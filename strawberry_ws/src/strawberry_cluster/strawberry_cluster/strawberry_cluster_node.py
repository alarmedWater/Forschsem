#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster strawberries across multiple views using 3D centroids in world frame
and export clustered point clouds to .ply.

Synchronized inputs:
  - masked depth image (/seg/depth_masked)
  - instance label image (/seg/label_image)
  - frame info (FrameInfo) aligned to depth stamp
  - camera intrinsics from CameraInfo (/camera/color/camera_info)

Pose comes from FrameInfo.camera_pose_world.

Exports per plant:
  - one PLY per cluster: cluster_XXX.ply
  - optional fused plant cloud: plant_fused.ply (all cluster points merged)
  - summary.txt + summary.csv

Robustness additions:
  - sync_mode: exact (default) or approx
  - process_once_per_plant: avoid overwriting when camera_folder loops
  - optional export_on_views_complete: export as soon as all expected views were seen
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from strawberry_msgs.msg import FrameInfo


# ---------------- Math helpers ----------------

def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        return np.eye(3, dtype=np.float32)

    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# ---------------- Data model ----------------

@dataclass
class Cluster:
    cluster_id: int
    centroid_world: np.ndarray
    num_points_weight: int
    views_seen: Set[int] = field(default_factory=set)
    last_frame_index: int = -1
    points_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))


# ---------------- Node ----------------

class StrawberryClusterNode(Node):
    def __init__(self) -> None:
        super().__init__("strawberry_cluster")

        # ---------------- Parameters ----------------
        self.declare_parameter("depth_topic", "/seg/depth_masked")
        self.declare_parameter("label_topic", "/seg/label_image")
        self.declare_parameter("frame_info_topic", "/seg/frame_info_depth_masked")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")

        # Sync
        self.declare_parameter("sync_mode", "exact")          # "exact" | "approx"
        self.declare_parameter("sync_queue_size", 200)
        self.declare_parameter("sync_slop", 0.2)              # only for approx

        # Point filtering / downsample
        self.declare_parameter("downsample_step", 1)
        self.declare_parameter("min_points", 50)
        self.declare_parameter("range_filter_enable", True)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 0.60)

        # Depth unit
        self.declare_parameter("depth_unit", "mm")            # "mm" | "realsense_units"
        self.declare_parameter("depth_scale_m_per_unit", 9.999999747378752e-05)

        # Clustering
        self.declare_parameter("distance_threshold", 0.05)
        self.declare_parameter("max_clusters", 50)
        self.declare_parameter("log_assignments", True)
        self.declare_parameter("profile", False)

        # Plant handling
        self.declare_parameter("expected_views_per_plant", 3)
        self.declare_parameter("export_on_views_complete", False)
        self.declare_parameter("process_once_per_plant", True)

        # Export
        self.declare_parameter("output_dir", str(Path.home() / "strawberry_ply"))
        self.declare_parameter("use_session_subdir", True)    # output_dir/<session>/plant_XXX/...
        self.declare_parameter("session_name", "")            # if empty -> timestamp
        self.declare_parameter("write_ply_on_plant_change", True)
        self.declare_parameter("write_ply_on_shutdown", True)

        self.declare_parameter("ply_ascii", True)
        self.declare_parameter("max_points_per_cluster", 200000)

        # Fused export
        self.declare_parameter("export_fused_plant_cloud", True)
        self.declare_parameter("fused_filename", "plant_fused.ply")
        self.declare_parameter("fused_voxel_size", 0.0)       # >0 => voxel downsample
        self.declare_parameter("fused_max_points", 0)         # 0 => unlimited

        # ---------------- Read params ----------------
        self._depth_topic = self._pstr("depth_topic", "/seg/depth_masked")
        self._label_topic = self._pstr("label_topic", "/seg/label_image")
        self._fi_topic = self._pstr("frame_info_topic", "/seg/frame_info_depth_masked")
        self._cam_info_topic = self._pstr("camera_info_topic", "/camera/color/camera_info")

        self._sync_mode = self._pstr("sync_mode", "exact").strip().lower()
        if self._sync_mode not in ("exact", "approx"):
            self._sync_mode = "exact"
        self._sync_queue = max(1, self._pint("sync_queue_size", 200))
        self._sync_slop = float(self._pfloat("sync_slop", 0.2))
        if self._sync_slop <= 0.0:
            self._sync_slop = 0.05

        self._step = max(1, self._pint("downsample_step", 1))
        self._min_points = max(0, self._pint("min_points", 50))

        self._range_filter_enable = self._pbool("range_filter_enable", True)
        self._min_depth_m = float(self._pfloat("min_depth_m", 0.05))
        self._max_depth_m = float(self._pfloat("max_depth_m", 0.60))
        if self._min_depth_m > self._max_depth_m:
            self._min_depth_m, self._max_depth_m = self._max_depth_m, self._min_depth_m

        self._depth_unit = self._pstr("depth_unit", "mm").strip().lower()
        self._depth_scale = float(self._pfloat("depth_scale_m_per_unit", 9.999999747378752e-05))

        self._dist_thresh = float(self._pfloat("distance_threshold", 0.05))
        self._max_clusters = max(1, self._pint("max_clusters", 50))
        self._log_assignments = self._pbool("log_assignments", True)
        self._profile = self._pbool("profile", False)

        self._expected_views = max(1, self._pint("expected_views_per_plant", 3))
        self._export_on_views_complete = self._pbool("export_on_views_complete", False)
        self._process_once_per_plant = self._pbool("process_once_per_plant", True)

        out_root = Path(self._pstr("output_dir", str(Path.home() / "strawberry_ply")))
        use_session = self._pbool("use_session_subdir", True)
        session_name = self._pstr("session_name", "").strip()
        if use_session:
            if not session_name:
                session_name = f"run_{now_str()}"
            self._output_dir = out_root / session_name
        else:
            self._output_dir = out_root

        self._write_on_plant_change = self._pbool("write_ply_on_plant_change", True)
        self._write_on_shutdown = self._pbool("write_ply_on_shutdown", True)
        self._ply_ascii = self._pbool("ply_ascii", True)
        self._max_points_per_cluster = max(1000, self._pint("max_points_per_cluster", 200000))

        self._export_fused = self._pbool("export_fused_plant_cloud", True)
        self._fused_filename = self._pstr("fused_filename", "plant_fused.ply")
        self._fused_voxel = float(self._pfloat("fused_voxel_size", 0.0))
        self._fused_max_points = int(self._pint("fused_max_points", 0))

        self.get_logger().info(
            "StrawberryClusterNode:\n"
            f"  depth_topic              = {self._depth_topic}\n"
            f"  label_topic              = {self._label_topic}\n"
            f"  frame_info_topic         = {self._fi_topic}\n"
            f"  camera_info_topic        = {self._cam_info_topic}\n"
            f"  sync_mode                = {self._sync_mode} (queue={self._sync_queue} slop={self._sync_slop})\n"
            f"  downsample_step          = {self._step}\n"
            f"  min_points               = {self._min_points}\n"
            f"  distance_threshold       = {self._dist_thresh:.3f} m\n"
            f"  max_clusters             = {self._max_clusters}\n"
            f"  depth_unit               = {self._depth_unit}\n"
            f"  depth_scale              = {self._depth_scale:.3e} m/unit\n"
            f"  range_filter_enable      = {self._range_filter_enable} [{self._min_depth_m:.3f},{self._max_depth_m:.3f}] m\n"
            f"  expected_views_per_plant = {self._expected_views}\n"
            f"  export_on_views_complete = {self._export_on_views_complete}\n"
            f"  process_once_per_plant   = {self._process_once_per_plant}\n"
            f"  output_dir               = {self._output_dir}\n"
            f"  write_on_plant_change    = {self._write_on_plant_change}\n"
            f"  write_on_shutdown        = {self._write_on_shutdown}\n"
            f"  max_points_per_cluster   = {self._max_points_per_cluster}\n"
            f"  ply_ascii                = {self._ply_ascii}\n"
            f"  export_fused_plant_cloud = {self._export_fused} ({self._fused_filename}, voxel={self._fused_voxel}, max={self._fused_max_points})\n"
            f"  profile                  = {self._profile}"
        )

        self._bridge = CvBridge()

        # Intrinsics
        self._fx: Optional[float] = None
        self._fy: Optional[float] = None
        self._cx: Optional[float] = None
        self._cy: Optional[float] = None
        self._warned_no_intrinsics = False
        self._warned_bad_depth_unit = False
        self._warned_missing_pose = False

        # Subscribe CameraInfo
        self.create_subscription(CameraInfo, self._cam_info_topic, self._camera_info_cb, qos_profile_sensor_data)

        # Clusters state
        self._active_plant_id: Optional[int] = None
        self._views_seen_current: Set[int] = set()
        self._processed_plants: Set[int] = set()

        self._clusters: List[Cluster] = []
        self._cluster_by_id: Dict[int, Cluster] = {}
        self._next_cluster_id: int = 1

        # message_filters sync
        self._sub_depth = message_filters.Subscriber(self, Image, self._depth_topic, qos_profile=qos_profile_sensor_data)
        self._sub_label = message_filters.Subscriber(self, Image, self._label_topic, qos_profile=qos_profile_sensor_data)
        self._sub_fi = message_filters.Subscriber(self, FrameInfo, self._fi_topic, qos_profile=qos_profile_sensor_data)

        if self._sync_mode == "exact":
            self._ts = message_filters.TimeSynchronizer(
                [self._sub_depth, self._sub_label, self._sub_fi],
                queue_size=self._sync_queue,
            )
        else:
            self._ts = message_filters.ApproximateTimeSynchronizer(
                [self._sub_depth, self._sub_label, self._sub_fi],
                queue_size=self._sync_queue,
                slop=self._sync_slop,
            )
        self._ts.registerCallback(self._sync_cb)

    # ---------------- Param helpers ----------------

    def _pstr(self, name: str, default: str) -> str:
        v: Any = self.get_parameter(name).value
        if v is None:
            return default
        s = str(v).strip()
        return s if s else default

    def _pbool(self, name: str, default: bool) -> bool:
        v: Any = self.get_parameter(name).value
        if isinstance(v, bool):
            return v
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "y", "on")
        return default

    def _pint(self, name: str, default: int) -> int:
        v: Any = self.get_parameter(name).value
        if v is None:
            return default
        try:
            return int(v)
        except Exception:
            return default

    def _pfloat(self, name: str, default: float) -> float:
        v: Any = self.get_parameter(name).value
        if v is None:
            return default
        try:
            return float(v)
        except Exception:
            return default

    # ---------------- Intrinsics ----------------

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self._fx = float(msg.k[0])
        self._fy = float(msg.k[4])
        self._cx = float(msg.k[2])
        self._cy = float(msg.k[5])

    # ---------------- Depth conversion ----------------

    def _depth_to_meters(self, depth: np.ndarray) -> np.ndarray:
        if depth.dtype == np.uint16:
            if self._depth_unit == "mm":
                return depth.astype(np.float32) / 1000.0
            if self._depth_unit == "realsense_units":
                return depth.astype(np.float32) * float(self._depth_scale)

            if not self._warned_bad_depth_unit:
                self.get_logger().warning(f"Unknown depth_unit='{self._depth_unit}'. Falling back to 'mm'.")
                self._warned_bad_depth_unit = True
            return depth.astype(np.float32) / 1000.0

        return depth.astype(np.float32)

    # ---------------- Plant handling ----------------

    def _reset_for_new_plant(self, plant_id: int) -> None:
        self._active_plant_id = int(plant_id)
        self._views_seen_current = set()
        self._clusters = []
        self._cluster_by_id = {}
        self._next_cluster_id = 1

    def _maybe_switch_plant(self, plant_id: int) -> None:
        if self._active_plant_id is None:
            self._reset_for_new_plant(plant_id)
            return

        if int(plant_id) != int(self._active_plant_id):
            # export previous plant before switching
            if self._write_on_plant_change:
                self._export_current_plant()

            if self._process_once_per_plant and self._active_plant_id is not None:
                self._processed_plants.add(int(self._active_plant_id))

            self.get_logger().info(f"Plant changed {self._active_plant_id} -> {plant_id}. Resetting clusters.")
            self._reset_for_new_plant(plant_id)

    # ---------------- Core sync callback ----------------

    def _sync_cb(self, depth_msg: Image, label_msg: Image, fi: FrameInfo) -> None:
        t0 = time.time()

        if self._fx is None or self._fy is None or self._cx is None or self._cy is None:
            if not self._warned_no_intrinsics:
                self.get_logger().warning("No CameraInfo received yet – cannot compute 3D points.")
                self._warned_no_intrinsics = True
            return

        plant_id = int(fi.plant_id)
        view_id = int(fi.view_id)
        frame_index = int(fi.frame_index)

        # If we already processed this plant (loop case), ignore.
        if self._process_once_per_plant and (plant_id in self._processed_plants):
            return

        # Plant switch handling
        self._maybe_switch_plant(plant_id)

        # Track views seen (for early export if enabled)
        self._views_seen_current.add(int(view_id))

        # Convert messages
        depth_raw = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        # label image expected mono16 (instance ids)
        label = self._bridge.imgmsg_to_cv2(label_msg, desired_encoding="mono16")

        if depth_raw is None or label is None:
            self.get_logger().warning("cv_bridge returned None for depth or label.")
            return

        if depth_raw.shape[:2] != label.shape[:2]:
            self.get_logger().warning(f"Shape mismatch depth={depth_raw.shape} label={label.shape} – check alignment.")
            return

        z_m = self._depth_to_meters(depth_raw)
        h, w = z_m.shape

        step = self._step
        if step > 1:
            z_sub = z_m[0:h:step, 0:w:step]
            lbl_sub = label[0:h:step, 0:w:step]
            v_grid, u_grid = np.mgrid[0:h:step, 0:w:step]
        else:
            z_sub = z_m
            lbl_sub = label
            v_grid, u_grid = np.mgrid[0:h, 0:w]

        valid = np.isfinite(z_sub) & (z_sub > 0.0)
        if self._range_filter_enable:
            valid &= (z_sub >= self._min_depth_m) & (z_sub <= self._max_depth_m)

        if not np.any(valid):
            self._maybe_export_if_complete(plant_id)
            return

        unique_ids = np.unique(lbl_sub[valid])
        unique_ids = unique_ids[unique_ids > 0]
        if unique_ids.size == 0:
            self._maybe_export_if_complete(plant_id)
            return

        # Intrinsics
        fx = float(self._fx)
        fy = float(self._fy)
        cx = float(self._cx)
        cy = float(self._cy)

        # Pose from FrameInfo
        try:
            pos = fi.camera_pose_world.position
            ori = fi.camera_pose_world.orientation
        except Exception:
            if not self._warned_missing_pose:
                self.get_logger().error("FrameInfo has no camera_pose_world. Cannot transform to world.")
                self._warned_missing_pose = True
            return

        r_world_cam = quaternion_to_rotation_matrix(ori.x, ori.y, ori.z, ori.w)
        t_world_cam = np.array([pos.x, pos.y, pos.z], dtype=np.float32)

        assignments: List[Tuple[int, int]] = []

        for inst_id in unique_ids.tolist():
            mask = valid & (lbl_sub == inst_id)
            if not np.any(mask):
                continue

            z_i = z_sub[mask]
            v_i = v_grid[mask].astype(np.float32)
            u_i = u_grid[mask].astype(np.float32)

            x_i = (u_i - cx) * z_i / fx
            y_i = (v_i - cy) * z_i / fy

            points_cam = np.stack((x_i, y_i, z_i), axis=-1).astype(np.float32)
            n_pts = int(points_cam.shape[0])
            if n_pts < self._min_points:
                continue

            centroid_cam = points_cam.mean(axis=0)
            centroid_world = (r_world_cam @ centroid_cam) + t_world_cam

            cid, _created_or_matched = self._assign_to_cluster(
                centroid_world=centroid_world,
                num_points=n_pts,
                view_id=view_id,
                frame_index=frame_index,
            )

            pts_w = (r_world_cam @ points_cam.T).T + t_world_cam
            self._append_points_to_cluster(cid, pts_w)

            assignments.append((int(inst_id), int(cid)))

        if self._log_assignments and assignments:
            lines = [
                f"Frame {frame_index} | plant {plant_id} | view {view_id} | "
                f"instances={len(assignments)} | clusters={len(self._clusters)}"
            ]
            for inst_id, cid in assignments:
                lines.append(f"  inst {inst_id} -> cluster {cid}")
            self.get_logger().info("\n".join(lines))

        if self._profile:
            dt_ms = (time.time() - t0) * 1000.0
            self.get_logger().info(f"Cluster callback: {dt_ms:.2f} ms | clusters={len(self._clusters)}")

        self._maybe_export_if_complete(plant_id)

    def _maybe_export_if_complete(self, plant_id: int) -> None:
        if not self._export_on_views_complete:
            return
        if self._active_plant_id is None:
            return
        if int(plant_id) != int(self._active_plant_id):
            return
        if len(self._views_seen_current) < self._expected_views:
            return

        # Export now
        self.get_logger().info(
            f"Views complete for plant {plant_id} ({sorted(self._views_seen_current)}). Exporting now."
        )
        self._export_current_plant()

        if self._process_once_per_plant:
            self._processed_plants.add(int(plant_id))

        # Reset for next plant (next callback will set active)
        self._active_plant_id = None
        self._views_seen_current = set()
        self._clusters = []
        self._cluster_by_id = {}
        self._next_cluster_id = 1

    # ---------------- Cluster management ----------------

    def _append_points_to_cluster(self, cluster_id: int, pts_world: np.ndarray) -> None:
        c = self._cluster_by_id.get(int(cluster_id))
        if c is None:
            return

        pts_world = np.asarray(pts_world, dtype=np.float32).reshape((-1, 3))

        if c.points_world.size == 0:
            c.points_world = pts_world
        else:
            c.points_world = np.vstack([c.points_world, pts_world])

        if c.points_world.shape[0] > self._max_points_per_cluster:
            idx = np.random.choice(
                c.points_world.shape[0],
                size=self._max_points_per_cluster,
                replace=False,
            )
            c.points_world = c.points_world[idx]

    def _assign_to_cluster(
        self,
        centroid_world: np.ndarray,
        num_points: int,
        view_id: int,
        frame_index: int,
    ) -> Tuple[int, bool]:
        if not self._clusters:
            cid = self._create_cluster(centroid_world, num_points, view_id, frame_index)
            return cid, True

        dists = [float(np.linalg.norm(centroid_world - c.centroid_world)) for c in self._clusters]
        min_idx = int(np.argmin(dists))
        min_dist = float(dists[min_idx])
        best = self._clusters[min_idx]

        if min_dist < self._dist_thresh:
            total = best.num_points_weight + int(num_points)
            w_old = best.num_points_weight / float(total)
            w_new = int(num_points) / float(total)

            best.centroid_world = (w_old * best.centroid_world) + (w_new * centroid_world)
            best.num_points_weight = int(total)
            best.views_seen.add(int(view_id))
            best.last_frame_index = int(frame_index)
            return best.cluster_id, True

        if len(self._clusters) < self._max_clusters:
            cid = self._create_cluster(centroid_world, num_points, view_id, frame_index)
            return cid, True

        # fallback: assign to closest anyway
        best.views_seen.add(int(view_id))
        best.last_frame_index = int(frame_index)
        return best.cluster_id, False

    def _create_cluster(
        self,
        centroid_world: np.ndarray,
        num_points: int,
        view_id: int,
        frame_index: int,
    ) -> int:
        cid = int(self._next_cluster_id)
        self._next_cluster_id += 1

        c = Cluster(
            cluster_id=cid,
            centroid_world=np.asarray(centroid_world, dtype=np.float32).copy(),
            num_points_weight=int(num_points),
            views_seen={int(view_id)},
            last_frame_index=int(frame_index),
        )
        self._clusters.append(c)
        self._cluster_by_id[cid] = c

        self.get_logger().info(
            "Created cluster "
            f"{cid} at ({float(centroid_world[0]):.3f}, {float(centroid_world[1]):.3f}, "
            f"{float(centroid_world[2]):.3f}) m | N={int(num_points)} | view={int(view_id)} "
            f"| frame={int(frame_index)}"
        )
        return cid

    # ---------------- Export helpers ----------------

    def _plant_dir(self, plant_id: int) -> Path:
        out_dir = self._output_dir / f"plant_{plant_id:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _export_current_plant(self) -> None:
        plant_id = self._active_plant_id
        if plant_id is None:
            return

        out_dir = self._plant_dir(int(plant_id))

        # Write per-cluster clouds
        exported = 0
        summary_lines: List[str] = [
            f"plant_id: {plant_id}",
            f"clusters_total: {len(self._clusters)}",
            f"views_seen_current: {sorted(list(self._views_seen_current))}",
            "",
        ]

        csv_rows: List[List[Any]] = []
        all_pts_for_fused: List[np.ndarray] = []

        for c in self._clusters:
            pts = c.points_world
            if pts.size == 0:
                continue

            ply_path = out_dir / f"cluster_{c.cluster_id:03d}.ply"
            self._write_ply(ply_path, pts, ascii_mode=self._ply_ascii)
            exported += 1
            all_pts_for_fused.append(pts)

            summary_lines.append(f"cluster_id: {c.cluster_id}")
            summary_lines.append(
                "centroid_world: "
                f"[{float(c.centroid_world[0]):.4f}, {float(c.centroid_world[1]):.4f}, {float(c.centroid_world[2]):.4f}]"
            )
            summary_lines.append(f"points_exported: {int(pts.shape[0])}")
            summary_lines.append(f"views_seen: {sorted(list(c.views_seen))}")
            summary_lines.append(f"last_frame_index: {c.last_frame_index}")
            summary_lines.append(f"ply: {ply_path.name}")
            summary_lines.append("")

            csv_rows.append([
                int(plant_id),
                int(c.cluster_id),
                float(c.centroid_world[0]),
                float(c.centroid_world[1]),
                float(c.centroid_world[2]),
                int(pts.shape[0]),
                ",".join(map(str, sorted(list(c.views_seen)))),
                int(c.last_frame_index),
                ply_path.name,
            ])

        # Optional fused plant cloud (all clusters merged)
        fused_written = False
        fused_points_n = 0
        if self._export_fused and all_pts_for_fused:
            pts_all = np.vstack(all_pts_for_fused).astype(np.float32, copy=False)

            if self._fused_voxel > 0.0:
                pts_all = self._voxel_downsample(pts_all, self._fused_voxel)

            if self._fused_max_points and pts_all.shape[0] > self._fused_max_points:
                idx = np.random.choice(pts_all.shape[0], size=self._fused_max_points, replace=False)
                pts_all = pts_all[idx]

            fused_path = out_dir / self._fused_filename
            self._write_ply(fused_path, pts_all, ascii_mode=self._ply_ascii)
            fused_written = True
            fused_points_n = int(pts_all.shape[0])

            summary_lines.append("fused_plant_cloud:")
            summary_lines.append(f"  file: {fused_path.name}")
            summary_lines.append(f"  points: {fused_points_n}")
            summary_lines.append(f"  voxel: {self._fused_voxel}")
            summary_lines.append(f"  max_points: {self._fused_max_points}")
            summary_lines.append("")

        # Write summary files
        (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

        csv_path = out_dir / "summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow([
                "plant_id",
                "cluster_id",
                "centroid_x",
                "centroid_y",
                "centroid_z",
                "points_exported",
                "views_seen",
                "last_frame_index",
                "ply_file",
            ])
            w.writerows(csv_rows)

        self.get_logger().info(
            f"Exported plant {int(plant_id):03d}: clusters_written={exported} "
            f"fused_written={fused_written} fused_points={fused_points_n} -> {out_dir}"
        )

    @staticmethod
    def _voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
        v = float(voxel)
        if v <= 0.0 or pts.size == 0:
            return pts
        q = np.floor(pts / v).astype(np.int32)
        _, idx = np.unique(q, axis=0, return_index=True)
        return pts[np.sort(idx)]

    @staticmethod
    def _write_ply(path: Path, points_xyz: np.ndarray, ascii_mode: bool = True) -> None:
        pts = np.asarray(points_xyz, dtype=np.float32).reshape((-1, 3))
        n = int(pts.shape[0])

        if ascii_mode:
            header = (
                "ply\n"
                "format ascii 1.0\n"
                f"element vertex {n}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "end_header\n"
            )
            lines = [header]
            lines.extend([f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n" for p in pts])
            path.write_text("".join(lines), encoding="utf-8")
            return

        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        ).encode("ascii")
        with path.open("wb") as f:
            f.write(header)
            f.write(pts.astype("<f4", copy=False).tobytes())

    # ---------------- Shutdown hook ----------------

    def shutdown(self) -> None:
        if not self._write_on_shutdown:
            return
        try:
            # avoid re-export if already processed
            if self._active_plant_id is not None:
                pid = int(self._active_plant_id)
                if not (self._process_once_per_plant and pid in self._processed_plants):
                    self._export_current_plant()
        except Exception as exc:
            self.get_logger().error(f"Failed to export on shutdown: {exc}")


def main() -> None:
    rclpy.init()
    node = StrawberryClusterNode()
    try:
        rclpy.spin(node)
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
