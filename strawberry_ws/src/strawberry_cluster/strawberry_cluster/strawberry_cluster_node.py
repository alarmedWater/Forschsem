#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster strawberries across multiple views using 3D centroids in world frame
and export clustered point clouds to .ply.

Synchronized inputs:
  - masked depth image (/seg/depth_masked)
  - instance label image (/seg/label_image)
  - camera pose in world (/camera_pose_world) [geometry_msgs/PoseStamped]
  - frame info (FrameInfo) aligned to depth stamp (recommended)

For each instance:
  - compute 3D points in camera frame (from depth + intrinsics)
  - compute centroid in camera frame
  - transform centroid + points to world frame using PoseStamped
  - assign/update a cluster in world frame (per plant_id)
  - accumulate world points per cluster

Exports:
  - on plant change (or shutdown): writes one PLY per cluster + summary txt

Assumes "3 views per plant":
  view_id: 0=links, 1=mitte, 2=rechts
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from strawberry_msgs.msg import FrameInfo


@dataclass
class Cluster:
    """Per-plant cluster representation in world coordinates."""

    cluster_id: int
    centroid_world: np.ndarray  # (3,)
    num_points: int
    views_seen: set[int] = field(default_factory=set)
    last_frame_index: int = -1
    points_world: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )


def quaternion_to_rotation_matrix(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> np.ndarray:
    """Convert quaternion (x,y,z,w) to 3x3 rotation matrix."""
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


class StrawberryClusterNode(Node):
    """Cluster strawberry instances across multiple views in world coordinates."""

    def __init__(self) -> None:
        super().__init__("strawberry_cluster")

        # ---------------- Parameters ----------------
        self.declare_parameter("depth_topic", "/seg/depth_masked")
        self.declare_parameter("label_topic", "/seg/label_image")
        self.declare_parameter("camera_pose_topic", "/camera_pose_world")
        self.declare_parameter("frame_info_topic", "/seg/frame_info_depth_masked")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")

        self.declare_parameter("downsample_step", 1)
        self.declare_parameter("min_points", 50)
        self.declare_parameter("distance_threshold", 0.02)  # meters
        self.declare_parameter("max_clusters", 50)
        self.declare_parameter("profile", False)

        # Sync tuning
        self.declare_parameter("sync_queue_size", 200)
        self.declare_parameter("sync_slop", 0.2)

        # Depth scaling
        self.declare_parameter("depth_unit", "mm")  # "mm" or "realsense_units"
        self.declare_parameter("depth_scale_m_per_unit", 9.999999747378752e-05)

        # Practical: reset clusters when plant_id changes
        self.declare_parameter("reset_on_new_plant", True)

        # Logging control
        self.declare_parameter("log_assignments", True)

        # -------- Optional range filter --------
        self.declare_parameter("range_filter_enable", True)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 0.60)

        # -------- PLY export --------
        self.declare_parameter("output_dir", str(Path.home() / "strawberry_ply"))
        self.declare_parameter("write_ply_on_plant_change", True)
        self.declare_parameter("write_ply_on_shutdown", True)
        self.declare_parameter("max_points_per_cluster", 200000)  # safety
        self.declare_parameter("ply_ascii", True)

        depth_topic = self._param_str("depth_topic", "/seg/depth_masked")
        label_topic = self._param_str("label_topic", "/seg/label_image")
        pose_topic = self._param_str("camera_pose_topic", "/camera_pose_world")
        frame_info_topic = self._param_str(
            "frame_info_topic", "/seg/frame_info_depth_masked"
        )
        cam_info_topic = self._param_str("camera_info_topic", "/camera/color/camera_info")

        self._step = max(1, self._param_int("downsample_step", 1))
        self._min_points = max(0, self._param_int("min_points", 50))
        self._dist_thresh = float(self._param_float("distance_threshold", 0.02))
        self._max_clusters = max(1, self._param_int("max_clusters", 50))
        self._profile = self._param_bool("profile", False)

        self._sync_queue_size = max(1, self._param_int("sync_queue_size", 200))
        self._sync_slop = float(self._param_float("sync_slop", 0.2))
        if self._sync_slop <= 0.0:
            self._sync_slop = 0.05

        self._depth_unit = self._param_str("depth_unit", "mm").strip().lower()
        self._depth_scale = float(
            self._param_float("depth_scale_m_per_unit", 9.999999747378752e-05)
        )

        self._reset_on_new_plant = self._param_bool("reset_on_new_plant", True)
        self._log_assignments = self._param_bool("log_assignments", True)

        self._range_filter_enable = self._param_bool("range_filter_enable", True)
        self._min_depth_m = float(self._param_float("min_depth_m", 0.05))
        self._max_depth_m = float(self._param_float("max_depth_m", 0.60))
        if self._min_depth_m > self._max_depth_m:
            self._min_depth_m, self._max_depth_m = self._max_depth_m, self._min_depth_m

        self._output_dir = Path(
            self._param_str("output_dir", str(Path.home() / "strawberry_ply"))
        )
        self._write_on_plant_change = self._param_bool("write_ply_on_plant_change", True)
        self._write_on_shutdown = self._param_bool("write_ply_on_shutdown", True)
        self._max_points_per_cluster = max(
            1000, self._param_int("max_points_per_cluster", 200000)
        )
        self._ply_ascii = self._param_bool("ply_ascii", True)

        self.get_logger().info(
            "StrawberryClusterNode starting:\n"
            f"  depth_topic              = {depth_topic}\n"
            f"  label_topic              = {label_topic}\n"
            f"  camera_pose_topic        = {pose_topic}\n"
            f"  frame_info_topic         = {frame_info_topic}\n"
            f"  camera_info_topic        = {cam_info_topic}\n"
            f"  downsample_step          = {self._step}\n"
            f"  min_points               = {self._min_points}\n"
            f"  distance_threshold       = {self._dist_thresh:.3f} m\n"
            f"  max_clusters             = {self._max_clusters}\n"
            f"  depth_unit               = {self._depth_unit}\n"
            f"  depth_scale              = {self._depth_scale:.3e} m/unit\n"
            f"  range_filter_enable      = {self._range_filter_enable}\n"
            f"  min_depth_m              = {self._min_depth_m:.3f}\n"
            f"  max_depth_m              = {self._max_depth_m:.3f}\n"
            f"  reset_on_new_plant       = {self._reset_on_new_plant}\n"
            f"  log_assignments          = {self._log_assignments}\n"
            f"  output_dir               = {self._output_dir}\n"
            f"  write_on_plant_change    = {self._write_on_plant_change}\n"
            f"  write_on_shutdown        = {self._write_on_shutdown}\n"
            f"  max_points_per_cluster   = {self._max_points_per_cluster}\n"
            f"  ply_ascii                = {self._ply_ascii}\n"
            f"  profile                  = {self._profile}"
        )

        self._bridge = CvBridge()

        # ---------------- Intrinsics ----------------
        self._fx: Optional[float] = None
        self._fy: Optional[float] = None
        self._cx: Optional[float] = None
        self._cy: Optional[float] = None

        self._warned_no_intrinsics = False
        self._warned_bad_depth_unit = False

        self.create_subscription(CameraInfo, cam_info_topic, self._camera_info_cb, 10)

        # ---------------- Clusters (per current plant) ----------------
        self._active_plant_id: Optional[int] = None
        self._clusters: List[Cluster] = []
        self._cluster_by_id: Dict[int, Cluster] = {}
        self._next_cluster_id: int = 1

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._sub_depth = message_filters.Subscriber(
            self, Image, depth_topic, qos_profile=qos
        )
        self._sub_label = message_filters.Subscriber(
            self, Image, label_topic, qos_profile=qos
        )
        self._sub_pose = message_filters.Subscriber(
            self, PoseStamped, pose_topic, qos_profile=qos
        )
        self._sub_frame_info = message_filters.Subscriber(
            self, FrameInfo, frame_info_topic, qos_profile=qos
        )

        self._ts = message_filters.ApproximateTimeSynchronizer(
            [self._sub_depth, self._sub_label, self._sub_pose, self._sub_frame_info],
            queue_size=self._sync_queue_size,
            slop=self._sync_slop,
        )
        self._ts.registerCallback(self._sync_cb)

    # ------------------------------------------------------------------ #
    # Param helpers
    # ------------------------------------------------------------------ #

    def _param_str(self, name: str, default: str) -> str:
        val: Any = self.get_parameter(name).value
        if val is None:
            return default
        s = str(val).strip()
        return s if s else default

    def _param_bool(self, name: str, default: bool) -> bool:
        val: Any = self.get_parameter(name).value
        if isinstance(val, bool):
            return val
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "y", "on")
        return default

    def _param_int(self, name: str, default: int) -> int:
        val: Any = self.get_parameter(name).value
        if val is None:
            return default
        try:
            return int(val)
        except Exception:  # noqa: BLE001
            return default

    def _param_float(self, name: str, default: float) -> float:
        val: Any = self.get_parameter(name).value
        if val is None:
            return default
        try:
            return float(val)
        except Exception:  # noqa: BLE001
            return default

    # ------------------------------------------------------------------ #
    # Intrinsics callback
    # ------------------------------------------------------------------ #

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self._fx = float(msg.k[0])
        self._fy = float(msg.k[4])
        self._cx = float(msg.k[2])
        self._cy = float(msg.k[5])

    # ------------------------------------------------------------------ #
    # Depth conversion
    # ------------------------------------------------------------------ #

    def _depth_to_meters(self, depth: np.ndarray) -> np.ndarray:
        if depth.dtype == np.uint16:
            if self._depth_unit == "mm":
                return depth.astype(np.float32) / 1000.0
            if self._depth_unit == "realsense_units":
                return depth.astype(np.float32) * float(self._depth_scale)

            if not self._warned_bad_depth_unit:
                self.get_logger().warning(
                    f"Unknown depth_unit='{self._depth_unit}'. Falling back to 'mm'."
                )
                self._warned_bad_depth_unit = True
            return depth.astype(np.float32) / 1000.0

        return depth.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Core callback
    # ------------------------------------------------------------------ #

    def _sync_cb(
        self,
        depth_msg: Image,
        label_msg: Image,
        pose_msg: PoseStamped,
        frame_info_msg: FrameInfo,
    ) -> None:
        t0 = time.time()

        if self._fx is None or self._fy is None or self._cx is None or self._cy is None:
            if not self._warned_no_intrinsics:
                self.get_logger().warning(
                    "No CameraInfo received yet – cannot compute 3D points."
                )
                self._warned_no_intrinsics = True
            return

        plant_id = int(frame_info_msg.plant_id)
        view_id = int(frame_info_msg.view_id)
        frame_index = int(frame_info_msg.frame_index)

        # Plant switch => export previous plant
        if self._reset_on_new_plant:
            if self._active_plant_id is None:
                self._active_plant_id = plant_id
            elif plant_id != self._active_plant_id:
                if self._write_on_plant_change:
                    self._export_current_plant()

                self.get_logger().info(
                    f"Plant changed {self._active_plant_id} -> {plant_id}. Resetting clusters."
                )
                self._active_plant_id = plant_id
                self._clusters = []
                self._cluster_by_id = {}
                self._next_cluster_id = 1
        elif self._active_plant_id is None:
            self._active_plant_id = plant_id

        depth_raw = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        label = self._bridge.imgmsg_to_cv2(label_msg, desired_encoding="mono16")

        if depth_raw is None or label is None:
            self.get_logger().warning("cv_bridge returned None for depth or label.")
            return

        if depth_raw.shape[:2] != label.shape[:2]:
            self.get_logger().warning(
                f"Shape mismatch depth={depth_raw.shape} label={label.shape} – check alignment."
            )
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
            return

        unique_ids = np.unique(lbl_sub[valid])
        unique_ids = unique_ids[unique_ids > 0]
        if unique_ids.size == 0:
            return

        fx = float(self._fx)
        fy = float(self._fy)
        cx = float(self._cx)
        cy = float(self._cy)

        pos = pose_msg.pose.position
        ori = pose_msg.pose.orientation
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

            cid, _ = self._assign_to_cluster(
                centroid_world=centroid_world,
                num_points=n_pts,
                view_id=view_id,
                frame_index=frame_index,
            )

            # accumulate points in world frame
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
            self.get_logger().info(
                f"Cluster callback: {dt_ms:.2f} ms | clusters={len(self._clusters)}"
            )

    # ------------------------------------------------------------------ #
    # Cluster management
    # ------------------------------------------------------------------ #

    def _append_points_to_cluster(self, cluster_id: int, pts_world: np.ndarray) -> None:
        c = self._cluster_by_id.get(int(cluster_id))
        if c is None:
            return

        pts_world = np.asarray(pts_world, dtype=np.float32).reshape((-1, 3))

        if c.points_world.size == 0:
            c.points_world = pts_world
        else:
            c.points_world = np.vstack([c.points_world, pts_world])

        # safety cap
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
            return (
                self._create_cluster(centroid_world, num_points, view_id, frame_index),
                True,
            )

        dists = [float(np.linalg.norm(centroid_world - c.centroid_world)) for c in self._clusters]
        min_idx = int(np.argmin(dists))
        min_dist = float(dists[min_idx])
        best = self._clusters[min_idx]

        if min_dist < self._dist_thresh:
            total = best.num_points + int(num_points)
            w_old = best.num_points / float(total)
            w_new = int(num_points) / float(total)

            best.centroid_world = (w_old * best.centroid_world) + (w_new * centroid_world)
            best.num_points = int(total)
            best.views_seen.add(int(view_id))
            best.last_frame_index = int(frame_index)
            return best.cluster_id, True

        if len(self._clusters) < self._max_clusters:
            return (
                self._create_cluster(centroid_world, num_points, view_id, frame_index),
                True,
            )

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
            centroid_world=centroid_world.copy(),
            num_points=int(num_points),
            views_seen={int(view_id)},
            last_frame_index=int(frame_index),
        )
        self._clusters.append(c)
        self._cluster_by_id[cid] = c

        # FIX: rclpy logger expects one message string → f-string
        self.get_logger().info(
            "Created cluster "
            f"{cid} at ({float(centroid_world[0]):.3f}, {float(centroid_world[1]):.3f}, "
            f"{float(centroid_world[2]):.3f}) m | N={int(num_points)} | view={int(view_id)} "
            f"| frame={int(frame_index)}"
        )
        return cid

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def _export_current_plant(self) -> None:
        plant_id = self._active_plant_id
        if plant_id is None:
            return

        out_dir = self._output_dir / f"plant_{plant_id:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_lines: List[str] = [
            f"plant_id: {plant_id}",
            f"clusters: {len(self._clusters)}",
            "",
        ]

        exported = 0
        for c in self._clusters:
            pts = c.points_world
            if pts.size == 0:
                continue

            ply_path = out_dir / f"cluster_{c.cluster_id:03d}.ply"
            self._write_ply(ply_path, pts, ascii_mode=self._ply_ascii)
            exported += 1

            summary_lines.append(f"cluster_id: {c.cluster_id}")
            summary_lines.append(
                "centroid_world: "
                f"[{float(c.centroid_world[0]):.4f}, {float(c.centroid_world[1]):.4f}, "
                f"{float(c.centroid_world[2]):.4f}]"
            )
            summary_lines.append(f"num_points_accum: {int(pts.shape[0])}")
            summary_lines.append(f"views_seen: {sorted(list(c.views_seen))}")
            summary_lines.append(f"last_frame_index: {c.last_frame_index}")
            summary_lines.append(f"ply: {ply_path.name}")
            summary_lines.append("")

        (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
        self.get_logger().info(
            f"Exported {exported} PLY file(s) for plant {plant_id} to: {out_dir}"
        )

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
            # (für sehr große Clouds nicht perfekt, aber ok bis ~200k)
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

    # ------------------------------------------------------------------ #
    # Shutdown hook (FIX: no destroy_node override)
    # ------------------------------------------------------------------ #

    def shutdown(self) -> None:
        """Call this from main() before destroy_node()."""
        if not self._write_on_shutdown:
            return
        try:
            self._export_current_plant()
        except Exception as exc:  # noqa: BLE001
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
