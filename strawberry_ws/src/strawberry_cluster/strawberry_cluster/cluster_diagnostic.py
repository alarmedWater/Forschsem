#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster / Cloud / Pose Diagnostic (robust, ohne message_filters Abhängigkeit)

Logs in eine .txt:
- plant_id, view_id, frame_index
- dt zwischen FrameInfo und gematchter Pose/Cloud
- pos_diff/rot_diff zwischen FrameInfo.camera_pose_world und /camera_pose_world
- Punkteanzahl in /seg/strawberry_cloud und /seg/strawberry_cloud_selected
- Bei Plant-Wechsel: scannt exportierte PLYs und loggt Vertex counts (fused Ergebnis)

Run:
  ros2 run strawberry_cluster cluster_diagnostic --ros-args \
    -p match_slop_s:=0.2 \
    -p cloud_topic_all:=/seg/strawberry_cloud \
    -p cloud_topic_selected:=/seg/strawberry_cloud_selected \
    -p output_dir:=/home/parallels/strawberry_ply
"""

from __future__ import annotations

import glob
import math
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from strawberry_msgs.msg import FrameInfo


# -----------------------------
# Helpers
# -----------------------------
def _now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _valid_header_stamp_sec(msg) -> Optional[float]:
    """
    Returns header stamp in seconds if present and not zero.
    Otherwise None (=> we fall back to receive-time).
    """
    try:
        s = _stamp_to_sec(msg.header.stamp)
        if s <= 0.0:
            return None
        return s
    except Exception:
        return None


def _pose_to_np(p: Pose) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.array([p.position.x, p.position.y, p.position.z], dtype=float)
    quat = np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w], dtype=float)
    n = float(np.linalg.norm(quat))
    if n > 0.0:
        quat /= n
    return pos, quat


def _rotation_diff_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    dot = float(np.dot(q1, q2))
    dot = abs(dot)
    dot = max(0.0, min(1.0, dot))
    ang = 2.0 * math.acos(dot)
    return float(ang * 180.0 / math.pi)


def _cloud_points(msg: PointCloud2) -> int:
    # schnell: organized/unorganized -> width*height
    return int(msg.width) * int(msg.height)


def _read_ply_vertex_count(path: str) -> Optional[int]:
    """
    Reads PLY header and extracts 'element vertex N'.
    Works for ASCII and binary (header is ASCII).
    """
    try:
        with open(path, "rb") as f:
            for _ in range(200):  # header is short
                line = f.readline()
                if not line:
                    return None
                try:
                    s = line.decode("ascii", errors="ignore").strip()
                except Exception:
                    continue
                if s.startswith("element vertex"):
                    parts = s.split()
                    if len(parts) == 3:
                        return int(parts[2])
                if s == "end_header":
                    break
    except Exception:
        return None
    return None


# -----------------------------
# Data containers
# -----------------------------
@dataclass
class BufEntry:
    t_match: float  # time used for matching (header stamp if valid else rx time)
    t_rx: float
    msg: object


@dataclass
class PlantAccum:
    views: set
    sum_points_all: int
    sum_points_selected: int
    frames: int


# -----------------------------
# Node
# -----------------------------
class ClusterDiagnostic(Node):
    def __init__(self) -> None:
        super().__init__("cluster_diagnostic")

        # Params
        self.declare_parameter("frame_info_topic", "/seg/frame_info_depth_masked")
        self.declare_parameter("pose_topic", "/camera_pose_world")
        self.declare_parameter("cloud_topic_all", "/seg/strawberry_cloud")
        self.declare_parameter("cloud_topic_selected", "/seg/strawberry_cloud_selected")

        self.declare_parameter("match_slop_s", 0.2)          # passt zu deiner Pipeline (sync_slop=0.2)
        self.declare_parameter("buffer_size", 500)
        self.declare_parameter("pos_tol_m", 0.001)
        self.declare_parameter("rot_tol_deg", 0.5)

        self.declare_parameter("log_file", "cluster_diagnostic.txt")

        # PLY scan (fused Ergebnis)
        self.declare_parameter("output_dir", "/home/parallels/strawberry_ply")
        self.declare_parameter("scan_ply_on_plant_change", True)

        # Summary
        self.declare_parameter("summary_period_s", 5.0)
        self.declare_parameter("warn_first_n_mismatches", 10)

        self.frame_info_topic = str(self.get_parameter("frame_info_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.cloud_topic_all = str(self.get_parameter("cloud_topic_all").value)
        self.cloud_topic_selected = str(self.get_parameter("cloud_topic_selected").value)

        self.match_slop_s = float(self.get_parameter("match_slop_s").value)
        self.buffer_size = int(self.get_parameter("buffer_size").value)
        self.pos_tol_m = float(self.get_parameter("pos_tol_m").value)
        self.rot_tol_deg = float(self.get_parameter("rot_tol_deg").value)

        self.log_file = str(self.get_parameter("log_file").value)
        self.output_dir = str(self.get_parameter("output_dir").value)
        self.scan_ply_on_plant_change = bool(self.get_parameter("scan_ply_on_plant_change").value)

        self.summary_period_s = float(self.get_parameter("summary_period_s").value)
        self.warn_first_n_mismatches = int(self.get_parameter("warn_first_n_mismatches").value)

        # Buffers
        self.pose_buf: Deque[BufEntry] = deque(maxlen=self.buffer_size)
        self.cloud_all_buf: Deque[BufEntry] = deque(maxlen=self.buffer_size)
        self.cloud_sel_buf: Deque[BufEntry] = deque(maxlen=self.buffer_size)

        # Last
        self.last_frame_info: Optional[FrameInfo] = None
        self.last_pose: Optional[PoseStamped] = None
        self.last_cloud_all_pts: Optional[int] = None
        self.last_cloud_sel_pts: Optional[int] = None

        # Counters
        self.rx_pose = 0
        self.rx_fi = 0
        self.rx_cloud_all = 0
        self.rx_cloud_sel = 0

        self.comparisons = 0
        self.mismatches = 0
        self.warned = 0

        # Plant accumulation
        self.current_plant: Optional[int] = None
        self.plant_accum: dict[int, PlantAccum] = {}

        # Log file
        self.log_path = os.path.abspath(self.log_file)
        self._fh = open(self.log_path, "a", encoding="utf-8")
        self._fh.write(f"\n# ===== ClusterDiagnostic start {_now_iso()} =====\n")
        self._fh.write(
            "# time_iso\tplant\tview\tframe\t"
            "dt_pose_s\tdt_cloud_all_s\tdt_cloud_sel_s\t"
            "pos_diff_m\trot_diff_deg\t"
            "pts_all\tpts_sel\tnote\n"
        )
        self._fh.flush()

        # Subs
        self.create_subscription(PoseStamped, self.pose_topic, self._pose_cb, 50)
        self.create_subscription(FrameInfo, self.frame_info_topic, self._fi_cb, 50)
        self.create_subscription(PointCloud2, self.cloud_topic_all, self._cloud_all_cb, 10)
        self.create_subscription(PointCloud2, self.cloud_topic_selected, self._cloud_sel_cb, 10)

        # Timer summary
        self.create_timer(self.summary_period_s, self._summary)

        self.get_logger().info("🚀 Cluster-Diagnostic gestartet")
        self.get_logger().info(f"  frame_info_topic     = {self.frame_info_topic}")
        self.get_logger().info(f"  pose_topic           = {self.pose_topic}")
        self.get_logger().info(f"  cloud_topic_all      = {self.cloud_topic_all}")
        self.get_logger().info(f"  cloud_topic_selected = {self.cloud_topic_selected}")
        self.get_logger().info(f"  match_slop_s         = {self.match_slop_s}")
        self.get_logger().info(f"  log_file             = {self.log_path}")
        self.get_logger().info(f"  output_dir           = {self.output_dir}")

    # -----------------------------
    # Buffering callbacks
    # -----------------------------
    def _pose_cb(self, msg: PoseStamped) -> None:
        self.rx_pose += 1
        self.last_pose = msg

        t_rx = self.get_clock().now().nanoseconds * 1e-9
        t_h = _valid_header_stamp_sec(msg)
        t_match = t_h if t_h is not None else t_rx
        self.pose_buf.append(BufEntry(t_match=t_match, t_rx=t_rx, msg=msg))

    def _cloud_all_cb(self, msg: PointCloud2) -> None:
        self.rx_cloud_all += 1
        pts = _cloud_points(msg)
        self.last_cloud_all_pts = pts

        t_rx = self.get_clock().now().nanoseconds * 1e-9
        t_h = _valid_header_stamp_sec(msg)
        t_match = t_h if t_h is not None else t_rx
        self.cloud_all_buf.append(BufEntry(t_match=t_match, t_rx=t_rx, msg=msg))

    def _cloud_sel_cb(self, msg: PointCloud2) -> None:
        self.rx_cloud_sel += 1
        pts = _cloud_points(msg)
        self.last_cloud_sel_pts = pts

        t_rx = self.get_clock().now().nanoseconds * 1e-9
        t_h = _valid_header_stamp_sec(msg)
        t_match = t_h if t_h is not None else t_rx
        self.cloud_sel_buf.append(BufEntry(t_match=t_match, t_rx=t_rx, msg=msg))

    # -----------------------------
    # Matching helpers
    # -----------------------------
    @staticmethod
    def _nearest(buf: Deque[BufEntry], t_ref: float) -> Tuple[Optional[BufEntry], float]:
        if not buf:
            return None, float("nan")

        best: Optional[BufEntry] = None
        best_dt = float("inf")
        for e in buf:
            dt = e.t_match - t_ref
            adt = abs(dt)
            if adt < best_dt:
                best_dt = adt
                best = e
        return best, float(best_dt)

    def _fi_cb(self, msg: FrameInfo) -> None:
        self.rx_fi += 1
        self.last_frame_info = msg

        # Reference time for matching: header stamp if valid else rx-time
        t_rx = self.get_clock().now().nanoseconds * 1e-9
        t_h = _valid_header_stamp_sec(msg)
        t_ref = t_h if t_h is not None else t_rx

        plant = int(getattr(msg, "plant_id", -1))
        view = int(getattr(msg, "view_id", -1))
        frame = int(getattr(msg, "frame_index", -1))

        # Plant change handling (scan PLYs)
        if self.current_plant is None:
            self.current_plant = plant
        elif plant != self.current_plant:
            old = self.current_plant
            self._log_line(old, -1, -1, float("nan"), float("nan"), float("nan"),
                           float("nan"), float("nan"), None, None,
                           note=f"PLANT_CHANGE {old} -> {plant}")
            if self.scan_ply_on_plant_change:
                self._scan_and_log_plys(old)
            self.current_plant = plant

        # Update per-plant accumulation container
        if plant not in self.plant_accum:
            self.plant_accum[plant] = PlantAccum(views=set(), sum_points_all=0, sum_points_selected=0, frames=0)
        self.plant_accum[plant].views.add(view)
        self.plant_accum[plant].frames += 1

        # Nearest pose + clouds
        pose_e, dt_pose = self._nearest(self.pose_buf, t_ref)
        all_e, dt_all = self._nearest(self.cloud_all_buf, t_ref)
        sel_e, dt_sel = self._nearest(self.cloud_sel_buf, t_ref)

        # Extract points
        pts_all = None
        pts_sel = None
        if all_e is not None and math.isfinite(dt_all) and dt_all <= self.match_slop_s:
            pts_all = _cloud_points(all_e.msg)  # type: ignore[arg-type]
            self.plant_accum[plant].sum_points_all += int(pts_all)
        if sel_e is not None and math.isfinite(dt_sel) and dt_sel <= self.match_slop_s:
            pts_sel = _cloud_points(sel_e.msg)  # type: ignore[arg-type]
            self.plant_accum[plant].sum_points_selected += int(pts_sel)

        # Compare poses (FrameInfo has camera_pose_world)
        pos_diff = float("nan")
        rot_diff = float("nan")
        note = "OK"

        if pose_e is None or (math.isfinite(dt_pose) and dt_pose > self.match_slop_s):
            note = "NO_MATCH_POSE"
        else:
            pose_msg: PoseStamped = pose_e.msg  # type: ignore[assignment]
            fi_pose = msg.camera_pose_world
            ps_pose = pose_msg.pose

            fi_pos, fi_q = _pose_to_np(fi_pose)
            ps_pos, ps_q = _pose_to_np(ps_pose)

            pos_diff = float(np.linalg.norm(fi_pos - ps_pos))
            rot_diff = _rotation_diff_deg(fi_q, ps_q)

            self.comparisons += 1

            if pos_diff > self.pos_tol_m or rot_diff > self.rot_tol_deg:
                self.mismatches += 1
                note = "MISMATCH"
                if self.warned < self.warn_first_n_mismatches:
                    self.warned += 1
                    self.get_logger().warning(
                        f"⚠️ MISMATCH plant={plant} view={view} frame={frame} "
                        f"pos={pos_diff:.4f}m rot={rot_diff:.1f}deg dt_pose={dt_pose:.3f}s"
                    )

        # If clouds not matched, annotate
        if pts_all is None:
            note += "|NO_MATCH_CLOUD_ALL"
        if pts_sel is None:
            note += "|NO_MATCH_CLOUD_SEL"

        self._log_line(
            plant=plant,
            view=view,
            frame=frame,
            dt_pose=dt_pose,
            dt_all=dt_all,
            dt_sel=dt_sel,
            pos_diff=pos_diff,
            rot_diff=rot_diff,
            pts_all=pts_all,
            pts_sel=pts_sel,
            note=note,
        )

    # -----------------------------
    # Logging
    # -----------------------------
    def _log_line(
        self,
        plant: int,
        view: int,
        frame: int,
        dt_pose: float,
        dt_all: float,
        dt_sel: float,
        pos_diff: float,
        rot_diff: float,
        pts_all: Optional[int],
        pts_sel: Optional[int],
        note: str,
    ) -> None:
        pts_all_str = str(pts_all) if pts_all is not None else "nan"
        pts_sel_str = str(pts_sel) if pts_sel is not None else "nan"

        line = (
            f"{_now_iso()}\t{plant}\t{view}\t{frame}\t"
            f"{dt_pose:.6f}\t{dt_all:.6f}\t{dt_sel:.6f}\t"
            f"{pos_diff:.6f}\t{rot_diff:.3f}\t"
            f"{pts_all_str}\t{pts_sel_str}\t{note}\n"
        )
        self._fh.write(line)
        self._fh.flush()

    def _scan_and_log_plys(self, plant: int) -> None:
        plant_dir = os.path.join(self.output_dir, f"plant_{plant:03d}")
        if not os.path.isdir(plant_dir):
            self._fh.write(f"{_now_iso()}\t{plant}\t-1\t-1\t"
                           f"nan\tnan\tnan\tnan\tnan\tnan\tnan\tNO_PLY_DIR {plant_dir}\n")
            self._fh.flush()
            return

        ply_paths = sorted(glob.glob(os.path.join(plant_dir, "*.ply")))
        if not ply_paths:
            self._fh.write(f"{_now_iso()}\t{plant}\t-1\t-1\t"
                           f"nan\tnan\tnan\tnan\tnan\tnan\tnan\tNO_PLY_FILES {plant_dir}\n")
            self._fh.flush()
            return

        self._fh.write(f"# PLY_SCAN plant={plant} dir={plant_dir} time={_now_iso()}\n")
        for p in ply_paths:
            n = _read_ply_vertex_count(p)
            n_str = str(n) if n is not None else "?"
            self._fh.write(f"#   {os.path.basename(p)}\tvertices={n_str}\n")
        self._fh.flush()

    # -----------------------------
    # Summary
    # -----------------------------
    def _summary(self) -> None:
        plant = self.current_plant if self.current_plant is not None else -1
        views = sorted(self.plant_accum.get(plant, PlantAccum(set(), 0, 0, 0)).views) if plant in self.plant_accum else []

        mismatch_rate = 100.0 * (self.mismatches / self.comparisons) if self.comparisons > 0 else 0.0

        self.get_logger().info(
            "\n📊 DIAGNOSTIC SUMMARY\n"
            f"   Rx: pose={self.rx_pose}, frame_info={self.rx_fi}, cloud_all={self.rx_cloud_all}, cloud_sel={self.rx_cloud_sel}\n"
            f"   Comparisons={self.comparisons}, mismatches={self.mismatches} ({mismatch_rate:.2f}%)\n"
            f"   Current plant={plant}, views_seen={views}\n"
            f"   Last cloud pts: all={self.last_cloud_all_pts}, sel={self.last_cloud_sel_pts}\n"
            f"   Log: {self.log_path}"
        )

    # -----------------------------
    # Shutdown
    # -----------------------------
    def destroy_node(self) -> bool:
        try:
            if self.current_plant is not None and self.scan_ply_on_plant_change:
                # Final scan for current plant (falls beim Shutdown geschrieben wird)
                self._scan_and_log_plys(self.current_plant)
            self._fh.write(f"# ===== ClusterDiagnostic stop {_now_iso()} =====\n")
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = ClusterDiagnostic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("👋 Diagnose beendet")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
