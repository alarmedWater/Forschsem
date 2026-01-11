#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from strawberry_msgs.msg import FrameInfo


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


class RawCloudSaver(Node):
    """
    Speichert pro Plant genau 3 ungeclusterte Punktwolken:
      plant_XXX/cloud_0.ply
      plant_XXX/cloud_1.ply
      plant_XXX/cloud_2.ply

    Quelle: PointCloud2 Topic aus strawberry_features (ungeclustert) + FrameInfo.
    Optional transformiert er von Camera -> World mit FrameInfo.camera_pose_world.
    """

    def __init__(self) -> None:
        super().__init__("raw_cloud_saver")

        # -------- Params --------
        self.declare_parameter("cloud_topic", "/seg/strawberry_cloud")
        self.declare_parameter("frame_info_topic", "/seg/frame_info_depth_masked")
        self.declare_parameter("output_dir_raw", str(Path.home() / "strawberry_raw_ply"))

        self.declare_parameter("sync_queue_size", 200)
        self.declare_parameter("sync_slop", 0.2)

        # Speichere pro (plant, view) nur ein einziges Mal
        self.declare_parameter("save_once_per_view", True)
        self.declare_parameter("overwrite", False)

        # Cloud-Frame Handling
        # assume_cloud_in: "camera" oder "world"
        # export_cloud_in: "camera" oder "world"
        self.declare_parameter("assume_cloud_in", "camera")
        self.declare_parameter("export_cloud_in", "world")

        # PLY
        self.declare_parameter("ply_ascii", True)

        self._cloud_topic = self._pstr("cloud_topic", "/seg/strawberry_cloud")
        self._frame_info_topic = self._pstr("frame_info_topic", "/seg/frame_info_depth_masked")
        self._out_raw = Path(self._pstr("output_dir_raw", str(Path.home() / "strawberry_raw_ply")))

        self._queue = max(1, self._pint("sync_queue_size", 200))
        self._slop = float(self._pfloat("sync_slop", 0.2))

        self._save_once = self._pbool("save_once_per_view", True)
        self._overwrite = self._pbool("overwrite", False)

        self._assume_cloud_in = self._pstr("assume_cloud_in", "camera").strip().lower()
        self._export_cloud_in = self._pstr("export_cloud_in", "world").strip().lower()
        if self._assume_cloud_in not in ("camera", "world"):
            self._assume_cloud_in = "camera"
        if self._export_cloud_in not in ("camera", "world"):
            self._export_cloud_in = "world"

        self._ply_ascii = self._pbool("ply_ascii", True)

        self.get_logger().info(
            "RawCloudSaver starting:\n"
            f"  cloud_topic       = {self._cloud_topic}\n"
            f"  frame_info_topic  = {self._frame_info_topic}\n"
            f"  output_dir_raw    = {self._out_raw}\n"
            f"  save_once_per_view= {self._save_once}\n"
            f"  overwrite         = {self._overwrite}\n"
            f"  assume_cloud_in   = {self._assume_cloud_in}\n"
            f"  export_cloud_in   = {self._export_cloud_in}\n"
            f"  ply_ascii         = {self._ply_ascii}\n"
            f"  slop              = {self._slop}"
        )

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._sub_cloud = message_filters.Subscriber(self, PointCloud2, self._cloud_topic, qos_profile=qos)
        self._sub_fi = message_filters.Subscriber(self, FrameInfo, self._frame_info_topic, qos_profile=qos)

        self._ts = message_filters.ApproximateTimeSynchronizer(
            [self._sub_cloud, self._sub_fi],
            queue_size=self._queue,
            slop=self._slop,
        )
        self._ts.registerCallback(self._cb)

        # dedup: plant_id -> set(view_id)
        self._saved: Dict[int, Set[int]] = {}
        # meta for summary
        self._meta: Dict[Tuple[int, int], Dict[str, Any]] = {}

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
        except Exception:  # noqa: BLE001
            return default

    def _pfloat(self, name: str, default: float) -> float:
        v: Any = self.get_parameter(name).value
        if v is None:
            return default
        try:
            return float(v)
        except Exception:  # noqa: BLE001
            return default

    # ---------------- Core ----------------

    def _cb(self, cloud_msg: PointCloud2, fi_msg: FrameInfo) -> None:
        plant_id = int(fi_msg.plant_id)
        view_id = int(fi_msg.view_id)
        frame_index = int(fi_msg.frame_index)

        if self._save_once:
            seen = self._saved.setdefault(plant_id, set())
            if view_id in seen and not self._overwrite:
                return

        plant_dir = self._out_raw / f"plant_{plant_id:03d}"
        plant_dir.mkdir(parents=True, exist_ok=True)

        out_path = plant_dir / f"cloud_{view_id}.ply"
        if out_path.exists() and (not self._overwrite) and self._save_once:
            # falls Datei schon da ist, behandeln wir es als "gespeichert"
            self._saved.setdefault(plant_id, set()).add(view_id)
            return

        pts = self._cloud2xyz(cloud_msg)
        if pts.size == 0:
            self.get_logger().warning(f"Empty cloud for plant={plant_id} view={view_id} frame={frame_index}")
            return

        pts_out = pts
        if self._assume_cloud_in == "camera" and self._export_cloud_in == "world":
            pts_out = self._cam_to_world(pts, fi_msg)
        elif self._assume_cloud_in == "world" and self._export_cloud_in == "camera":
            self.get_logger().warning(
                "Requested world->camera transform, not implemented. Exporting as-is."
            )

        self._write_ply(out_path, pts_out, ascii_mode=self._ply_ascii)

        self._saved.setdefault(plant_id, set()).add(view_id)
        self._meta[(plant_id, view_id)] = {
            "frame_index": frame_index,
            "num_points": int(pts_out.shape[0]),
            "file": out_path.name,
            "export_frame": self._export_cloud_in,
        }
        self._write_summary_csv(plant_dir, plant_id)

        self.get_logger().info(
            f"Saved RAW cloud: plant={plant_id:03d} view={view_id} frame={frame_index} "
            f"N={int(pts_out.shape[0])} -> {out_path}"
        )

    @staticmethod
    def _cloud2xyz(msg: PointCloud2) -> np.ndarray:
        # liest x,y,z und droppt NaNs
        pts_list = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pts_list.append((float(p[0]), float(p[1]), float(p[2])))
        if not pts_list:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(pts_list, dtype=np.float32)

    @staticmethod
    def _cam_to_world(pts_cam: np.ndarray, fi_msg: FrameInfo) -> np.ndarray:
        pos = fi_msg.camera_pose_world.position
        ori = fi_msg.camera_pose_world.orientation
        r = quaternion_to_rotation_matrix(ori.x, ori.y, ori.z, ori.w)
        t = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
        return (r @ pts_cam.T).T + t

    def _write_summary_csv(self, plant_dir: Path, plant_id: int) -> None:
        csv_path = plant_dir / "raw_summary.csv"
        rows = []
        for (pid, vid), m in sorted(self._meta.items()):
            if pid != plant_id:
                continue
            rows.append([pid, vid, m["frame_index"], m["num_points"], m["export_frame"], m["file"]])

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["plant_id", "view_id", "frame_index", "num_points", "export_frame", "file"])
            w.writerows(rows)

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


def main() -> None:
    rclpy.init()
    node = RawCloudSaver()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
