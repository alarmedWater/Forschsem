#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Folder-based "dummy camera" publisher (ROS 2).

Publishes:
  - /camera/color/image_raw (sensor_msgs/Image, rgb8)
  - /camera/aligned_depth_to_color/image_raw (sensor_msgs/Image, 16UC1)
  - /camera/color/camera_info (sensor_msgs/CameraInfo)
  - /camera_pose_world (geometry_msgs/PoseStamped) [optional]
  - /camera/frame_info (strawberry_msgs/FrameInfo) [optional]
    fallback if msg missing: /camera/frame_info_json (std_msgs/String)

Dataset modes:
  - Mode A: plants_root_dir + plant_* folders containing color_*.png and depth_*.png
  - Mode B: rgb_dir/depth_dir flat folders

Pose notes:
  This version uses measured robot TRF/TCP poses (WRF=BRF) for view_id {0,1,2}
  and converts robot Euler angles (XY'Z'' convention) into quaternions.
"""

from __future__ import annotations

import glob
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, String

try:
    from strawberry_msgs.msg import FrameInfo  # type: ignore
except Exception:  # noqa: BLE001
    FrameInfo = None  # type: ignore


@dataclass(frozen=True)
class FrameItem:
    frame_index: int
    plant_id: int
    view_id: int
    rgb_path: str
    depth_path: Optional[str]


def to_image_msg_rgb8(arr: np.ndarray, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = arr.tobytes()
    return msg


def to_image_msg_depth16(arr_u16: np.ndarray, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height, msg.width = arr_u16.shape[:2]
    msg.encoding = "16UC1"
    msg.is_bigendian = 0
    msg.step = msg.width * 2
    msg.data = arr_u16.tobytes()
    return msg


_INDEX_RE = re.compile(r".*_(\d+)(?:\.[^.]+)?$")


def extract_index(path: str) -> int:
    name = Path(path).name
    m = _INDEX_RE.match(name)
    if not m:
        return 10**9
    try:
        return int(m.group(1))
    except ValueError:
        return 10**9


def rotation_matrix_to_quaternion(rot: np.ndarray) -> Tuple[float, float, float, float]:
    r = rot.astype(np.float64)
    tr = r[0, 0] + r[1, 1] + r[2, 2]

    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif (r[0, 0] > r[1, 1]) and (r[0, 0] > r[2, 2]):
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s

    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n <= 0.0:
        return (0.0, 0.0, 0.0, 1.0)

    return (float(qx / n), float(qy / n), float(qz / n), float(qw / n))


def _rotx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def _roty(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def _rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def euler_xyprime_zdoubleprime_to_rot(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """
    Convert robot Euler angles in XY'Z'' convention to a rotation matrix.

    We treat XY'Z'' as intrinsic XYZ:
      R = Rx(rx) @ Ry(ry) @ Rz(rz)
    """
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    return _rotx(rx) @ _roty(ry) @ _rotz(rz)


class CameraFolderNode(Node):
    """Replay RGB + depth frames and publish as ROS camera topics."""

    # Updated TRF/TCP poses (WRF=BRF) measured on robot:
    # - translation in mm (t_mm)
    # - orientation in degrees (r_deg) as (rx, ry, rz) in XY'Z'' convention
    _POSES: Dict[int, Dict[str, Any]] = {
        0: {  # links
            "name": "links",
            "t_mm": (188.205, 58.384, 249.002),
            "r_deg": (63.286, 55.563, 127.337),
        },
        1: {  # mitte
            "name": "mitte",
            "t_mm": (148.567, 0.0, 283.585),
            "r_deg": (-180.0, 90.0, 0.0),
        },
        2: {  # rechts
            "name": "rechts",
            "t_mm": (195.183, -34.680, 243.936),
            "r_deg": (-71.161, 50.263, -104.430),
        },
    }

    def __init__(self) -> None:
        super().__init__("camera_folder")

        # ---------------- Parameters ----------------
        self.declare_parameter("rgb_dir", "")
        self.declare_parameter("depth_dir", "")
        self.declare_parameter("rgb_pattern", "color_*.png")
        self.declare_parameter("depth_pattern", "depth_*.png")

        self.declare_parameter(
            "plants_root_dir",
            "/home/parallels/Forschsemrep/strawberry_ws/data/plant_views",
        )
        self.declare_parameter("plant_glob", "plant_*")
        self.declare_parameter("use_plants_root", True)

        self.declare_parameter("fps", 2.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("publish_depth", True)

        # Fallback intrinsics (YAML preferred!)
        self.declare_parameter("fx", 900.0)
        self.declare_parameter("fy", 900.0)
        self.declare_parameter("cx", 640.0)
        self.declare_parameter("cy", 360.0)

        self.declare_parameter("frame_color", "camera_color_optical_frame")
        self.declare_parameter("frame_depth", "camera_color_optical_frame")

        self.declare_parameter("calib_yaml", "")
        self.declare_parameter("camera_info_source", "color")

        self.declare_parameter("publish_pose", True)
        self.declare_parameter("pose_topic", "/camera_pose_world")
        self.declare_parameter("world_frame_id", "world")

        self.declare_parameter("publish_frame_info", True)
        self.declare_parameter("frame_info_topic", "/camera/frame_info")
        self.declare_parameter("frame_info_json_topic", "/camera/frame_info_json")

        # ---------------- Publishers ----------------
        self.pub_rgb = self.create_publisher(Image, "/camera/color/image_raw", 10)
        self.pub_depth = self.create_publisher(
            Image, "/camera/aligned_depth_to_color/image_raw", 10
        )
        self.pub_info = self.create_publisher(CameraInfo, "/camera/color/camera_info", 10)

        self._publish_pose = self._param_bool("publish_pose", True)
        self._pose_topic = self._param_str("pose_topic", "/camera_pose_world")
        self._world_frame_id = self._param_str("world_frame_id", "world")
        self.pub_pose = (
            self.create_publisher(PoseStamped, self._pose_topic, 10)
            if self._publish_pose
            else None
        )

        self._publish_frame_info = self._param_bool("publish_frame_info", True)
        self._frame_info_topic = self._param_str("frame_info_topic", "/camera/frame_info")
        self._frame_info_json_topic = self._param_str(
            "frame_info_json_topic", "/camera/frame_info_json"
        )

        self.pub_frame_info = None
        self.pub_frame_info_json = None
        if self._publish_frame_info:
            if FrameInfo is not None:
                self.pub_frame_info = self.create_publisher(
                    FrameInfo, self._frame_info_topic, 10
                )
            else:
                self.get_logger().warning(
                    "FrameInfo msg not available. Falling back to JSON String on "
                    "'frame_info_json_topic'."
                )
                self.pub_frame_info_json = self.create_publisher(
                    String, self._frame_info_json_topic, 10
                )

        # ---------------- Load calibration (optional) ----------------
        self._calib: Optional[Dict[str, Any]] = None
        self._load_calibration()

        # ---------------- Gather frames ----------------
        self._frames = self._collect_frames()
        if not self._frames:
            raise RuntimeError(
                "No frames found. Check plants_root_dir (mode A) or rgb_dir/depth_dir (mode B)."
            )

        # ---------------- Timer ----------------
        fps = float(self._param_float("fps", 2.0))
        fps = max(fps, 0.1)
        self._period_s = max(1.0 / fps, 1e-3)
        self._timer = self.create_timer(self._period_s, self._tick)

        self._i = 0
        self._done = False
        self._logged_depth_warn = False

        self.get_logger().info(
            "camera_folder started:\n"
            f"  frames             = {len(self._frames)}\n"
            f"  fps                = {fps:.3f}\n"
            f"  loop               = {self._param_bool('loop', False)}\n"
            f"  publish_depth      = {self._param_bool('publish_depth', True)}\n"
            f"  publish_pose       = {self._publish_pose} ({self._pose_topic})\n"
            f"  publish_frame_info = {self._publish_frame_info}\n"
        )

    # ------------------------------------------------------------------ #
    # Parameter helpers
    # ------------------------------------------------------------------ #

    def _param_float(self, name: str, default: float) -> float:
        val: Any = self.get_parameter(name).value
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _param_str(self, name: str, default: str = "") -> str:
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

    # ------------------------------------------------------------------ #
    # YAML helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _as_float(value: Any, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _as_float_list(
        values: Any, fallback: Sequence[float], length: int = 5
    ) -> list[float]:
        out: list[float] = []
        if isinstance(values, (list, tuple)):
            for v in values:
                try:
                    out.append(float(v))
                except (TypeError, ValueError):
                    out.append(0.0)
        else:
            out = list(fallback)

        if len(out) < length:
            out += [0.0] * (length - len(out))
        return out[:length]

    # ------------------------------------------------------------------ #
    # Calibration loading
    # ------------------------------------------------------------------ #

    def _load_calibration(self) -> None:
        self._calib = None
        calib_path_param = self._param_str("calib_yaml", "").strip()

        if calib_path_param:
            calib_path = calib_path_param
        else:
            try:
                share_dir = get_package_share_directory("strawberry_camera")
                calib_path = str(
                    Path(share_dir) / "config" / "realsense_d405_640x480_30fps.yml"
                )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(
                    "Could not resolve 'strawberry_camera' share dir. "
                    "Falling back to parameter intrinsics. "
                    f"({exc})"
                )
                return

        calib_file = Path(calib_path)
        if not calib_file.is_file():
            self.get_logger().warning(
                f"Calibration YAML not found at '{calib_path}'. Falling back to parameter intrinsics."
            )
            return

        try:
            with calib_file.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError("Calibration YAML root must be a mapping/dict.")
            self._calib = data
            source = self._param_str("camera_info_source", "color")
            self.get_logger().info(
                f"Loaded calibration YAML: {calib_path} (camera_info_source={source})"
            )
        except Exception as exc:  # noqa: BLE001
            self._calib = None
            self.get_logger().warning(
                f"Could not load calibration YAML '{calib_path}'. Falling back to "
                f"parameter intrinsics. ({exc})"
            )

    # ------------------------------------------------------------------ #
    # Frame collection
    # ------------------------------------------------------------------ #

    def _collect_frames(self) -> list[FrameItem]:
        use_plants_root = self._param_bool("use_plants_root", True)
        plants_root_dir = self._param_str("plants_root_dir", "")
        plant_glob = self._param_str("plant_glob", "plant_*")
        rgb_pattern = self._param_str("rgb_pattern", "color_*.png")
        depth_pattern = self._param_str("depth_pattern", "depth_*.png")

        if use_plants_root and plants_root_dir:
            root = Path(plants_root_dir)
            if root.is_dir():
                return self._collect_frames_plants_root(
                    root, plant_glob, rgb_pattern, depth_pattern
                )

            self.get_logger().warning(
                f"use_plants_root=True but plants_root_dir is not a directory: {plants_root_dir}"
            )

        rgb_dir = self._param_str("rgb_dir", "")
        depth_dir = self._param_str("depth_dir", "")

        return self._collect_frames_flat(
            rgb_dir=Path(rgb_dir) if rgb_dir else Path(),
            depth_dir=Path(depth_dir) if depth_dir else None,
            rgb_pattern=rgb_pattern,
            depth_pattern=depth_pattern,
        )

    def _collect_frames_plants_root(
        self,
        root: Path,
        plant_glob: str,
        rgb_pattern: str,
        depth_pattern: str,
    ) -> list[FrameItem]:
        plant_dirs = [p for p in root.glob(plant_glob) if p.is_dir()]
        plant_dirs.sort(key=lambda p: (extract_index(str(p)), p.name))

        frames: list[FrameItem] = []
        frame_index = 0

        for plant_id, plant_dir in enumerate(plant_dirs):
            rgb_files = sorted(
                glob.glob(str(plant_dir / rgb_pattern)),
                key=lambda s: (extract_index(s), s),
            )
            depth_files = sorted(
                glob.glob(str(plant_dir / depth_pattern)),
                key=lambda s: (extract_index(s), s),
            )

            rgb_by_idx = {extract_index(p): p for p in rgb_files}
            depth_by_idx = {extract_index(p): p for p in depth_files}

            indices = sorted(rgb_by_idx.keys())
            for idx_in_folder in indices:
                rgb_path = rgb_by_idx[idx_in_folder]
                depth_path = depth_by_idx.get(idx_in_folder)

                view_id = int(idx_in_folder)
                if view_id not in (0, 1, 2):
                    view_id = view_id % 3

                frames.append(
                    FrameItem(
                        frame_index=frame_index,
                        plant_id=plant_id,
                        view_id=view_id,
                        rgb_path=rgb_path,
                        depth_path=depth_path,
                    )
                )
                frame_index += 1

        if frames:
            self.get_logger().info(
                f"Collected {len(frames)} frames from plants_root_dir='{root}' "
                f"(plants={len(plant_dirs)})"
            )
        return frames

    def _collect_frames_flat(
        self,
        rgb_dir: Path,
        depth_dir: Optional[Path],
        rgb_pattern: str,
        depth_pattern: str,
    ) -> list[FrameItem]:
        if not rgb_dir.is_dir():
            self.get_logger().warning("rgb_dir is empty or not a directory (flat mode).")
            return []

        rgb_paths = sorted(
            glob.glob(str(rgb_dir / rgb_pattern)),
            key=lambda s: (extract_index(s), s),
        )

        depth_paths: list[str] = []
        if depth_dir and depth_dir.is_dir():
            depth_paths = sorted(
                glob.glob(str(depth_dir / depth_pattern)),
                key=lambda s: (extract_index(s), s),
            )

        if not rgb_paths:
            return []

        if depth_paths and len(depth_paths) != len(rgb_paths):
            self.get_logger().warning(
                f"Depth/RGB count mismatch ({len(depth_paths)} vs {len(rgb_paths)}). "
                "Depth will be published by index if present."
            )

        frames: list[FrameItem] = []
        for i, rgb_path in enumerate(rgb_paths):
            depth_path = depth_paths[i] if i < len(depth_paths) else None
            frames.append(
                FrameItem(
                    frame_index=i,
                    plant_id=i // 3,
                    view_id=i % 3,
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                )
            )
        self.get_logger().info(
            f"Collected {len(frames)} frames from flat folders: rgb_dir='{rgb_dir}'"
        )
        return frames

    # ------------------------------------------------------------------ #
    # CameraInfo
    # ------------------------------------------------------------------ #

    def _make_camera_info(self, stamp, w: int, h: int) -> CameraInfo:
        frame = self._param_str("frame_color", "camera_color_optical_frame")

        msg = CameraInfo()
        msg.header = Header(stamp=stamp, frame_id=frame)
        msg.width = int(w)
        msg.height = int(h)

        fx_fb = self._param_float("fx", 900.0)
        fy_fb = self._param_float("fy", 900.0)
        cx_fb = self._param_float("cx", 640.0)
        cy_fb = self._param_float("cy", 360.0)

        if self._calib is None:
            fx, fy, cx, cy = fx_fb, fy_fb, cx_fb, cy_fb
            msg.distortion_model = "plumb_bob"
            msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            src = self._param_str("camera_info_source", "color").lower()
            if src not in ("color", "depth"):
                self.get_logger().warning(
                    f"camera_info_source='{src}' invalid. Using 'color'."
                )
                src = "color"

            intr_any = self._calib.get("intrinsics", {}).get(src, {})
            intr: Dict[str, Any] = intr_any if isinstance(intr_any, dict) else {}

            fx = self._as_float(intr.get("fx"), fx_fb)
            fy = self._as_float(intr.get("fy"), fy_fb)
            cx = self._as_float(intr.get("cx"), cx_fb)
            cy = self._as_float(intr.get("cy"), cy_fb)

            msg.d = self._as_float_list(
                intr.get("distortion_coeffs"),
                fallback=[0.0, 0.0, 0.0, 0.0, 0.0],
                length=5,
            )
            msg.distortion_model = "plumb_bob"

        msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    # ------------------------------------------------------------------ #
    # Pose + FrameInfo publishing
    # ------------------------------------------------------------------ #

    def _make_pose_msg(self, stamp, view_id: int) -> PoseStamped:
        pose_def = self._POSES.get(int(view_id), self._POSES[1])

        # Translation (mm -> m)
        x_mm, y_mm, z_mm = pose_def["t_mm"]
        t_m = (float(x_mm) / 1000.0, float(y_mm) / 1000.0, float(z_mm) / 1000.0)

        # Rotation from robot Euler (degrees)
        rx_deg, ry_deg, rz_deg = pose_def["r_deg"]
        rot = euler_xyprime_zdoubleprime_to_rot(rx_deg, ry_deg, rz_deg)

        # Orthonormalize (robust)
        u, _, vt = np.linalg.svd(rot)
        rot_ortho = (u @ vt).astype(np.float64)

        qx, qy, qz, qw = rotation_matrix_to_quaternion(rot_ortho)

        msg = PoseStamped()
        msg.header = Header(stamp=stamp, frame_id=self._world_frame_id)
        msg.pose.position.x = t_m[0]
        msg.pose.position.y = t_m[1]
        msg.pose.position.z = t_m[2]
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    def _publish_frame_info_msg(
        self,
        stamp,
        frame_id_for_info: str,
        item: FrameItem,
        pose_msg: Optional[PoseStamped],
    ) -> None:
        if not self._publish_frame_info:
            return

        # IMPORTANT: FrameInfo should carry the *camera/image* frame_id (not world)
        if self.pub_frame_info is not None and FrameInfo is not None:
            msg = FrameInfo()
            msg.header = Header(stamp=stamp, frame_id=frame_id_for_info)
            msg.frame_index = int(item.frame_index)
            msg.plant_id = int(item.plant_id)
            msg.view_id = int(item.view_id)
            msg.rgb_path = str(item.rgb_path)
            msg.depth_path = str(item.depth_path) if item.depth_path else ""

            # NEW: hard-coupled pose (same stamp as images)
            if pose_msg is not None:
                msg.camera_pose_world = pose_msg.pose
                msg.world_frame_id = str(self._world_frame_id)
            else:
                msg.world_frame_id = str(self._world_frame_id)

            self.pub_frame_info.publish(msg)
            return

        if self.pub_frame_info_json is not None:
            payload: Dict[str, Any] = {
                "frame_index": int(item.frame_index),
                "plant_id": int(item.plant_id),
                "view_id": int(item.view_id),
                "rgb_path": str(item.rgb_path),
                "depth_path": str(item.depth_path) if item.depth_path else "",
            }

            if pose_msg is not None:
                payload["camera_pose_world"] = {
                    "position": {
                        "x": float(pose_msg.pose.position.x),
                        "y": float(pose_msg.pose.position.y),
                        "z": float(pose_msg.pose.position.z),
                    },
                    "orientation": {
                        "x": float(pose_msg.pose.orientation.x),
                        "y": float(pose_msg.pose.orientation.y),
                        "z": float(pose_msg.pose.orientation.z),
                        "w": float(pose_msg.pose.orientation.w),
                    },
                }
                payload["world_frame_id"] = str(self._world_frame_id)
            else:
                payload["world_frame_id"] = str(self._world_frame_id)

            out = String()
            out.data = json.dumps(payload)
            self.pub_frame_info_json.publish(out)

    # ------------------------------------------------------------------ #
    # Main tick
    # ------------------------------------------------------------------ #

    def _tick(self) -> None:
        if self._done:
            return

        loop = self._param_bool("loop", False)
        publish_depth = self._param_bool("publish_depth", True)

        idx = self._i
        if idx >= len(self._frames):
            if loop:
                idx = idx % len(self._frames)
                self._i = idx
            else:
                self.get_logger().info("Finished all frames (loop=False). Stopping timer.")
                self._timer.cancel()
                self._done = True
                return

        item = self._frames[idx]

        stamp = self.get_clock().now().to_msg()
        frame_color = self._param_str("frame_color", "camera_color_optical_frame")
        frame_depth = self._param_str("frame_depth", "camera_color_optical_frame")

        bgr = cv2.imread(item.rgb_path, cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warning(f"Failed to read RGB: {item.rgb_path}")
            self._i += 1
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        rgb_msg = to_image_msg_rgb8(rgb, stamp, frame_color)

        depth_msg = None
        if publish_depth and item.depth_path:
            depth_raw = cv2.imread(item.depth_path, cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                if not self._logged_depth_warn:
                    self.get_logger().warning(
                        f"Failed to read depth: {item.depth_path} (publishing RGB only)."
                    )
                    self._logged_depth_warn = True
            else:
                if depth_raw.ndim == 3:
                    depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)

                if depth_raw.dtype != np.uint16:
                    depth_raw = (
                        (depth_raw.astype(np.float32) * 1000.0)
                        .clip(0, 65535)
                        .astype(np.uint16)
                    )

                if depth_raw.shape[:2] != (height, width):
                    depth_raw = cv2.resize(
                        depth_raw,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )

                depth_msg = to_image_msg_depth16(depth_raw, stamp, frame_depth)

        info_msg = self._make_camera_info(stamp, width, height)

        # NEW: compute pose once (needed for pose topic AND FrameInfo)
        pose_msg: Optional[PoseStamped] = None
        if self._publish_pose or self._publish_frame_info:
            pose_msg = self._make_pose_msg(stamp, item.view_id)

        # Pose topic (optional)
        if self.pub_pose is not None and pose_msg is not None:
            self.pub_pose.publish(pose_msg)

        # FrameInfo should be aligned with image stamp and use camera frame_id
        self._publish_frame_info_msg(stamp, frame_color, item, pose_msg)

        # Publish streams
        self.pub_rgb.publish(rgb_msg)
        if depth_msg is not None:
            self.pub_depth.publish(depth_msg)
        self.pub_info.publish(info_msg)

        self._i += 1


def main() -> None:
    rclpy.init()
    node = CameraFolderNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
