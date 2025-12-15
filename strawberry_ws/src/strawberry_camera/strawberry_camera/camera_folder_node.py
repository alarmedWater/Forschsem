#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Folder-based "dummy camera" publisher.

Publishes:
  - /camera/color/image_raw (sensor_msgs/Image, rgb8)
  - /camera/aligned_depth_to_color/image_raw (sensor_msgs/Image, 16UC1)
  - /camera/color/camera_info (sensor_msgs/CameraInfo)

Calibration YAML:
  - If parameter 'calib_yaml' is set, loads that YAML.
  - Otherwise loads the default YAML from:
      <share>/strawberry_camera/config/realsense_d405_640x480_30fps.yml
  - If loading fails, falls back to fx/fy/cx/cy parameters.

Notes:
  - For aligned depth-to-color images, use camera_info_source="color" (default).
  - RealSense distortion models are mapped to ROS CameraInfo distortion_model "plumb_bob".
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header


def to_image_msg_rgb8(arr: np.ndarray, stamp, frame_id: str) -> Image:
    """Convert HxWx3 RGB uint8 array to sensor_msgs/Image (rgb8)."""
    msg = Image()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = arr.tobytes()
    return msg


def to_image_msg_depth16(arr_u16: np.ndarray, stamp, frame_id: str) -> Image:
    """Convert HxW uint16 array to sensor_msgs/Image (16UC1)."""
    msg = Image()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height, msg.width = arr_u16.shape[:2]
    msg.encoding = "16UC1"
    msg.is_bigendian = 0
    msg.step = msg.width * 2
    msg.data = arr_u16.tobytes()
    return msg


class CameraFolderNode(Node):
    """Replay RGB + depth frames from folders and publish as ROS camera topics."""

    def __init__(self) -> None:
        super().__init__("camera_folder")

        # ---------- Parameters ----------
        self.declare_parameter("rgb_dir", "")
        self.declare_parameter("depth_dir", "")
        self.declare_parameter("rgb_pattern", "color_*.png")
        self.declare_parameter("depth_pattern", "depth_*.png")
        self.declare_parameter("fps", 2.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("publish_depth", True)

        # intrinsics fallback (only used if YAML cannot be loaded)
        self.declare_parameter("fx", 900.0)
        self.declare_parameter("fy", 900.0)
        self.declare_parameter("cx", 640.0)
        self.declare_parameter("cy", 360.0)

        # frames
        self.declare_parameter("frame_color", "camera_color_optical_frame")
        self.declare_parameter("frame_depth", "camera_color_optical_frame")

        # calibration config
        self.declare_parameter("calib_yaml", "")
        self.declare_parameter("camera_info_source", "color")  # "color" or "depth"

        # ---------- Publishers ----------
        self.pub_rgb = self.create_publisher(Image, "/camera/color/image_raw", 10)
        self.pub_depth = self.create_publisher(
            Image, "/camera/aligned_depth_to_color/image_raw", 10
        )
        self.pub_info = self.create_publisher(CameraInfo, "/camera/color/camera_info", 10)

        # ---------- Load calibration (optional) ----------
        self._calib: Optional[Dict[str, Any]] = None
        self._load_calibration()

        # ---------- Gather files ----------
        self.rgb_paths, self.depth_paths = self._collect_file_pairs()
        if not self.rgb_paths:
            raise RuntimeError("No RGB images found. Check rgb_dir and rgb_pattern.")

        # ---------- Timer ----------
        fps = self._param_float("fps", 2.0)
        self.period = max(1.0 / max(fps, 0.1), 1e-3)
        self.timer = self.create_timer(self.period, self._tick)

        self.i = 0
        self.done = False
        self.logged_depth_warn = False

        self.get_logger().info(
            f"camera_folder: {len(self.rgb_paths)} RGB files, {len(self.depth_paths)} depth files "
            f"| fps={fps:.3f} loop={self.get_parameter('loop').value}"
        )

    # ------------------------------------------------------------------ #
    # Parameter helpers (Pylance/Pyright friendly)
    # ------------------------------------------------------------------ #

    def _param_float(self, name: str, default: float) -> float:
        """Read ROS parameter and return as float with safe fallback."""
        val: Any = self.get_parameter(name).value
        if val is None:
            return default

        if isinstance(val, (int, float)):
            return float(val)

        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return default

        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _param_str(self, name: str, default: str = "") -> str:
        """Read ROS parameter and return as str with safe fallback."""
        val: Any = self.get_parameter(name).value
        if val is None:
            return default
        return str(val)

    def _param_bool(self, name: str, default: bool) -> bool:
        """Read ROS parameter and return as bool with safe fallback."""
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
        """Convert value to float; fallback if None/invalid."""
        if value is None:
            return fallback
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _as_float_list(
        values: Any,
        fallback: Sequence[float],
        length: int = 5,
    ) -> list[float]:
        """Convert values to list[float] of fixed length; fallback if invalid."""
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
        """Load calibration YAML from calib_yaml or default share/<pkg>/config/..."""
        self._calib = None

        calib_path_param = self._param_str("calib_yaml", "").strip()

        if calib_path_param:
            calib_path = calib_path_param
        else:
            try:
                share_dir = get_package_share_directory("strawberry_camera")
                calib_path = str(
                    Path(share_dir)
                    / "config"
                    / "realsense_d405_640x480_30fps.yml"
                )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(
                    "Could not resolve package share directory for 'strawberry_camera'. "
                    f"Falling back to parameter intrinsics. ({exc})"
                )
                return

        calib_file = Path(calib_path)
        if not calib_file.is_file():
            self.get_logger().warning(
                f"Calibration YAML not found at '{calib_path}'. "
                "Falling back to parameter intrinsics."
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
                f"Could not load calibration YAML '{calib_path}'. "
                f"Falling back to parameter intrinsics. ({exc})"
            )

    # ------------------------------------------------------------------ #
    # File collection
    # ------------------------------------------------------------------ #

    def _collect_file_pairs(self) -> tuple[list[str], list[str]]:
        """Collect sorted RGB and depth file lists."""
        rgb_dir = self._param_str("rgb_dir", "")
        depth_dir = self._param_str("depth_dir", "")
        rgb_pattern = self._param_str("rgb_pattern", "color_*.png")
        depth_pattern = self._param_str("depth_pattern", "depth_*.png")

        rgb_paths = sorted(glob.glob(str(Path(rgb_dir) / rgb_pattern)))
        depth_paths = (
            sorted(glob.glob(str(Path(depth_dir) / depth_pattern)))
            if depth_dir
            else []
        )

        if depth_paths and len(depth_paths) != len(rgb_paths):
            self.get_logger().warning(
                f"Depth/RGB count mismatch ({len(depth_paths)} vs {len(rgb_paths)}). "
                "Will still publish RGB; depth will be published by index if present."
            )

        return rgb_paths, depth_paths

    # ------------------------------------------------------------------ #
    # CameraInfo
    # ------------------------------------------------------------------ #

    def _make_camera_info(self, stamp, w: int, h: int) -> CameraInfo:
        """Create CameraInfo from YAML (preferred) or fallback fx/fy/cx/cy."""
        frame = self._param_str("frame_color", "camera_color_optical_frame")

        msg = CameraInfo()
        msg.header = Header(stamp=stamp, frame_id=frame)
        msg.width = int(w)
        msg.height = int(h)

        # Fallback intrinsics from parameters
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

            # Keep compatibility with ROS tooling
            msg.distortion_model = "plumb_bob"

        msg.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0,
        ]
        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]
        msg.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        return msg

    # ------------------------------------------------------------------ #
    # Main tick
    # ------------------------------------------------------------------ #

    def _tick(self) -> None:
        if self.done:
            return

        loop = self._param_bool("loop", False)
        idx = self.i

        if idx >= len(self.rgb_paths):
            if loop:
                idx = idx % len(self.rgb_paths)
                self.i = idx
            else:
                self.get_logger().info("Finished all files (loop=False). Stopping timer.")
                self.timer.cancel()
                self.done = True
                return

        stamp = self.get_clock().now().to_msg()
        frame_color = self._param_str("frame_color", "camera_color_optical_frame")
        frame_depth = self._param_str("frame_depth", "camera_color_optical_frame")

        # ---- RGB ----
        rgb_path = self.rgb_paths[idx]
        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warning(f"Failed to read RGB: {rgb_path}")
            self.i += 1
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        rgb_msg = to_image_msg_rgb8(rgb, stamp, frame_color)

        # ---- Depth (optional) ----
        depth_msg = None
        if self._param_bool("publish_depth", True) and idx < len(self.depth_paths):
            depth_path = self.depth_paths[idx]
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            if depth_raw is None:
                if not self.logged_depth_warn:
                    self.get_logger().warning(
                        f"Failed to read depth: {depth_path} (publishing RGB only)."
                    )
                    self.logged_depth_warn = True
            else:
                if depth_raw.ndim == 3:
                    depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)

                if depth_raw.dtype != np.uint16:
                    # Fallback conversion: assume meters -> mm
                    depth_raw = (
                        (depth_raw.astype(np.float32) * 1000.0)
                        .clip(0, 65535)
                        .astype(np.uint16)
                    )

                # Resize to color if needed (aligned)
                if depth_raw.shape[:2] != (height, width):
                    depth_raw = cv2.resize(
                        depth_raw,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )

                depth_msg = to_image_msg_depth16(depth_raw, stamp, frame_depth)

        # ---- CameraInfo ----
        info_msg = self._make_camera_info(stamp, width, height)

        # ---- Publish ----
        self.pub_rgb.publish(rgb_msg)
        if depth_msg is not None:
            self.pub_depth.publish(depth_msg)
        self.pub_info.publish(info_msg)

        self.i += 1


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
