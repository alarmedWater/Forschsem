#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Depth masking node (ROS 2).

Applies instance label masks to the aligned depth image and optionally gates depth
to a known working range (e.g., strawberries are within 0.05..0.60 m).

Inputs (synchronized):
  - depth_topic: depth image (typically 16UC1) aligned to color
  - label_topic: instance label image (mono16), 0=background, >0 instance ids
  - frame_info_topic: strawberry_msgs/FrameInfo aligned to label stamp

Outputs:
  - output_topic: masked depth (same type as input; outside -> 0)
  - frame_info_out_topic: FrameInfo passthrough aligned to output stamp

Key idea:
  Run YOLO on RGB first, then apply depth constraints AFTERWARDS to avoid
  distribution shift.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header

# If Pylance complains but runtime is fine, you can keep this ignore:
from strawberry_msgs.msg import FrameInfo  # type: ignore


class DepthMaskNode(Node):
    """Mask aligned depth with instance labels and optional depth range filtering."""

    def __init__(self) -> None:
        super().__init__("strawberry_depth_mask")

        # ---------------- Parameters ----------------
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("label_topic", "/seg/label_image")
        self.declare_parameter("output_topic", "/seg/depth_masked")

        self.declare_parameter("frame_info_topic", "/seg/frame_info")
        self.declare_parameter("publish_frame_info", True)
        self.declare_parameter("frame_info_out_topic", "/seg/frame_info_depth_masked")

        self.declare_parameter("zero_background", True)

        # Sync tuning
        self.declare_parameter("sync_queue_size", 200)
        self.declare_parameter("sync_slop", 0.2)

        # Profiling / debug
        self.declare_parameter("profile", False)
        self.declare_parameter("debug_stamps_once", False)

        # -------- Depth range gating --------
        self.declare_parameter("range_filter_enable", True)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 0.60)

        # Depth unit handling
        self.declare_parameter("depth_unit", "mm")  # "mm" or "realsense_units"
        self.declare_parameter("depth_scale_m_per_unit", 9.999999747378752e-05)
        self.declare_parameter("treat_65535_as_invalid", True)

        # ---------------- Read parameters ----------------
        depth_topic = self._param_str("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        label_topic = self._param_str("label_topic", "/seg/label_image")
        output_topic = self._param_str("output_topic", "/seg/depth_masked")

        frame_info_topic = self._param_str("frame_info_topic", "/seg/frame_info")
        self._publish_frame_info = self._param_bool("publish_frame_info", True)
        self._frame_info_out_topic = self._param_str(
            "frame_info_out_topic", "/seg/frame_info_depth_masked"
        )

        self._zero_background = self._param_bool("zero_background", True)

        self._sync_queue_size = max(1, self._param_int("sync_queue_size", 200))
        self._sync_slop = float(self._param_float("sync_slop", 0.2))
        if self._sync_slop <= 0.0:
            self._sync_slop = 0.05

        self._profile = self._param_bool("profile", False)
        self._debug_stamps_once = self._param_bool("debug_stamps_once", False)
        self._did_debug_stamps = False

        self._range_filter_enable = self._param_bool("range_filter_enable", True)
        self._min_depth_m = float(self._param_float("min_depth_m", 0.05))
        self._max_depth_m = float(self._param_float("max_depth_m", 0.60))
        if self._max_depth_m <= 0.0:
            self._max_depth_m = 0.60
        if self._min_depth_m < 0.0:
            self._min_depth_m = 0.0
        if self._min_depth_m > self._max_depth_m:
            self._min_depth_m, self._max_depth_m = self._max_depth_m, self._min_depth_m

        self._depth_unit = self._param_str("depth_unit", "mm").strip().lower()
        self._depth_scale = float(
            self._param_float("depth_scale_m_per_unit", 9.999999747378752e-05)
        )
        self._treat_65535_as_invalid = self._param_bool("treat_65535_as_invalid", True)

        self.get_logger().info(
            "DepthMaskNode starting:\n"
            f"  depth_topic          = {depth_topic}\n"
            f"  label_topic          = {label_topic}\n"
            f"  output_topic         = {output_topic}\n"
            f"  frame_info_topic     = {frame_info_topic}\n"
            f"  publish_frame_info   = {self._publish_frame_info}\n"
            f"  frame_info_out_topic = {self._frame_info_out_topic}\n"
            f"  zero_background      = {self._zero_background}\n"
            f"  range_filter_enable  = {self._range_filter_enable}\n"
            f"  min_depth_m          = {self._min_depth_m:.3f}\n"
            f"  max_depth_m          = {self._max_depth_m:.3f}\n"
            f"  depth_unit           = {self._depth_unit}\n"
            f"  depth_scale          = {self._depth_scale:.3e} m/unit\n"
            f"  sync_queue_size      = {self._sync_queue_size}\n"
            f"  sync_slop            = {self._sync_slop}\n"
            f"  debug_stamps_once    = {self._debug_stamps_once}\n"
            f"  profile              = {self._profile}"
        )

        self._bridge = CvBridge()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ---------------- Subscribers (synced) ----------------
        self._sub_depth = message_filters.Subscriber(self, Image, depth_topic, qos_profile=qos)
        self._sub_label = message_filters.Subscriber(self, Image, label_topic, qos_profile=qos)
        self._sub_frame_info = message_filters.Subscriber(
            self, FrameInfo, frame_info_topic, qos_profile=qos
        )

        self._ts = message_filters.ApproximateTimeSynchronizer(
            [self._sub_depth, self._sub_label, self._sub_frame_info],
            queue_size=self._sync_queue_size,
            slop=self._sync_slop,
        )
        self._ts.registerCallback(self._sync_cb)

        # ---------------- Publishers ----------------
        self._pub_depth = self.create_publisher(Image, output_topic, 10)
        self._pub_frame_info = (
            self.create_publisher(FrameInfo, self._frame_info_out_topic, 10)
            if self._publish_frame_info
            else None
        )

        self._warned_depth_dtype = False
        self._warned_bad_depth_unit = False

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
    # Depth range mask (fast, in raw units)
    # ------------------------------------------------------------------ #

    def _compute_range_mask_u16(self, depth_u16: np.ndarray) -> np.ndarray:
        """Return boolean mask for 'in valid range' on uint16 depth."""
        valid = depth_u16 != 0
        if self._treat_65535_as_invalid:
            valid &= depth_u16 != np.uint16(65535)

        if not self._range_filter_enable:
            return valid

        min_m = float(self._min_depth_m)
        max_m = float(self._max_depth_m)

        if self._depth_unit == "mm":
            lo = int(round(min_m * 1000.0))
            hi = int(round(max_m * 1000.0))
        elif self._depth_unit == "realsense_units":
            scale = float(self._depth_scale) if self._depth_scale > 0.0 else 1e-4
            lo = int(round(min_m / scale))
            hi = int(round(max_m / scale))
        else:
            if not self._warned_bad_depth_unit:
                self.get_logger().warning(
                    f"Unknown depth_unit='{self._depth_unit}'. Using 'mm' thresholds."
                )
                self._warned_bad_depth_unit = True
            lo = int(round(min_m * 1000.0))
            hi = int(round(max_m * 1000.0))

        lo = max(0, min(lo, 65535))
        hi = max(0, min(hi, 65535))
        if lo > hi:
            lo, hi = hi, lo

        return valid & (depth_u16 >= np.uint16(lo)) & (depth_u16 <= np.uint16(hi))

    @staticmethod
    def _encoding_for_depth(arr: np.ndarray) -> str:
        if arr.dtype == np.uint16:
            return "16UC1"
        if arr.dtype == np.float32:
            return "32FC1"
        return "passthrough"

    # ------------------------------------------------------------------ #
    # Callback
    # ------------------------------------------------------------------ #

    def _sync_cb(self, depth_msg: Image, label_msg: Image, fi_msg: FrameInfo) -> None:
        t0 = time.time()

        depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        label = self._bridge.imgmsg_to_cv2(label_msg, desired_encoding="mono16")

        # Guard (rare but prevents hard crashes)
        if depth is None or label is None:
            self.get_logger().warning("cv_bridge returned None for depth or label.")
            return

        if depth.shape[:2] != label.shape[:2]:
            self.get_logger().warning(
                f"Shape mismatch depth={depth.shape} label={label.shape} -> check alignment!"
            )
            return

        # FIX (important): rclpy logger expects a single string message
        if self._debug_stamps_once and not self._did_debug_stamps:
            self._did_debug_stamps = True
            self.get_logger().info(
                "Stamps (sec.nanosec): "
                f"depth={depth_msg.header.stamp.sec}.{depth_msg.header.stamp.nanosec:09d} "
                f"label={label_msg.header.stamp.sec}.{label_msg.header.stamp.nanosec:09d} "
                f"frame_info={fi_msg.header.stamp.sec}.{fi_msg.header.stamp.nanosec:09d}"
            )

        # We primarily support uint16 depth (RealSense / PNG dumps)
        if depth.dtype != np.uint16:
            if not self._warned_depth_dtype:
                self._warned_depth_dtype = True
                self.get_logger().warning(
                    f"Depth dtype is {depth.dtype}, expected uint16. "
                    "Proceeding, but range gating may be disabled."
                )

        depth_out = depth.copy()

        # Valid depth mask (range-gated if enabled)
        if depth_out.dtype == np.uint16:
            range_ok = self._compute_range_mask_u16(depth_out)
        else:
            # Fallback for float depths: compute in meters (assumes meters already)
            range_ok = np.isfinite(depth_out) & (depth_out > 0.0)
            if self._range_filter_enable:
                range_ok &= (depth_out >= self._min_depth_m) & (depth_out <= self._max_depth_m)

        if self._zero_background:
            keep = (label > 0) & range_ok
            depth_out[~keep] = 0
        else:
            # Only apply range gating (keep labels untouched)
            depth_out[~range_ok] = 0

        out_msg = self._bridge.cv2_to_imgmsg(
            depth_out, encoding=self._encoding_for_depth(depth_out)
        )
        out_msg.header = depth_msg.header
        self._pub_depth.publish(out_msg)

        if self._pub_frame_info is not None:
            out_fi = FrameInfo()
            out_fi.header = Header(
                stamp=depth_msg.header.stamp, frame_id=fi_msg.header.frame_id
            )
            out_fi.frame_index = int(fi_msg.frame_index)
            out_fi.plant_id = int(fi_msg.plant_id)
            out_fi.view_id = int(fi_msg.view_id)
            out_fi.rgb_path = str(fi_msg.rgb_path)
            out_fi.depth_path = str(fi_msg.depth_path)
            self._pub_frame_info.publish(out_fi)

        if self._profile:
            dt_ms = (time.time() - t0) * 1000.0
            self.get_logger().info(f"DepthMask callback: {dt_ms:.2f} ms")


def main() -> None:
    rclpy.init()
    node = DepthMaskNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
