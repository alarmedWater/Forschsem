#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth mask node (ROS 2).

Combines a depth image and an instance label image into a masked depth image.

Subscribes:
  - depth_topic: depth image (e.g. /camera/aligned_depth_to_color/image_raw)
  - label_topic: instance label image (mono16, e.g. /seg/label_image)

Publishes:
  - output_topic: masked depth image where label > 0 is kept (e.g. /seg/depth_masked)

Notes:
  - Uses message_filters ApproximateTimeSynchronizer.
  - If your segmentation is slower / jittery, increase sync_queue_size and/or sync_slop.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image


class DepthMaskNode(Node):
    """Create a masked depth image using an instance label image."""

    def __init__(self) -> None:
        super().__init__("strawberry_depth_mask")

        # ---------------- Parameters ----------------
        self.declare_parameter(
            "depth_topic", "/camera/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter("label_topic", "/seg/label_image")
        self.declare_parameter("output_topic", "/seg/depth_masked")

        # If True -> background set to 0.
        # If False -> background set to NaN if float, else 0.
        self.declare_parameter("zero_background", True)

        self.declare_parameter("profile", False)

        # Sync tuning (IMPORTANT when segmentation is slow / jittery)
        self.declare_parameter("sync_queue_size", 200)
        self.declare_parameter("sync_slop", 0.2)

        # Optional: print stamp mismatch once for debugging
        self.declare_parameter("debug_stamps_once", False)

        depth_topic = self._param_str(
            "depth_topic", "/camera/aligned_depth_to_color/image_raw"
        )
        label_topic = self._param_str("label_topic", "/seg/label_image")
        output_topic = self._param_str("output_topic", "/seg/depth_masked")

        self._zero_bg = self._param_bool("zero_background", True)
        self._profile = self._param_bool("profile", False)

        self._sync_queue_size = self._param_int("sync_queue_size", 200)
        self._sync_slop = self._param_float("sync_slop", 0.2)

        self._debug_stamps_once = self._param_bool("debug_stamps_once", False)
        self._debug_once = True

        if self._sync_queue_size < 1:
            self._sync_queue_size = 1
        if self._sync_slop <= 0.0:
            self._sync_slop = 0.05

        self.get_logger().info(
            "DepthMaskNode starting:\n"
            f"  depth_topic        = {depth_topic}\n"
            f"  label_topic        = {label_topic}\n"
            f"  output_topic       = {output_topic}\n"
            f"  zero_background    = {self._zero_bg}\n"
            f"  sync_queue_size    = {self._sync_queue_size}\n"
            f"  sync_slop          = {self._sync_slop}\n"
            f"  debug_stamps_once  = {self._debug_stamps_once}\n"
            f"  profile            = {self._profile}"
        )

        self.bridge = CvBridge()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ---------------- Subscribers ----------------
        self.sub_depth = Subscriber(self, Image, depth_topic, qos_profile=qos)
        self.sub_label = Subscriber(self, Image, label_topic, qos_profile=qos)

        # ---------------- Synchronizer ----------------
        self.ts = ApproximateTimeSynchronizer(
            [self.sub_depth, self.sub_label],
            queue_size=self._sync_queue_size,
            slop=self._sync_slop,
        )
        self.ts.registerCallback(self.sync_cb)

        # ---------------- Publisher ----------------
        self.pub_depth_masked = self.create_publisher(Image, output_topic, 10)

    # ------------------------------------------------------------------ #
    # Param helpers (Pylance-friendly)
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
    # Callback
    # ------------------------------------------------------------------ #

    def sync_cb(self, depth_msg: Image, label_msg: Image) -> None:
        t0 = time.time()

        if self._debug_stamps_once and self._debug_once:
            ds = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec * 1e-9
            ls = label_msg.header.stamp.sec + label_msg.header.stamp.nanosec * 1e-9
            self.get_logger().info(
                "Stamp debug (once): depth=%.6f label=%.6f delta=%.3f ms",
                ds,
                ls,
                (ds - ls) * 1000.0,
            )
            self._debug_once = False

        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        label = self.bridge.imgmsg_to_cv2(label_msg, desired_encoding="mono16")

        if depth.shape[:2] != label.shape[:2]:
            self.get_logger().warning(
                f"Shape mismatch depth={depth.shape} label={label.shape} "
                "-> check that depth and label are aligned!"
            )
            return

        # Mask: all instances > 0
        mask = label > 0

        masked = depth.copy()
        if self._zero_bg:
            masked[~mask] = 0
        else:
            if np.issubdtype(masked.dtype, np.floating):
                masked[~mask] = np.nan
            else:
                masked[~mask] = 0

        out_msg = self.bridge.cv2_to_imgmsg(masked, encoding=depth_msg.encoding)
        out_msg.header = depth_msg.header
        self.pub_depth_masked.publish(out_msg)

        if self._profile:
            dt_ms = (time.time() - t0) * 1000.0
            self.get_logger().info(f"Depth mask callback: {dt_ms:.2f} ms")


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
