#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth mask node (ROS 2).

Subscribes (synchronized):
  - depth_topic: depth image
  - label_topic: instance label image (mono16)
  - frame_info_topic: strawberry_msgs/FrameInfo

Publishes:
  - output_topic: masked depth image
  - frame_info_out_topic: FrameInfo passthrough aligned to output stamp
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

from strawberry_msgs.msg import FrameInfo


class DepthMaskNode(Node):
    """Create a masked depth image using an instance label image."""

    def __init__(self) -> None:
        super().__init__("strawberry_depth_mask")

        # ---------------- Parameters ----------------
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("label_topic", "/seg/label_image")
        self.declare_parameter("output_topic", "/seg/depth_masked")

        # NOTE: consume passthrough from seg_ultra to avoid mismatch
        self.declare_parameter("frame_info_topic", "/seg/frame_info")
        self.declare_parameter("publish_frame_info", True)
        self.declare_parameter("frame_info_out_topic", "/seg/frame_info_depth_masked")

        self.declare_parameter("zero_background", True)
        self.declare_parameter("profile", False)

        self.declare_parameter("sync_queue_size", 200)
        self.declare_parameter("sync_slop", 0.2)

        self.declare_parameter("debug_stamps_once", False)

        depth_topic = self._param_str("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        label_topic = self._param_str("label_topic", "/seg/label_image")
        output_topic = self._param_str("output_topic", "/seg/depth_masked")

        frame_info_topic = self._param_str("frame_info_topic", "/seg/frame_info")
        self._publish_frame_info = self._param_bool("publish_frame_info", True)
        frame_info_out_topic = self._param_str("frame_info_out_topic", "/seg/frame_info_depth_masked")

        self._zero_bg = self._param_bool("zero_background", True)
        self._profile = self._param_bool("profile", False)

        self._sync_queue_size = max(1, self._param_int("sync_queue_size", 200))
        self._sync_slop = self._param_float("sync_slop", 0.2)
        if self._sync_slop <= 0.0:
            self._sync_slop = 0.05

        self._debug_stamps_once = self._param_bool("debug_stamps_once", False)
        self._debug_once = True

        self.get_logger().info(
            "DepthMaskNode starting:\n"
            f"  depth_topic          = {depth_topic}\n"
            f"  label_topic          = {label_topic}\n"
            f"  output_topic         = {output_topic}\n"
            f"  frame_info_topic     = {frame_info_topic}\n"
            f"  publish_frame_info   = {self._publish_frame_info}\n"
            f"  frame_info_out_topic = {frame_info_out_topic}\n"
            f"  zero_background      = {self._zero_bg}\n"
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

        self._pub_depth_masked = self.create_publisher(Image, output_topic, 10)
        self._pub_frame_info = (
            self.create_publisher(FrameInfo, frame_info_out_topic, 10)
            if self._publish_frame_info
            else None
        )

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

    @staticmethod
    def _copy_frame_info(src: FrameInfo, stamp) -> FrameInfo:
        out = FrameInfo()
        out.header = Header(stamp=stamp, frame_id=src.header.frame_id)
        out.frame_index = int(src.frame_index)
        out.plant_id = int(src.plant_id)
        out.view_id = int(src.view_id)
        out.rgb_path = str(src.rgb_path)
        out.depth_path = str(src.depth_path)
        return out

    # ------------------------------------------------------------------ #
    # Callback
    # ------------------------------------------------------------------ #

    def _sync_cb(self, depth_msg: Image, label_msg: Image, frame_info: FrameInfo) -> None:
        t0 = time.time()

        if self._debug_stamps_once and self._debug_once:
            ds = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec * 1e-9
            ls = label_msg.header.stamp.sec + label_msg.header.stamp.nanosec * 1e-9
            fs = frame_info.header.stamp.sec + frame_info.header.stamp.nanosec * 1e-9
            self.get_logger().info(
                "Stamp debug (once): "
                f"depth={ds:.6f} label={ls:.6f} frame_info={fs:.6f} "
                f"delta(depth-label)={(ds - ls) * 1000.0:.3f} ms "
                f"delta(depth-frame)={(ds - fs) * 1000.0:.3f} ms "
                f"| frame_index={int(frame_info.frame_index)} plant={int(frame_info.plant_id)} "
                f"view={int(frame_info.view_id)}"
            )
            self._debug_once = False

        depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        label = self._bridge.imgmsg_to_cv2(label_msg, desired_encoding="mono16")

        if depth.shape[:2] != label.shape[:2]:
            self.get_logger().warning(
                f"Shape mismatch depth={depth.shape} label={label.shape} "
                "-> check that depth and label are aligned!"
            )
            return

        mask = label > 0
        masked = depth.copy()

        if self._zero_bg:
            masked[~mask] = 0
        else:
            if np.issubdtype(masked.dtype, np.floating):
                masked[~mask] = np.nan
            else:
                masked[~mask] = 0

        out_msg = self._bridge.cv2_to_imgmsg(masked, encoding=depth_msg.encoding)
        out_msg.header = depth_msg.header
        self._pub_depth_masked.publish(out_msg)

        if self._pub_frame_info is not None:
            self._pub_frame_info.publish(self._copy_frame_info(frame_info, depth_msg.header.stamp))

        if self._profile:
            self.get_logger().info(f"Depth mask callback: {(time.time() - t0) * 1000.0:.2f} ms")


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
