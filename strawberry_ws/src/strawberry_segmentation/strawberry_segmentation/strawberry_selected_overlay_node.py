#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strawberry selected overlay node (ROS 2).

Visualizes the currently selected strawberry instance on the RGB image.

Subscribes (synchronized):
  - image_topic:      RGB image (rgb8), e.g. /camera/color/image_raw
  - label_topic:      instance label image (mono16), e.g. /seg/label_image
  - frame_info_topic: FrameInfo, e.g. /seg/frame_info (recommended) or /camera/frame_info

Publishes:
  - output_topic:     overlay RGB image (rgb8), e.g. /seg/selected_overlay
  - frame_info_out_topic (optional): FrameInfo passthrough aligned to output stamp

Parameters:
  - selected_instance_id (int): instance ID in the label image
  - min_pixels (int): minimum number of pixels required to consider valid
  - darken_factor (float): background darkening factor in [0, 1]
  - draw_bbox (bool): draw bounding box and text
  - sync_queue_size (int)
  - sync_slop (float)
"""

from __future__ import annotations

from typing import Any

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from strawberry_msgs.msg import FrameInfo


class StrawberrySelectedOverlayNode(Node):
    """Highlight the selected instance in the RGB stream."""

    def __init__(self) -> None:
        super().__init__("strawberry_selected_overlay")

        # ---------------- Parameters ----------------
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("label_topic", "/seg/label_image")
        self.declare_parameter("output_topic", "/seg/selected_overlay")

        # IMPORTANT: prefer /seg/frame_info (from seg_ultra) so it is aligned to labels
        self.declare_parameter("frame_info_topic", "/seg/frame_info")
        self.declare_parameter("publish_frame_info", True)
        self.declare_parameter("frame_info_out_topic", "/seg/frame_info_selected_overlay")

        self.declare_parameter("selected_instance_id", 1)
        self.declare_parameter("min_pixels", 50)
        self.declare_parameter("darken_factor", 0.3)
        self.declare_parameter("draw_bbox", True)

        self.declare_parameter("sync_queue_size", 50)
        self.declare_parameter("sync_slop", 0.1)

        image_topic = self._param_str("image_topic", "/camera/color/image_raw")
        label_topic = self._param_str("label_topic", "/seg/label_image")
        output_topic = self._param_str("output_topic", "/seg/selected_overlay")

        frame_info_topic = self._param_str("frame_info_topic", "/seg/frame_info")
        self._publish_frame_info = self._param_bool("publish_frame_info", True)
        frame_info_out_topic = self._param_str(
            "frame_info_out_topic", "/seg/frame_info_selected_overlay"
        )

        self._min_pixels = max(0, self._param_int("min_pixels", 50))
        self._darken_factor = float(self._param_float("darken_factor", 0.3))
        self._draw_bbox = self._param_bool("draw_bbox", True)

        self._sync_queue_size = max(1, self._param_int("sync_queue_size", 50))
        self._sync_slop = self._param_float("sync_slop", 0.1)
        if self._sync_slop <= 0.0:
            self._sync_slop = 0.05

        self.get_logger().info(
            "StrawberrySelectedOverlayNode starting:\n"
            f"  image_topic          = {image_topic}\n"
            f"  label_topic          = {label_topic}\n"
            f"  frame_info_topic     = {frame_info_topic}\n"
            f"  output_topic         = {output_topic}\n"
            f"  publish_frame_info   = {self._publish_frame_info}\n"
            f"  frame_info_out_topic = {frame_info_out_topic}\n"
            f"  min_pixels           = {self._min_pixels}\n"
            f"  darken_factor        = {self._darken_factor:.3f}\n"
            f"  draw_bbox            = {self._draw_bbox}\n"
            f"  sync_queue_size      = {self._sync_queue_size}\n"
            f"  sync_slop            = {self._sync_slop:.3f}"
        )

        self._bridge = CvBridge()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ---------------- Subscribers (synced) ----------------
        self._sub_img = message_filters.Subscriber(self, Image, image_topic, qos_profile=qos)
        self._sub_label = message_filters.Subscriber(self, Image, label_topic, qos_profile=qos)
        self._sub_frame_info = message_filters.Subscriber(
            self, FrameInfo, frame_info_topic, qos_profile=qos
        )

        self._ts = message_filters.ApproximateTimeSynchronizer(
            [self._sub_img, self._sub_label, self._sub_frame_info],
            queue_size=self._sync_queue_size,
            slop=self._sync_slop,
        )
        self._ts.registerCallback(self._sync_cb)

        # ---------------- Publishers ----------------
        self._pub_overlay = self.create_publisher(Image, output_topic, 10)
        self._pub_frame_info = (
            self.create_publisher(FrameInfo, frame_info_out_topic, 10)
            if self._publish_frame_info
            else None
        )

    # ------------------------------------------------------------------ #
    # Param helpers                                                      #
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
    # Callback                                                           #
    # ------------------------------------------------------------------ #

    def _sync_cb(self, img_msg: Image, label_msg: Image, frame_info: FrameInfo) -> None:
        selected_id = self._param_int("selected_instance_id", 1)

        img_rgb = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        label = self._bridge.imgmsg_to_cv2(label_msg, desired_encoding="mono16")

        if img_rgb.shape[:2] != label.shape[:2]:
            self.get_logger().warning(
                f"Shape mismatch image={img_rgb.shape} label={label.shape} "
                "-> check alignment!"
            )
            return

        mask = label == int(selected_id)
        n_pix = int(mask.sum())

        if n_pix < self._min_pixels:
            overlay = img_rgb
        else:
            df = float(np.clip(self._darken_factor, 0.0, 1.0))
            overlay = (img_rgb.astype(np.float32) * df).astype(np.uint8)
            overlay[mask] = img_rgb[mask]

            if self._draw_bbox:
                ys, xs = np.where(mask)
                y_min, y_max = int(ys.min()), int(ys.max())
                x_min, x_max = int(xs.min()), int(xs.max())

                cv2.rectangle(
                    overlay,
                    (x_min, y_min),
                    (x_max, y_max),
                    color=(255, 0, 0),  # red in RGB
                    thickness=2,
                )
                cv2.putText(
                    overlay,
                    f"id={selected_id}",
                    (x_min, max(y_min - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        out_msg = self._bridge.cv2_to_imgmsg(overlay, encoding="rgb8")
        out_msg.header = img_msg.header
        self._pub_overlay.publish(out_msg)

        if self._pub_frame_info is not None:
            out_fi = FrameInfo()
            out_fi.header = Header(stamp=img_msg.header.stamp, frame_id=frame_info.header.frame_id)
            out_fi.frame_index = int(frame_info.frame_index)
            out_fi.plant_id = int(frame_info.plant_id)
            out_fi.view_id = int(frame_info.view_id)
            out_fi.rgb_path = str(frame_info.rgb_path)
            out_fi.depth_path = str(frame_info.depth_path)
            self._pub_frame_info.publish(out_fi)


def main() -> None:
    rclpy.init()
    node = StrawberrySelectedOverlayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
