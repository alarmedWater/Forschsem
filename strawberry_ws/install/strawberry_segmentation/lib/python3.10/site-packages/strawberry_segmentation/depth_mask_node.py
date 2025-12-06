#!/usr/bin/env python3
# strawberry_segmentation/depth_mask_node.py

from __future__ import annotations

import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image


class DepthMaskNode(Node):
    """
    Combine depth and instance label image to create a masked depth image.

    Subscribes:
      - depth_topic: depth image (e.g. /camera/aligned_depth_to_color/image_raw)
      - label_topic: instance label image (mono16, e.g. /seg/label_image)

    Publishes:
      - output_topic: masked depth image where label > 0 is kept
        (e.g. /seg/depth_masked)
    """

    def __init__(self) -> None:
        super().__init__("strawberry_depth_mask")

        # Parameters
        self.declare_parameter(
            "depth_topic", "/camera/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter("label_topic", "/seg/label_image")
        self.declare_parameter("output_topic", "/seg/depth_masked")
        self.declare_parameter(
            "zero_background", True
        )  # False -> set background to NaN if float
        self.declare_parameter("profile", False)

        depth_topic = self.get_parameter("depth_topic").value
        label_topic = self.get_parameter("label_topic").value
        output_topic = self.get_parameter("output_topic").value
        self._zero_bg = bool(self.get_parameter("zero_background").value)
        self._profile = bool(self.get_parameter("profile").value)

        self.get_logger().info(
            "DepthMaskNode starting:\n"
            f"  depth_topic  = {depth_topic}\n"
            f"  label_topic  = {label_topic}\n"
            f"  output_topic = {output_topic}"
        )

        self.bridge = CvBridge()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Synchronized subscribers
        self.sub_depth = Subscriber(self, Image, depth_topic, qos_profile=qos)
        self.sub_label = Subscriber(self, Image, label_topic, qos_profile=qos)

        self.ts = ApproximateTimeSynchronizer(
            [self.sub_depth, self.sub_label],
            queue_size=10,
            slop=0.05,  # 50 ms tolerance
        )
        self.ts.registerCallback(self.sync_cb)

        self.pub_depth_masked = self.create_publisher(Image, output_topic, 10)

        self._debug_once = True

    # ----------------- Callback -----------------
    def sync_cb(self, depth_msg: Image, label_msg: Image) -> None:
        t0 = time.time()

        depth = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding="passthrough"
        )
        label = self.bridge.imgmsg_to_cv2(label_msg, desired_encoding="mono16")

        if depth.shape[:2] != label.shape[:2]:
            self.get_logger().warn(
                f"Shape mismatch depth={depth.shape} label={label.shape} – "
                "check that depth and color are aligned!"
            )
            return

        # Mask: all instances > 0
        mask = label > 0
        if self._debug_once:
            unique_ids = np.unique(label)
            self.get_logger().info(
                f"First mask: unique instance IDs = {unique_ids[:10]}..."
            )
            self._debug_once = False

        masked = depth.copy()

        if self._zero_bg:
            # Background to 0
            masked[~mask] = 0
        else:
            # Background to NaN (only useful for float depth)
            if np.issubdtype(masked.dtype, np.floating):
                masked[~mask] = np.nan
            else:
                masked[~mask] = 0

        out_msg = self.bridge.cv2_to_imgmsg(
            masked, encoding=depth_msg.encoding
        )
        out_msg.header = depth_msg.header
        self.pub_depth_masked.publish(out_msg)

        if self._profile:
            dt = (time.time() - t0) * 1000.0
            self.get_logger().info(f"Depth mask callback: {dt:.2f} ms")


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
