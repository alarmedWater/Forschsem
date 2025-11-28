#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

import message_filters   # apt: ros-humble-message-filters, in package.xml als dep eintragen


class DepthMaskNode(Node):
    """
    Nimmt:
      - depth_topic: Tiefenbild (z.B. /camera/depth/image_raw)
      - label_topic: Instanz-Labelbild (mono16) von seg_onnx/seg_ultra

    Gibt aus:
      - /seg/depth_masked: Tiefenbild, wo nur Pixel mit label>0 erhalten bleiben
    """

    def __init__(self):
        super().__init__("strawberry_depth_mask")

        # --- Parameter ---
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("label_topic", "/seg/label_image")
        self.declare_parameter("output_topic", "/seg/depth_masked")
        self.declare_parameter("zero_background", True)  # sonst NaN
        self.declare_parameter("profile", False)

        depth_topic  = self.get_parameter("depth_topic").value
        label_topic  = self.get_parameter("label_topic").value
        output_topic = self.get_parameter("output_topic").value
        self._zero_bg = bool(self.get_parameter("zero_background").value)
        self._profile = bool(self.get_parameter("profile").value)

        self.get_logger().info(
            f"DepthMaskNode startet:\n"
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

        # --- Synchronisierte Subscriber ---
        self.sub_depth = message_filters.Subscriber(self, Image, depth_topic, qos_profile=qos)
        self.sub_label = message_filters.Subscriber(self, Image, label_topic, qos_profile=qos)

        # ApproximateTimeSynchronizer erlaubt etwas Zeitversatz
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_depth, self.sub_label],
            queue_size=10,
            slop=0.05,   # 50ms Toleranz
        )
        self.ts.registerCallback(self.sync_cb)

        self.pub_depth_masked = self.create_publisher(Image, output_topic, 10)

        self._debug_once = True

    # ----------------- Callback -----------------
    def sync_cb(self, depth_msg: Image, label_msg: Image):
        import time
        t0 = time.time()

        # Depth so lassen wie es ist (16UC1 oder 32FC1)
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        # Labels als uint16
        label = self.bridge.imgmsg_to_cv2(label_msg, desired_encoding="mono16")

        if depth.shape[:2] != label.shape[:2]:
            self.get_logger().warn(
                f"Shape mismatch depth={depth.shape} label={label.shape} – "
                f"prüfe, ob depth & color registriert/aligned sind!"
            )
            return

        # Maske: alle Instanzen > 0
        mask = label > 0  # bool
        if self._debug_once:
            unique_ids = np.unique(label)
            self.get_logger().info(f"Erste Maske: unique instance IDs = {unique_ids[:10]}...")
            self._debug_once = False

        # Tiefenbild maskieren
        masked = depth.copy()

        if self._zero_bg:
            # Hintergrund auf 0
            masked[~mask] = 0
        else:
            # Hintergrund auf NaN (nur sinnvoll bei float32-depth)
            if np.issubdtype(masked.dtype, np.floating):
                masked[~mask] = np.nan
            else:
                masked[~mask] = 0

        # --- Publish ---
        out_msg = self.bridge.cv2_to_imgmsg(masked, encoding=depth_msg.encoding)
        out_msg.header = Header(
            stamp=depth_msg.header.stamp,
            frame_id=depth_msg.header.frame_id,
        )
        self.pub_depth_masked.publish(out_msg)

        if self._profile:
            dt = (time.time() - t0) * 1000.0
            self.get_logger().info(f"Depth mask callback: {dt:.2f} ms")

def main():
    rclpy.init()
    node = DepthMaskNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
