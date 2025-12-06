#!/usr/bin/env python3
# strawberry_segmentation/strawberry_selected_overlay_node.py

from __future__ import annotations

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image


class StrawberrySelectedOverlayNode(Node):
    """
    Visualize the currently selected strawberry instance on the RGB image.

    Subscribes:
      - image_topic: RGB image (e.g. /camera/color/image_raw, rgb8)
      - label_topic: instance label image (mono16, e.g. /seg/label_image)

    Parameters:
      - selected_instance_id (int): instance ID in the label image
      - min_pixels (int): minimum number of pixels required to consider
                          the instance valid
      - darken_factor (float): background darkening factor in [0, 1]
      - output_topic (str): overlay topic (default: /seg/selected_overlay)

    Publishes:
      - output_topic (default: /seg/selected_overlay):
        RGB image where the background is darkened and the selected
        strawberry is highlighted with original brightness and a bounding box.
    """

    def __init__(self) -> None:
        super().__init__("strawberry_selected_overlay")

        # Parameters
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("label_topic", "/seg/label_image")
        self.declare_parameter("output_topic", "/seg/selected_overlay")

        self.declare_parameter("selected_instance_id", 1)
        self.declare_parameter("min_pixels", 50)
        self.declare_parameter("darken_factor", 0.3)

        image_topic = self.get_parameter("image_topic").value
        label_topic = self.get_parameter("label_topic").value
        self._output_topic = self.get_parameter("output_topic").value
        self._selected_instance_id = int(
            self.get_parameter("selected_instance_id").value
        )
        self._min_pixels = int(self.get_parameter("min_pixels").value)
        self._darken_factor = float(
            self.get_parameter("darken_factor").value
        )

        self.get_logger().info(
            "StrawberrySelectedOverlayNode starting:\n"
            f"  image_topic      = {image_topic}\n"
            f"  label_topic      = {label_topic}\n"
            f"  output_topic     = {self._output_topic}\n"
            f"  selected_inst_id = {self._selected_instance_id}"
        )

        self.bridge = CvBridge()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Synchronized subscribers for image + labels
        self.sub_img = Subscriber(
            self, Image, image_topic, qos_profile=qos
        )
        self.sub_label = Subscriber(
            self, Image, label_topic, qos_profile=qos
        )

        self.ts = ApproximateTimeSynchronizer(
            [self.sub_img, self.sub_label],
            queue_size=10,
            slop=0.05,
        )
        self.ts.registerCallback(self.sync_cb)

        # Publisher for overlay image
        self.pub_overlay = self.create_publisher(
            Image, self._output_topic, 10
        )

    # ----------------- Callback -----------------
    def sync_cb(self, img_msg: Image, label_msg: Image) -> None:
        # Read current selected_instance_id at runtime (ros2 param set ...)
        self._selected_instance_id = int(
            self.get_parameter("selected_instance_id").value
        )

        # RGB image
        img_rgb = self.bridge.imgmsg_to_cv2(
            img_msg, desired_encoding="rgb8"
        )
        # Label image (mono16)
        label = self.bridge.imgmsg_to_cv2(
            label_msg, desired_encoding="mono16"
        )

        if img_rgb.shape[:2] != label.shape[:2]:
            self.get_logger().warn(
                f"Shape mismatch image={img_rgb.shape} label={label.shape} – "
                "check that seg_ultra and camera are aligned!"
            )
            return

        inst_id = self._selected_instance_id
        mask = label == inst_id
        n_pix = int(mask.sum())

        if n_pix < self._min_pixels:
            # If this instance is not present or too small, just pass through
            out_msg = self.bridge.cv2_to_imgmsg(img_rgb, encoding="rgb8")
            out_msg.header = img_msg.header
            self.pub_overlay.publish(out_msg)
            return

        # Darken background
        overlay = (img_rgb.astype(np.float32) * self._darken_factor).astype(
            np.uint8
        )
        # Restore original brightness for selected strawberry pixels
        overlay[mask] = img_rgb[mask]

        # Bounding box around the selected instance
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

        # Optional: draw instance ID as text
        text = f"id={inst_id}"
        cv2.putText(
            overlay,
            text,
            (x_min, max(y_min - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

        out_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="rgb8")
        out_msg.header = img_msg.header
        self.pub_overlay.publish(out_msg)


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
