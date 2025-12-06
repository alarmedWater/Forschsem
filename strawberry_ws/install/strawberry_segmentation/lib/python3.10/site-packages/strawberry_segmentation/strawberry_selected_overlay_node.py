#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

import message_filters


class StrawberrySelectedOverlayNode(Node):
    """
    Visualisiert die aktuell ausgewählte Erdbeer-Instanz im RGB-Bild.

    Subscribes:
      - image_topic: RGB-Bild (z.B. /camera/color/image_raw, rgb8)
      - label_topic: Instanz-Labelbild (mono16, z.B. /seg/label_image)

    Parameter:
      - selected_instance_id (int): ID der Erdbeer-Instanz (wie im label_image)
      - output_topic (str): Topic für das Overlay (default: /seg/selected_overlay)

    Ausgabe:
      - /seg/selected_overlay: RGB-Bild, in dem der Hintergrund abgedunkelt
        und die gewählte Erdbeere hervorgehoben ist (+ Bounding Box).
    """

    def __init__(self):
        super().__init__("strawberry_selected_overlay")

        # --- Parameter ---
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("label_topic", "/seg/label_image")
        self.declare_parameter("output_topic", "/seg/selected_overlay")

        self.declare_parameter("selected_instance_id", 1)
        self.declare_parameter("min_pixels", 50)     # minimale Pixelanzahl, damit es sich lohnt
        self.declare_parameter("darken_factor", 0.3) # wie stark Hintergrund abgedunkelt wird (0..1)

        image_topic = self.get_parameter("image_topic").value
        label_topic = self.get_parameter("label_topic").value
        self._output_topic = self.get_parameter("output_topic").value
        self._selected_instance_id = int(self.get_parameter("selected_instance_id").value)
        self._min_pixels = int(self.get_parameter("min_pixels").value)
        self._darken_factor = float(self.get_parameter("darken_factor").value)

        self.get_logger().info(
            f"StrawberrySelectedOverlayNode startet:\n"
            f"  image_topic      = {image_topic}\n"
            f"  label_topic      = {label_topic}\n"
            f"  output_topic     = {self._output_topic}\n"
            f"  selected_inst_id = {self._selected_instance_id}"
        )

        self.bridge = CvBridge()

        # QoS (analog zu deinen anderen Nodes)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Synchronisierte Subscriber
        self.sub_img = message_filters.Subscriber(self, Image, image_topic, qos_profile=qos)
        self.sub_label = message_filters.Subscriber(self, Image, label_topic, qos_profile=qos)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_img, self.sub_label],
            queue_size=10,
            slop=0.05,
        )
        self.ts.registerCallback(self.sync_cb)

        # Publisher
        self.pub_overlay = self.create_publisher(Image, self._output_topic, 10)

    # ----------------- Callback -----------------
    def sync_cb(self, img_msg: Image, label_msg: Image):
        # aktuellen selected_instance_id-Parameter holen (kann zur Laufzeit geändert werden)
        self._selected_instance_id = int(self.get_parameter("selected_instance_id").value)

        # RGB-Bild holen
        img_rgb = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        # Labelbild (mono16)
        label = self.bridge.imgmsg_to_cv2(label_msg, desired_encoding="mono16")

        if img_rgb.shape[:2] != label.shape[:2]:
            self.get_logger().warn(
                f"Shape mismatch image={img_rgb.shape} label={label.shape} – "
                f"prüfe, ob seg_ultra & Kamera aligned sind!"
            )
            return

        # Maske der ausgewählten Instanz
        inst_id = self._selected_instance_id
        mask = (label == inst_id)

        n_pix = int(mask.sum())
        if n_pix < self._min_pixels:
            # Falls gerade keine oder zu kleine Erdbeere dieser ID sichtbar ist, kannst du
            # hier entweder: nichts publishen oder einfach das Originalbild durchreichen.
            # Wir nehmen hier das Originalbild:
            out_msg = self.bridge.cv2_to_imgmsg(img_rgb, encoding="rgb8")
            out_msg.header = Header(
                stamp=img_msg.header.stamp,
                frame_id=img_msg.header.frame_id,
            )
            self.pub_overlay.publish(out_msg)
            return

        # Hintergrund abdunkeln
        overlay = (img_rgb.astype(np.float32) * self._darken_factor).astype(np.uint8)
        # Erdbeeren-Pixel wieder mit Original-Helligkeit einsetzen
        overlay[mask] = img_rgb[mask]

        # Bounding Box um die Erdbeere zeichnen
        ys, xs = np.where(mask)
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())

        # cv2 arbeitet typischerweise in BGR, aber für Rechteckfarbe ist das egal (wir bleiben bei RGB)
        cv2.rectangle(
            overlay,
            (x_min, y_min),
            (x_max, y_max),
            color=(255, 0, 0),  # Rot in RGB
            thickness=2,
        )

        # Optional: Instanz-ID als Text
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

        # Zurück nach ROS
        out_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="rgb8")
        out_msg.header = Header(
            stamp=img_msg.header.stamp,
            frame_id=img_msg.header.frame_id,
        )
        self.pub_overlay.publish(out_msg)


def main():
    rclpy.init()
    node = StrawberrySelectedOverlayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
