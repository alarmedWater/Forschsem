#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publiziert RGB-Bilder aus einem Verzeichnis als /camera/color/image_raw (rgb8)
und ein passendes /camera/color/camera_info. Optional loop.
"""

import os, glob, time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

class CameraFolder(Node):
    def __init__(self):
        super().__init__('camera_folder')

        # Parameter
        self.declare_parameter('image_dir', '')                 # Pfad zu Testbildern (required)
        self.declare_parameter('fps', 5.0)                      # Abspielrate
        self.declare_parameter('frame_color', 'camera_color_optical_frame')
        self.declare_parameter('publish_info', True)            # CameraInfo publizieren
        self.declare_parameter('loop', True)                    # nach letztem Bild wieder von vorn
        # Intrinsics optional (wenn 0/None -> einfache Defaults aus Bildgröße)
        self.declare_parameter('fx', 0.0)
        self.declare_parameter('fy', 0.0)
        self.declare_parameter('cx', 0.0)
        self.declare_parameter('cy', 0.0)
        self.declare_parameter('distortion_model', 'plumb_bob')
        self.declare_parameter('D', [0.0, 0.0, 0.0, 0.0, 0.0])

        self.bridge = CvBridge()
        self.pub_rgb  = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.pub_info = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)

        # Bilder einsammeln
        image_dir = self.get_parameter('image_dir').value
        if not image_dir or not os.path.isdir(image_dir):
            raise RuntimeError(f"Parameter image_dir ist ungültig: '{image_dir}'")
        files = [p for p in glob.glob(os.path.join(image_dir, '*')) if os.path.splitext(p)[1].lower() in IMG_EXTS]
        files.sort(key=natural_key)
        if not files:
            raise RuntimeError(f"Keine Bilder in {image_dir} gefunden.")
        self.files = files
        self.idx = 0

        # FPS/Timer
        fps = float(self.get_parameter('fps').value)
        period = max(1.0 / max(fps, 0.001), 0.001)
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(f"camera_folder streaming {len(files)} Bilder aus: {image_dir} @ {fps:.2f} FPS")

    def _make_info(self, stamp, w, h):
        fx = float(self.get_parameter('fx').value)
        fy = float(self.get_parameter('fy').value)
        cx = float(self.get_parameter('cx').value)
        cy = float(self.get_parameter('cy').value)

        # simple Defaults, falls nicht gesetzt
        if fx <= 0 or fy <= 0:
            fx = fy = 1.2 * max(w, h)
        if cx <= 0 or cy <= 0:
            cx, cy = w / 2.0, h / 2.0

        info = CameraInfo()
        info.header = Header(stamp=stamp, frame_id=self.get_parameter('frame_color').value)
        info.width, info.height = w, h
        info.distortion_model = self.get_parameter('distortion_model').value
        info.d = list(self.get_parameter('D').value)

        info.k = [fx, 0.0, cx,
                  0.0, fy, cy,
                  0.0, 0.0, 1.0]
        info.r = [1.0,0.0,0.0,
                  0.0,1.0,0.0,
                  0.0,0.0,1.0]
        info.p = [fx, 0.0, cx, 0.0,
                  0.0, fy, cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        return info

    def tick(self):
        if self.idx >= len(self.files):
            if bool(self.get_parameter('loop').value):
                self.idx = 0
            else:
                self.get_logger().info("Ende erreicht (loop=False) – stoppe Timer.")
                self.timer.cancel()
                return

        path = self.files[self.idx]
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn(f"Konnte Bild nicht lesen: {path}")
            self.idx += 1
            return

        # BGR->RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        stamp = self.get_clock().now().to_msg()

        # publish RGB
        msg = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
        msg.header.stamp = stamp
        msg.header.frame_id = self.get_parameter('frame_color').value
        self.pub_rgb.publish(msg)

        # optional CameraInfo
        if bool(self.get_parameter('publish_info').value):
            info_msg = self._make_info(stamp, w, h)
            self.pub_info.publish(info_msg)

        self.get_logger().debug(f"Published {os.path.basename(path)} ({w}x{h})")
        self.idx += 1


def main():
    rclpy.init()
    node = CameraFolder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
