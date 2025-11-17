#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 2 Dummy-Kameranode für eine Intel RealSense D405-ähnliche Publikation.
Publiziert:
  - /camera/color/image_raw                (sensor_msgs/Image,  rgb8)
  - /camera/aligned_depth_to_color/image_raw (sensor_msgs/Image, 16UC1 in Millimetern)
  - /camera/color/camera_info              (sensor_msgs/CameraInfo)
Alle Header-Stamps sind identisch pro Frame; frame_id passt zur Color-Optical-Frame.
Depth-Auflösung folgt standardmäßig der Color-Auflösung (aligned).
"""

import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class CameraDummy(Node):
    def __init__(self) -> None:
        super().__init__('camera_dummy')

        # -----------------------------
        # Parameter (via Launch/YAML änderbar)
        # -----------------------------
        # Color-Auflösung
        self.declare_parameter('color_width', 1280)
        self.declare_parameter('color_height', 720)

        # Depth-Auflösung (aligned: wenn None -> wie Color)
        self.declare_parameter('depth_width', None)
        self.declare_parameter('depth_height', None)

        # FPS
        self.declare_parameter('fps', 30.0)

        # Intrinsics (Platzhalter; bitte an eure reale Auflösung anpassen)
        self.declare_parameter('fx', 900.0)
        self.declare_parameter('fy', 900.0)
        self.declare_parameter('cx', 640.0)
        self.declare_parameter('cy', 360.0)

        # Frames
        self.declare_parameter('frame_color', 'camera_color_optical_frame')
        # Depth ist auf Color registriert -> gleicher Optical-Frame
        self.declare_parameter('frame_depth', 'camera_color_optical_frame')

        # Verzeichnung
        self.declare_parameter('distortion_model', 'plumb_bob')
        self.declare_parameter('D', [0.0, 0.0, 0.0, 0.0, 0.0])  # k1, k2, t1, t2, k3

        # -----------------------------
        # Publisher
        # -----------------------------
        self._bridge = CvBridge()
        self._pub_rgb = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self._pub_depth = self.create_publisher(Image, '/camera/aligned_depth_to_color/image_raw', 10)
        self._pub_info = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)

        # Timer
        fps = float(self.get_parameter('fps').get_parameter_value().double_value)
        period = max(1.0 / fps, 1e-3)
        self._timer = self.create_timer(period, self._tick)

        self._frame_idx = 0
        self.get_logger().info('camera_dummy started (RGB:rgb8, Depth:16UC1 mm, info:CameraInfo)')

    # ---------------------------------------------------------
    # Hilfsfunktionen
    # ---------------------------------------------------------
    def _p(self, name):
        """Kurzform: Parameterwert holen (liefert .value direkt)."""
        return self.get_parameter(name).get_parameter_value()._value

    def _make_cam_info(self, stamp):
        cw = int(self._p('color_width'))
        ch = int(self._p('color_height'))

        fx = float(self._p('fx'))
        fy = float(self._p('fy'))
        cx = float(self._p('cx'))
        cy = float(self._p('cy'))
        frame = str(self._p('frame_color'))

        info = CameraInfo()
        info.header = Header(stamp=stamp, frame_id=frame)
        info.width = cw
        info.height = ch
        info.distortion_model = str(self._p('distortion_model'))
        D_list = list(self._p('D'))
        info.d = D_list

        # Kameramatrix K
        info.k = [fx, 0.0, cx,
                  0.0, fy, cy,
                  0.0, 0.0, 1.0]

        # R (Identität) und P (Mono-Projektionsmatrix)
        info.r = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]

        info.p = [fx, 0.0, cx, 0.0,
                  0.0, fy, cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        return info

    # ---------------------------------------------------------
    # Haupt-Tick: einmal pro Frame alles mit identischem Timestamp publizieren
    # ---------------------------------------------------------
    def _tick(self):
        stamp = self.get_clock().now().to_msg()

        # ---- Auflösungen (Depth folgt Color, wenn nicht explizit gesetzt)
        cw = int(self._p('color_width'))
        ch = int(self._p('color_height'))
        dw = int(self._p('depth_width')) if self._p('depth_width') is not None else cw
        dh = int(self._p('depth_height')) if self._p('depth_height') is not None else ch

        # -----------------------------
        # RGB dummy (rgb8)
        # -----------------------------
        rgb = np.zeros((ch, cw, 3), dtype=np.uint8)
        # einfacher Farbverlauf + animierte zweite Kanalebene
        rgb[..., 0] = np.linspace(0, 255, cw, dtype=np.uint8)
        rgb[..., 1] = (self._frame_idx * 5) % 255
        rgb_msg = self._bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = str(self._p('frame_color'))

        # -----------------------------
        # Depth dummy (16UC1, Millimeter), aligned zu Color
        # -----------------------------
        xv = np.linspace(0.0, 1.0, dw, dtype=np.float32)
        yv = np.linspace(0.0, 1.0, dh, dtype=np.float32)[:, None]
        depth_mm = 300.0 + 200.0 * (xv + yv)  # geneigte Ebene 300..700 mm
        depth_np = depth_mm.astype(np.uint16)

        depth_msg = Image()
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = str(self._p('frame_depth'))  # gleiches Optical-Frame wie Color
        depth_msg.height = dh
        depth_msg.width = dw
        depth_msg.encoding = '16UC1'      # wichtig: Millimeter als unsigned short
        depth_msg.is_bigendian = 0
        depth_msg.step = dw * 2
        depth_msg.data = depth_np.tobytes()

        # -----------------------------
        # CameraInfo (zu Color-Auflösung)
        # -----------------------------
        info_msg = self._make_cam_info(stamp)

        # -----------------------------
        # Publish
        # -----------------------------
        self._pub_rgb.publish(rgb_msg)
        self._pub_depth.publish(depth_msg)
        self._pub_info.publish(info_msg)

        self._frame_idx += 1


def main():
    rclpy.init()
    node = CameraDummy()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
