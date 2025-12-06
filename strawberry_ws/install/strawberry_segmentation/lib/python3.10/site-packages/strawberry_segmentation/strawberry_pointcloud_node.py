#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2


class StrawberryPointCloudNode(Node):
    """
    Nimmt:
      - depth_topic: maskiertes Tiefenbild (z.B. /seg/depth_masked, 16UC1 in mm oder 32FC1 in m)
      - camera_info_topic: Kameraintrinsics (fx, fy, cx, cy) für die gleiche Kamera

    Gibt aus:
      - cloud_topic: PointCloud2 mit nur Erdbeer-Punkten (nicht-null Tiefen)
    """

    def __init__(self):
        super().__init__("strawberry_pointcloud")

        # --- Parameter ---
        self.declare_parameter("depth_topic", "/seg/depth_masked")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("cloud_topic", "/seg/strawberry_cloud")
        self.declare_parameter("frame_id", "")          # leer -> Frame aus Depth-Image
        self.declare_parameter("downsample_step", 1)    # >1 um Punkte zu sparen (z.B. 2, 4, ...)

        depth_topic = self.get_parameter("depth_topic").value
        cam_info_topic = self.get_parameter("camera_info_topic").value
        cloud_topic = self.get_parameter("cloud_topic").value
        self._frame_id = self.get_parameter("frame_id").value
        self._step = int(self.get_parameter("downsample_step").value)

        if self._step < 1:
            self._step = 1

        self.get_logger().info(
            f"StrawberryPointCloudNode startet:\n"
            f"  depth_topic       = {depth_topic}\n"
            f"  camera_info_topic = {cam_info_topic}\n"
            f"  cloud_topic       = {cloud_topic}\n"
            f"  downsample_step   = {self._step}"
        )

        self.bridge = CvBridge()

        # Intrinsics
        self._fx = None
        self._fy = None
        self._cx = None
        self._cy = None

        # QoS: Depth & CameraInfo kommen mit BEST_EFFORT/KEEP_LAST in deinem Setup zurecht
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Subscriber
        self.sub_depth = self.create_subscription(
            Image, depth_topic, self.depth_cb, qos
        )
        self.sub_info = self.create_subscription(
            CameraInfo, cam_info_topic, self.camera_info_cb, 10
        )

        # Publisher
        self.pub_cloud = self.create_publisher(PointCloud2, cloud_topic, 10)

        self._warned_no_intrinsics = False

    # ----------------- CameraInfo Callback -----------------
    def camera_info_cb(self, msg: CameraInfo):
        # K = [fx, 0, cx,  0, fy, cy,  0, 0, 1]
        self._fx = msg.k[0]
        self._fy = msg.k[4]
        self._cx = msg.k[2]
        self._cy = msg.k[5]

        if not self._frame_id:
            self._frame_id = msg.header.frame_id

    # ----------------- Depth Callback -----------------
    def depth_cb(self, depth_msg: Image):
        # Warten, bis Intrinsics da sind
        if self._fx is None or self._fy is None or self._cx is None or self._cy is None:
            if not self._warned_no_intrinsics:
                self.get_logger().warn(
                    "Noch keine CameraInfo empfangen – kann keine Punktwolke erzeugen."
                )
                self._warned_no_intrinsics = True
            return

        # Depth in numpy holen (16UC1 in mm oder 32FC1 in m)
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        if depth.ndim != 2:
            self.get_logger().warn(f"Depth hat unerwartete Shape: {depth.shape}")
            return

        # In Meter umwandeln
        if depth.dtype == np.uint16:
            z_m = depth.astype(np.float32) / 1000.0  # mm -> m
        else:
            z_m = depth.astype(np.float32)          # hoffentlich schon m

        H, W = z_m.shape

        # Downsampling (für Performance): nur jedes n-te Pixel
        step = self._step
        if step > 1:
            z_sub = z_m[0:H:step, 0:W:step]
            v_grid, u_grid = np.mgrid[0:H:step, 0:W:step]
        else:
            z_sub = z_m
            v_grid, u_grid = np.mgrid[0:H, 0:W]

        # Nur gültige Tiefen (z > 0)
        valid = z_sub > 0.0
        if not np.any(valid):
            # nichts zu tun
            return

        z = z_sub[valid]
        v = v_grid[valid].astype(np.float32)
        u = u_grid[valid].astype(np.float32)

        fx, fy, cx, cy = float(self._fx), float(self._fy), float(self._cx), float(self._cy)

        # Kamera-Pinhole-Modell: X, Y, Z in Kamerakoordinaten
        X = (u - cx) * z / fx
        Y = (v - cy) * z / fy
        Z = z

        # Punkte als Nx3-Array
        points = np.stack((X, Y, Z), axis=-1).astype(np.float32)
        points_list = points.reshape(-1, 3).tolist()

        # Header
        header = Header()
        header.stamp = depth_msg.header.stamp
        header.frame_id = self._frame_id if self._frame_id else depth_msg.header.frame_id

        # PointCloud2 erstellen
        cloud_msg = point_cloud2.create_cloud_xyz32(header, points_list)
        self.pub_cloud.publish(cloud_msg)


def main():
    rclpy.init()
    node = StrawberryPointCloudNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
