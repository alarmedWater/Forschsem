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

import message_filters


class StrawberryInstanceCloudNode(Node):
    """
    Erzeugt Punktwolken pro Erdbeer-Instanz.

    Subscribes:
      - depth_topic: maskiertes Tiefenbild (z.B. /seg/depth_masked)
      - label_topic: Instanz-Labelbild (mono16, z.B. /seg/label_image)
      - camera_info_topic: Kameraintrinsics (fx, fy, cx, cy)

    Publishes:
      - cloud_all_topic: alle Erdbeer-Punkte zusammen (z.B. /seg/strawberry_cloud_all)
      - cloud_topic_prefix + "_<i>": pro Instanz i (z.B. /seg/strawberry_cloud_1, _2, ...)
    """

    def __init__(self):
        super().__init__("strawberry_instance_cloud")

        # --- Parameter ---
        self.declare_parameter("depth_topic", "/seg/depth_masked")
        self.declare_parameter("label_topic", "/seg/label_image")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("cloud_all_topic", "/seg/strawberry_cloud_all")
        self.declare_parameter("cloud_topic_prefix", "/seg/strawberry_cloud")
        self.declare_parameter("max_instances", 10)      # maximale Anzahl Topics für Instanzen
        self.declare_parameter("downsample_step", 1)     # Pixel-Skip für Performance
        self.declare_parameter("frame_id", "")           # leer -> Frame aus CameraInfo

        depth_topic = self.get_parameter("depth_topic").value
        label_topic = self.get_parameter("label_topic").value
        cam_info_topic = self.get_parameter("camera_info_topic").value
        cloud_all_topic = self.get_parameter("cloud_all_topic").value
        cloud_topic_prefix = self.get_parameter("cloud_topic_prefix").value
        self._max_instances = int(self.get_parameter("max_instances").value)
        self._step = int(self.get_parameter("downsample_step").value)
        self._frame_id = self.get_parameter("frame_id").value

        if self._step < 1:
            self._step = 1

        self.get_logger().info(
            f"StrawberryInstanceCloudNode startet:\n"
            f"  depth_topic        = {depth_topic}\n"
            f"  label_topic        = {label_topic}\n"
            f"  camera_info_topic  = {cam_info_topic}\n"
            f"  cloud_all_topic    = {cloud_all_topic}\n"
            f"  cloud_topic_prefix = {cloud_topic_prefix}_<i>\n"
            f"  max_instances      = {self._max_instances}\n"
            f"  downsample_step    = {self._step}"
        )

        self.bridge = CvBridge()

        # Intrinsics
        self._fx = None
        self._fy = None
        self._cx = None
        self._cy = None

        # QoS
        qos_depth_label = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # --- Synchronisierte Subscriber für Depth & Label ---
        self.sub_depth = message_filters.Subscriber(self, Image, depth_topic, qos_profile=qos_depth_label)
        self.sub_label = message_filters.Subscriber(self, Image, label_topic, qos_profile=qos_depth_label)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_depth, self.sub_label],
            queue_size=10,
            slop=0.05,
        )
        self.ts.registerCallback(self.sync_cb)

        # Separater Subscriber für CameraInfo (nicht synchronisiert)
        self.sub_info = self.create_subscription(
            CameraInfo, cam_info_topic, self.camera_info_cb, 10
        )

        # Publisher: alle Instanzen zusammen + pro Instanz
        self.pub_cloud_all = self.create_publisher(PointCloud2, cloud_all_topic, 10)

        self.cloud_topic_prefix = cloud_topic_prefix
        self.pub_cloud_instances = [
            self.create_publisher(
                PointCloud2, f"{cloud_topic_prefix}_{i+1}", 10
            )
            for i in range(self._max_instances)
        ]

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

    # ----------------- Depth + Label Callback -----------------
    def sync_cb(self, depth_msg: Image, label_msg: Image):
        import time
        t0 = time.time()

        # Warten, bis Intrinsics da sind
        if self._fx is None or self._fy is None or self._cx is None or self._cy is None:
            if not self._warned_no_intrinsics:
                self.get_logger().warn(
                    "Noch keine CameraInfo empfangen – kann keine Punktwolke erzeugen."
                )
                self._warned_no_intrinsics = True
            return

        # Depth und Label in numpy holen
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        label = self.bridge.imgmsg_to_cv2(label_msg, desired_encoding="mono16")

        if depth.shape[:2] != label.shape[:2]:
            self.get_logger().warn(
                f"Shape mismatch depth={depth.shape} label={label.shape} – "
                f"prüfe, ob depth & label aligned sind!"
            )
            return

        # In Meter umwandeln
        if depth.dtype == np.uint16:
            z_m = depth.astype(np.float32) / 1000.0  # mm -> m
        else:
            z_m = depth.astype(np.float32)          # hoffentlich schon m

        H, W = z_m.shape

        # Downsampling
        step = self._step
        if step > 1:
            z_sub = z_m[0:H:step, 0:W:step]
            label_sub = label[0:H:step, 0:W:step]
            v_grid, u_grid = np.mgrid[0:H:step, 0:W:step]
        else:
            z_sub = z_m
            label_sub = label
            v_grid, u_grid = np.mgrid[0:H, 0:W]

        # Nur gültige Tiefen
        valid_depth = z_sub > 0.0
        if not np.any(valid_depth):
            return

        # Alle Instanz-IDs > 0
        unique_ids = np.unique(label_sub[valid_depth])
        unique_ids = unique_ids[unique_ids > 0]

        if unique_ids.size == 0:
            return

        fx, fy, cx, cy = float(self._fx), float(self._fy), float(self._cx), float(self._cy)

        # ----- 1) ALLE Instanzen in eine Cloud packen -----
        mask_all = valid_depth & (label_sub > 0)
        z_all = z_sub[mask_all]
        v_all = v_grid[mask_all].astype(np.float32)
        u_all = u_grid[mask_all].astype(np.float32)

        X_all = (u_all - cx) * z_all / fx
        Y_all = (v_all - cy) * z_all / fy
        Z_all = z_all

        points_all = np.stack((X_all, Y_all, Z_all), axis=-1).astype(np.float32)
        points_all_list = points_all.reshape(-1, 3).tolist()

        header = Header()
        header.stamp = depth_msg.header.stamp
        header.frame_id = self._frame_id if self._frame_id else depth_msg.header.frame_id

        cloud_all_msg = point_cloud2.create_cloud_xyz32(header, points_all_list)
        self.pub_cloud_all.publish(cloud_all_msg)

        # ----- 2) Pro Instanz eine eigene Cloud -----
        # Hinweis: Instanz-ID (1..N) wird hier einfach der Reihenfolge nach auf Topics gemappt
        # /seg/strawberry_cloud_1, _2, ... bis max_instances.
        for idx, inst_id in enumerate(unique_ids[: self._max_instances]):
            mask_inst = mask_all & (label_sub == inst_id)
            if not np.any(mask_inst):
                continue

            z_i = z_sub[mask_inst]
            v_i = v_grid[mask_inst].astype(np.float32)
            u_i = u_grid[mask_inst].astype(np.float32)

            X_i = (u_i - cx) * z_i / fx
            Y_i = (v_i - cy) * z_i / fy
            Z_i = z_i

            points_i = np.stack((X_i, Y_i, Z_i), axis=-1).astype(np.float32)
            points_i_list = points_i.reshape(-1, 3).tolist()

            cloud_i_msg = point_cloud2.create_cloud_xyz32(header, points_i_list)
            self.pub_cloud_instances[idx].publish(cloud_i_msg)

        # Optional: Profiling-Log
        # dt = (time.time() - t0) * 1000.0
        # self.get_logger().info(f"Instance cloud callback: {dt:.2f} ms, inst={len(unique_ids)}")


def main():
    rclpy.init()
    node = StrawberryInstanceCloudNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
