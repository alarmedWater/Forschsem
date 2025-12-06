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


class StrawberryFeaturesNode(Node):
    """
    Berechnet pro Erdbeer-Instanz einfache 3D-Features aus Depth + Label:

    Subscribes:
      - depth_topic: maskiertes Tiefenbild (z.B. /seg/depth_masked, 16UC1 in mm oder 32FC1 in m)
      - label_topic: Instanz-Labelbild (mono16, z.B. /seg/label_image)
      - camera_info_topic: Kameraintrinsics (fx, fy, cx, cy)

    Features pro Instanz:
      - N Punkte
      - 3D-Schwerpunkt (x,y,z)
      - Axis-aligned Bounding Box (min/max) und Volumen (dx*dy*dz)

    Zusätzlich:
      - Optionales Topic /seg/strawberry_cloud_selected mit der Punktwolke
        der aktuell ausgewählten Instanz (selected_instance_id), um diese
        z.B. in RViz zu visualisieren.

    Intern:
      - self._last_clouds[inst_id] = Nx3-Array im Kamerakoordinatensystem
    """

    def __init__(self):
        super().__init__("strawberry_features")

        # --- Parameter ---
        self.declare_parameter("depth_topic", "/seg/depth_masked")
        self.declare_parameter("label_topic", "/seg/label_image")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")

        self.declare_parameter("downsample_step", 1)    # Pixel-Skip für Performance
        self.declare_parameter("min_points", 50)        # minimale Punktanzahl pro Instanz
        self.declare_parameter("frame_id", "")          # leer -> Frame aus CameraInfo
        self.declare_parameter("profile", False)

        # 👉 neu für RViz-Ansicht einer einzelnen Erdbeere
        self.declare_parameter("selected_instance_id", 1)
        self.declare_parameter("publish_selected_cloud", True)

        depth_topic = self.get_parameter("depth_topic").value
        label_topic = self.get_parameter("label_topic").value
        cam_info_topic = self.get_parameter("camera_info_topic").value
        self._step = int(self.get_parameter("downsample_step").value)
        self._min_points = int(self.get_parameter("min_points").value)
        self._frame_id = self.get_parameter("frame_id").value
        self._profile = bool(self.get_parameter("profile").value)
        self._selected_instance_id = int(self.get_parameter("selected_instance_id").value)
        self._publish_selected_cloud = bool(self.get_parameter("publish_selected_cloud").value)

        if self._step < 1:
            self._step = 1

        self.get_logger().info(
            f"StrawberryFeaturesNode startet:\n"
            f"  depth_topic       = {depth_topic}\n"
            f"  label_topic       = {label_topic}\n"
            f"  camera_info_topic = {cam_info_topic}\n"
            f"  downsample_step   = {self._step}\n"
            f"  min_points        = {self._min_points}\n"
            f"  selected_instance = {self._selected_instance_id}\n"
            f"  publish_selected  = {self._publish_selected_cloud}"
        )

        self.bridge = CvBridge()

        # Intrinsics
        self._fx = None
        self._fy = None
        self._cx = None
        self._cy = None

        # Hier speichern wir letzte Punktwolken pro Instanz (für spätere Weitergabe/Verarbeitung)
        # Dict: inst_id -> np.ndarray mit Shape (N, 3)
        self._last_clouds = {}

        # QoS für Depth/Label
        qos_depth_label = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # --- Synchronisierte Subscriber für Depth & Label ---
        self.sub_depth = message_filters.Subscriber(
            self, Image, depth_topic, qos_profile=qos_depth_label
        )
        self.sub_label = message_filters.Subscriber(
            self, Image, label_topic, qos_profile=qos_depth_label
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_depth, self.sub_label],
            queue_size=10,
            slop=0.05,
        )
        self.ts.registerCallback(self.sync_cb)

        # CameraInfo separat
        self.sub_info = self.create_subscription(
            CameraInfo, cam_info_topic, self.camera_info_cb, 10
        )

        # Publisher für die Punktwolke der ausgewählten Instanz
        self.pub_selected_cloud = self.create_publisher(
            PointCloud2, "/seg/strawberry_cloud_selected", 10
        )

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

        # Intrinsics vorhanden?
        if self._fx is None or self._fy is None or self._cx is None or self._cy is None:
            if not self._warned_no_intrinsics:
                self.get_logger().warn(
                    "Noch keine CameraInfo empfangen – kann keine Features berechnen."
                )
                self._warned_no_intrinsics = True
            return

        # Depth & Label in numpy holen
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        label = self.bridge.imgmsg_to_cv2(label_msg, desired_encoding="mono16")

        if depth.shape[:2] != label.shape[:2]:
            self.get_logger().warn(
                f"Shape mismatch depth={depth.shape} label={label.shape} – "
                f"prüfe, ob depth & label aligned sind!"
            )
            return

        # Depth in Meter umwandeln
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

        # Alle Instanz-IDs > 0 mit mind. einem gültigen Tiefenpixel
        unique_ids = np.unique(label_sub[valid_depth])
        unique_ids = unique_ids[unique_ids > 0]

        if unique_ids.size == 0:
            return

        fx, fy, cx, cy = float(self._fx), float(self._fy), float(self._cx), float(self._cy)

        # Dict für aktuelle Clouds
        current_clouds = {}

        # ---------- pro Instanz Punkte & Features ----------
        log_lines = []
        for inst_id in unique_ids:
            mask_inst = valid_depth & (label_sub == inst_id)
            if not np.any(mask_inst):
                continue

            z_i = z_sub[mask_inst]
            v_i = v_grid[mask_inst].astype(np.float32)
            u_i = u_grid[mask_inst].astype(np.float32)

            # 3D-Koordinaten in Kameraframe
            X_i = (u_i - cx) * z_i / fx
            Y_i = (v_i - cy) * z_i / fy
            Z_i = z_i

            points_i = np.stack((X_i, Y_i, Z_i), axis=-1).astype(np.float32)
            N = points_i.shape[0]

            if N < self._min_points:
                # zu wenige Punkte, vermutlich Rausch- oder Mini-Masken
                continue

            # Features
            centroid = points_i.mean(axis=0)  # (3,)
            p_min = points_i.min(axis=0)
            p_max = points_i.max(axis=0)
            extent = p_max - p_min
            volume_box = float(extent[0] * extent[1] * extent[2])

            current_clouds[int(inst_id)] = points_i

            log_lines.append(
                f"  Instanz {int(inst_id)}: N={N}, "
                f"Centroid=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}) m, "
                f"Extent=({extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}) m, "
                f"BoxVol={volume_box:.6f} m^3"
            )

        # Wenn es keine Instanzen mit genug Punkten gab -> fertig
        if not current_clouds:
            return

        # Clouds für möglichen späteren Zugriff zwischenspeichern
        self._last_clouds = current_clouds

        # Logging (ein kompakter Block pro Frame)
        self.get_logger().info("Strawberry-Features aktueller Frame:\n" + "\n".join(log_lines))

        # ---- Optional: Punktwolke der ausgewählten Instanz publishen ----
        if self._publish_selected_cloud and self.pub_selected_cloud.get_subscription_count() > 0:
            # aktuellen Wert von selected_instance_id aus Params holen (zur Laufzeit veränderbar)
            self._selected_instance_id = int(self.get_parameter("selected_instance_id").value)

            if self._selected_instance_id in current_clouds:
                pts = current_clouds[self._selected_instance_id]  # (N,3)
                pts_list = pts.tolist()

                header = Header()
                header.stamp = depth_msg.header.stamp
                header.frame_id = self._frame_id if self._frame_id else depth_msg.header.frame_id

                cloud_msg = point_cloud2.create_cloud_xyz32(header, pts_list)
                self.pub_selected_cloud.publish(cloud_msg)

        if self._profile:
            dt = (time.time() - t0) * 1000.0
            self.get_logger().info(f"Feature-Callback: {dt:.2f} ms")


def main():
    rclpy.init()
    node = StrawberryFeaturesNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
