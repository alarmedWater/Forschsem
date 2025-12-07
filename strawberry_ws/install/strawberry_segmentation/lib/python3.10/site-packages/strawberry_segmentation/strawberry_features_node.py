#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strawberry features and point cloud node.

This node:

- Subscribes:
    * depth_topic: masked depth image (e.g. /seg/depth_masked)
    * label_topic: instance label image (mono16, e.g. /seg/label_image)
    * camera_info_topic: camera intrinsics (fx, fy, cx, cy)

- For each strawberry instance (label > 0, with enough points):
    * builds a 3D point cloud in the camera frame
    * computes simple 3D features:
        - N points
        - centroid (x, y, z)
        - axis-aligned bounding box (extent and box volume)

- Publishes (configurable via parameters):
    * /seg/strawberry_cloud:
        all strawberry points in a single PointCloud2
    * /seg/strawberry_cloud_selected:
        only the currently selected instance (for RViz)

- Logs features per instance for debugging / evaluation.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

import message_filters


class StrawberryFeaturesNode(Node):
    """Compute per-instance strawberry features and point clouds."""

    def __init__(self) -> None:
        super().__init__("strawberry_features")

        # --- Parameters ---
        self.declare_parameter("depth_topic", "/seg/depth_masked")
        self.declare_parameter("label_topic", "/seg/label_image")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")

        self.declare_parameter("downsample_step", 1)   # pixel skip for performance
        self.declare_parameter("min_points", 50)       # minimum points per instance
        self.declare_parameter("frame_id", "")         # empty -> use from CameraInfo
        self.declare_parameter("profile", False)

        # Experiment metadata
        self.declare_parameter("plant_id", 0)
        self.declare_parameter("view_id", 0)

        # One selected instance for RViz, etc.
        self.declare_parameter("selected_instance_id", 1)
        self.declare_parameter("publish_selected_cloud", True)

        # Optional: publish all strawberries in one cloud
        self.declare_parameter("publish_all_cloud", True)
        self.declare_parameter("cloud_topic_all", "/seg/strawberry_cloud")
        self.declare_parameter(
            "cloud_topic_selected", "/seg/strawberry_cloud_selected"
        )

        depth_topic = self.get_parameter("depth_topic").value
        label_topic = self.get_parameter("label_topic").value
        cam_info_topic = self.get_parameter("camera_info_topic").value

        self._step = int(self.get_parameter("downsample_step").value)
        self._min_points = int(self.get_parameter("min_points").value)
        self._frame_id = self.get_parameter("frame_id").value
        self._profile = bool(self.get_parameter("profile").value)

        self._selected_instance_id = int(
            self.get_parameter("selected_instance_id").value
        )
        self._publish_selected_cloud = bool(
            self.get_parameter("publish_selected_cloud").value
        )
        self._publish_all_cloud = bool(
            self.get_parameter("publish_all_cloud").value
        )

        self._cloud_topic_all = self.get_parameter("cloud_topic_all").value
        self._cloud_topic_selected = self.get_parameter(
            "cloud_topic_selected"
        ).value

        if self._step < 1:
            self._step = 1

        self.get_logger().info(
            "StrawberryFeaturesNode starting:\n"
            f"  depth_topic         = {depth_topic}\n"
            f"  label_topic         = {label_topic}\n"
            f"  camera_info_topic   = {cam_info_topic}\n"
            f"  downsample_step     = {self._step}\n"
            f"  min_points          = {self._min_points}\n"
            f"  selected_instance   = {self._selected_instance_id}\n"
            f"  publish_selected    = {self._publish_selected_cloud}\n"
            f"  publish_all_cloud   = {self._publish_all_cloud}\n"
            f"  cloud_topic_all     = {self._cloud_topic_all}\n"
            f"  cloud_topic_selected= {self._cloud_topic_selected}"
        )

        self.bridge = CvBridge()

        # Camera intrinsics
        self._fx: float | None = None
        self._fy: float | None = None
        self._cx: float | None = None
        self._cy: float | None = None

        # Last per-instance point clouds (for possible later use)
        # Dict: inst_id -> np.ndarray of shape (N, 3)
        self._last_clouds: Dict[int, np.ndarray] = {}

        # QoS for depth/label
        qos_depth_label = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # --- Synchronized subscribers for depth & label ---
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

        # CameraInfo subscription (not synchronized)
        self.sub_info = self.create_subscription(
            CameraInfo, cam_info_topic, self.camera_info_cb, 10
        )

        # Publishers for point clouds
        self.pub_cloud_all = self.create_publisher(
            PointCloud2, self._cloud_topic_all, 10
        )
        self.pub_cloud_selected = self.create_publisher(
            PointCloud2, self._cloud_topic_selected, 10
        )

        self._warned_no_intrinsics = False

    # --------------------------------------------------------------------- #
    # CameraInfo callback
    # --------------------------------------------------------------------- #
    def camera_info_cb(self, msg: CameraInfo) -> None:
        """Store camera intrinsics from CameraInfo message."""
        # K = [fx, 0, cx,  0, fy, cy,  0, 0, 1]
        self._fx = msg.k[0]
        self._fy = msg.k[4]
        self._cx = msg.k[2]
        self._cy = msg.k[5]

        if not self._frame_id:
            self._frame_id = msg.header.frame_id

    # --------------------------------------------------------------------- #
    # Depth + label synchronized callback
    # --------------------------------------------------------------------- #
    def sync_cb(self, depth_msg: Image, label_msg: Image) -> None:
        """Process one synchronized depth + label pair."""
        t0 = time.time()

        # Experiment metadata (can be changed at runtime via ros2 param set)
        plant_id = int(self.get_parameter("plant_id").value)
        view_id = int(self.get_parameter("view_id").value)

        # Wait until intrinsics are available.
        if (
            self._fx is None
            or self._fy is None
            or self._cx is None
            or self._cy is None
        ):
            if not self._warned_no_intrinsics:
                self.get_logger().warn(
                    "No CameraInfo received yet – cannot compute features."
                )
                self._warned_no_intrinsics = True
            return

        # Convert depth & label to numpy
        depth = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding="passthrough"
        )
        label = self.bridge.imgmsg_to_cv2(
            label_msg, desired_encoding="mono16"
        )

        if depth.shape[:2] != label.shape[:2]:
            self.get_logger().warn(
                "Shape mismatch depth=%s label=%s – "
                "check if depth & label are aligned!",
                depth.shape,
                label.shape,
            )
            return

        # Convert depth to meters
        if depth.dtype == np.uint16:
            z_m = depth.astype(np.float32) / 1000.0  # mm -> m
        else:
            z_m = depth.astype(np.float32)          # assume already in meters

        height, width = z_m.shape

        # Downsample if requested
        step = self._step
        if step > 1:
            z_sub = z_m[0:height:step, 0:width:step]
            label_sub = label[0:height:step, 0:width:step]
            v_grid, u_grid = np.mgrid[0:height:step, 0:width:step]
        else:
            z_sub = z_m
            label_sub = label
            v_grid, u_grid = np.mgrid[0:height, 0:width]

        # Only valid depth pixels
        valid_depth = z_sub > 0.0
        if not np.any(valid_depth):
            return

        # All instance IDs > 0 with at least one valid depth pixel
        unique_ids = np.unique(label_sub[valid_depth])
        unique_ids = unique_ids[unique_ids > 0]

        if unique_ids.size == 0:
            return

        fx = float(self._fx)
        fy = float(self._fy)
        cx = float(self._cx)
        cy = float(self._cy)

        # Current per-instance clouds
        current_clouds: Dict[int, np.ndarray] = {}

        # ------------------------------------------------------------------ #
        # 1) Build per-instance clouds and features
        # ------------------------------------------------------------------ #
        log_lines: list[str] = [
            f"Plant {plant_id}, view {view_id} – "
            f"instances: {unique_ids.tolist()}"
        ]

        for inst_id in unique_ids:
            mask_inst = valid_depth & (label_sub == inst_id)
            if not np.any(mask_inst):
                continue

            z_i = z_sub[mask_inst]
            v_i = v_grid[mask_inst].astype(np.float32)
            u_i = u_grid[mask_inst].astype(np.float32)

            # Back-project into camera frame (pinhole model)
            x_i = (u_i - cx) * z_i / fx
            y_i = (v_i - cy) * z_i / fy

            points_i = np.stack(
                (x_i, y_i, z_i), axis=-1
            ).astype(np.float32)
            n_points = points_i.shape[0]

            if n_points < self._min_points:
                # Too few points, likely noise or tiny masks.
                continue

            inst_key = int(inst_id)
            current_clouds[inst_key] = points_i

            centroid = points_i.mean(axis=0)
            p_min = points_i.min(axis=0)
            p_max = points_i.max(axis=0)
            extent = p_max - p_min
            volume_box = float(extent[0] * extent[1] * extent[2])

            log_lines.append(
                "  Instance %d: N=%d, "
                "Centroid=(%.3f, %.3f, %.3f) m, "
                "Extent=(%.3f, %.3f, %.3f) m, "
                "BoxVol=%.6f m^3"
                % (
                    inst_key,
                    n_points,
                    centroid[0],
                    centroid[1],
                    centroid[2],
                    extent[0],
                    extent[1],
                    extent[2],
                    volume_box,
                )
            )

        if not current_clouds:
            # No instance had enough points.
            return

        # Store clouds for possible later use
        self._last_clouds = current_clouds

        # Compact per-frame log
        self.get_logger().info(
            "Strawberry features for current frame:\n"
            + "\n".join(log_lines)
        )

        # Common header for point clouds
        header = Header()
        header.stamp = depth_msg.header.stamp
        header.frame_id = (
            self._frame_id if self._frame_id else depth_msg.header.frame_id
        )

        # ------------------------------------------------------------------ #
        # 2) Publish all strawberries in one cloud (optional)
        # ------------------------------------------------------------------ #
        if (
            self._publish_all_cloud
            and self.pub_cloud_all.get_subscription_count() > 0
        ):
            all_points_list = []
            for pts in current_clouds.values():
                if pts.size == 0:
                    continue
                all_points_list.extend(pts.tolist())

            cloud_all_msg = point_cloud2.create_cloud_xyz32(
                header, all_points_list
            )
            self.pub_cloud_all.publish(cloud_all_msg)

        # ------------------------------------------------------------------ #
        # 3) Publish selected instance as its own cloud (optional)
        # ------------------------------------------------------------------ #
        if (
            self._publish_selected_cloud
            and self.pub_cloud_selected.get_subscription_count() > 0
        ):
            # Read parameter at runtime so you can change it with ros2 param set.
            self._selected_instance_id = int(
                self.get_parameter("selected_instance_id").value
            )

            pts_sel = current_clouds.get(self._selected_instance_id, None)
            if pts_sel is not None and pts_sel.size > 0:
                cloud_sel_msg = point_cloud2.create_cloud_xyz32(
                    header, pts_sel.tolist()
                )
            else:
                # Publish an empty cloud if the selected instance is not present.
                cloud_sel_msg = point_cloud2.create_cloud_xyz32(header, [])

            self.pub_cloud_selected.publish(cloud_sel_msg)

        if self._profile:
            dt_ms = (time.time() - t0) * 1000.0
            self.get_logger().info("Feature callback: %.2f ms", dt_ms)


def main() -> None:
    rclpy.init()
    node = StrawberryFeaturesNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
