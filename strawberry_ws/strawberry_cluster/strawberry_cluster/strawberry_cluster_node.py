#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Cluster strawberries across multiple views using 3D centroids in world frame.

This node does **not** change your existing pipeline. It simply listens to:

  - depth_topic        (masked depth image, e.g. /seg/depth_masked)
  - label_topic        (instance label image, mono16, e.g. /seg/label_image)
  - camera_info_topic  (intrinsics, e.g. /camera/color/camera_info)
  - camera_pose_topic  (PoseStamped, camera pose in world frame)

For each synchronized depth+label frame and a valid camera pose:

  - compute per-instance 3D points in camera frame
  - compute per-instance 3D centroids in camera frame
  - transform centroids into world frame using the latest camera pose
  - assign instances to clusters in world frame based on a distance threshold

Clusters are stored in memory:

  self._clusters = [
      {
          "id": int,
          "centroid_world": np.ndarray shape (3,),
          "num_points": int,
      },
      ...
  ]

Currently, the node:
  - prints a short summary to the log for each frame
  - maintains cluster centroids over time

You can later extend it to:
  - publish fused per-cluster point clouds
  - dump clusters to disk
"""

from __future__ import annotations

import math
from typing import List, Dict, Any

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge

import message_filters


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float,
                                  qw: float) -> np.ndarray:
    """Convert a unit quaternion into a 3x3 rotation matrix.

    The quaternion is assumed to be in the form (x, y, z, w).
    """
    # Normalize to be safe
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        # Degenerate quaternion → identity
        return np.eye(3, dtype=np.float32)

    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    rot = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return rot


class StrawberryClusterNode(Node):
    """Cluster strawberry instances across multiple views in world coordinates."""

    def __init__(self) -> None:
        super().__init__("strawberry_cluster")

        # ----- Parameters -----
        self.declare_parameter("depth_topic", "/seg/depth_masked")
        self.declare_parameter("label_topic", "/seg/label_image")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("camera_pose_topic", "/camera_pose_world")

        self.declare_parameter("downsample_step", 1)
        self.declare_parameter("min_points", 50)
        self.declare_parameter("distance_threshold", 0.02)  # [m], ~2 cm
        self.declare_parameter("max_clusters", 50)
        self.declare_parameter("profile", False)

        depth_topic = self.get_parameter("depth_topic").value
        label_topic = self.get_parameter("label_topic").value
        cam_info_topic = self.get_parameter("camera_info_topic").value
        cam_pose_topic = self.get_parameter("camera_pose_topic").value

        self._step = int(self.get_parameter("downsample_step").value)
        self._min_points = int(self.get_parameter("min_points").value)
        self._dist_thresh = float(
            self.get_parameter("distance_threshold").value
        )
        self._max_clusters = int(self.get_parameter("max_clusters").value)
        self._profile = bool(self.get_parameter("profile").value)

        if self._step < 1:
            self._step = 1

        self.get_logger().info(
            "StrawberryClusterNode started:\n"
            f"  depth_topic        = {depth_topic}\n"
            f"  label_topic        = {label_topic}\n"
            f"  camera_info_topic  = {cam_info_topic}\n"
            f"  camera_pose_topic  = {cam_pose_topic}\n"
            f"  downsample_step    = {self._step}\n"
            f"  min_points         = {self._min_points}\n"
            f"  distance_threshold = {self._dist_thresh:.3f} m\n"
            f"  max_clusters       = {self._max_clusters}"
        )

        self.bridge = CvBridge()

        # Intrinsics
        self._fx: float | None = None
        self._fy: float | None = None
        self._cx: float | None = None
        self._cy: float | None = None

        # Latest camera pose in world
        self._R_world_cam: np.ndarray | None = None  # 3x3
        self._t_world_cam: np.ndarray | None = None  # (3,)

        # Simple in-memory cluster structure
        # each cluster: {"id": int, "centroid_world": np.ndarray (3,), "num_points": int}
        self._clusters: List[Dict[str, Any]] = []
        self._next_cluster_id: int = 1

        # Flags / warnings
        self._warned_no_intrinsics = False
        self._warned_no_pose = False

        # QoS
        qos_depth_label = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ----- Subscribers -----
        # Depth + label synchronized
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

        # CameraInfo
        self.sub_info = self.create_subscription(
            CameraInfo, cam_info_topic, self.camera_info_cb, 10
        )

        # Camera pose (PoseStamped in world frame)
        self.sub_pose = self.create_subscription(
            PoseStamped, cam_pose_topic, self.camera_pose_cb, 10
        )

    # --------------------------------------------------------------------- #
    # Subscribers                                                           #
    # --------------------------------------------------------------------- #

    def camera_info_cb(self, msg: CameraInfo) -> None:
        """Store camera intrinsics from CameraInfo."""
        self._fx = msg.k[0]
        self._fy = msg.k[4]
        self._cx = msg.k[2]
        self._cy = msg.k[5]

    def camera_pose_cb(self, msg: PoseStamped) -> None:
        """Store latest camera pose (world <- camera) from PoseStamped."""
        pos = msg.pose.position
        ori = msg.pose.orientation

        R = quaternion_to_rotation_matrix(
            ori.x, ori.y, ori.z, ori.w
        )
        t = np.array([pos.x, pos.y, pos.z], dtype=np.float32)

        self._R_world_cam = R
        self._t_world_cam = t

    def sync_cb(self, depth_msg: Image, label_msg: Image) -> None:
        """Depth + label callback: compute instance centroids and cluster."""
        import time

        t0 = time.time()

        # Need intrinsics
        if (self._fx is None or self._fy is None or
                self._cx is None or self._cy is None):
            if not self._warned_no_intrinsics:
                self.get_logger().warn(
                    "No CameraInfo received yet – cannot compute 3D points."
                )
                self._warned_no_intrinsics = True
            return

        # Need camera pose in world
        if self._R_world_cam is None or self._t_world_cam is None:
            if not self._warned_no_pose:
                self.get_logger().warn(
                    "No camera pose received yet on camera_pose_topic – "
                    "cannot cluster in world frame."
                )
                self._warned_no_pose = True
            return

        depth = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding="passthrough"
        )
        label = self.bridge.imgmsg_to_cv2(
            label_msg, desired_encoding="mono16"
        )

        if depth.shape[:2] != label.shape[:2]:
            self.get_logger().warn(
                f"Shape mismatch depth={depth.shape} label={label.shape} – "
                "check depth & label alignment."
            )
            return

        # Convert to meters
        if depth.dtype == np.uint16:
            z_m = depth.astype(np.float32) / 1000.0
        else:
            z_m = depth.astype(np.float32)

        height, width = z_m.shape

        # Downsample
        step = self._step
        if step > 1:
            z_sub = z_m[0:height:step, 0:width:step]
            lbl_sub = label[0:height:step, 0:width:step]
            v_grid, u_grid = np.mgrid[0:height:step, 0:width:step]
        else:
            z_sub = z_m
            lbl_sub = label
            v_grid, u_grid = np.mgrid[0:height, 0:width]

        valid_depth = z_sub > 0.0
        if not np.any(valid_depth):
            return

        # All instance ids with valid depth
        unique_ids = np.unique(lbl_sub[valid_depth])
        unique_ids = unique_ids[unique_ids > 0]

        if unique_ids.size == 0:
            return

        fx = float(self._fx)
        fy = float(self._fy)
        cx = float(self._cx)
        cy = float(self._cy)

        R_wc = self._R_world_cam  # 3x3
        t_wc = self._t_world_cam  # (3,)

        # For logging
        frame_assignments = []

        # ------------------------------------------------------------------
        # Per instance: compute 3D points (camera frame) and centroid,
        # then transform centroid to world and assign to cluster.
        # ------------------------------------------------------------------
        for inst_id in unique_ids:
            mask_inst = valid_depth & (lbl_sub == inst_id)
            if not np.any(mask_inst):
                continue

            z_i = z_sub[mask_inst]
            v_i = v_grid[mask_inst].astype(np.float32)
            u_i = u_grid[mask_inst].astype(np.float32)

            X_i = (u_i - cx) * z_i / fx
            Y_i = (v_i - cy) * z_i / fy
            Z_i = z_i

            points_cam = np.stack(
                (X_i, Y_i, Z_i), axis=-1
            ).astype(np.float32)
            num_pts = points_cam.shape[0]

            if num_pts < self._min_points:
                continue

            centroid_cam = points_cam.mean(axis=0)  # (3,)

            # Transform centroid into world frame
            centroid_world = R_wc @ centroid_cam + t_wc  # (3,)

            cluster_id = self._assign_to_cluster(
                centroid_world, num_pts
            )
            frame_assignments.append(
                (int(inst_id), cluster_id, num_pts)
            )

        if frame_assignments:
            lines = [
                "Cluster assignments this frame:"
            ]
            for inst_id, cid, n_pts in frame_assignments:
                lines.append(
                    f"  instance {inst_id} -> cluster {cid} "
                    f"(N={n_pts})"
                )
            self.get_logger().info("\n".join(lines))

        if self._profile:
            dt = (time.time() - t0) * 1000.0
            self.get_logger().info(
                f"Cluster callback time: {dt:.2f} ms, "
                f"clusters={len(self._clusters)}"
            )

    # ------------------------------------------------------------------ #
    # Clustering logic                                                   #
    # ------------------------------------------------------------------ #

    def _assign_to_cluster(
        self,
        centroid_world: np.ndarray,
        num_points: int,
    ) -> int:
        """Assign an instance to an existing cluster or create a new one.

        Returns:
            cluster_id (int): ID of the chosen or created cluster.
        """
        if not self._clusters:
            cid = self._create_cluster(centroid_world, num_points)
            return cid

        # Find nearest cluster in Euclidean distance
        dists = [
            np.linalg.norm(
                centroid_world - c["centroid_world"]
            )
            for c in self._clusters
        ]
        min_idx = int(np.argmin(dists))
        min_dist = dists[min_idx]

        if min_dist < self._dist_thresh:
            cluster = self._clusters[min_idx]
            # Update centroid with weighted average
            total_points = cluster["num_points"] + num_points
            w_old = cluster["num_points"] / total_points
            w_new = num_points / total_points

            new_centroid = (
                w_old * cluster["centroid_world"]
                + w_new * centroid_world
            )

            cluster["centroid_world"] = new_centroid
            cluster["num_points"] = total_points
            return cluster["id"]

        # No close cluster found -> create new if allowed
        if len(self._clusters) < self._max_clusters:
            cid = self._create_cluster(centroid_world, num_points)
            return cid

        # Otherwise: assign to the nearest anyway (to avoid unbounded growth)
        cluster = self._clusters[min_idx]
        return cluster["id"]

    def _create_cluster(
        self,
        centroid_world: np.ndarray,
        num_points: int,
    ) -> int:
        """Create a new cluster and return its ID."""
        cid = self._next_cluster_id
        self._next_cluster_id += 1

        self._clusters.append(
            {
                "id": cid,
                "centroid_world": centroid_world.copy(),
                "num_points": int(num_points),
            }
        )

        self.get_logger().info(
            f"Created new cluster {cid} at "
            f"({centroid_world[0]:.3f}, "
            f"{centroid_world[1]:.3f}, "
            f"{centroid_world[2]:.3f}) m"
        )
        return cid


def main() -> None:
    rclpy.init()
    node = StrawberryClusterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
