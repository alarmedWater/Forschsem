#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Snapshot node: save synced RGB + depth on request.

Subscribes:
  - /camera/color/image_raw
  - /camera/aligned_depth_to_color/image_raw

Provides:
  - /capture_snapshot  (strawberry_msgs/srv/CaptureSnapshot)

The service request carries:
  - plant_id (int32)
  - view_id  (int32)

Parameters:
  - rgb_topic   (string)
  - depth_topic (string)
  - output_dir  (string)
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image

# Neuer eigener Service-Typ
from strawberry_msgs.srv import CaptureSnapshot


class StrawberrySnapshotNode(Node):
    """Store the latest RGB+depth and save them on service call."""

    def __init__(self) -> None:
        super().__init__("strawberry_snapshot")

        # Parameters
        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter(
            "depth_topic",
            "/camera/aligned_depth_to_color/image_raw",
        )
        self.declare_parameter("output_dir", "snapshots")

        rgb_topic = self.get_parameter("rgb_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        self._output_dir = self.get_parameter("output_dir").value

        os.makedirs(self._output_dir, exist_ok=True)

        self.bridge = CvBridge()

        # Latest synced images (numpy arrays)
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None

        # Simple counter per plant+view (index innerhalb eines Laufs)
        self._counter: int = 0

        # Subscribers with approximate sync
        self.sub_rgb = Subscriber(self, Image, rgb_topic)
        self.sub_depth = Subscriber(self, Image, depth_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth],
            queue_size=10,
            slop=0.05,
        )
        self.sync.registerCallback(self._sync_cb)

        # Service mit eigenem Typ
        self.srv = self.create_service(
            CaptureSnapshot,
            "capture_snapshot",
            self._capture_snapshot_cb,
        )

        self.get_logger().info(
            "StrawberrySnapshotNode started.\n"
            f"  rgb_topic   = {rgb_topic}\n"
            f"  depth_topic = {depth_topic}\n"
            f"  output_dir  = {self._output_dir}"
        )

    # ------------------------------------------------------------------ #
    # Callback: synced RGB + depth                                       #
    # ------------------------------------------------------------------ #

    def _sync_cb(self, rgb_msg: Image, depth_msg: Image) -> None:
        """Store the latest synchronized RGB + depth frames."""
        # Keep in memory; we don't save yet.
        self._latest_rgb = self.bridge.imgmsg_to_cv2(
            rgb_msg,
            desired_encoding="rgb8",
        )
        # Depth might be float32 or uint16, keep as-is
        self._latest_depth = self.bridge.imgmsg_to_cv2(
            depth_msg,
            desired_encoding="passthrough",
        )

    # ------------------------------------------------------------------ #
    # Service: capture_snapshot (CaptureSnapshot.srv)                    #
    # ------------------------------------------------------------------ #

    def _capture_snapshot_cb(
        self,
        request: CaptureSnapshot.Request,
        response: CaptureSnapshot.Response,
    ) -> CaptureSnapshot.Response:
        """Save the latest RGB + depth images to disk."""
        if self._latest_rgb is None or self._latest_depth is None:
            response.success = False
            response.message = "No synced RGB+depth available yet."
            return response

        plant_id = int(request.plant_id)
        view_id = int(request.view_id)

        # Build file names
        idx = self._counter
        self._counter += 1

        base = f"plant{plant_id}_view{view_id}_{idx:03d}"
        rgb_name = f"{base}_color.png"
        depth_name = f"{base}_depth.png"

        rgb_path = os.path.join(self._output_dir, rgb_name)
        depth_path = os.path.join(self._output_dir, depth_name)

        # Convert RGB (we stored as RGB; cv2 wants BGR for writing)
        rgb_bgr = cv2.cvtColor(self._latest_rgb, cv2.COLOR_RGB2BGR)

        # For depth: if float32 meters, convert to 16-bit mm for PNG
        depth = self._latest_depth
        if depth.dtype == np.float32:
            depth_mm = (depth * 1000.0).clip(0, 65535).astype(np.uint16)
        else:
            depth_mm = depth

        ok_rgb = cv2.imwrite(rgb_path, rgb_bgr)
        ok_depth = cv2.imwrite(depth_path, depth_mm)

        if not ok_rgb or not ok_depth:
            response.success = False
            response.message = (
                "Failed to write one or both images: "
                f"{rgb_path}, {depth_path}"
            )
            return response

        response.success = True
        response.message = (
            f"Saved snapshot: {rgb_name}, {depth_name} "
            f"(plant_id={plant_id}, view_id={view_id}, idx={idx})"
        )
        self.get_logger().info(response.message)
        return response


def main() -> None:
    rclpy.init()
    node = StrawberrySnapshotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
