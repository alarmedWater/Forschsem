#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Intel RealSense D405 camera as a ROS 2 node.

Publishes:
  - /camera/color/image_raw              (sensor_msgs/Image, rgb8)
  - /camera/aligned_depth_to_color/image_raw (sensor_msgs/Image, 32FC1, meters)
  - /camera/color/camera_info            (sensor_msgs/CameraInfo)
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header


class RealsenseCameraNode(Node):
    """ROS 2 node wrapping Intel RealSense streaming."""

    def __init__(self) -> None:
        super().__init__("realsense_camera")

        # Parameters
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("frame_id", "camera_color_optical_frame")

        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        fps = float(self.get_parameter("fps").value)

        self.frame_id = self.get_parameter("frame_id").value

        # Publishers
        self.pub_color = self.create_publisher(
            Image, "/camera/color/image_raw", 10
        )
        self.pub_depth = self.create_publisher(
            Image, "/camera/aligned_depth_to_color/image_raw", 10
        )
        self.pub_info = self.create_publisher(
            CameraInfo, "/camera/color/camera_info", 10
        )

        # Set up RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(
            rs.stream.depth, width, height, rs.format.z16, int(fps)
        )
        self.config.enable_stream(
            rs.stream.color, width, height, rs.format.bgr8, int(fps)
        )

        self.profile = self.pipeline.start(self.config)

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # Get intrinsics from the color stream
        color_profile = self.profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()
        intr = color_profile.get_intrinsics()

        self.width = intr.width
        self.height = intr.height
        self.fx = intr.fx
        self.fy = intr.fy
        self.cx = intr.ppx
        self.cy = intr.ppy

        # Depth scale: meters per depth unit
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        self.get_logger().info("=== RealSense intrinsics ===")
        self.get_logger().info(f"resolution: {self.width} x {self.height}")
        self.get_logger().info(
            f"fx, fy    : {self.fx:.3f}, {self.fy:.3f}"
        )
        self.get_logger().info(
            f"cx, cy    : {self.cx:.3f}, {self.cy:.3f}"
        )
        self.get_logger().info(
            f"depth_scale (m/unit): {self.depth_scale:g}"
        )

        # Timer for streaming (approx. fps, but real timing is driven by camera)
        period = max(1.0 / max(fps, 1.0), 1e-3)
        self.timer = self.create_timer(period, self._timer_cb)

    # ------------------------------------------------------------------ #
    # CameraInfo helper                                                  #
    # ------------------------------------------------------------------ #

    def _make_camera_info(self, stamp) -> CameraInfo:
        msg = CameraInfo()
        msg.header = Header(stamp=stamp, frame_id=self.frame_id)
        msg.width = int(self.width)
        msg.height = int(self.height)

        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        msg.k = [
            float(self.fx),
            0.0,
            float(self.cx),
            0.0,
            float(self.fy),
            float(self.cy),
            0.0,
            0.0,
            1.0,
        ]

        msg.r = [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]

        msg.p = [
            float(self.fx),
            0.0,
            float(self.cx),
            0.0,
            0.0,
            float(self.fy),
            float(self.cy),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]

        return msg

    # ------------------------------------------------------------------ #
    # Timer callback                                                     #
    # ------------------------------------------------------------------ #

    def _timer_cb(self) -> None:
        """Grab a frame from the camera and publish color + depth + info."""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"RealSense wait_for_frames error: {exc}")
            return

        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            self.get_logger().warn("No depth or color frame received.")
            return

        # Depth and color as numpy arrays
        depth_raw = np.asanyarray(depth_frame.get_data())  # uint16
        color_bgr = np.asanyarray(color_frame.get_data())  # BGR8

        # Convert color to RGB8 for your YOLO node
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        # Convert depth to meters (float32)
        depth_m = depth_raw.astype(np.float32) * self.depth_scale

        # Build ROS messages
        stamp = self.get_clock().now().to_msg()

        color_msg = Image()
        color_msg.header = Header(stamp=stamp, frame_id=self.frame_id)
        color_msg.height, color_msg.width = color_rgb.shape[:2]
        color_msg.encoding = "rgb8"
        color_msg.is_bigendian = 0
        color_msg.step = color_msg.width * 3
        color_msg.data = color_rgb.tobytes()

        depth_msg = Image()
        depth_msg.header = Header(stamp=stamp, frame_id=self.frame_id)
        depth_msg.height, depth_msg.width = depth_m.shape
        depth_msg.encoding = "32FC1"
        depth_msg.is_bigendian = 0
        depth_msg.step = depth_msg.width * 4
        depth_msg.data = depth_m.tobytes()

        info_msg = self._make_camera_info(stamp)

        # Publish
        self.pub_color.publish(color_msg)
        self.pub_depth.publish(depth_msg)
        self.pub_info.publish(info_msg)

    # ------------------------------------------------------------------ #
    # Cleanup                                                            #
    # ------------------------------------------------------------------ #

    def destroy_node(self) -> bool:
        """Stop the RealSense pipeline on shutdown."""
        try:
            self.pipeline.stop()
        except Exception:  # noqa: BLE001
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node: RealsenseCameraNode | None = None

    try:
        node = RealsenseCameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
        print("[ROS2] RealSense node shutdown complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
