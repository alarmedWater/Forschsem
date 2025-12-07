#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, time
from pathlib import Path
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo


def to_image_msg_rgb8(arr: np.ndarray, stamp, frame_id: str) -> Image:
    # arr is HxWx3 RGB uint8
    msg = Image()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = 'rgb8'
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = arr.tobytes()
    return msg


def to_image_msg_depth16(arr_u16: np.ndarray, stamp, frame_id: str) -> Image:
    # arr_u16 is HxW uint16 (millimeters)
    msg = Image()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height, msg.width = arr_u16.shape[:2]
    msg.encoding = '16UC1'
    msg.is_bigendian = 0
    msg.step = msg.width * 2
    msg.data = arr_u16.tobytes()
    return msg


class CameraFolderNode(Node):
    def __init__(self):
        super().__init__('camera_folder')

        # ---------- Parameters with defaults ----------
        self.declare_parameter('rgb_dir', '')
        self.declare_parameter('depth_dir', '')
        self.declare_parameter('rgb_pattern', 'color_*.png')
        self.declare_parameter('depth_pattern', 'depth_*.png')
        self.declare_parameter('fps', 2.0)
        self.declare_parameter('loop', False)
        self.declare_parameter('publish_depth', True)

        # intrinsics (placeholders; set to your real values)
        self.declare_parameter('fx', 900.0)
        self.declare_parameter('fy', 900.0)
        self.declare_parameter('cx', 640.0)
        self.declare_parameter('cy', 360.0)

        # frames
        self.declare_parameter('frame_color', 'camera_color_optical_frame')
        self.declare_parameter('frame_depth', 'camera_color_optical_frame')

        # ---------- Publishers ----------
        self.pub_rgb   = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.pub_depth = self.create_publisher(Image, '/camera/aligned_depth_to_color/image_raw', 10)
        self.pub_info  = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)

        # ---------- Gather files ----------
        self.rgb_paths, self.depth_paths = self._collect_file_pairs()
        if not self.rgb_paths:
            raise RuntimeError("No RGB images found. Check rgb_dir and rgb_pattern.")

        # ---------- Timer ----------
        fps = float(self.get_parameter('fps').value)
        self.period = max(1.0 / max(fps, 0.1), 1e-3)
        self.timer = self.create_timer(self.period, self._tick)

        self.i = 0
        self.done = False
        self.logged_depth_warn = False

        self.get_logger().info(
            f"camera_folder: {len(self.rgb_paths)} RGB files, "
            f"{len(self.depth_paths)} depth files | fps={fps} loop={self.get_parameter('loop').value}"
        )

    def P(self, name):
        return self.get_parameter(name).value

    def _collect_file_pairs(self):
        rgb_dir = str(self.P('rgb_dir'))
        depth_dir = str(self.P('depth_dir'))
        rgb_pattern = str(self.P('rgb_pattern'))
        depth_pattern = str(self.P('depth_pattern'))

        rgb_paths = sorted(glob.glob(str(Path(rgb_dir) / rgb_pattern)))
        depth_paths = sorted(glob.glob(str(Path(depth_dir) / depth_pattern))) if depth_dir else []

        # We’ll pair by index (sorted order). If counts differ, we’ll publish RGB only.
        if depth_paths and len(depth_paths) != len(rgb_paths):
            self.get_logger().warn(
                f"Depth/RGB count mismatch ({len(depth_paths)} vs {len(rgb_paths)}). "
                f"Will publish RGB only unless indices match."
            )
        return rgb_paths, depth_paths

    def _make_camera_info(self, stamp, w: int, h: int) -> CameraInfo:
        fx, fy, cx, cy = float(self.P('fx')), float(self.P('fy')), float(self.P('cx')), float(self.P('cy'))
        frame = str(self.P('frame_color'))
        msg = CameraInfo()
        msg.header = Header(stamp=stamp, frame_id=frame)
        msg.width, msg.height = w, h
        msg.distortion_model = 'plumb_bob'
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = [fx, 0.0, cx,
                 0.0, fy, cy,
                 0.0, 0.0, 1.0]
        msg.r = [1.0,0.0,0.0,
                 0.0,1.0,0.0,
                 0.0,0.0,1.0]
        msg.p = [fx, 0.0, cx, 0.0,
                 0.0, fy, cy, 0.0,
                 0.0, 0.0, 1.0, 0.0]
        return msg

    def _tick(self):
        if self.done:
            return

        loop = bool(self.P('loop'))
        idx = self.i
        if idx >= len(self.rgb_paths):
            if loop:
                idx = idx % len(self.rgb_paths)
                self.i = idx
            else:
                self.get_logger().info("Finished all files (loop=False). Stopping timer.")
                self.timer.cancel()
                self.done = True
                return

        stamp = self.get_clock().now().to_msg()
        frame_color = str(self.P('frame_color'))
        frame_depth = str(self.P('frame_depth'))

        # ---- RGB ----
        rgb_path = self.rgb_paths[idx]
        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn(f"Failed to read RGB: {rgb_path}")
            self.i += 1
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        rgb_msg = to_image_msg_rgb8(rgb, stamp, frame_color)

        # ---- Depth (optional) ----
        depth_msg = None
        if bool(self.P('publish_depth')) and idx < len(self.depth_paths):
            depth_path = self.depth_paths[idx]
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                if not self.logged_depth_warn:
                    self.get_logger().warn(f"Failed to read depth: {depth_path} (publishing RGB only).")
                    self.logged_depth_warn = True
            else:
                # ensure 2D and uint16 (millimeters). If your PNG is 16-bit, this preserves it.
                if depth_raw.ndim == 3:
                    depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
                if depth_raw.dtype != np.uint16:
                    # assume meters in float or 8-bit; convert to mm uint16 conservatively
                    depth_raw = (depth_raw.astype(np.float32) * 1000.0).clip(0, 65535).astype(np.uint16)
                # Resize to color if needed (aligned)
                if depth_raw.shape[:2] != (H, W):
                    depth_raw = cv2.resize(depth_raw, (W, H), interpolation=cv2.INTER_NEAREST)
                depth_msg = to_image_msg_depth16(depth_raw, stamp, frame_depth)

        # ---- CameraInfo ----
        info_msg = self._make_camera_info(stamp, W, H)

        # ---- Publish ----
        self.pub_rgb.publish(rgb_msg)
        if depth_msg is not None:
            self.pub_depth.publish(depth_msg)
        self.pub_info.publish(info_msg)

        self.i += 1


def main():
    rclpy.init()
    node = CameraFolderNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
