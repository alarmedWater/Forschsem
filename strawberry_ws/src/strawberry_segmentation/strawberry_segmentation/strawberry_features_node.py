#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strawberry features and point cloud node (ROS 2).

Computes per-instance 3D point clouds and simple 3D features from:
  - masked depth image (/seg/depth_masked)
  - instance label image (/seg/label_image)
  - camera intrinsics (/camera/color/camera_info)

Additionally consumes (synchronized):
  - frame info (FrameInfo)
    Recommended default: /seg/frame_info_depth_masked
    (passthrough from depth_mask, aligned to the masked depth stamp)

Depth scaling:
- If depth is RealSense raw units (uint16): depth_unit="realsense_units"
      z_m = depth_u16 * depth_scale_m_per_unit
- If depth is millimeters (uint16): depth_unit="mm"
      z_m = depth_u16 / 1000.0
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from strawberry_msgs.msg import FrameInfo


class StrawberryFeaturesNode(Node):
    """Compute per-instance strawberry features and point clouds."""

    def __init__(self) -> None:
        super().__init__("strawberry_features")

        # ---------------- Parameters ----------------
        self.declare_parameter("depth_topic", "/seg/depth_masked")
        self.declare_parameter("label_topic", "/seg/label_image")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")

        # IMPORTANT: default is the aligned/passthrough FrameInfo from depth_mask
        self.declare_parameter("frame_info_topic", "/seg/frame_info_depth_masked")

        self.declare_parameter("downsample_step", 1)
        self.declare_parameter("min_points", 50)
        self.declare_parameter("frame_id", "")
        self.declare_parameter("profile", False)

        self.declare_parameter("depth_unit", "realsense_units")
        self.declare_parameter("depth_scale_m_per_unit", 9.999999747378752e-05)

        self.declare_parameter("sync_queue_size", 200)
        self.declare_parameter("sync_slop", 0.2)

        self.declare_parameter("selected_instance_id", 1)
        self.declare_parameter("publish_selected_cloud", True)

        self.declare_parameter("publish_all_cloud", True)
        self.declare_parameter("cloud_topic_all", "/seg/strawberry_cloud")
        self.declare_parameter("cloud_topic_selected", "/seg/strawberry_cloud_selected")

        self.declare_parameter("log_features", True)

        # ---------------- Read parameters ----------------
        depth_topic = self._param_str("depth_topic", "/seg/depth_masked")
        label_topic = self._param_str("label_topic", "/seg/label_image")
        cam_info_topic = self._param_str("camera_info_topic", "/camera/color/camera_info")
        frame_info_topic = self._param_str("frame_info_topic", "/seg/frame_info_depth_masked")

        self._step = max(1, self._param_int("downsample_step", 1))
        self._min_points = max(0, self._param_int("min_points", 50))
        self._frame_id = self._param_str("frame_id", "")
        self._profile = self._param_bool("profile", False)

        self._depth_unit = self._param_str("depth_unit", "realsense_units").strip().lower()
        self._depth_scale = float(
            self._param_float("depth_scale_m_per_unit", 9.999999747378752e-05)
        )

        self._sync_queue_size = max(1, self._param_int("sync_queue_size", 200))
        self._sync_slop = float(self._param_float("sync_slop", 0.2))
        if self._sync_slop <= 0.0:
            self._sync_slop = 0.05

        self._selected_instance_id = self._param_int("selected_instance_id", 1)
        self._publish_selected_cloud = self._param_bool("publish_selected_cloud", True)
        self._publish_all_cloud = self._param_bool("publish_all_cloud", True)

        self._cloud_topic_all = self._param_str("cloud_topic_all", "/seg/strawberry_cloud")
        self._cloud_topic_selected = self._param_str(
            "cloud_topic_selected", "/seg/strawberry_cloud_selected"
        )

        self._log_features = self._param_bool("log_features", True)

        self.get_logger().info(
            "StrawberryFeaturesNode starting:\n"
            f"  depth_topic           = {depth_topic}\n"
            f"  label_topic           = {label_topic}\n"
            f"  frame_info_topic      = {frame_info_topic}\n"
            f"  camera_info_topic     = {cam_info_topic}\n"
            f"  downsample_step       = {self._step}\n"
            f"  min_points            = {self._min_points}\n"
            f"  depth_unit            = {self._depth_unit}\n"
            f"  depth_scale_m_per_unit= {self._depth_scale:.3e}\n"
            f"  sync_queue_size       = {self._sync_queue_size}\n"
            f"  sync_slop             = {self._sync_slop}\n"
            f"  selected_instance     = {self._selected_instance_id}\n"
            f"  publish_selected      = {self._publish_selected_cloud}\n"
            f"  publish_all_cloud     = {self._publish_all_cloud}\n"
            f"  cloud_topic_all       = {self._cloud_topic_all}\n"
            f"  cloud_topic_selected  = {self._cloud_topic_selected}\n"
            f"  log_features          = {self._log_features}\n"
            f"  profile               = {self._profile}"
        )

        self._bridge = CvBridge()

        # ---------------- Intrinsics cache ----------------
        self._fx: Optional[float] = None
        self._fy: Optional[float] = None
        self._cx: Optional[float] = None
        self._cy: Optional[float] = None

        # Last clouds for selected publishing
        self._last_clouds: Dict[int, np.ndarray] = {}

        self._warned_no_intrinsics = False
        self._warned_bad_depth_unit = False

        # ---------------- QoS ----------------
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ---------------- Subscribers (sync) ----------------
        self._sub_depth = message_filters.Subscriber(self, Image, depth_topic, qos_profile=qos)
        self._sub_label = message_filters.Subscriber(self, Image, label_topic, qos_profile=qos)
        self._sub_frame_info = message_filters.Subscriber(
            self, FrameInfo, frame_info_topic, qos_profile=qos
        )

        self._ts = message_filters.ApproximateTimeSynchronizer(
            [self._sub_depth, self._sub_label, self._sub_frame_info],
            queue_size=self._sync_queue_size,
            slop=self._sync_slop,
        )
        self._ts.registerCallback(self._sync_cb)

        # ---------------- CameraInfo (async) ----------------
        self._sub_info = self.create_subscription(
            CameraInfo, cam_info_topic, self._camera_info_cb, 10
        )

        # ---------------- Publishers ----------------
        self._pub_cloud_all = self.create_publisher(PointCloud2, self._cloud_topic_all, 10)
        self._pub_cloud_selected = self.create_publisher(
            PointCloud2, self._cloud_topic_selected, 10
        )

    # ------------------------------------------------------------------ #
    # Param helpers                                                      #
    # ------------------------------------------------------------------ #

    def _param_str(self, name: str, default: str) -> str:
        val: Any = self.get_parameter(name).value
        if val is None:
            return default
        s = str(val).strip()
        return s if s else default

    def _param_bool(self, name: str, default: bool) -> bool:
        val: Any = self.get_parameter(name).value
        if isinstance(val, bool):
            return val
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "y", "on")
        return default

    def _param_int(self, name: str, default: int) -> int:
        val: Any = self.get_parameter(name).value
        if val is None:
            return default
        try:
            return int(val)
        except Exception:  # noqa: BLE001
            return default

    def _param_float(self, name: str, default: float) -> float:
        val: Any = self.get_parameter(name).value
        if val is None:
            return default
        try:
            return float(val)
        except Exception:  # noqa: BLE001
            return default

    # ------------------------------------------------------------------ #
    # CameraInfo callback                                                #
    # ------------------------------------------------------------------ #

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self._fx = float(msg.k[0])
        self._fy = float(msg.k[4])
        self._cx = float(msg.k[2])
        self._cy = float(msg.k[5])

        if not self._frame_id:
            self._frame_id = msg.header.frame_id

    # ------------------------------------------------------------------ #
    # Depth conversion                                                   #
    # ------------------------------------------------------------------ #

    def _depth_to_meters(self, depth: np.ndarray) -> np.ndarray:
        if depth.dtype == np.uint16:
            if self._depth_unit == "realsense_units":
                return depth.astype(np.float32) * float(self._depth_scale)
            if self._depth_unit == "mm":
                return depth.astype(np.float32) / 1000.0

            if not self._warned_bad_depth_unit:
                self.get_logger().warning(
                    f"Unknown depth_unit='{self._depth_unit}'. Falling back to "
                    f"'realsense_units' with scale={self._depth_scale:.3e}."
                )
                self._warned_bad_depth_unit = True
            return depth.astype(np.float32) * float(self._depth_scale)

        return depth.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Sync callback                                                      #
    # ------------------------------------------------------------------ #

    def _sync_cb(self, depth_msg: Image, label_msg: Image, info_msg: FrameInfo) -> None:
        t0 = time.time()

        # Intrinsics required
        if self._fx is None or self._fy is None or self._cx is None or self._cy is None:
            if not self._warned_no_intrinsics:
                self.get_logger().warning(
                    "No CameraInfo received yet – cannot compute features."
                )
                self._warned_no_intrinsics = True
            return

        plant_id = int(info_msg.plant_id)
        view_id = int(info_msg.view_id)
        frame_index = int(info_msg.frame_index)

        depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        label = self._bridge.imgmsg_to_cv2(label_msg, desired_encoding="mono16")

        if depth.shape[:2] != label.shape[:2]:
            self.get_logger().warning(
                f"Shape mismatch depth={depth.shape} label={label.shape} – "
                "check if depth & label are aligned!"
            )
            return

        z_m = self._depth_to_meters(depth)

        height, width = z_m.shape

        # Downsample
        step = self._step
        if step > 1:
            z_sub = z_m[0:height:step, 0:width:step]
            label_sub = label[0:height:step, 0:width:step]
            v_grid, u_grid = np.mgrid[0:height:step, 0:width:step]
        else:
            z_sub = z_m
            label_sub = label
            v_grid, u_grid = np.mgrid[0:height, 0:width]

        valid = np.isfinite(z_sub) & (z_sub > 0.0)
        if not np.any(valid):
            return

        unique_ids = np.unique(label_sub[valid])
        unique_ids = unique_ids[unique_ids > 0]
        if unique_ids.size == 0:
            return

        fx = float(self._fx)
        fy = float(self._fy)
        cx = float(self._cx)
        cy = float(self._cy)

        current_clouds: Dict[int, np.ndarray] = {}

        for inst_id in unique_ids.tolist():
            mask_inst = valid & (label_sub == inst_id)
            if not np.any(mask_inst):
                continue

            z_i = z_sub[mask_inst]
            v_i = v_grid[mask_inst].astype(np.float32)
            u_i = u_grid[mask_inst].astype(np.float32)

            x_i = (u_i - cx) * z_i / fx
            y_i = (v_i - cy) * z_i / fy
            points_i = np.stack((x_i, y_i, z_i), axis=-1).astype(np.float32)

            n_points = int(points_i.shape[0])
            if n_points < self._min_points:
                continue

            current_clouds[int(inst_id)] = points_i

        if not current_clouds:
            return

        self._last_clouds = current_clouds

        # Logging
        if self._log_features:
            inst_list = sorted(current_clouds.keys())
            lines: List[str] = [
                f"Frame {frame_index} | plant {plant_id} | view {view_id} | instances={inst_list}"
            ]
            for inst_id in inst_list:
                pts = current_clouds[inst_id]
                centroid = pts.mean(axis=0)
                p_min = pts.min(axis=0)
                p_max = pts.max(axis=0)
                extent = p_max - p_min
                volume_box = float(extent[0] * extent[1] * extent[2])

                lines.append(
                    "  Instance %d: N=%d, Centroid=(%.3f, %.3f, %.3f) m, "
                    "Extent=(%.3f, %.3f, %.3f) m, BoxVol=%.6f m^3"
                    % (
                        inst_id,
                        int(pts.shape[0]),
                        float(centroid[0]),
                        float(centroid[1]),
                        float(centroid[2]),
                        float(extent[0]),
                        float(extent[1]),
                        float(extent[2]),
                        float(volume_box),
                    )
                )
            self.get_logger().info("\n".join(lines))

        header = Header()
        header.stamp = depth_msg.header.stamp
        header.frame_id = self._frame_id if self._frame_id else depth_msg.header.frame_id

        # Publish all cloud
        if self._publish_all_cloud and self._pub_cloud_all.get_subscription_count() > 0:
            all_points: List[List[float]] = []
            for pts in current_clouds.values():
                all_points.extend(pts.tolist())
            cloud_all_msg = point_cloud2.create_cloud_xyz32(header, all_points)
            self._pub_cloud_all.publish(cloud_all_msg)

        # Publish selected cloud
        if self._publish_selected_cloud and self._pub_cloud_selected.get_subscription_count() > 0:
            self._selected_instance_id = self._param_int(
                "selected_instance_id", self._selected_instance_id
            )
            pts_sel = current_clouds.get(int(self._selected_instance_id))
            cloud_sel_msg = point_cloud2.create_cloud_xyz32(
                header,
                pts_sel.tolist() if (pts_sel is not None and pts_sel.size > 0) else [],
            )
            self._pub_cloud_selected.publish(cloud_sel_msg)

        if self._profile:
            dt_ms = (time.time() - t0) * 1000.0
            self.get_logger().info(f"Feature callback: {dt_ms:.2f} ms")


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
