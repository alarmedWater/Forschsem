#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strawberry selected overlay node (ROS 2).

Konvention (wie bei DepthMask / Pipeline-Vorgaben):
- QoS: qos_profile_sensor_data (BEST_EFFORT)
- Sync: TimeSynchronizer (exakt gleiche stamps)
- Token stamp: FrameInfo.source_stamp (fallback: FrameInfo.header.stamp)
- Outputs übernehmen immer token_stamp
- Hard stamp guards: image/label/frame_info header stamps müssen übereinstimmen
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from strawberry_msgs.msg import FrameInfo  # type: ignore


class StrawberrySelectedOverlayNode(Node):
    """Highlight the selected instance in the RGB stream."""

    def __init__(self) -> None:
        super().__init__("strawberry_selected_overlay")

        # ---------------- Parameters ----------------
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("label_topic", "/seg/label_image")
        self.declare_parameter("output_topic", "/seg/selected_overlay")

        # IMPORTANT: prefer /seg/frame_info (from seg_ultra) so it is aligned to labels
        self.declare_parameter("frame_info_topic", "/seg/frame_info")
        self.declare_parameter("publish_frame_info", True)
        self.declare_parameter("frame_info_out_topic", "/seg/frame_info_selected_overlay")

        self.declare_parameter("selected_instance_id", 1)
        self.declare_parameter("min_pixels", 50)
        self.declare_parameter("darken_factor", 0.3)
        self.declare_parameter("draw_bbox", True)

        # Sync tuning (queue only; slop is deprecated under exact sync)
        self.declare_parameter("sync_queue_size", 50)
        self.declare_parameter("sync_slop", 0.1)  # kept for backward compat / unused

        # Profiling / debug
        self.declare_parameter("profile", False)
        self.declare_parameter("debug_stamps_once", True)

        # ---------------- Read parameters ----------------
        image_topic = self._param_str("image_topic", "/camera/color/image_raw")
        label_topic = self._param_str("label_topic", "/seg/label_image")
        output_topic = self._param_str("output_topic", "/seg/selected_overlay")

        frame_info_topic = self._param_str("frame_info_topic", "/seg/frame_info")
        self._publish_frame_info = self._param_bool("publish_frame_info", True)
        self._frame_info_out_topic = self._param_str(
            "frame_info_out_topic", "/seg/frame_info_selected_overlay"
        )

        self._min_pixels = max(0, self._param_int("min_pixels", 50))
        self._darken_factor = float(self._param_float("darken_factor", 0.3))
        self._draw_bbox = self._param_bool("draw_bbox", True)

        self._sync_queue_size = max(1, self._param_int("sync_queue_size", 50))

        self._profile = self._param_bool("profile", False)
        self._debug_stamps_once = self._param_bool("debug_stamps_once", True)
        self._did_debug_stamps = False

        self.get_logger().info(
            "StrawberrySelectedOverlayNode starting:\n"
            f"  image_topic          = {image_topic}\n"
            f"  label_topic          = {label_topic}\n"
            f"  frame_info_topic     = {frame_info_topic}\n"
            f"  output_topic         = {output_topic}\n"
            f"  publish_frame_info   = {self._publish_frame_info}\n"
            f"  frame_info_out_topic = {self._frame_info_out_topic}\n"
            f"  min_pixels           = {self._min_pixels}\n"
            f"  darken_factor        = {self._darken_factor:.3f}\n"
            f"  draw_bbox            = {self._draw_bbox}\n"
            f"  sync_queue_size      = {self._sync_queue_size}\n"
            f"  debug_stamps_once    = {self._debug_stamps_once}\n"
            f"  profile              = {self._profile}"
        )

        self._bridge = CvBridge()

        # ---------------- Subscribers (exact sync) ----------------
        self._sub_img = message_filters.Subscriber(
            self, Image, image_topic, qos_profile=qos_profile_sensor_data
        )
        self._sub_label = message_filters.Subscriber(
            self, Image, label_topic, qos_profile=qos_profile_sensor_data
        )
        self._sub_frame_info = message_filters.Subscriber(
            self, FrameInfo, frame_info_topic, qos_profile=qos_profile_sensor_data
        )

        self._ts = message_filters.TimeSynchronizer(
            [self._sub_img, self._sub_label, self._sub_frame_info],
            queue_size=self._sync_queue_size,
        )
        self._ts.registerCallback(self._sync_cb)

        # ---------------- Publishers ----------------
        self._pub_overlay = self.create_publisher(Image, output_topic, qos_profile_sensor_data)
        self._pub_frame_info = (
            self.create_publisher(FrameInfo, self._frame_info_out_topic, qos_profile_sensor_data)
            if self._publish_frame_info
            else None
        )

    # ------------------------------------------------------------------ #
    # Param helpers
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
    # FrameInfo helpers (copy token fields if present)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _token_stamp(fi: FrameInfo):
        return getattr(fi, "source_stamp", fi.header.stamp)

    @staticmethod
    def _copy_frame_info(src: FrameInfo, token_stamp) -> FrameInfo:
        out = FrameInfo()
        out.header = Header(stamp=token_stamp, frame_id=src.header.frame_id)
        out.frame_index = int(src.frame_index)
        out.plant_id = int(src.plant_id)
        out.view_id = int(src.view_id)
        out.rgb_path = str(src.rgb_path)
        out.depth_path = str(src.depth_path)
        out.camera_pose_world = src.camera_pose_world
        out.world_frame_id = str(src.world_frame_id)

        # Optional new fields
        if hasattr(src, "source_stamp"):
            out.source_stamp = src.source_stamp
        if hasattr(src, "frame_uid"):
            out.frame_uid = src.frame_uid
        if hasattr(src, "pipeline_version"):
            out.pipeline_version = src.pipeline_version

        return out

    # ------------------------------------------------------------------ #
    # Callback
    # ------------------------------------------------------------------ #

    def _sync_cb(self, img_msg: Image, label_msg: Image, frame_info: FrameInfo) -> None:
        t0 = time.time()

        token_stamp = self._token_stamp(frame_info)
        uid = getattr(frame_info, "frame_uid", "")

        # Hard stamp guards (deterministic pipeline)
        # (Wenn hier Drops passieren, stimmt die Tokenisierung/Synchronisation upstream nicht.)
        if (
            img_msg.header.stamp != label_msg.header.stamp
            or img_msg.header.stamp != frame_info.header.stamp
        ):
            self.get_logger().warning(
                "DROP stamp_mismatch "
                f"uid={uid} idx={int(frame_info.frame_index)} "
                f"img={img_msg.header.stamp.sec}.{img_msg.header.stamp.nanosec:09d} "
                f"label={label_msg.header.stamp.sec}.{label_msg.header.stamp.nanosec:09d} "
                f"fi={frame_info.header.stamp.sec}.{frame_info.header.stamp.nanosec:09d}"
            )
            return

        if self._debug_stamps_once and not self._did_debug_stamps:
            self._did_debug_stamps = True
            self.get_logger().info(
                "Stamps (sec.nanosec): "
                f"img={img_msg.header.stamp.sec}.{img_msg.header.stamp.nanosec:09d} "
                f"label={label_msg.header.stamp.sec}.{label_msg.header.stamp.nanosec:09d} "
                f"fi={frame_info.header.stamp.sec}.{frame_info.header.stamp.nanosec:09d} "
                f"token={token_stamp.sec}.{token_stamp.nanosec:09d} "
                f"uid={uid}"
            )

        # Dynamic param: selected id may change at runtime
        selected_id = int(self._param_int("selected_instance_id", 1))

        img_rgb = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        label = self._bridge.imgmsg_to_cv2(label_msg, desired_encoding="mono16")

        if img_rgb is None or label is None:
            self.get_logger().warning("cv_bridge returned None for image or label.")
            return

        if img_rgb.shape[:2] != label.shape[:2]:
            self.get_logger().warning(
                f"Shape mismatch image={img_rgb.shape} label={label.shape} -> check alignment!"
            )
            return

        mask = label == np.uint16(selected_id)
        n_pix = int(mask.sum())

        # Default: passthrough
        overlay = img_rgb

        if n_pix >= self._min_pixels:
            df = float(np.clip(self._darken_factor, 0.0, 1.0))
            overlay = (img_rgb.astype(np.float32) * df).astype(np.uint8)
            overlay[mask] = img_rgb[mask]

            if self._draw_bbox:
                ys, xs = np.where(mask)
                if ys.size > 0 and xs.size > 0:
                    y_min, y_max = int(ys.min()), int(ys.max())
                    x_min, x_max = int(xs.min()), int(xs.max())

                    # OpenCV arbeitet hier auf RGB-Array, wir nutzen bewusst (255,0,0) als "rot in RGB"
                    cv2.rectangle(
                        overlay,
                        (x_min, y_min),
                        (x_max, y_max),
                        color=(255, 0, 0),
                        thickness=2,
                    )
                    cv2.putText(
                        overlay,
                        f"id={selected_id} pix={n_pix}",
                        (x_min, max(y_min - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

        out_msg = self._bridge.cv2_to_imgmsg(overlay, encoding="rgb8")
        out_msg.header = Header(stamp=token_stamp, frame_id=img_msg.header.frame_id)
        self._pub_overlay.publish(out_msg)

        if self._pub_frame_info is not None:
            self._pub_frame_info.publish(self._copy_frame_info(frame_info, token_stamp))

        if self._profile:
            dt_ms = (time.time() - t0) * 1000.0
            self.get_logger().info(f"SelectedOverlay callback: {dt_ms:.2f} ms")


def main() -> None:
    rclpy.init()
    node = StrawberrySelectedOverlayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
