#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 2 node: Ultralytics YOLOv8 segmentation (.pt).

Subscribes (synchronized):
  - topic_in         (sensor_msgs/Image, rgb8)
  - frame_info_topic (strawberry_msgs/FrameInfo)

Publishes:
  - /seg/label_image       (sensor_msgs/Image, mono16): instance IDs (0 = BG, 1..N)
  - /seg/label_image_vis   (sensor_msgs/Image, mono8):  debug mask (0/255)
  - /seg/overlay           (sensor_msgs/Image, rgb8):   YOLO overlay (optional)
  - frame_info_out_topic   (strawberry_msgs/FrameInfo): passthrough for downstream sync
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import message_filters
import numpy as np
import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType as PT
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from strawberry_msgs.msg import FrameInfo

try:
    from ultralytics.yolo.engine.model import YOLO  # type: ignore
except Exception:  # noqa: BLE001
    from ultralytics import YOLO  # type: ignore


class YoloSegUltralyticsNode(Node):
    """ROS 2 node running YOLOv8 segmentation on RGB images."""

    def __init__(self) -> None:
        super().__init__("strawberry_seg_ultra")

        # ---------------- Parameters ----------------
        self.declare_parameter("model_path", "")
        self.declare_parameter("topic_in", "/camera/color/image_raw")
        self.declare_parameter("publish_overlay", True)

        self.declare_parameter("frame_info_topic", "/camera/frame_info")
        self.declare_parameter("publish_frame_info", True)
        self.declare_parameter("frame_info_out_topic", "/seg/frame_info")

        self.declare_parameter("sync_queue_size", 200)
        self.declare_parameter("sync_slop", 0.2)

        self.declare_parameter("device", "auto")  # auto|cpu|cuda:0
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf_thres", 0.65)
        self.declare_parameter("iou_thres", 0.50)
        self.declare_parameter("max_det", 100)
        self.declare_parameter("min_mask_area_px", 1500)
        self.declare_parameter("profile", False)

        self.declare_parameter(
            "classes",
            [],
            ParameterDescriptor(
                type=PT.PARAMETER_INTEGER_ARRAY,
                description="Optional: restrict detection to these class IDs.",
            ),
        )

        # ---------------- Read parameters ----------------
        topic_in = self._param_str("topic_in", "/camera/color/image_raw")
        self._publish_overlay = self._param_bool("publish_overlay", True)

        frame_info_topic = self._param_str("frame_info_topic", "/camera/frame_info")
        self._publish_frame_info = self._param_bool("publish_frame_info", True)
        frame_info_out_topic = self._param_str("frame_info_out_topic", "/seg/frame_info")

        self._sync_queue_size = max(1, self._param_int("sync_queue_size", 200))
        self._sync_slop = self._param_float("sync_slop", 0.2)
        if self._sync_slop <= 0.0:
            self._sync_slop = 0.05

        self._imgsz = self._param_int("imgsz", 640)
        self._conf = self._param_float("conf_thres", 0.65)
        self._iou = self._param_float("iou_thres", 0.50)
        self._max_det = self._param_int("max_det", 100)
        self._min_area = self._param_int("min_mask_area_px", 1500)
        self._profile = self._param_bool("profile", False)
        self._classes = self._param_int_list("classes")

        model_path = self._resolve_model_path()
        self._device, self._half = self._resolve_device()

        self.get_logger().info(
            "seg_ultra params:\n"
            f"  topic_in             = {topic_in}\n"
            f"  frame_info_topic     = {frame_info_topic}\n"
            f"  publish_frame_info   = {self._publish_frame_info}\n"
            f"  frame_info_out_topic = {frame_info_out_topic}\n"
            f"  sync_queue_size      = {self._sync_queue_size}\n"
            f"  sync_slop            = {self._sync_slop}\n"
            f"  publish_overlay      = {self._publish_overlay}\n"
            f"  device               = {self._device}\n"
            f"  half                 = {self._half}\n"
            f"  imgsz                = {self._imgsz}\n"
            f"  conf                 = {self._conf:.3f}\n"
            f"  iou                  = {self._iou:.3f}\n"
            f"  max_det              = {self._max_det}\n"
            f"  min_area             = {self._min_area}\n"
            f"  classes              = {self._classes}\n"
            f"  profile              = {self._profile}"
        )

        # ---------------- ROS I/O ----------------
        self._bridge = CvBridge()

        self._pub_label = self.create_publisher(Image, "/seg/label_image", 10)
        self._pub_label_vis = self.create_publisher(Image, "/seg/label_image_vis", 10)
        self._pub_overlay = (
            self.create_publisher(Image, "/seg/overlay", 10)
            if self._publish_overlay
            else None
        )
        self._pub_frame_info = (
            self.create_publisher(FrameInfo, frame_info_out_topic, 10)
            if self._publish_frame_info
            else None
        )

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._sub_rgb = message_filters.Subscriber(self, Image, topic_in, qos_profile=qos)
        self._sub_frame_info = message_filters.Subscriber(
            self, FrameInfo, frame_info_topic, qos_profile=qos
        )

        self._ts = message_filters.ApproximateTimeSynchronizer(
            [self._sub_rgb, self._sub_frame_info],
            queue_size=self._sync_queue_size,
            slop=self._sync_slop,
        )
        self._ts.registerCallback(self._sync_cb)

        # ---------------- Load YOLO model ----------------
        t0 = time.time()
        self._model = YOLO(model_path)
        self.get_logger().info(
            f"YOLO loaded in {time.time() - t0:.2f}s | model={model_path} "
            f"| device={self._device} half={self._half}"
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

    def _param_int_list(self, name: str) -> List[int]:
        val: Any = self.get_parameter(name).value
        if not isinstance(val, (list, tuple)):
            return []
        out: List[int] = []
        for x in val:
            try:
                out.append(int(x))
            except Exception:  # noqa: BLE001
                continue
        return out

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _resolve_model_path(self) -> str:
        model_path = self._param_str("model_path", "").strip()
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"YOLO .pt model not found at '{model_path}'")
            return model_path

        share_dir = get_package_share_directory("strawberry_segmentation")
        default_model = os.path.join(share_dir, "models", "best.pt")
        if os.path.exists(default_model):
            self.get_logger().info(f"model_path empty -> using package model: {default_model}")
            return default_model

        raise FileNotFoundError(
            "No model_path set and no package model found at share/.../models/best.pt."
        )

    def _resolve_device(self) -> Tuple[str, bool]:
        device_param = self._param_str("device", "auto").lower()
        if device_param == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device = device_param
        return device, device.startswith("cuda")

    def _masks_to_numpy_u8(self, res: Any, h0: int, w0: int) -> Optional[np.ndarray]:
        if res is None or getattr(res, "masks", None) is None:
            return None
        if getattr(res.masks, "data", None) is None:
            return None

        data = res.masks.data
        if isinstance(data, np.ndarray):
            t = torch.from_numpy(data)
        elif isinstance(data, torch.Tensor):
            t = data
        else:
            return None

        masks = t.detach().cpu().numpy()
        if masks.ndim != 3:
            return None

        masks = masks.astype(np.uint8)
        hm, wm = int(masks.shape[1]), int(masks.shape[2])

        if (hm, wm) != (h0, w0):
            resized = np.zeros((masks.shape[0], h0, w0), dtype=np.uint8)
            for i in range(masks.shape[0]):
                resized[i] = (
                    cv2.resize(
                        masks[i] * 255,
                        (w0, h0),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    // 255
                ).astype(np.uint8)
            masks = resized

        return masks

    def _predict(self, img_bgr: np.ndarray) -> Any:
        kwargs: Dict[str, Any] = {
            "source": img_bgr,
            "imgsz": self._imgsz,
            "conf": self._conf,
            "iou": self._iou,
            "max_det": self._max_det,
            "device": self._device,
            "verbose": False,
            "half": self._half,
            "agnostic_nms": False,
            "retina_masks": True,
        }
        if self._classes:
            kwargs["classes"] = self._classes

        try:
            return self._model.predict(**kwargs)
        except TypeError:
            for k in ("classes", "retina_masks", "agnostic_nms", "half"):
                kwargs.pop(k, None)
            return self._model.predict(**kwargs)

    @staticmethod
    def _copy_frame_info(src: FrameInfo, stamp) -> FrameInfo:
        out = FrameInfo()
        out.header = Header(stamp=stamp, frame_id=src.header.frame_id)
        out.frame_index = int(src.frame_index)
        out.plant_id = int(src.plant_id)
        out.view_id = int(src.view_id)
        out.rgb_path = str(src.rgb_path)
        out.depth_path = str(src.depth_path)
        out.camera_pose_world = src.camera_pose_world
        out.world_frame_id = str(src.world_frame_id)
        return out

    # ------------------------------------------------------------------ #
    # Sync callback
    # ------------------------------------------------------------------ #

    def _sync_cb(self, img_msg: Image, frame_info: FrameInfo) -> None:
        t_all0 = time.time()

        img_rgb = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h0, w0 = img_rgb.shape[:2]

        t_inf0 = time.time()
        results = self._predict(img_bgr)
        t_inf_s = time.time() - t_inf0

        label = np.zeros((h0, w0), dtype=np.uint16)

        if results:
            res = results[0]
            masks_np = self._masks_to_numpy_u8(res, h0, w0)

            if masks_np is not None and masks_np.shape[0] > 0:
                order = np.arange(masks_np.shape[0])

                boxes = getattr(res, "boxes", None)
                conf = getattr(boxes, "conf", None) if boxes is not None else None
                if isinstance(conf, torch.Tensor) and conf.numel() == masks_np.shape[0]:
                    order = torch.argsort(conf).detach().cpu().numpy()[::-1]

                k_out = 0
                for k in order:
                    mask = masks_np[int(k)]
                    if int(mask.sum()) < self._min_area:
                        continue
                    k_out += 1
                    newpix = (mask == 1) & (label == 0)
                    label[newpix] = k_out

        # Overlay nur berechnen, wenn wirklich abonniert
        overlay_rgb: Optional[np.ndarray] = None
        if results and self._pub_overlay is not None and self._pub_overlay.get_subscription_count() > 0:
            overlay_bgr = results[0].plot()
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            if overlay_rgb.shape[:2] != (h0, w0):
                overlay_rgb = cv2.resize(overlay_rgb, (w0, h0), interpolation=cv2.INTER_LINEAR)

        self._publish(img_msg, frame_info, label, overlay_rgb)

        if self._profile:
            t_all_ms = (time.time() - t_all0) * 1000.0
            n_inst = int(label.max())
            self.get_logger().info(
                f"Frame {int(frame_info.frame_index)} | plant {int(frame_info.plant_id)} "
                f"| view {int(frame_info.view_id)} | inference {t_inf_s * 1000.0:.1f} ms "
                f"| total {t_all_ms:.1f} ms | instances={n_inst}"
            )

    def _publish(
        self,
        src_msg: Image,
        frame_info: FrameInfo,
        label_u16: np.ndarray,
        overlay_rgb: Optional[np.ndarray],
    ) -> None:
        stamp = src_msg.header.stamp

        if self._pub_frame_info is not None:
            self._pub_frame_info.publish(self._copy_frame_info(frame_info, stamp))

        if overlay_rgb is not None and self._pub_overlay is not None:
            ov_msg = self._bridge.cv2_to_imgmsg(overlay_rgb, encoding="rgb8")
            ov_msg.header = Header(stamp=stamp, frame_id=src_msg.header.frame_id)
            self._pub_overlay.publish(ov_msg)

        height, width = label_u16.shape
        lbl = Image()
        lbl.header = Header(stamp=stamp, frame_id=src_msg.header.frame_id)
        lbl.height = int(height)
        lbl.width = int(width)
        lbl.encoding = "mono16"
        lbl.is_bigendian = 0
        lbl.step = int(width) * 2
        lbl.data = label_u16.tobytes()
        self._pub_label.publish(lbl)

        if self._pub_label_vis.get_subscription_count() > 0:
            label_vis = (label_u16 > 0).astype(np.uint8) * 255
            lbl_vis_msg = self._bridge.cv2_to_imgmsg(label_vis, encoding="mono8")
            lbl_vis_msg.header = lbl.header
            self._pub_label_vis.publish(lbl_vis_msg)


def main() -> None:
    rclpy.init()
    node = YoloSegUltralyticsNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
