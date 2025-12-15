#!/usr/bin/env python3
# strawberry_segmentation/seg_ultra_node.py
# -*- coding: utf-8 -*-

"""
ROS 2 node: Ultralytics YOLOv8 segmentation (.pt).

Subscribes:
  - /camera/color/image_raw (sensor_msgs/Image, rgb8) [param: topic_in]

Publishes:
  - /seg/label_image       (sensor_msgs/Image, mono16): instance IDs (0 = BG, 1..N)
  - /seg/label_image_vis   (sensor_msgs/Image, mono8):  debug mask (0/255)
  - /seg/overlay           (sensor_msgs/Image, rgb8):   YOLO overlay (boxes + masks)

Important:
  - rclpy logger methods take ONLY ONE positional argument (the message string).
    So we always use f-strings (no printf-style formatting args).
  - Ultralytics API varies between versions. predict() is called with a full set of
    kwargs first, then falls back to a reduced set if TypeError occurs.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType as PT
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

try:
    # Pylance-friendly for many Ultralytics versions
    from ultralytics.yolo.engine.model import YOLO  # type: ignore
except Exception:  # noqa: BLE001
    # Runtime fallback (often works even if typing stubs complain)
    from ultralytics import YOLO  # type: ignore


class YoloSegUltralyticsNode(Node):
    """ROS 2 node running YOLOv8 segmentation on RGB images."""

    def __init__(self) -> None:
        super().__init__("strawberry_seg_ultra")

        # ---------------- Parameters ----------------
        self.declare_parameter("model_path", "")
        self.declare_parameter("topic_in", "/camera/color/image_raw")
        self.declare_parameter("publish_overlay", True)

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

        # Read params (Pylance-safe)
        topic_in = self._param_str("topic_in", "/camera/color/image_raw")
        publish_overlay = self._param_bool("publish_overlay", True)

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
            "seg_ultra params: "
            f"topic_in={topic_in} "
            f"publish_overlay={publish_overlay} "
            f"device={self._device} "
            f"half={self._half} "
            f"imgsz={self._imgsz} "
            f"conf={self._conf:.3f} "
            f"iou={self._iou:.3f} "
            f"max_det={self._max_det} "
            f"min_area={self._min_area} "
            f"classes={self._classes}"
        )

        # ---------------- ROS I/O ----------------
        self.bridge = CvBridge()

        self.sub_rgb = self.create_subscription(Image, topic_in, self.on_image, 10)
        self.pub_label = self.create_publisher(Image, "/seg/label_image", 10)
        self.pub_label_vis = self.create_publisher(Image, "/seg/label_image_vis", 10)
        self.pub_overlay = (
            self.create_publisher(Image, "/seg/overlay", 10) if publish_overlay else None
        )

        # ---------------- Load YOLO model ----------------
        t0 = time.time()
        self.model = YOLO(model_path)
        dt = time.time() - t0
        self.get_logger().info(
            f"YOLO loaded in {dt:.2f}s | model={model_path} | device={self._device} half={self._half}"
        )

    # ------------------------------------------------------------------ #
    # Param helpers (Pylance-friendly)
    # ------------------------------------------------------------------ #

    def _param_str(self, name: str, default: str) -> str:
        val: Any = self.get_parameter(name).value
        return default if val is None else str(val)

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
        if val is None:
            return []
        if isinstance(val, (list, tuple)):
            out: List[int] = []
            for x in val:
                try:
                    out.append(int(x))
                except Exception:  # noqa: BLE001
                    continue
            return out
        return []

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _resolve_model_path(self) -> str:
        """Resolve model_path parameter or fall back to the package share."""
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
        """Select device and half-precision flag."""
        device_param = self._param_str("device", "auto").lower()
        if device_param == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device = device_param
        half = device.startswith("cuda")
        return device, half

    def _masks_to_numpy_u8(self, res: Any, h0: int, w0: int) -> Optional[np.ndarray]:
        """Return masks as uint8 numpy array (n, h0, w0) with values {0,1}."""
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
        """Call Ultralytics predict() in a version-tolerant way."""
        kwargs: Dict[str, Any] = {
            "source": img_bgr,
            "imgsz": self._imgsz,
            "conf": self._conf,
            "iou": self._iou,
            "max_det": self._max_det,
            "device": self._device,
            "verbose": False,
        }

        # Optional kwargs (not supported in every Ultralytics version)
        kwargs["half"] = self._half
        kwargs["agnostic_nms"] = False
        kwargs["retina_masks"] = True
        if self._classes:
            kwargs["classes"] = self._classes

        try:
            return self.model.predict(**kwargs)
        except TypeError:
            # Drop optional kwargs that frequently break on older versions
            for k in ("classes", "retina_masks", "agnostic_nms", "half"):
                kwargs.pop(k, None)
            return self.model.predict(**kwargs)

    # ------------------------------------------------------------------ #
    # Image callback
    # ------------------------------------------------------------------ #

    def on_image(self, msg: Image) -> None:
        t_all0 = time.time()

        img_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h0, w0 = img_rgb.shape[:2]

        t_inf0 = time.time()
        results = self._predict(img_bgr)
        t_inf = time.time() - t_inf0

        if not results:
            self._publish(msg, img_rgb, np.zeros((h0, w0), dtype=np.uint16))
            return

        res = results[0]

        overlay_bgr = res.plot()
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        if overlay_rgb.shape[:2] != (h0, w0):
            overlay_rgb = cv2.resize(overlay_rgb, (w0, h0), interpolation=cv2.INTER_LINEAR)

        label = np.zeros((h0, w0), dtype=np.uint16)

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

        self._publish(msg, overlay_rgb if self.pub_overlay else img_rgb, label)

        if self._profile:
            t_all = (time.time() - t_all0) * 1000.0
            n_inst = int(label.max())
            self.get_logger().info(
                f"inference {t_inf * 1000.0:.1f} ms | total {t_all:.1f} ms | instances={n_inst}"
            )

    # ------------------------------------------------------------------ #
    # Publish helper
    # ------------------------------------------------------------------ #

    def _publish(self, src_msg: Image, overlay_rgb: np.ndarray, label_u16: np.ndarray) -> None:
        if self.pub_overlay is not None and self.pub_overlay.get_subscription_count() > 0:
            ov_msg = self.bridge.cv2_to_imgmsg(overlay_rgb, encoding="rgb8")
            ov_msg.header = Header(
                stamp=src_msg.header.stamp,
                frame_id=src_msg.header.frame_id,
            )
            self.pub_overlay.publish(ov_msg)

        height, width = label_u16.shape
        lbl = Image()
        lbl.header = Header(
            stamp=src_msg.header.stamp,
            frame_id=src_msg.header.frame_id,
        )
        lbl.height = int(height)
        lbl.width = int(width)
        lbl.encoding = "mono16"
        lbl.is_bigendian = 0
        lbl.step = int(width) * 2
        lbl.data = label_u16.tobytes()
        self.pub_label.publish(lbl)

        if self.pub_label_vis.get_subscription_count() > 0:
            label_vis = (label_u16 > 0).astype(np.uint8) * 255
            lbl_vis_msg = self.bridge.cv2_to_imgmsg(label_vis, encoding="mono8")
            lbl_vis_msg.header = lbl.header
            self.pub_label_vis.publish(lbl_vis_msg)


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
