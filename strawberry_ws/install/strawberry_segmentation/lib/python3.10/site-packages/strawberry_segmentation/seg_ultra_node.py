#!/usr/bin/env python3
# strawberry_segmentation/seg_ultra_node.py

"""
ROS 2 node: Ultralytics YOLOv8 segmentation (.pt).

Subscribes:
  - /camera/color/image_raw (sensor_msgs/Image, rgb8) [param: topic_in]

Publishes:
  - /seg/label_image       (sensor_msgs/Image, mono16): instance IDs (0 = BG, 1..N)
  - /seg/label_image_vis   (sensor_msgs/Image, mono8):  debug mask (0/255)
  - /seg/overlay           (sensor_msgs/Image, rgb8):  YOLO overlay (boxes + masks)

Key parameters:
  - model_path: path to best.pt (empty -> <pkg share>/models/best.pt)
  - device: 'auto' | 'cpu' | 'cuda:0'
  - imgsz, conf_thres, iou_thres, max_det
  - min_mask_area_px: minimum area per instance (filter)
  - publish_overlay, profile
"""

from __future__ import annotations

import os
import time

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
from ultralytics import YOLO


class YoloSegUltralyticsNode(Node):
    """ROS 2 node running YOLOv8 segmentation on RGB images."""

    def __init__(self) -> None:
        super().__init__("strawberry_seg_ultra")

        # Parameters
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

        # Optional class filter (not used now, but kept as parameter)
        self.declare_parameter(
            "classes",
            [],
            ParameterDescriptor(
                type=PT.PARAMETER_INTEGER_ARRAY,
                description="Optional: restrict detection to these class IDs.",
            ),
        )

        topic_in = self.get_parameter("topic_in").value
        publish_overlay = bool(self.get_parameter("publish_overlay").value)

        model_path = self._resolve_model_path()
        self._device, self._half = self._resolve_device()

        self._imgsz = int(self.get_parameter("imgsz").value)
        self._conf = float(self.get_parameter("conf_thres").value)
        self._iou = float(self.get_parameter("iou_thres").value)
        self._max_det = int(self.get_parameter("max_det").value)
        self._min_area = int(self.get_parameter("min_mask_area_px").value)
        self._profile = bool(self.get_parameter("profile").value)
        self._classes = list(self.get_parameter("classes").value or [])

        # ROS I/O
        self.bridge = CvBridge()

        self.sub_rgb = self.create_subscription(
            Image, topic_in, self.on_image, 10
        )

        self.pub_label = self.create_publisher(Image, "/seg/label_image", 10)
        self.pub_label_vis = self.create_publisher(
            Image, "/seg/label_image_vis", 10
        )
        self.pub_overlay = (
            self.create_publisher(Image, "/seg/overlay", 10)
            if publish_overlay
            else None
        )

        # Load YOLO model
        t0 = time.time()
        self.model = YOLO(model_path)
        self.get_logger().info(
            "YOLO loaded in "
            f"{time.time() - t0:.2f}s | device={self._device} "
            f"half={self._half} | imgsz={self._imgsz} conf={self._conf} "
            f"iou={self._iou}"
        )

    # ---------------- Helpers ----------------
    def _resolve_model_path(self) -> str:
        """Resolve model_path parameter or fall back to the package share."""
        model_path = self.get_parameter("model_path").value
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"YOLO .pt model not found at '{model_path}'"
                )
            return model_path

        try:
            share_dir = get_package_share_directory("strawberry_segmentation")
            default_model = os.path.join(share_dir, "models", "best.pt")
            if os.path.exists(default_model):
                self.get_logger().info(
                    f"model_path empty -> using package model: {default_model}"
                )
                return default_model
            self.get_logger().error(
                "No model_path set and no package model found at "
                "share/.../models/best.pt."
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f"Could not get package share directory: {exc}"
            )

        raise FileNotFoundError("YOLO .pt model could not be resolved.")

    def _resolve_device(self) -> tuple[str, bool]:
        """Select device and half-precision flag."""
        device_param = str(self.get_parameter("device").value).lower()
        if device_param == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device = device_param

        half = device.startswith("cuda")
        return device, half

    # ---------------- Image callback ----------------
    def on_image(self, msg: Image) -> None:
        t_all0 = time.time()

        # ROS -> RGB -> BGR (Ultralytics expects BGR like OpenCV)
        img_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h0, w0 = img_rgb.shape[:2]

        # Inference (single image)
        t_inf0 = time.time()
        results = self.model.predict(
            source=img_bgr,
            imgsz=self._imgsz,
            conf=self._conf,
            iou=self._iou,
            max_det=self._max_det,
            device=self._device,
            half=self._half,
            agnostic_nms=False,
            retina_masks=True,
            verbose=False,
        )
        t_inf = time.time() - t_inf0

        if not results:
            self._publish(msg, img_rgb, np.zeros((h0, w0), dtype=np.uint16))
            return

        res = results[0]

        # Overlay directly from Ultralytics (BGR -> RGB)
        overlay_bgr = res.plot()
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        if overlay_rgb.shape[:2] != (h0, w0):
            overlay_rgb = cv2.resize(
                overlay_rgb,
                (w0, h0),
                interpolation=cv2.INTER_LINEAR,
            )

        # Instance label map
        label = np.zeros((h0, w0), dtype=np.uint16)

        if res.masks is not None and res.masks.data is not None and len(res.masks):
            masks_np = res.masks.data.cpu().numpy().astype(np.uint8)  # (n,H,W)
            hm, wm = masks_np.shape[1:]

            # Resize masks to input size if needed
            if (hm, wm) != (h0, w0):
                resized = np.zeros(
                    (masks_np.shape[0], h0, w0), dtype=np.uint8
                )
                for i in range(masks_np.shape[0]):
                    resized[i] = cv2.resize(
                        masks_np[i] * 255,
                        (w0, h0),
                        interpolation=cv2.INTER_NEAREST,
                    ) // 255
                masks_np = resized

            # Sort by confidence (stable Z-order)
            if res.boxes is not None and res.boxes.conf is not None:
                order = (
                    torch.argsort(res.boxes.conf).cpu().numpy()[::-1]
                )
            else:
                order = np.arange(masks_np.shape[0])

            k_out = 0
            for k in order:
                mask = masks_np[k]
                if int(mask.sum()) < self._min_area:
                    continue
                k_out += 1
                newpix = (mask == 1) & (label == 0)
                label[newpix] = k_out  # 1..N

        self._publish(
            msg,
            overlay_rgb if self.pub_overlay else img_rgb,
            label,
        )

        if self._profile:
            t_all = (time.time() - t_all0) * 1000.0
            n_inst = int(label.max())
            self.get_logger().info(
                f"inference {t_inf * 1000:.1f} ms | total {t_all:.1f} ms "
                f"| instances={n_inst}"
            )

    # ---------------- Publish helper ----------------
    def _publish(
        self,
        src_msg: Image,
        overlay_rgb: np.ndarray,
        label_u16: np.ndarray,
    ) -> None:
        # Overlay (optional)
        if (
            self.pub_overlay is not None
            and self.pub_overlay.get_subscription_count() > 0
        ):
            ov_msg = self.bridge.cv2_to_imgmsg(overlay_rgb, encoding="rgb8")
            ov_msg.header = Header(
                stamp=src_msg.header.stamp,
                frame_id=src_msg.header.frame_id,
            )
            self.pub_overlay.publish(ov_msg)

        # mono16 label image
        height, width = label_u16.shape
        lbl = Image()
        lbl.header = Header(
            stamp=src_msg.header.stamp,
            frame_id=src_msg.header.frame_id,
        )
        lbl.height = height
        lbl.width = width
        lbl.encoding = "mono16"
        lbl.is_bigendian = 0
        lbl.step = width * 2
        lbl.data = label_u16.tobytes()
        self.pub_label.publish(lbl)

        # mono8 visualization (0/255) for rqt_image_view
        if self.pub_label_vis.get_subscription_count() > 0:
            if np.any(label_u16 > 0):
                label_vis = (label_u16 > 0).astype(np.uint8) * 255
            else:
                label_vis = np.zeros_like(label_u16, dtype=np.uint8)
            lbl_vis_msg = self.bridge.cv2_to_imgmsg(
                label_vis, encoding="mono8"
            )
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
