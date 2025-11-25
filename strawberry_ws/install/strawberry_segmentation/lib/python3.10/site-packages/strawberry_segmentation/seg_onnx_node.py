#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 node: Ultralytics YOLOv8-Seg (.pt) – identisches Verhalten wie `yolo predict`.

Subscribes:
  /camera/color/image_raw  (sensor_msgs/Image, rgb8)  [param: topic_in]

Publishes:
  /seg/label_image   (sensor_msgs/Image, mono16)  -> Instanz-IDs (0=BG, 1..N)
  /seg/overlay       (sensor_msgs/Image, rgb8)    -> Overlay aus Ultralytics .plot()

Parameter (wichtigste):
  model_path: Pfad zu best.pt (leer -> <pkg share>/models/best.pt)
  device    : 'auto' | 'cpu' | 'cuda:0' (default: auto)
  imgsz     : 640 (Training-Größe)
  conf_thres, iou_thres, max_det
  min_mask_area_px: Minimumfläche je Instanz (Filter)
  publish_overlay, profile
"""

from __future__ import annotations
import os
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType as PT
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from ament_index_python.packages import get_package_share_directory

# Ultralytics / Torch
from ultralytics import YOLO
import torch


class YoloSegUltralyticsNode(Node):
    def __init__(self):
        super().__init__('strawberry_seg_ultra')

        # ---------------- Parameter ----------------
        self.declare_parameter('model_path', '')
        self.declare_parameter('topic_in', '/camera/color/image_raw')
        self.declare_parameter('publish_overlay', True)

        self.declare_parameter('device', 'auto')       # auto|cpu|cuda:0
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('conf_thres', 0.65)
        self.declare_parameter('iou_thres', 0.50)
        self.declare_parameter('max_det', 100)
        self.declare_parameter('min_mask_area_px', 1500)
        self.declare_parameter('profile', False)

        # liest STR-ARRAY nicht zwingend, aber kompatibel bleiben
        self.declare_parameter(
            'classes', [],
            ParameterDescriptor(
                type=PT.PARAMETER_INTEGER_ARRAY,
                description='Optional: nur diese Klassen-IDs detektieren'
            )
        )

        # ---------------- Parameter lesen ----------------
        topic_in = self.get_parameter('topic_in').value
        publish_overlay = bool(self.get_parameter('publish_overlay').value)

        model_path = self.get_parameter('model_path').value
        if not model_path:
            try:
                share_dir = get_package_share_directory('strawberry_segmentation')
                default_model = os.path.join(share_dir, 'models', 'best.pt')
                if os.path.exists(default_model):
                    model_path = default_model
                    self.get_logger().info(f"model_path leer -> nutze Paketmodell: {model_path}")
                else:
                    self.get_logger().error("Kein model_path gesetzt und kein Paketmodell gefunden (share/.../models/best.pt).")
            except Exception as e:
                self.get_logger().error(f"Konnte Paket-Share-Verzeichnis nicht ermitteln: {e}")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO .pt-Modell nicht gefunden: '{model_path}'")

        # device/half
        device_param = str(self.get_parameter('device').value).lower()
        if device_param == 'auto':
            self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device_param
        self._half = self._device.startswith('cuda')

        # weitere
        self._imgsz   = int(self.get_parameter('imgsz').value)
        self._conf    = float(self.get_parameter('conf_thres').value)
        self._iou     = float(self.get_parameter('iou_thres').value)
        self._max_det = int(self.get_parameter('max_det').value)
        self._min_area = int(self.get_parameter('min_mask_area_px').value)
        self._profile = bool(self.get_parameter('profile').value)
        self._classes = list(self.get_parameter('classes').value or [])

        # ---------------- ROS I/O ----------------
        self.bridge = CvBridge()
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=5)

        self.sub_rgb = self.create_subscription(Image, topic_in, self.on_image, qos)
        self.pub_label = self.create_publisher(Image, '/seg/label_image', 10)
        self.pub_overlay = self.create_publisher(Image, '/seg/overlay', 10) if publish_overlay else None

        # ---------------- YOLO laden ----------------
        t0 = time.time()
        self.model = YOLO(model_path)  # seg-Variante (yolov8s-seg.pt / best.pt)
        # kein explizites .to(); Ultralytics nutzt 'device' in predict()
        self.get_logger().info(
            f"YOLO geladen ({time.time()-t0:.2f}s) | device={self._device} half={self._half} | "
            f"imgsz={self._imgsz} conf={self._conf} iou={self._iou}"
        )

    # ---------------- Callback ----------------
    def on_image(self, msg: Image):
        t_all0 = time.time()
        # ROS -> RGB -> BGR (Ultralytics erwartet BGR wie OpenCV)
        img_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h0, w0 = img_rgb.shape[:2]

        # Inferenz (ein Bild; identisch zu `yolo predict`)
        t1 = time.time()
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
            verbose=False
        )
        t_inf = time.time() - t1
        if not results:
            self._publish(msg, img_rgb, np.zeros((h0, w0), dtype=np.uint16))
            return

        res = results[0]

        # Overlay direkt aus Ultralytics (BGR) -> RGB für ROS
        overlay_bgr = res.plot()  # zeichnet Boxen + Masken
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        if overlay_rgb.shape[:2] != (h0, w0):
            overlay_rgb = cv2.resize(overlay_rgb, (w0, h0), interpolation=cv2.INTER_LINEAR)

        # Label-Map aus Masks
        label = np.zeros((h0, w0), dtype=np.uint16)
        if res.masks is not None and res.masks.data is not None and len(res.masks) > 0:
            # (n, H, W) float/bool → np.uint8
            masks_np = res.masks.data.cpu().numpy().astype(np.uint8)  # 0/1
            # auf Eingabebildgröße sichern
            Hm, Wm = masks_np.shape[1:]
            if (Hm, Wm) != (h0, w0):
                # resize jede Maske
                resized = np.zeros((masks_np.shape[0], h0, w0), dtype=np.uint8)
                for i in range(masks_np.shape[0]):
                    resized[i] = cv2.resize(masks_np[i] * 255, (w0, h0), interpolation=cv2.INTER_NEAREST) // 255
                masks_np = resized

            # nach Score sortieren (stabile Z-Reihenfolge)
            if res.boxes is not None and res.boxes.conf is not None:
                order = torch.argsort(res.boxes.conf).cpu().numpy()[::-1]
            else:
                order = np.arange(masks_np.shape[0])

            k_out = 0
            for k in order:
                m = masks_np[k]
                if int(m.sum()) < self._min_area:
                    continue
                k_out += 1
                newpix = (m == 1) & (label == 0)
                label[newpix] = k_out  # 1..N

        self._publish(msg, overlay_rgb if self.pub_overlay else img_rgb, label)

        if self._profile:
            t_all = (time.time() - t_all0) * 1000.0
            n_inst = int(label.max())
            self.get_logger().info(f"inf {t_inf*1000:.1f} ms | inst={n_inst}")

    # ---------------- Publish Helper ----------------
    def _publish(self, src_msg: Image, overlay_rgb: np.ndarray, label_u16: np.ndarray):
        # overlay (optional)
        if self.pub_overlay is not None and self.pub_overlay.get_subscription_count() > 0:
            ov_msg = self.bridge.cv2_to_imgmsg(overlay_rgb, encoding='rgb8')
            ov_msg.header = Header(stamp=src_msg.header.stamp, frame_id=src_msg.header.frame_id)
            self.pub_overlay.publish(ov_msg)

        # mono16 Label
        H, W = label_u16.shape
        lbl = Image()
        lbl.header = Header(stamp=src_msg.header.stamp, frame_id=src_msg.header.frame_id)
        lbl.height = H
        lbl.width = W
        lbl.encoding = 'mono16'
        lbl.is_bigendian = 0
        lbl.step = W * 2
        lbl.data = label_u16.tobytes()
        self.pub_label.publish(lbl)


def main():
    rclpy.init()
    node = YoloSegUltralyticsNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
