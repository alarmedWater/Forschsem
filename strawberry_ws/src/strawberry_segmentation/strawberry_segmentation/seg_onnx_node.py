#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 node: YOLOv8-seg (ONNXRuntime) inference-only.

Subscribes:
  - /camera/color/image_raw  (sensor_msgs/Image, rgb8)   [param: 'topic_in']

Publishes:
  - /seg/label_image         (sensor_msgs/Image, mono16)  instance-id map (0=bg, 1..N)
  - /seg/overlay             (sensor_msgs/Image, rgb8)    farbiges Overlay (optional)

Ann.: Ultralytics YOLOv8-Seg Export (.onnx)
  - Output A: (1, N, 4+nc+nm)  -> boxes(xywh), cls-scores, mask-coeffs
  - Output B: (1, nm, Mh, Mw)  -> mask prototype
Die Output-Namen/Reihenfolge können variieren – der Node erkennt sie automatisch über Shapes.

Wichtige Parameter:
  - model_path: Pfad zur .onnx  (leer => auto: <pkg share>/models/best.onnx)
  - providers:  STRING_ARRAY, z.B. ['CPUExecutionProvider'] oder
                ['CUDAExecutionProvider','CPUExecutionProvider']
                leer/auto => wählt selbst je nach Verfügbarkeit
  - imgsz, conf_thres, iou_thres, mask_thresh, max_det, num_classes, mask_dim
  - topic_in: Eingangsbild-Topic (default '/camera/color/image_raw')
  - publish_overlay: bool (default True)
  - min_mask_area_px: int (default 20)
  - profile: bool (default False) -> Laufzeiten loggen
  
  GPU call:  
  ros2 run strawberry_segmentation seg_onnx --ros-args \
  -p providers:="['CUDAExecutionProvider','CPUExecutionProvider']"

"""

import os
import time
import math
import numpy as np
import cv2
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType as PT
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from ament_index_python.packages import get_package_share_directory


# ----------------- helpers -----------------
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), stride=32):
    """Resize+pad to target shape, keeping aspect ratio (Ultralytics-like)."""
    shape = im.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # w, h
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x))


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def iou(box, boxes):
    # box: (4,), boxes: (M,4) [x1,y1,x2,y2]
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    a = (box[2]-box[0])*(box[3]-box[1])
    b = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    return inter / (a + b - inter + 1e-7)


def nms_boxes(boxes, scores, iou_th=0.5, top_k=300):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0 and len(keep) < top_k:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_th]
    return np.array(keep, dtype=np.int32)


def color_from_id(k: int):
    np.random.seed(k * 12345 + 7)
    return tuple(int(v) for v in np.random.randint(64, 255, size=3, dtype=np.uint8))


def pick_outputs_by_shape(session, prefer_names=None):
    """
    Versucht, YOLOv8-Seg-Outputs robust zu erkennen:
    - boxes/masks coeff: rank==3  (1, N, D)
    - proto:            rank==4  (1, nm, Mh, Mw)
    """
    outs = session.get_outputs()
    names = [o.name for o in outs]
    shapes = [tuple(o.shape) for o in outs]

    # 1) Falls Namen vorgegeben & vorhanden:
    if prefer_names:
        a, b = prefer_names
        if a in names and b in names:
            return a, b

    # 2) Nach Rank suchen:
    idx_rank3 = [i for i, s in enumerate(shapes) if len([d for d in s if d is not None]) == 3]
    idx_rank4 = [i for i, s in enumerate(shapes) if len([d for d in s if d is not None]) == 4]
    if idx_rank3 and idx_rank4:
        return names[idx_rank3[0]], names[idx_rank4[0]]

    # 3) Fallback: Erste zwei Outputs
    return names[0], names[1]


# ----------------- node -----------------
class YoloSegOnnxNode(Node):
    def __init__(self):
        super().__init__('strawberry_seg_onnx')

        # --------- Parameter deklarieren ---------
        # I/O
        self.declare_parameter('model_path', '')  # leer => auto (Paket-Share)
        self.declare_parameter('topic_in', '/camera/color/image_raw')
        self.declare_parameter('publish_overlay', True)

        # Inferenz-Settings
        self.declare_parameter('imgsz', 1024)
        self.declare_parameter('conf_thres', 0.25)
        self.declare_parameter('iou_thres', 0.50)
        self.declare_parameter('mask_thresh', 0.50)
        self.declare_parameter('max_det', 300)
        self.declare_parameter('num_classes', 1)   # strawberries only
        self.declare_parameter('mask_dim', 32)     # YOLOv8 default
        self.declare_parameter('stride', 32)
        self.declare_parameter('min_mask_area_px', 20)
        self.declare_parameter('profile', False)

        # ONNXRuntime: providers als STRING_ARRAY deklarieren!
        self.declare_parameter(
            'providers',
            ['CPUExecutionProvider'],  # sinnvoller Default für VM
            ParameterDescriptor(
                type=PT.PARAMETER_STRING_ARRAY,
                description='ONNX Runtime providers, e.g. [CPUExecutionProvider] or [CUDAExecutionProvider, CPUExecutionProvider]'
            )
        )
        # Optionale Outputnamen (falls Export anders heißt)
        self.declare_parameter('out_boxes_name', '')
        self.declare_parameter('out_proto_name', '')

        # --------- Parameter lesen ---------
        topic_in = self.get_parameter('topic_in').value
        publish_overlay = bool(self.get_parameter('publish_overlay').value)

        # Modellpfad bestimmen
        model_path = self.get_parameter('model_path').value
        if not model_path:
            try:
                share_dir = get_package_share_directory('strawberry_segmentation')
                default_model = os.path.join(share_dir, 'models', 'best.onnx')
                if os.path.exists(default_model):
                    model_path = default_model
                    self.get_logger().info(f"model_path leer -> nutze Paketmodell: {model_path}")
                else:
                    self.get_logger().error(
                        "Kein model_path gesetzt und kein Paketmodell gefunden "
                        "(share/strawberry_segmentation/models/best.onnx)."
                    )
            except Exception as e:
                self.get_logger().error(f"Konnte Paket-Share-Verzeichnis nicht ermitteln: {e}")

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX-Modell nicht gefunden: '{model_path}'")

        # Provider bestimmen (robust gegen versehentliche String-Übergaben)
        user_prov = self.get_parameter('providers').value
        if isinstance(user_prov, str):
            # z.B. "['CPUExecutionProvider']"
            user_prov = [s.strip(" '\"\t") for s in user_prov.strip('[]').split(',') if s.strip()]
        elif not isinstance(user_prov, (list, tuple)):
            user_prov = []

        avail = ort.get_available_providers()
        if user_prov:
            providers = list(user_prov)
        else:
            providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                         if 'CUDAExecutionProvider' in avail else ['CPUExecutionProvider'])

        self.get_logger().info(f"ONNX providers (gewählt): {providers}; verfügbar: {avail}")

        # CvBridge + QoS
        self.bridge = CvBridge()
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        # I/O
        self.sub_rgb = self.create_subscription(Image, topic_in, self.on_image, qos)
        self.pub_label = self.create_publisher(Image, '/seg/label_image', 10)
        self.pub_overlay = self.create_publisher(Image, '/seg/overlay', 10) if publish_overlay else None

        # ONNX Session
        t0 = time.time()
        sess_opts = ort.SessionOptions()
        # (Optional) leichte Optimierungen, aber konservativ halten:
        sess_opts.intra_op_num_threads = 0  # auto
        sess_opts.inter_op_num_threads = 0  # auto
        try:
            self.sess = ort.InferenceSession(model_path, providers=providers, sess_options=sess_opts)
        except Exception as e:
            self.get_logger().error(f"ONNX Session konnte nicht erstellt werden: {e}")
            raise

        # Input/Output-Namen
        self.inp_name = self.sess.get_inputs()[0].name

        # Outputnamen aus Param (falls gesetzt) oder per Shape erkennen
        pref_a = self.get_parameter('out_boxes_name').value or None
        pref_b = self.get_parameter('out_proto_name').value or None
        self.out_boxes_name, self.out_proto_name = pick_outputs_by_shape(
            self.sess,
            (pref_a, pref_b) if (pref_a and pref_b) else None
        )

        self.get_logger().info(
            f"Modell geladen: {model_path}  ({time.time()-t0:.2f}s). "
            f"Outputs: boxes='{self.out_boxes_name}', proto='{self.out_proto_name}'. "
            f"Input='{self.inp_name}'"
        )

        # Cache param-Werte (Performance)
        self._imgsz     = int(self.get_parameter('imgsz').value)
        self._stride    = int(self.get_parameter('stride').value)
        self._conf_th   = float(self.get_parameter('conf_thres').value)
        self._iou_th    = float(self.get_parameter('iou_thres').value)
        self._mask_th   = float(self.get_parameter('mask_thresh').value)
        self._max_det   = int(self.get_parameter('max_det').value)
        self._nm        = int(self.get_parameter('mask_dim').value)
        self._nc        = int(self.get_parameter('num_classes').value)
        self._min_area  = int(self.get_parameter('min_mask_area_px').value)
        self._profile   = bool(self.get_parameter('profile').value)

        # Optional: einmal „warmlaufen“, um JIT/Allocator zu triggern
        try:
            dummy = np.zeros((1, 3, self._imgsz, self._imgsz), dtype=np.float32)
            _ = self.sess.run([self.out_boxes_name, self.out_proto_name], {self.inp_name: dummy})
        except Exception:
            pass

    # ----------------- callback -----------------
    def on_image(self, msg: Image):
        t_all0 = time.time()
        # ROS -> RGB (np)
        img_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        h0, w0 = img_rgb.shape[:2]

        # preprocess
        t0 = time.time()
        img_lb, r, (dw, dh) = letterbox(img_rgb, (self._imgsz, self._imgsz), stride=self._stride)
        img_in = img_lb.astype(np.float32) / 255.0
        img_in = np.transpose(img_in, (2, 0, 1))  # CHW
        img_in = np.ascontiguousarray(img_in[None, ...])  # NCHW
        t_prep = time.time() - t0

        # inference
        t1 = time.time()
        try:
            pred, proto = self.sess.run([self.out_boxes_name, self.out_proto_name],
                                        {self.inp_name: img_in})
        except Exception as e:
            self.get_logger().error(f"Inferenz-Fehler: {e}")
            return
        t_inf = time.time() - t1

        # shapes normalisieren
        pred = np.squeeze(pred, axis=0)  # (N, D) erwartet
        if pred.ndim != 2:
            pred = pred.reshape(-1, pred.shape[-1])
        if pred.size == 0:
            self.publish_outputs(msg, img_rgb, np.zeros((h0, w0), dtype=np.uint16))
            if self._profile:
                self.get_logger().info(f"prep {t_prep*1000:.1f} ms | inf {t_inf*1000:.1f} ms | post ~0 ms")
            return

        # decode (xywh, cls confs, mask coeffs)
        D = pred.shape[1]
        expect = 4 + self._nc + self._nm
        if D < expect:
            self.get_logger().warn(f"Unerwartete Prädiktionslänge D={D} < {expect}. Prüfe Export/Param.")
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:4+self._nc]
        mask_coeff = pred[:, 4+self._nc:4+self._nc+self._nm]

        # Klassenkonfidenz/-ID
        cls_conf = cls_scores.max(axis=1)
        cls_ids  = cls_scores.argmax(axis=1)

        # Filter nach conf
        keep0 = cls_conf >= self._conf_th
        boxes_xywh = boxes_xywh[keep0]
        cls_conf   = cls_conf[keep0]
        cls_ids    = cls_ids[keep0]
        mask_coeff = mask_coeff[keep0]
        if boxes_xywh.shape[0] == 0:
            self.publish_outputs(msg, img_rgb, np.zeros((h0, w0), dtype=np.uint16))
            if self._profile:
                self.get_logger().info(f"prep {t_prep*1000:.1f} ms | inf {t_inf*1000:.1f} ms | post ~0 ms")
            return

        # Boxen zurück auf Originalbild
        boxes_xyxy = xywh2xyxy(boxes_xywh)
        boxes_xyxy[:, [0, 2]] -= dw
        boxes_xyxy[:, [1, 3]] -= dh
        boxes_xyxy /= r
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w0 - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h0 - 1)

        # NMS
        keep = nms_boxes(boxes_xyxy, cls_conf, iou_th=self._iou_th, top_k=self._max_det)
        boxes_xyxy = boxes_xyxy[keep]
        cls_conf   = cls_conf[keep]
        cls_ids    = cls_ids[keep]
        mask_coeff = mask_coeff[keep]
        n = boxes_xyxy.shape[0]
        if n == 0:
            self.publish_outputs(msg, img_rgb, np.zeros((h0, w0), dtype=np.uint16))
            if self._profile:
                self.get_logger().info(f"prep {t_prep*1000:.1f} ms | inf {t_inf*1000:.1f} ms | post ~0 ms")
            return

        # proto -> masks
        t2 = time.time()
        proto = np.squeeze(proto, axis=0)  # (nm, Mh, Mw) erwartet
        if proto.ndim == 4:
            # Manche Exporte liefern (1, nm, Mh, Mw) -> squeeze vergessen
            proto = np.squeeze(proto, axis=0)
        if proto.shape[0] != self._nm:
            # zur Not auf gemeinsame Mini-Dim schneiden
            nm_eff = min(proto.shape[0], self._nm)
            proto = proto[:nm_eff, ...]
            mask_coeff = mask_coeff[:, :nm_eff]
            self.get_logger().warn(f"mask_dim angepasst: proto_nm={proto.shape[0]} vs. param_nm={self._nm}")

        nm_, Mh, Mw = proto.shape
        proto_flat = proto.reshape(nm_, Mh * Mw)  # (nm, P)
        masks = sigmoid(np.matmul(mask_coeff, proto_flat))  # (n, P)
        masks = masks.reshape(n, Mh, Mw)

        # auf imgsz hoch, Padding weg, aufs Original skaliert
        masks_up = np.zeros((n, self._imgsz, self._imgsz), dtype=np.float32)
        for i in range(n):
            masks_up[i] = cv2.resize(masks[i], (self._imgsz, self._imgsz), interpolation=cv2.INTER_LINEAR)

        x0, y0 = int(round(dw)), int(round(dh))
        x1, y1 = self._imgsz - x0, self._imgsz - y0
        masks_up = masks_up[:, y0:y1, x0:x1]

        final_masks = np.zeros((n, h0, w0), dtype=np.uint8)
        for i in range(n):
            m = cv2.resize(masks_up[i], (w0, h0), interpolation=cv2.INTER_LINEAR)
            m = (m > self._mask_th).astype(np.uint8)
            # Masken auf Box beschränken (sauberere Kanten)
            x1i, y1i, x2i, y2i = boxes_xyxy[i].astype(int)
            if x2i > x1i and y2i > y1i:
                crop = np.zeros_like(m)
                crop[y1i:y2i+1, x1i:x2i+1] = 1
                m = m * crop
            final_masks[i] = m
        t_post = time.time() - t2

        # Instanz-Labelbild + Overlay
        order = np.argsort(cls_conf)[::-1]  # z-order nach Konfidenz
        label = np.zeros((h0, w0), dtype=np.uint16)
        overlay = img_rgb.copy()
        draw_overlay = self.pub_overlay is not None and self.pub_overlay.get_subscription_count() > 0

        for k, idx in enumerate(order, start=1):
            m = final_masks[idx]
            if int(m.sum()) < self._min_area:
                continue
            write = (m == 1) & (label == 0)
            label[write] = k

            if draw_overlay:
                col = np.array(color_from_id(k), dtype=np.uint8)
                overlay[write] = (0.6 * overlay[write] + 0.4 * col).astype(np.uint8)
                cnts, _ = cv2.findContours(m.astype(np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, cnts, -1,
                                 (int(col[0]), int(col[1]), int(col[2])), 2)

        # Publish
        self.publish_outputs(msg, overlay if draw_overlay else img_rgb, label)

        if self._profile:
            t_all = (time.time() - t_all0) * 1000.0
            self.get_logger().info(
                f"prep {t_prep*1000:.1f} ms | inf {t_inf*1000:.1f} ms | post {t_post*1000:.1f} ms | total {t_all:.1f} ms"
            )

    # ----------------- publish -----------------
    def publish_outputs(self, src_msg: Image, overlay_rgb: np.ndarray, label_u16: np.ndarray):
        # overlay (optional)
        if self.pub_overlay is not None and self.pub_overlay.get_subscription_count() > 0:
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay_rgb, encoding='rgb8')
            overlay_msg.header = Header(stamp=src_msg.header.stamp, frame_id=src_msg.header.frame_id)
            self.pub_overlay.publish(overlay_msg)

        # label image (mono16)
        H, W = label_u16.shape
        label_msg = Image()
        label_msg.header = Header(stamp=src_msg.header.stamp, frame_id=src_msg.header.frame_id)
        label_msg.height = H
        label_msg.width = W
        label_msg.encoding = 'mono16'
        label_msg.is_bigendian = 0
        label_msg.step = W * 2
        label_msg.data = label_u16.tobytes()
        self.pub_label.publish(label_msg)


def main():
    rclpy.init()
    node = YoloSegOnnxNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
