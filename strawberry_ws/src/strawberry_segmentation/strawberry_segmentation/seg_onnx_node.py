#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 node: YOLOv8-seg (ONNXRuntime) inference-only.
Subscribes:
  - /camera/color/image_raw (sensor_msgs/Image, rgb8)
Publishes:
  - /seg/label_image (sensor_msgs/Image, mono16)  instance-id map (0=bg, 1..N)
  - /seg/overlay     (sensor_msgs/Image, rgb8)    color overlay for RViz

Assumptions (Ultralytics YOLOv8-seg ONNX export):
  outputs:
    - 'output0' : (1, N, 4+nc+nm)  -> boxes(xywh), cls scores, mask coeffs
    - 'output1' : (1, nm, Mh, Mw)  -> mask prototype
You can override tensor names & dims via parameters if your export differs.
"""

import math
import numpy as np
import cv2
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

# ----------------- helpers -----------------
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), stride=32):
    shape = im.shape[:2]  # h,w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # w,h
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)

def sigmoid(x): return 1 / (1 + np.exp(-x))

def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y

def nms_boxes(boxes, scores, iou_th=0.5, top_k=300):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0 and len(keep) < top_k:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1: break
        ious = iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_th]
    return np.array(keep, dtype=np.int32)

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

def color_from_id(k: int):
    np.random.seed(k * 12345 + 7)
    return tuple(int(v) for v in np.random.randint(64, 255, size=3, dtype=np.uint8))

# ----------------- node -----------------
class YoloSegOnnxNode(Node):
    def __init__(self):
        super().__init__('strawberry_seg_onnx')

        # Params
        self.declare_parameter('model_path', '/ABSOLUTE/PATH/TO/best.onnx')
        self.declare_parameter('imgsz', 1024)
        self.declare_parameter('conf_thres', 0.25)
        self.declare_parameter('iou_thres', 0.50)
        self.declare_parameter('max_det', 300)
        self.declare_parameter('num_classes', 1)   # strawberries only
        self.declare_parameter('mask_dim', 32)     # YOLOv8 default
        self.declare_parameter('mask_thresh', 0.5)
        self.declare_parameter('stride', 32)
        self.declare_parameter('providers', ['CPUExecutionProvider'])  # or CUDAExecutionProvider if available

        # Tensor names (can differ by export; override if needed)
        self.declare_parameter('out_boxes_name', 'output0')
        self.declare_parameter('out_proto_name', 'output1')

        model_path = self.get_parameter('model_path').value
        providers  = self.get_parameter('providers').value

        self.bridge = CvBridge()

        # QoS
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5
        )

        # I/O
        self.sub_rgb = self.create_subscription(Image, '/camera/color/image_raw', self.on_image, qos)
        self.pub_label = self.create_publisher(Image, '/seg/label_image', 10)
        self.pub_overlay = self.create_publisher(Image, '/seg/overlay', 10)

        # ONNX session
        sess_opts = ort.SessionOptions()
        self.sess = ort.InferenceSession(model_path, providers=providers, sess_options=sess_opts)

        self.inp_name = self.sess.get_inputs()[0].name
        self.out_boxes_name = self.get_parameter('out_boxes_name').value
        self.out_proto_name = self.get_parameter('out_proto_name').value

        out_names = [o.name for o in self.sess.get_outputs()]
        if self.out_boxes_name not in out_names or self.out_proto_name not in out_names:
            self.get_logger().warn(f"Output names not found in model: {out_names}. Using first two outputs as fallback.")
            self.out_boxes_name, self.out_proto_name = out_names[:2]

        self.get_logger().info(f"Loaded ONNX model: {model_path}")
        self.get_logger().info(f"Outputs: boxes='{self.out_boxes_name}', proto='{self.out_proto_name}'")

    def on_image(self, msg: Image):
        # ROS -> np RGB
        img_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        h0, w0 = img_rgb.shape[:2]

        imgsz = int(self.get_parameter('imgsz').value)
        stride = int(self.get_parameter('stride').value)

        # preprocess: letterbox, normalize, CHW
        img_lb, r, (dw, dh) = letterbox(img_rgb, (imgsz, imgsz), stride=stride)
        img_in = img_lb.astype(np.float32) / 255.0
        img_in = np.transpose(img_in, (2, 0, 1))[None, ...]   # 1x3xHxW
        # run
        out = self.sess.run([self.out_boxes_name, self.out_proto_name], {self.inp_name: img_in})
        pred, proto = out[0], out[1]  # pred: (1,N,4+nc+nm), proto: (1, nm, Mh, Mw)

        pred = np.squeeze(pred, axis=0)  # (N, D)
        nm = int(self.get_parameter('mask_dim').value)
        nc = int(self.get_parameter('num_classes').value)
        conf_th = float(self.get_parameter('conf_thres').value)
        iou_th  = float(self.get_parameter('iou_thres').value)
        max_det = int(self.get_parameter('max_det').value)
        mask_thr = float(self.get_parameter('mask_thresh').value)

        if pred.size == 0:
            # publish empty label/overlay
            self.publish_outputs(msg, img_rgb, np.zeros((h0, w0), dtype=np.uint16))
            return

        # decode (xywh, cls confs, mask coeffs)
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:4+nc]
        mask_coeff = pred[:, 4+nc:4+nc+nm]

        cls_conf = cls_scores.max(axis=1)
        cls_ids  = cls_scores.argmax(axis=1)

        # filter by conf
        keep0 = cls_conf >= conf_th
        boxes_xywh = boxes_xywh[keep0]
        cls_conf   = cls_conf[keep0]
        cls_ids    = cls_ids[keep0]
        mask_coeff = mask_coeff[keep0]
        if boxes_xywh.shape[0] == 0:
            self.publish_outputs(msg, img_rgb, np.zeros((h0, w0), dtype=np.uint16))
            return

        # scale boxes back to letterboxed image, then to original
        boxes_xyxy = xywh2xyxy(boxes_xywh)
        # undo padding
        boxes_xyxy[:, [0,2]] -= dw
        boxes_xyxy[:, [1,3]] -= dh
        boxes_xyxy /= r
        # clip
        boxes_xyxy[:, [0,2]] = np.clip(boxes_xyxy[:, [0,2]], 0, w0 - 1)
        boxes_xyxy[:, [1,3]] = np.clip(boxes_xyxy[:, [1,3]], 0, h0 - 1)

        # NMS
        keep = nms_boxes(boxes_xyxy, cls_conf, iou_th=iou_th, top_k=max_det)
        boxes_xyxy = boxes_xyxy[keep]
        cls_conf   = cls_conf[keep]
        cls_ids    = cls_ids[keep]
        mask_coeff = mask_coeff[keep]
        n = boxes_xyxy.shape[0]
        if n == 0:
            self.publish_outputs(msg, img_rgb, np.zeros((h0, w0), dtype=np.uint16))
            return

        # proto -> masks
        # proto: (1, nm, Mh, Mw) -> (nm, Mh*Mw)
        proto = np.squeeze(proto, axis=0)
        nm_, Mh, Mw = proto.shape
        proto_flat = proto.reshape(nm_, Mh * Mw)  # (nm, P)
        # (n, nm) x (nm, P) = (n, P)
        masks = sigmoid(np.matmul(mask_coeff, proto_flat))  # (n, P)
        masks = masks.reshape(n, Mh, Mw)

        # upsample to letterbox size, then crop to original (remove pad, then scale 1/r)
        # first to imgsz x imgsz
        masks_up = np.zeros((n, imgsz, imgsz), dtype=np.float32)
        for i in range(n):
            masks_up[i] = cv2.resize(masks[i], (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

        # remove padding
        x0, y0 = int(round(dw)), int(round(dh))
        x1, y1 = imgsz - x0, imgsz - y0
        masks_up = masks_up[:, y0:y1, x0:x1]  # (n, ch, cw) where ch≈r*h0

        # scale to original size
        final_masks = np.zeros((n, h0, w0), dtype=np.uint8)
        for i in range(n):
            m = cv2.resize(masks_up[i], (w0, h0), interpolation=cv2.INTER_LINEAR)
            m = (m > mask_thr).astype(np.uint8)
            # optional: restrict by its box (clean edges)
            x1i,y1i,x2i,y2i = boxes_xyxy[i].astype(int)
            if x2i > x1i and y2i > y1i:
                crop = np.zeros_like(m)
                crop[y1i:y2i+1, x1i:x2i+1] = 1
                m = m * crop
            final_masks[i] = m

        # compose label image (mono16) with occlusion ordering by score
        order = np.argsort(cls_conf)[::-1]
        label = np.zeros((h0, w0), dtype=np.uint16)
        overlay = img_rgb.copy()
        for k, idx in enumerate(order, start=1):
            m = final_masks[idx]
            if m.sum() < 20:
                continue
            # write label only where 0 (z-order by confidence)
            write = (m == 1) & (label == 0)
            label[write] = k
            # overlay color
            col = np.array(color_from_id(k), dtype=np.uint8)
            overlay[write] = (0.6 * overlay[write] + 0.4 * col).astype(np.uint8)
            # draw contour
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, (int(col[0]), int(col[1]), int(col[2])), 2)

        self.publish_outputs(msg, overlay, label)

    def publish_outputs(self, src_msg: Image, overlay_rgb: np.ndarray, label_u16: np.ndarray):
        # overlay
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay_rgb, encoding='rgb8')
        overlay_msg.header = Header(stamp=src_msg.header.stamp, frame_id=src_msg.header.frame_id)
        # label
        H, W = label_u16.shape
        label_msg = Image()
        label_msg.header = overlay_msg.header
        label_msg.height = H
        label_msg.width = W
        label_msg.encoding = 'mono16'
        label_msg.is_bigendian = 0
        label_msg.step = W * 2
        label_msg.data = label_u16.tobytes()

        self.pub_overlay.publish(overlay_msg)
        self.pub_label.publish(label_msg)

def main():
    rclpy.init()
    node = YoloSegOnnxNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
