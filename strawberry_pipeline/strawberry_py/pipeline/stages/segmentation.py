# strawberry_py/pipeline/stages/segmentation.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from strawberry_py.config import SegmentationCfg
from strawberry_py.st_types import RGBImage, SegmentationResult, assert_label_u16, assert_rgb


class YoloV8Segmenter:
    """
    YOLOv8 instance segmentation wrapper with:
      - mask area filtering
      - border-relaxation (cropped objects)
      - fallback inference if nothing survives filtering
      - OPTIONAL postprocessing per mask (close + fill holes + largest CC)
    """

    @classmethod
    def from_cfg(cls, cfg: SegmentationCfg) -> "YoloV8Segmenter":
        pp = cfg.postprocess
        return cls(
            model_path=str(cfg.model_path),
            device=str(cfg.device),
            imgsz=int(cfg.imgsz),
            conf=float(cfg.conf),
            iou=float(cfg.iou),
            max_det=int(cfg.max_det),
            min_mask_area_px=int(cfg.min_mask_area_px),
            classes=list(cfg.classes),

            border_relax_factor=float(cfg.border_relax_factor),
            fallback_enabled=bool(cfg.fallback_enabled),
            fallback_conf=float(cfg.fallback_conf),
            fallback_imgsz=(int(cfg.fallback_imgsz) if cfg.fallback_imgsz is not None else None),
            fallback_min_mask_area_px=(
                int(cfg.fallback_min_mask_area_px) if cfg.fallback_min_mask_area_px is not None else None
            ),

            postprocess_enabled=bool(pp.enabled),
            postprocess_keep_largest_cc=bool(pp.keep_largest_cc),
            postprocess_morph_open=bool(pp.morph_open),
            postprocess_morph_close=bool(pp.morph_close),
            postprocess_kernel_size=int(pp.kernel_size),
            postprocess_open_iters=int(pp.open_iters),
            postprocess_close_iters=int(pp.close_iters),
            postprocess_fill_holes=bool(pp.fill_holes),
        )

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        imgsz: int = 640,
        conf: float = 0.65,
        iou: float = 0.50,
        max_det: int = 100,
        min_mask_area_px: int = 1500,
        classes: Optional[List[int]] = None,

        border_relax_factor: float = 0.25,
        fallback_enabled: bool = True,
        fallback_conf: float = 0.35,
        fallback_imgsz: Optional[int] = None,
        fallback_min_mask_area_px: Optional[int] = None,

        # Postprocess (fix holes / speckles)
        postprocess_enabled: bool = True,
        postprocess_keep_largest_cc: bool = True,
        postprocess_morph_open: bool = True,
        postprocess_morph_close: bool = True,
        postprocess_kernel_size: int = 5,
        postprocess_open_iters: int = 1,
        postprocess_close_iters: int = 1,
        postprocess_fill_holes: bool = True,
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise ImportError("ultralytics is required: pip install ultralytics") from exc

        mp = Path(model_path)
        if not mp.exists():
            raise FileNotFoundError(f"YOLO model not found: {mp}")

        # Robust "auto" device handling: if no CUDA -> CPU.
        dev = str(device).strip().lower()
        if dev in ("", "auto"):
            try:
                import torch  # type: ignore
                dev = "0" if torch.cuda.is_available() else "cpu"
            except Exception:
                dev = "cpu"

        self._model = YOLO(str(mp))
        self.device = dev
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.min_mask_area_px = int(min_mask_area_px)
        self.classes = list(classes) if classes else []

        self.border_relax_factor = float(np.clip(border_relax_factor, 0.0, 1.0))

        self.fallback_enabled = bool(fallback_enabled)
        self.fallback_conf = float(fallback_conf)
        self.fallback_imgsz = int(fallback_imgsz) if fallback_imgsz is not None else None
        if fallback_min_mask_area_px is None:
            self.fallback_min_mask_area_px = int(min(self.min_mask_area_px, 600))
        else:
            self.fallback_min_mask_area_px = int(fallback_min_mask_area_px)

        # postprocess params
        self.postprocess_enabled = bool(postprocess_enabled)
        self.postprocess_keep_largest_cc = bool(postprocess_keep_largest_cc)
        self.postprocess_morph_open = bool(postprocess_morph_open)
        self.postprocess_morph_close = bool(postprocess_morph_close)
        self.postprocess_fill_holes = bool(postprocess_fill_holes)

        k = int(postprocess_kernel_size)
        if k < 1:
            k = 1
        if (k % 2) == 0:
            k += 1
        self.postprocess_kernel_size = k
        self.postprocess_open_iters = max(0, int(postprocess_open_iters))
        self.postprocess_close_iters = max(0, int(postprocess_close_iters))

    def _predict(self, img_bgr: np.ndarray, *, conf: Optional[float] = None, imgsz: Optional[int] = None) -> Any:
        kwargs: Dict[str, Any] = {
            "source": img_bgr,
            "imgsz": int(self.imgsz if imgsz is None else imgsz),
            "conf": float(self.conf if conf is None else conf),
            "iou": float(self.iou),
            "max_det": int(self.max_det),
            "device": str(self.device),
            "verbose": False,
            "retina_masks": True,
        }
        if self.classes:
            kwargs["classes"] = self.classes

        try:
            return self._model.predict(**kwargs)
        except TypeError:
            kwargs.pop("retina_masks", None)
            if "classes" in kwargs:
                kwargs.pop("classes", None)
            return self._model.predict(**kwargs)

    @staticmethod
    def _masks_to_numpy_u8(res: Any, h0: int, w0: int) -> Optional[np.ndarray]:
        """Return masks as uint8 {0,1} array shape (N,H,W) in original image size."""
        if res is None or getattr(res, "masks", None) is None:
            return None
        if getattr(res.masks, "data", None) is None:
            return None

        data = res.masks.data
        if hasattr(data, "detach"):
            masks = data.detach().cpu().numpy()
        else:
            masks = np.asarray(data)

        if masks.ndim != 3:
            return None

        masks = (masks > 0.5).astype(np.uint8)  # (N,H,W) in model output resolution
        hm, wm = int(masks.shape[1]), int(masks.shape[2])

        if (hm, wm) != (h0, w0):
            resized = np.zeros((masks.shape[0], h0, w0), dtype=np.uint8)
            for i in range(masks.shape[0]):
                mi = (masks[i] * 255).astype(np.uint8)
                ri = cv2.resize(mi, (w0, h0), interpolation=cv2.INTER_NEAREST)
                resized[i] = (ri > 127).astype(np.uint8)
            masks = resized

        return masks

    @staticmethod
    def _get_confidences(res0: Any, n: int) -> Optional[np.ndarray]:
        """Try to fetch confidences aligned with masks."""
        boxes = getattr(res0, "boxes", None)
        if boxes is None:
            return None
        conf = getattr(boxes, "conf", None)
        if conf is None:
            return None

        try:
            if hasattr(conf, "detach"):
                c = conf.detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                c = np.asarray(conf, dtype=np.float32)
        except Exception:
            return None

        if c.ndim != 1 or c.shape[0] != n:
            return None
        return c

    @staticmethod
    def _touches_border(mask01: np.ndarray) -> bool:
        """mask01 is uint8 {0,1} shape (H,W)."""
        if mask01.size == 0:
            return False
        return (
            bool(mask01[0, :].any())
            or bool(mask01[-1, :].any())
            or bool(mask01[:, 0].any())
            or bool(mask01[:, -1].any())
        )

    @staticmethod
    def _largest_cc(mask01: np.ndarray) -> np.ndarray:
        """Keep only largest connected component in a {0,1} mask."""
        m = (mask01 > 0).astype(np.uint8)
        if m.sum() == 0:
            return m
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 2:
            return m
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        return (labels == best).astype(np.uint8)

    @staticmethod
    def _fill_holes(mask01: np.ndarray) -> np.ndarray:
        """
        Fill holes in a binary mask via floodfill on the inverted image.
        Works robustly even if the object touches the border (we pad first).
        """
        m = (mask01 > 0).astype(np.uint8) * 255
        if m.sum() == 0:
            return (m > 0).astype(np.uint8)

        # pad with 1px background border
        mp = cv2.copyMakeBorder(m, 1, 1, 1, 1, borderType=cv2.BORDER_CONSTANT, value=0)
        inv = cv2.bitwise_not(mp)

        h, w = inv.shape[:2]
        ff = inv.copy()
        mask_ff = np.zeros((h + 2, w + 2), dtype=np.uint8)

        # Flood-fill background (in inverted image) starting from (0,0)
        cv2.floodFill(ff, mask_ff, (0, 0), 0)

        # remaining >0 in ff are holes
        holes = ff > 0
        mp[holes] = 255

        # unpad
        out = mp[1:-1, 1:-1]
        return (out > 0).astype(np.uint8)

    def _postprocess_one(self, mask01: np.ndarray) -> np.ndarray:
        if not self.postprocess_enabled:
            return (mask01 > 0).astype(np.uint8)

        m = (mask01 > 0).astype(np.uint8) * 255
        if m.sum() == 0:
            return (m > 0).astype(np.uint8)

        k = self.postprocess_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        if self.postprocess_morph_open and self.postprocess_open_iters > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=self.postprocess_open_iters)

        if self.postprocess_morph_close and self.postprocess_close_iters > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=self.postprocess_close_iters)

        m01 = (m > 0).astype(np.uint8)

        if self.postprocess_fill_holes:
            m01 = self._fill_holes(m01)

        if self.postprocess_keep_largest_cc:
            m01 = self._largest_cc(m01)

        return m01

    def _postprocess_masks(self, masks01: np.ndarray) -> np.ndarray:
        if masks01 is None or masks01.size == 0:
            return masks01
        if not self.postprocess_enabled:
            return (masks01 > 0).astype(np.uint8)

        out = np.zeros_like(masks01, dtype=np.uint8)
        for i in range(int(masks01.shape[0])):
            out[i] = self._postprocess_one(masks01[i])
        return out

    def _build_label_from_masks(
        self,
        masks01: np.ndarray,
        confs: Optional[np.ndarray],
        *,
        min_area_px: int,
        border_relax_factor: float,
    ) -> np.ndarray:
        """Return label image uint16 with instance ids 1..K."""
        h0, w0 = int(masks01.shape[1]), int(masks01.shape[2])
        label = np.zeros((h0, w0), dtype=np.uint16)

        n = int(masks01.shape[0])
        if n <= 0:
            return label

        if confs is not None:
            order = np.argsort(confs)[::-1]
        else:
            order = np.arange(n, dtype=np.int32)

        k_out = 0
        min_area_px = int(max(0, min_area_px))
        border_relax_factor = float(np.clip(border_relax_factor, 0.0, 1.0))
        min_area_border = int(round(min_area_px * border_relax_factor))

        for k in order.tolist():
            m = masks01[int(k)]
            area = int(m.sum())

            if area < min_area_px:
                if min_area_border > 0 and area >= min_area_border and self._touches_border(m):
                    pass
                else:
                    continue

            newpix = (m == 1) & (label == 0)
            if not np.any(newpix):
                continue
            k_out += 1
            label[newpix] = np.uint16(k_out)

        return label

    def __call__(self, rgb: RGBImage) -> SegmentationResult:
        assert_rgb(rgb)
        h0, w0 = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # --- pass 1 ---
        results = self._predict(bgr, conf=self.conf, imgsz=self.imgsz)
        label = np.zeros((h0, w0), dtype=np.uint16)
        overlay_rgb: Optional[RGBImage] = None

        if results:
            res0 = results[0]
            masks01 = self._masks_to_numpy_u8(res0, h0, w0)
            if masks01 is not None and masks01.shape[0] > 0:
                masks01 = self._postprocess_masks(masks01)
                confs = self._get_confidences(res0, int(masks01.shape[0]))
                label = self._build_label_from_masks(
                    masks01,
                    confs,
                    min_area_px=self.min_mask_area_px,
                    border_relax_factor=self.border_relax_factor,
                )

            # overlay (best-effort)
            try:
                ov_bgr = res0.plot()
                if isinstance(ov_bgr, np.ndarray):
                    ov_rgb = cv2.cvtColor(ov_bgr, cv2.COLOR_BGR2RGB)
                    ov_rgb = ov_rgb.astype(np.uint8, copy=False)
                    if ov_rgb.shape[:2] != (h0, w0):
                        ov_rgb = cv2.resize(ov_rgb, (w0, h0), interpolation=cv2.INTER_LINEAR)
                    overlay_rgb = ov_rgb
            except Exception:
                overlay_rgb = None

        # --- fallback pass if empty ---
        if self.fallback_enabled and int(label.max()) == 0:
            fb_imgsz = self.imgsz if self.fallback_imgsz is None else int(self.fallback_imgsz)
            fb_conf = float(self.fallback_conf)
            fb_min_area = int(self.fallback_min_mask_area_px)

            results2 = self._predict(bgr, conf=fb_conf, imgsz=fb_imgsz)
            if results2:
                res0 = results2[0]
                masks01 = self._masks_to_numpy_u8(res0, h0, w0)
                if masks01 is not None and masks01.shape[0] > 0:
                    masks01 = self._postprocess_masks(masks01)
                    confs = self._get_confidences(res0, int(masks01.shape[0]))
                    label = self._build_label_from_masks(
                        masks01,
                        confs,
                        min_area_px=fb_min_area,
                        border_relax_factor=self.border_relax_factor,
                    )

                try:
                    ov_bgr = res0.plot()
                    if isinstance(ov_bgr, np.ndarray):
                        ov_rgb = cv2.cvtColor(ov_bgr, cv2.COLOR_BGR2RGB)
                        ov_rgb = ov_rgb.astype(np.uint8, copy=False)
                        if ov_rgb.shape[:2] != (h0, w0):
                            ov_rgb = cv2.resize(ov_rgb, (w0, h0), interpolation=cv2.INTER_LINEAR)
                        overlay_rgb = ov_rgb
                except Exception:
                    pass

        label_vis = ((label > 0).astype(np.uint8) * np.uint8(255))
        assert_label_u16(label)
        return SegmentationResult(label=label, overlay_rgb=overlay_rgb, label_vis=label_vis)
