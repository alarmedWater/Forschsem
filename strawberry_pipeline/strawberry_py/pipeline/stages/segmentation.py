from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from strawberry_py.types import RGBImage, SegmentationResult, assert_rgb, assert_label_u16


class YoloV8Segmenter:
    """
    YOLOv8 instance segmentation wrapper with:
      - mask area filtering
      - border-relaxation (cropped objects)
      - fallback inference if nothing survives filtering
    """

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
        # --- New robustness options ---
        border_relax_factor: float = 0.25,     # if mask touches border, allow min_area * factor
        fallback_enabled: bool = True,
        fallback_conf: float = 0.35,           # used only if first pass yields zero instances
        fallback_imgsz: Optional[int] = None,  # if None -> keep imgsz
        fallback_min_mask_area_px: Optional[int] = None,  # if None -> smaller of min_area or 600
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

        self._YOLO = YOLO
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

        # ultralytics versions differ a bit; keep it resilient.
        try:
            return self._model.predict(**kwargs)
        except TypeError:
            kwargs.pop("retina_masks", None)
            # Some versions also don't like 'classes' in predict(**kwargs)
            # but that's usually fine. Keep conservative:
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
        # Any pixel on border?
        return (
            bool(mask01[0, :].any())
            or bool(mask01[-1, :].any())
            or bool(mask01[:, 0].any())
            or bool(mask01[:, -1].any())
        )

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

        # Sort by confidence descending (if available) so "best" wins in overlaps.
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

            # area filtering with border relax
            if area < min_area_px:
                if min_area_border > 0 and area >= min_area_border and self._touches_border(m):
                    pass  # accept as cropped/border instance
                else:
                    continue

            # Keep only pixels not assigned yet (so instances don't overwrite each other)
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
                    confs = self._get_confidences(res0, int(masks01.shape[0]))
                    label = self._build_label_from_masks(
                        masks01,
                        confs,
                        min_area_px=fb_min_area,
                        border_relax_factor=self.border_relax_factor,
                    )

                # (optional) overwrite overlay with fallback overlay
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

        # label_vis: mono8 0/255
        label_vis = ((label > 0).astype(np.uint8) * np.uint8(255))

        assert_label_u16(label)
        return SegmentationResult(label=label, overlay_rgb=overlay_rgb, label_vis=label_vis)
