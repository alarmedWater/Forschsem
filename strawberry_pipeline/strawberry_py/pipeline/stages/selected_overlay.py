from __future__ import annotations

import cv2
import numpy as np

from strawberry_py.st_types import RGBImage, LabelImage, assert_rgb, assert_label_u16


def selected_overlay(
    rgb: RGBImage,
    label: LabelImage,
    selected_id: int,
    min_pixels: int = 50,
    darken_factor: float = 0.3,
    draw_bbox: bool = True,
) -> RGBImage:
    assert_rgb(rgb)
    assert_label_u16(label)
    if rgb.shape[:2] != label.shape:
        raise ValueError("Shape mismatch RGB vs label")

    mask = label == np.uint16(int(selected_id))
    n_pix = int(mask.sum())

    out = rgb.copy()
    if n_pix >= int(min_pixels):
        df = float(np.clip(darken_factor, 0.0, 1.0))
        out = (rgb.astype(np.float32) * df).astype(np.uint8)
        out[mask] = rgb[mask]

        if draw_bbox:
            ys, xs = np.where(mask)
            if ys.size > 0 and xs.size > 0:
                y0, y1 = int(ys.min()), int(ys.max())
                x0, x1 = int(xs.min()), int(xs.max())
                cv2.rectangle(out, (x0, y0), (x1, y1), color=(255, 0, 0), thickness=2)
                cv2.putText(
                    out, f"id={selected_id} pix={n_pix}", (x0, max(y0 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA
                )
    return out
