from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2


def save_depth_preview(depth_u16: np.ndarray, out_png: Path) -> None:
    d = depth_u16.astype(np.float32)
    d[d <= 0] = np.nan

    if np.all(~np.isfinite(d)):
        img = np.zeros_like(depth_u16, dtype=np.uint8)
    else:
        hi = np.nanpercentile(d, 99.0)
        lo = np.nanpercentile(d, 1.0)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, float(np.nanmax(d))
        x = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        img = (255.0 * np.nan_to_num(x)).astype(np.uint8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)
