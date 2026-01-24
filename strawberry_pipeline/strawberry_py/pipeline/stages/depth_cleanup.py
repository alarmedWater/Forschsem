# strawberry_py/pipeline/stages/depth_cleanup.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from strawberry_py.config import DepthCfg
from strawberry_py.pipeline.stages.transforms import depth_u16_to_meters
from strawberry_py.utils.masks import largest_cc_mask


@dataclass(frozen=True)
class DepthCleanupResult:
    depth_u16: np.ndarray
    stats: Dict[str, float]


class DepthCleaner:
    """
    Post-process already masked depth image:
      - robust median/MAD outlier removal (in meters)
      - optionally keep only largest 2D connected component afterwards
    """

    def __init__(
        self,
        depth_cfg: DepthCfg,
        min_valid_px: int = 50,
        band_k: float = 3.5,
        min_band_m: float = 0.010,
        keep_largest_cc: bool = True,
    ) -> None:
        self.depth_cfg = depth_cfg
        self.min_valid_px = int(min_valid_px)
        self.band_k = float(band_k)
        self.min_band_m = float(min_band_m)
        self.keep_largest_cc = bool(keep_largest_cc)

    def __call__(self, depth_u16: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        d = np.asarray(depth_u16)
        if d.ndim != 2:
            raise ValueError(f"depth must be 2D, got shape={d.shape}")

        # Convert to meters for statistics
        z_m = depth_u16_to_meters(d, self.depth_cfg)
        valid = np.isfinite(z_m) & (z_m > 0.0)
        n_valid = int(valid.sum())

        if n_valid < self.min_valid_px:
            out = d.copy()
            if self.keep_largest_cc:
                out = self._keep_largest_cc(out)
            return out, {"n_valid": float(n_valid), "removed": 0.0, "band_m": 0.0}

        z = z_m[valid].astype(np.float64)
        med = float(np.median(z))
        mad = float(np.median(np.abs(z - med)))
        sigma = 1.4826 * mad  # MAD->sigma estimate
        band = max(self.min_band_m, self.band_k * sigma)

        keep = valid & (z_m >= (med - band)) & (z_m <= (med + band))
        removed = float(int(valid.sum()) - int(keep.sum()))

        out = d.copy()
        out[~keep] = 0

        if self.keep_largest_cc:
            out = self._keep_largest_cc(out)

        stats = {
            "n_valid": float(n_valid),
            "removed": float(removed),
            "med_m": float(med),
            "mad_m": float(mad),
            "band_m": float(band),
        }
        return out, stats

    @staticmethod
    def _keep_largest_cc(depth_u16: np.ndarray) -> np.ndarray:
        out = np.asarray(depth_u16).copy()
        m = (out > 0).astype(np.uint8) * 255
        m = largest_cc_mask(m)
        out[m == 0] = 0
        return out
