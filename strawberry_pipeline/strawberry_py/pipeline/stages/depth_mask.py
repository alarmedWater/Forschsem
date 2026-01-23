from __future__ import annotations

import numpy as np

from strawberry_py.config import DepthCfg
from strawberry_py.st_types import DepthU16, LabelImage, DepthMaskResult, assert_depth_u16, assert_label_u16
from strawberry_py.pipeline.stages.transforms import compute_valid_range_mask


class DepthMasker:
    def __init__(self, depth_cfg: DepthCfg, zero_background: bool = True) -> None:
        self.depth_cfg = depth_cfg
        self.zero_background = bool(zero_background)

    def __call__(self, depth: DepthU16, label: LabelImage) -> DepthMaskResult:
        assert_depth_u16(depth)
        assert_label_u16(label)
        if depth.shape != label.shape:
            raise ValueError(f"Shape mismatch depth={depth.shape} label={label.shape}")

        range_ok = compute_valid_range_mask(depth, self.depth_cfg)
        out = depth.copy()

        if self.zero_background:
            keep = (label > 0) & range_ok
            out[~keep] = np.uint16(0)
        else:
            out[~range_ok] = np.uint16(0)

        return DepthMaskResult(depth_masked=out, valid_range_mask=range_ok)
