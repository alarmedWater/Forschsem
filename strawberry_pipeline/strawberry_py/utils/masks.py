# strawberry_py/utils/masks.py
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def morph_cleanup(
    mask_u8: np.ndarray,
    ksize: int = 5,
    open_iter: int = 1,
    close_iter: int = 1,
) -> np.ndarray:
    """
    Mild morphology to remove speckles and close small holes.

    Expects binary-ish uint8 mask (0/255). Returns uint8 mask (0/255).
    """
    if mask_u8 is None or getattr(mask_u8, "size", 0) == 0:
        return mask_u8

    m = (mask_u8 > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    if open_iter > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=int(open_iter))
    if close_iter > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=int(close_iter))

    return m.astype(np.uint8, copy=False)


def largest_cc_mask(mask_u8: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """
    Keep only the largest connected component in a binary uint8 mask (0/255).
    Returns uint8 mask (0/255).
    """
    if mask_u8 is None or getattr(mask_u8, "size", 0) == 0:
        return mask_u8

    m = (mask_u8 > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return (m * 255).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=int(connectivity))
    # n includes background label 0
    if n <= 1:
        return (m * 255).astype(np.uint8)
    if n == 2:
        return (m * 255).astype(np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    out = (labels == best).astype(np.uint8) * 255
    return out


def reduce_to_selected_label(
    label_full: np.ndarray,
    selected_id: Optional[int] = None,
    do_morph: bool = True,
    connectivity: int = 8,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Reduce an instance label image to exactly ONE selected instance.

    Args:
      label_full: 2D label image, background=0, instances>0 (any int dtype)
      selected_id:
        - None -> pick the largest instance (by pixel count)
        - int  -> try that id; if absent, fallback to largest
      do_morph: apply morph cleanup + keep largest 2D CC
      connectivity: CC connectivity

    Returns:
      label_sel (uint16): binary label encoded as 1 (else 0), shape like input
      stats: debug numbers
    """
    lbl = np.asarray(label_full)
    if lbl.ndim != 2:
        raise ValueError(f"label must be 2D, got shape={lbl.shape}")

    ids, counts = np.unique(lbl[lbl > 0], return_counts=True)
    if ids.size == 0:
        empty = np.zeros_like(lbl, dtype=np.uint16)
        return empty, {"picked_id": -1.0, "area_px": 0.0, "fallback_used": 1.0}

    fallback_used = 0.0

    # choose id
    if selected_id is None:
        picked = int(ids[int(np.argmax(counts))])
    else:
        picked = int(selected_id)
        if picked not in set(ids.tolist()):
            picked = int(ids[int(np.argmax(counts))])
            fallback_used = 1.0

    # binary mask for picked
    mask = (lbl == picked).astype(np.uint8) * 255

    if do_morph:
        mask = morph_cleanup(mask)
        mask = largest_cc_mask(mask, connectivity=int(connectivity))

    area_px = float(int((mask > 0).sum()))

    # encode as label=1 for extractor + depth masker
    label_sel = (mask > 0).astype(np.uint16)

    return label_sel, {
        "picked_id": float(picked),
        "area_px": area_px,
        "fallback_used": float(fallback_used),
    }
