#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path, flags: int) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def component_count(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return max(0, count - 1)


def refine_mask(
    rgb_path: Path,
    mask_path: Path,
    hue_min: int,
    hue_max: int,
    sat_min: int,
    green_red_margin: int,
    morph_kernel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    bgr = read_image(rgb_path, cv2.IMREAD_COLOR)
    mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE) > 0

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(bgr)
    green = (
        mask
        & (hsv[:, :, 0] >= hue_min)
        & (hsv[:, :, 0] <= hue_max)
        & (hsv[:, :, 1] >= sat_min)
        & (g.astype(np.int16) >= r.astype(np.int16) + green_red_margin)
    )

    refined = mask & ~green
    if morph_kernel:
        if morph_kernel < 1 or morph_kernel % 2 == 0:
            raise ValueError("--morph-kernel must be 0 or a positive odd integer")
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        refined = cv2.morphologyEx(refined.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        refined = cv2.morphologyEx(refined.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0

    stats = {
        "original_area": int(mask.sum()),
        "refined_area": int(refined.sum()),
        "removed_area": int((mask & ~refined).sum()),
        "removed_green_area": int(green.sum()),
        "original_components": component_count(mask),
        "refined_components": component_count(refined),
    }
    stats["removed_pct"] = 100.0 * stats["removed_area"] / max(1, stats["original_area"])
    return bgr, mask, refined, stats


def make_overlay(bgr: np.ndarray, mask: np.ndarray, refined: np.ndarray) -> np.ndarray:
    original = bgr.copy()
    original[mask] = (0.55 * original[mask] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)

    refined_view = bgr.copy()
    refined_view[refined] = (0.55 * refined_view[refined] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)

    removed_view = bgr.copy()
    removed = mask & ~refined
    removed_view[refined] = (0.70 * removed_view[refined] + 0.30 * np.array([0, 255, 0])).astype(np.uint8)
    removed_view[removed] = (0.40 * removed_view[removed] + 0.60 * np.array([0, 0, 255])).astype(np.uint8)

    return np.concatenate([original, refined_view, removed_view], axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove green leaf-like pixels from one selected mask.")
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--out-mask", type=Path, required=True)
    parser.add_argument("--out-overlay", type=Path, required=True)
    parser.add_argument("--hue-min", type=int, required=True)
    parser.add_argument("--hue-max", type=int, required=True)
    parser.add_argument("--sat-min", type=int, required=True)
    parser.add_argument("--green-red-margin", type=int, required=True)
    parser.add_argument("--morph-kernel", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bgr, mask, refined, stats = refine_mask(
        args.rgb,
        args.mask,
        args.hue_min,
        args.hue_max,
        args.sat_min,
        args.green_red_margin,
        args.morph_kernel,
    )

    args.out_mask.parent.mkdir(parents=True, exist_ok=True)
    args.out_overlay.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_mask), refined.astype(np.uint8) * 255)
    cv2.imwrite(str(args.out_overlay), make_overlay(bgr, mask, refined))
    print(
        "original_area={original_area} refined_area={refined_area} "
        "removed_area={removed_area} removed_pct={removed_pct:.2f} "
        "original_components={original_components} refined_components={refined_components}".format(**stats)
    )


if __name__ == "__main__":
    main()
