from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Copy a pipeline dataset and red-boost only color images.")
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--sat-scale", required=True, type=float)
    p.add_argument("--red-gain", required=True, type=float)
    p.add_argument("--min-red-mask", default=35, type=int, help="Minimum HSV saturation for red/pink mask.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def redboost_bgr(img_bgr: np.ndarray, sat_scale: float, red_gain: float, min_red_mask: int) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    red_mask = ((h <= 15) | (h >= 165)) & (s >= int(min_red_mask))

    out_hsv = hsv.copy()
    out_hsv[:, :, 1][red_mask] = np.clip(
        out_hsv[:, :, 1][red_mask].astype(np.float32) * float(sat_scale),
        0,
        255,
    ).astype(np.uint8)

    out = cv2.cvtColor(out_hsv, cv2.COLOR_HSV2BGR)
    red = out[:, :, 2].astype(np.float32)
    red[red_mask] = np.clip(red[red_mask] * float(red_gain), 0, 255)
    out[:, :, 2] = red.astype(np.uint8)
    return out


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"--src is not a directory: {src}")
    if src == dst:
        raise ValueError("--src and --dst must be different")
    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"--dst exists: {dst} (use --overwrite)")
        shutil.rmtree(dst)

    color_count = 0
    depth_count = 0

    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        out_path = dst / rel
        if path.is_dir():
            out_path.mkdir(parents=True, exist_ok=True)
            continue
        if not path.is_file():
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if path.name.startswith("color_") and path.suffix.lower() == ".png":
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise OSError(f"Failed to read color image: {path}")
            boosted = redboost_bgr(img, args.sat_scale, args.red_gain, args.min_red_mask)
            if not cv2.imwrite(str(out_path), boosted):
                raise OSError(f"Failed to write color image: {out_path}")
            color_count += 1
        elif path.name.startswith("depth_") and path.suffix.lower() == ".png":
            shutil.copy2(path, out_path)
            depth_count += 1
        else:
            raise ValueError(f"Unexpected file in source dataset: {path}")

    print(f"[DONE] wrote {color_count} color images and {depth_count} depth images to {dst}")


if __name__ == "__main__":
    main()
