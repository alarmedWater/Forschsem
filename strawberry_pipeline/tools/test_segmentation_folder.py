from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from strawberry_py.config import load_config
from strawberry_py.pipeline.stages.segmentation import YoloV8Segmenter


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--image_dir", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config.resolve())
    segmenter = YoloV8Segmenter.from_cfg(cfg.segmentation)

    image_dir = args.image_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    for image_path in image_paths:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] Skip unreadable file: {image_path}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = segmenter(rgb)

        stem = image_path.stem

        cv2.imwrite(str(out_dir / f"{stem}_label_vis.png"), result.label_vis)
        cv2.imwrite(str(out_dir / f"{stem}_label_u16.png"), result.label)

        if result.overlay_rgb is not None:
            overlay_bgr = cv2.cvtColor(result.overlay_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay_bgr)

        print(f"[OK] {image_path.name}")

    print(f"[DONE] Results written to: {out_dir}")


if __name__ == "__main__":
    main()

