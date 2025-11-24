#!/usr/bin/env python3
"""
Convert StrawDI_Db1 layout to:
  1) COCO JSON for instance segmentation
     -> images at <out_coco>/images/{train,val,test}/
     -> file_name in JSON is "<split>/<filename>"
     -> JSON at <out_coco>/instances_<split>.json
  2) YOLOv8-seg labels (polygon format)
     -> <out_yolo>/{train,val,test}/{images,labels}
     -> optional: writes data.yaml into <out_yolo>

Mask handling:
- Supports single-channel instance-id maps (0 = bg) and color-coded masks.
- For color masks, builds a unique 24-bit id: id = R + (G<<8) + (B<<16).

Usage example:
  python convert_strawdi.py \
      --src /path/StrawDI_Db1 \
      --out_coco converted/coco \
      --out_yolo converted/yolo \
      --min_area 20 \
      --class_name strawberry \
      --poly_simplify 0.002 \
      --write_empty_labels \
      --write_data_yaml
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
MASK_EXTS = [".png", ".bmp", ".tif", ".tiff"]


def find_mask_for_image(mask_dir: Path, stem: str) -> Path | None:
    # exact name first
    for ext in MASK_EXTS:
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # fallback: any stem.*
    cands = list(mask_dir.glob(f"{stem}.*"))
    return cands[0] if cands else None


def iter_images(img_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMG_EXTS:
        files.extend(sorted(img_dir.glob(f"*{ext}")))
    return files


def read_instance_mask(mask_path: Path) -> np.ndarray:
    """Read mask as instance-id map (int32); 0 = background.
    Supports single-channel or color-coded instance masks.
    """
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    if m.ndim == 2:
        return m.astype(np.int32)
    if m.ndim == 3:
        # Build 24-bit id from BGR
        b, g, r = m[..., 0].astype(np.int32), m[..., 1].astype(np.int32), m[..., 2].astype(np.int32)
        return (r + (g << 8) + (b << 16)).astype(np.int32)
    raise ValueError(f"Unsupported mask shape: {m.shape} for {mask_path}")


def contours_from_binary(bin_mask: np.ndarray, simplify_eps_frac: float | None = None) -> List[np.ndarray]:
    """Return list of Nx2 polygon points (float). Optional simplification by eps = frac * arcLength."""
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[np.ndarray] = []
    for c in contours:
        if c is None or len(c) < 3:
            continue
        c = c.squeeze(1)
        if simplify_eps_frac is not None and simplify_eps_frac > 0:
            eps = simplify_eps_frac * cv2.arcLength(c, True)
            c = cv2.approxPolyDP(c, eps, True).squeeze(1)
            if c.ndim != 2 or c.shape[0] < 3:
                continue
        polys.append(c.astype(np.float32))
    return polys


def mask_to_coco_anns(mask: np.ndarray, image_id: int, start_ann_id: int,
                      category_id: int = 1, min_area: int = 20,
                      simplify_eps_frac: float | None = None) -> Tuple[List[dict], int]:
    anns, ann_id = [], start_ann_id
    ids = np.unique(mask)
    ids = ids[ids > 0]  # skip background
    for iid in ids:
        m = (mask == iid).astype(np.uint8)
        area = int(m.sum())
        if area < min_area:
            continue
        polys = contours_from_binary(m, simplify_eps_frac)
        if not polys:
            continue
        ys, xs = np.where(m > 0)
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        w, h = int(xmax - xmin + 1), int(ymax - ymin + 1)

        segs = [p.flatten().astype(float).tolist() for p in polys if p.shape[0] >= 3]
        if not segs:
            continue

        anns.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segs,
            "bbox": [xmin, ymin, w, h],
            "iscrowd": 0,
            "area": float(area),
        })
        ann_id += 1
    return anns, ann_id


def write_yolo_seg(txt_path: Path, mask: np.ndarray, class_id: int = 0,
                   min_area: int = 20, simplify_eps_frac: float | None = None,
                   write_empty: bool = True) -> None:
    """Write YOLOv8-seg polygons (normalized). One line per polygon."""
    H, W = mask.shape
    ids = np.unique(mask)
    ids = ids[ids > 0]  # 0 = background

    lines: List[str] = []
    for iid in ids:
        m = (mask == iid).astype(np.uint8)
        if int(m.sum()) < min_area:
            continue
        polys = contours_from_binary(m, simplify_eps_frac)
        for p in polys:
            if p.shape[0] < 3:
                continue
            coords = [f"{x / W:.6f} {y / H:.6f}" for (x, y) in p]
            lines.append(f"{class_id} " + " ".join(coords))

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    if lines or write_empty:
        txt_path.write_text("\n".join(lines), encoding="utf-8")


def _place_image(src: Path, dst: Path, mode: str = "hardlink") -> None:
    """Create image in target dir using hardlink/symlink/copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        if mode == "hardlink":
            os.link(src, dst)
        elif mode == "symlink":
            os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def process_split(src_root: Path, split: str, out_coco_dir: Path, out_yolo_dir: Path,
                  link_mode: str, class_name: str, min_area: int,
                  simplify_eps_frac: float | None, write_empty_labels: bool) -> None:
    img_dir = src_root / split / "img"
    mask_dir = src_root / split / "label"
    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Missing directories in {src_root}/{split} (need img/ and label/)")

    images, annotations = [], []
    ann_id = 1
    image_id = 1

    # Prepare output directories
    coco_img_dir = out_coco_dir / "images" / split
    coco_img_dir.mkdir(parents=True, exist_ok=True)
    yolo_img_dir = out_yolo_dir / split / "images"
    yolo_lbl_dir = out_yolo_dir / split / "labels"
    yolo_img_dir.mkdir(parents=True, exist_ok=True)
    yolo_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_paths = iter_images(img_dir)
    for img_path in tqdm(img_paths, desc=f"{split}: converting"):
        stem = img_path.stem
        mask_path = find_mask_for_image(mask_dir, stem)
        if mask_path is None:
            raise FileNotFoundError(f"No mask for {img_path.name}")

        # read image sizes from mask to be consistent
        mask = read_instance_mask(mask_path)
        h, w = mask.shape[:2]

        # COCO bookkeeping
        images.append({
            "id": image_id,
            "file_name": f"{split}/{img_path.name}",
            "width": w,
            "height": h,
        })
        anns, ann_id = mask_to_coco_anns(mask, image_id, ann_id,
                                         category_id=1, min_area=min_area,
                                         simplify_eps_frac=simplify_eps_frac)
        annotations.extend(anns)

        # Place image (COCO & YOLO share same source image)
        _place_image(img_path, coco_img_dir / img_path.name, mode=link_mode)
        _place_image(img_path, yolo_img_dir / img_path.name, mode="copy")

        # YOLO labels
        write_yolo_seg(yolo_lbl_dir / f"{stem}.txt", mask, class_id=0,
                       min_area=min_area, simplify_eps_frac=simplify_eps_frac,
                       write_empty=write_empty_labels)

        image_id += 1

    # Save COCO JSON
    out_coco_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_coco_dir / f"instances_{split}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": class_name}],
        }, f)
    print(f"[COCO] Wrote {json_path}")
    print(f"[COLO] Images @ {coco_img_dir}")
    print(f"[YOLO] Images @ {yolo_img_dir}")
    print(f"[YOLO] Labels @ {yolo_lbl_dir}")


def maybe_write_data_yaml(out_yolo_dir: Path, class_name: str) -> Path:
    data_yaml = out_yolo_dir / "data.yaml"
    content = {
        "path": str(out_yolo_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "names": [class_name],
    }
    out_yolo_dir.mkdir(parents=True, exist_ok=True)
    import yaml
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False, allow_unicode=True)
    print(f"[YOLO] Wrote {data_yaml}")
    return data_yaml


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="Path to StrawDI_Db1 root (contains train/ val/ test/ with img/ and label/)")
    ap.add_argument("--out_coco", default="converted/coco", help="Output dir for COCO dataset")
    ap.add_argument("--out_yolo", default="converted/yolo", help="Output dir for YOLOv8 dataset")
    ap.add_argument("--link_mode", choices=["copy", "hardlink", "symlink"], default="hardlink")
    ap.add_argument("--class_name", default="strawberry")
    ap.add_argument("--min_area", type=int, default=20, help="Ignore instances smaller than this area (px)")
    ap.add_argument("--poly_simplify", type=float, default=0.002,
                    help="Douglas-Peucker eps as fraction of contour perimeter (0 to disable)")
    ap.add_argument("--write_empty_labels", action="store_true",
                    help="Write empty YOLO .txt for images with no objects")
    ap.add_argument("--write_data_yaml", action="store_true",
                    help="Also write <out_yolo>/data.yaml")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src).resolve()
    out_coco = Path(args.out_coco).resolve()
    out_yolo = Path(args.out_yolo).resolve()

    if not (src / "train").exists():
        raise FileNotFoundError(f"{src} must contain 'train', 'val', 'test' subfolders")

    for split in ["train", "val", "test"]:
        process_split(
            src_root=src, split=split,
            out_coco_dir=out_coco, out_yolo_dir=out_yolo,
            link_mode=args.link_mode,
            class_name=args.class_name,
            min_area=args.min_area,
            simplify_eps_frac=(None if args.poly_simplify <= 0 else args.poly_simplify),
            write_empty_labels=args.write_empty_labels
        )

    if args.write_data_yaml:
        maybe_write_data_yaml(out_yolo, args.class_name)


if __name__ == "__main__":
    main()
