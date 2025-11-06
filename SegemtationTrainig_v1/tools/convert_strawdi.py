#!/usr/bin/env python3
"""
Convert StrawDI_Db1 layout to:
  1) COCO JSON for instance segmentation (Detectron2 / YOLACT++)
     -> images at converted/coco/images/{train,val,test}/
     -> file_name in JSON is "<split>/<filename>"
  2) YOLO-seg labels for Ultralytics YOLOv8-seg
     -> images+labels at converted/yolo/{train,val,test}/{images,labels}
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTS = [".png", ".jpg", ".jpeg"]
MASK_EXTS = [".png"]


def find_mask_for_image(mask_dir: Path, stem: str) -> Path | None:
    for ext in MASK_EXTS:
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    cands = list(mask_dir.glob(f"{stem}.*"))
    return cands[0] if cands else None


def iter_images(img_dir: Path):
    files = []
    for ext in IMG_EXTS:
        files.extend(sorted(img_dir.glob(f"*{ext}")))
    return files


def contours_from_instance(bin_mask: np.ndarray):
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segs = []
    for c in contours:
        if len(c) < 3:
            continue
        c = c.squeeze(1)
        seg = c.flatten().astype(float).tolist()
        if len(seg) >= 6:
            segs.append(seg)
    return segs


def mask_to_coco_anns(mask: np.ndarray, image_id: int, start_ann_id: int, category_id: int = 1):
    anns, ann_id = [], start_ann_id
    ids = np.unique(mask)
    ids = ids[ids > 0]
    for iid in ids:
        m = (mask == iid).astype(np.uint8)
        if int(m.sum()) < 20:
            continue
        segs = contours_from_instance(m)
        if not segs:
            continue
        ys, xs = np.where(m > 0)
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        w, h = int(xmax - xmin + 1), int(ymax - ymin + 1)
        anns.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segs,
            "bbox": [xmin, ymin, w, h],
            "iscrowd": 0,
            "area": float(m.sum()),
        })
        ann_id += 1
    return anns, ann_id


def write_yolo_seg(txt_path: Path, mask: np.ndarray, class_id: int = 0):
    """Write YOLOv8-seg polygons (normalized)."""
    H, W = mask.shape
    ids = np.unique(mask)[1:]  # skip background
    lines = []
    for iid in ids:
        m = (mask == iid).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if len(c) < 3:
                continue
            c = c.squeeze(1)
            coords = [f"{x / W:.6f} {y / H:.6f}" for (x, y) in c]
            if len(coords) >= 3:
                lines.append(f"{class_id} " + " ".join(coords))
    if lines:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text("\n".join(lines), encoding="utf-8")


def _place_image(src: Path, dst: Path, mode: str = "hardlink"):
    """Create image in COCO dir using hardlink/symlink/copy."""
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


def process_split(src_root: Path, split: str, out_coco_dir: Path, out_yolo_dir: Path, link_mode: str):
    img_dir = src_root / split / "img"
    mask_dir = src_root / split / "label"
    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Missing directories in {src_root}/{split}")

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

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        h, w = mask.shape[:2]

        # COCO bookkeeping
        images.append({
            "id": image_id,
            "file_name": f"{split}/{img_path.name}",
            "width": w,
            "height": h,
        })
        anns, ann_id = mask_to_coco_anns(mask, image_id, ann_id)
        annotations.extend(anns)

        # Place COCO image
        _place_image(img_path, coco_img_dir / img_path.name, mode=link_mode)
        # YOLO export
        _place_image(img_path, yolo_img_dir / img_path.name, mode="copy")
        write_yolo_seg(yolo_lbl_dir / f"{stem}.txt", mask)

        image_id += 1

    # Save COCO JSON
    out_coco_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_coco_dir / f"instances_{split}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "strawberry"}],
        }, f)
    print(f"[COCO] Wrote {json_path}")
    print(f"[COCO] Images @ {coco_img_dir}")
    print(f"[YOLO] Images @ {yolo_img_dir}")
    print(f"[YOLO] Labels @ {yolo_lbl_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to StrawDI_Db1 root (train/val/test with img/label)")
    ap.add_argument("--out_coco", default="converted/coco", help="Output dir for COCO dataset")
    ap.add_argument("--out_yolo", default="converted/yolo", help="Output dir for YOLOv8 dataset")
    ap.add_argument("--link_mode", choices=["copy", "hardlink", "symlink"], default="hardlink")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out_coco = Path(args.out_coco).resolve()
    out_yolo = Path(args.out_yolo).resolve()

    for split in ["train", "val", "test"]:
        process_split(src, split, out_coco, out_yolo, link_mode=args.link_mode)


if __name__ == "__main__":
    main()
