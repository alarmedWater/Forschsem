#!/usr/bin/env python3
"""
Convert StrawDI_Db1 layout to:
  1) COCO JSON for instance segmentation (Detectron2 / YOLACT++)
  2) YOLO-seg labels for Ultralytics YOLOv8-seg

Input layout (do NOT reshuffle):
StrawDI_Db1/
  train/
    img/*.png|jpg
    label/*.png        # instance-ID masks: 0=bg, 1..N=one berry each
  val/
    img/*.png|jpg
    label/*.png
  test/
    img/*.png|jpg
    label/*.png

Outputs:
converted/
  coco/
    instances_train.json
    instances_val.json
    instances_test.json
  yolo/
    train/{images,labels}
    val/{images,labels}
    test/{images,labels}
"""

import os, json, argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTS = [".png", ".jpg", ".jpeg"]
MASK_EXTS = [".png"]  

def find_mask_for_image(mask_dir: Path, stem: str) -> Path | None:
    # Try common mask extensions
    for ext in MASK_EXTS:
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # Fallback: any file starting with stem (rare)
    cands = list(mask_dir.glob(f"{stem}.*"))
    return cands[0] if cands else None

def iter_images(img_dir: Path):
    files = []
    for ext in IMG_EXTS:
        files.extend(sorted(img_dir.glob(f"*{ext}")))
    return files

def contours_from_instance(bin_mask: np.ndarray):
    # bin_mask: uint8 {0,1} for a single instance
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segs = []
    for c in contours:
        if len(c) < 3:
            continue
        c = c.squeeze(1)           # shape (N,2)
        seg = c.flatten().astype(float).tolist()  # [x1,y1,x2,y2,...]
        if len(seg) >= 6:
            segs.append(seg)
    return segs

def mask_to_coco_anns(mask: np.ndarray, image_id: int, start_ann_id: int, category_id: int = 1):
    """
    Converts an instance-ID mask to a list of COCO-style annotations (polygons+box) for one image.
    Returns: (annotations, next_ann_id)
    """
    anns, ann_id = [], start_ann_id
    ids = np.unique(mask)
    ids = ids[ids > 0]  # 0 = background
    for iid in ids:
        m = (mask == iid).astype(np.uint8)
        if int(m.sum()) < 20:  # ignore tiny fragments
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
            "area": float(m.sum())
        })
        ann_id += 1
    return anns, ann_id

def write_yolo_seg(txt_path: Path, mask: np.ndarray, class_id: int = 0):
    """
    Write a YOLO-seg .txt: one line per contour
    Format: class x1 y1 x2 y2 ... (coords normalized to [0,1])
    """
    H, W = mask.shape
    ids = np.unique(mask); ids = ids[ids > 0]
    lines = []
    for iid in ids:
        m = (mask == iid).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if len(c) < 3:
                continue
            c = c.squeeze(1)
            coords = []
            for (x, y) in c:
                coords += [x / W, y / H]
            if len(coords) >= 6:
                lines.append(" ".join([str(class_id)] + [f"{v:.6f}" for v in coords]))
    if lines:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

def process_split(src_root: Path, split: str, out_coco_dir: Path, out_yolo_dir: Path):
    img_dir = src_root / split / "img"
    mask_dir = src_root / split / "label"
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing {img_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing {mask_dir}")

    images, annotations = [], []
    ann_id = 1
    image_id = 1

    # YOLO output dirs
    y_img = out_yolo_dir / split / "images"
    y_lbl = out_yolo_dir / split / "labels"
    y_img.mkdir(parents=True, exist_ok=True)
    y_lbl.mkdir(parents=True, exist_ok=True)

    img_paths = iter_images(img_dir)
    for img_path in tqdm(img_paths, desc=f"{split}: convert"):
        stem = img_path.stem
        mask_path = find_mask_for_image(mask_dir, stem)
        if mask_path is None:
            raise FileNotFoundError(f"No mask for image {img_path.name} in {mask_dir}")

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(mask_path)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        h, w = img.shape[:2]
        images.append({
            "id": image_id,
            "file_name": f"{split}/{img_path.name}",
            "width": w, "height": h
        })
        anns, ann_id = mask_to_coco_anns(mask, image_id, ann_id)
        annotations.extend(anns)

        # For YOLO training, copy image and write label txt
        cv2.imwrite(str(y_img / img_path.name), img)
        write_yolo_seg(y_lbl / f"{stem}.txt", mask, class_id=0)

        image_id += 1

    out_coco_dir.mkdir(parents=True, exist_ok=True)
    with open(out_coco_dir / f"instances_{split}.json", "w") as f:
        json.dump(
            {"images": images, "annotations": annotations, "categories": [{"id":1, "name":"strawberry"}]},
            f
        )
    print(f"Wrote COCO {split}:", out_coco_dir / f"instances_{split}.json")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to StrawDI_Db1 root (has train/val/test with img/label)")
    ap.add_argument("--out_coco", default="converted/coco")
    ap.add_argument("--out_yolo", default="converted/yolo")
    args = ap.parse_args()

    src = Path(args.src)
    out_coco = Path(args.out_coco)
    out_yolo = Path(args.out_yolo)

    for split in ["train", "val", "test"]:
        process_split(src, split, out_coco, out_yolo)

if __name__ == "__main__":
    main()
