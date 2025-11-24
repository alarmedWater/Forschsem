#!/usr/bin/env python3
"""
Train YOLOv8-seg on the converted YOLO dataset produced by convert_strawdi.py.

- Expects:
    <yolo_root>/
      ├─ train/images, train/labels
      ├─ val/images,   val/labels
      └─ test/images,  test/labels (optional)
    and a data.yaml at <yolo_root>/data.yaml
- Uses yolov8s-seg.pt (segmentation backbone)
- Exports ONNX after training

Example:
  python train_yolov8.py --yolo_root converted/yolo --epochs 100 --imgsz 640
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

from ultralytics import YOLO
import torch
import yaml


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_root", type=str, default="converted/yolo",
                    help="Path to YOLO dataset root (contains data.yaml)")
    ap.add_argument("--model", type=str, default="yolov8s-seg.pt",
                    help="Ultralytics segmentation model (.pt)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", type=str, default="runs/segment")
    ap.add_argument("--name", type=str, default="strawdi_y8s")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--cos_lr", action="store_true")
    ap.add_argument("--val", action="store_true", help="Run val() after training")
    return ap.parse_args()


def ensure_data_yaml(yolo_root: Path) -> Path:
    data_yaml = yolo_root / "data.yaml"
    if not data_yaml.exists():
        # fallback minimal data.yaml
        content = {
            "path": str(yolo_root.resolve()),
            "train": "train/images",
            "val":   "val/images",
            "test":  "test/images",
            "names": ["strawberry"],
        }
        with open(data_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(content, f, sort_keys=False, allow_unicode=True)
        print(f"[INFO] Wrote fallback data.yaml at {data_yaml}")
    return data_yaml


def pick_device() -> str:
    if torch.cuda.is_available():
        print(f"[INFO] CUDA available ✓  Using GPU: {torch.cuda.get_device_name(0)}")
        return "0"  # first GPU
    print("[INFO] CUDA not available -> using CPU")
    return "cpu"


def main() -> None:
    args = parse_args()
    yolo_root = Path(args.yolo_root).resolve()

    # basic checks
    for sub in ["train/images", "train/labels", "val/images", "val/labels"]:
        if not (yolo_root / sub).exists():
            print(f"[FATAL] Missing folder: {yolo_root/sub}")
            sys.exit(1)

    data_yaml = ensure_data_yaml(yolo_root)
    device = pick_device()

    # Load segmentation model
    model = YOLO(args.model)  # e.g. yolov8s-seg.pt

    # Train
    print("[INFO] Starting segmentation training…")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        cos_lr=args.cos_lr,
        pretrained=True,
        verbose=False,
        plots=False,
        val=True,  # keep validation during training
    )

    # Optional explicit val with best weights
    if args.val:
        best = Path(args.project) / args.name / "weights" / "best.pt"
        m_eval = YOLO(str(best)) if best.exists() else model
        print("[INFO] Running .val() on validation split…")
        m_eval.val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            device=device,
            workers=args.workers,
            verbose=False,
            plots=False,
        )

    # Export ONNX (will end up next to best.pt by default)
    print("[INFO] Exporting ONNX…")
    try:
        (Path(args.project) / args.name).mkdir(parents=True, exist_ok=True)
        model.export(format="onnx", imgsz=args.imgsz, dynamic=False, verbose=False)
        print("[INFO] ONNX export done. Look in the run's weights/ folder.")
    except Exception as e:
        print(f"[WARN] ONNX export failed: {e}")


if __name__ == "__main__":
    main()
