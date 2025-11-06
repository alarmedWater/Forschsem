#!/usr/bin/env python3
from __future__ import annotations
import os
import glob
import shutil
from pathlib import Path
from typing import Iterable, Optional

import yaml
import torch
from ultralytics import YOLO

# ================== GPU-ONLY SETTINGS ==================
GPU_DEVICE = "0"          # Erste GPU
FULL_EPOCHS  = 100
FULL_IMGSZ   = 640
FULL_BATCH   = 16
FULL_WORKERS = 8

SEED = 42
PATIENCE = 50
COS_LR = False
CLASS_NAMES: Optional[list[str]] = ["strawberry"]
# =======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = (PROJECT_ROOT / "converted" / "yolo").resolve()
TRAIN_IMG = DATA_ROOT / "train" / "images"
TRAIN_LBL = DATA_ROOT / "train" / "labels"
VAL_IMG   = DATA_ROOT / "val" / "images"
VAL_LBL   = DATA_ROOT / "val" / "labels"
TEST_IMG  = DATA_ROOT / "test" / "images"
TEST_LBL  = DATA_ROOT / "test" / "labels"
DATA_YAML = DATA_ROOT / "data.yaml"

MODEL_WEIGHTS = "yolov8s.pt"
PROJECT = str(PROJECT_ROOT / "runs" / "detect")
RUN_NAME = "train_yolov8s_gpu"


def _infer_names_from_labels(lbl_dir: Path) -> Optional[list[str]]:
    if not lbl_dir.exists():
        return None
    label_files = glob.glob(str(lbl_dir / "**/*.txt"), recursive=True)
    max_id = -1
    for lf in label_files:
        with open(lf, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                cid = int(line.split()[0])
                max_id = max(max_id, cid)
    return [f"class_{i}" for i in range(max_id + 1)] if max_id >= 0 else None


def _count_files(paths: Iterable[Path], patterns=("*.jpg", "*.jpeg", "*.png", "*.bmp")) -> int:
    return sum(sum(1 for _ in p.rglob(ext)) for p in paths for ext in patterns)


def _print_split_summary() -> None:
    print("\n[DATASET] Summary")
    print(f"  Train images: {_count_files([TRAIN_IMG])} @ {TRAIN_IMG}")
    print(f"  Val images:   {_count_files([VAL_IMG])} @ {VAL_IMG}")
    ti = _count_files([TEST_IMG]) if TEST_IMG.exists() else 0
    print(f"  Test images:  {ti} @ {TEST_IMG if TEST_IMG.exists() else 'N/A'}\n")


def _ensure_data_yaml(path_train: Path, path_val: Path, path_test: Optional[Path]) -> Path:
    names = CLASS_NAMES or _infer_names_from_labels(TRAIN_LBL) or ["class_0"]
    data = {
        "path": str(DATA_ROOT),
        "train": str(path_train),
        "val": str(path_val),
        "names": names,
    }
    if path_test is not None:
        data["test"] = str(path_test)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"[INFO] Wrote data.yaml -> {DATA_YAML}")
    return DATA_YAML


def main() -> None:
    # --- Sicherstellen, dass CUDA verfügbar ist (GPU-ONLY) ---
    if not torch.cuda.is_available():
        raise SystemExit(
            "[FATAL] CUDA ist nicht verfügbar, aber dieses Skript ist GPU-only.\n"
            "Bitte stelle sicher, dass du im 'yolo8-seg' Env bist und eine NVIDIA-GPU mit passenden Treibern hast."
        )

    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    print(f"[INFO] Using GPU device: {GPU_DEVICE}")
    print(f"[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] GPU name: {torch.cuda.get_device_name(0)}")

    # --- Daten prüfen ---
    for p in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
        assert p.exists(), f"Missing: {p}"

    _print_split_summary()

    # Volles Dataset für GPU-Training
    tr, va = TRAIN_IMG, VAL_IMG
    te = TEST_IMG if TEST_IMG.exists() and TEST_LBL.exists() else None

    workers = FULL_WORKERS
    epochs  = FULL_EPOCHS
    batch   = FULL_BATCH
    imgsz   = FULL_IMGSZ
    device  = GPU_DEVICE   # "0" = erste GPU
    do_val  = True

    data_yaml = _ensure_data_yaml(tr, va, te)

    # --- Modell laden ---
    model = YOLO(MODEL_WEIGHTS)

    # --- Training ---
    print("[INFO] Starting GPU training on full dataset…")
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=PROJECT,
        name=RUN_NAME,
        seed=SEED,
        patience=PATIENCE,
        cos_lr=COS_LR,
        pretrained=True,
        verbose=False,
        plots=False,
        val=do_val,
    )

    # --- Validierung ---
    if do_val:
        print("[INFO] Validating on val split…")
        best = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
        model_eval = YOLO(str(best)) if best.exists() else model
        val_metrics = model_eval.val(
            data=str(data_yaml),
            imgsz=imgsz,
            device=device,
            workers=workers,
            verbose=False,
            plots=False,
        )
        try:
            from pprint import pprint
            pprint(val_metrics.results_dict)
        except Exception:
            pass

    # --- ONNX Export ---
    print("[INFO] Exporting ONNX…")
    try:
        model.export(format="onnx", verbose=False)
    except Exception as e:
        print(f"[WARN] Export failed: {e}")

    print("[DONE] GPU training pipeline finished.")


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    main()
