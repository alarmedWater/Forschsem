#!/usr/bin/env python3
"""
YOLO (Ultralytics) training & testing with hardcoded paths — quiet console.

- GPU mode: trains/evaluates on full dataset
- CPU sanity mode: copies a ~50-image subset and trains briefly (3 epochs)
- Writes/updates data.yaml with absolute paths
- Prints compact metrics (no huge arrays)
- Suppresses plotting and verbose logs
- Exports ONNX at the end

"""
from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

import yaml
from ultralytics import YOLO


# =========================
# HARD-CODED PATHS & PARAMS
# =========================
PROJECT_ROOT = Path("SegemtationTrainig_v1").resolve()
DATA_ROOT = (PROJECT_ROOT / "converted" / "yolo").resolve()

TRAIN_IMG = DATA_ROOT / "train" / "images"
TRAIN_LBL = DATA_ROOT / "train" / "labels"
VAL_IMG = DATA_ROOT / "val" / "images"
VAL_LBL = DATA_ROOT / "val" / "labels"
TEST_IMG = DATA_ROOT / "test" / "images"
TEST_LBL = DATA_ROOT / "test" / "labels"

DATA_YAML = DATA_ROOT / "data.yaml"

# Model & training configuration
MODEL_WEIGHTS = "yolov8n.pt"  # switch to 'yolov8s.pt' later if you like
EPOCHS = 100
IMGSZ = 640
BATCH = 16
WORKERS = 8
SEED = 42
PROJECT = str(PROJECT_ROOT / "runs" / "detect")
RUN_NAME = "train_hardcoded"
PATIENCE = 50
COS_LR = False

# Device selection
USE_CUDA = False              # True = GPU full run, False = CPU sanity subset
GPU_DEVICE = "0"
DEVICE = GPU_DEVICE if USE_CUDA else "cpu"

# CPU sanity subset
SANITY_MAX_IMAGES = 50
SANITY_EPOCHS = 3
SANITY_BATCH = 8
SANITY_WORKERS = 0            # 0 = no multiprocessing)

# Optional class names (set to None to infer count from labels)
CLASS_NAMES: Optional[list[str]] = ["strawberry"]


# =========================
# Helpers
# =========================
def _infer_names_from_labels(lbl_dir: Path) -> Optional[list[str]]:
    """Infer class count from YOLO *.txt labels if CLASS_NAMES is None."""
    if not lbl_dir.exists():
        return None
    label_files = glob.glob(str(lbl_dir / "**/*.txt"), recursive=True)
    max_id = -1
    for lf in label_files:
        try:
            with open(lf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cid = int(line.split()[0])
                    if cid > max_id:
                        max_id = cid
        except Exception:
            pass
    if max_id >= 0:
        return [f"class_{i}" for i in range(max_id + 1)]
    return None


def _count_files(paths: Iterable[Path],
                 patterns: tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png", "*.bmp")) -> int:
    """Count image files under paths matching given patterns."""
    total = 0
    for p in paths:
        for ext in patterns:
            total += sum(1 for _ in p.rglob(ext))
    return total


def _print_split_summary() -> None:
    """Print a compact dataset summary to console."""
    train_imgs = _count_files([TRAIN_IMG])
    val_imgs = _count_files([VAL_IMG])
    test_imgs = _count_files([TEST_IMG]) if TEST_IMG.exists() else 0
    print("\n[DATASET] Summary")
    print(f"  Train images: {train_imgs} @ {TRAIN_IMG}")
    print(f"  Val images:   {val_imgs} @ {VAL_IMG}")
    print(f"  Test images:  {test_imgs} @ {TEST_IMG if TEST_IMG.exists() else 'N/A'}\n")


def _print_metrics(metrics) -> None:
    """Print compact metrics (mAP etc.) without dumping arrays."""
    try:
        box = metrics.box  # type: ignore[attr-defined]
        m_map = getattr(box, "map", None)
        m_map50 = getattr(box, "map50", None)
        m_map75 = getattr(box, "map75", None)
        print("[METRICS] Detection (Boxes):")
        if m_map is not None:
            print(f"  mAP50-95: {m_map:.4f}")
        if m_map50 is not None:
            print(f"  mAP50:    {m_map50:.4f}")
        if m_map75 is not None:
            print(f"  mAP75:    {m_map75:.4f}")
        # Per-class summary (short)
        maps = getattr(box, "maps", None)
        names = getattr(metrics, "names", None)
        if maps is not None and names is not None and len(maps) > 0:
            # print only first few to keep console tidy
            k = min(5, len(maps))
            print("  Per-class mAP50-95 (first few):")
            for i in range(k):
                v = maps[i]
                if v is None:
                    continue
                cname = names[i] if i < len(names) else f"class_{i}"
                print(f"    - {cname}: {v:.4f}")
    except Exception as exc:
        print(f"[WARN] Could not format metrics: {exc}")
        try:
            print("  Fallback:", getattr(metrics, "results_dict", "<unavailable>"))
        except Exception:
            pass


def _ensure_data_yaml(path_train: Path, path_val: Path, path_test: Optional[Path]) -> Path:
    """Create or update a minimal data.yaml pointing to provided splits."""
    names = CLASS_NAMES
    if names is None:
        inferred = _infer_names_from_labels(TRAIN_LBL)
        names = inferred if inferred else ["class_0"]
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


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _make_subset(split_img: Path, split_lbl: Path,
                 out_img: Path, out_lbl: Path, n: int) -> int:
    """Copy up to n matched image/label pairs to a subset folder."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    img_files: list[Path] = []
    for ext in exts:
        img_files.extend(sorted(split_img.rglob(ext)))

    count = 0
    for img in img_files:
        if count >= n:
            break
        rel = img.relative_to(split_img)
        lbl = (split_lbl / rel).with_suffix(".txt")
        if not lbl.exists():
            continue
        _copy(img.resolve(), (out_img / rel.name).resolve())
        _copy(lbl.resolve(), (out_lbl / rel.with_suffix(".txt").name).resolve())
        count += 1
    return count


def _prepare_cpu_sanity_subset() -> tuple[Path, Path, Optional[Path]]:
    """Build a small subset dataset for CPU sanity testing."""
    subset_root = DATA_ROOT / "_cpu_sanity"
    s_train_img = subset_root / "train" / "images"
    s_train_lbl = subset_root / "train" / "labels"
    s_val_img = subset_root / "val" / "images"
    s_val_lbl = subset_root / "val" / "labels"
    s_test_img = subset_root / "test" / "images"
    s_test_lbl = subset_root / "test" / "labels"

    if subset_root.exists():
        shutil.rmtree(subset_root)

    n_tr = _make_subset(TRAIN_IMG, TRAIN_LBL, s_train_img, s_train_lbl, SANITY_MAX_IMAGES)
    n_va = _make_subset(VAL_IMG, VAL_LBL, s_val_img, s_val_lbl, min(SANITY_MAX_IMAGES, 25))
    has_test = TEST_IMG.exists() and TEST_LBL.exists()
    n_te = 0
    if has_test:
        n_te = _make_subset(TEST_IMG, TEST_LBL, s_test_img, s_test_lbl, min(SANITY_MAX_IMAGES, 50))

    print("[SANITY] Subset created:", f"train={n_tr}, val={n_va}, test={n_te if has_test else 0}")
    return s_train_img.resolve(), s_val_img.resolve(), (s_test_img.resolve() if has_test else None)


# =========================
# Main
# =========================
def main() -> None:
    # Require train & val
    assert TRAIN_IMG.exists(), f"Missing: {TRAIN_IMG}"
    assert TRAIN_LBL.exists(), f"Missing: {TRAIN_LBL}"
    assert VAL_IMG.exists(), f"Missing: {VAL_IMG}"
    assert VAL_LBL.exists(), f"Missing: {VAL_LBL}"

    _print_split_summary()

    # Decide dataset
    if USE_CUDA:
        print("[INFO] GPU mode: full dataset.")
        train_path = TRAIN_IMG
        val_path = VAL_IMG
        test_path = TEST_IMG if TEST_IMG.exists() and TEST_LBL.exists() else None
        workers = WORKERS
        epochs = EPOCHS
        batch = BATCH
        device = GPU_DEVICE
    else:
        print("[INFO] CPU sanity mode: building small subset...")
        s_train, s_val, s_test = _prepare_cpu_sanity_subset()
        train_path, val_path, test_path = s_train, s_val, s_test
        workers = SANITY_WORKERS
        epochs = SANITY_EPOCHS
        batch = SANITY_BATCH
        device = "cpu"

    data_yaml_path = _ensure_data_yaml(train_path, val_path, test_path)

    # Build/Load model
    model = YOLO(MODEL_WEIGHTS)

    # Train (quiet)
    print("[INFO] Starting training…")
    model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=IMGSZ,
        batch=batch,
        device=device,
        workers=workers,
        project=PROJECT,
        name=RUN_NAME,
        seed=SEED,
        patience=PATIENCE,
        cos_lr=COS_LR,
        pretrained=True,
        verbose=False,   # keep console tidy
        plots=False      # do not generate PR/Confusion plots
    )

    # Validate (val split) — quiet
    print("[INFO] Validating on val split…")
    best = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
    model_eval = YOLO(str(best)) if best.exists() else model
    val_metrics = model_eval.val(
        data=str(data_yaml_path),
        imgsz=IMGSZ,
        device=device,
        workers=workers,
        verbose=False,
        plots=False
    )
    _print_metrics(val_metrics)

    # Test (if configured) — quiet
    if test_path is not None:
        print("[INFO] Evaluating on test split…")
        test_metrics = model_eval.val(
            data=str(data_yaml_path),
            imgsz=IMGSZ,
            device=device,
            workers=workers,
            split="test",
            verbose=False,
            plots=False
        )
        _print_metrics(test_metrics)

    # Export ONNX
    print("[INFO] Exporting ONNX…")
    try:
        model_eval.export(format="onnx", verbose=False)
    except Exception as exc:
        print(f"[WARN] Export failed: {exc}")

    print("[DONE] Pipeline finished.")


if __name__ == "__main__":
    # Make cuBLAS behavior deterministic where applicable
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    main()
