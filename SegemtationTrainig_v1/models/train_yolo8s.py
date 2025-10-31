#!/usr/bin/env python3
from __future__ import annotations
import os, glob, shutil
from pathlib import Path
from typing import Iterable, Optional
import yaml
from ultralytics import YOLO

# ================== SWITCH ==================
USE_CUDA = False      # False = schneller CPU-Check, True = richtig trainieren
GPU_DEVICE = "0"
# ============================================

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
RUN_NAME = "train_yolov8s"

# CPU-SANITY
SANITY_MAX_IMAGES  = 8
SANITY_EPOCHS      = 1
SANITY_IMGSZ       = 320
SANITY_BATCH       = 2
SANITY_WORKERS     = 0

# GPU-FULL
FULL_EPOCHS  = 100
FULL_IMGSZ   = 640
FULL_BATCH   = 16
FULL_WORKERS = 8

SEED = 42
PATIENCE = 50
COS_LR = False
CLASS_NAMES: Optional[list[str]] = ["strawberry"]


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


def _count_files(paths: Iterable[Path], patterns=("*.jpg","*.jpeg","*.png","*.bmp")) -> int:
    return sum(sum(1 for _ in p.rglob(ext)) for p in paths for ext in patterns)


def _print_split_summary() -> None:
    print("\n[DATASET] Summary")
    print(f"  Train images: {_count_files([TRAIN_IMG])} @ {TRAIN_IMG}")
    print(f"  Val images:   {_count_files([VAL_IMG])} @ {VAL_IMG}")
    ti = _count_files([TEST_IMG]) if TEST_IMG.exists() else 0
    print(f"  Test images:  {ti} @ {TEST_IMG if TEST_IMG.exists() else 'N/A'}\n")


def _ensure_data_yaml(path_train: Path, path_val: Path, path_test: Optional[Path]) -> Path:
    names = CLASS_NAMES or _infer_names_from_labels(TRAIN_LBL) or ["class_0"]
    data = {"path": str(DATA_ROOT), "train": str(path_train), "val": str(path_val), "names": names}
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


def _make_subset(split_img: Path, split_lbl: Path, out_img: Path, out_lbl: Path, n: int) -> int:
    exts = ("*.jpg","*.jpeg","*.png","*.bmp")
    imgs = [p for ext in exts for p in sorted(split_img.rglob(ext))]
    count = 0
    for img in imgs:
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
    subset = DATA_ROOT / "_cpu_sanity"
    s_tr_i, s_tr_l = subset/"train/images", subset/"train/labels"
    s_va_i, s_va_l = subset/"val/images",   subset/"val/labels"
    s_te_i, s_te_l = subset/"test/images",  subset/"test/labels"
    if subset.exists():
        shutil.rmtree(subset)
    n_tr = _make_subset(TRAIN_IMG, TRAIN_LBL, s_tr_i, s_tr_l, SANITY_MAX_IMAGES)
    n_va = _make_subset(VAL_IMG,   VAL_LBL, s_va_i, s_va_l, min(SANITY_MAX_IMAGES, 25))
    has_test = TEST_IMG.exists() and TEST_LBL.exists()
    n_te = _make_subset(TEST_IMG, TEST_LBL, s_te_i, s_te_l, min(SANITY_MAX_IMAGES, 50)) if has_test else 0
    print("[SANITY] Subset created:", f"train={n_tr}, val={n_va}, test={n_te if has_test else 0}")
    return s_tr_i.resolve(), s_va_i.resolve(), (s_te_i.resolve() if has_test else None)


def main() -> None:
    for p in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
        assert p.exists(), f"Missing: {p}"
    _print_split_summary()

    if USE_CUDA:
        print("[INFO] GPU mode: full dataset.")
        tr, va = TRAIN_IMG, VAL_IMG
        te = TEST_IMG if TEST_IMG.exists() and TEST_LBL.exists() else None
        workers = FULL_WORKERS
        epochs  = FULL_EPOCHS
        batch   = FULL_BATCH
        imgsz   = FULL_IMGSZ
        device  = GPU_DEVICE
        do_val  = True
    else:
        print("[INFO] CPU sanity mode: building small subset...")
        tr, va, te = _prepare_cpu_sanity_subset()
        workers = SANITY_WORKERS
        epochs  = SANITY_EPOCHS
        batch   = SANITY_BATCH
        imgsz   = SANITY_IMGSZ
        device  = "cpu"
        do_val  = False   # <- wichtig, sonst hängt er an torchvision.nms

    data_yaml = _ensure_data_yaml(tr, va, te)
    model = YOLO(MODEL_WEIGHTS)

    print("[INFO] Starting training…")
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

    print("[INFO] Exporting ONNX…")
    try:
        model.export(format="onnx", verbose=False)
    except Exception as e:
        print(f"[WARN] Export failed: {e}")
    print("[DONE] Pipeline finished.")


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    main()
