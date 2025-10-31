#!/usr/bin/env python3
"""
YOLACT++ (ResNet-101) – hardcoded COCO paths, quiet-ish training via subprocess.

- GPU mode: full dataset
- CPU sanity mode: writes tiny subset JSONs and uses few epochs/iters
- Requires 'yolact' package installed (editable) and its configs available.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

# =========================
# PATHS
# =========================
PROJECT_ROOT = Path("SegemtationTrainig_v1").resolve()
COCO_ROOT = (PROJECT_ROOT / "converted" / "coco").resolve()

TRAIN_JSON = COCO_ROOT / "train.json"
VAL_JSON = COCO_ROOT / "val.json"
TEST_JSON = COCO_ROOT / "test.json"  # not used by yolact training; evaluate separately if needed

IMG_ROOT = COCO_ROOT / "images"        # yolact expects a root with 'train' and 'val' subdirs
TRAIN_IMG_DIR = IMG_ROOT / "train"
VAL_IMG_DIR = IMG_ROOT / "val"

OUTPUT_DIR = PROJECT_ROOT / "runs" / "yolactpp_r101"

# =========================
# CONFIG
# =========================
USE_CUDA = False
CPU_SANITY_MAX_IMAGES = 120
CPU_MAX_ITERS = 1200           # tiny run
GPU_MAX_ITERS = 80000          # typical scale (adjust)
BATCH_SIZE = 8                 # global batch size
LEARNING_RATE = 1e-3
CONFIG_NAME = "yolact_plus_resnet101_config"  # from yolact configs

PYTHON = "python"              # interpreter to call yolact runner


# =========================
# HELPERS
# =========================
def _make_subset_json(src_json: Path, dst_json: Path, max_images: int) -> None:
    data = json.loads(src_json.read_text(encoding="utf-8"))
    keep_images = data["images"][:max_images]
    keep_ids = {im["id"] for im in keep_images}
    anns = [a for a in data["annotations"] if a["image_id"] in keep_ids]
    out = {"images": keep_images, "annotations": anns, "categories": data["categories"]}
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    dst_json.write_text(json.dumps(out), encoding="utf-8")


def _ensure_cpu_subsets() -> tuple[Path, Path]:
    sub = COCO_ROOT / "_cpu_sanity_yolact"
    tr_json = sub / "train.json"
    va_json = sub / "val.json"
    if sub.exists():
        shutil.rmtree(sub)
    _make_subset_json(TRAIN_JSON, tr_json, CPU_SANITY_MAX_IMAGES)
    _make_subset_json(VAL_JSON, va_json, max(1, CPU_SANITY_MAX_IMAGES // 2))
    return tr_json, va_json


def _cmd(args: list[str]) -> None:
    print("[CMD]", " ".join(args))
    subprocess.run(args, check=True)


# =========================
# MAIN
# =========================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if USE_CUDA:
        tr_json, va_json = TRAIN_JSON, VAL_JSON
        max_iters = GPU_MAX_ITERS
    else:
        print("[INFO] CPU sanity mode: building tiny COCO subsets…")
        tr_json, va_json = _ensure_cpu_subsets()
        max_iters = CPU_MAX_ITERS

    # YOLACT expects dataset registration via CLI flags:
    #   --coco_train_images, --coco_train_annotation,
    #   --coco_val_images,   --coco_val_annotation
    # Training entrypoint is typically `yolact.py` or `train.py` in the project.
    # Many forks support: `python -m yolact train --config=... --batch=...`
    # Below we call the module; adapt if your yolact fork differs.

    args = [
        PYTHON, "-m", "yolact",
        "train",
        f"--config={CONFIG_NAME}",
        f"--batch_size={BATCH_SIZE}",
        f"--lr={LEARNING_RATE}",
        f"--max_iter={max_iters}",
        f"--save_folder={str(OUTPUT_DIR)}",
        f"--coco_train_images={str(TRAIN_IMG_DIR)}",
        f"--coco_train_annotation={str(tr_json)}",
        f"--coco_val_images={str(VAL_IMG_DIR)}",
        f"--coco_val_annotation={str(va_json)}",
        "--loggi",  # some forks use --logging; ignore if unknown
    ]

    if not USE_CUDA:
        # many yolact forks accept this to force CPU; if not, it will be ignored
        args.append("--cuda=False")

    # Quiet-ish: many forks support --display=False to suppress vis; optional
    args.append("--display=False")

    _cmd(args)
    print("[DONE] Check weights/logs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
