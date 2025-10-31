#!/usr/bin/env python3
"""
Mask R-CNN (Detectron2, R-50-FPN) – hardcoded COCO paths, quiet-ish logs.

- GPU mode: full dataset
- CPU sanity mode: writes tiny subset JSONs and runs few iters
- Prints key AP metrics
"""
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Optional

import detectron2
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.model_zoo import get_config_file, get_checkpoint_url
from detectron2.utils.logger import setup_logger

# =========================
# PATHS
# =========================
PROJECT_ROOT = Path("SegemtationTrainig_v1").resolve()
COCO_ROOT = (PROJECT_ROOT / "converted" / "coco").resolve()

TRAIN_JSON = COCO_ROOT / "train.json"
VAL_JSON = COCO_ROOT / "val.json"
TEST_JSON = COCO_ROOT / "test.json"

TRAIN_IMG_DIR = COCO_ROOT / "images" / "train"
VAL_IMG_DIR = COCO_ROOT / "images" / "val"
TEST_IMG_DIR = COCO_ROOT / "images" / "test"

OUTPUT_DIR = PROJECT_ROOT / "runs" / "maskrcnn_r50fpn"

# =========================
# CONFIG
# =========================
USE_CUDA = False
CPU_SANITY_MAX_IMAGES = 80     # ~80 images == quick check
CPU_MAX_ITER = 300
GPU_MAX_ITER = 20000           # adjust for full training
BATCH_IMS = 4                  # per-batch images (global)
BASE_LR = 0.002
NUM_CLASSES = 1                # "strawberry"

random.seed(42)


# =========================
# HELPERS
# =========================
def _register(name: str, json_path: Path, img_dir: Path) -> None:
    register_coco_instances(name, {}, str(json_path), str(img_dir))


def _make_subset_json(src_json: Path, dst_json: Path, max_images: int) -> None:
    data = json.loads(src_json.read_text(encoding="utf-8"))
    keep_images = data["images"][:max_images]
    keep_ids = {im["id"] for im in keep_images}
    anns = [a for a in data["annotations"] if a["image_id"] in keep_ids]
    cats = data["categories"]
    out = {"images": keep_images, "annotations": anns, "categories": cats}
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    dst_json.write_text(json.dumps(out), encoding="utf-8")


def _ensure_cpu_subsets() -> tuple[Path, Path, Optional[Path]]:
    sub = COCO_ROOT / "_cpu_sanity"
    tr_json = sub / "train.json"
    va_json = sub / "val.json"
    te_json = sub / "test.json"
    if sub.exists():
        shutil.rmtree(sub)
    _make_subset_json(TRAIN_JSON, tr_json, CPU_SANITY_MAX_IMAGES)
    _make_subset_json(VAL_JSON, va_json, max(1, CPU_SANITY_MAX_IMAGES // 2))
    test_path = None
    if TEST_JSON.exists():
        _make_subset_json(TEST_JSON, te_json, CPU_SANITY_MAX_IMAGES)
        test_path = te_json
    return tr_json, va_json, test_path


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        out = output_folder or (Path(cfg.OUTPUT_DIR) / "inference" / dataset_name)
        out.mkdir(parents=True, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, False, output_folder=str(out))


def main() -> None:
    logger = setup_logger()
    print(f"[INFO] Detectron2 {detectron2.__version__}")

    # Register datasets
    if USE_CUDA:
        train_name, val_name, test_name = "straw_train", "straw_val", "straw_test"
        _register(train_name, TRAIN_JSON, TRAIN_IMG_DIR)
        _register(val_name, VAL_JSON, VAL_IMG_DIR)
        if TEST_JSON.exists():
            _register(test_name, TEST_JSON, TEST_IMG_DIR)
        else:
            test_name = None
    else:
        print("[INFO] CPU sanity mode: building tiny COCO subsets…")
        tr_json, va_json, te_json = _ensure_cpu_subsets()
        train_name, val_name, test_name = "straw_train_cpu", "straw_val_cpu", "straw_test_cpu"
        _register(train_name, tr_json, TRAIN_IMG_DIR)  # images from full dir (ok)
        _register(val_name, va_json, VAL_IMG_DIR)
        if te_json is not None:
            _register(test_name, te_json, TEST_IMG_DIR)
        else:
            test_name = None

    # Build config
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)
    cfg.DATALOADER.NUM_WORKERS = 2 if USE_CUDA else 0
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.DEVICE = "cuda" if USE_CUDA else "cpu"
    cfg.SOLVER.IMS_PER_BATCH = BATCH_IMS
    cfg.SOLVER.BASE_LR = BASE_LR
    cfg.SOLVER.MAX_ITER = GPU_MAX_ITER if USE_CUDA else CPU_MAX_ITER
    cfg.SOLVER.STEPS = []  # disable lr step schedule for simplicity
    cfg.SOLVER.CHECKPOINT_PERIOD = 0
    cfg.TEST.EVAL_PERIOD = 0
    cfg.OUTPUT_DIR = str(OUTPUT_DIR)
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    default_setup(cfg, {})

    # Train
    print("[INFO] Training Mask R-CNN…")
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Eval on val
    print("[INFO] Evaluating on val…")
    evaluator = COCOEvaluator(val_name, cfg, False, output_dir=str(Path(cfg.OUTPUT_DIR) / "inference" / "val"))
    val_loader = detectron2.data.build_detection_test_loader(cfg, val_name)
    val_results = inference_on_dataset(trainer.model, val_loader, evaluator)
    print("[VAL] COCO metrics:", val_results)

    # Eval on test (optional)
    if test_name is not None:
        print("[INFO] Evaluating on test…")
        evaluator = COCOEvaluator(test_name, cfg, False, output_dir=str(Path(cfg.OUTPUT_DIR) / "inference" / "test"))
        test_loader = detectron2.data.build_detection_test_loader(cfg, test_name)
        test_results = inference_on_dataset(trainer.model, test_loader, evaluator)
        print("[TEST] COCO metrics:", test_results)

    print("[DONE] Output ->", cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()
