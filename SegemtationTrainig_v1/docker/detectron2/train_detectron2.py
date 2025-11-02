#!/usr/bin/env python3
"""
Mask R-CNN (Detectron2, R-50-FPN) – Docker-compatible with CPU/GPU support

- CPU: creates small subset, few iters
- GPU: full json, many iters
- Docker: automatic path detection and compatibility
"""
from __future__ import annotations
import numpy as np
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Optional



try:
    import numpy as np
    print(f"[INFO] NumPy version: {np.__version__}")
except ImportError as e:
    print("[ERROR] NumPy not available. Please install numpy first.")
    sys.exit(1)


import detectron2
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.model_zoo import get_config_file, get_checkpoint_url
from detectron2.utils.logger import setup_logger
import detectron2.data

# --- Smart Path Detection (Works in Docker and Local) ---
def get_project_root() -> Path:
    """Detect project root automatically for Docker and local execution."""
    # Try Docker path first
    docker_root = Path("/workspace").resolve()
    if docker_root.exists():
        return docker_root
    
    # Fallback to local path detection
    current_file = Path(__file__).resolve()
    
    # If running from docker/detectron2/ folder
    if "docker/detectron2" in str(current_file):
        return current_file.parent.parent.parent
    
    # If running from models/ folder
    if current_file.parent.name == "models":
        return current_file.parent
    
    # Default: assume we're in project root
    return current_file

PROJECT_ROOT = get_project_root()
COCO_ROOT = PROJECT_ROOT / "converted" / "coco"

# KORRIGIERTE JSON PFADE - verwende instances_*.json
TRAIN_JSON = COCO_ROOT / "instances_train.json"
VAL_JSON = COCO_ROOT / "instances_val.json"
TEST_JSON = COCO_ROOT / "instances_test.json"

TRAIN_IMG_DIR = COCO_ROOT / "images" / "train"
VAL_IMG_DIR = COCO_ROOT / "images" / "val"
TEST_IMG_DIR = COCO_ROOT / "images" / "test"

OUTPUT_DIR = PROJECT_ROOT / "runs" / "maskrcnn_r50fpn"

# --- Configuration ---
USE_CUDA = os.getenv("DETECTRON2_USE_CUDA", "0") == "1"
CPU_SANITY_MAX_IMAGES = 60
CPU_MAX_ITER = 200
GPU_MAX_ITER = 20000
CPU_BATCH_IMS = 2
GPU_BATCH_IMS = 4
BASE_LR = 0.002
NUM_CLASSES = 1

random.seed(42)


def _register(name: str, json_path: Path, img_dir: Path) -> None:
    """Register COCO dataset with proper error handling."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    print(f"[DATA] Registering {name}: {json_path.name}")
    register_coco_instances(name, {}, str(json_path), str(img_dir))


def _make_subset_json(src_json: Path, dst_json: Path, max_images: int) -> None:
    """Create a subset of COCO JSON for CPU testing."""
    data = json.loads(src_json.read_text(encoding="utf-8"))
    
    if len(data["images"]) < max_images:
        print(f"[WARN] Only {len(data['images'])} images available, using all")
        max_images = len(data["images"])
    
    keep_images = data["images"][:max_images]
    keep_ids = {im["id"] for im in keep_images}
    anns = [a for a in data["annotations"] if a["image_id"] in keep_ids]
    
    out = {
        "images": keep_images, 
        "annotations": anns, 
        "categories": data["categories"]
    }
    
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    dst_json.write_text(json.dumps(out), encoding="utf-8")
    print(f"[SUBSET] Created {dst_json.name} with {len(keep_images)} images, {len(anns)} annotations")


def _ensure_cpu_subsets() -> tuple[Path, Path, Optional[Path]]:
    """Create CPU subsets and return paths to subset JSON files."""
    sub = COCO_ROOT / "_cpu_sanity"
    
    if sub.exists():
        shutil.rmtree(sub)
    sub.mkdir(parents=True, exist_ok=True)
    
    # Verwende instances_*.json auch für Subsets für Konsistenz
    tr_json = sub / "instances_train.json"
    va_json = sub / "instances_val.json"
    te_json = sub / "instances_test.json"
    
    _make_subset_json(TRAIN_JSON, tr_json, CPU_SANITY_MAX_IMAGES)
    _make_subset_json(VAL_JSON, va_json, max(1, CPU_SANITY_MAX_IMAGES // 2))
    
    test_path = None
    if TEST_JSON.exists():
        _make_subset_json(TEST_JSON, te_json, CPU_SANITY_MAX_IMAGES)
        test_path = te_json
    
    return tr_json, va_json, test_path


def _validate_paths() -> None:
    """Validate that all required paths exist."""
    print("[VALIDATION] Checking dataset paths...")
    
    required_paths = [
        (COCO_ROOT, "COCO root"),
        (TRAIN_JSON, "Train JSON"),
        (VAL_JSON, "Val JSON"),
        (TRAIN_IMG_DIR, "Train images"),
        (VAL_IMG_DIR, "Val images")
    ]
    
    # Prüfe ob Pfade existieren
    missing_paths = []
    for path, desc in required_paths:
        if not path.exists():
            missing_paths.append(f"{desc}: {path}")
        else:
            print(f"  ✓ {desc}: {path}")
    
    # Zusätzliche Prüfungen für bessere Debug-Info
    if TRAIN_JSON.exists():
        data = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
        print(f"  ✓ Train JSON: {len(data['images'])} images, {len(data['annotations'])} annotations")
    
    if VAL_JSON.exists():
        data = json.loads(VAL_JSON.read_text(encoding="utf-8"))
        print(f"  ✓ Val JSON: {len(data['images'])} images, {len(data['annotations'])} annotations")
    
    # Prüfe Bilder-Verzeichnisse
    if TRAIN_IMG_DIR.exists():
        image_count = len(list(TRAIN_IMG_DIR.glob("*.*")))
        print(f"  ✓ Train images: {image_count} files")
    
    if VAL_IMG_DIR.exists():
        image_count = len(list(VAL_IMG_DIR.glob("*.*")))
        print(f"  ✓ Val images: {image_count} files")
    
    if missing_paths:
        print("\n[ERROR] Missing required paths:")
        for missing in missing_paths:
            print(f"  ✗ {missing}")
        
        # Vorschläge für häufige Probleme
        print("\n[DEBUG] Available files in COCO directory:")
        if COCO_ROOT.exists():
            for item in COCO_ROOT.iterdir():
                print(f"    - {item.name}{'/' if item.is_dir() else ''}")
        
        raise FileNotFoundError("Required dataset files not found")
    
    print("[VALIDATION] All dataset paths validated successfully")


def _find_json_files() -> None:
    """Debug function to find all JSON files in COCO directory."""
    print("\n[DEBUG] Searching for JSON files in COCO directory...")
    json_files = list(COCO_ROOT.glob("*.json"))
    if json_files:
        print("Found JSON files:")
        for json_file in json_files:
            print(f"  - {json_file.name}")
    else:
        print("No JSON files found!")
    
    # Prüfe images Unterverzeichnisse
    print("\n[DEBUG] Checking images subdirectories...")
    images_dir = COCO_ROOT / "images"
    if images_dir.exists():
        for subdir in images_dir.iterdir():
            if subdir.is_dir():
                file_count = len(list(subdir.glob("*.*")))
                print(f"  - {subdir.name}: {file_count} files")
    else:
        print("  - images directory not found!")


class Trainer(DefaultTrainer):
    """Custom trainer with COCO evaluation support."""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        out = output_folder or (Path(cfg.OUTPUT_DIR) / "inference" / dataset_name)
        out.mkdir(parents=True, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, False, output_folder=str(out))


def setup_config() -> None:
    """Setup and return Detectron2 configuration."""
    cfg = get_cfg()
    
    # Load base config
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    # Dataset configuration
    cfg.DATALOADER.NUM_WORKERS = 2 if USE_CUDA else 0
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.DEVICE = "cuda" if USE_CUDA else "cpu"
    
    # Training configuration
    cfg.SOLVER.IMS_PER_BATCH = GPU_BATCH_IMS if USE_CUDA else CPU_BATCH_IMS
    cfg.SOLVER.BASE_LR = BASE_LR
    cfg.SOLVER.MAX_ITER = GPU_MAX_ITER if USE_CUDA else CPU_MAX_ITER
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # Save checkpoints periodically
    cfg.TEST.EVAL_PERIOD = 1000  # Evaluate periodically
    
    # Output configuration
    cfg.OUTPUT_DIR = str(OUTPUT_DIR)
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    return cfg


def main() -> None:
    """Main training function."""
    setup_logger()
    
    print("=" * 60)
    print("DETECTRON2 TRAINING - MASK R-CNN R50-FPN")
    print("=" * 60)
    print(f"[INFO] Detectron2 version: {detectron2.__version__}")
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] CUDA enabled: {USE_CUDA}")
    print(f"[INFO] Mode: {'GPU (Full training)' if USE_CUDA else 'CPU (Sanity check)'}")
    print("=" * 60)
    
    # Debug: Zeige verfügbare JSON Dateien
    _find_json_files()
    print("")
    
    # Validate paths
    try:
        _validate_paths()
        print("[INFO] All dataset paths validated successfully")
    except FileNotFoundError as e:
        print(f"[ERROR] Dataset validation failed: {e}")
        print("\n[SOLUTION] Make sure you ran the data conversion first:")
        print("  python tools/convert_strawdi.py --src /path/to/StrawDI_Db1 --out_coco converted/coco")
        sys.exit(1)
    
    # Register datasets
    if USE_CUDA:
        # Full dataset for GPU training
        train_name, val_name, test_name = "straw_train", "straw_val", "straw_test"
        print(f"[DATA] Using full dataset for GPU training")
        _register(train_name, TRAIN_JSON, TRAIN_IMG_DIR)
        _register(val_name, VAL_JSON, VAL_IMG_DIR)
        
        if TEST_JSON.exists():
            _register(test_name, TEST_JSON, TEST_IMG_DIR)
        else:
            test_name = None
            print("[INFO] Test set not found, skipping test evaluation")
    else:
        # Subset for CPU testing
        print("[INFO] CPU mode: Creating dataset subsets for faster testing")
        tr_json, va_json, te_json = _ensure_cpu_subsets()
        train_name, val_name, test_name = "straw_train_cpu", "straw_val_cpu", "straw_test_cpu"
        _register(train_name, tr_json, TRAIN_IMG_DIR)
        _register(val_name, va_json, VAL_IMG_DIR)
        
        if te_json is not None:
            _register(test_name, te_json, TEST_IMG_DIR)
        else:
            test_name = None
    
    # Setup configuration
    cfg = setup_config()
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)
    
    default_setup(cfg, {})
    
    # Start training
    print("\n[INFO] Starting training...")
    print(f"[INFO] Output directory: {cfg.OUTPUT_DIR}")
    print(f"[INFO] Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"[INFO] Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"[INFO] Device: {cfg.MODEL.DEVICE}")
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    # Evaluation
    print("\n[INFO] Training completed. Starting evaluation...")
    
    # Validation evaluation
    print("[INFO] Evaluating on validation set...")
    val_loader = detectron2.data.build_detection_test_loader(cfg, val_name)
    evaluator = COCOEvaluator(
        val_name, cfg, False, 
        output_dir=str(Path(cfg.OUTPUT_DIR) / "inference" / "val")
    )
    val_results = inference_on_dataset(trainer.model, val_loader, evaluator)
    print("[VAL RESULTS]", val_results)
    
    # Test evaluation (if available)
    if test_name is not None:
        print("[INFO] Evaluating on test set...")
        test_loader = detectron2.data.build_detection_test_loader(cfg, test_name)
        evaluator = COCOEvaluator(
            test_name, cfg, False,
            output_dir=str(Path(cfg.OUTPUT_DIR) / "inference" / "test")
        )
        test_results = inference_on_dataset(trainer.model, test_loader, evaluator)
        print("[TEST RESULTS]", test_results)
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Training and evaluation completed!")
    print(f"[OUTPUT] Results saved to: {cfg.OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()