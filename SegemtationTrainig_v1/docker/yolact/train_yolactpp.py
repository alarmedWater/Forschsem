#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ==================== CONFIGURATION ====================

# Smart path detection for Docker and local
PROJECT_ROOT = Path("/workspace").resolve()
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

YOLACT_ROOT = PROJECT_ROOT / "yolact"
YOLACT_PY = YOLACT_ROOT / "train.py"  # YOLACT uses train.py for training

# Dataset paths
COCO_ROOT = PROJECT_ROOT / "converted" / "coco"
TRAIN_JSON = COCO_ROOT / "instances_train.json"
VAL_JSON = COCO_ROOT / "instances_val.json"
TRAIN_IMG_DIR = COCO_ROOT / "images" / "train"
VAL_IMG_DIR = COCO_ROOT / "images" / "val"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "runs" / "yolactpp_r101"

# Environment-based configuration
USE_CUDA = os.getenv("YOLACT_USE_CUDA", "0") == "1"
QUICK_TEST = os.getenv("YOLACT_QUICK_TEST", "0") == "1"

# Training parameters based on mode
if QUICK_TEST:
    # Extreme Sanity (1-2 minutes)
    MAX_IMAGES = 10
    MAX_ITERS = 100
    BATCH_SIZE = 2
    MODE_NAME = "🧪 EXTREME SANITY MODE"
elif not USE_CUDA:
    # CPU Sanity (5-15 minutes)  
    MAX_IMAGES = 50
    MAX_ITERS = 500
    BATCH_SIZE = 4
    MODE_NAME = "🚀 CPU SANITY MODE"
else:
    # GPU Full Training
    MAX_IMAGES = None  # Use all images
    MAX_ITERS = 80000
    BATCH_SIZE = 8
    MODE_NAME = "🏎️ GPU FULL TRAINING MODE"

# Fixed parameters
LEARNING_RATE = 1e-3
CONFIG_NAME = "yolact_plus_resnet101_config"
NUM_WORKERS = 0 if not USE_CUDA else 2

# ==================== HELPER FUNCTIONS ====================

def _make_subset_json(src: Path, dst: Path, max_images: int) -> None:
    """
    Create a subset of COCO JSON for faster testing.
    """
    print(f"[DATA] Creating subset: {src.name} -> {max_images} images")
    
    data = json.loads(src.read_text(encoding="utf-8"))
    
    # Safety check
    available_images = len(data["images"])
    if available_images < max_images:
        print(f"[WARN] Only {available_images} images available, using all")
        max_images = available_images
    
    keep_images = data["images"][:max_images]
    keep_ids = {im["id"] for im in keep_images}
    anns = [a for a in data["annotations"] if a["image_id"] in keep_ids]
    
    out = {
        "images": keep_images, 
        "annotations": anns, 
        "categories": data["categories"]
    }
    
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out), encoding="utf-8")
    print(f"[DATA] Created {dst.name} with {len(keep_images)} images, {len(anns)} annotations")


def _ensure_cpu_subsets() -> tuple[Path, Path]:
    """
    Create CPU subsets and return paths to subset JSON files.
    """
    sub_dir = COCO_ROOT / "_cpu_sanity_yolact"
    
    if sub_dir.exists():
        shutil.rmtree(sub_dir)
    sub_dir.mkdir(parents=True, exist_ok=True)
    
    train_subset = sub_dir / "instances_train.json"
    val_subset = sub_dir / "instances_val.json"
    
    _make_subset_json(TRAIN_JSON, train_subset, MAX_IMAGES)
    _make_subset_json(VAL_JSON, val_subset, max(1, MAX_IMAGES // 2))
    
    return train_subset, val_subset


def _validate_environment() -> None:
    """
    Validate that all required paths and dependencies exist.
    """
    print("[VALIDATION] Checking environment...")
    
    required_paths = [
        (YOLACT_ROOT, "YOLACT root"),
        (YOLACT_PY, "YOLACT train.py"),
        (COCO_ROOT, "COCO root"),
        (TRAIN_JSON, "Train JSON"),
        (VAL_JSON, "Val JSON"), 
        (TRAIN_IMG_DIR, "Train images"),
        (VAL_IMG_DIR, "Val images")
    ]
    
    missing_paths = []
    for path, description in required_paths:
        if not path.exists():
            missing_paths.append(f"{description}: {path}")
    
    if missing_paths:
        print("[ERROR] Missing required paths:")
        for missing in missing_paths:
            print(f"  - {missing}")
        raise FileNotFoundError("Required files not found")
    
    # Check if YOLACT is properly set up
    config_path = YOLACT_ROOT / "data" / "config.py"
    if not config_path.exists():
        print("[WARN] YOLACT config not found - training might fail")
    
    print("[VALIDATION] All paths validated successfully")


def _setup_output_directory() -> None:
    """
    Create and prepare output directory.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] Results will be saved to: {OUTPUT_DIR}")


# ==================== MAIN TRAINING FUNCTION ====================

def main():
    """
    Main YOLACT training function with CPU/GPU support.
    """
    print("=" * 60)
    print("YOLACT++ TRAINING - ResNet101 Backbone")
    print("=" * 60)
    print(f"[MODE] {MODE_NAME}")
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] CUDA enabled: {USE_CUDA}")
    print(f"[INFO] Max images: {MAX_IMAGES or 'ALL'}")
    print(f"[INFO] Max iterations: {MAX_ITERS}")
    print(f"[INFO] Batch size: {BATCH_SIZE}")
    print(f"[INFO] Learning rate: {LEARNING_RATE}")
    print("=" * 60)
    
    try:
        # Step 1: Validate environment
        _validate_environment()
        
        # Step 2: Setup output directory
        _setup_output_directory()
        
        # Step 3: Prepare datasets
        if USE_CUDA and not QUICK_TEST:
            # Use full dataset for GPU training
            train_json, val_json = TRAIN_JSON, VAL_JSON
            print("[DATA] Using full dataset for GPU training")
        else:
            # Create subsets for CPU/testing
            print("[DATA] Creating dataset subsets for faster testing")
            train_json, val_json = _ensure_cpu_subsets()
        
        # Step 4: Build YOLACT training command
        cmd = [
            "python", str(YOLACT_PY),
            f"--config={CONFIG_NAME}",
            f"--batch_size={BATCH_SIZE}",
            f"--lr={LEARNING_RATE}",
            f"--max_iter={MAX_ITERS}",
            f"--save_folder={OUTPUT_DIR}",
            f"--num_workers={NUM_WORKERS}",
            "--dataset=coco2017_dataset",
            "--validation_size=0.1",
            "--validation_epoch=5",
            "--keep_latest=1",
            "--save_intervals=1000",
            "--no_bar",  # Disable progress bar for cleaner logs
        ]
        
        # Add CUDA flag
        if USE_CUDA:
            cmd.append("--cuda=True")
            print("[TRAINING] Using CUDA/GPU acceleration")
        else:
            cmd.append("--cuda=False") 
            print("[TRAINING] Using CPU only")
        
        # For quick test, reduce logging
        if QUICK_TEST:
            cmd.extend(["--no_bar", "--log=False"])
        
        print("[CMD]", " ".join(cmd))
        
        # Step 5: Run training
        print("\n[TRAINING] Starting YOLACT training...")
        print("[TRAINING] This may take a while. Please wait...")
        
        # Set Python path for YOLACT
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{YOLACT_ROOT}:{env.get('PYTHONPATH', '')}"
        
        # Run in YOLACT directory
        result = subprocess.run(cmd, cwd=YOLACT_ROOT, env=env, check=True)
        
        # Step 6: Training completed
        print("\n" + "=" * 60)
        print("[SUCCESS] YOLACT training completed!")
        print(f"[OUTPUT] Results saved to: {OUTPUT_DIR}")
        
        # List generated files
        if OUTPUT_DIR.exists():
            generated_files = list(OUTPUT_DIR.glob("*"))
            if generated_files:
                print("[OUTPUT] Generated files:")
                for file in generated_files:
                    print(f"  - {file.name}")
        
        print("=" * 60)
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Training failed with exit code: {e.returncode}")
        print("[INFO] Check YOLACT logs above for details")
        sys.exit(1)
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("[INFO] Make sure your dataset is properly converted")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        print("[INFO] Check your environment and dependencies")
        sys.exit(1)


if __name__ == "__main__":
    main()