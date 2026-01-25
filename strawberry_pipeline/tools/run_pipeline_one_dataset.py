#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# --- make repo importable even when executed from tools/ ---
REPO_ROOT = Path(__file__).resolve().parents[1]  # .../strawberry_pipeline
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strawberry_py.config import load_config
from strawberry_py.pipeline.runner import PipelineRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pipeline for exactly one dataset root.")
    p.add_argument("--config", required=True, type=Path, help="e.g. configs/meca_d405.yaml")
    p.add_argument("--dataset_root", required=True, type=Path, help="e.g. data/Basisdatensatz")
    p.add_argument(
        "--out_base",
        default=Path("outputs"),
        type=Path,
        help="Base output dir. Final out_root = out_base/<dataset_name> (default: outputs/<dataset_name>)",
    )
    p.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        help="Optional: override name used for output folder; default is dataset_root folder name",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve relative to current working directory (you should run from strawberry_pipeline/)
    cfg_path = args.config.resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"--config not found: {cfg_path}")

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"--dataset_root is not a directory: {dataset_root}")

    dataset_name = args.dataset_name or dataset_root.name
    out_root = (args.out_base / dataset_name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cfg_path)

    # Override config paths
    try:
        cfg.dataset.root = str(dataset_root)
    except Exception:
        cfg.dataset.root = dataset_root

    try:
        cfg.outputs.out_root = str(out_root)
    except Exception:
        cfg.outputs.out_root = out_root

    print(f"[RUN] repo_root   : {REPO_ROOT}")
    print(f"[RUN] config      : {cfg_path}")
    print(f"[RUN] dataset_root: {dataset_root}")
    print(f"[RUN] out_root    : {out_root}")

    runner = PipelineRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
