from __future__ import annotations

import argparse
from pathlib import Path

from strawberry_py.config import load_config
from strawberry_py.pipeline.runner import PipelineRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline strawberry pipeline (no ROS2).")
    p.add_argument("--config", type=str, default="config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    runner = PipelineRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
