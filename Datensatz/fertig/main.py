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

    print(f"[Config] euler_convention = {cfg.robot.euler_convention}")
    print(f"[Config] views loaded = {sorted(cfg.robot.views.keys())}")
    print(f"[Config] intrinsics = fx={cfg.camera.fx} fy={cfg.camera.fy} cx={cfg.camera.cx} cy={cfg.camera.cy}")
    print(f"[Config] R_trf_cam = {cfg.robot.cam_axes_correction_R_trf_cam_row_major_3x3}")

    runner = PipelineRunner(cfg)
    runner.run()



if __name__ == "__main__":
    main()
