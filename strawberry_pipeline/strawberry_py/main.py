from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from strawberry_py.config import load_config
from strawberry_py.pipeline.runner import PipelineRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline strawberry pipeline (no ROS2).")

    p.add_argument("--config", required=True, type=Path, help="e.g. configs/meca_d405.yaml")

    # optional overrides (ohne cfg zu mutieren)
    p.add_argument("--dataset_root", type=Path, default=None, help="e.g. data/Basisdatensatz")
    p.add_argument(
        "--out_base",
        type=Path,
        default=None,
        help="If set, final out_root = <out_base>/<dataset_name>. (Ignored if --out_root is set.)",
    )
    p.add_argument(
        "--out_root",
        type=Path,
        default=None,
        help="Explicit output root. If set, overrides config outputs.out_root and --out_base.",
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name used under out_base. Default: dataset_root folder name.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = args.config.resolve()
    cfg = load_config(cfg_path)

    # dataset root bestimmen
    dataset_root: Optional[Path]
    if args.dataset_root is not None:
        dataset_root = args.dataset_root.resolve()
        if not dataset_root.exists() or not dataset_root.is_dir():
            raise FileNotFoundError(f"--dataset_root is not a directory: {dataset_root}")
    else:
        # kommt aus config
        dataset_root = Path(cfg.dataset.root).resolve()

    # output root bestimmen
    if args.out_root is not None:
        out_root = args.out_root.resolve()
    else:
        if args.out_base is not None:
            out_base = args.out_base.resolve()
            name = args.dataset_name or dataset_root.name
            out_root = (out_base / name).resolve()
        else:
            # kommt aus config
            out_root = Path(cfg.outputs.out_root).resolve()

    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[RUN] config       : {cfg_path}")
    print(f"[RUN] dataset_root : {dataset_root}")
    print(f"[RUN] out_root     : {out_root}")

    print(f"[Config] euler_convention = {cfg.robot.euler_convention}")
    print(f"[Config] views loaded     = {sorted(cfg.robot.views.keys())}")
    print(
        f"[Config] intrinsics       = fx={cfg.camera.fx} fy={cfg.camera.fy} "
        f"cx={cfg.camera.cx} cy={cfg.camera.cy}"
    )
    print(f"[Config] R_trf_cam        = {cfg.robot.cam_axes_correction_R_trf_cam_row_major_3x3}")

    runner = PipelineRunner(cfg, dataset_root=dataset_root, out_root=out_root)
    runner.run()


if __name__ == "__main__":
    main()
