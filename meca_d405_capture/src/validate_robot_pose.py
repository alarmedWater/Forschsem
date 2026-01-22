#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_robot_pose.py

Ziel:
- Sicherstellen, dass Robot-Posen reproduzierbar sind
- WRF/TRF korrekt gesetzt
- Pose-Stabilität nach Move prüfen
- Optional: Orientierung als Matrix + forward-vector ausgeben

Start:
python src/validate_robot_pose.py --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import yaml
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from robot_meca import Meca500Controller  # noqa: E402
from transforms import pose_mm_deg_to_T  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--samples", type=int, default=6, help="pose stability samples per position")
    ap.add_argument("--dt", type=float, default=0.05, help="seconds between stability samples")
    ap.add_argument("--loops", type=int, default=1, help="repeat full sequence N times")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    seq = raw.get("robot", {}).get("sequence", ["l", "m", "r"])
    euler = str(raw.get("robot", {}).get("euler_convention", "RzRyRx_deg"))

    robot = Meca500Controller.from_config_yaml(cfg_path, verbose=True)

    try:
        robot.connect()
        robot.activate_and_home()
        robot.set_wrf_trf_from_config(cfg_path)

        print(f"\n[VALIDATE] sequence={seq}")
        print(f"[VALIDATE] euler_convention={euler}\n")

        # optional: check repeatability by returning to m between moves
        for loop in range(int(args.loops)):
            print(f"\n========== LOOP {loop+1}/{args.loops} ==========")

            for key in seq:
                print(f"\n[VALIDATE] Move -> {key}")
                robot.move_to(str(key))
                robot.print_pose(prefix=f"[VALIDATE] Pose({key})")

                st = robot.pose_stability_check(samples=args.samples, dt_s=args.dt)
                print("[VALIDATE] stability:",
                      f"dx={st['max_dx_mm']:.3f}mm dy={st['max_dy_mm']:.3f}mm dz={st['max_dz_mm']:.3f}mm | "
                      f"drx={st['max_drx_deg']:.4f}° dry={st['max_dry_deg']:.4f}° drz={st['max_drz_deg']:.4f}°")

                p = robot.get_pose_mm_deg()
                T = pose_mm_deg_to_T(
                    p.x_mm, p.y_mm, p.z_mm, p.rx_deg, p.ry_deg, p.rz_deg,
                    euler_convention=euler, t_in_m=True,
                )
                R = T[:3, :3]
                fw = R[:, 2]  # z-axis of TRF in world coords (forward if your TRF z points forward)
                print(f"[VALIDATE] forward(z) in WORLD: [{fw[0]: .4f}, {fw[1]: .4f}, {fw[2]: .4f}]")
                time.sleep(0.1)

        print("\n[VALIDATE] DONE.")

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
