#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
capture_dataset.py

Nimmt ein Dataset mit mehreren Plants und mehreren Views auf.
Speichert:
- config_used.yaml
- camera_meta.yaml
- run_meta.yaml (Robot snapshot inkl. WRF/TRF + pose_source)
- poses.csv (Pose + pose_source + optional joints)
- pro plant:
    color_{view}.png
    depth_{view}.png          (aligned depth-to-color, uint16)
    optional depth_raw_{view}.png
    optional cloud_aligned_{view}.ply
    optional cloud_raw_{view}.ply

Start:
  python src/capture_dataset.py --config config.yaml
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any

import yaml
import cv2

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from robot_meca import Meca500Controller  # noqa: E402
from camera_d405 import RealsenseD405     # noqa: E402


@dataclass(frozen=True)
class CaptureCfg:
    out_root: Path
    run_name: str
    plant_prefix: str
    start_plant_id: int
    num_plants: int
    view_ids: List[int]
    view_to_pose_key: Dict[int, str]
    save_depth_raw: bool
    save_ply_aligned: bool
    save_ply_raw: bool


POSES_HEADER = [
    "plant_id", "view_id", "pose_key",
    "pose_source",
    "x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg",
    "j1_deg", "j2_deg", "j3_deg", "j4_deg", "j5_deg", "j6_deg",
    "timestamp",
]


def load_cfg(cfg_path: Path) -> Tuple[CaptureCfg, dict]:
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    ds = raw.get("dataset", {}) or {}
    out_root = Path(ds.get("out_root", "runs")).resolve()
    run_name = str(ds.get("run_name", "")).strip()
    if not run_name:
        run_name = time.strftime("dataset_%Y%m%d_%H%M%S")

    plant_prefix = str(ds.get("plant_prefix", "plant_"))
    start_plant_id = int(ds.get("start_plant_id", 0))
    num_plants = int(ds.get("num_plants", 1))
    view_ids = [int(v) for v in ds.get("view_ids", [0, 1, 2])]

    view_map_raw = ds.get("view_to_pose_key", {})
    if not isinstance(view_map_raw, dict) or not view_map_raw:
        raise ValueError("dataset.view_to_pose_key missing")
    view_to_pose_key = {int(k): str(v) for k, v in view_map_raw.items()}

    out = raw.get("outputs", {}) or {}
    save_depth_raw = bool(out.get("save_depth_raw", True))
    save_ply_aligned = bool(out.get("save_ply_aligned", True))
    save_ply_raw = bool(out.get("save_ply_raw", False))

    return (
        CaptureCfg(
            out_root=out_root,
            run_name=run_name,
            plant_prefix=plant_prefix,
            start_plant_id=start_plant_id,
            num_plants=num_plants,
            view_ids=view_ids,
            view_to_pose_key=view_to_pose_key,
            save_depth_raw=save_depth_raw,
            save_ply_aligned=save_ply_aligned,
            save_ply_raw=save_ply_raw,
        ),
        raw,
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plant_dir(run_dir: Path, prefix: str, plant_id: int) -> Path:
    return run_dir / f"{prefix}{plant_id:03d}"


def write_yaml(path: Path, data: Any) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def save_png_rgb(path: Path, bgr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise IOError(f"Failed to write {path}")


def save_png_depth_u16(path: Path, depth_u16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), depth_u16)  # keeps 16-bit
    if not ok:
        raise IOError(f"Failed to write {path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg, raw = load_cfg(cfg_path)

    run_dir = cfg.out_root / cfg.run_name
    ensure_dir(run_dir)

    # Copy config for provenance
    write_yaml(run_dir / "config_used.yaml", raw)

    # init robot + camera using same config
    robot = Meca500Controller.from_config_yaml(cfg_path, verbose=True)

    cam_cfg = raw.get("camera", {}) or {}
    cam = RealsenseD405(
        width=int(cam_cfg.get("width", 640)),
        height=int(cam_cfg.get("height", 480)),
        fps=int(cam_cfg.get("fps", 30)),
    )

    poses_csv = run_dir / "poses.csv"
    poses_exists = poses_csv.exists()

    print(f"[CAPTURE] run_dir={run_dir}")

    try:
        # --- Robot init ---
        robot.connect()
        robot.activate_and_home()
        robot.set_wrf_trf_from_config(cfg_path)

        # pose stability quick check
        stab = robot.pose_stability_check(samples=8, dt_s=0.05)
        print(f"[CAPTURE] pose stability: {stab}")

        # --- Camera init ---
        cam.start()
        cam.save_meta_yaml(run_dir / "camera_meta.yaml")
        print(f"[CAPTURE] wrote {run_dir / 'camera_meta.yaml'}")

        # --- store robot snapshot meta once ---
        run_meta = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pose_stability": stab,
            "robot_state": robot.get_state_dict(),
        }
        write_yaml(run_dir / "run_meta.yaml", run_meta)
        print(f"[CAPTURE] wrote {run_dir / 'run_meta.yaml'}")

        # --- capture loop ---
        with poses_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=POSES_HEADER, delimiter=";")
            if not poses_exists:
                w.writeheader()

            for plant_idx in range(cfg.num_plants):
                pid = cfg.start_plant_id + plant_idx
                pdir = plant_dir(run_dir, cfg.plant_prefix, pid)
                ensure_dir(pdir)

                print(f"\n[CAPTURE] Plant {pid:03d} -> {pdir}")

                for view_id in cfg.view_ids:
                    pose_key = cfg.view_to_pose_key.get(int(view_id))
                    if pose_key is None:
                        raise ValueError(f"Missing view_to_pose_key for view_id={view_id}")

                    print(f"[CAPTURE] view={view_id} move='{pose_key}'")
                    robot.move_to(pose_key)

                    fr = cam.get_frames()
                    color = fr["color_bgr"]
                    depth_al = fr["depth_aligned_u16"]
                    depth_raw = fr["depth_raw_u16"]

                    # save images
                    save_png_rgb(pdir / f"color_{view_id}.png", color)
                    save_png_depth_u16(pdir / f"depth_{view_id}.png", depth_al)
                    if cfg.save_depth_raw:
                        save_png_depth_u16(pdir / f"depth_raw_{view_id}.png", depth_raw)

                    # save PLY
                    if cfg.save_ply_aligned:
                        cam.export_ply_rs(
                            pdir / f"cloud_aligned_{view_id}.ply",
                            fr["rs_depth_aligned"],
                            fr["rs_color_aligned"],
                        )
                    if cfg.save_ply_raw:
                        cam.export_ply_rs(
                            pdir / f"cloud_raw_{view_id}.ply",
                            fr["rs_depth_raw"],
                            fr["rs_color_raw"],
                        )

                    pose = robot.get_pose_mm_deg()
                    joints = robot.get_joints_deg()
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")

                    row = {
                        "plant_id": pid,
                        "view_id": int(view_id),
                        "pose_key": str(pose_key),
                        "pose_source": pose.source,
                        "x_mm": pose.x_mm,
                        "y_mm": pose.y_mm,
                        "z_mm": pose.z_mm,
                        "rx_deg": pose.rx_deg,
                        "ry_deg": pose.ry_deg,
                        "rz_deg": pose.rz_deg,
                        "timestamp": ts,
                    }

                    # joints optional
                    if joints is None:
                        row.update({f"j{i}_deg": "" for i in range(1, 7)})
                    else:
                        row.update({f"j{i}_deg": float(joints[i - 1]) for i in range(1, 7)})

                    w.writerow(row)
                    f.flush()

                    print(f"[CAPTURE] saved view={view_id} + pose({pose.source})")

        print(f"\n[CAPTURE] DONE: {run_dir}")

    finally:
        try:
            cam.stop()
        except Exception:
            pass
        robot.disconnect()


if __name__ == "__main__":
    main()
