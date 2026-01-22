#!/usr/bin/env python3
# tools/sweep_pose_convention.py

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml

# ---------- rotations ----------
def Rx(a): 
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)

def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

def R_from_order(rx_deg: float, ry_deg: float, rz_deg: float, order: str) -> np.ndarray:
    """
    order examples: 'RzRyRx', 'RxRyRz', ...
    Interpretiert als: R = A @ B @ C (rightmost applied first).
    """
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    mats = {"Rx": Rx(rx), "Ry": Ry(ry), "Rz": Rz(rz)}
    seq = [order[i:i+2] for i in range(0, 6, 2)]
    return mats[seq[0]] @ mats[seq[1]] @ mats[seq[2]]

# ---------- load features centroids ----------
def load_centroids(features_csv: Path, instance_id: Optional[int] = None) -> Dict[int, np.ndarray]:
    if not features_csv.exists():
        raise FileNotFoundError(features_csv)

    # delimiter sniff
    sample = features_csv.read_text(encoding="utf-8", errors="replace").splitlines()[:5]
    sniff = "\n".join(sample)
    try:
        dialect = csv.Sniffer().sniff(sniff, delimiters=";,\t")
        delim = dialect.delimiter
    except Exception:
        delim = ";"

    rows = []
    with features_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter=delim)
        for row in r:
            rows.append(row)

    by_view: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for row in rows:
        try:
            vid = int(row["view_id"])
            iid = int(row["instance_id"])
            npt = int(float(row["num_points"]))
            c = np.array([float(row["cx"]), float(row["cy"]), float(row["cz"])], dtype=np.float64)
        except Exception:
            continue
        if instance_id is not None and iid != instance_id:
            continue
        by_view.setdefault(vid, []).append((npt, c))

    out: Dict[int, np.ndarray] = {}
    for vid, lst in by_view.items():
        npt, c = max(lst, key=lambda x: x[0])  # largest instance per view
        out[vid] = c
    return out

# ---------- load robot poses from yaml, but DO NOT use strawberry_py.config ----------
@dataclass(frozen=True)
class PoseWorldTRF:
    t_m: np.ndarray
    R: np.ndarray

def load_robot_views(cfg_path: Path, order: str) -> Dict[int, PoseWorldTRF]:
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    robot = raw.get("robot", {})
    views = robot.get("views", {})
    out: Dict[int, PoseWorldTRF] = {}

    for vid_str, v in views.items():
        try:
            vid = int(vid_str)
        except Exception:
            continue
        pose = v.get("pose_wrf_mm_deg", None)
        if not (isinstance(pose, list) and len(pose) == 6):
            continue
        x_mm, y_mm, z_mm, rx, ry, rz = [float(p) for p in pose]
        t = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0
        R = R_from_order(rx, ry, rz, order)
        out[vid] = PoseWorldTRF(t_m=t, R=R)
    return out

def load_cam_trf(cfg_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    robot = raw.get("robot", {})
    R_list = robot.get("cam_axes_correction_R_trf_cam_row_major_3x3", None)
    t_mm = robot.get("camera_in_trf_translation_mm", None)
    if not (isinstance(R_list, list) and len(R_list) == 9):
        raise ValueError("robot.cam_axes_correction_R_trf_cam_row_major_3x3 missing/invalid")
    if not (isinstance(t_mm, list) and len(t_mm) == 3):
        raise ValueError("robot.camera_in_trf_translation_mm missing/invalid")
    R = np.array(R_list, dtype=np.float64).reshape(3,3)
    t = np.array(t_mm, dtype=np.float64).reshape(3,) / 1000.0
    return R, t

def score_orders(cfg: Path, features: Path, view_ids: List[int]) -> None:
    cents = load_centroids(features, instance_id=1)  # du nutzt i.d.R. 1
    R_trf_cam, t_trf_cam = load_cam_trf(cfg)

    orders = ["RzRyRx", "RzRxRy", "RyRzRx", "RyRxRz", "RxRyRz", "RxRzRy"]

    best = None
    for order in orders:
        poses = load_robot_views(cfg, order)

        world_pts = []
        used = []
        for vid in view_ids:
            if vid not in poses or vid not in cents:
                continue
            pc = cents[vid]
            p_trf = (R_trf_cam @ pc) + t_trf_cam
            pw = (poses[vid].R @ p_trf) + poses[vid].t_m
            world_pts.append(pw)
            used.append(vid)

        if len(world_pts) < 2:
            continue

        W = np.vstack(world_pts)  # (N,3)
        mean = W.mean(axis=0, keepdims=True)
        rms = float(np.sqrt(np.mean(np.sum((W - mean)**2, axis=1))))  # meters

        if best is None or rms < best[0]:
            best = (rms, order, used, W)

        print(f"{order}: RMS to mean = {rms*1000.0:.2f} mm  (views={used})")

    if best:
        rms, order, used, W = best
        print("\nBEST:", order, f"RMS={rms*1000.0:.2f} mm views={used}")
        for vid, p in zip(used, W):
            print(f"  view {vid}: {p.tolist()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--view-ids", default="0,1,2")
    args = ap.parse_args()

    cfg = Path(args.config)
    feat = Path(args.features)
    view_ids = [int(x.strip()) for x in args.view_ids.split(",") if x.strip()]

    score_orders(cfg, feat, view_ids)

if __name__ == "__main__":
    main()
