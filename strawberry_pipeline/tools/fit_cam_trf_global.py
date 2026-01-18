#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


# ---------- rotations ----------

def Rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float64)

def Ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)

def Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float64)

def meca_mobile_xyz_deg_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    # Mecademic intrinsic XYZ -> body->world: R = Rz(rz) @ Ry(ry) @ Rx(rx)
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    return Rz(rz) @ Ry(ry) @ Rx(rx)

def generate_axis_rotations_24() -> List[np.ndarray]:
    rots: List[np.ndarray] = []
    axes = [0, 1, 2]
    for perm in itertools.permutations(axes):
        P = np.zeros((3, 3), dtype=np.float64)
        for r, c in enumerate(perm):
            P[r, c] = 1.0
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    S = np.diag([sx, sy, sz])
                    R = S @ P
                    if np.linalg.det(R) > 0.5:
                        rots.append(R)
    uniq: List[np.ndarray] = []
    for R in rots:
        if not any(np.array_equal(R, U) for U in uniq):
            uniq.append(R)
    return uniq

def axis_mapping_str(R_trf_cam: np.ndarray) -> str:
    names = ["X", "Y", "Z"]
    out = []
    for j, ax in enumerate(names):
        col = R_trf_cam[:, j]
        i = int(np.argmax(np.abs(col)))
        sign = "+" if col[i] >= 0 else "-"
        out.append(f"{ax}_cam -> {sign}{names[i]}_trf")
    return ", ".join(out)


# ---------- data ----------

@dataclass(frozen=True)
class ViewPose:
    t_world_trf_m: np.ndarray  # (3,)
    R_world_trf: np.ndarray    # (3,3)

def load_robot_poses_from_yaml(cfg_path: Path) -> Dict[int, ViewPose]:
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    robot = raw.get("robot", {})
    views = robot.get("views", {})
    if not isinstance(views, dict) or not views:
        raise ValueError("robot.views missing or empty in YAML")

    out: Dict[int, ViewPose] = {}
    for vid_str, v in views.items():
        try:
            vid = int(vid_str)
        except Exception:
            continue
        pose = v.get("pose_wrf_mm_deg", None) if isinstance(v, dict) else None
        if not (isinstance(pose, list) and len(pose) == 6):
            continue
        x_mm, y_mm, z_mm, rx, ry, rz = [float(p) for p in pose]
        t = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0
        R = meca_mobile_xyz_deg_to_R(rx, ry, rz)
        out[vid] = ViewPose(t_world_trf_m=t, R_world_trf=R)

    if not out:
        raise ValueError("No valid robot.views.*.pose_wrf_mm_deg entries found")
    return out

@dataclass(frozen=True)
class Obs:
    plant_key: str
    view_id: int
    centroid_cam_m: np.ndarray
    num_points: int

def sniff_delim(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="replace").splitlines()[:8]
    sniff = "\n".join(sample)
    try:
        d = csv.Sniffer().sniff(sniff, delimiters=";,\t").delimiter
        return d
    except Exception:
        return ";"

def load_observations_from_features(features_csv: Path, instance_id: Optional[int]) -> List[Obs]:
    delim = sniff_delim(features_csv)
    rows = []
    with features_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter=delim)
        for row in r:
            rows.append(row)

    needed = {"view_id", "instance_id", "num_points", "cx", "cy", "cz"}
    if not rows or not needed.issubset(rows[0].keys()):
        raise ValueError(f"{features_csv} must have columns {sorted(needed)}")

    # per view: choose largest instance unless instance_id fixed
    by_view: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for row in rows:
        try:
            vid = int(row["view_id"])
            iid = int(row["instance_id"])
            npt = int(float(row["num_points"]))
            c = np.array([float(row["cx"]), float(row["cy"]), float(row["cz"])], dtype=np.float64)
        except Exception:
            continue
        if instance_id is not None and iid != int(instance_id):
            continue
        by_view.setdefault(vid, []).append((npt, c))

    out: List[Obs] = []
    for vid, items in by_view.items():
        npt, c = max(items, key=lambda x: x[0])
        out.append(Obs(
            plant_key=features_csv.parent.name,  # plant_000 ...
            view_id=vid,
            centroid_cam_m=c,
            num_points=int(npt),
        ))
    return out


# ---------- global solve ----------

@dataclass(frozen=True)
class GlobalResult:
    R_trf_cam: np.ndarray
    t_trf_cam_m: np.ndarray
    rms_m: float
    num_obs: int
    num_plants: int

def solve_global(
    poses: Dict[int, ViewPose],
    observations: List[Obs],
    R_trf_cam: np.ndarray,
    view_ids: List[int],
    weighted: bool = True,
) -> GlobalResult:
    # unknowns: p_world for each plant (3*K) + t_trf_cam (3)
    plant_keys = sorted({o.plant_key for o in observations})
    key_to_idx = {k: i for i, k in enumerate(plant_keys)}
    K = len(plant_keys)

    # build A x = b
    # equation for each obs (plant k, view i):
    # p_k - Rw_i * t = (Rw_i * (R_tc * p_cam)) + tw_i
    A_blocks = []
    b_blocks = []
    w_blocks = []

    use = [o for o in observations if o.view_id in view_ids and o.view_id in poses]
    if len(use) < 2:
        raise ValueError("Not enough observations")

    # weights
    if weighted:
        nmax = max(o.num_points for o in use)
    else:
        nmax = 1

    for o in use:
        Rw = poses[o.view_id].R_world_trf
        tw = poses[o.view_id].t_world_trf_m
        pc = o.centroid_cam_m.reshape(3,)

        rhs = (Rw @ (R_trf_cam @ pc)) + tw  # (3,)

        Ai = np.zeros((3, 3*K + 3), dtype=np.float64)
        k = key_to_idx[o.plant_key]
        Ai[:, 3*k:3*k+3] = np.eye(3)
        Ai[:, 3*K:3*K+3] = -Rw

        A_blocks.append(Ai)
        b_blocks.append(rhs.reshape(3, 1))

        w = np.sqrt(max(o.num_points, 1) / float(nmax)) if weighted else 1.0
        w_blocks.append(w)

    A = np.vstack(A_blocks)
    b = np.vstack(b_blocks)

    # apply weights by scaling each 3-row block
    for i, w in enumerate(w_blocks):
        A[3*i:3*i+3, :] *= w
        b[3*i:3*i+3, :] *= w

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    t = x[3*K:3*K+3, 0]

    # compute RMS in world (unweighted, physical)
    errs = []
    for o in use:
        Rw = poses[o.view_id].R_world_trf
        tw = poses[o.view_id].t_world_trf_m
        pc = o.centroid_cam_m.reshape(3,)
        k = key_to_idx[o.plant_key]
        p_k = x[3*k:3*k+3, 0]
        pred = (Rw @ (R_trf_cam @ pc + t)) + tw
        errs.append(pred - p_k)
    E = np.vstack(errs)
    rms = float(np.sqrt(np.mean(np.sum(E*E, axis=1))))

    return GlobalResult(
        R_trf_cam=R_trf_cam,
        t_trf_cam_m=t.astype(np.float64),
        rms_m=rms,
        num_obs=len(use),
        num_plants=K,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--features-glob", required=True, type=str, help="e.g. configs/outputs/plant_*/features.csv")
    ap.add_argument("--view-ids", default="0,1,2", type=str)
    ap.add_argument("--instance-id", default=None, type=int)
    ap.add_argument("--topk", default=5, type=int)
    ap.add_argument("--no-weight", action="store_true")
    args = ap.parse_args()

    cfg = Path(args.config)
    poses = load_robot_poses_from_yaml(cfg)
    view_ids = [int(x.strip()) for x in args.view_ids.split(",") if x.strip()]

    feats = sorted(Path().glob(args.features_glob))
    if not feats:
        raise FileNotFoundError(f"No files matched: {args.features_glob}")

    obs: List[Obs] = []
    for f in feats:
        obs.extend(load_observations_from_features(f, args.instance_id))

    rots = generate_axis_rotations_24()
    results: List[GlobalResult] = []
    for R in rots:
        try:
            res = solve_global(poses, obs, R, view_ids, weighted=(not args.no_weight))
            results.append(res)
        except Exception:
            continue

    if not results:
        raise RuntimeError("No global solution found")

    results.sort(key=lambda r: r.rms_m)

    print("\n=== Best global candidates ===")
    for i, r in enumerate(results[: max(1, args.topk)], start=1):
        t_mm = r.t_trf_cam_m * 1000.0
        print(f"\n--- #{i} ---")
        print(f"RMS: {r.rms_m*1000.0:.2f} mm  | obs={r.num_obs}  plants={r.num_plants}")
        print("Axis mapping:", axis_mapping_str(r.R_trf_cam))
        print("R_trf_cam:")
        print(np.array2string(r.R_trf_cam.astype(np.int32), separator=", "))
        print(f"t_trf_cam (mm): [{t_mm[0]:.3f}, {t_mm[1]:.3f}, {t_mm[2]:.3f}]")

    best = results[0]
    t_mm = best.t_trf_cam_m * 1000.0
    print("\n=== Suggested YAML snippet (GLOBAL) ===")
    print("robot:")
    print(f"  cam_axes_correction_R_trf_cam_row_major_3x3: [{', '.join(str(int(x)) for x in best.R_trf_cam.reshape(-1))}]")
    print(f"  camera_in_trf_translation_mm: [{t_mm[0]:.3f}, {t_mm[1]:.3f}, {t_mm[2]:.3f}]")


if __name__ == "__main__":
    main()
