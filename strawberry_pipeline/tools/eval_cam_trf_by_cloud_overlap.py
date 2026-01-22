#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml


# ---------------- rotations (24 cube rotations) ----------------

def generate_axis_rotations_24() -> List[np.ndarray]:
    import itertools
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


# ---------------- Mecademic pose: R = Rz @ Ry @ Rx ----------------

def Rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

def Ry(b: float) -> np.ndarray:
    c, s = np.cos(b), np.sin(b)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

def Rz(g: float) -> np.ndarray:
    c, s = np.cos(g), np.sin(g)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

def meca_mobile_xyz_deg_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    return Rz(rz) @ Ry(ry) @ Rx(rx)


@dataclass(frozen=True)
class ViewPose:
    R_world_trf: np.ndarray      # (3,3)
    t_world_trf_m: np.ndarray    # (3,)


def load_robot_poses(cfg_path: Path) -> Dict[int, ViewPose]:
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    robot = raw.get("robot", {})
    views = robot.get("views", {})
    out: Dict[int, ViewPose] = {}
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
        R = meca_mobile_xyz_deg_to_R(rx, ry, rz)
        out[vid] = ViewPose(R_world_trf=R, t_world_trf_m=t)
    if not out:
        raise ValueError("No robot.views.*.pose_wrf_mm_deg found")
    return out


# ---------------- PLY loader (xyz only, ASCII) ----------------

def load_ply_xyz(path: Path) -> np.ndarray:
    txt = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(txt) and txt[i].strip() != "end_header":
        i += 1
    if i >= len(txt):
        raise ValueError(f"Not a valid ASCII PLY: {path}")
    pts = []
    for line in txt[i + 1 :]:
        s = line.strip().split()
        if len(s) < 3:
            continue
        pts.append([float(s[0]), float(s[1]), float(s[2])])
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


# ---------------- centroid LS fit for translation ----------------

def fit_t_trf_cam_for_plant(
    poses: Dict[int, ViewPose],
    centroids_cam: Dict[int, np.ndarray],
    R_trf_cam: np.ndarray,
    view_ids: List[int],
) -> np.ndarray:
    A_blocks = []
    b_blocks = []
    for vid in view_ids:
        if vid not in poses or vid not in centroids_cam:
            continue
        Rw = poses[vid].R_world_trf
        tw = poses[vid].t_world_trf_m
        pc = centroids_cam[vid].reshape(3,)
        ai = (Rw @ (R_trf_cam @ pc)) + tw
        Ai = np.hstack([np.eye(3), -Rw])
        A_blocks.append(Ai)
        b_blocks.append(ai.reshape(3, 1))

    if len(A_blocks) < 2:
        raise ValueError("Need at least 2 views with centroid + pose")

    A = np.vstack(A_blocks)
    b = np.vstack(b_blocks)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x[3:, 0]


# ---------------- NN mean distance without SciPy ----------------
# For each point in A, compute min ||a - b|| over B (chunked)

def nn_mean_distance(a: np.ndarray, b: np.ndarray, chunk: int = 2048) -> float:
    if a.size == 0 or b.size == 0:
        return float("inf")
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    s = 0.0
    n = a.shape[0]
    for i in range(0, n, chunk):
        aa = a[i : i + chunk]  # (m,3)
        # (m,1,3) - (1,k,3) -> (m,k,3) -> squared -> (m,k)
        d2 = np.sum((aa[:, None, :] - b[None, :, :]) ** 2, axis=2)
        mins = np.min(d2, axis=1)
        s += float(np.sum(np.sqrt(mins)))
    return s / float(n)


def downsample_random(pts: np.ndarray, n: int) -> np.ndarray:
    if pts.shape[0] <= n:
        return pts
    idx = np.random.choice(pts.shape[0], size=n, replace=False)
    return pts[idx]


def find_features_csv(plant_dir: Path) -> Path:
    # prefer plant_dir/features.csv
    p1 = plant_dir / "features.csv"
    if p1.exists():
        return p1
    # fallback: maybe plant_dir is outputs root and we passed plant_XXX wrong
    cand = list(plant_dir.glob("plant_*/features.csv"))
    if len(cand) == 1:
        return cand[0]
    raise FileNotFoundError(p1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--plant-dir", required=True, type=str,
                    help="Directory containing features.csv and raw_clouds/cloud_<vid>.ply (CAM frame)")
    ap.add_argument("--view-ids", default="0,1,2", type=str)
    ap.add_argument("--max-points", default=8000, type=int,
                    help="Downsample per view for speed (noscipy version is O(N*M))")
    ap.add_argument("--topk", default=5, type=int)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    plant_dir = Path(args.plant_dir)
    view_ids = [int(x.strip()) for x in args.view_ids.split(",") if x.strip()]

    poses = load_robot_poses(cfg_path)

    feat = find_features_csv(plant_dir)

    # parse delimiter
    txt = feat.read_text(encoding="utf-8", errors="replace").splitlines()
    delim = ";" if (";" in txt[0]) else ","
    import csv
    rows = list(csv.DictReader(txt, delimiter=delim))

    centroids_cam: Dict[int, np.ndarray] = {}
    # take largest instance per view (robust)
    best_n: Dict[int, int] = {}
    for r in rows:
        try:
            vid = int(r["view_id"])
            if vid not in view_ids:
                continue
            npt = int(float(r["num_points"]))
            cx, cy, cz = float(r["cx"]), float(r["cy"]), float(r["cz"])
        except Exception:
            continue
        if (vid not in best_n) or (npt > best_n[vid]):
            best_n[vid] = npt
            centroids_cam[vid] = np.array([cx, cy, cz], dtype=np.float64)

    # load clouds
    clouds_cam: Dict[int, np.ndarray] = {}
    for vid in view_ids:
        ply = plant_dir / "raw_clouds" / f"cloud_{vid}.ply"
        if not ply.exists():
            raise FileNotFoundError(ply)
        pts = load_ply_xyz(ply)
        pts = downsample_random(pts, int(args.max_points))
        clouds_cam[vid] = pts

    rots = generate_axis_rotations_24()

    scored: List[Tuple[float, np.ndarray, np.ndarray]] = []
    for R_trf_cam in rots:
        try:
            t_trf_cam = fit_t_trf_cam_for_plant(poses, centroids_cam, R_trf_cam, view_ids)
        except Exception:
            continue

        clouds_world: Dict[int, np.ndarray] = {}
        for vid in view_ids:
            Rw = poses[vid].R_world_trf
            tw = poses[vid].t_world_trf_m
            pc = clouds_cam[vid]
            pts_trf = (R_trf_cam @ pc.T).T + t_trf_cam
            pts_w = (Rw @ pts_trf.T).T + tw
            clouds_world[vid] = pts_w

        pair_scores = []
        for i in range(len(view_ids)):
            for j in range(i + 1, len(view_ids)):
                a = clouds_world[view_ids[i]]
                b = clouds_world[view_ids[j]]
                pair_scores.append(nn_mean_distance(a, b))
                pair_scores.append(nn_mean_distance(b, a))
        score = float(np.mean(pair_scores))
        scored.append((score, R_trf_cam, t_trf_cam))

    if not scored:
        raise RuntimeError("No candidates scored.")

    scored.sort(key=lambda x: x[0])

    print("\n=== Best by cloud-overlap (mean NN distance) ===")
    for k, (s, R, t) in enumerate(scored[: max(1, int(args.topk))], start=1):
        print(f"\n--- #{k} ---")
        print(f"score: {s*1000.0:.2f} mm")
        print("R_trf_cam:")
        print(np.array2string(R.astype(np.int32), separator=", "))
        t_mm = t * 1000.0
        print(f"t_trf_cam (mm): [{t_mm[0]:.3f}, {t_mm[1]:.3f}, {t_mm[2]:.3f}]")

    best_s, best_R, best_t = scored[0]
    t_mm = best_t * 1000.0
    print("\n=== Suggested YAML (for THIS plant) ===")
    print("robot:")
    print(f"  cam_axes_correction_R_trf_cam_row_major_3x3: [{', '.join(str(int(x)) for x in best_R.reshape(-1))}]")
    print(f"  camera_in_trf_translation_mm: [{t_mm[0]:.3f}, {t_mm[1]:.3f}, {t_mm[2]:.3f}]")
    print(f"\n(best overlap score: {best_s*1000.0:.2f} mm)")


if __name__ == "__main__":
    main()
