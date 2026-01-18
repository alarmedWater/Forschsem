#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


# ---------------- Rotation helpers ----------------

def Rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float64)

def Ry(b: float) -> np.ndarray:
    c, s = np.cos(b), np.sin(b)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)

def Rz(g: float) -> np.ndarray:
    c, s = np.cos(g), np.sin(g)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float64)

def meca_mobile_xyz_deg_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """
    Mecademic: mobile/intrinsic XYZ (rotate about x, then new y, then new z).
    This corresponds to: R = Rz(rz) @ Ry(ry) @ Rx(rx)  (body->world)
    """
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    return Rz(rz) @ Ry(ry) @ Rx(rx)


# ---------------- Candidate rotations (24 proper axis rotations) ----------------

def generate_axis_rotations_24() -> List[np.ndarray]:
    """
    All proper rotations that map axes to axes (cube rotations): 24 matrices with det=+1.
    Each matrix has exactly one ±1 per row/col.
    """
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
                    d = np.linalg.det(R)
                    if d > 0.5:  # det ~ +1
                        rots.append(R)
    # remove duplicates (floating safe here because entries are exact -1/0/1)
    uniq: List[np.ndarray] = []
    for R in rots:
        if not any(np.array_equal(R, U) for U in uniq):
            uniq.append(R)
    return uniq


def axis_mapping_str(R_trf_cam: np.ndarray) -> str:
    """
    v_trf = R_trf_cam @ v_cam
    Columns tell where camera axes land in TRF.
    """
    names = ["X", "Y", "Z"]
    out = []
    for j, ax in enumerate(names):
        col = R_trf_cam[:, j]
        i = int(np.argmax(np.abs(col)))
        sign = "+" if col[i] >= 0 else "-"
        out.append(f"{ax}_cam -> {sign}{names[i]}_trf")
    return ", ".join(out)


# ---------------- Data loading ----------------

@dataclass(frozen=True)
class ViewPose:
    t_world_trf_m: np.ndarray  # (3,)
    R_world_trf: np.ndarray    # (3,3)

def load_robot_poses_from_yaml(cfg_path: Path) -> Dict[int, ViewPose]:
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping")

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
        if not isinstance(v, dict):
            continue
        pose = v.get("pose_wrf_mm_deg", None)
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
class ViewCentroid:
    view_id: int
    centroid_cam_m: np.ndarray  # (3,)
    num_points: int

def load_centroids_from_features_csv(features_csv: Path, instance_id: Optional[int]) -> Dict[int, ViewCentroid]:
    if not features_csv.exists():
        raise FileNotFoundError(features_csv)

    # try to sniff delimiter
    sample = features_csv.read_text(encoding="utf-8", errors="replace").splitlines()[:5]
    sniff = "\n".join(sample)
    try:
        dialect = csv.Sniffer().sniff(sniff, delimiters=";,\t")
        delim = dialect.delimiter
    except Exception:
        delim = ";"

    rows: List[dict] = []
    with features_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter=delim)
        for row in r:
            rows.append(row)

    needed = {"view_id", "instance_id", "num_points", "cx", "cy", "cz"}
    if not rows or not needed.issubset(set(rows[0].keys())):
        raise ValueError(f"features.csv must have columns {sorted(needed)} (got {list(rows[0].keys()) if rows else 'no rows'})")

    by_view: Dict[int, List[ViewCentroid]] = {}
    for row in rows:
        try:
            vid = int(row["view_id"])
            iid = int(row["instance_id"])
            npt = int(float(row["num_points"]))
            cx = float(row["cx"]); cy = float(row["cy"]); cz = float(row["cz"])
        except Exception:
            continue

        if instance_id is not None and iid != int(instance_id):
            continue

        vc = ViewCentroid(vid, np.array([cx, cy, cz], dtype=np.float64), npt)
        by_view.setdefault(vid, []).append(vc)

    out: Dict[int, ViewCentroid] = {}
    for vid, cands in by_view.items():
        # if multiple, take largest (robust when segmentation produces extras)
        best = max(cands, key=lambda x: x.num_points)
        out[vid] = best

    if not out:
        msg = "No centroids loaded. "
        msg += "Try without --instance-id (largest per view) or check your features.csv delimiter."
        raise ValueError(msg)
    return out


# ---------------- Solve best fixed T_trf<-cam ----------------

@dataclass(frozen=True)
class Result:
    R_trf_cam: np.ndarray
    t_trf_cam_m: np.ndarray
    p_world_m: np.ndarray
    rms_m: float

def solve_best_translation_for_rotation(
    poses: Dict[int, ViewPose],
    cents: Dict[int, ViewCentroid],
    R_trf_cam: np.ndarray,
    view_ids: List[int],
) -> Result:
    """
    Model:
      p_world = R_world_trf[i] @ ( R_trf_cam @ p_cam[i] + t_trf_cam ) + t_world_trf[i]
    Unknowns:
      p_world (3) and t_trf_cam (3)
    Stack linear system:
      [ I , -R_world_trf[i] ] [p_world; t_trf_cam] = a_i
      where a_i = R_world_trf[i] @ R_trf_cam @ p_cam[i] + t_world_trf[i]
    """
    A_blocks = []
    b_blocks = []

    for vid in view_ids:
        if vid not in poses or vid not in cents:
            continue
        Rw = poses[vid].R_world_trf
        tw = poses[vid].t_world_trf_m
        pc = cents[vid].centroid_cam_m.reshape(3,)

        ai = (Rw @ (R_trf_cam @ pc)) + tw  # (3,)
        # 3x6 block
        Ai = np.hstack([np.eye(3), -Rw])
        A_blocks.append(Ai)
        b_blocks.append(ai.reshape(3, 1))

    if len(A_blocks) < 2:
        raise ValueError("Need at least 2 views with both pose and centroid")

    A = np.vstack(A_blocks)          # (3N, 6)
    b = np.vstack(b_blocks)          # (3N, 1)

    x, *_ = np.linalg.lstsq(A, b, rcond=None)  # (6,1)
    p_world = x[:3, 0]
    t_trf_cam = x[3:, 0]

    # compute residuals
    errs = []
    for vid in view_ids:
        if vid not in poses or vid not in cents:
            continue
        Rw = poses[vid].R_world_trf
        tw = poses[vid].t_world_trf_m
        pc = cents[vid].centroid_cam_m.reshape(3,)
        pred = (Rw @ (R_trf_cam @ pc + t_trf_cam)) + tw
        errs.append(pred - p_world)

    E = np.vstack(errs)  # (N,3)
    rms = float(np.sqrt(np.mean(np.sum(E * E, axis=1))))
    return Result(R_trf_cam=R_trf_cam, t_trf_cam_m=t_trf_cam, p_world_m=p_world, rms_m=rms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="Path to YAML config with robot.views.*.pose_wrf_mm_deg")
    ap.add_argument("--features", required=True, type=str, help="Path to outputs/plant_XXX/features.csv")
    ap.add_argument("--view-ids", type=str, default="0,1,2", help="Comma list of view ids to use, default 0,1,2")
    ap.add_argument("--instance-id", type=int, default=None, help="Optional instance_id to force (else: largest per view)")
    ap.add_argument("--topk", type=int, default=5, help="Print top-k candidates")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    feat_path = Path(args.features)
    view_ids = [int(x.strip()) for x in args.view_ids.split(",") if x.strip()]

    poses = load_robot_poses_from_yaml(cfg_path)
    cents = load_centroids_from_features_csv(feat_path, args.instance_id)

    missing = [vid for vid in view_ids if vid not in poses]
    if missing:
        print(f"[WARN] Missing robot pose for views: {missing}")
    missing_c = [vid for vid in view_ids if vid not in cents]
    if missing_c:
        print(f"[WARN] Missing centroid for views: {missing_c}")

    rots = generate_axis_rotations_24()
    results: List[Result] = []
    for R_tc in rots:
        try:
            res = solve_best_translation_for_rotation(poses, cents, R_tc, view_ids)
            results.append(res)
        except Exception:
            continue

    if not results:
        raise RuntimeError("No valid solutions found (check inputs).")

    results.sort(key=lambda r: r.rms_m)

    print("\n=== Centroids used (camera frame, meters) ===")
    for vid in view_ids:
        if vid in cents:
            c = cents[vid]
            print(f"  view {vid}: centroid_cam = {c.centroid_cam_m.tolist()}  (num_points={c.num_points})")

    print("\n=== Best candidates (lower RMS is better) ===")
    for k, r in enumerate(results[: max(1, args.topk)], start=1):
        t_mm = (r.t_trf_cam_m * 1000.0).tolist()
        print(f"\n--- #{k} ---")
        print(f"RMS alignment error: {r.rms_m*1000.0:.2f} mm")
        print("Axis mapping:", axis_mapping_str(r.R_trf_cam))
        print("R_trf_cam (row-major):")
        print(np.array2string(r.R_trf_cam.astype(np.int32), separator=", "))
        print(f"t_trf_cam (mm): [{t_mm[0]:.3f}, {t_mm[1]:.3f}, {t_mm[2]:.3f}]")
        print(f"estimated common strawberry centroid in WORLD (m): {r.p_world_m.tolist()}")

    best = results[0]
    t_mm = (best.t_trf_cam_m * 1000.0)
    print("\n=== Suggested YAML snippet ===")
    print("robot:")
    print("  # fixed correction so that v_trf = R_trf_cam @ v_cam, and p_trf = R_trf_cam@p_cam + t_trf_cam")
    print(f"  cam_axes_correction_R_trf_cam_row_major_3x3: [{', '.join(str(int(x)) for x in best.R_trf_cam.reshape(-1))}]")
    print(f"  camera_in_trf_translation_mm: [{t_mm[0]:.3f}, {t_mm[1]:.3f}, {t_mm[2]:.3f}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
