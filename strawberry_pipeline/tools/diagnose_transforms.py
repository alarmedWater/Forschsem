#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_transforms.py (batch + chain-aware + bridge ICP + pose-vs-ICP + handeye candidates + freeze_cam)

Ziel:
- systematisch eingrenzen, warum View-Clouds im WORLD "random" liegen
- batch über mehrere plants, optional mehrere instances (falls exportiert)

Wichtige Erkenntnis aus deinem Log / deiner Beobachtung:
- cloud_*_cam und cloud_*_trf sind sehr ähnlich  => CAM->TRF (R_trf_cam) ist sehr wahrscheinlich OK
- cloud_*_world ist "random"                     => TRF->WORLD (Pose-Interpretation) ist sehr wahrscheinlich falsch

Neu:
- --freeze_cam:
    Fixiert CAM->TRF exakt auf YAML (keine zusätzliche Cam-Search, kein cam_flip, kein cam_t)
    => Fokus auf Pose / Euler / mm<->m / Signs / invert_pose
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


# ============================================================
# PLY I/O
# ============================================================

def read_ply_xyz(path: Path) -> np.ndarray:
    """
    Minimal PLY reader for vertex x y z (float), ASCII or binary_little_endian.
    Returns (N,3) float32.
    Designed for your pipeline export (x,y,z float32).
    """
    with path.open("rb") as f:
        fmt = None
        num_verts = None
        props: List[str] = []
        in_vertex = False

        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY (EOF before end_header): {path}")
            s = line.decode("ascii", errors="ignore").strip()

            if s.startswith("format "):
                fmt = s.split()[1]
            elif s.startswith("element vertex "):
                num_verts = int(s.split()[-1])
                in_vertex = True
                props = []
            elif s.startswith("element ") and not s.startswith("element vertex"):
                in_vertex = False
            elif s.startswith("property ") and in_vertex:
                parts = s.split()
                if len(parts) >= 3:
                    props.append(parts[-1])
            elif s == "end_header":
                break

        if fmt is None or num_verts is None:
            raise ValueError(f"PLY missing format/vertex count: {path}")

        try:
            ix = props.index("x")
            iy = props.index("y")
            iz = props.index("z")
        except ValueError:
            raise ValueError(f"PLY has no x/y/z properties: {path} props={props}")

        if fmt == "ascii":
            rest = f.read().decode("utf-8", errors="ignore").splitlines()
            pts = np.zeros((num_verts, 3), dtype=np.float32)
            for i in range(num_verts):
                parts = rest[i].split()
                pts[i, 0] = float(parts[ix])
                pts[i, 1] = float(parts[iy])
                pts[i, 2] = float(parts[iz])
            return pts

        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format '{fmt}': {path}")

        stride = len(props) * 4
        raw = f.read(num_verts * stride)
        if len(raw) < num_verts * stride:
            raise ValueError(f"PLY binary too short: {path}")

        data = np.frombuffer(raw, dtype="<f4").reshape((num_verts, len(props)))
        pts = data[:, [ix, iy, iz]].astype(np.float32, copy=False)
        return pts


def write_ply_xyzrgb_ascii(path: Path, pts: np.ndarray, rgb_u8: np.ndarray) -> None:
    pts = np.asarray(pts, dtype=np.float32).reshape((-1, 3))
    rgb_u8 = np.asarray(rgb_u8, dtype=np.uint8).reshape((-1, 3))
    if pts.shape[0] != rgb_u8.shape[0]:
        raise ValueError("pts and rgb size mismatch")

    n = pts.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    # Write as one big string (fast enough for debug sizes)
    out_lines = [header]
    for p, c in zip(pts, rgb_u8):
        out_lines.append(
            f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n"
        )
    path.write_text("".join(out_lines), encoding="utf-8")


# ============================================================
# Math / Rotations / SE(3)
# ============================================================

def rotx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float64)


def roty(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)


def rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float64)


AXMAT = {"x": rotx, "y": roty, "z": rotz}


def compose_euler(rx: float, ry: float, rz: float,
                  order: Tuple[str, str, str],
                  mode: str) -> np.ndarray:
    """
    order: tuple like ("z","y","x") for RzRyRx
    mode:
      - written_extrinsic: R = Ra @ Rb @ Rc   (as written)
      - reverse_extrinsic: R = Rc @ Rb @ Ra
      - intrinsic: treated as intrinsic rotations; here equal to reverse_extrinsic
    """
    mats = {"x": AXMAT["x"](rx), "y": AXMAT["y"](ry), "z": AXMAT["z"](rz)}
    a, b, c = order
    if mode == "written_extrinsic":
        return mats[a] @ mats[b] @ mats[c]
    if mode in ("reverse_extrinsic", "intrinsic"):
        return mats[c] @ mats[b] @ mats[a]
    raise ValueError(f"Unknown mode {mode}")


def rotation_sanity(R: np.ndarray) -> Tuple[float, float]:
    R = np.asarray(R, dtype=np.float64).reshape((3, 3))
    err = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
    det = float(np.linalg.det(R))
    return err, det


def angle_from_R(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64).reshape((3, 3))
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.acos(tr))


def T_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape((3, 3))
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape((3,))
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def T_delta_stats(T: np.ndarray) -> Tuple[float, float]:
    """
    Returns (rot_deg, trans_mm) of a transform close to identity.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = float(np.arccos(tr)) * 180.0 / np.pi
    trans_mm = float(np.linalg.norm(t)) * 1000.0
    return ang, trans_mm


# ============================================================
# Point cloud utils (downsample / NN / ICP)
# ============================================================

def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape((-1, 3))
    if pts.shape[0] == 0 or voxel <= 0:
        return pts
    q = np.floor(pts / float(voxel)).astype(np.int32)
    _, idx = np.unique(q, axis=0, return_index=True)
    return pts[idx]


def _nn_backend():
    try:
        from scipy.spatial import cKDTree  # type: ignore
        return ("scipy", cKDTree)
    except Exception:
        pass
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        return ("sklearn", NearestNeighbors)
    except Exception:
        pass
    return ("brute", None)


def nn_query(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=np.float32).reshape((-1, 3))
    B = np.asarray(B, dtype=np.float32).reshape((-1, 3))
    if A.shape[0] == 0 or B.shape[0] == 0:
        return (
            np.full((A.shape[0],), np.inf, dtype=np.float64),
            np.full((A.shape[0],), -1, dtype=np.int64),
        )

    kind, backend = _nn_backend()
    if kind == "scipy":
        tree = backend(B)  # type: ignore
        d, idx = tree.query(A, k=1)
        return np.asarray(d, dtype=np.float64), np.asarray(idx, dtype=np.int64)

    if kind == "sklearn":
        nn = backend(n_neighbors=1, algorithm="auto")  # type: ignore
        nn.fit(B)
        d, idx = nn.kneighbors(A, return_distance=True)
        return d[:, 0].astype(np.float64), idx[:, 0].astype(np.int64)

    # brute (slow, but fine as fallback)
    D = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)).astype(np.float64)
    idx = D.argmin(axis=1).astype(np.int64)
    d = D[np.arange(D.shape[0]), idx]
    return d, idx


def nn_dist_stats(A: np.ndarray, B: np.ndarray, sample_n: int) -> Tuple[float, float]:
    A = np.asarray(A, dtype=np.float32).reshape((-1, 3))
    B = np.asarray(B, dtype=np.float32).reshape((-1, 3))
    if A.size == 0 or B.size == 0:
        return float("inf"), float("inf")

    if A.shape[0] > sample_n:
        A = A[np.random.choice(A.shape[0], size=sample_n, replace=False)]
    if B.shape[0] > sample_n:
        B = B[np.random.choice(B.shape[0], size=sample_n, replace=False)]

    d, _ = nn_query(A, B)
    med = float(np.median(d))
    p90 = float(np.quantile(d, 0.9))
    return med, p90


def overlap_ratio(A: np.ndarray, B: np.ndarray, thr_m: float, sample_n: int = 4000) -> float:
    A = np.asarray(A, dtype=np.float32).reshape((-1, 3))
    B = np.asarray(B, dtype=np.float32).reshape((-1, 3))
    if A.size == 0 or B.size == 0:
        return 0.0
    if A.shape[0] > sample_n:
        A = A[np.random.choice(A.shape[0], size=sample_n, replace=False)]
    d, _ = nn_query(A, B)
    return float(np.mean(d < float(thr_m)))


def rigid_fit(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=np.float64).reshape((-1, 3))
    B = np.asarray(B, dtype=np.float64).reshape((-1, 3))
    if A.shape[0] != B.shape[0] or A.shape[0] < 3:
        return np.eye(3), np.zeros(3)

    ca = A.mean(axis=0)
    cb = B.mean(axis=0)
    AA = A - ca
    BB = B - cb
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    return R, t


def icp_point_to_point(
    src: np.ndarray,
    dst: np.ndarray,
    init_R: Optional[np.ndarray] = None,
    init_t: Optional[np.ndarray] = None,
    max_iter: int = 35,
    tol: float = 1e-5,
    sample_n: int = 4000,
    max_corr_dist: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    src = np.asarray(src, dtype=np.float32).reshape((-1, 3))
    dst = np.asarray(dst, dtype=np.float32).reshape((-1, 3))

    if src.shape[0] == 0 or dst.shape[0] == 0:
        return np.eye(3), np.zeros(3), float("inf"), 0.0

    if src.shape[0] > sample_n:
        src_s = src[np.random.choice(src.shape[0], size=sample_n, replace=False)]
    else:
        src_s = src

    if dst.shape[0] > sample_n:
        dst_s = dst[np.random.choice(dst.shape[0], size=sample_n, replace=False)]
    else:
        dst_s = dst

    R = np.eye(3, dtype=np.float64) if init_R is None else np.asarray(init_R, dtype=np.float64).reshape((3, 3))
    t = np.zeros(3, dtype=np.float64) if init_t is None else np.asarray(init_t, dtype=np.float64).reshape((3,))

    prev_rmse = None
    for _ in range(max_iter):
        X = (R @ src_s.T).T + t
        d, idx = nn_query(X.astype(np.float32), dst_s.astype(np.float32))

        inl = d < float(max_corr_dist)
        if np.count_nonzero(inl) < 20:
            rmse = float(np.sqrt(np.mean(d**2)))
            return R, t, rmse, float(np.count_nonzero(inl) / max(1, d.size))

        A = X[inl].astype(np.float64)
        B = dst_s[idx[inl]].astype(np.float64)

        dR, dt = rigid_fit(A, B)
        R = dR @ R
        t = dR @ t + dt

        rmse = float(np.sqrt(np.mean((d[inl]) ** 2)))
        if prev_rmse is not None and abs(prev_rmse - rmse) < tol:
            break
        prev_rmse = rmse

    X = (R @ src_s.T).T + t
    d, _ = nn_query(X.astype(np.float32), dst_s.astype(np.float32))
    inl = d < float(max_corr_dist)
    rmse = float(np.sqrt(np.mean((d[inl]) ** 2))) if np.any(inl) else float(np.sqrt(np.mean(d**2)))
    inlier_ratio = float(np.count_nonzero(inl) / max(1, d.size))
    return R, t, rmse, inlier_ratio


# ============================================================
# YAML loading + clouds
# ============================================================

def load_yaml(path: Path) -> Dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping")
    return raw


def load_view_clouds_cam(out_root: Path, plant_id: int, view_ids: List[int], instance_id: Optional[int]) -> Dict[int, np.ndarray]:
    """
    Loads CAM clouds.

    Supported filenames:
      - cloud_{vid}_cam.ply
      - cloud_{vid}_inst{instance}_cam.ply
      - cloud_{vid}_cam_inst{instance}.ply
    """
    plant_dir = out_root / f"plant_{plant_id:03d}" / "raw_clouds"
    clouds: Dict[int, np.ndarray] = {}

    for vid in view_ids:
        if instance_id is None:
            candidates = [plant_dir / f"cloud_{vid}_cam.ply"]
        else:
            candidates = [
                plant_dir / f"cloud_{vid}_inst{instance_id}_cam.ply",
                plant_dir / f"cloud_{vid}_cam_inst{instance_id}.ply",
                plant_dir / f"cloud_{vid}_cam.ply",  # fallback
            ]

        p = next((c for c in candidates if c.exists()), None)
        if p is not None:
            clouds[vid] = read_ply_xyz(p)

    return clouds


def load_centroids_from_features(out_root: Path, plant_id: int, instance_id: int = 1) -> Dict[int, np.ndarray]:
    fcsv = out_root / f"plant_{plant_id:03d}" / "features.csv"
    if not fcsv.exists():
        return {}

    rows = fcsv.read_text(encoding="utf-8").splitlines()
    if not rows:
        return {}

    header = rows[0].split(";")
    idx = {k: i for i, k in enumerate(header)}

    out: Dict[int, np.ndarray] = {}
    for line in rows[1:]:
        if not line.strip():
            continue
        p = line.split(";")
        pid = int(p[idx["plant_id"]])
        if pid != plant_id:
            continue
        vid = int(p[idx["view_id"]])
        inst = int(p[idx["instance_id"]])
        if inst != int(instance_id):
            continue
        cx = float(p[idx["cx"]])
        cy = float(p[idx["cy"]])
        cz = float(p[idx["cz"]])
        out[vid] = np.array([cx, cy, cz], dtype=np.float32)
    return out


# ============================================================
# Search space
# ============================================================

EULER_ORDERS = {
    "RzRyRx": ("z", "y", "x"),
    "RxRyRz": ("x", "y", "z"),
    "RzRxRy": ("z", "x", "y"),
    "RyRxRz": ("y", "x", "z"),
    "RyRzRx": ("y", "z", "x"),
    "RxRzRy": ("x", "z", "y"),
}

EULER_MODES = ["written_extrinsic", "reverse_extrinsic", "intrinsic"]

ANGLE_MAPS = {
    "xyz": (0, 1, 2),
    "xzy": (0, 2, 1),
    "yxz": (1, 0, 2),
    "yzx": (1, 2, 0),
    "zxy": (2, 0, 1),
    "zyx": (2, 1, 0),
}


@dataclass(frozen=True)
class Candidate:
    euler_order: str
    euler_mode: str
    angle_map: str
    neg_angles: Tuple[int, int, int]
    invert_pose: bool
    t_sign: Tuple[int, int, int]
    t_scale: float
    cam_corr_id: int
    cam_corr_name: str
    cam_axis_flip: Tuple[int, int, int]
    use_cam_t: bool


# ============================================================
# Transform chain + scoring + reporting
# ============================================================

def build_pose_Rt(view_pose_wrf_mm_deg: List[float], cand: Candidate) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds (R, t) for WORLD<-TRF based on candidate settings.
    """
    x, y, z, rx_deg, ry_deg, rz_deg = [float(v) for v in view_pose_wrf_mm_deg]

    # translation scaling + sign
    t = np.array([x, y, z], dtype=np.float64) * float(cand.t_scale)
    t = t * np.array(cand.t_sign, dtype=np.float64)

    # angles: apply permutation (angle_map), then sign
    vals_deg = np.array([rx_deg, ry_deg, rz_deg], dtype=np.float64)
    perm = ANGLE_MAPS[cand.angle_map]
    vals_deg = vals_deg[list(perm)]
    vals_deg = vals_deg * np.array(cand.neg_angles, dtype=np.float64)

    rx, ry, rz = np.deg2rad(vals_deg).tolist()
    order_axes = EULER_ORDERS[cand.euler_order]
    R = compose_euler(rx, ry, rz, order_axes, cand.euler_mode)

    if cand.invert_pose:
        # invert the WORLD<-TRF you just built (common when pose is "TRF in WORLD" vs "WORLD in TRF")
        R = R.T
        t = -R @ t

    return R.astype(np.float64), t.astype(np.float64)


def apply_chain_points_world(
    points_cam: np.ndarray,
    Rw: np.ndarray,
    tw: np.ndarray,
    R_trf_cam: np.ndarray,
    t_trf_cam_m: np.ndarray,
    cand: Candidate,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    CAM -> (cam_axis_flip) -> TRF via R_trf_cam (+ optional t_trf_cam) -> WORLD via (Rw, tw)
    Returns:
      points_world (N,3)
      forward_dot
      cam_origin_world (3,)
    """
    pts = np.asarray(points_cam, dtype=np.float64).reshape((-1, 3))
    if pts.size == 0:
        return pts.astype(np.float32), float("nan"), np.zeros(3, dtype=np.float64)

    # flip in CAM (diag only)
    F = np.diag(np.array(cand.cam_axis_flip, dtype=np.float64))
    pts = (F @ pts.T).T

    # CAM -> TRF
    if cand.use_cam_t:
        pts_trf = (R_trf_cam @ pts.T).T + t_trf_cam_m
        cam_origin_trf = t_trf_cam_m.copy()
    else:
        pts_trf = (R_trf_cam @ pts.T).T
        cam_origin_trf = np.zeros(3, dtype=np.float64)

    # TRF -> WORLD
    pts_w = (Rw @ pts_trf.T).T + tw

    # forward dot sanity (camera forward should point roughly towards the object)
    f_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    f_cam = F @ f_cam
    f_trf = R_trf_cam @ f_cam
    f_world = Rw @ f_trf

    cam_origin_world = (Rw @ cam_origin_trf) + tw
    centroid_world = pts_w.mean(axis=0)
    v = centroid_world - cam_origin_world
    f_dot = float(np.dot(f_world, v))

    return pts_w.astype(np.float32), f_dot, cam_origin_world


def summarize_cloud(pts: np.ndarray) -> Dict[str, float]:
    pts = np.asarray(pts, dtype=np.float32).reshape((-1, 3))
    if pts.shape[0] == 0:
        return {"n": 0.0}
    c = pts.mean(axis=0)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    ext = mx - mn
    return {
        "n": float(pts.shape[0]),
        "cx": float(c[0]), "cy": float(c[1]), "cz": float(c[2]),
        "ex": float(ext[0]), "ey": float(ext[1]), "ez": float(ext[2]),
        "diam": float(np.linalg.norm(ext)),
    }


def make_pairs(view_ids: List[int], mode: str) -> List[Tuple[int, int]]:
    vids = sorted([int(v) for v in view_ids])
    if len(vids) < 2:
        return []
    if mode == "chain":
        return [(vids[i], vids[i + 1]) for i in range(len(vids) - 1)]
    if mode == "all":
        out = []
        for i in range(len(vids)):
            for j in range(i + 1, len(vids)):
                out.append((vids[i], vids[j]))
        return out
    raise ValueError("pair_mode must be 'chain' or 'all'")


def score_overlap_alignment(
    world_clouds: Dict[int, np.ndarray],
    forward_dots: Dict[int, float],
    view_ids: List[int],
    pair_mode: str,
    sample_n: int,
    forward_penalty: float,
    aabb_penalty: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Lower score is better. (overlap-based proxy)
    """
    vids_present = sorted(world_clouds.keys())
    if len(vids_present) < 2:
        return float("inf"), {"reason": 1.0}

    centroids = np.stack([world_clouds[v].mean(axis=0) for v in vids_present], axis=0)
    spread_xyz = np.ptp(centroids, axis=0)
    spread_norm = float(np.linalg.norm(spread_xyz))

    pairs = make_pairs([v for v in view_ids if v in world_clouds], mode=pair_mode)
    pair_meds, pair_p90 = [], []
    for (i, j) in pairs:
        A, B = world_clouds[i], world_clouds[j]
        med1, p901 = nn_dist_stats(A, B, sample_n=sample_n)
        med2, p902 = nn_dist_stats(B, A, sample_n=sample_n)
        pair_meds.append(0.5 * (med1 + med2))
        pair_p90.append(0.5 * (p901 + p902))

    nn_med = float(np.median(pair_meds)) if pair_meds else float("inf")
    nn_p90 = float(np.median(pair_p90)) if pair_p90 else float("inf")

    bad_forward = sum(1 for v in vids_present if not np.isfinite(forward_dots[v]) or forward_dots[v] <= 0.0)
    forward_pen = forward_penalty * float(bad_forward)

    def aabb(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        return mn, mx

    aabb_pen = 0.0
    boxes = {v: aabb(world_clouds[v]) for v in vids_present}
    for (i, j) in pairs:
        mn1, mx1 = boxes[i]
        mn2, mx2 = boxes[j]
        inter = np.maximum(0.0, np.minimum(mx1, mx2) - np.maximum(mn1, mn2))
        size1 = np.maximum(1e-9, mx1 - mn1)
        size2 = np.maximum(1e-9, mx2 - mn2)
        ratio = float(np.min(inter / np.minimum(size1, size2)))
        if ratio < 0.05:
            aabb_pen += aabb_penalty

    score = (
        2.0 * spread_norm +
        8.0 * nn_med +
        4.0 * nn_p90 +
        forward_pen +
        aabb_pen
    )

    details = {
        "spread_norm": spread_norm,
        "spread_x": float(spread_xyz[0]),
        "spread_y": float(spread_xyz[1]),
        "spread_z": float(spread_xyz[2]),
        "nn_med": nn_med,
        "nn_p90": nn_p90,
        "bad_forward": float(bad_forward),
        "forward_pen": forward_pen,
        "aabb_pen": aabb_pen,
        "score_overlap": score,
    }
    return score, details


def free_icp_compute_pairs(
    clouds_cam: Dict[int, np.ndarray],
    pairs: List[Tuple[int, int]],
    voxel: float,
    sample_n: int,
    max_corr: float,
) -> Dict[Tuple[int, int], np.ndarray]:
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for (i, j) in pairs:
        if i not in clouds_cam or j not in clouds_cam:
            continue
        a = voxel_downsample(clouds_cam[i], voxel)
        b = voxel_downsample(clouds_cam[j], voxel)
        R, t, _, _ = icp_point_to_point(a, b, max_iter=35, sample_n=sample_n, max_corr_dist=max_corr)
        out[(i, j)] = T_from_Rt(R, t)
    return out


def print_free_icp_report(
    clouds_cam: Dict[int, np.ndarray],
    view_ids: List[int],
    pair_mode: str,
    voxel: float,
    sample_n: int,
    max_corr: float,
    bridge_via: Optional[int] = None,
) -> Dict[Tuple[int, int], np.ndarray]:
    vids = [v for v in sorted(view_ids) if v in clouds_cam]
    if len(vids) < 2:
        print("[FREE-ICP] Not enough CAM clouds for free ICP.")
        return {}

    pairs = make_pairs(vids, mode=pair_mode)

    print("\n" + "=" * 90)
    print("FREE-ICP CHECK (ignores robot poses) -> tests if clouds are same object")
    print("=" * 90)

    for v in vids:
        pts = voxel_downsample(clouds_cam[v], voxel)
        s = summarize_cloud(pts)
        print(
            f"[view {v}] n={int(s['n'])} diam={s['diam']:.3f} "
            f"ext=({s['ex']:.3f},{s['ey']:.3f},{s['ez']:.3f}) "
            f"centroid=({s['cx']:.3f},{s['cy']:.3f},{s['cz']:.3f})"
        )

    icp_T: Dict[Tuple[int, int], np.ndarray] = {}
    for (i, j) in pairs:
        a = voxel_downsample(clouds_cam[i], voxel)
        b = voxel_downsample(clouds_cam[j], voxel)

        ov5_ab = overlap_ratio(a, b, thr_m=0.005, sample_n=sample_n)
        ov10_ab = overlap_ratio(a, b, thr_m=0.010, sample_n=sample_n)
        ov5_ba = overlap_ratio(b, a, thr_m=0.005, sample_n=sample_n)
        ov10_ba = overlap_ratio(b, a, thr_m=0.010, sample_n=sample_n)

        R, t, rmse, inl = icp_point_to_point(a, b, max_iter=35, sample_n=sample_n, max_corr_dist=max_corr)
        ang = math.degrees(angle_from_R(R))
        print(
            f"[free ICP] {i} -> {j} : rmse={rmse*1000:.2f}mm inlier={inl*100:.1f}% | "
            f"rot={ang:.1f}deg | t=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f}) | "
            f"ov@5mm A->B={ov5_ab*100:.1f}% B->A={ov5_ba*100:.1f}% ov@10mm A->B={ov10_ab*100:.1f}% B->A={ov10_ba*100:.1f}%"
        )
        icp_T[(i, j)] = T_from_Rt(R, t)

    if bridge_via is not None and len(vids) >= 3:
        left = vids[0]
        right = vids[-1]
        mid = int(bridge_via)
        if left in clouds_cam and mid in clouds_cam and right in clouds_cam:
            key_lm = (left, mid)
            key_mr = (mid, right)

            if key_lm not in icp_T:
                icp_T.update(free_icp_compute_pairs(clouds_cam, [key_lm], voxel, sample_n, max_corr))
            if key_mr not in icp_T:
                icp_T.update(free_icp_compute_pairs(clouds_cam, [key_mr], voxel, sample_n, max_corr))

            if key_lm in icp_T and key_mr in icp_T:
                T_mid_from_left = icp_T[key_lm]
                T_right_from_mid = icp_T[key_mr]
                T_init_right_from_left = T_right_from_mid @ T_mid_from_left

                a = voxel_downsample(clouds_cam[left], voxel)
                b = voxel_downsample(clouds_cam[right], voxel)

                R0 = T_init_right_from_left[:3, :3]
                t0 = T_init_right_from_left[:3, 3]

                R, t, rmse, inl = icp_point_to_point(
                    a, b,
                    init_R=R0, init_t=t0,
                    max_iter=35, sample_n=sample_n, max_corr_dist=max_corr
                )
                ang = math.degrees(angle_from_R(R))
                print("\n" + "-" * 90)
                print(f"[bridge ICP] {left} -> {right} (init via {mid}) : rmse={rmse*1000:.2f}mm inlier={inl*100:.1f}% | rot={ang:.1f}deg | t=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})")
                print("-" * 90)

    print("\nInterpretation:")
    print("- Wenn rmse ~ wenige mm bis 1-2cm und inlier hoch: Clouds sind kompatibel (selbe Erdbeere), Problem ist Pose/Frame-Interpretation.")
    print("- Wenn rmse groß (z.B. >3-5cm) oder inlier sehr niedrig: meistens NICHT dieselbe Erdbeere / Maske falsch / Depth-Müll / ID inkonsistent.")
    print("- Bei 3 Views: 0<->2 kann legitimerweise schlechter sein (kleiner Overlap). Nutze BRIDGE über View 1.")
    return icp_T


def predicted_T_camj_from_cami(
    pose6_i: List[float],
    pose6_j: List[float],
    cand: Candidate,
    cam_corr_mat: np.ndarray,
    t_trf_cam_m: np.ndarray,
) -> np.ndarray:
    """
    Computes T_pred_camj_cami = inv(T_world_cam_j) @ T_world_cam_i

    Includes cam_axis_flip as diag(F) and cam_corr_mat may include permutations (handeye/full).
    """
    Rw_i, tw_i = build_pose_Rt(pose6_i, cand)
    Rw_j, tw_j = build_pose_Rt(pose6_j, cand)

    F = np.diag(np.array(cand.cam_axis_flip, dtype=np.float64))
    R_trf_cam_eff = cam_corr_mat @ F
    t_eff = t_trf_cam_m if cand.use_cam_t else np.zeros(3, dtype=np.float64)

    T_world_trf_i = T_from_Rt(Rw_i, tw_i)
    T_world_trf_j = T_from_Rt(Rw_j, tw_j)
    T_trf_cam = T_from_Rt(R_trf_cam_eff, t_eff)

    T_world_cam_i = T_world_trf_i @ T_trf_cam
    T_world_cam_j = T_world_trf_j @ T_trf_cam

    return invert_T(T_world_cam_j) @ T_world_cam_i


def score_pose_vs_icp(
    view_pose_data: Dict[int, List[float]],
    icp_cam_T: Dict[Tuple[int, int], np.ndarray],
    pairs: List[Tuple[int, int]],
    cand: Candidate,
    cam_corr_mat: np.ndarray,
    t_trf_cam_m: np.ndarray,
    w_rot_deg: float,
    w_trans_mm: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Lower is better. Compares predicted relative motion in CAM vs free-ICP relative motion in CAM.
    """
    drots: List[float] = []
    dtrans: List[float] = []

    for (i, j) in pairs:
        if (i not in view_pose_data) or (j not in view_pose_data):
            continue
        if (i, j) not in icp_cam_T:
            continue

        T_icp = icp_cam_T[(i, j)]
        T_pred = predicted_T_camj_from_cami(
            pose6_i=view_pose_data[i],
            pose6_j=view_pose_data[j],
            cand=cand,
            cam_corr_mat=cam_corr_mat,
            t_trf_cam_m=t_trf_cam_m,
        )

        Delta = T_pred @ invert_T(T_icp)
        drot, dtr = T_delta_stats(Delta)
        drots.append(float(drot))
        dtrans.append(float(dtr))

    if not drots:
        return float("inf"), {"pose_icp_pairs": 0.0}

    med_rot = float(np.median(drots))
    med_tr = float(np.median(dtrans))
    score = (w_rot_deg * med_rot) + (w_trans_mm * med_tr)

    return score, {
        "pose_icp_pairs": float(len(drots)),
        "pose_icp_med_rot_deg": med_rot,
        "pose_icp_med_trans_mm": med_tr,
        "score_pose_icp": score,
    }


# ============================================================
# CAM correction candidates
# ============================================================

def build_cam_corr_candidates(R_yaml: np.ndarray, mode: str) -> List[Tuple[str, np.ndarray]]:
    """
    Returns list of (name, R_trf_cam) candidates.

    limited:
      - R_yaml, R_yaml.T, I

    handeye:
      - limited + a small set of common axis permutations used in hand-eye
        including Rz(±90), Rx(±90), Ry(±90) and some diag flips.

    full:
      - generates all signed permutation matrices combined with base (R, RT, I)
    """
    R_yaml = np.asarray(R_yaml, dtype=np.float64).reshape((3, 3))
    bases: List[Tuple[str, np.ndarray]] = [
        ("R", R_yaml),
        ("RT", R_yaml.T),
        ("I", np.eye(3, dtype=np.float64)),
    ]

    if mode == "limited":
        return bases

    if mode == "handeye":
        extra: List[Tuple[str, np.ndarray]] = []
        rots = [
            ("Rz+90", rotz(np.deg2rad(90.0))),
            ("Rz-90", rotz(np.deg2rad(-90.0))),
            ("Rx+90", rotx(np.deg2rad(90.0))),
            ("Rx-90", rotx(np.deg2rad(-90.0))),
            ("Ry+90", roty(np.deg2rad(90.0))),
            ("Ry-90", roty(np.deg2rad(-90.0))),
        ]
        flips = [
            ("F111", np.diag([1.0, 1.0, 1.0])),
            ("F1-11", np.diag([1.0, -1.0, 1.0])),
            ("F11-1", np.diag([1.0, 1.0, -1.0])),
            ("F-111", np.diag([-1.0, 1.0, 1.0])),
        ]
        for bname, B in bases:
            for rname, R in rots:
                for fname, F in flips:
                    M = R @ F
                    e, d = rotation_sanity(M)
                    if e < 1e-8 and abs(abs(d) - 1.0) < 1e-6:
                        extra.append((f"{bname}*{rname}*{fname}", B @ M))

        uniq: Dict[Tuple[float, ...], Tuple[str, np.ndarray]] = {}
        for name, M in (bases + extra):
            key = tuple(np.round(M.reshape(-1), 6).tolist())
            if key not in uniq:
                uniq[key] = (name, M)
        return list(uniq.values())

    if mode == "full":
        mats = []
        axes = [0, 1, 2]
        for perm in itertools.permutations(axes, 3):
            P = np.zeros((3, 3), dtype=np.float64)
            P[0, perm[0]] = 1.0
            P[1, perm[1]] = 1.0
            P[2, perm[2]] = 1.0
            for sx, sy, sz in itertools.product([1.0, -1.0], repeat=3):
                S = np.diag([sx, sy, sz]).astype(np.float64)
                M = S @ P
                e, d = rotation_sanity(M)
                if e < 1e-8 and abs(abs(d) - 1.0) < 1e-6:
                    mats.append(M)

        extra = []
        for base_name, base_R in bases:
            for i, M in enumerate(mats):
                X = base_R @ M
                e, d = rotation_sanity(X)
                if e < 1e-7 and abs(abs(d) - 1.0) < 1e-6:
                    extra.append((f"{base_name}*M{i}", X))

        uniq = {}
        for name, M in (bases + extra):
            key = tuple(np.round(M.reshape(-1), 6).tolist())
            if key not in uniq:
                uniq[key] = (name, M)
        return list(uniq.values())

    raise ValueError("search_cam_axes must be one of: limited, handeye, full")


# ============================================================
# Batch helpers
# ============================================================

def parse_plant_ids(args) -> List[int]:
    if args.plant_ids:
        ids = [int(x) for x in args.plant_ids]
    elif args.plant_range:
        a, b = int(args.plant_range[0]), int(args.plant_range[1])
        if a > b:
            a, b = b, a
        ids = list(range(a, b + 1))
    else:
        raise ValueError("Provide either --plant_ids or --plant_range")

    skip = set(int(x) for x in (args.skip_plant_ids or []))
    ids = [i for i in ids if i not in skip]
    return ids


# ============================================================
# Per-plant runner
# ============================================================

def run_one(
    cfg: Dict,
    config_path: Path,
    out_root: Path,
    plant_id: int,
    instance_id: Optional[int],
    args,
) -> Dict[str, object]:
    robot = cfg.get("robot", {})
    views = robot.get("views", {})
    if not isinstance(views, dict) or not views:
        raise ValueError("robot.views missing in config")

    ds = cfg.get("dataset", {})
    if isinstance(ds, dict) and "view_ids" in ds:
        view_ids = [int(v) for v in ds["view_ids"]]
    else:
        view_ids = sorted([int(k) for k in views.keys()])

    # ------------------------------------------------------------
    # CAM correction in YAML (robust keys)
    # ------------------------------------------------------------
    R_list = (
        robot.get("cam_axes_correction_R_trf_cam_row_major_3x3")
        or robot.get("R_trf_cam_row_major_3x3")
        or robot.get("R_trf_cam")
    )
    if not (isinstance(R_list, list) and len(R_list) == 9):
        raise ValueError(
            "Missing 3x3 R_trf_cam. Expected one of keys:\n"
            "  robot.cam_axes_correction_R_trf_cam_row_major_3x3\n"
            "  robot.R_trf_cam_row_major_3x3\n"
            "  robot.R_trf_cam\n"
        )
    R_yaml = np.array([float(x) for x in R_list], dtype=np.float64).reshape((3, 3))

    # optional cam translation in TRF (meters)
    t_mm = robot.get("camera_in_trf_translation_mm", [0.0, 0.0, 0.0])
    if not (isinstance(t_mm, list) and len(t_mm) == 3):
        raise ValueError("robot.camera_in_trf_translation_mm must be list of 3")
    t_trf_cam_m = (np.array([float(x) for x in t_mm], dtype=np.float64) / 1000.0).reshape((3,))

    err, det = rotation_sanity(R_yaml)
    print(f"[Plant {plant_id:03d}] [Config] R_trf_cam sanity: ortho_err={err:.3e} det={det:.6f} | t_trf_cam_m={t_trf_cam_m.tolist()}")

    # ------------------------------------------------------------
    # Load CAM clouds
    # ------------------------------------------------------------
    clouds_cam = load_view_clouds_cam(out_root, plant_id, view_ids, instance_id=instance_id)
    use_clouds = True
    centroids_cam: Dict[int, np.ndarray] = {}

    if len(clouds_cam) < 2:
        print(f"[Plant {plant_id:03d}] [WARN] Not enough CAM clouds found. Falling back to features.csv centroids only.")
        centroids_cam = load_centroids_from_features(out_root, plant_id, instance_id=int(instance_id or 1))
        if len(centroids_cam) < 2:
            raise ValueError(f"[Plant {plant_id:03d}] Neither CAM clouds nor features.csv centroids found for >=2 views.")
        use_clouds = False

    # ------------------------------------------------------------
    # Prepare view poses
    # ------------------------------------------------------------
    view_pose_data: Dict[int, List[float]] = {}
    for vid in view_ids:
        v = views.get(str(vid), None) or views.get(vid, None)
        if not isinstance(v, dict):
            continue
        pose = v.get("pose_wrf_mm_deg", None)
        if isinstance(pose, list) and len(pose) == 6:
            view_pose_data[vid] = [float(x) for x in pose]

    if len(view_pose_data) < 2:
        raise ValueError(f"[Plant {plant_id:03d}] Not enough robot.view poses found for >=2 views.")

    # ------------------------------------------------------------
    # Optional FREE-ICP check
    # ------------------------------------------------------------
    icp_cam_T: Dict[Tuple[int, int], np.ndarray] = {}
    if args.free_icp and use_clouds:
        icp_cam_T = print_free_icp_report(
            clouds_cam=clouds_cam,
            view_ids=view_ids,
            pair_mode=args.pair_mode,
            voxel=args.voxel,
            sample_n=args.sample_n,
            max_corr=args.icp_max_corr,
            bridge_via=args.bridge_via,
        )

    # For scoring pose-vs-ICP we need chain pairs
    chain_pairs = make_pairs(view_ids, mode="chain")
    if use_clouds and (args.score_mode in ("icp", "hybrid")):
        missing = [p for p in chain_pairs if p not in icp_cam_T]
        if missing:
            icp_cam_T.update(
                free_icp_compute_pairs(
                    clouds_cam=clouds_cam,
                    pairs=missing,
                    voxel=args.voxel,
                    sample_n=args.sample_n,
                    max_corr=args.icp_max_corr,
                )
            )

    # ------------------------------------------------------------
    # cam_corr candidates (optionally frozen)
    # ------------------------------------------------------------
    cam_corr_candidates = build_cam_corr_candidates(R_yaml, mode=args.search_cam_axes)
    if args.freeze_cam:
        cam_corr_candidates = [("YAML_FIXED", R_yaml)]
        print(f"[Plant {plant_id:03d}] [Search] freeze_cam=True => cam_corr candidates forced to 1 (YAML_FIXED)")
    else:
        print(f"[Plant {plant_id:03d}] [Search] cam_corr candidates ({args.search_cam_axes}) = {len(cam_corr_candidates)}")

    # ------------------------------------------------------------
    # Candidate options (pose search)
    # ------------------------------------------------------------
    neg_angle_opts = [(1, 1, 1), (-1, -1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)]
    t_sign_opts = [(1, 1, 1), (-1, -1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)]
    t_scale_opts = [0.001, 1.0]  # mm->m, or already meters
    invert_pose_opts = [False, True]
    use_cam_t_opts = [False, True]
    cam_axis_flip_opts = [(1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, 1, 1), (-1, -1, 1)]

    # Freeze CAM: no extra axis flips and no cam_t search
    if args.freeze_cam:
        use_cam_t_opts = [False]
        cam_axis_flip_opts = [(1, 1, 1)]

    euler_order_opts = list(EULER_ORDERS.keys())
    euler_mode_opts = EULER_MODES
    angle_map_opts = list(ANGLE_MAPS.keys()) if args.allow_angle_maps == "full" else ["xyz"]

    est = (len(euler_order_opts) * len(euler_mode_opts) * len(angle_map_opts) * len(neg_angle_opts) *
           len(invert_pose_opts) * len(t_sign_opts) * len(t_scale_opts) *
           len(cam_corr_candidates) * len(use_cam_t_opts) * len(cam_axis_flip_opts))
    print(f"[Plant {plant_id:03d}] [Search] estimated candidates = {est} | score_mode={args.score_mode} | freeze_cam={args.freeze_cam}")

    # auto reduce if huge
    if est > args.max_candidates:
        print(f"[Plant {plant_id:03d}] [WARN] Candidate space huge. Reducing automatically.")
        t_scale_opts = [0.001]
        use_cam_t_opts = [False]
        if not args.freeze_cam:
            cam_axis_flip_opts = [(1, 1, 1), (1, -1, 1)]
        else:
            cam_axis_flip_opts = [(1, 1, 1)]
        if args.allow_angle_maps == "full":
            angle_map_opts = ["xyz", "zyx"]

    forward_penalty = 5.0
    aabb_penalty = 1.0

    ranked: List[Tuple[float, Candidate, Dict[str, float]]] = []
    best_pack: Optional[Tuple[float, Candidate, Dict[str, float], Dict[int, np.ndarray], Dict[int, np.ndarray]]] = None

    tested = 0
    for euler_order, euler_mode, angle_map, neg_angles, invert_pose, t_sign, t_scale, cam_corr_id, use_cam_t, cam_flip in itertools.product(
        euler_order_opts,
        euler_mode_opts,
        angle_map_opts,
        neg_angle_opts,
        invert_pose_opts,
        t_sign_opts,
        t_scale_opts,
        range(len(cam_corr_candidates)),
        use_cam_t_opts,
        cam_axis_flip_opts,
    ):
        tested += 1
        if tested > args.max_candidates:
            break

        cam_name, cam_mat = cam_corr_candidates[int(cam_corr_id)]

        cand = Candidate(
            euler_order=euler_order,
            euler_mode=euler_mode,
            angle_map=angle_map,
            neg_angles=neg_angles,
            invert_pose=invert_pose,
            t_sign=t_sign,
            t_scale=t_scale,
            cam_corr_id=int(cam_corr_id),
            cam_corr_name=str(cam_name),
            cam_axis_flip=cam_flip,
            use_cam_t=use_cam_t,
        )

        world_clouds: Dict[int, np.ndarray] = {}
        forward_dots: Dict[int, float] = {}
        cam_origins_world: Dict[int, np.ndarray] = {}

        for vid, pose6 in view_pose_data.items():
            Rw, tw = build_pose_Rt(pose6, cand)

            if use_clouds:
                pts_cam = clouds_cam.get(vid, None)
                if pts_cam is None:
                    continue
                pts_cam_ds = voxel_downsample(pts_cam, args.voxel)
                pts_w, f_dot, cam_o_w = apply_chain_points_world(
                    pts_cam_ds, Rw, tw, cam_mat, t_trf_cam_m, cand
                )
                world_clouds[vid] = pts_w
                forward_dots[vid] = f_dot
                cam_origins_world[vid] = cam_o_w
            else:
                if vid not in centroids_cam:
                    continue
                pts_cam = centroids_cam[vid].reshape((1, 3))
                pts_w, f_dot, cam_o_w = apply_chain_points_world(
                    pts_cam, Rw, tw, cam_mat, t_trf_cam_m, cand
                )
                world_clouds[vid] = pts_w
                forward_dots[vid] = f_dot
                cam_origins_world[vid] = cam_o_w

        if len(world_clouds) < 2:
            continue

        # overlap score (optional)
        s_overlap, det_overlap = score_overlap_alignment(
            world_clouds=world_clouds,
            forward_dots=forward_dots,
            view_ids=view_ids,
            pair_mode=args.pair_mode,
            sample_n=args.sample_n,
            forward_penalty=forward_penalty,
            aabb_penalty=aabb_penalty,
        )

        # pose-vs-icp score
        s_icp = float("inf")
        det_icp: Dict[str, float] = {}
        if args.score_mode in ("icp", "hybrid") and use_clouds:
            s_icp, det_icp = score_pose_vs_icp(
                view_pose_data=view_pose_data,
                icp_cam_T=icp_cam_T,
                pairs=chain_pairs,
                cand=cand,
                cam_corr_mat=cam_mat,
                t_trf_cam_m=t_trf_cam_m,
                w_rot_deg=args.w_rot_deg,
                w_trans_mm=args.w_trans_mm,
            )

        if args.score_mode == "overlap":
            score = s_overlap
        elif args.score_mode == "icp":
            score = s_icp
        else:  # hybrid
            score = s_overlap + (args.alpha_icp * s_icp)

        details = dict(det_overlap)
        details.update(det_icp)
        details["score_total"] = float(score)
        details["score_overlap"] = float(s_overlap)
        details["score_pose_icp"] = float(s_icp)

        ranked.append((score, cand, details))

        if best_pack is None or score < best_pack[0]:
            best_pack = (score, cand, details, world_clouds, cam_origins_world)

    ranked.sort(key=lambda x: x[0])

    print("\n" + "=" * 120)
    inst_txt = "inst=ALL" if instance_id is None else f"inst={instance_id}"
    print(f"[Plant {plant_id:03d}] RANKING (top {min(args.topk, len(ranked))}) tested={len(ranked)} cap={args.max_candidates} | {inst_txt} | score_mode={args.score_mode} | freeze_cam={args.freeze_cam}")
    print("=" * 120)
    for i, (score, cand, det) in enumerate(ranked[: args.topk], start=1):
        print(
            f"{i:02d}) score={score:.6f} "
            f"ov={det.get('score_overlap', float('nan')):.4f} icp={det.get('score_pose_icp', float('nan')):.4f} "
            f"spread={det['spread_norm']:.4f} nn_med={det['nn_med']:.4f} nn_p90={det['nn_p90']:.4f} bad_fwd={int(det['bad_forward'])} | "
            f"euler={cand.euler_order}/{cand.euler_mode} map={cand.angle_map} neg={cand.neg_angles} inv={cand.invert_pose} "
            f"t_sign={cand.t_sign} t_scale={cand.t_scale} cam_corr={cand.cam_corr_name} cam_t={cand.use_cam_t} flip={cand.cam_axis_flip} "
            f"| icp_med_rot={det.get('pose_icp_med_rot_deg', float('nan')):.2f}deg icp_med_tr={det.get('pose_icp_med_trans_mm', float('nan')):.1f}mm"
        )

    if best_pack is None:
        print(f"[Plant {plant_id:03d}] [ERROR] No valid candidates found.")
        return {"plant_id": plant_id, "instance": instance_id, "ok": False}

    best_score, best_cand, best_det, best_world, best_cam_origins = best_pack
    print("\n" + "-" * 120)
    print(f"[Plant {plant_id:03d}] BEST CANDIDATE ({inst_txt})")
    print("-" * 120)
    print(f"score_total={best_score:.6f} | overlap={best_det.get('score_overlap', float('nan')):.6f} | icp={best_det.get('score_pose_icp', float('nan')):.6f}")
    print(f"pose-vs-icp median: rot={best_det.get('pose_icp_med_rot_deg', float('nan')):.2f}deg trans={best_det.get('pose_icp_med_trans_mm', float('nan')):.1f}mm")
    print(f"cand={best_cand}")

    # Extra: top-K pose-vs-ICP report for chain pairs
    if use_clouds and args.icp_topk > 0:
        K = min(args.icp_topk, len(ranked))
        pairs_for_diag = chain_pairs

        # ensure ICP
        missing = [p for p in pairs_for_diag if p not in icp_cam_T]
        if missing:
            icp_cam_T.update(
                free_icp_compute_pairs(
                    clouds_cam=clouds_cam,
                    pairs=missing,
                    voxel=args.voxel,
                    sample_n=args.sample_n,
                    max_corr=args.icp_max_corr,
                )
            )

        print("\n" + "=" * 120)
        print(f"[Plant {plant_id:03d}] TOP-{K} CANDIDATE DIAGNOSTICS: POSE vs FREE-ICP (chain pairs)")
        print("=" * 120)

        for rank_i in range(K):
            score, cand, _det = ranked[rank_i]
            cam_name, cam_mat = cam_corr_candidates[cand.cam_corr_id]

            print(
                f"\n[Cand rank {rank_i+1}] score={score:.6f} | "
                f"euler={cand.euler_order}/{cand.euler_mode} map={cand.angle_map} neg={cand.neg_angles} "
                f"inv={cand.invert_pose} t_sign={cand.t_sign} t_scale={cand.t_scale} "
                f"cam_corr={cam_name} cam_t={cand.use_cam_t} flip={cand.cam_axis_flip}"
            )

            for (i, j) in pairs_for_diag:
                if (i not in view_pose_data) or (j not in view_pose_data):
                    continue
                if (i, j) not in icp_cam_T:
                    print(f"  [pair {i}->{j}] missing ICP transform")
                    continue

                T_icp = icp_cam_T[(i, j)]
                T_pred = predicted_T_camj_from_cami(
                    pose6_i=view_pose_data[i],
                    pose6_j=view_pose_data[j],
                    cand=cand,
                    cam_corr_mat=cam_mat,
                    t_trf_cam_m=t_trf_cam_m,
                )

                Delta = T_pred @ invert_T(T_icp)
                drot, dtrans = T_delta_stats(Delta)

                rot_pred, trans_pred = T_delta_stats(T_pred)
                rot_icp, trans_icp = T_delta_stats(T_icp)

                print(
                    f"  [pair {i}->{j}] "
                    f"delta(rot={drot:.2f}deg, trans={dtrans:.1f}mm) | "
                    f"pred(rot={rot_pred:.2f}deg, trans={trans_pred:.1f}mm) "
                    f"icp(rot={rot_icp:.2f}deg, trans={trans_icp:.1f}mm)"
                )

    # Export best
    if args.export_best or args.export_combined:
        dbg_dir = out_root / "diagnostics" / f"plant_{plant_id:03d}"
        if instance_id is not None:
            dbg_dir = dbg_dir / f"inst_{instance_id:02d}"
        dbg_dir.mkdir(parents=True, exist_ok=True)

        palette = [
            (255, 50, 50),
            (50, 255, 50),
            (50, 50, 255),
            (255, 255, 50),
            (255, 50, 255),
            (50, 255, 255),
            (200, 200, 200),
        ]

        all_pts = []
        all_rgb = []

        for k, vid in enumerate(sorted(best_world.keys())):
            pts = best_world[vid]
            col = np.array(palette[k % len(palette)], dtype=np.uint8)
            rgb = np.repeat(col.reshape(1, 3), pts.shape[0], axis=0)
            out_p = dbg_dir / f"best_view_{vid}_world_xyzrgb.ply"
            write_ply_xyzrgb_ascii(out_p, pts, rgb)
            all_pts.append(pts)
            all_rgb.append(rgb)

        cam_pts = []
        cam_rgb = []
        for k, vid in enumerate(sorted(best_cam_origins.keys())):
            cam_o = best_cam_origins[vid].reshape(1, 3).astype(np.float32)
            col = np.array(palette[k % len(palette)], dtype=np.uint8).reshape(1, 3)
            cam_pts.append(cam_o)
            cam_rgb.append(col)
        if cam_pts:
            cam_pts = np.vstack(cam_pts)
            cam_rgb = np.vstack(cam_rgb)
            write_ply_xyzrgb_ascii(dbg_dir / "camera_origins_world_xyzrgb.ply", cam_pts, cam_rgb)

        if args.export_combined and all_pts:
            P = np.vstack(all_pts).astype(np.float32)
            C = np.vstack(all_rgb).astype(np.uint8)
            write_ply_xyzrgb_ascii(dbg_dir / "best_combined_world_xyzrgb.ply", P, C)

        (dbg_dir / "best_candidate.txt").write_text(
            f"CONFIG={config_path}\nPLANT={plant_id}\nINSTANCE={instance_id}\n"
            f"SCORE_TOTAL={best_score}\n{best_det}\n{best_cand}\n",
            encoding="utf-8",
        )
        print(f"[Plant {plant_id:03d}] [Export] Wrote diagnostics to: {dbg_dir}")

    return {
        "plant_id": plant_id,
        "instance": instance_id,
        "ok": True,
        "score_total": float(best_score),
        "score_overlap": float(best_det.get("score_overlap", float("nan"))),
        "score_pose_icp": float(best_det.get("score_pose_icp", float("nan"))),
        "pose_icp_med_rot_deg": float(best_det.get("pose_icp_med_rot_deg", float("nan"))),
        "pose_icp_med_trans_mm": float(best_det.get("pose_icp_med_trans_mm", float("nan"))),
        "cand": str(best_cand),
    }


# ============================================================
# Main (batch)
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out_root", default=Path("outputs"), type=Path)

    # batch selection
    ap.add_argument("--plant_ids", nargs="*", type=int, default=None, help="Explicit list, e.g. --plant_ids 6 7 9")
    ap.add_argument("--plant_range", nargs=2, type=int, default=None, help="Range inclusive, e.g. --plant_range 6 9")
    ap.add_argument("--skip_plant_ids", nargs="*", type=int, default=None, help="Plants to skip, e.g. --skip_plant_ids 8")

    # instances (optional)
    ap.add_argument(
        "--instances", nargs="*", type=int, default=None,
        help="If provided: runs per instance using cloud_{vid}_inst{n}_cam.ply (or fallback). Example: --instances 1 2"
    )

    # core params
    ap.add_argument("--sample_n", default=4000, type=int)
    ap.add_argument("--topk", default=20, type=int)
    ap.add_argument("--max_candidates", default=20000, type=int)

    ap.add_argument("--pair_mode", choices=["chain", "all"], default="chain")
    ap.add_argument("--bridge_via", type=int, default=1)

    ap.add_argument("--export_best", action="store_true")
    ap.add_argument("--export_combined", action="store_true")

    ap.add_argument("--free_icp", action="store_true")
    ap.add_argument("--voxel", default=0.0025, type=float)
    ap.add_argument("--icp_max_corr", default=0.03, type=float)

    # CAM focus
    ap.add_argument(
        "--freeze_cam", action="store_true",
        help="Fix CAM->TRF exactly to YAML: no extra cam candidates, no cam flip, no cam_t search."
    )

    # search & scoring
    ap.add_argument(
        "--search_cam_axes", choices=["limited", "handeye", "full"], default="handeye",
        help="handeye includes Rz(±90) etc; full tries all signed permutations"
    )
    ap.add_argument("--icp_topk", default=8, type=int)
    ap.add_argument("--allow_angle_maps", choices=["limited", "full"], default="limited")

    ap.add_argument(
        "--score_mode", choices=["overlap", "icp", "hybrid"], default="hybrid",
        help="hybrid recommended: overlap + alpha*pose-vs-icp"
    )
    ap.add_argument("--alpha_icp", default=0.25, type=float,
                    help="Only used in hybrid: total = overlap + alpha*icp_score")

    ap.add_argument("--w_rot_deg", default=0.10, type=float,
                    help="ICP score weight for rotation degrees")
    ap.add_argument("--w_trans_mm", default=0.02, type=float,
                    help="ICP score weight for translation millimeters")

    args = ap.parse_args()
    np.random.seed(0)

    cfg = load_yaml(args.config)
    plant_ids = parse_plant_ids(args)

    print("\n" + "=" * 120)
    print(f"BATCH RUN: plants={plant_ids} | skip={args.skip_plant_ids or []} | out_root={args.out_root} | config={args.config}")
    print(f"score_mode={args.score_mode} search_cam_axes={args.search_cam_axes} allow_angle_maps={args.allow_angle_maps} pair_mode={args.pair_mode} freeze_cam={args.freeze_cam}")
    print("=" * 120)

    instances: List[Optional[int]]
    if args.instances:
        instances = [int(i) for i in args.instances]
    else:
        instances = [None]  # single run (no inst suffix)

    summaries: List[Dict[str, object]] = []
    for pid in plant_ids:
        for inst in instances:
            print("\n" + "#" * 120)
            inst_txt = "ALL" if inst is None else f"{inst}"
            print(f"RUN plant_{pid:03d} | instance={inst_txt}")
            print("#" * 120)
            try:
                s = run_one(
                    cfg=cfg,
                    config_path=args.config,
                    out_root=args.out_root,
                    plant_id=pid,
                    instance_id=inst,
                    args=args,
                )
            except Exception as e:
                print(f"[Plant {pid:03d}] [ERROR] {e}")
                s = {"plant_id": pid, "instance": inst, "ok": False, "error": str(e)}
            summaries.append(s)

    print("\n" + "=" * 120)
    print("BATCH SUMMARY (best per run)")
    print("=" * 120)
    for s in summaries:
        pid = int(s.get("plant_id", -1))
        inst = s.get("instance", None)
        ok = bool(s.get("ok", False))
        inst_txt = "ALL" if inst is None else f"{int(inst)}"
        if not ok:
            print(f"plant_{pid:03d} inst={inst_txt}: FAIL | {s.get('error', '')}")
            continue
        print(
            f"plant_{pid:03d} inst={inst_txt}: "
            f"score_total={s.get('score_total'):.4f} "
            f"overlap={s.get('score_overlap'):.4f} icp={s.get('score_pose_icp'):.4f} "
            f"icp_med_rot={s.get('pose_icp_med_rot_deg'):.2f}deg icp_med_tr={s.get('pose_icp_med_trans_mm'):.1f}mm"
        )


if __name__ == "__main__":
    main()
