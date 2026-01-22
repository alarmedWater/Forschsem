#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
transforms.py

Mathe-Helfer:
- Rx/Ry/Rz
- Euler-Konventionen -> R
- 4x4 Transformationsmatrizen
- Backprojection (u,v,depth -> xyz_cam)
- Rotation sanity checks
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

EulerConvention = Literal[
    "RxRyRz_deg",
    "RzRyRx_deg",
    "RxRyRz",
    "RzRyRx",
]


# -----------------------------
# Rotations
# -----------------------------

def rotx(rad: float) -> np.ndarray:
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c,  -s],
                     [0.0, s,   c]], dtype=np.float64)


def roty(rad: float) -> np.ndarray:
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return np.array([[c,  0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=np.float64)


def rotz(rad: float) -> np.ndarray:
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def euler_xyz_to_R(rx: float, ry: float, rz: float, convention: EulerConvention) -> np.ndarray:
    """
    Build R from Euler angles.

    convention:
      - RzRyRx_deg: inputs degrees, R = Rz(rz) @ Ry(ry) @ Rx(rx)
      - RxRyRz_deg: inputs degrees, R = Rx(rx) @ Ry(ry) @ Rz(rz)
      - Without _deg: inputs radians.
    """
    is_deg = convention.endswith("_deg")
    base = convention.replace("_deg", "")

    if is_deg:
        rx, ry, rz = np.deg2rad([rx, ry, rz]).tolist()

    Rx = rotx(float(rx))
    Ry = roty(float(ry))
    Rz = rotz(float(rz))

    if base == "RzRyRx":
        return (Rz @ Ry @ Rx).astype(np.float64)
    if base == "RxRyRz":
        return (Rx @ Ry @ Rz).astype(np.float64)

    raise ValueError(f"Unknown Euler convention: {convention}")


# -----------------------------
# Homogeneous transforms
# -----------------------------

def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3,)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    Rt = R.T
    tt = -(Rt @ t)
    return make_T(Rt, tt)


def compose_T(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (np.asarray(A, dtype=np.float64) @ np.asarray(B, dtype=np.float64)).astype(np.float64)


def pose_mm_deg_to_T(
    x_mm: float, y_mm: float, z_mm: float,
    rx_deg: float, ry_deg: float, rz_deg: float,
    euler_convention: EulerConvention = "RzRyRx_deg",
    t_in_m: bool = True,
) -> np.ndarray:
    R = euler_xyz_to_R(rx_deg, ry_deg, rz_deg, euler_convention)
    t = np.array([x_mm, y_mm, z_mm], dtype=np.float64)
    if t_in_m:
        t = t / 1000.0
    return make_T(R, t)


# -----------------------------
# Intrinsics + Backprojection
# -----------------------------

@dataclass(frozen=True)
class PinholeIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def uv_depth_to_xyz_cam(u_px: float, v_px: float, depth_m: float, K: PinholeIntrinsics) -> np.ndarray:
    """
    Camera optical frame convention:
      x right, y down, z forward
    """
    z = float(depth_m)
    x = (float(u_px) - float(K.cx)) * z / float(K.fx)
    y = (float(v_px) - float(K.cy)) * z / float(K.fy)
    return np.array([x, y, z], dtype=np.float64)


# -----------------------------
# Rotation sanity / helpers
# -----------------------------

def R_from_row_major_3x3(vals9) -> np.ndarray:
    return np.asarray(vals9, dtype=np.float64).reshape(3, 3)


def rotation_sanity(R: np.ndarray) -> Tuple[float, float]:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
    det = float(np.linalg.det(R))
    return ortho_err, det


def orthonormalize_R(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("pts must be Nx3")
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    ph = np.hstack([pts, ones])
    out = (ph @ T.T)[:, :3]
    return out


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return transform_points(T, np.asarray(p, dtype=np.float64).reshape(1, 3)).reshape(3,)
