from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from strawberry_py.config import DepthCfg
from strawberry_py.st_types import DepthU16, DepthF32, Mask, PointCloud, Pose


# ============================================================
# Rotation / Pose helpers
# ============================================================

def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x,y,z,w) -> Rotation matrix (3x3)."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n <= 0.0:
        return np.eye(3, dtype=np.float32)
    q /= n
    x, y, z, w = q.tolist()

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return R


def apply_pose(points_src: PointCloud, pose_dst_src: Pose) -> PointCloud:
    """
    Apply pose: p_dst = R_dst_src * p_src + t_dst_src
    pose_dst_src is Pose(t_xyz in meters, q_xyzw).
    """
    pts = np.asarray(points_src, dtype=np.float32).reshape((-1, 3))
    t = np.asarray(pose_dst_src.t_xyz, dtype=np.float32).reshape((3,))
    qx, qy, qz, qw = pose_dst_src.q_xyzw
    R = quaternion_to_rotation_matrix(float(qx), float(qy), float(qz), float(qw))
    return (R @ pts.T).T + t


def invert_pose(pose_dst_src: Pose) -> Pose:
    """Return pose_src_dst (inverse of pose_dst_src)."""
    t = np.asarray(pose_dst_src.t_xyz, dtype=np.float32).reshape((3,))
    qx, qy, qz, qw = pose_dst_src.q_xyzw
    R = quaternion_to_rotation_matrix(float(qx), float(qy), float(qz), float(qw))
    R_inv = R.T
    t_inv = -(R_inv @ t)

    # invert quaternion: (x,y,z,w) -> (-x,-y,-z,w) for unit quaternion
    return Pose(t_xyz=(float(t_inv[0]), float(t_inv[1]), float(t_inv[2])),
                q_xyzw=(-float(qx), -float(qy), -float(qz), float(qw)))


# ============================================================
# Mecademic Euler convention (XY'Z'')
# ============================================================

def _rotx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float32)


def _roty(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float32)


def _rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float32)


def meca_euler_xyzprime_zdouble_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """
    Mecademic manual: Euler angles in XY'Z'' convention:
      rotate about x by rx, then about y' by ry, then about z'' by rz.
    For the body->world rotation matrix this corresponds to:
      R = Rz(rz) @ Ry(ry) @ Rx(rx)
    """
    rx = math.radians(float(rx_deg))
    ry = math.radians(float(ry_deg))
    rz = math.radians(float(rz_deg))
    return (_rotz(rz) @ _roty(ry) @ _rotx(rx)).astype(np.float32)


def rotmat_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Rotation matrix -> quaternion (x,y,z,w), normalized."""
    R = np.asarray(R, dtype=np.float64).reshape((3, 3))
    tr = float(np.trace(R))

    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n > 0:
        q /= n
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


# ============================================================
# Camera frame conversion (Optical -> TRF)
# ============================================================

def optical_to_trf(points_optical: PointCloud, flip_y: bool = False) -> PointCloud:
    """
    Convert camera OPTICAL frame (x right, y down, z forward)
    to a more "robot tool / link-like" frame (x forward, y left, z up).

    ROS REP-103 common mapping:
      x_trf =  z_opt
      y_trf = -x_opt
      z_trf = -y_opt

    This is a pure rotation (no translation). If your camera is mounted differently,
    you can extend this with an extra constant R.

    flip_y: quick hack to test handedness (rarely needed if mapping is correct).
    """
    pts = np.asarray(points_optical, dtype=np.float32).reshape((-1, 3))
    if pts.size == 0:
        return pts

    x = pts[:, 0].copy()
    y = pts[:, 1].copy()
    z = pts[:, 2].copy()

    if flip_y:
        y = -y

    # mapping:
    x_trf = z
    y_trf = -x
    z_trf = -y
    out = np.stack([x_trf, y_trf, z_trf], axis=1).astype(np.float32, copy=False)
    return out


# ============================================================
# Depth conversions / valid mask
# ============================================================

def depth_u16_to_meters(depth_u16: DepthU16, depth_cfg: DepthCfg) -> DepthF32:
    d = np.asarray(depth_u16)
    if depth_cfg.unit.value == "realsense_units":
        return d.astype(np.float32) * float(depth_cfg.scale_m_per_unit)
    return d.astype(np.float32) / 1000.0


def compute_valid_range_mask(depth_u16: DepthU16, depth_cfg: DepthCfg) -> Mask:
    d = np.asarray(depth_u16)
    valid = d != np.uint16(0)
    if depth_cfg.treat_65535_as_invalid:
        valid &= d != np.uint16(65535)

    if not depth_cfg.range_filter.enabled:
        return valid.astype(np.bool_, copy=False)

    min_m = float(depth_cfg.range_filter.min_m)
    max_m = float(depth_cfg.range_filter.max_m)
    if min_m > max_m:
        min_m, max_m = max_m, min_m

    if depth_cfg.unit.value == "realsense_units":
        lo = int(round(min_m / float(depth_cfg.scale_m_per_unit)))
        hi = int(round(max_m / float(depth_cfg.scale_m_per_unit)))
    else:
        lo = int(round(min_m * 1000.0))
        hi = int(round(max_m * 1000.0))

    lo = max(0, min(lo, 65535))
    hi = max(0, min(hi, 65535))
    if lo > hi:
        lo, hi = hi, lo

    return (valid & (d >= np.uint16(lo)) & (d <= np.uint16(hi))).astype(np.bool_, copy=False)
