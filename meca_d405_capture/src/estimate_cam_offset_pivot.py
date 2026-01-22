#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
estimate_cam_offset_pivot.py

Schätzt Kamera-Translation t_trf_cam (in TRF, Meter) mit Pivot-Methode:
- Fester Referenzpunkt in Welt (Ellipse-Marker)
- Viele Robot-Posen (sequence)
- Pro Pose: Ellipse-fit -> center (u,v), depth median -> 3D Punkt p_cam
- Bekannte Rotation R_trf_cam (CAM optical -> TRF)
- Unbekannt: t_trf_cam und P_world (fester Weltpunkt)

Messmodell:
  P_world = Rw_i @ (R_trf_cam @ p_cam_i + t_trf_cam) + tw_i

Linear in [P_world, t_trf_cam]:
  [ I  -Rw_i ] [P_world] = Rw_i @ (R_trf_cam @ p_cam_i) + tw_i
              [t_trf_cam]

CLI:
  python src/estimate_cam_offset_pivot.py --config config.yaml capture --run_dir runs/pivot_001 --debug
  python src/estimate_cam_offset_pivot.py --config config.yaml solve --run_dir runs/pivot_001
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
import cv2

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from camera_d405 import RealsenseD405  # noqa: E402
from robot_meca import Meca500Controller  # noqa: E402
from transforms import (  # noqa: E402
    PinholeIntrinsics,
    uv_depth_to_xyz_cam,
    pose_mm_deg_to_T,
    R_from_row_major_3x3,
    rotation_sanity,
)


@dataclass(frozen=True)
class PivotCfg:
    euler_convention: str
    R_trf_cam: np.ndarray
    depth_scale_m_per_unit: float
    positions: Dict[str, Tuple[float, float, float, float, float, float]]
    sequence: List[str]
    settle_s: float
    min_contour_area: float
    canny1: int
    canny2: int


def load_cfg(path: Path) -> Tuple[PivotCfg, dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    rob = raw.get("robot", {})
    euler = str(rob.get("euler_convention", "RzRyRx_deg")).strip()

    R_list = rob.get("cam_axes_correction_R_trf_cam_row_major_3x3", None)
    if R_list is None:
        raise ValueError("robot.cam_axes_correction_R_trf_cam_row_major_3x3 missing")
    R_trf_cam = R_from_row_major_3x3(R_list)

    ortho_err, det = rotation_sanity(R_trf_cam)
    if ortho_err > 1e-6 or abs(det - 1.0) > 1e-6:
        print(f"[WARN] R_trf_cam sanity: ortho_err={ortho_err:.3e} det={det:.6f}")

    depth = raw.get("depth", {})
    depth_scale = float(depth.get("scale_m_per_unit", 1e-4))

    positions_raw = rob.get("positions", {})
    if not isinstance(positions_raw, dict) or not positions_raw:
        raise ValueError("robot.positions missing")
    positions: Dict[str, Tuple[float, float, float, float, float, float]] = {}
    for k, v in positions_raw.items():
        if not (isinstance(v, list) and len(v) == 6):
            raise ValueError(f"robot.positions['{k}'] must be list of 6")
        positions[str(k)] = tuple(float(x) for x in v)  # type: ignore

    seq = rob.get("sequence", ["l", "m", "r"])
    seq = [str(s) for s in seq]
    settle_s = float(rob.get("settle_s", 0.5))

    ell = raw.get("ellipse", {})
    return (
        PivotCfg(
            euler_convention=euler,
            R_trf_cam=R_trf_cam,
            depth_scale_m_per_unit=depth_scale,
            positions=positions,
            sequence=seq,
            settle_s=settle_s,
            min_contour_area=float(ell.get("min_contour_area", 500.0)),
            canny1=int(ell.get("canny1", 60)),
            canny2=int(ell.get("canny2", 160)),
        ),
        raw,
    )


@dataclass(frozen=True)
class EllipseObs:
    u: float
    v: float
    axis_a: float
    axis_b: float
    angle_deg: float
    depth_m: float
    n_depth_px: int


def _ellipse_mask(shape_hw: Tuple[int, int], ellipse) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, ellipse, color=255, thickness=-1)
    return mask


def detect_ellipse_and_depth(
    rgb_bgr: np.ndarray,
    depth_u16: np.ndarray,
    cfg: PivotCfg,
    treat_65535_as_invalid: bool = True,
    debug_dir: Optional[Path] = None,
    idx: int = 0,
) -> Optional[EllipseObs]:
    h, w = rgb_bgr.shape[:2]
    if depth_u16.shape[:2] != (h, w):
        raise ValueError("Depth must be aligned to color (same resolution).")

    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, cfg.canny1, cfg.canny2)
    edges = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    best, best_area = None, 0.0
    for c in contours:
        a = float(cv2.contourArea(c))
        if a > best_area and a >= cfg.min_contour_area and len(c) >= 5:
            best, best_area = c, a

    if best is None:
        return None

    ellipse = cv2.fitEllipse(best)  # ((cx,cy),(MA,ma),angle)
    (cx, cy), (MA, ma), ang = ellipse
    axis_a = float(max(MA, ma)) * 0.5
    axis_b = float(min(MA, ma)) * 0.5

    mask = _ellipse_mask((h, w), ellipse).astype(bool)
    d = depth_u16
    valid = (d > 0) & mask
    if treat_65535_as_invalid:
        valid = valid & (d < 65535)

    vals = d[valid]
    if vals.size < 30:
        return None

    depth_units_med = float(np.median(vals))
    depth_m = depth_units_med * float(cfg.depth_scale_m_per_unit)

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        dbg = rgb_bgr.copy()
        cv2.ellipse(dbg, ellipse, (0, 255, 255), 2)
        cv2.circle(dbg, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        cv2.putText(
            dbg,
            f"u={cx:.1f} v={cy:.1f} z={depth_m:.3f}m n={int(vals.size)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(debug_dir / f"dbg_{idx:03d}.png"), dbg)

    return EllipseObs(
        u=float(cx),
        v=float(cy),
        axis_a=float(axis_a),
        axis_b=float(axis_b),
        angle_deg=float(ang),
        depth_m=float(depth_m),
        n_depth_px=int(vals.size),
    )


CSV_HEADER = [
    "idx",
    "x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg",
    "u_px", "v_px", "depth_m",
    "axis_a_px", "axis_b_px", "angle_deg",
    "n_depth_px",
    "timestamp",
    "fx", "fy", "cx", "cy", "width", "height",
    "depth_scale_m_per_unit",
]


def write_measurements_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_measurements_csv(path: Path) -> Tuple[List[Tuple[float, float, float, float, float, float]], List[EllipseObs], PinholeIntrinsics, float]:
    poses: List[Tuple[float, float, float, float, float, float]] = []
    obs: List[EllipseObs] = []
    K: Optional[PinholeIntrinsics] = None
    depth_scale: Optional[float] = None

    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            poses.append((
                float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"]),
                float(row["rx_deg"]), float(row["ry_deg"]), float(row["rz_deg"]),
            ))
            obs.append(EllipseObs(
                u=float(row["u_px"]),
                v=float(row["v_px"]),
                depth_m=float(row["depth_m"]),
                axis_a=float(row["axis_a_px"]),
                axis_b=float(row["axis_b_px"]),
                angle_deg=float(row["angle_deg"]),
                n_depth_px=int(float(row["n_depth_px"])),
            ))
            if K is None:
                K = PinholeIntrinsics(
                    width=int(float(row["width"])),
                    height=int(float(row["height"])),
                    fx=float(row["fx"]),
                    fy=float(row["fy"]),
                    cx=float(row["cx"]),
                    cy=float(row["cy"]),
                )
            if depth_scale is None:
                depth_scale = float(row["depth_scale_m_per_unit"])

    if K is None or depth_scale is None:
        raise ValueError("No rows in measurements.csv")
    return poses, obs, K, depth_scale


@dataclass(frozen=True)
class SolveResult:
    t_trf_cam_m: np.ndarray
    P_world_m: np.ndarray
    rms_mm: float
    max_mm: float
    n: int


def solve_translation_only(
    euler_convention: str,
    R_trf_cam: np.ndarray,
    K: PinholeIntrinsics,
    poses_mm_deg: List[Tuple[float, float, float, float, float, float]],
    obs: List[EllipseObs],
) -> SolveResult:
    if len(poses_mm_deg) != len(obs):
        raise ValueError("poses and obs length mismatch")
    n = len(obs)
    if n < 6:
        raise ValueError("Need >= 6 observations (better 15-30).")

    A = np.zeros((3 * n, 6), dtype=np.float64)
    b = np.zeros((3 * n,), dtype=np.float64)

    R_trf_cam = np.asarray(R_trf_cam, dtype=np.float64).reshape(3, 3)

    for i, (pose, ob) in enumerate(zip(poses_mm_deg, obs)):
        x_mm, y_mm, z_mm, rx, ry, rz = pose
        T = pose_mm_deg_to_T(x_mm, y_mm, z_mm, rx, ry, rz, euler_convention=euler_convention, t_in_m=True)
        Rw = T[:3, :3]
        tw = T[:3, 3]

        p_cam = uv_depth_to_xyz_cam(ob.u, ob.v, ob.depth_m, K)  # CAM optical
        p_trf0 = (R_trf_cam @ p_cam.reshape(3, 1)).reshape(3,)

        rhs = (Rw @ p_trf0) + tw

        A[3*i:3*i+3, 0:3] = np.eye(3)
        A[3*i:3*i+3, 3:6] = -Rw
        b[3*i:3*i+3] = rhs

    x_hat, *_ = np.linalg.lstsq(A, b, rcond=None)
    P = x_hat[0:3]
    t = x_hat[3:6]

    errs = []
    for pose, ob in zip(poses_mm_deg, obs):
        x_mm, y_mm, z_mm, rx, ry, rz = pose
        T = pose_mm_deg_to_T(x_mm, y_mm, z_mm, rx, ry, rz, euler_convention=euler_convention, t_in_m=True)
        Rw = T[:3, :3]
        tw = T[:3, 3]
        p_cam = uv_depth_to_xyz_cam(ob.u, ob.v, ob.depth_m, K)
        p_trf0 = (R_trf_cam @ p_cam.reshape(3, 1)).reshape(3,)
        P_i = (Rw @ (p_trf0 + t)) + tw
        errs.append(float(np.linalg.norm(P_i - P)))

    errs = np.asarray(errs, dtype=np.float64)
    rms_mm = float(np.sqrt(np.mean(errs**2)) * 1000.0)
    max_mm = float(np.max(errs) * 1000.0)

    return SolveResult(
        t_trf_cam_m=t.astype(np.float64),
        P_world_m=P.astype(np.float64),
        rms_mm=rms_mm,
        max_mm=max_mm,
        n=n,
    )


def cmd_capture(cfg_path: Path, run_dir: Path, debug: bool) -> None:
    cfg, raw = load_cfg(cfg_path)

    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir = run_dir / "debug" if debug else None

    robot = Meca500Controller.from_config_yaml(cfg_path, verbose=True)
    cam_cfg = raw.get("camera", {})
    cam = RealsenseD405(int(cam_cfg.get("width", 640)), int(cam_cfg.get("height", 480)), int(cam_cfg.get("fps", 30)))

    rows: List[dict] = []
    csv_path = run_dir / "measurements.csv"

    try:
        robot.connect()
        robot.activate_and_home()
        robot.set_wrf_trf_from_config(cfg_path)

        cam.start()
        cam.save_meta_yaml(run_dir / "camera_meta.yaml")

        meta = cam.get_meta()
        K = PinholeIntrinsics(
            width=meta.intrinsics_color.width,
            height=meta.intrinsics_color.height,
            fx=meta.intrinsics_color.fx,
            fy=meta.intrinsics_color.fy,
            cx=meta.intrinsics_color.cx,
            cy=meta.intrinsics_color.cy,
        )
        depth_scale = float(meta.depth_scale_m_per_unit or cfg.depth_scale_m_per_unit)

        print(f"[CAPTURE] run_dir={run_dir}")
        print(f"[CAPTURE] sequence={cfg.sequence}")
        print(f"[CAPTURE] writing {csv_path}")
        print(f"[CAPTURE] intrinsics: fx={K.fx:.3f} fy={K.fy:.3f} cx={K.cx:.3f} cy={K.cy:.3f}")
        print(f"[CAPTURE] depth_scale_m_per_unit: {depth_scale:.9e}")

        for i, key in enumerate(cfg.sequence):
            print(f"\n[CAPTURE] Move to '{key}' ({i+1}/{len(cfg.sequence)})")
            robot.move_to(key)
            time.sleep(cfg.settle_s)

            pose = robot.get_pose_mm_deg()

            fr = cam.get_frames()
            rgb = fr["color_bgr"]
            depth_al = fr["depth_aligned_u16"]

            ob = detect_ellipse_and_depth(
                rgb, depth_al, cfg,
                treat_65535_as_invalid=True,
                debug_dir=dbg_dir,
                idx=i,
            )
            if ob is None:
                print("[WARN] Ellipse/Depth not found, skipping")
                continue

            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            row = dict(
                idx=int(i),
                x_mm=float(pose.x_mm), y_mm=float(pose.y_mm), z_mm=float(pose.z_mm),
                rx_deg=float(pose.rx_deg), ry_deg=float(pose.ry_deg), rz_deg=float(pose.rz_deg),
                u_px=float(ob.u), v_px=float(ob.v), depth_m=float(ob.depth_m),
                axis_a_px=float(ob.axis_a), axis_b_px=float(ob.axis_b), angle_deg=float(ob.angle_deg),
                n_depth_px=int(ob.n_depth_px),
                timestamp=str(ts),
                fx=float(K.fx), fy=float(K.fy), cx=float(K.cx), cy=float(K.cy),
                width=int(K.width), height=int(K.height),
                depth_scale_m_per_unit=float(depth_scale),
            )
            rows.append(row)
            write_measurements_csv(csv_path, rows)

            print(f"[OK] u={ob.u:.1f} v={ob.v:.1f} depth={ob.depth_m:.4f}m n={ob.n_depth_px}")

        print(f"\n[CAPTURE] done. saved {len(rows)} measurements to {csv_path}")

    finally:
        try:
            cam.stop()
        except Exception:
            pass
        robot.disconnect()


def cmd_solve(cfg_path: Path, run_dir: Path) -> None:
    cfg, _ = load_cfg(cfg_path)
    run_dir = run_dir.resolve()
    csv_path = run_dir / "measurements.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run capture first.")

    poses, obs, K, depth_scale = read_measurements_csv(csv_path)
    print(f"[SOLVE] loaded n={len(obs)} from {csv_path}")
    print(f"[SOLVE] intrinsics: fx={K.fx:.3f} fy={K.fy:.3f} cx={K.cx:.3f} cy={K.cy:.3f}")
    print(f"[SOLVE] depth_scale_m_per_unit: {depth_scale:.9e}")

    res = solve_translation_only(cfg.euler_convention, cfg.R_trf_cam, K, poses, obs)
    t_mm = res.t_trf_cam_m * 1000.0
    P_mm = res.P_world_m * 1000.0

    print("\n==============================")
    print("   RESULT (translation-only)")
    print("==============================")
    print(f"n              = {res.n}")
    print(f"RMS error      = {res.rms_mm:.2f} mm")
    print(f"MAX error      = {res.max_mm:.2f} mm")
    print("")
    print("t_trf_cam (mm) =", [float(x) for x in t_mm.tolist()])
    print("P_world (mm)   =", [float(x) for x in P_mm.tolist()])
    print("==============================\n")

    snippet_path = run_dir / "estimated_cam_offset.yaml"
    snippet = {"robot": {"camera_in_trf_translation_mm": [float(x) for x in t_mm.tolist()]}}
    snippet_path.write_text(yaml.safe_dump(snippet, sort_keys=False), encoding="utf-8")
    print(f"[SOLVE] wrote {snippet_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture")
    cap.add_argument("--run_dir", type=str, required=True)
    cap.add_argument("--debug", action="store_true")

    sol = sub.add_parser("solve")
    sol.add_argument("--run_dir", type=str, required=True)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    if args.cmd == "capture":
        cmd_capture(cfg_path, Path(args.run_dir), bool(args.debug))
    elif args.cmd == "solve":
        cmd_solve(cfg_path, Path(args.run_dir))
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
