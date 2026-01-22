#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
camera_d405.py

RealSense D405/D4xx Kamera-Wrapper:
- RGB (BGR) + Depth RAW + Depth aligned-to-color
- Intrinsics/Extrinsics/Depth scale lesen und als YAML speichern
- Punktwolken:
  - RealSense SDK Export PLY (texturiert)
  - Numpy XYZ (aligned depth + COLOR intrinsics)
  - Optional Open3D

Konvention:
- aligned depth -> color: Backprojection (u,v,z) nutzt COLOR intrinsics.
- Kamera-Optical-Frame: x rechts, y runter, z vorwärts
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pyrealsense2 as rs

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

try:
    import open3d as o3d  # type: ignore
    _HAS_O3D = True
except Exception:
    o3d = None
    _HAS_O3D = False


# ============================================================
# Meta models
# ============================================================

@dataclass(frozen=True)
class RSIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    distortion_coeffs: Tuple[float, ...]


@dataclass(frozen=True)
class RSExtrinsics:
    rotation_row_major_3x3: Tuple[float, ...]  # len=9
    translation_m: Tuple[float, float, float]


@dataclass(frozen=True)
class RSDeviceInfo:
    name: str
    serial_number: str
    firmware_version: str
    product_line: str


@dataclass(frozen=True)
class RSStreamMode:
    width: int
    height: int
    fps: int
    depth_format: str
    color_format: str


@dataclass(frozen=True)
class RSCameraMeta:
    timestamp_utc: str
    device: RSDeviceInfo
    stream_mode: RSStreamMode
    depth_scale_m_per_unit: Optional[float]
    intrinsics_depth: RSIntrinsics
    intrinsics_color: RSIntrinsics
    extrinsics_depth_to_color: RSExtrinsics
    aligned_depth_uses_color_intrinsics: bool = True


# ============================================================
# Helpers
# ============================================================

def _safe_info(dev: rs.device, key: rs.camera_info) -> str:
    try:
        return str(dev.get_info(key))
    except Exception:
        return "unknown"


def _distortion_to_str(model: Any) -> str:
    mapping = {
        rs.distortion.none: "none",
        rs.distortion.modified_brown_conrady: "modified_brown_conrady",
        rs.distortion.inverse_brown_conrady: "inverse_brown_conrady",
        rs.distortion.brown_conrady: "brown_conrady",
        rs.distortion.ftheta: "ftheta",
        rs.distortion.kannala_brandt4: "kannala_brandt4",
    }
    return mapping.get(model, str(model))


def _intrinsics_from_vsp(vsp: rs.video_stream_profile) -> RSIntrinsics:
    intr = vsp.get_intrinsics()
    coeffs = tuple(float(c) for c in intr.coeffs)
    return RSIntrinsics(
        width=int(intr.width),
        height=int(intr.height),
        fx=float(intr.fx),
        fy=float(intr.fy),
        cx=float(intr.ppx),
        cy=float(intr.ppy),
        distortion_model=_distortion_to_str(intr.model),
        distortion_coeffs=coeffs,
    )


def _extrinsics_depth_to_color(depth_vsp: rs.video_stream_profile, color_vsp: rs.video_stream_profile) -> RSExtrinsics:
    ex = depth_vsp.get_extrinsics_to(color_vsp)
    R = tuple(float(v) for v in ex.rotation)
    t = tuple(float(v) for v in ex.translation)
    return RSExtrinsics(rotation_row_major_3x3=R, translation_m=(t[0], t[1], t[2]))


# ============================================================
# Camera wrapper
# ============================================================

class RealsenseD405:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        self.profile: Optional[rs.pipeline_profile] = None
        self.align: Optional[rs.align] = None

        self.rs_pc = rs.pointcloud()
        self._meta: Optional[RSCameraMeta] = None

    # ---------------- lifecycle ----------------

    def start(self, warmup_frames: int = 15) -> None:
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        for _ in range(int(warmup_frames)):
            _ = self.pipeline.wait_for_frames()

        self._meta = self._read_meta()

    def stop(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass
        self.profile = None
        self.align = None

    def close(self) -> None:
        self.stop()

    # ---------------- meta ----------------

    def _read_meta(self) -> RSCameraMeta:
        if self.profile is None:
            raise RuntimeError("Camera not started. Call start() first.")

        dev = self.profile.get_device()
        dev_info = RSDeviceInfo(
            name=_safe_info(dev, rs.camera_info.name),
            serial_number=_safe_info(dev, rs.camera_info.serial_number),
            firmware_version=_safe_info(dev, rs.camera_info.firmware_version),
            product_line=_safe_info(dev, rs.camera_info.product_line),
        )

        depth_scale: Optional[float] = None
        try:
            depth_sensor = dev.first_depth_sensor()
            depth_scale = float(depth_sensor.get_depth_scale())
        except Exception:
            depth_scale = None

        depth_vsp = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_vsp = self.profile.get_stream(rs.stream.color).as_video_stream_profile()

        intr_depth = _intrinsics_from_vsp(depth_vsp)
        intr_color = _intrinsics_from_vsp(color_vsp)
        extr_d2c = _extrinsics_depth_to_color(depth_vsp, color_vsp)

        stream_mode = RSStreamMode(
            width=self.width,
            height=self.height,
            fps=self.fps,
            depth_format="z16",
            color_format="bgr8",
        )

        ts_utc = datetime.now(timezone.utc).isoformat()

        return RSCameraMeta(
            timestamp_utc=ts_utc,
            device=dev_info,
            stream_mode=stream_mode,
            depth_scale_m_per_unit=depth_scale,
            intrinsics_depth=intr_depth,
            intrinsics_color=intr_color,
            extrinsics_depth_to_color=extr_d2c,
            aligned_depth_uses_color_intrinsics=True,
        )

    def get_meta(self) -> RSCameraMeta:
        if self._meta is None:
            self._meta = self._read_meta()
        return self._meta

    def save_meta_yaml(self, path: str | Path) -> None:
        import yaml
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(asdict(self.get_meta()), sort_keys=False), encoding="utf-8")

    def print_meta(self) -> None:
        m = self.get_meta()
        print("=== RealSense Meta ===")
        print(f"timestamp_utc: {m.timestamp_utc}")
        print(f"device: {m.device.name}  serial={m.device.serial_number} fw={m.device.firmware_version} line={m.device.product_line}")
        print(f"stream: {m.stream_mode.width}x{m.stream_mode.height}@{m.stream_mode.fps}")
        print(f"depth_scale_m_per_unit: {m.depth_scale_m_per_unit}")
        print(f"aligned_depth_uses_color_intrinsics: {m.aligned_depth_uses_color_intrinsics}")
        ic = m.intrinsics_color
        idp = m.intrinsics_depth
        print("--- intrinsics(color) ---")
        print(f"  fx={ic.fx:.6f} fy={ic.fy:.6f} cx={ic.cx:.6f} cy={ic.cy:.6f} w={ic.width} h={ic.height}")
        print(f"  distortion={ic.distortion_model} coeffs={list(ic.distortion_coeffs)}")
        print("--- intrinsics(depth) ---")
        print(f"  fx={idp.fx:.6f} fy={idp.fy:.6f} cx={idp.cx:.6f} cy={idp.cy:.6f} w={idp.width} h={idp.height}")
        print(f"  distortion={idp.distortion_model} coeffs={list(idp.distortion_coeffs)}")
        ex = m.extrinsics_depth_to_color
        print("--- extrinsics depth->color ---")
        print(f"  R(row-major)={list(ex.rotation_row_major_3x3)}")
        print(f"  t(m)={list(ex.translation_m)}")
        print("======================")

    # ---------------- frames ----------------

    def get_frames(self, timeout_ms: int = 5000) -> Dict[str, Any]:
        """
        Returns dict:
          color_bgr: uint8 (H,W,3) color (aligned)
          depth_aligned_u16: uint16 (H,W) depth aligned to color
          depth_raw_u16: uint16 (H,W) raw depth

          rs_depth_raw, rs_color_raw
          rs_depth_aligned, rs_color_aligned
        """
        if self.profile is None or self.align is None:
            raise RuntimeError("Camera not started. Call start() first.")

        frames = self.pipeline.wait_for_frames(timeout_ms)

        depth_raw = frames.get_depth_frame()
        color_raw = frames.get_color_frame()

        aligned = self.align.process(frames)
        depth_aligned = aligned.get_depth_frame()
        color_aligned = aligned.get_color_frame()

        if not depth_raw or not color_raw or not depth_aligned or not color_aligned:
            raise RuntimeError("Missing frames from RealSense (depth/color/aligned).")

        depth_raw_u16 = np.asanyarray(depth_raw.get_data()).astype(np.uint16, copy=False)
        depth_aligned_u16 = np.asanyarray(depth_aligned.get_data()).astype(np.uint16, copy=False)
        color_bgr = np.asanyarray(color_aligned.get_data()).astype(np.uint8, copy=False)

        return {
            "color_bgr": color_bgr,
            "depth_aligned_u16": depth_aligned_u16,
            "depth_raw_u16": depth_raw_u16,
            "rs_depth_raw": depth_raw,
            "rs_color_raw": color_raw,
            "rs_depth_aligned": depth_aligned,
            "rs_color_aligned": color_aligned,
            "timestamp_ms": float(color_aligned.get_timestamp()),
            "frame_number": int(color_aligned.get_frame_number()),
        }

    def get_aligned_rgb_depth(self, timeout_ms: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        fr = self.get_frames(timeout_ms=timeout_ms)
        return fr["color_bgr"], fr["depth_aligned_u16"]

    # ---------------- self-check ----------------

    def self_check(
        self,
        patch_px: int = 20,
        treat_65535_as_invalid: bool = True,
        debug_out: Optional[str | Path] = None,
        *,
        compute_global_stats: bool = True,
        compare_rs_deprojection: bool = True,
    ) -> Dict[str, Any]:
        """
        Self-check:
        - depth_scale ausgeben
        - intrinsics ausgeben (color + depth)
        - z_center_m: Median depth im patch_px x patch_px Patch um (cx,cy)
        - optional: globale Depth-Stats
        - optional: RS deprojection vs numpy backprojection Vergleich
        """
        m = self.get_meta()
        if m.depth_scale_m_per_unit is None:
            raise RuntimeError("depth_scale_m_per_unit is None (cannot convert depth to meters).")

        fr = self.get_frames()
        depth = fr["depth_aligned_u16"]
        color = fr["color_bgr"]

        H, W = depth.shape[:2]
        ic = m.intrinsics_color
        ds = float(m.depth_scale_m_per_unit)

        # optischer Mittelpunkt (gerundet), fallback Bildmitte
        u0 = int(round(ic.cx)) if 0 <= ic.cx < W else (W // 2)
        v0 = int(round(ic.cy)) if 0 <= ic.cy < H else (H // 2)

        r = max(1, int(patch_px) // 2)
        u1, u2 = max(0, u0 - r), min(W, u0 + r)
        v1, v2 = max(0, v0 - r), min(H, v0 + r)

        patch = depth[v1:v2, u1:u2].astype(np.uint16, copy=False)

        valid = (patch > 0)
        if treat_65535_as_invalid:
            valid = valid & (patch < 65535)

        vals = patch[valid].astype(np.float64)
        if vals.size == 0:
            z_center_m = float("nan")
            depth_u16_center = int(depth[v0, u0])
        else:
            depth_u16_center = int(np.median(vals))
            z_center_m = float(depth_u16_center) * ds

        # --- optional global stats ---
        global_stats = {}
        if compute_global_stats:
            d = depth.astype(np.uint16, copy=False)
            valid_g = d > 0
            if treat_65535_as_invalid:
                valid_g = valid_g & (d < 65535)
            vg = d[valid_g].astype(np.float64)
            if vg.size > 0:
                z = vg * ds
                global_stats = {
                    "depth_valid_ratio": float(vg.size) / float(d.size),
                    "depth_min_m": float(np.min(z)),
                    "depth_p05_m": float(np.quantile(z, 0.05)),
                    "depth_median_m": float(np.median(z)),
                    "depth_p95_m": float(np.quantile(z, 0.95)),
                    "depth_max_m": float(np.max(z)),
                }
            else:
                global_stats = {
                    "depth_valid_ratio": 0.0,
                }

        # --- optional RS vs numpy deprojection consistency check ---
        proj_check = {}
        if compare_rs_deprojection:
            # numpy backprojection (optical frame)
            z = float(depth_u16_center) * ds
            x = (float(u0) - float(ic.cx)) * z / float(ic.fx)
            y = (float(v0) - float(ic.cy)) * z / float(ic.fy)
            xyz_np = np.array([x, y, z], dtype=np.float64)

            try:
                xyz_rs = self.deproject_rs_aligned(float(u0), float(v0), int(depth_u16_center))
                diff_mm = float(np.linalg.norm(xyz_rs - xyz_np) * 1000.0)
                proj_check = {
                    "xyz_numpy_m": xyz_np.tolist(),
                    "xyz_rs_m": xyz_rs.tolist(),
                    "diff_mm": diff_mm,
                }
            except Exception as exc:
                proj_check = {"error": repr(exc)}

        # Printout
        print("\n========== D405 SELF CHECK ==========")
        print(f"depth_scale_m_per_unit = {m.depth_scale_m_per_unit}")
        print("--- intrinsics (color) ---")
        print(f"w={ic.width} h={ic.height} fx={ic.fx:.6f} fy={ic.fy:.6f} cx={ic.cx:.6f} cy={ic.cy:.6f}")
        print(f"distortion={ic.distortion_model} coeffs={list(ic.distortion_coeffs)}")
        idp = m.intrinsics_depth
        print("--- intrinsics (depth) ---")
        print(f"w={idp.width} h={idp.height} fx={idp.fx:.6f} fy={idp.fy:.6f} cx={idp.cx:.6f} cy={idp.cy:.6f}")
        print(f"distortion={idp.distortion_model} coeffs={list(idp.distortion_coeffs)}")
        print("--- center depth (aligned depth) ---")
        print(f"center_px=(u={u0}, v={v0}) patch={patch_px}x{patch_px} valid_n={int(vals.size)} z_center_m={z_center_m:.6f}")

        if global_stats:
            print("--- global depth stats (aligned) ---")
            for k, v in global_stats.items():
                print(f"{k}: {v}")

        if proj_check:
            print("--- deprojection check ---")
            print(proj_check)

        print("=====================================\n")

        if debug_out is not None:
            if not _HAS_CV2:
                raise RuntimeError("cv2 not installed (needed for debug_out). pip install opencv-python")

            dbg = color.copy()
            cv2.circle(dbg, (u0, v0), 5, (0, 0, 255), -1)
            cv2.rectangle(dbg, (u1, v1), (u2 - 1, v2 - 1), (0, 255, 255), 2)
            cv2.putText(
                dbg,
                f"z_center={z_center_m:.4f}m n={int(vals.size)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            outp = Path(debug_out)
            outp.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(outp), dbg)
            print(f"[SELF_CHECK] wrote debug image: {outp}")

        return {
            "depth_scale_m_per_unit": float(ds),
            "intrinsics_color": asdict(m.intrinsics_color),
            "intrinsics_depth": asdict(m.intrinsics_depth),
            "center_px": (int(u0), int(v0)),
            "patch_px": int(patch_px),
            "valid_n": int(vals.size),
            "z_center_m": float(z_center_m),
            "global_stats": global_stats,
            "deprojection_check": proj_check,
            "timestamp_ms": fr.get("timestamp_ms", None),
            "frame_number": fr.get("frame_number", None),
        }

    # ---------------- PLY export (RealSense) ----------------

    def export_ply_rs(
        self,
        ply_path: str | Path,
        depth_frame: rs.depth_frame,
        color_frame: Optional[rs.video_frame] = None,
    ) -> None:
        p = Path(ply_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if color_frame is not None:
            self.rs_pc.map_to(color_frame)
        pts = self.rs_pc.calculate(depth_frame)

        if color_frame is not None:
            pts.export_to_ply(str(p), color_frame)
        else:
            pts.export_to_ply(str(p), depth_frame)

    # ---------------- file save helpers ----------------

    def save_rgb_depth_png(self, rgb_path: Path, depth_path: Path, color_bgr: np.ndarray, depth_u16: np.ndarray) -> None:
        if not _HAS_CV2:
            raise RuntimeError("cv2 not installed. pip install opencv-python")
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        ok1 = cv2.imwrite(str(rgb_path), color_bgr)
        ok2 = cv2.imwrite(str(depth_path), depth_u16.astype(np.uint16, copy=False))
        if not (ok1 and ok2):
            raise IOError(f"Could not write {rgb_path} / {depth_path}")

    def save_depth_png(self, depth_path: Path, depth_u16: np.ndarray) -> None:
        if not _HAS_CV2:
            raise RuntimeError("cv2 not installed. pip install opencv-python")
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(depth_path), depth_u16.astype(np.uint16, copy=False))
        if not ok:
            raise IOError(f"Could not write {depth_path}")


# ============================================================
# CLI / Quick test
# ============================================================

def _parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--self-check", action="store_true", help="Print depth_scale + intrinsics + z_center_m")
    ap.add_argument("--patch", type=int, default=20, help="Patch size for self-check (px)")
    ap.add_argument("--debug-out", type=str, default="", help="Optional debug png path for self-check")
    ap.add_argument("--save-meta", type=str, default="", help="Write camera_meta.yaml to given path")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cam = RealsenseD405(args.width, args.height, args.fps)
    cam.start()
    try:
        cam.print_meta()

        if args.save_meta:
            cam.save_meta_yaml(Path(args.save_meta))
            print(f"[CLI] wrote meta yaml: {args.save_meta}")

        if args.self_check:
            dbg = args.debug_out.strip() or None
            cam.self_check(patch_px=int(args.patch), debug_out=dbg)

    finally:
        cam.stop()
