#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
capture_dataset_legacy_positions.py

Verbesserte, aber "legacy-kompatible" Version deines alten Skripts.

Was es macht:
- Meca500 fährt die Joint-Posen: m -> l (idx0) -> m (idx1) -> r (idx2) -> m
- Pro Position nimmt RealSense D405:
    - color_aligned (BGR)
    - depth_aligned_u16 (Depth auf Color aligned)
    - depth_raw_u16 (roher Depth)
    - optional: cloud_raw_{idx}.ply (raw depth + raw color)
    - optional: cloud_aligned_{idx}.ply (aligned depth + aligned color)
- Speichert alles in: Datensatz/Aufnahme###/
- Schreibt Koordinaten.txt mit GetPose() (TCP Pose im aktuellen WRF/TRF)
- Wartet pro "Erdbeere" auf ENTER, damit du die nächste Probe einhängen kannst.

Start:
  python src/capture_dataset_legacy_positions.py
oder mit Parametern:
  python src/capture_dataset_legacy_positions.py --dataset_dir Datensatz --ip 192.168.0.100

Hinweis:
- Der Code nutzt exakt deine alten Joint-Positionen (POSITIONS l/m/r).
- TRF/WRF wird wie früher gesetzt (WRF=BRF, TRF fix).
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

import mecademicpy.robot as mdr
from mecademicpy.robot_classes import CommunicationError


# -------------------------
# Defaults (wie dein alter Code)
# -------------------------
DEFAULT_DATASET_DIR = "Datensatz"

DEFAULT_ROBOT_IP = "192.168.0.100"
DEFAULT_ROBOT_VEL = 20
DEFAULT_ROBOT_ACC = 20

# Gelenkpositionen (J1..J6) für links/mitte/rechts (EXAKT aus deinem alten Code)
POSITIONS = {
    "l": (48, 14, 24, -108, 91, -81),
    "m": (0, -35, 35, 0, 0, 135),
    "r": (-41, 14, 24, 106, 89, 6),
}

# RealSense Defaults
DEFAULT_RS_WIDTH = 640
DEFAULT_RS_HEIGHT = 480
DEFAULT_RS_FPS = 30

# Wartezeit nach Motion (Robot + Auto-Exposure)
DEFAULT_SETTLE_SECONDS = 0.5


# -------------------------
# RealSense Kamera (nahe an deinem alten Code, aber robuster)
# -------------------------
@dataclass(frozen=True)
class RsMeta:
    device_name: str
    serial: str
    firmware: str
    product_line: str
    depth_scale_m_per_unit: Optional[float]


class RealsenseCamera:
    def __init__(self, width: int, height: int, fps: int):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()

        # Meta
        dev = self.profile.get_device()

        def safe_info(key) -> str:
            try:
                return dev.get_info(key)
            except Exception:
                return "unknown"

        depth_scale = None
        try:
            depth_sensor = dev.first_depth_sensor()
            depth_scale = float(depth_sensor.get_depth_scale())
        except Exception:
            depth_scale = None

        self.meta = RsMeta(
            device_name=safe_info(rs.camera_info.name),
            serial=safe_info(rs.camera_info.serial_number),
            firmware=safe_info(rs.camera_info.firmware_version),
            product_line=safe_info(rs.camera_info.product_line),
            depth_scale_m_per_unit=depth_scale,
        )

    def _distortion_to_str(self, model) -> str:
        mapping = {
            rs.distortion.none: "none",
            rs.distortion.modified_brown_conrady: "modified_brown_conrady",
            rs.distortion.inverse_brown_conrady: "inverse_brown_conrady",
            rs.distortion.brown_conrady: "brown_conrady",
            rs.distortion.ftheta: "ftheta",
            rs.distortion.kannala_brandt4: "kannala_brandt4",
        }
        return mapping.get(model, str(model))

    def _get_extrinsics_depth_to_color(self) -> Tuple[list, list]:
        depth_vsp = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_vsp = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        ex = depth_vsp.get_extrinsics_to(color_vsp)
        R = list(ex.rotation)      # 9 Werte, row-major
        t = list(ex.translation)   # 3 Werte, Meter
        return R, t

    def save_camera_calib_yaml_txt(self, txt_path: Path) -> None:
        """Speichert Kamera-Infos, Intrinsics, Extrinsics depth->color in eine Textdatei."""
        from datetime import datetime, timezone

        ts_utc = datetime.now(timezone.utc).isoformat()

        depth_vsp = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_vsp = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_intr = depth_vsp.intrinsics
        color_intr = color_vsp.intrinsics
        R, t = self._get_extrinsics_depth_to_color()

        def intr_block(intr, label: str) -> str:
            coeff_lines = "".join(f"      - {c}\n" for c in intr.coeffs)  # 6 spaces before '-'
            return (
                f"  {label}:\n"
                f"    width: {intr.width}\n"
                f"    height: {intr.height}\n"
                f"    fx: {intr.fx}\n"
                f"    fy: {intr.fy}\n"
                f"    cx: {intr.ppx}\n"
                f"    cy: {intr.ppy}\n"
                f"    distortion_model: {self._distortion_to_str(intr.model)}\n"
                f"    distortion_coeffs:\n"
                f"{coeff_lines}"
            )

        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with txt_path.open("w", encoding="utf-8") as f:
            f.write(f"timestamp_utc: '{ts_utc}'\n")
            f.write("device:\n")
            f.write(f"  name: {self.meta.device_name}\n")
            f.write(f"  serial_number: '{self.meta.serial}'\n")
            f.write(f"  firmware_version: {self.meta.firmware}\n")
            f.write(f"  product_line: {self.meta.product_line}\n")

            f.write("stream_mode:\n")
            f.write(f"  width: {self.width}\n")
            f.write(f"  height: {self.height}\n")
            f.write(f"  fps: {self.fps}\n")

            if self.meta.depth_scale_m_per_unit is not None:
                f.write(f"depth_scale_m_per_unit: {self.meta.depth_scale_m_per_unit}\n")
            else:
                f.write("depth_scale_m_per_unit: unknown\n")

            f.write("intrinsics:\n")
            f.write(intr_block(depth_intr, "depth"))
            f.write(intr_block(color_intr, "color"))

            f.write("extrinsics:\n")
            f.write("  depth_to_color:\n")
            f.write("    rotation_row_major_3x3:\n")
            for v in R:
                f.write(f"      - {v}\n")
            f.write("    translation_m:\n")
            for v in t:
                f.write(f"      - {v}\n")

            f.write("notes:\n")
            f.write("  - Intrinsics are constant for a given stream configuration.\n")
            f.write("  - If you resize/crop images later, update fx/fy/cx/cy accordingly.\n")
            f.write("  - Aligned depth-to-color images typically use the color intrinsics.\n")

    def warmup(self, n: int = 15) -> None:
        """Ein paar Frames verwerfen, damit Auto-Exposure/Auto-WhiteBalance stabil wird."""
        for _ in range(int(n)):
            _ = self.pipeline.wait_for_frames()

    def get_frames(self):
        """
        Liefert:
        - color_image (aligned): np.uint8 (H,W,3) BGR
        - depth_aligned_image: np.uint16 (H,W)
        - depth_raw_image: np.uint16 (H,W)
        - depth_frame_raw: rs.depth_frame
        - color_frame_raw: rs.video_frame
        - depth_frame_aligned: rs.depth_frame
        - color_frame_aligned: rs.video_frame
        """
        frames = self.pipeline.wait_for_frames()

        depth_frame_raw = frames.get_depth_frame()
        color_frame_raw = frames.get_color_frame()

        aligned_frames = self.align.process(frames)
        depth_frame_aligned = aligned_frames.get_depth_frame()
        color_frame_aligned = aligned_frames.get_color_frame()

        if (not depth_frame_raw) or (not color_frame_raw) or (not depth_frame_aligned) or (not color_frame_aligned):
            return None, None, None, None, None, None, None

        depth_raw_image = np.asanyarray(depth_frame_raw.get_data())
        depth_aligned_image = np.asanyarray(depth_frame_aligned.get_data())
        color_image = np.asanyarray(color_frame_aligned.get_data())

        return (
            color_image,
            depth_aligned_image,
            depth_raw_image,
            depth_frame_raw,
            color_frame_raw,
            depth_frame_aligned,
            color_frame_aligned,
        )

    def save_ply(self, ply_path: Path, depth_frame, color_frame=None) -> None:
        """Speichert eine Pointcloud als .ply (texturiert, wenn color_frame übergeben wird)."""
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        if color_frame is not None:
            self.pc.map_to(color_frame)
        points = self.pc.calculate(depth_frame)
        if color_frame is not None:
            points.export_to_ply(str(ply_path), color_frame)
        else:
            points.export_to_ply(str(ply_path), depth_frame)

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


# -------------------------
# Robot Helpers (wie früher, aber mit etwas mehr Robustheit)
# -------------------------
def ensure_robot_ready(robot: mdr.Robot) -> None:
    """Roboter aktivieren & homen (inkl. ResetError/ClearMotion)."""
    try:
        robot.ResetError()
    except Exception as exc:
        print(f"[ROBOT] ResetError() failed: {exc!r}")

    try:
        robot.ClearMotion()
    except Exception as exc:
        print(f"[ROBOT] ClearMotion() failed: {exc!r}")

    robot.ActivateAndHome()
    robot.WaitHomed()


def set_wrf_to_brf(robot: mdr.Robot) -> None:
    robot.SetWrf(0, 0, 0, 0, 0, 0)


def set_trf_fixed(robot: mdr.Robot, *, trf_x: float, trf_y: float, trf_z: float, trf_rx: float, trf_ry: float, trf_rz: float) -> None:
    """
    TRF setzen. (Genau wie dein alter Code: (0,0,36mm) und rz=45°)
    Wichtig: Das ist "Tool Reference Frame", nicht dein Kamerapivot-offset.
    """
    robot.SetTrf(float(trf_x), float(trf_y), float(trf_z), float(trf_rx), float(trf_ry), float(trf_rz))


def move_to(robot: mdr.Robot, key: str) -> None:
    if key not in POSITIONS:
        raise ValueError(f"Unbekannte Position '{key}'. Erlaubt: {list(POSITIONS.keys())}")
    robot.MoveJoints(*POSITIONS[key])
    robot.WaitIdle()


def get_tcp_pose(robot: mdr.Robot) -> Tuple[float, float, float, float, float, float]:
    """GetPose() -> TCP pose im aktuellen WRF/TRF."""
    x, y, z, rx, ry, rz = robot.GetPose()
    return float(x), float(y), float(z), float(rx), float(ry), float(rz)


# -------------------------
# Filesystem Helpers
# -------------------------
def create_next_recording_folder(base_dir: Path) -> Path:
    """
    Legt Datensatz/Aufnahme### an (nächster freier Index).
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    pat = re.compile(r"^Aufnahme(\d{3})$")

    nums = []
    for name in os.listdir(str(base_dir)):
        full = base_dir / name
        if not full.is_dir():
            continue
        m = pat.match(name)
        if m:
            nums.append(int(m.group(1)))

    next_num = (max(nums) + 1) if nums else 1
    out_dir = base_dir / f"Aufnahme{next_num:03d}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def save_png_rgb(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise IOError(f"Failed to write {path}")


def save_png_depth_u16(path: Path, depth_u16: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), depth_u16)  # keeps 16-bit
    if not ok:
        raise IOError(f"Failed to write {path}")


# -------------------------
# Main
# -------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--ip", type=str, default=DEFAULT_ROBOT_IP)
    ap.add_argument("--vel", type=float, default=DEFAULT_ROBOT_VEL)
    ap.add_argument("--acc", type=float, default=DEFAULT_ROBOT_ACC)

    ap.add_argument("--width", type=int, default=DEFAULT_RS_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_RS_HEIGHT)
    ap.add_argument("--fps", type=int, default=DEFAULT_RS_FPS)

    ap.add_argument("--settle", type=float, default=DEFAULT_SETTLE_SECONDS)

    # TRF wie früher fest verdrahtet:
    ap.add_argument("--trf_x", type=float, default=0.0)
    ap.add_argument("--trf_y", type=float, default=0.0)
    ap.add_argument("--trf_z", type=float, default=36.0)
    ap.add_argument("--trf_rx", type=float, default=0.0)
    ap.add_argument("--trf_ry", type=float, default=0.0)
    ap.add_argument("--trf_rz", type=float, default=45.0)

    ap.add_argument("--no_ply", action="store_true", help="wenn gesetzt: keine PLYs exportieren")
    ap.add_argument("--no_raw_depth_png", action="store_true", help="wenn gesetzt: depth_raw*.png nicht speichern")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    run_dir = create_next_recording_folder(dataset_dir)

    coord_path = run_dir / "Koordinaten.txt"
    calib_path = run_dir / "CameraCalibration.txt"

    print(f"[DATA] Speichere in: {run_dir}")

    cam = RealsenseCamera(width=args.width, height=args.height, fps=args.fps)

    robot = mdr.Robot()

    try:
        # ---- Robot connect/init ----
        print("[ROBOT] Verbinde ...")
        robot.Connect(args.ip)
        print("[ROBOT] Verbunden.")

        ensure_robot_ready(robot)
        robot.SetJointVel(float(args.vel))
        robot.SetJointAcc(float(args.acc))

        print("[ROBOT] Setze WRF = BRF ...")
        set_wrf_to_brf(robot)

        print("[ROBOT] Setze TRF (fixed) ...")
        set_trf_fixed(
            robot,
            trf_x=args.trf_x, trf_y=args.trf_y, trf_z=args.trf_z,
            trf_rx=args.trf_rx, trf_ry=args.trf_ry, trf_rz=args.trf_rz,
        )

        # ---- Camera warmup + calib once ----
        cam.warmup(15)
        cam.save_camera_calib_yaml_txt(calib_path)
        print(f"[CAM] Gespeichert: {calib_path.name}")

        # ---- One "Strawberry" capture per ENTER ----
        input("\n[USER] Hänge Erdbeere ein / richte Szene aus. ENTER = 3 Views aufnehmen (Ctrl+C=Abbruch) ...")

        # Ablauf: Mitte -> Links (0) -> Mitte (1) -> Rechts (2) -> Mitte
        print("[SEQ] Fahre zunächst nach Mitte ...")
        move_to(robot, "m")
        time.sleep(float(args.settle))

        sequence = [("l", 0), ("m", 1), ("r", 2)]

        with coord_path.open("w", encoding="utf-8") as f:
            f.write("# Koordinaten (GetPose) in mm/deg, WRF=BRF, TRF gesetzt\n")
            f.write("# idx;pos;x_mm;y_mm;z_mm;rx_deg;ry_deg;rz_deg;timestamp\n")

            for pos_key, idx in sequence:
                print(f"\n[SEQ] Fahre nach {pos_key} (idx={idx}) ...")
                move_to(robot, pos_key)
                time.sleep(float(args.settle))

                fr = cam.get_frames()
                if fr[0] is None:
                    raise RuntimeError("Keine Frames von der RealSense erhalten (Frames=None).")

                color, depth_aligned, depth_raw, depth_frame_raw, color_frame_raw, depth_frame_aligned, color_frame_aligned = fr

                # Save PNGs
                save_png_rgb(run_dir / f"color{idx}.png", color)
                save_png_depth_u16(run_dir / f"depth_aligned{idx}.png", depth_aligned)
                if not args.no_raw_depth_png:
                    save_png_depth_u16(run_dir / f"depth_raw{idx}.png", depth_raw)

                print(f"[CAM] PNGs gespeichert (idx={idx})")

                # Save PLYs (optional)
                if not args.no_ply:
                    ply_raw_path = run_dir / f"cloud_raw{idx}.ply"
                    cam.save_ply(ply_raw_path, depth_frame_raw, color_frame_raw)
                    print(f"[CAM] Gespeichert: {ply_raw_path.name}")

                    ply_aligned_path = run_dir / f"cloud_aligned{idx}.ply"
                    cam.save_ply(ply_aligned_path, depth_frame_aligned, color_frame_aligned)
                    print(f"[CAM] Gespeichert: {ply_aligned_path.name}")

                # Write TCP pose
                x, y, z, rx, ry, rz = get_tcp_pose(robot)
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{idx};{pos_key};{x:.3f};{y:.3f};{z:.3f};{rx:.3f};{ry:.3f};{rz:.3f};{ts}\n")
                f.flush()

                print(f"[ROBOT] Pose gespeichert: x={x:.3f} y={y:.3f} z={z:.3f}  rx={rx:.3f} ry={ry:.3f} rz={rz:.3f}")

        print("\n[SEQ] Fahre zurück nach Mitte ...")
        move_to(robot, "m")
        robot.WaitIdle()

        print(f"\n[DONE] Fertig. Output: {run_dir}")

    except (TimeoutError, CommunicationError) as e:
        print("[ROBOT] Kommunikationsfehler:", e)

    except KeyboardInterrupt:
        print("\n[ABORT] Abgebrochen per Ctrl+C.")

    except Exception as e:
        print("[ERROR]", repr(e))

    finally:
        # Kamera stoppen
        cam.close()

        # Roboter sauber deaktivieren
        try:
            robot.WaitIdle()
        except Exception:
            pass
        try:
            robot.DeactivateRobot()
            if hasattr(robot, "WaitDeactivated"):
                robot.WaitDeactivated()
        except Exception:
            pass
        try:
            robot.Disconnect()
        except Exception:
            pass

        print("[CLEANUP] Roboter getrennt, Kamera gestoppt.")


if __name__ == "__main__":
    main()
