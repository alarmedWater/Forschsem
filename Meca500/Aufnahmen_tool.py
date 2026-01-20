#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kombiniertes Programm:
- Meca500 Roboterarm fährt in die Positionen: Mitte -> Links (Aufnahme0) -> Mitte (Aufnahme1) -> Rechts (Aufnahme2) -> Mitte
- Intel RealSense nimmt pro Position ein aligned Color- & Depth-Frame auf
- Speichert alles in: Datensatz/Aufnahme###/  (automatisch nächster freier Ordner)
- Schreibt pro Aufnahme die aktuelle TRF/TCP Pose (GetPose) in Koordinaten.txt
"""

import os
import re
import time

import cv2
import numpy as np
import pyrealsense2 as rs

import mecademicpy.robot as mdr
from mecademicpy.robot_classes import CommunicationError


# -------------------------
# Einstellungen
# -------------------------
DATASET_DIR = "Datensatz"

ROBOT_IP = "192.168.0.100"
ROBOT_VEL = 20
ROBOT_ACC = 20

# Gelenkpositionen (J1..J6) für links/mitte/rechts
POSITIONS = {
    "l":  (48, 14, 24, -108, 91, -81),
    "m":  (0, -35, 35, 0, 0, 135),
    "r": (-41, 14, 24, 106, 89, 6)
}

# RealSense Settings
RS_WIDTH = 640
RS_HEIGHT = 480
RS_FPS = 30

# kleine Wartezeit nach Erreichen einer Position, bevor das Bild aufgenommen wird
SETTLE_SECONDS = 0.5


# -------------------------
# RealSense Kamera
# -------------------------
class RealsenseCamera:
    def __init__(self, width=RS_WIDTH, height=RS_HEIGHT, fps=RS_FPS):
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()

        # Device-Infos + Depth-Scale
        dev = self.profile.get_device()

        def safe_info(key):
            try:
                return dev.get_info(key)
            except Exception:
                return "unknown"

        self.device_name = safe_info(rs.camera_info.name)
        self.serial = safe_info(rs.camera_info.serial_number)
        self.firmware = safe_info(rs.camera_info.firmware_version)
        self.product_line = safe_info(rs.camera_info.product_line)

        try:
            depth_sensor = dev.first_depth_sensor()
            self.depth_scale = float(depth_sensor.get_depth_scale())
        except Exception:
            self.depth_scale = None

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

    def _get_extrinsics_depth_to_color(self):
        depth_vsp = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_vsp = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        ex = depth_vsp.get_extrinsics_to(color_vsp)
        R = list(ex.rotation)      # 9 Werte, row-major
        t = list(ex.translation)   # 3 Werte, Meter
        return R, t

    def save_camera_calib_yaml_txt(self, txt_path: str):
        """
        Speichert Kamera-Infos 1x pro Aufnahme-Ordner im YAML-ähnlichen Textformat:
        - device info, stream_mode, depth_scale
        - intrinsics (native depth + native color)
        - extrinsics depth_to_color
        """
        from datetime import datetime, timezone
        ts_utc = datetime.now(timezone.utc).isoformat()

        depth_vsp = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_vsp = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_intr = depth_vsp.intrinsics
        color_intr = color_vsp.intrinsics

        R, t = self._get_extrinsics_depth_to_color()

        def intr_block(intr, label: str):
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

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"timestamp_utc: '{ts_utc}'\n")
            f.write("device:\n")
            f.write(f"  name: {self.device_name}\n")
            f.write(f"  serial_number: '{self.serial}'\n")
            f.write(f"  firmware_version: {self.firmware}\n")
            f.write(f"  product_line: {self.product_line}\n")
            f.write("stream_mode:\n")
            f.write(f"  width: {self.width}\n")
            f.write(f"  height: {self.height}\n")
            f.write(f"  fps: {self.fps}\n")

            if self.depth_scale is not None:
                f.write(f"depth_scale_m_per_unit: {self.depth_scale}\n")
            else:
                f.write("depth_scale_m_per_unit: unknown\n")

            f.write("intrinsics:\n")
            f.write(intr_block(depth_intr, "depth"))
            f.write(intr_block(color_intr, "color"))

            f.write("extrinsics:\n")
            f.write("  depth_to_color:\n")
            f.write("    rotation_row_major_3x3:\n")
            for v in R:
                f.write(f"      - {v}\n")   # 6 spaces before '-'
            f.write("    translation_m:\n")
            for v in t:
                f.write(f"      - {v}\n")   # 6 spaces before '-'

            f.write("notes:\n")
            f.write("  - Intrinsics are constant for a given stream configuration.\n")
            f.write("  - If you resize/crop images later, update fx/fy/cx/cy accordingly.\n")
            f.write("  - Aligned depth-to-color images typically use the color intrinsics.\n")


    def get_frames_alt(self):
        """Gibt aligned color & depth als numpy arrays zurück UND die rs.frames für .ply Export"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None, None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image, depth_frame, color_frame
    
    def get_frames(self):
        """
        Liefert:
        - color_image: numpy (Color-Frame aus aligned frameset)
        - depth_aligned_image: numpy (Depth auf Color aligned)
        - depth_raw_image: numpy (Depth roh)
        - depth_frame_raw: rs.depth_frame (roh)
        - color_frame_raw: rs.video_frame (roh)
        - depth_frame_aligned: rs.depth_frame (aligned)
        - color_frame_aligned: rs.video_frame (aligned)
        """
        frames = self.pipeline.wait_for_frames()

        # Rohframes
        depth_frame_raw = frames.get_depth_frame()
        color_frame_raw = frames.get_color_frame()

        # Aligned frames (Depth -> Color)
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



    def save_ply(self, ply_path: str, depth_frame, color_frame=None):
        """Speichert eine Pointcloud als .ply (texturiert, wenn color_frame übergeben wird)."""
        if color_frame is not None:
            self.pc.map_to(color_frame)

        points = self.pc.calculate(depth_frame)
        if color_frame is not None:
            points.export_to_ply(ply_path, color_frame)
        else:
            points.export_to_ply(ply_path, depth_frame)

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


# -------------------------
# Roboter Helpers
# -------------------------
def ensure_robot_ready(robot) -> None:
    """Roboter aktivieren & homen (inkl. Fehler/ClearMotion)."""
    try:
        robot.ResetError()
    except Exception as exc:
        print(f"[ensure_robot_ready] ResetError() failed: {exc!r}")

    try:
        robot.ClearMotion()
    except Exception as exc:
        print(f"[ensure_robot_ready] ClearMotion() failed: {exc!r}")

    robot.ActivateAndHome()
    robot.WaitHomed()


def set_wrf_to_brf(robot) -> None:
    """WRF auf BRF (Basis) setzen."""
    robot.SetWrf(0, 0, 0, 0, 0, 0)


def set_trf_from_tcp(robot) -> None:
    """
    TRF setzen.
    Hier wie in deinem Skript fest verdrahtet:
      Translation: (0,0,36mm)
      Rotation: rz=45°
    """
    trf_x = 0
    trf_y = 0
    trf_z = 36
    trf_rx = 0
    trf_ry = 0
    trf_rz = 45
    robot.SetTrf(trf_x, trf_y, trf_z, trf_rx, trf_ry, trf_rz)


def move_to(robot, key: str) -> None:
    """Fährt in eine definierte Position (l/m/r)."""
    if key not in POSITIONS:
        raise ValueError(f"Unbekannte Position '{key}'. Erlaubt: {list(POSITIONS.keys())}")
    robot.MoveJoints(*POSITIONS[key])
    robot.WaitIdle()


def get_trf_pose(robot):
    """Pose des TCP/TRF im aktuellen WRF (hier BRF), wie GetPose() liefert."""
    x, y, z, rx, ry, rz = robot.GetPose()
    return x, y, z, rx, ry, rz


# -------------------------
# Dateisystem Helpers
# -------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def create_next_recording_folder(base_dir: str) -> str:
    """
    Legt Datensatz/Aufnahme### an.
    Sucht bestehende Aufnahme-Ordner und nimmt die nächsthöhere Nummer.
    """
    ensure_dir(base_dir)
    pat = re.compile(r"^Aufnahme(\d{3})$")

    nums = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        m = pat.match(name)
        if m:
            nums.append(int(m.group(1)))

    next_num = (max(nums) + 1) if nums else 1
    out_dir = os.path.join(base_dir, f"Aufnahme{next_num:03d}")
    os.makedirs(out_dir, exist_ok=False)
    return out_dir


def save_capture(out_dir: str, idx: int, color, depth) -> None:
    """Speichert color/depth als PNG mit gewünschtem Namen."""
    color_path = os.path.join(out_dir, f"color{idx}.png")
    depth_path = os.path.join(out_dir, f"depth{idx}.png")

    ok1 = cv2.imwrite(color_path, color)
    ok2 = cv2.imwrite(depth_path, depth)  # depth ist uint16 -> 16-bit PNG

    if not ok1 or not ok2:
        raise IOError(f"Konnte Bilder nicht speichern: {color_path} / {depth_path}")

def save_capture_all(out_dir: str, idx: int, color, depth_aligned, depth_raw) -> None:
    """Speichert color + depth_aligned + depth_raw als PNG."""
    color_path = os.path.join(out_dir, f"color{idx}.png")
    depth_aligned_path = os.path.join(out_dir, f"depth_aligned{idx}.png")
    depth_raw_path = os.path.join(out_dir, f"depth_raw{idx}.png")

    ok1 = cv2.imwrite(color_path, color)
    ok2 = cv2.imwrite(depth_aligned_path, depth_aligned)  # uint16 -> 16-bit PNG
    ok3 = cv2.imwrite(depth_raw_path, depth_raw)          # uint16 -> 16-bit PNG

    if not (ok1 and ok2 and ok3):
        raise IOError(f"Konnte Bilder nicht speichern: {color_path} / {depth_aligned_path} / {depth_raw_path}")

# -------------------------
# Hauptablauf
# -------------------------
def main():
    # 1) Datensatz-Ordner prüfen/anlegen + neuen Aufnahme-Ordner erstellen
    ensure_dir(DATASET_DIR)
    run_dir = create_next_recording_folder(DATASET_DIR)
    coord_path = os.path.join(run_dir, "Koordinaten.txt")

    print(f"[DATA] Speichere in: {run_dir}")

    # 2) Kamera starten
    cam = RealsenseCamera()

    # 3) Roboter verbinden & konfigurieren
    robot = mdr.Robot()

    try:
        print("[ROBOT] Verbinde ...")
        robot.Connect(ROBOT_IP)
        print("[ROBOT] Verbunden.")

        ensure_robot_ready(robot)
        robot.SetJointVel(ROBOT_VEL)
        robot.SetJointAcc(ROBOT_ACC)

        print("[ROBOT] Setze WRF = BRF ...")
        set_wrf_to_brf(robot)
        print("[ROBOT] Setze TRF ...")
        set_trf_from_tcp(robot)

        # Ablauf: Mitte -> Links (0) -> Mitte (1) -> Rechts (2) -> Mitte
        print("[SEQ] Fahre zunächst nach Mitte ...")
        move_to(robot, "m")
        time.sleep(SETTLE_SECONDS)

        sequence = [("l", 0), ("m", 1), ("r", 2)]

        calib_written = False
        calib_path = os.path.join(run_dir, "CameraCalibration.txt")

        with open(coord_path, "w", encoding="utf-8") as f:
            f.write("# Koordinaten (GetPose) in mm/deg, WRF=BRF, TRF gesetzt\n")
            f.write("# idx;pos;x_mm;y_mm;z_mm;rx_deg;ry_deg;rz_deg;timestamp\n")

            for pos_key, idx in sequence:
                print(f"[SEQ] Fahre nach {pos_key} ...")
                move_to(robot, pos_key)
                time.sleep(SETTLE_SECONDS)

                # Bild aufnehmen
                color, depth_aligned, depth_raw, depth_frame_raw, color_frame_raw, depth_frame_aligned, color_frame_aligned = cam.get_frames()
                if color is None or depth_aligned is None or depth_raw is None:
                    raise RuntimeError("Keine Frames von der RealSense erhalten (Frames = None).")

                if not calib_written:
                    cam.save_camera_calib_yaml_txt(calib_path)
                    calib_written = True
                    print("[CAM] Gespeichert: CameraCalibration.txt")

                # PNGs speichern
                save_capture_all(run_dir, idx, color, depth_aligned, depth_raw)
                print(f"[CAM] Gespeichert: color{idx}.png + depth_aligned{idx}.png + depth_raw{idx}.png")

                # PLY aus RAW depth + RAW color (wie bisher)
                ply_raw_path = os.path.join(run_dir, f"cloud_raw{idx}.ply")
                cam.save_ply(ply_raw_path, depth_frame_raw, color_frame_raw)
                print(f"[CAM] Gespeichert: cloud_raw{idx}.ply")

                # PLY aus ALIGNED depth + ALIGNED color
                ply_aligned_path = os.path.join(run_dir, f"cloud_aligned{idx}.ply")
                cam.save_ply(ply_aligned_path, depth_frame_aligned, color_frame_aligned)
                print(f"[CAM] Gespeichert: cloud_aligned{idx}.ply")


                # Koordinaten abfragen und in Datei schreiben
                x, y, z, rx, ry, rz = get_trf_pose(robot)
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{idx};{pos_key};{x:.3f};{y:.3f};{z:.3f};{rx:.3f};{ry:.3f};{rz:.3f};{ts}\n")
                f.flush()
                print(f"[ROBOT] Pose gespeichert (idx={idx}, pos={pos_key}).")

        print("[SEQ] Fahre zurück nach Mitte ...")
        move_to(robot, "m")
        robot.WaitIdle()

        print("[DONE] Fertig.")

    except (TimeoutError, CommunicationError) as e:
        print("[ROBOT] Kommunikationsfehler:", e)

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
