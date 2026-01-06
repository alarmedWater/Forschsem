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
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

    def get_frames(self):
        """Gibt aligned color & depth als numpy arrays zurück."""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image

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

        with open(coord_path, "w", encoding="utf-8") as f:
            f.write("# Koordinaten (GetPose) in mm/deg, WRF=BRF, TRF gesetzt\n")
            f.write("# idx;pos;x_mm;y_mm;z_mm;rx_deg;ry_deg;rz_deg;timestamp\n")

            for pos_key, idx in sequence:
                print(f"[SEQ] Fahre nach {pos_key} ...")
                move_to(robot, pos_key)
                time.sleep(SETTLE_SECONDS)

                # Bild aufnehmen
                color, depth = cam.get_frames()
                if color is None or depth is None:
                    raise RuntimeError("Keine Frames von der RealSense erhalten (color/depth = None).")

                save_capture(run_dir, idx, color, depth)
                print(f"[CAM] Gespeichert: color{idx}.png + depth{idx}.png")

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
