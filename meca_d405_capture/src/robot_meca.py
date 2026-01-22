#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
robot_meca.py

Robuster Wrapper für Mecademic Meca500:
- connect / activate+home
- WRF setzen (typisch BRF)
- TRF setzen (wie beim Capturen)
- MoveJoints zu benannten Positionen
- Pose lesen (bevorzugt GetRtCartPos, fallback GetPose)
- WRF/TRF/Joints loggen + Pose-Stabilitätscheck
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import time
import yaml
import numpy as np

import mecademicpy.robot as mdr
from mecademicpy.robot_classes import CommunicationError


@dataclass(frozen=True)
class MecaPose:
    """Cartesian pose in mm/deg."""
    x_mm: float
    y_mm: float
    z_mm: float
    rx_deg: float
    ry_deg: float
    rz_deg: float
    source: str  # "GetRtCartPos" or "GetPose"

    def as_tuple6(self) -> Tuple[float, float, float, float, float, float]:
        return (self.x_mm, self.y_mm, self.z_mm, self.rx_deg, self.ry_deg, self.rz_deg)


class Meca500Controller:
    def __init__(
        self,
        ip: str,
        positions_joints_deg: Dict[str, Tuple[float, float, float, float, float, float]],
        joint_vel: float = 20.0,
        joint_acc: float = 20.0,
        settle_s: float = 0.5,
        verbose: bool = True,
    ) -> None:
        self.ip = str(ip)
        self.positions = dict(positions_joints_deg)
        self.joint_vel = float(joint_vel)
        self.joint_acc = float(joint_acc)
        self.settle_s = float(settle_s)
        self.verbose = bool(verbose)
        self.robot: Optional[mdr.Robot] = None

    # ---------- config helper ----------

    @staticmethod
    def from_config_yaml(cfg_path: str | Path, verbose: bool = True) -> "Meca500Controller":
        """
        Erwartet in YAML:
        robot:
          ip: "..."
          joint_vel: 20
          joint_acc: 20
          settle_s: 0.5
          positions: { l: [..6..], m: [..6..], ... }

        (optional) alternative Stelle:
        capture:
          positions: { ... }  # wird als Fallback akzeptiert
        """
        p = Path(cfg_path)
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        rob = raw.get("robot", {}) or {}

        ip = rob.get("ip", "192.168.0.100")
        joint_vel = float(rob.get("joint_vel", 20))
        joint_acc = float(rob.get("joint_acc", 20))
        settle_s = float(rob.get("settle_s", 0.5))

        pos_raw = rob.get("positions", None)
        if not isinstance(pos_raw, dict) or not pos_raw:
            # fallback: some configs store under capture.positions
            cap = raw.get("capture", {}) or {}
            pos_raw = cap.get("positions", {})
        if not isinstance(pos_raw, dict) or not pos_raw:
            raise ValueError("robot.positions (oder capture.positions) fehlt in config.yaml")

        positions: Dict[str, Tuple[float, float, float, float, float, float]] = {}
        for k, v in pos_raw.items():
            if not (isinstance(v, list) and len(v) == 6):
                raise ValueError(f"positions['{k}'] must be list of 6 numbers")
            positions[str(k)] = tuple(float(x) for x in v)  # type: ignore

        return Meca500Controller(
            ip=str(ip),
            positions_joints_deg=positions,
            joint_vel=joint_vel,
            joint_acc=joint_acc,
            settle_s=settle_s,
            verbose=verbose,
        )

    # ---------- connection ----------

    def connect(self) -> None:
        if self.robot is not None:
            return
        self.robot = mdr.Robot()
        if self.verbose:
            print(f"[MECA] Connecting to {self.ip} ...")
        self.robot.Connect(self.ip)
        if self.verbose:
            print("[MECA] Connected.")

    def activate_and_home(self) -> None:
        if self.robot is None:
            raise RuntimeError("Robot not connected. Call connect().")

        # try to clear state
        try:
            self.robot.ResetError()
        except Exception as exc:
            if self.verbose:
                print(f"[MECA] ResetError failed: {exc!r}")

        try:
            self.robot.ClearMotion()
        except Exception as exc:
            if self.verbose:
                print(f"[MECA] ClearMotion failed: {exc!r}")

        if self.verbose:
            print("[MECA] ActivateAndHome ...")
        self.robot.ActivateAndHome()
        self.robot.WaitHomed()

        self.set_joint_vel_acc(self.joint_vel, self.joint_acc)

    def set_joint_vel_acc(self, vel: float, acc: float) -> None:
        if self.robot is None:
            raise RuntimeError("Robot not connected.")
        self.joint_vel = float(vel)
        self.joint_acc = float(acc)
        self.robot.SetJointVel(self.joint_vel)
        self.robot.SetJointAcc(self.joint_acc)
        if self.verbose:
            print(f"[MECA] JointVel={self.joint_vel} JointAcc={self.joint_acc}")

    # ---------- WRF / TRF ----------

    def set_wrf_brf(self) -> None:
        """Set WRF = BRF."""
        if self.robot is None:
            raise RuntimeError("Robot not connected.")
        self.robot.SetWrf(0, 0, 0, 0, 0, 0)
        if self.verbose:
            print("[MECA] WRF=BRF set (0,0,0,0,0,0)")

    def set_trf_mm_deg(self, x_mm: float, y_mm: float, z_mm: float, rx_deg: float, ry_deg: float, rz_deg: float) -> None:
        """Set TRF in mm/deg."""
        if self.robot is None:
            raise RuntimeError("Robot not connected.")
        self.robot.SetTrf(float(x_mm), float(y_mm), float(z_mm), float(rx_deg), float(ry_deg), float(rz_deg))
        if self.verbose:
            print(f"[MECA] TRF set: t=({x_mm:.3f},{y_mm:.3f},{z_mm:.3f})mm r=({rx_deg:.3f},{ry_deg:.3f},{rz_deg:.3f})deg")

    def _try_get_wrf(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Best-effort: returns (x,y,z,rx,ry,rz) in mm/deg if supported by mecademicpy."""
        if self.robot is None:
            return None
        for name in ("GetWrf", "GetWRF"):
            if hasattr(self.robot, name):
                try:
                    v = getattr(self.robot, name)()
                    if isinstance(v, (list, tuple)) and len(v) >= 6:
                        return tuple(float(x) for x in v[:6])  # type: ignore
                except Exception:
                    return None
        return None

    def _try_get_trf(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Best-effort: returns (x,y,z,rx,ry,rz) in mm/deg if supported by mecademicpy."""
        if self.robot is None:
            return None
        for name in ("GetTrf", "GetTRF"):
            if hasattr(self.robot, name):
                try:
                    v = getattr(self.robot, name)()
                    if isinstance(v, (list, tuple)) and len(v) >= 6:
                        return tuple(float(x) for x in v[:6])  # type: ignore
                except Exception:
                    return None
        return None

    def set_wrf_trf_from_config(self, cfg_path: str | Path) -> None:
        """
        Unterstützt Keys:
          robot.wrf_equals_brf: true/false
          robot.trf_set_mm_deg: [..6..]
          robot.trf_set_during_capture_mm_deg: [..6..]   (dein bisheriger Key)
        """
        raw = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
        rob = raw.get("robot", {}) or {}

        if bool(rob.get("wrf_equals_brf", True)):
            self.set_wrf_brf()

        trf = None
        for key in ("trf_set_during_capture_mm_deg", "trf_set_mm_deg", "trf_set_mmdeg", "trf_set"):
            v = rob.get(key, None)
            if isinstance(v, list) and len(v) == 6:
                trf = [float(x) for x in v]
                break

        if trf is not None:
            self.set_trf_mm_deg(*trf)
        elif self.verbose:
            print("[MECA] No robot.trf_set_* in config -> TRF unchanged")

        # log current WRF/TRF if possible
        wrf = self._try_get_wrf()
        trf_now = self._try_get_trf()
        if self.verbose:
            print(f"[MECA] GetWRF available? {'yes' if wrf is not None else 'no'}")
            print(f"[MECA] GetTRF available? {'yes' if trf_now is not None else 'no'}")
            if wrf is not None:
                print(f"[MECA] WRF(mm/deg) = {wrf}")
            if trf_now is not None:
                print(f"[MECA] TRF(mm/deg) = {trf_now}")

    # ---------- motion ----------

    def move_to(self, key: str) -> None:
        if self.robot is None:
            raise RuntimeError("Robot not connected.")
        if key not in self.positions:
            raise ValueError(f"Unknown pose key '{key}'. Available: {sorted(self.positions.keys())}")

        j = self.positions[key]
        if self.verbose:
            print(f"[MECA] MoveJoints {key}: {j}")
        self.robot.MoveJoints(*[float(v) for v in j])
        self.robot.WaitIdle()
        time.sleep(self.settle_s)

    def wait_idle(self) -> None:
        if self.robot is not None:
            self.robot.WaitIdle()

    # ---------- pose / joints ----------

    def get_joints_deg(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Best-effort: returns joints (J1..J6) in deg, if API offers it.
        mecademicpy sometimes has GetJoints() or GetRtJointPos() depending on version.
        """
        if self.robot is None:
            return None

        # try common method names
        for name in ("GetJoints", "GetRtJointPos", "GetRtJoints", "GetJointPos"):
            if hasattr(self.robot, name):
                try:
                    v = getattr(self.robot, name)()
                    if isinstance(v, (list, tuple)) and len(v) >= 6:
                        return tuple(float(x) for x in v[:6])  # type: ignore
                except Exception:
                    return None
        return None

    def get_pose_mm_deg(self) -> MecaPose:
        """
        Prefer realtime encoder-based pose if available: GetRtCartPos()
        Fallback to GetPose().

        Note: exact availability depends on mecademicpy version.
        """
        if self.robot is None:
            raise RuntimeError("Robot not connected.")

        # Prefer realtime cart pose if available
        for name in ("GetRtCartPos", "GetRtCartPose"):
            if hasattr(self.robot, name):
                try:
                    v = getattr(self.robot, name)()
                    if isinstance(v, (list, tuple)) and len(v) >= 6:
                        x, y, z, rx, ry, rz = (float(v[i]) for i in range(6))
                        return MecaPose(x, y, z, rx, ry, rz, source=name)
                except Exception:
                    pass

        # fallback
        x, y, z, rx, ry, rz = self.robot.GetPose()
        return MecaPose(float(x), float(y), float(z), float(rx), float(ry), float(rz), source="GetPose")

    def pose_stability_check(self, samples: int = 8, dt_s: float = 0.05) -> Dict[str, float]:
        """
        Reads pose multiple times and reports max deviation (mm/deg).
        """
        poses: List[MecaPose] = []
        for _ in range(int(samples)):
            poses.append(self.get_pose_mm_deg())
            time.sleep(float(dt_s))

        xs = np.array([p.x_mm for p in poses], dtype=np.float64)
        ys = np.array([p.y_mm for p in poses], dtype=np.float64)
        zs = np.array([p.z_mm for p in poses], dtype=np.float64)
        rxs = np.array([p.rx_deg for p in poses], dtype=np.float64)
        rys = np.array([p.ry_deg for p in poses], dtype=np.float64)
        rzs = np.array([p.rz_deg for p in poses], dtype=np.float64)

        return {
            "pose_source": poses[-1].source if poses else "unknown",
            "max_dx_mm": float(xs.max() - xs.min()),
            "max_dy_mm": float(ys.max() - ys.min()),
            "max_dz_mm": float(zs.max() - zs.min()),
            "max_drx_deg": float(rxs.max() - rxs.min()),
            "max_dry_deg": float(rys.max() - rys.min()),
            "max_drz_deg": float(rzs.max() - rzs.min()),
        }

    def print_pose(self, prefix: str = "[MECA] Pose") -> None:
        p = self.get_pose_mm_deg()
        print(
            f"{prefix}({p.source}): x={p.x_mm:.3f} y={p.y_mm:.3f} z={p.z_mm:.3f}  "
            f"rx={p.rx_deg:.3f} ry={p.ry_deg:.3f} rz={p.rz_deg:.3f} (mm/deg)"
        )

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Snapshot that is useful to store per run.
        """
        wrf = self._try_get_wrf()
        trf = self._try_get_trf()
        joints = self.get_joints_deg()
        pose = self.get_pose_mm_deg()

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ip": self.ip,
            "joint_vel": self.joint_vel,
            "joint_acc": self.joint_acc,
            "wrf_mm_deg": list(wrf) if wrf is not None else None,
            "trf_mm_deg": list(trf) if trf is not None else None,
            "joints_deg": list(joints) if joints is not None else None,
            "pose_mm_deg": list(pose.as_tuple6()),
            "pose_source": pose.source,
            "positions_keys": sorted(list(self.positions.keys())),
        }

    # ---------- shutdown ----------

    def disconnect(self) -> None:
        if self.robot is None:
            return
        try:
            self.robot.WaitIdle()
        except Exception:
            pass
        try:
            self.robot.DeactivateRobot()
            self.robot.WaitDeactivated()
        except Exception:
            pass
        try:
            self.robot.Disconnect()
        except Exception:
            pass
        if self.verbose:
            print("[MECA] Disconnected.")
        self.robot = None
