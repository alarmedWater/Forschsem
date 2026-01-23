#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
robot_meca.py

Robuster Wrapper für Mecademic Meca500 (mecademicpy):

Ziele / Improvements gegenüber deiner Version:
- verhindert Auto-Disconnect bei Exceptions (disconnect_on_exception=False, wenn verfügbar)
- robustes Activate+Home (Fallback, falls ActivateAndHome "not activated" o.ä. liefert)
- robuste Motion-Wrapper:
  - WaitIdle() mit sauberem Fehler-Reset, damit nach "robot is in error" weitergearbeitet werden kann
  - move_pose_mm_deg() mit:
      * optionalem "Safe-Approach" (hoch -> XY -> runter)
      * optionalem Retry via "safe joint pose" (aus config)
      * besserer Fehlermeldung (inkl. Zielpose)
- aus YAML:
  robot:
    ip: "192.168.0.100"
    joint_vel: 20
    joint_acc: 20
    settle_s: 0.5
    positions: { key: [j1..j6], ... }   # Joint-Posen (optional, wenn du move_to nutzt)
    safe_joints_deg: [j1..j6]           # optional: Safe Joint Pose (z.B. "Mitte/oben")
    wrf_equals_brf: true
    trf_set_during_capture_mm_deg: [x,y,z,rx,ry,rz]   # oder trf_set_mm_deg
    pivot_z_clearance_mm: 60.0          # optional Default für move_pose_mm_deg safe approach

Hinweis:
- Cartesian Posen sind (x,y,z,rx,ry,rz) in mm/deg im aktuellen WRF/TRF.
- "Pose out of reach" (1016) heißt: IK/Workspace/Joint-Limits/Orientierung/WRF/TRF -> nicht erreichbar.
  Dagegen hilft meistens:
  - andere Zwischenpose (safe joints)
  - andere Orientierung
  - kleinerer Clearance oder ein anderer Weg (MoveLin vs MovePose)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import time
import yaml
import numpy as np

import mecademicpy.robot as mdr
from mecademicpy.robot_classes import CommunicationError, InterruptException, DisconnectError


# ----------------------------
# Data models
# ----------------------------

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


# ----------------------------
# Controller
# ----------------------------

class Meca500Controller:
    def __init__(
        self,
        ip: str,
        positions_joints_deg: Dict[str, Tuple[float, float, float, float, float, float]],
        joint_vel: float = 20.0,
        joint_acc: float = 20.0,
        settle_s: float = 0.5,
        verbose: bool = True,
        *,
        safe_joints_deg: Optional[Tuple[float, float, float, float, float, float]] = None,
        default_z_clearance_mm: float = 60.0,
    ) -> None:
        self.ip = str(ip)
        self.positions = dict(positions_joints_deg)
        self.joint_vel = float(joint_vel)
        self.joint_acc = float(joint_acc)
        self.settle_s = float(settle_s)
        self.verbose = bool(verbose)

        self.safe_joints_deg = safe_joints_deg
        self.default_z_clearance_mm = float(default_z_clearance_mm)

        self.robot: Optional[mdr.Robot] = None

    # ---------- config helper ----------

    @staticmethod
    def from_config_yaml(cfg_path: Union[str, Path], verbose: bool = True) -> "Meca500Controller":
        """
        Erwartet YAML:
        robot:
          ip: "192.168.0.100"
          joint_vel: 20
          joint_acc: 20
          settle_s: 0.5
          positions: { name: [j1..j6], ... }   # optional
          safe_joints_deg: [j1..j6]            # optional
          pivot_z_clearance_mm: 60.0           # optional

        Fallback: capture.positions
        """
        p = Path(cfg_path)
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        rob = raw.get("robot", {}) or {}

        ip = str(rob.get("ip", "192.168.0.100"))
        joint_vel = float(rob.get("joint_vel", 20.0))
        joint_acc = float(rob.get("joint_acc", 20.0))
        settle_s = float(rob.get("settle_s", 0.5))
        default_z_clearance_mm = float(rob.get("pivot_z_clearance_mm", 60.0))

        pos_raw = rob.get("positions", None)
        if not isinstance(pos_raw, dict) or not pos_raw:
            cap = raw.get("capture", {}) or {}
            pos_raw = cap.get("positions", {})
        positions: Dict[str, Tuple[float, float, float, float, float, float]] = {}
        if isinstance(pos_raw, dict) and pos_raw:
            for k, v in pos_raw.items():
                if not (isinstance(v, list) and len(v) == 6):
                    raise ValueError(f"positions['{k}'] must be list of 6 numbers (j1..j6)")
                positions[str(k)] = tuple(float(x) for x in v)  # type: ignore

        safe_j = rob.get("safe_joints_deg", None)
        safe_joints_deg: Optional[Tuple[float, float, float, float, float, float]] = None
        if isinstance(safe_j, list) and len(safe_j) == 6:
            safe_joints_deg = tuple(float(x) for x in safe_j)  # type: ignore

        return Meca500Controller(
            ip=ip,
            positions_joints_deg=positions,
            joint_vel=joint_vel,
            joint_acc=joint_acc,
            settle_s=settle_s,
            verbose=verbose,
            safe_joints_deg=safe_joints_deg,
            default_z_clearance_mm=default_z_clearance_mm,
        )

    # ---------- internal helpers ----------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _ensure_connected(self) -> None:
        if self.robot is None:
            raise RuntimeError("Robot not connected. Call connect().")

    def _clear_error_if_any(self) -> None:
        if self.robot is None:
            return
        try:
            self.robot.ResetError()
        except Exception:
            pass
        try:
            self.robot.ClearMotion()
        except Exception:
            pass

    def _wait_idle_safe(self, *, context: str = "") -> None:
        """
        WaitIdle, aber fängt typische mecademicpy-Exceptions ab und resettet sauber,
        damit nicht der nächste Befehl sofort wieder scheitert.
        """
        self._ensure_connected()
        try:
            self.robot.WaitIdle()
        except InterruptException as e:
            # "robot is in error"
            self._log(f"[MECA] WaitIdle interrupted{(' ('+context+')') if context else ''}: {e}")
            self._clear_error_if_any()
            raise
        except DisconnectError as e:
            # sollte bei disconnect_on_exception=False seltener passieren, aber sicher ist sicher
            self._log(f"[MECA] DisconnectError{(' ('+context+')') if context else ''}: {e}")
            raise
        except CommunicationError as e:
            self._log(f"[MECA] CommunicationError{(' ('+context+')') if context else ''}: {e}")
            raise

    # ---------- connection ----------

    def connect(self) -> None:
        if self.robot is not None:
            return

        self.robot = mdr.Robot()

        # Verhindert: "Automatically disconnected as a result of exception"
        # (je nach mecademicpy-Version ist das als property vorhanden)
        try:
            self.robot.disconnect_on_exception = False  # type: ignore[attr-defined]
        except Exception:
            pass

        self._log(f"[MECA] Connecting to {self.ip} ...")
        self.robot.Connect(self.ip)
        self._log("[MECA] Connected.")

    def activate_and_home(self) -> None:
        """
        Robust gegen den Fall, dass ActivateAndHome / WaitHomed einen "not activated" Zustand auslöst.
        """
        self._ensure_connected()

        # try to clear state first
        self._clear_error_if_any()

        self._log("[MECA] ActivateAndHome ...")
        try:
            # 1) preferred: one-shot
            self.robot.ActivateAndHome()
            self.robot.WaitHomed()
        except Exception as e:
            # 2) fallback: ActivateRobot + Home (verschiedene API-Versionen)
            self._log(f"[MECA] ActivateAndHome failed -> fallback ActivateRobot+Home. Reason: {e!r}")
            self._clear_error_if_any()

            # Some firmwares have ActivateRobot/Home, some may have ActivateAndHome only.
            # We'll try a few sensible combinations.
            activated = False
            for fn in ("ActivateRobot", "Activate", "ActivateRob"):
                if hasattr(self.robot, fn):
                    try:
                        getattr(self.robot, fn)()
                        activated = True
                        break
                    except Exception:
                        pass

            if not activated:
                # try ActivateAndHome again as last resort
                self.robot.ActivateAndHome()
            else:
                homed = False
                for fn in ("Home", "Homing", "StartHoming"):
                    if hasattr(self.robot, fn):
                        try:
                            getattr(self.robot, fn)()
                            homed = True
                            break
                        except Exception:
                            pass
                if not homed:
                    # best effort
                    self.robot.ActivateAndHome()

            # wait homed if available
            try:
                self.robot.WaitHomed()
            except Exception:
                pass

        # always set speed after homing attempt
        self.set_joint_vel_acc(self.joint_vel, self.joint_acc)

    def set_joint_vel_acc(self, vel: float, acc: float) -> None:
        self._ensure_connected()
        self.joint_vel = float(vel)
        self.joint_acc = float(acc)
        self.robot.SetJointVel(self.joint_vel)
        self.robot.SetJointAcc(self.joint_acc)
        self._log(f"[MECA] JointVel={self.joint_vel} JointAcc={self.joint_acc}")

    # ---------- WRF / TRF ----------

    def set_wrf_brf(self) -> None:
        """Set WRF = BRF."""
        self._ensure_connected()
        self.robot.SetWrf(0, 0, 0, 0, 0, 0)
        self._log("[MECA] WRF=BRF set (0,0,0,0,0,0)")

    def set_trf_mm_deg(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        rx_deg: float,
        ry_deg: float,
        rz_deg: float,
    ) -> None:
        """Set TRF in mm/deg."""
        self._ensure_connected()
        self.robot.SetTrf(float(x_mm), float(y_mm), float(z_mm), float(rx_deg), float(ry_deg), float(rz_deg))
        self._log(
            f"[MECA] TRF set: t=({x_mm:.3f},{y_mm:.3f},{z_mm:.3f})mm r=({rx_deg:.3f},{ry_deg:.3f},{rz_deg:.3f})deg"
        )

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

    def set_wrf_trf_from_config(self, cfg_path: Union[str, Path]) -> None:
        """
        Unterstützt Keys:
          robot.wrf_equals_brf: true/false
          robot.trf_set_mm_deg: [..6..]
          robot.trf_set_during_capture_mm_deg: [..6..]
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
        else:
            self._log("[MECA] No robot.trf_set_* in config -> TRF unchanged")

        wrf = self._try_get_wrf()
        trf_now = self._try_get_trf()
        self._log(f"[MECA] GetWRF available? {'yes' if wrf is not None else 'no'}")
        self._log(f"[MECA] GetTRF available? {'yes' if trf_now is not None else 'no'}")
        if wrf is not None:
            self._log(f"[MECA] WRF(mm/deg) = {wrf}")
        if trf_now is not None:
            self._log(f"[MECA] TRF(mm/deg) = {trf_now}")

    # ---------- motion: joints ----------

    def move_to(self, key: str) -> None:
        """MoveJoints to a named joint pose stored in config."""
        self._ensure_connected()
        if key not in self.positions:
            raise ValueError(f"Unknown pose key '{key}'. Available: {sorted(self.positions.keys())}")

        j = self.positions[key]
        self._log(f"[MECA] MoveJoints {key}: {j}")
        self.robot.MoveJoints(*[float(v) for v in j])
        self._wait_idle_safe(context=f"MoveJoints({key})")
        time.sleep(self.settle_s)

    def move_joints_deg(self, joints: Tuple[float, float, float, float, float, float], *, label: str = "MoveJoints") -> None:
        """Direct MoveJoints with explicit tuple."""
        self._ensure_connected()
        self._log(f"[MECA] {label}: {tuple(float(x) for x in joints)}")
        self.robot.MoveJoints(*[float(v) for v in joints])
        self._wait_idle_safe(context=label)
        time.sleep(self.settle_s)

    def wait_idle(self) -> None:
        self._wait_idle_safe(context="wait_idle()")

    # ---------- motion: cartesian ----------

    def _move_pose_raw(
        self,
        x_mm: float, y_mm: float, z_mm: float,
        rx_deg: float, ry_deg: float, rz_deg: float,
        *,
        linear: bool = False,
        context: str = "MovePose",
    ) -> None:
        """
        Ein einzelner MovePose/MoveLin Schritt.
        """
        self._ensure_connected()
        if linear and hasattr(self.robot, "MoveLin"):
            self.robot.MoveLin(float(x_mm), float(y_mm), float(z_mm), float(rx_deg), float(ry_deg), float(rz_deg))
        else:
            self.robot.MovePose(float(x_mm), float(y_mm), float(z_mm), float(rx_deg), float(ry_deg), float(rz_deg))
        self._wait_idle_safe(context=context)
        time.sleep(self.settle_s)

    def move_pose_mm_deg(
        self,
        x_mm: float, y_mm: float, z_mm: float,
        rx_deg: float, ry_deg: float, rz_deg: float,
        *,
        safe_approach: bool = True,
        z_clearance_mm: Optional[float] = None,
        use_lin_for_vertical: bool = False,
        retry_via_safe_joints: bool = True,
        direct_first: bool = True,
    ) -> None:
        """
        Robustes Anfahren einer kartesischen Pose.

        Strategie:
        - optional: erst direkt versuchen (direct_first)
        - wenn fail: Safe-Approach (hoch -> XY -> runter)
        - wenn immer noch fail und safe_joints_deg vorhanden: via safe joints (MoveJoints) und nochmal Safe-Approach

        Das "Pose out of reach" (1016) ist *nicht* nur "Wegproblem", sondern oft "Pose nicht erreichbar".
        Dann hilft nur: andere Pose/Orientierung/WRF/TRF oder bewusst via safe joints in einen IK-Zweig.
        """
        self._ensure_connected()
        zc = float(self.default_z_clearance_mm if z_clearance_mm is None else z_clearance_mm)

        target = (float(x_mm), float(y_mm), float(z_mm), float(rx_deg), float(ry_deg), float(rz_deg))

        def _attempt_direct() -> None:
            self._log(f"[MECA] MovePose direct -> {target}")
            self._move_pose_raw(*target, linear=False, context="MovePose(direct)")

        def _attempt_safe_approach() -> None:
            cur = self.get_pose_mm_deg()
            # Clearance: über beiden Z liegen (current und target)
            z_up = max(cur.z_mm + zc, float(z_mm) + zc)

            self._log(f"[MECA] MovePose safe-approach (zc={zc:.1f}mm) -> {target}")
            # 1) hoch am aktuellen XY
            self._move_pose_raw(
                cur.x_mm, cur.y_mm, z_up,
                rx_deg, ry_deg, rz_deg,
                linear=use_lin_for_vertical,
                context="MovePose(step1-up)",
            )
            # 2) XY rüber bei hohem Z
            self._move_pose_raw(
                x_mm, y_mm, z_up,
                rx_deg, ry_deg, rz_deg,
                linear=False,
                context="MovePose(step2-xy)",
            )
            # 3) runter
            self._move_pose_raw(
                x_mm, y_mm, z_mm,
                rx_deg, ry_deg, rz_deg,
                linear=use_lin_for_vertical,
                context="MovePose(step3-down)",
            )

        last_err: Optional[BaseException] = None

        # 0) immer erst clearen, falls vorher etwas schiefging
        self._clear_error_if_any()

        # 1) direct first?
        if direct_first:
            try:
                _attempt_direct()
                return
            except Exception as e:
                last_err = e
                self._log(f"[MECA] Direct MovePose failed -> will try fallback. Reason: {e!r}")
                self._clear_error_if_any()

        # 2) safe approach
        if safe_approach:
            try:
                _attempt_safe_approach()
                return
            except Exception as e:
                last_err = e
                self._log(f"[MECA] Safe-approach failed. Reason: {e!r}")
                self._clear_error_if_any()

        # 3) retry via safe joints if configured
        if retry_via_safe_joints and self.safe_joints_deg is not None:
            try:
                self._log(f"[MECA] Retry via safe_joints_deg: {self.safe_joints_deg}")
                self.move_joints_deg(self.safe_joints_deg, label="MoveJoints(safe_joints_deg)")
                # nochmal safe approach (oder direct falls safe_approach aus ist)
                if safe_approach:
                    _attempt_safe_approach()
                else:
                    _attempt_direct()
                return
            except Exception as e:
                last_err = e
                self._log(f"[MECA] Retry via safe_joints failed. Reason: {e!r}")
                self._clear_error_if_any()

        # 4) endgültig scheitern
        msg = (
            "move_pose_mm_deg failed.\n"
            f"  target(mm/deg) = {target}\n"
            f"  safe_approach={safe_approach} z_clearance_mm={zc}\n"
            f"  retry_via_safe_joints={retry_via_safe_joints} safe_joints_deg={self.safe_joints_deg}\n"
            f"  last_error={repr(last_err)}"
        )
        raise RuntimeError(msg) from last_err

    # ---------- pose / joints ----------

    def get_joints_deg(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Best-effort: returns joints (J1..J6) in deg, if API offers it.
        """
        if self.robot is None:
            return None
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
        """
        self._ensure_connected()

        for name in ("GetRtCartPos", "GetRtCartPose"):
            if hasattr(self.robot, name):
                try:
                    v = getattr(self.robot, name)()
                    if isinstance(v, (list, tuple)) and len(v) >= 6:
                        x, y, z, rx, ry, rz = (float(v[i]) for i in range(6))
                        return MecaPose(x, y, z, rx, ry, rz, source=name)
                except Exception:
                    pass

        x, y, z, rx, ry, rz = self.robot.GetPose()
        return MecaPose(float(x), float(y), float(z), float(rx), float(ry), float(rz), source="GetPose")

    def pose_stability_check(self, samples: int = 8, dt_s: float = 0.05) -> Dict[str, float]:
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
            "safe_joints_deg": list(self.safe_joints_deg) if self.safe_joints_deg is not None else None,
            "default_z_clearance_mm": self.default_z_clearance_mm,
        }

    # ---------- shutdown ----------

    def disconnect(self) -> None:
        if self.robot is None:
            return

        # best-effort graceful shutdown
        try:
            try:
                self.robot.WaitIdle()
            except Exception:
                pass

            try:
                self.robot.DeactivateRobot()
                # nicht jede Version hat WaitDeactivated
                if hasattr(self.robot, "WaitDeactivated"):
                    self.robot.WaitDeactivated()
            except Exception:
                pass

            try:
                self.robot.Disconnect()
            except Exception:
                pass
        finally:
            self._log("[MECA] Disconnected.")
            self.robot = None
