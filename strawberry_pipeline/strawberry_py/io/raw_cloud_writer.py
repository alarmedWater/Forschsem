# strawberry_py/io/raw_cloud_writer.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np

from strawberry_py.pipeline.stages.transforms import apply_pose
from strawberry_py.st_types import Pose
from strawberry_py.utils.ply import write_ply_xyz


class RawCloudWriter:
    """
    IO service for exporting raw point clouds and maintaining a summary CSV.

    Responsibilities:
      - dedup per (plant_id, view_id)
      - decide which frames to export (camera/trf/world/all)
      - write PLY files
      - write/update raw_summary.csv
    """

    def __init__(
        self,
        raw_cloud_cfg: object,
        R_trf_cam: np.ndarray,
        t_trf_cam_m: np.ndarray,
        debug: bool = False,
    ) -> None:
        self.cfg = raw_cloud_cfg
        self.debug = bool(debug)

        self.enabled = bool(getattr(raw_cloud_cfg, "enabled", False))
        self.export_frame = str(getattr(raw_cloud_cfg, "export_frame", "all")).strip().lower()
        self.save_once_per_view = bool(getattr(raw_cloud_cfg, "save_once_per_view", True))
        self.overwrite = bool(getattr(raw_cloud_cfg, "overwrite", False))
        self.ply_ascii = bool(getattr(raw_cloud_cfg, "ply_ascii", True))

        self.R_trf_cam = np.asarray(R_trf_cam, dtype=np.float32).reshape((3, 3))
        self.t_trf_cam_m = np.asarray(t_trf_cam_m, dtype=np.float32).reshape((3,))

        self._saved: Set[Tuple[int, int]] = set()

    def maybe_write(
        self,
        pid: int,
        vid: int,
        plant_dir: Path,
        points_cam_m: Optional[np.ndarray],
        pose_world_trf: Pose,
    ) -> None:
        if not self.enabled:
            return

        key = (int(pid), int(vid))
        if self.save_once_per_view and (key in self._saved) and (not self.overwrite):
            return

        if points_cam_m is None or getattr(points_cam_m, "size", 0) == 0:
            return

        pts_cam = np.asarray(points_cam_m, dtype=np.float32).reshape((-1, 3))
        if pts_cam.size == 0:
            return

        out_dir = plant_dir / "raw_clouds"
        out_dir.mkdir(parents=True, exist_ok=True)

        export_frame = self._normalize_export_frame(self.export_frame)

        pts_trf = self._cam_to_trf(pts_cam)
        pts_world = apply_pose(pts_trf, pose_world_trf)  # WORLD <- TRF

        if export_frame in ("camera", "all"):
            write_ply_xyz(out_dir / f"cloud_{vid}_cam.ply", pts_cam, ascii_mode=self.ply_ascii)
        if export_frame in ("trf", "all"):
            write_ply_xyz(out_dir / f"cloud_{vid}_trf.ply", pts_trf, ascii_mode=self.ply_ascii)
        if export_frame in ("world", "all"):
            write_ply_xyz(out_dir / f"cloud_{vid}_world.ply", pts_world, ascii_mode=self.ply_ascii)

        self._saved.add(key)
        self._write_raw_summary(out_dir=out_dir, plant_id=int(pid))

    def _cam_to_trf(self, pts_cam: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_cam, dtype=np.float32).reshape((-1, 3))
        return (self.R_trf_cam @ pts.T).T + self.t_trf_cam_m

    @staticmethod
    def _normalize_export_frame(v: str) -> str:
        vv = str(v).strip().lower()
        if vv == "cam":
            vv = "camera"
        if vv not in ("camera", "trf", "world", "all"):
            raise ValueError("outputs.raw_cloud.export_frame must be one of: camera | trf | world | all")
        return vv

    @staticmethod
    def _write_raw_summary(out_dir: Path, plant_id: int) -> None:
        csv_path = out_dir / "raw_summary.csv"
        files = sorted(out_dir.glob("cloud_*_*.ply"))
        out_rows: List[List[object]] = []

        for f in files:
            stem = f.stem
            parts = stem.split("_")
            if len(parts) < 3:
                continue
            try:
                vid = int(parts[1])
            except Exception:
                continue
            frame = parts[2]
            out_rows.append([int(plant_id), int(vid), str(frame), f.name])

        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp, delimiter=";")
            w.writerow(["plant_id", "view_id", "frame", "file"])
            w.writerows(out_rows)
