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
      - dedup per (sample_name, image_id, view_id)
      - decide which frames to export (camera/trf/world/all)
      - write PLY files
      - write/update raw_summary.csv

    Backward-compatible:
      - accepts old arguments like pid / vid / plant_dir / capture_id
      - accepts new arguments like plant_id / view_id / image_id / image_dir
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

        self._saved: Set[Tuple[str, int, int]] = set()

    def maybe_write(
        self,
        pid: Optional[int] = None,
        vid: Optional[int] = None,
        plant_dir: Optional[Path] = None,
        points_cam_m: Optional[np.ndarray] = None,
        pose_world_trf: Optional[Pose] = None,
        *,
        plant_id: Optional[int] = None,
        sample_name: Optional[str] = None,
        variant: Optional[str] = None,
        image_id: Optional[int] = None,
        capture_id: Optional[int] = None,
        view_id: Optional[int] = None,
        image_dir: Optional[Path] = None,
        capture_dir: Optional[Path] = None,
    ) -> None:
        if not self.enabled:
            return

        resolved_plant_id = int(plant_id if plant_id is not None else pid) if (plant_id is not None or pid is not None) else None
        if resolved_plant_id is None:
            raise ValueError("maybe_write: plant_id/pid must be provided.")

        resolved_view_id = self._resolve_view_id(vid=vid, view_id=view_id)
        resolved_image_id = self._resolve_image_id(image_id=image_id, capture_id=capture_id)
        resolved_sample_name = self._resolve_sample_name(sample_name=sample_name, plant_id=resolved_plant_id)
        resolved_variant = self._resolve_variant(variant=variant, sample_name=resolved_sample_name)
        resolved_out_base_dir = self._resolve_out_base_dir(
            plant_dir=plant_dir,
            image_dir=image_dir,
            capture_dir=capture_dir,
        )

        if pose_world_trf is None:
            raise ValueError("maybe_write: pose_world_trf must be provided.")

        if points_cam_m is None or getattr(points_cam_m, "size", 0) == 0:
            return

        pts_cam = np.asarray(points_cam_m, dtype=np.float32).reshape((-1, 3))
        if pts_cam.size == 0:
            return

        dedup_key = (resolved_sample_name, resolved_image_id, resolved_view_id)
        if self.save_once_per_view and (dedup_key in self._saved) and (not self.overwrite):
            return

        out_dir = resolved_out_base_dir / "raw_clouds"
        out_dir.mkdir(parents=True, exist_ok=True)

        export_frame = self._normalize_export_frame(self.export_frame)

        pts_trf = self._cam_to_trf(pts_cam)
        pts_world = apply_pose(pts_trf, pose_world_trf)

        if export_frame in ("camera", "all"):
            write_ply_xyz(out_dir / f"cloud_{resolved_view_id}_cam.ply", pts_cam, ascii_mode=self.ply_ascii)
        if export_frame in ("trf", "all"):
            write_ply_xyz(out_dir / f"cloud_{resolved_view_id}_trf.ply", pts_trf, ascii_mode=self.ply_ascii)
        if export_frame in ("world", "all"):
            write_ply_xyz(out_dir / f"cloud_{resolved_view_id}_world.ply", pts_world, ascii_mode=self.ply_ascii)

        self._saved.add(dedup_key)
        self._write_raw_summary(
            out_dir=out_dir,
            plant_id=resolved_plant_id,
            sample_name=resolved_sample_name,
            variant=resolved_variant,
            image_id=resolved_image_id,
        )

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
    def _resolve_view_id(vid: Optional[int], view_id: Optional[int]) -> int:
        if view_id is not None:
            return int(view_id)
        if vid is not None:
            return int(vid)
        raise ValueError("maybe_write: view_id/vid must be provided.")

    @staticmethod
    def _resolve_image_id(image_id: Optional[int], capture_id: Optional[int]) -> int:
        if image_id is not None:
            return int(image_id)
        if capture_id is not None:
            return int(capture_id)
        raise ValueError("maybe_write: image_id/capture_id must be provided.")

    @staticmethod
    def _resolve_sample_name(sample_name: Optional[str], plant_id: int) -> str:
        if sample_name is not None and str(sample_name).strip():
            return str(sample_name).strip()
        return str(int(plant_id))

    @staticmethod
    def _resolve_variant(variant: Optional[str], sample_name: str) -> str:
        if variant is not None and str(variant).strip():
            return str(variant).strip()

        if sample_name.endswith("_hok"):
            return "hok"
        if sample_name.endswith("_lok"):
            return "lok"
        return "base"

    @staticmethod
    def _resolve_out_base_dir(
        plant_dir: Optional[Path],
        image_dir: Optional[Path],
        capture_dir: Optional[Path],
    ) -> Path:
        if image_dir is not None:
            return Path(image_dir)
        if capture_dir is not None:
            return Path(capture_dir)
        if plant_dir is not None:
            return Path(plant_dir)
        raise ValueError("maybe_write: one of plant_dir/image_dir/capture_dir must be provided.")

    @staticmethod
    def _write_raw_summary(
        out_dir: Path,
        plant_id: int,
        sample_name: str,
        variant: str,
        image_id: int,
    ) -> None:
        csv_path = out_dir / "raw_summary.csv"
        files = sorted(out_dir.glob("cloud_*_*.ply"))
        out_rows: List[List[object]] = []

        for f in files:
            stem = f.stem
            parts = stem.split("_")
            if len(parts) < 3:
                continue

            try:
                view_id = int(parts[1])
            except Exception:
                continue

            frame = parts[2]
            out_rows.append(
                [
                    int(plant_id),
                    str(sample_name),
                    str(variant),
                    int(image_id),
                    int(view_id),
                    str(frame),
                    f.name,
                ]
            )

        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp, delimiter=";")
            writer.writerow(["plant_id", "sample_name", "variant", "image_id", "view_id", "frame", "file"])
            writer.writerows(out_rows)