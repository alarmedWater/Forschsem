# strawberry_py/pipeline/runner.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Set, Tuple

import cv2
import numpy as np

from strawberry_py.config import AppCfg
from strawberry_py.io.plant_views_dataset import PlantViewsDataset
from strawberry_py.pipeline.stages.clustering import StrawberryClusterer
from strawberry_py.pipeline.stages.depth_mask import DepthMasker
from strawberry_py.pipeline.stages.features import FeatureExtractor
from strawberry_py.pipeline.stages.segmentation import YoloV8Segmenter
from strawberry_py.pipeline.stages.selected_overlay import selected_overlay
from strawberry_py.pipeline.stages.transforms import quaternion_to_rotation_matrix
from strawberry_py.st_types import Pose
from strawberry_py.utils.ply import write_ply_xyz
from strawberry_py.utils.vis import save_depth_preview


class PipelineRunner:
    """
    Offline pipeline:
      - per view: YOLO seg -> depth mask -> 3D clouds + features
      - save per-view raw clouds in CAM / TRF / WORLD (or all)
      - accumulate instances into per-plant clustering
      - export clusters after all views processed

    Frames / transforms:
      FeatureExtractor outputs points in CAM (RealSense optical) frame.
      Robot pose provides WORLD <- TRF.
      We additionally need fixed TRF <- CAM (R_trf_cam, t_trf_cam_m).
      Then:
         p_trf   = R_trf_cam @ p_cam + t_trf_cam
         p_world = R_world_trf @ p_trf + t_world_trf
    """

    def __init__(self, cfg: AppCfg) -> None:
        self.cfg = cfg

        # output root
        self.out_root = Path(cfg.outputs.out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)

        # dataset
        self.dataset = PlantViewsDataset(
            root=cfg.dataset.root,
            plant_glob=cfg.dataset.plant_glob,
            rgb_pattern=cfg.dataset.rgb_pattern,
            depth_pattern=cfg.dataset.depth_pattern,
            view_ids=cfg.dataset.view_ids,
        )

        # stages
        self.segmenter = YoloV8Segmenter(
            model_path=str(cfg.segmentation.model_path),
            device=str(cfg.segmentation.device),
            imgsz=int(cfg.segmentation.imgsz),
            conf=float(cfg.segmentation.conf),
            iou=float(cfg.segmentation.iou),
            max_det=int(cfg.segmentation.max_det),
            min_mask_area_px=int(cfg.segmentation.min_mask_area_px),
            classes=list(cfg.segmentation.classes),
        )
        self.depth_masker = DepthMasker(cfg.depth, zero_background=True)
        self.extractor = FeatureExtractor(cfg.camera, cfg.depth, cfg.features)

        # clustering (per plant)
        self.clusterer = StrawberryClusterer(cfg.outputs.cluster)

        # dedup raw saving (plant_id, view_id)
        self._saved_raw: Set[Tuple[int, int]] = set()

        # optional debug knob (read if present; default True)
        # NOTE: cfg.outputs.raw_cloud has no "debug" in your dataclass,
        # so we read it safely with getattr.
        self._debug_raw = bool(getattr(cfg.outputs.raw_cloud, "debug", True))

        # fixed CAM->TRF correction
        self.R_trf_cam: np.ndarray
        self.t_trf_cam_m: np.ndarray
        self.R_trf_cam, self.t_trf_cam_m = self._load_cam_in_trf_correction()

    # ----------------- config helpers -----------------

    def _load_cam_in_trf_correction(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads from cfg.robot:
          cam_axes_correction_R_trf_cam_row_major_3x3: len=9
          camera_in_trf_translation_mm: len=3
        Returns:
          R_trf_cam (3,3) float32
          t_trf_cam_m (3,) float32  in meters
        """
        robot = getattr(self.cfg, "robot", None)
        if robot is None:
            raise ValueError(
                "cfg.robot missing. Need robot.cam_axes_correction_R_trf_cam_row_major_3x3 "
                "and robot.camera_in_trf_translation_mm."
            )

        R_list = getattr(robot, "cam_axes_correction_R_trf_cam_row_major_3x3", None)
        t_mm = getattr(robot, "camera_in_trf_translation_mm", None)

        if R_list is None or t_mm is None:
            raise ValueError(
                "robot.cam_axes_correction_R_trf_cam_row_major_3x3 and/or "
                "robot.camera_in_trf_translation_mm missing in config."
            )

        R = np.asarray(R_list, dtype=np.float32).reshape((3, 3))
        t = np.asarray(t_mm, dtype=np.float32).reshape((3,)) / 1000.0

        if self._debug_raw:
            ortho_err, det = self._rotation_sanity(R)
            print(f"[Runner] R_trf_cam sanity: ortho_err={ortho_err:.3e} det={det:.6f}")
            print(f"[Runner] t_trf_cam_m = {t.tolist()}")

        return R, t

    def _pose_for_view_world_trf(self, vid: int) -> Pose:
        """
        Pose for this view: WORLD <- TRF.
        STRICT: No fallback allowed. Missing view pose must fail fast,
        otherwise clustering silently breaks.
        """
        rv = self.cfg.robot.views.get(int(vid))
        if rv is None:
            raise KeyError(
                f"Missing robot.views[{vid}] in config. "
                f"dataset.view_ids={self.cfg.dataset.view_ids}. "
                "No fallback pose allowed."
            )
        return rv.pose_world


    @staticmethod
    def _pose_to_Rt(pose_world_trf: Pose) -> Tuple[np.ndarray, np.ndarray]:
        qx, qy, qz, qw = pose_world_trf.q_xyzw
        R = quaternion_to_rotation_matrix(float(qx), float(qy), float(qz), float(qw)).astype(np.float32)
        t = np.asarray(pose_world_trf.t_xyz, dtype=np.float32).reshape((3,))
        return R, t

    @staticmethod
    def _rotation_sanity(R: np.ndarray) -> Tuple[float, float]:
        """
        Returns (orthonormal_error, det).
        orthonormal_error ~ ||R^T R - I||_F
        """
        R = np.asarray(R, dtype=np.float64).reshape((3, 3))
        I = np.eye(3, dtype=np.float64)
        err = float(np.linalg.norm(R.T @ R - I, ord="fro"))
        det = float(np.linalg.det(R))
        return err, det

    # ----------------- transform helpers -----------------

    def _cam_to_trf(self, pts_cam: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_cam, dtype=np.float32).reshape((-1, 3))
        if pts.size == 0:
            return pts
        return (self.R_trf_cam @ pts.T).T + self.t_trf_cam_m

    def _trf_to_world(self, pts_trf: np.ndarray, pose_world_trf: Pose) -> np.ndarray:
        pts = np.asarray(pts_trf, dtype=np.float32).reshape((-1, 3))
        if pts.size == 0:
            return pts
        Rw, tw = self._pose_to_Rt(pose_world_trf)
        return (Rw @ pts.T).T + tw

    def _forward_result_dot(self, pts_world: np.ndarray, pose_world_trf: Pose) -> float:
        """
        Sanity check:
          forward_world • (centroid_world - cam_origin_world) should be > 0.

        Uses optical forward axis (0,0,1) in CAM, mapped CAM->TRF->WORLD.
        """
        pts = np.asarray(pts_world, dtype=np.float32).reshape((-1, 3))
        if pts.size == 0:
            return float("nan")

        Rw, tw = self._pose_to_Rt(pose_world_trf)

        # CAM origin in TRF is t_trf_cam_m (because p_trf = R*p_cam + t)
        cam_origin_world = (Rw @ self.t_trf_cam_m) + tw

        # CAM forward axis (optical z) mapped to TRF then WORLD
        f_cam = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        f_trf = self.R_trf_cam @ f_cam
        f_world = Rw @ f_trf

        centroid = pts.mean(axis=0)
        v = centroid - cam_origin_world
        return float(np.dot(f_world, v))

    # ----------------- main run -----------------

    def run(self) -> None:
        print(f"[Pipeline] dataset_root={self.cfg.dataset.root}")
        print(f"[Pipeline] output_root={self.out_root}")
        print(f"[Pipeline] view_ids={self.cfg.dataset.view_ids}")

        for plant in self.dataset.iter_plants():
            pid = int(plant.plant_id)
            plant_dir = self.out_root / f"plant_{pid:03d}"
            plant_dir.mkdir(parents=True, exist_ok=True)

            # reset per-plant clustering state
            self.clusterer.reset()

            features_csv = plant_dir / "features.csv"
            with features_csv.open("w", newline="", encoding="utf-8") as fcsv:
                w = csv.writer(fcsv, delimiter=";")
                w.writerow(
                    [
                        "plant_id",
                        "view_id",
                        "instance_id",
                        "num_points",
                        "cx",
                        "cy",
                        "cz",
                        "ex",
                        "ey",
                        "ez",
                        "box_vol_m3",
                    ]
                )

                for view in plant.views:
                    vid = int(view.info.view_id)
                    view_dir = plant_dir / f"view_{vid}"
                    view_dir.mkdir(parents=True, exist_ok=True)

                    pose_world_trf = self._pose_for_view_world_trf(vid)
                    Rw, tw = self._pose_to_Rt(pose_world_trf)
                    print(f"[view {vid}] t_world_trf = {tw.tolist()}")

                    # ---- segmentation ----
                    seg = self.segmenter(view.rgb)
                    label = seg.label

                    if self.cfg.outputs.save_overlay and seg.overlay_rgb is not None:
                        cv2.imwrite(
                            str(view_dir / "overlay.png"),
                            cv2.cvtColor(seg.overlay_rgb, cv2.COLOR_RGB2BGR),
                        )
                    if self.cfg.outputs.save_label_vis and seg.label_vis is not None:
                        cv2.imwrite(str(view_dir / "label_vis.png"), seg.label_vis)

                    # ---- depth mask ----
                    dm = self.depth_masker(view.depth, label)
                    depth_masked = dm.depth_masked
                    cv2.imwrite(str(view_dir / "depth_masked.png"), depth_masked)
                    if self.cfg.outputs.save_depth_mask_preview:
                        save_depth_preview(depth_masked, view_dir / "depth_masked_preview.png")

                    # ---- features (clouds + per-instance features) ----
                    fr = self.extractor(depth_masked, label)
                    if self.cfg.features.log_features:
                        print(f"[plant {pid:03d} view {vid}] instances={sorted(fr.features.keys())}")

                    for inst_id, feat in fr.features.items():
                        w.writerow(
                            [
                                pid,
                                vid,
                                inst_id,
                                feat.num_points,
                                feat.centroid_m[0],
                                feat.centroid_m[1],
                                feat.centroid_m[2],
                                feat.extent_m[0],
                                feat.extent_m[1],
                                feat.extent_m[2],
                                feat.box_volume_m3,
                            ]
                        )

                    # ---- selected overlay ----
                    if self.cfg.selected.enabled:
                        sel_img = selected_overlay(
                            view.rgb,
                            label,
                            selected_id=self.cfg.selected.instance_id,
                            min_pixels=self.cfg.selected.min_pixels,
                            darken_factor=self.cfg.selected.darken_factor,
                            draw_bbox=self.cfg.selected.draw_bbox,
                        )
                        cv2.imwrite(
                            str(view_dir / "selected_overlay.png"),
                            cv2.cvtColor(sel_img, cv2.COLOR_RGB2BGR),
                        )

                    # ---- raw cloud save (per view) ----
                    if self.cfg.outputs.raw_cloud.enabled:
                        self._save_raw_cloud(
                            pid=pid,
                            vid=vid,
                            plant_dir=plant_dir,
                            points_cam_m=fr.all_points,
                            pose_world_trf=pose_world_trf,
                        )

                    # ---- clustering accumulation (per view) ----
                    if self.cfg.outputs.cluster.enabled:
                        self.clusterer.add_view(
                            plant_id=pid,
                            view_id=vid,
                            clouds_by_instance=fr.clouds_by_instance,
                            features=fr.features,
                            pose_world_trf=pose_world_trf,
                            R_trf_cam=self.R_trf_cam,
                            t_trf_cam_m=self.t_trf_cam_m,
                        )

            if self.cfg.outputs.cluster.enabled:
                self.clusterer.export(plant_dir)

    # ----------------- raw cloud saving -----------------

    def _save_raw_cloud(
        self,
        pid: int,
        vid: int,
        plant_dir: Path,
        points_cam_m: np.ndarray,
        pose_world_trf: Pose,
    ) -> None:
        """
        Save raw cloud per view to plant_XXX/raw_clouds/

        export_frame:
          - "camera" / "cam":   saves cloud_{vid}_cam.ply
          - "trf":              saves cloud_{vid}_trf.ply
          - "world":            saves cloud_{vid}_world.ply
          - "all":              saves all three
        """
        key = (pid, vid)
        raw_cfg = self.cfg.outputs.raw_cloud

        out_dir = plant_dir / "raw_clouds"
        out_dir.mkdir(parents=True, exist_ok=True)

        if raw_cfg.save_once_per_view and (key in self._saved_raw) and (not raw_cfg.overwrite):
            return

        if points_cam_m is None or getattr(points_cam_m, "size", 0) == 0:
            return

        pts_cam = np.asarray(points_cam_m, dtype=np.float32).reshape((-1, 3))

        export_frame = str(raw_cfg.export_frame).strip().lower()
        if export_frame in ("cam",):
            export_frame = "camera"

        if export_frame not in ("camera", "trf", "world", "all"):
            raise ValueError("outputs.raw_cloud.export_frame must be one of: camera | trf | world | all")

        pts_trf = self._cam_to_trf(pts_cam)
        pts_world = self._trf_to_world(pts_trf, pose_world_trf)

        if self._debug_raw and export_frame in ("world", "all"):
            dot = self._forward_result_dot(pts_world, pose_world_trf)
            print(f"[plant {pid:03d} view {vid}] forward_dot={dot:.4f} (should be > 0)")

        # write files
        if export_frame in ("camera", "all"):
            write_ply_xyz(out_dir / f"cloud_{vid}_cam.ply", pts_cam, ascii_mode=bool(raw_cfg.ply_ascii))

        if export_frame in ("trf", "all"):
            write_ply_xyz(out_dir / f"cloud_{vid}_trf.ply", pts_trf, ascii_mode=bool(raw_cfg.ply_ascii))

        if export_frame in ("world", "all"):
            write_ply_xyz(out_dir / f"cloud_{vid}_world.ply", pts_world, ascii_mode=bool(raw_cfg.ply_ascii))

        self._saved_raw.add(key)
        self._write_raw_summary(out_dir=out_dir, plant_id=pid)

    def _write_raw_summary(self, out_dir: Path, plant_id: int) -> None:
        """
        Writes raw_clouds/raw_summary.csv by scanning the folder.
        Rows:
          plant_id;view_id;frame;file
        """
        csv_path = out_dir / "raw_summary.csv"

        files = sorted(out_dir.glob("cloud_*_*.ply"))
        out_rows: List[List[object]] = []

        for f in files:
            # expected: cloud_{vid}_{frame}.ply
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

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["plant_id", "view_id", "frame", "file"])
            w.writerows(out_rows)
