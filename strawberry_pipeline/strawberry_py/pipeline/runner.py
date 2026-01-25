# strawberry_py/pipeline/runner.py
from __future__ import annotations

import csv
import inspect
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import cv2
import numpy as np

from strawberry_py.config import AppCfg
from strawberry_py.io.plant_views_dataset import PlantViewsDataset
from strawberry_py.io.raw_cloud_writer import RawCloudWriter
from strawberry_py.pipeline.stages.clustering import StrawberryClusterer
from strawberry_py.pipeline.stages.depth_cleanup import DepthCleaner
from strawberry_py.pipeline.stages.depth_mask import DepthMasker
from strawberry_py.pipeline.stages.features import FeatureExtractor
from strawberry_py.pipeline.stages.segmentation import YoloV8Segmenter
from strawberry_py.pipeline.stages.selected_overlay import selected_overlay
from strawberry_py.pipeline.stages.transforms import quaternion_to_rotation_matrix
from strawberry_py.st_types import Pose
from strawberry_py.utils.masks import reduce_to_selected_label
from strawberry_py.utils.vis import save_depth_preview


class PipelineRunner:
    """
    Offline pipeline:
      - per view: YOLO seg -> (selected-only label) -> depth mask -> depth cleanup -> 3D clouds + features
      - optional: save per-view raw clouds in CAM / TRF / WORLD (or all)
      - optional: accumulate instances into per-plant clustering
      - export clusters after all views processed

    Änderung (wichtig):
      - cfg ist frozen => wir mutieren cfg NICHT.
      - Stattdessen kann man dataset_root/out_root als Overrides beim Runner übergeben.
    """

    def __init__(
        self,
        cfg: AppCfg,
        *,
        dataset_root: Optional[Union[str, Path]] = None,
        out_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.cfg = cfg

        # ---------- output root (override möglich) ----------
        self.out_root = Path(out_root) if out_root is not None else Path(cfg.outputs.out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)

        # ---------- dataset root (override möglich) ----------
        self.dataset_root = Path(dataset_root) if dataset_root is not None else Path(cfg.dataset.root)
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

        # ---------- dataset ----------
        self.dataset = PlantViewsDataset(
            root=self.dataset_root,
            plant_glob=cfg.dataset.plant_glob,
            rgb_pattern=cfg.dataset.rgb_pattern,
            depth_pattern=cfg.dataset.depth_pattern,
            view_ids=cfg.dataset.view_ids,
        )

        # ---------- stages ----------
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
        self.depth_cleaner = DepthCleaner(cfg.depth, keep_largest_cc=True)
        self.extractor = FeatureExtractor(cfg.camera, cfg.depth, cfg.features)

        # clustering (per plant)
        self.clusterer = StrawberryClusterer(cfg.outputs.cluster)

        # optional debug knob (safe read)
        self._debug_raw = bool(getattr(cfg.outputs.raw_cloud, "debug", True))

        # fixed CAM->TRF correction from config (may be identity or real mapping)
        self.R_trf_cam, self.t_trf_cam_m = self._load_cam_in_trf_correction()

        # raw cloud writer (IO service)
        self.raw_cloud_writer = RawCloudWriter(
            raw_cloud_cfg=cfg.outputs.raw_cloud,
            R_trf_cam=self.R_trf_cam,
            t_trf_cam_m=self.t_trf_cam_m,
            debug=self._debug_raw,
        )

    # ----------------- config helpers -----------------

    def _load_cam_in_trf_correction(self) -> Tuple[np.ndarray, np.ndarray]:
        robot = getattr(self.cfg, "robot", None)
        if robot is None:
            raise ValueError("cfg.robot missing.")

        R_list = getattr(robot, "cam_axes_correction_R_trf_cam_row_major_3x3", None)
        t_mm = getattr(robot, "camera_in_trf_translation_mm", None)
        if R_list is None or t_mm is None:
            raise ValueError(
                "robot.cam_axes_correction_R_trf_cam_row_major_3x3 and/or "
                "robot.camera_in_trf_translation_mm missing in config."
            )

        R = np.asarray(R_list, dtype=np.float32).reshape((3, 3))
        t = (np.asarray(t_mm, dtype=np.float32).reshape((3,)) / 1000.0)

        ortho_err, det = self._rotation_sanity(R)
        if self._debug_raw:
            print(f"[Runner] R_trf_cam sanity: ortho_err={ortho_err:.3e} det={det:.6f}")
            print(f"[Runner] t_trf_cam_m = {t.tolist()}")

        return R, t

    def _pose_for_view_world_trf(self, vid: int) -> Pose:
        rv = self.cfg.robot.views.get(int(vid))
        if rv is None:
            raise KeyError(f"Missing robot.views[{vid}] in config.")
        return rv.pose_world  # WORLD <- TRF

    @staticmethod
    def _pose_to_Rt(pose_world_trf: Pose) -> Tuple[np.ndarray, np.ndarray]:
        qx, qy, qz, qw = pose_world_trf.q_xyzw
        R = quaternion_to_rotation_matrix(float(qx), float(qy), float(qz), float(qw)).astype(np.float32)
        t = np.asarray(pose_world_trf.t_xyz, dtype=np.float32).reshape((3,))
        return R, t

    @staticmethod
    def _rotation_sanity(R: np.ndarray) -> Tuple[float, float]:
        R = np.asarray(R, dtype=np.float64).reshape((3, 3))
        I = np.eye(3, dtype=np.float64)
        err = float(np.linalg.norm(R.T @ R - I, ord="fro"))
        det = float(np.linalg.det(R))
        return err, det

    # ----------------- main run -----------------

    def run(self) -> None:
        print(f"[Pipeline] dataset_root={self.dataset_root}")
        print(f"[Pipeline] output_root={self.out_root}")
        print(f"[Pipeline] view_ids={self.cfg.dataset.view_ids}")

        for plant in self.dataset.iter_plants():
            pid = int(plant.plant_id)
            plant_dir = self.out_root / f"plant_{pid:03d}"
            plant_dir.mkdir(parents=True, exist_ok=True)

            self.clusterer.reset()

            features_csv = plant_dir / "features.csv"
            with features_csv.open("w", newline="", encoding="utf-8") as fcsv:
                w = csv.writer(fcsv, delimiter=";")
                w.writerow(
                    ["plant_id", "view_id", "instance_id", "num_points", "cx", "cy", "cz", "ex", "ey", "ez", "box_vol_m3"]
                )

                for view in plant.views:
                    vid = int(view.info.view_id)
                    view_dir = plant_dir / f"view_{vid}"
                    view_dir.mkdir(parents=True, exist_ok=True)

                    pose_world_trf = self._pose_for_view_world_trf(vid)
                    _, tw = self._pose_to_Rt(pose_world_trf)
                    if self._debug_raw:
                        print(f"[view {vid}] t_world_trf = {tw.tolist()}")

                    # ---- segmentation ----
                    seg = self.segmenter(view.rgb)
                    label_full = seg.label

                    if self.cfg.outputs.save_overlay and seg.overlay_rgb is not None:
                        cv2.imwrite(str(view_dir / "overlay.png"), cv2.cvtColor(seg.overlay_rgb, cv2.COLOR_RGB2BGR))
                    if self.cfg.outputs.save_label_vis and seg.label_vis is not None:
                        cv2.imwrite(str(view_dir / "label_vis.png"), seg.label_vis)

                    # ---- reduce to selected-only label ----
                    selected_id = int(getattr(self.cfg.selected, "instance_id", 1)) if self.cfg.selected.enabled else None
                    label_sel, sel_stats = reduce_to_selected_label(label_full, selected_id=selected_id, do_morph=True)

                    if self._debug_raw:
                        print(
                            f"[plant {pid:03d} view {vid}] selected_id={int(sel_stats['picked_id'])} "
                            f"area_px={int(sel_stats['area_px'])} fallback={int(sel_stats['fallback_used'])}"
                        )

                    cv2.imwrite(str(view_dir / "selected_mask.png"), (label_sel > 0).astype(np.uint8) * 255)

                    if self.cfg.selected.enabled:
                        sel_img = selected_overlay(
                            view.rgb,
                            label_full,
                            selected_id=self.cfg.selected.instance_id,
                            min_pixels=self.cfg.selected.min_pixels,
                            darken_factor=self.cfg.selected.darken_factor,
                            draw_bbox=self.cfg.selected.draw_bbox,
                        )
                        cv2.imwrite(str(view_dir / "selected_overlay.png"), cv2.cvtColor(sel_img, cv2.COLOR_RGB2BGR))

                    # ---- depth mask (ONLY selected label) ----
                    dm = self.depth_masker(view.depth, label_sel)
                    depth_masked = dm.depth_masked

                    # ---- depth cleanup ----
                    depth_masked, dstat = self.depth_cleaner(depth_masked)
                    if self._debug_raw:
                        print(
                            f"[plant {pid:03d} view {vid}] depth_valid={int(dstat.get('n_valid', 0))} "
                            f"removed={int(dstat.get('removed', 0))} band={dstat.get('band_m', 0.0):.4f}m"
                        )

                    cv2.imwrite(str(view_dir / "depth_masked.png"), depth_masked)
                    if self.cfg.outputs.save_depth_mask_preview:
                        save_depth_preview(depth_masked, view_dir / "depth_masked_preview.png")

                    # ---- features ----
                    fr = self.extractor(depth_masked, label_sel)
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

                    # ---- raw cloud save ----
                    if self.cfg.outputs.raw_cloud.enabled:
                        self.raw_cloud_writer.maybe_write(
                            pid=pid,
                            vid=vid,
                            plant_dir=plant_dir,
                            points_cam_m=fr.all_points,
                            pose_world_trf=pose_world_trf,
                        )

                    # ---- clustering accumulation ----
                    if self.cfg.outputs.cluster.enabled:
                        self._clusterer_add_view_compat(
                            plant_id=pid,
                            view_id=vid,
                            clouds_by_instance=fr.clouds_by_instance,
                            features=fr.features,
                            pose_world_trf=pose_world_trf,
                        )

            if self.cfg.outputs.cluster.enabled:
                self.clusterer.export(plant_dir)

    # ----------------- clusterer call compat -----------------

    def _clusterer_add_view_compat(
        self,
        plant_id: int,
        view_id: int,
        clouds_by_instance: Dict[int, np.ndarray],
        features: Dict[int, object],
        pose_world_trf: Pose,
    ) -> None:
        """
        Avoid breaking if StrawberryClusterer.add_view() signature differs.
        We pass only kwargs that exist in the current implementation.
        """
        fn = self.clusterer.add_view
        sig = inspect.signature(fn)
        params = set(sig.parameters.keys())

        payload = {
            "plant_id": plant_id,
            "view_id": view_id,
            "clouds_by_instance": clouds_by_instance,
            "features": features,
            "pose_world_trf": pose_world_trf,
            "pose_world": pose_world_trf,
            "R_trf_cam": self.R_trf_cam,
            "t_trf_cam_m": self.t_trf_cam_m,
        }
        kwargs = {k: v for k, v in payload.items() if k in params}
        fn(**kwargs)
