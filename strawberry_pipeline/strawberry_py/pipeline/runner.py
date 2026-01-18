# strawberry_py/pipeline/runner.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Set, Tuple

import cv2
import numpy as np

from strawberry_py.config import AppCfg
from strawberry_py.io.plant_views_dataset import PlantViewsDataset
from strawberry_py.pipeline.stages.clustering import StrawberryClusterer
from strawberry_py.pipeline.stages.depth_mask import DepthMasker
from strawberry_py.pipeline.stages.features import FeatureExtractor
from strawberry_py.pipeline.stages.segmentation import YoloV8Segmenter
from strawberry_py.pipeline.stages.selected_overlay import selected_overlay
from strawberry_py.pipeline.stages.transforms import apply_pose, optical_to_trf
from strawberry_py.types import Pose
from strawberry_py.utils.ply import write_ply_xyz
from strawberry_py.utils.vis import save_depth_preview


class PipelineRunner:
    """
    Offline pipeline:
      - iterate plants from PlantViewsDataset
      - per view: YOLO seg -> depth mask -> 3D clouds + features
      - save per-view raw clouds (camera/optical or world)
      - accumulate per-view instances into per-plant clustering
      - export clusters after all views are processed
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

    # ----------------- pose helper -----------------

    def _pose_for_view_world(self, vid: int) -> Pose:
        """
        Pose mapping for a view: WORLD <- CAM (tool/TRF-like frame).
        Prefer cfg.robot.views[vid].pose_world; fallback to cfg.poses.default_pose.
        """
        rv = self.cfg.robot.views.get(int(vid))
        if rv is not None:
            return rv.pose_world
        return self.cfg.poses.default_pose

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

            # per-plant features csv
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

                    # pose for this view (WORLD <- CAM/TRF-like)
                    pose_world_cam = self._pose_for_view_world(vid)

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
                            pose_world_cam=pose_world_cam,
                        )

                    # ---- clustering accumulation (per view) ----
                    if self.cfg.outputs.cluster.enabled:
                        self.clusterer.add_view(
                            plant_id=pid,
                            view_id=vid,
                            clouds_by_instance=fr.clouds_by_instance,
                            features=fr.features,
                            pose_world_cam=pose_world_cam,
                        )

            # ---- export clusters AFTER all views for this plant ----
            if self.cfg.outputs.cluster.enabled:
                self.clusterer.export(plant_dir)

    # ----------------- raw cloud saving -----------------

    def _save_raw_cloud(
        self,
        pid: int,
        vid: int,
        plant_dir: Path,
        points_cam_m: np.ndarray,
        pose_world_cam: Pose,
    ) -> None:
        """
        Save one raw cloud per view to:
          plant_XXX/raw_clouds/cloud_<vid>.ply

        IMPORTANT:
        FeatureExtractor points are in RealSense OPTICAL frame (x right, y down, z forward).
        If exporting in 'world', we do:
          optical -> trf/tool-like (optical_to_trf) -> apply WORLD<-CAM pose.
        """
        key = (pid, vid)
        raw_cfg = self.cfg.outputs.raw_cloud

        out_dir = plant_dir / "raw_clouds"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"cloud_{vid}.ply"

        # dedup / overwrite logic
        if raw_cfg.save_once_per_view and (key in self._saved_raw) and (not raw_cfg.overwrite):
            return
        if out_path.exists() and raw_cfg.save_once_per_view and (not raw_cfg.overwrite):
            self._saved_raw.add(key)
            return

        if points_cam_m is None or getattr(points_cam_m, "size", 0) == 0:
            return

        pts_optical = np.asarray(points_cam_m, dtype=np.float32).reshape((-1, 3))
        export_frame = str(raw_cfg.export_frame).strip().lower()

        if export_frame == "world":
            # optional debug flag comes from cluster config (same symptom source)
            flip_y = bool(getattr(self.cfg.outputs.cluster, "flip_y", False))
            pts_trf = optical_to_trf(pts_optical, flip_y=flip_y)
            pts_out = apply_pose(pts_trf, pose_world_cam)
        else:
            # export in optical/camera frame
            pts_out = pts_optical

        write_ply_xyz(out_path, pts_out, ascii_mode=bool(raw_cfg.ply_ascii))
        self._saved_raw.add(key)

        # optional summary
        self._write_raw_summary(out_dir=out_dir, plant_id=pid, export_frame=export_frame)

    def _write_raw_summary(self, out_dir: Path, plant_id: int, export_frame: str) -> None:
        """
        Writes raw_summary.csv listing which view clouds exist for this plant.
        """
        csv_path = out_dir / "raw_summary.csv"
        rows = []
        for (pid, vid) in sorted(self._saved_raw):
            if pid != plant_id:
                continue
            rows.append([pid, vid, f"cloud_{vid}.ply", export_frame])

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["plant_id", "view_id", "file", "export_frame"])
            w.writerows(rows)
