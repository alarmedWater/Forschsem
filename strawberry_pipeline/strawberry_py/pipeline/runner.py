from __future__ import annotations

import csv
import inspect
from pathlib import Path
from typing import Dict, Tuple, Optional, Union

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
      - per image sample: 3 views (usually left/mid/right)
      - per view: YOLO seg -> selected-only label -> depth mask -> depth cleanup -> 3D clouds + features
      - optional: save per-view raw clouds in CAM / TRF / WORLD (or all)
      - optional: accumulate instances into clustering
      - export clustering results after all views of one image were processed

    Struktur:
      - sample_name: "1", "1_hok", "1_lok"
      - image_id: 1, 2, 3, ...
    """

    def __init__(
        self,
        cfg: AppCfg,
        *,
        dataset_root: Optional[Union[str, Path]] = None,
        out_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.cfg = cfg

        self.out_root = Path(out_root) if out_root is not None else Path(cfg.outputs.out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)

        self.dataset_root = Path(dataset_root) if dataset_root is not None else Path(cfg.dataset.root)
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

        self.dataset = PlantViewsDataset(
            root=self.dataset_root,
            plant_glob=cfg.dataset.plant_glob,
            rgb_pattern=cfg.dataset.rgb_pattern,
            depth_pattern=cfg.dataset.depth_pattern,
            view_ids=cfg.dataset.view_ids,
        )

        self.segmenter = YoloV8Segmenter.from_cfg(cfg.segmentation)
        self.depth_masker = DepthMasker(cfg.depth, zero_background=True)
        self.depth_cleaner = DepthCleaner(cfg.depth, keep_largest_cc=True)
        self.extractor = FeatureExtractor(cfg.camera, cfg.depth, cfg.features)

        self.clusterer = StrawberryClusterer(cfg.outputs.cluster)

        self._debug_raw = bool(getattr(cfg.outputs.raw_cloud, "debug", True))

        self.R_trf_cam, self.t_trf_cam_m = self._load_cam_in_trf_correction()

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
        t = np.asarray(t_mm, dtype=np.float32).reshape((3,)) / 1000.0

        ortho_err, det = self._rotation_sanity(R)
        if self._debug_raw:
            print(f"[Runner] R_trf_cam sanity: ortho_err={ortho_err:.3e} det={det:.6f}")
            print(f"[Runner] t_trf_cam_m = {t.tolist()}")

        return R, t

    def _pose_for_view_world_trf(self, vid: int) -> Pose:
        rv = self.cfg.robot.views.get(int(vid))
        if rv is None:
            raise KeyError(f"Missing robot.views[{vid}] in config.")
        return rv.pose_world

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

    @staticmethod
    def _image_dir_name(image_id: int) -> str:
        return f"image_{int(image_id):03d}"

    @staticmethod
    def _resolve_image_id(plant: object) -> int:
        """
        Backward-compatible:
        - preferred: plant.image_id
        - fallback: plant.capture_id
        """
        if hasattr(plant, "image_id"):
            return int(getattr(plant, "image_id"))
        if hasattr(plant, "capture_id"):
            return int(getattr(plant, "capture_id"))
        raise AttributeError("PlantSample has neither 'image_id' nor 'capture_id'.")

    # ----------------- main run -----------------

    def run(self) -> None:
        print(f"[Pipeline] dataset_root={self.dataset_root}")
        print(f"[Pipeline] output_root={self.out_root}")
        print(f"[Pipeline] view_ids={self.cfg.dataset.view_ids}")
        if self.cfg.selected.enabled:
            selected_id_cfg: Optional[int] = int(getattr(self.cfg.selected, "instance_id", 1))
            print(f"[Pipeline] selected_mode=configured_id:{selected_id_cfg}")
        else:
            # This pipeline remains single-mask: disabled selection means auto-pick largest instance.
            selected_id_cfg = None
            print("[Pipeline] selected_mode=auto_largest_instance_per_view")

        for plant in self.dataset.iter_plants():
            pid = int(plant.plant_id)
            sample_name = str(plant.sample_name)
            variant = str(plant.variant)
            image_id = self._resolve_image_id(plant)

            sample_root_dir = self.out_root / sample_name
            image_dir = sample_root_dir / self._image_dir_name(image_id)
            image_dir.mkdir(parents=True, exist_ok=True)

            if self._debug_raw:
                print(
                    f"[Pipeline] sample={sample_name} variant={variant} "
                    f"plant_id={pid} image_id={image_id}"
                )

            self.clusterer.reset()

            features_csv = image_dir / "features.csv"
            with features_csv.open("w", newline="", encoding="utf-8") as fcsv:
                writer = csv.writer(fcsv, delimiter=";")
                writer.writerow(
                    [
                        "plant_id",
                        "sample_name",
                        "variant",
                        "image_id",
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
                    view_dir = image_dir / f"view_{vid}"
                    view_dir.mkdir(parents=True, exist_ok=True)

                    pose_world_trf = self._pose_for_view_world_trf(vid)
                    _, tw = self._pose_to_Rt(pose_world_trf)

                    if self._debug_raw:
                        print(
                            f"[sample {sample_name} image {image_id} view {vid}] "
                            f"t_world_trf = {tw.tolist()}"
                        )

                    # ---- segmentation ----
                    seg = self.segmenter(view.rgb)
                    label_full = seg.label

                    if self.cfg.outputs.save_overlay and seg.overlay_rgb is not None:
                        cv2.imwrite(
                            str(view_dir / "overlay.png"),
                            cv2.cvtColor(seg.overlay_rgb, cv2.COLOR_RGB2BGR),
                        )

                    if self.cfg.outputs.save_label_vis and seg.label_vis is not None:
                        cv2.imwrite(str(view_dir / "label_vis.png"), seg.label_vis)

                    # ---- reduce to selected-only label ----
                    label_sel, sel_stats = reduce_to_selected_label(
                        label_full,
                        selected_id=selected_id_cfg,
                        do_morph=True,
                    )

                    if self._debug_raw:
                        print(
                            f"[sample {sample_name} image {image_id} view {vid}] "
                            f"selected_id={int(sel_stats['picked_id'])} "
                            f"area_px={int(sel_stats['area_px'])} "
                            f"fallback={int(sel_stats['fallback_used'])}"
                        )

                    cv2.imwrite(
                        str(view_dir / "selected_mask.png"),
                        (label_sel > 0).astype(np.uint8) * 255,
                    )

                    if self.cfg.selected.enabled:
                        sel_img = selected_overlay(
                            view.rgb,
                            label_full,
                            selected_id=self.cfg.selected.instance_id,
                            min_pixels=self.cfg.selected.min_pixels,
                            darken_factor=self.cfg.selected.darken_factor,
                            draw_bbox=self.cfg.selected.draw_bbox,
                        )
                        cv2.imwrite(
                            str(view_dir / "selected_overlay.png"),
                            cv2.cvtColor(sel_img, cv2.COLOR_RGB2BGR),
                        )

                    # ---- depth mask ----
                    dm = self.depth_masker(view.depth, label_sel)
                    depth_masked = dm.depth_masked

                    # ---- depth cleanup ----
                    depth_masked, dstat = self.depth_cleaner(depth_masked)
                    if self._debug_raw:
                        print(
                            f"[sample {sample_name} image {image_id} view {vid}] "
                            f"depth_valid={int(dstat.get('n_valid', 0))} "
                            f"removed={int(dstat.get('removed', 0))} "
                            f"band={dstat.get('band_m', 0.0):.4f}m"
                        )

                    cv2.imwrite(str(view_dir / "depth_masked.png"), depth_masked)

                    if self.cfg.outputs.save_depth_mask_preview:
                        save_depth_preview(
                            depth_masked,
                            view_dir / "depth_masked_preview.png",
                        )

                    # ---- features ----
                    fr = self.extractor(depth_masked, label_sel)

                    if self.cfg.features.log_features:
                        print(
                            f"[sample {sample_name} image {image_id} view {vid}] "
                            f"instances={sorted(fr.features.keys())}"
                        )

                    for inst_id, feat in fr.features.items():
                        writer.writerow(
                            [
                                pid,
                                sample_name,
                                variant,
                                image_id,
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
                        self._raw_cloud_write_compat(
                            plant_id=pid,
                            sample_name=sample_name,
                            variant=variant,
                            image_id=image_id,
                            view_id=vid,
                            image_dir=image_dir,
                            points_cam_m=fr.all_points,
                            pose_world_trf=pose_world_trf,
                        )

                    # ---- clustering accumulation ----
                    if self.cfg.outputs.cluster.enabled:
                        self._clusterer_add_view_compat(
                            plant_id=pid,
                            sample_name=sample_name,
                            variant=variant,
                            image_id=image_id,
                            view_id=vid,
                            clouds_by_instance=fr.clouds_by_instance,
                            features=fr.features,
                            pose_world_trf=pose_world_trf,
                        )

            if self.cfg.outputs.cluster.enabled:
                self.clusterer.export(image_dir)

    # ----------------- raw cloud compat -----------------

    def _raw_cloud_write_compat(
        self,
        plant_id: int,
        sample_name: str,
        variant: str,
        image_id: int,
        view_id: int,
        image_dir: Path,
        points_cam_m: np.ndarray | None,
        pose_world_trf: Pose,
    ) -> None:
        fn = self.raw_cloud_writer.maybe_write
        sig = inspect.signature(fn)
        params = set(sig.parameters.keys())

        payload = {
            "pid": plant_id,
            "plant_id": plant_id,
            "sample_name": sample_name,
            "variant": variant,
            "image_id": image_id,
            "capture_id": image_id,   # backward compat alias
            "vid": view_id,
            "view_id": view_id,
            "plant_dir": image_dir,
            "image_dir": image_dir,
            "capture_dir": image_dir,  # backward compat alias
            "points_cam_m": points_cam_m,
            "pose_world_trf": pose_world_trf,
        }
        kwargs = {k: v for k, v in payload.items() if k in params}
        fn(**kwargs)

    # ----------------- clusterer compat -----------------

    def _clusterer_add_view_compat(
        self,
        plant_id: int,
        sample_name: str,
        variant: str,
        image_id: int,
        view_id: int,
        clouds_by_instance: Dict[int, np.ndarray],
        features: Dict[int, object],
        pose_world_trf: Pose,
    ) -> None:
        fn = self.clusterer.add_view
        sig = inspect.signature(fn)
        params = set(sig.parameters.keys())

        payload = {
            "plant_id": plant_id,
            "sample_name": sample_name,
            "variant": variant,
            "image_id": image_id,
            "capture_id": image_id,   # backward compat alias
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
