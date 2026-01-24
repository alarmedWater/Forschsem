from __future__ import annotations

import numpy as np

from strawberry_py.config import DepthCfg, FeaturesCfg
from strawberry_py.st_types import (
    CameraIntrinsics,
    DepthU16,
    LabelImage,
    FeaturesResult,
    InstanceFeatures,
    PointCloud,
    assert_depth_u16,
    assert_label_u16,
)
from strawberry_py.pipeline.stages.transforms import depth_u16_to_meters


class FeatureExtractor:
    """
    Builds per-instance point clouds + simple geometric features from:
      - depth_masked (u16)
      - label image (u16 instance ids)

    Coordinate frame:
      - Points are computed in CAMERA (RealSense optical) convention:
          x = right, y = down, z = forward
      - Optional axis flip can be applied via FeaturesCfg.cam_axis_flip,
        e.g. (1, -1, 1) to convert y-down to y-up.
    """

    def __init__(self, intr: CameraIntrinsics, depth_cfg: DepthCfg, feats_cfg: FeaturesCfg) -> None:
        self.intr = intr
        self.depth_cfg = depth_cfg
        self.cfg = feats_cfg

    def __call__(
        self,
        depth_masked: DepthU16,
        label: LabelImage,
        valid_range_mask: np.ndarray | None = None,
    ) -> FeaturesResult:
        assert_depth_u16(depth_masked)
        assert_label_u16(label)
        if depth_masked.shape != label.shape:
            raise ValueError(f"Shape mismatch depth={depth_masked.shape} label={label.shape}")

        step = max(1, int(self.cfg.downsample_step))
        h, w = depth_masked.shape

        if step > 1:
            d_sub = depth_masked[0:h:step, 0:w:step]
            l_sub = label[0:h:step, 0:w:step]
            v_grid, u_grid = np.mgrid[0:h:step, 0:w:step]
            vr_sub = valid_range_mask[0:h:step, 0:w:step] if valid_range_mask is not None else None
        else:
            d_sub = depth_masked
            l_sub = label
            v_grid, u_grid = np.mgrid[0:h, 0:w]
            vr_sub = valid_range_mask

        z_m = depth_u16_to_meters(d_sub, self.depth_cfg)

        valid = np.isfinite(z_m) & (z_m > 0.0)
        if vr_sub is not None:
            valid = valid & vr_sub.astype(bool)

        if not np.any(valid):
            return FeaturesResult({}, np.zeros((0, 3), dtype=np.float32), {})

        ids = np.unique(l_sub[valid])
        ids = ids[ids > 0]
        if ids.size == 0:
            return FeaturesResult({}, np.zeros((0, 3), dtype=np.float32), {})

        fx, fy, cx, cy = float(self.intr.fx), float(self.intr.fy), float(self.intr.cx), float(self.intr.cy)
        if fx == 0.0 or fy == 0.0:
            raise ValueError(f"Invalid intrinsics: fx={fx}, fy={fy}")

        # Optional axis flip, e.g. (1, -1, 1) to flip camera Y.
        # This is intentionally tolerant (works even if FeaturesCfg has no such field yet).
        cam_axis_flip = np.asarray(getattr(self.cfg, "cam_axis_flip", (1.0, 1.0, 1.0)), dtype=np.float32).reshape(
            (1, 3)
        )

        clouds_by_instance: dict[int, PointCloud] = {}
        features: dict[int, InstanceFeatures] = {}
        all_pts_list: list[np.ndarray] = []

        for inst_id in ids.tolist():
            mask = valid & (l_sub == inst_id)
            if not np.any(mask):
                continue

            z_i = z_m[mask].astype(np.float32, copy=False)
            v_i = v_grid[mask].astype(np.float32, copy=False)
            u_i = u_grid[mask].astype(np.float32, copy=False)

            x_i = (u_i - cx) * z_i / fx
            y_i = (v_i - cy) * z_i / fy
            pts = np.stack((x_i, y_i, z_i), axis=-1).astype(np.float32, copy=False)

            # Apply optional axis flip in CAM frame (elementwise multiply).
            if not np.allclose(cam_axis_flip, 1.0):
                pts = pts * cam_axis_flip

            if pts.shape[0] < int(self.cfg.min_points):
                continue

            clouds_by_instance[int(inst_id)] = pts
            all_pts_list.append(pts)

            centroid = pts.mean(axis=0)
            pmin = pts.min(axis=0)
            pmax = pts.max(axis=0)
            ext = pmax - pmin
            vol = float(ext[0] * ext[1] * ext[2])

            features[int(inst_id)] = InstanceFeatures(
                instance_id=int(inst_id),
                num_points=int(pts.shape[0]),
                centroid_m=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                extent_m=(float(ext[0]), float(ext[1]), float(ext[2])),
                box_volume_m3=vol,
            )

        all_points = (
            np.vstack(all_pts_list).astype(np.float32, copy=False)
            if all_pts_list
            else np.zeros((0, 3), dtype=np.float32)
        )
        return FeaturesResult(clouds_by_instance=clouds_by_instance, all_points=all_points, features=features)
