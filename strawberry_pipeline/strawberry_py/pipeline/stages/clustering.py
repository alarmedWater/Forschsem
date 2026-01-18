# strawberry_py/pipeline/stages/clustering.py
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from strawberry_py.config import ClusterCfg
from strawberry_py.types import InstanceFeatures, PointCloud, Pose
from strawberry_py.pipeline.stages.transforms import quaternion_to_rotation_matrix


@dataclass
class Cluster:
    cluster_id: int
    centroid_world: np.ndarray  # (3,)
    num_points_weight: int
    views_seen: Set[int] = field(default_factory=set)
    points_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))


class StrawberryClusterer:
    """
    Cluster instances across views in WORLD coordinates.

    Points/features arrive in CAMERA (RealSense optical) frame.
    We apply:
      TRF <- CAM  (fixed correction from YAML)
      WORLD <- TRF (pose from robot GetPose, per view)
    """

    def __init__(self, cfg: ClusterCfg) -> None:
        self.cfg = cfg
        self._clusters: List[Cluster] = []
        self._by_id: Dict[int, Cluster] = {}
        self._next_id = 1
        self._assignments: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self._clusters.clear()
        self._by_id.clear()
        self._next_id = 1
        self._assignments.clear()

    @staticmethod
    def _pose_to_Rt_world_trf(pose_world_trf: Pose) -> Tuple[np.ndarray, np.ndarray]:
        qx, qy, qz, qw = pose_world_trf.q_xyzw
        R = quaternion_to_rotation_matrix(float(qx), float(qy), float(qz), float(qw)).astype(np.float32)
        t = np.asarray(pose_world_trf.t_xyz, dtype=np.float32).reshape((3,))
        return R, t

    @staticmethod
    def _cam_to_trf(points_cam: np.ndarray, R_trf_cam: np.ndarray, t_trf_cam_m: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_cam, dtype=np.float32).reshape((-1, 3))
        if pts.size == 0:
            return pts
        R = np.asarray(R_trf_cam, dtype=np.float32).reshape((3, 3))
        t = np.asarray(t_trf_cam_m, dtype=np.float32).reshape((3,))
        return (R @ pts.T).T + t

    def add_view(
        self,
        plant_id: int,
        view_id: int,
        clouds_by_instance: Dict[int, PointCloud],
        features: Dict[int, InstanceFeatures],
        pose_world_trf: Pose,          # WORLD <- TRF from robot GetPose
        R_trf_cam: np.ndarray,         # TRF <- CAM correction
        t_trf_cam_m: np.ndarray,       # TRF <- CAM translation (meters)
    ) -> None:
        if not self.cfg.enabled:
            return

        Rw, tw = self._pose_to_Rt_world_trf(pose_world_trf)

        for inst_id, pts_cam in clouds_by_instance.items():
            feat = features.get(int(inst_id))
            if feat is None or int(feat.num_points) <= 0:
                continue

            # -------- centroid: CAM -> TRF -> WORLD --------
            centroid_cam = np.asarray(feat.centroid_m, dtype=np.float32).reshape((1, 3))
            centroid_trf = self._cam_to_trf(centroid_cam, R_trf_cam, t_trf_cam_m)[0]
            centroid_world = (Rw @ centroid_trf) + tw

            # IMPORTANT: clustering uses WORLD centroids
            cid, created = self._assign_to_cluster(centroid_world, int(feat.num_points), int(view_id))

            # -------- points: CAM -> TRF -> WORLD --------
            pts_trf = self._cam_to_trf(np.asarray(pts_cam, dtype=np.float32), R_trf_cam, t_trf_cam_m)
            pts_world = (Rw @ pts_trf.T).T + tw
            self._append_points(cid, pts_world, int(view_id))

            self._assignments.append(
                dict(
                    plant_id=int(plant_id),
                    view_id=int(view_id),
                    instance_id=int(inst_id),
                    cluster_id=int(cid),
                    num_points=int(feat.num_points),
                    created=bool(created),
                    centroid_world_x=float(centroid_world[0]),
                    centroid_world_y=float(centroid_world[1]),
                    centroid_world_z=float(centroid_world[2]),
                )
            )

    def _assign_to_cluster(self, centroid_world: np.ndarray, num_points: int, view_id: int) -> Tuple[int, bool]:
        if not self._clusters:
            return self._create_cluster(centroid_world, num_points, view_id), True

        dists = [float(np.linalg.norm(centroid_world - c.centroid_world)) for c in self._clusters]
        min_idx = int(np.argmin(dists))
        best = self._clusters[min_idx]
        min_dist = float(dists[min_idx])

        if min_dist < float(self.cfg.distance_threshold_m):
            total = int(best.num_points_weight + int(num_points))
            if total > 0:
                w_old = best.num_points_weight / float(total)
                w_new = int(num_points) / float(total)
                best.centroid_world = (w_old * best.centroid_world) + (w_new * centroid_world.astype(np.float32))
                best.num_points_weight = total
            best.views_seen.add(int(view_id))
            return best.cluster_id, False

        if len(self._clusters) < int(self.cfg.max_clusters):
            return self._create_cluster(centroid_world, num_points, view_id), True

        best.views_seen.add(int(view_id))
        return best.cluster_id, False

    def _create_cluster(self, centroid_world: np.ndarray, num_points: int, view_id: int) -> int:
        cid = int(self._next_id)
        self._next_id += 1
        c = Cluster(
            cluster_id=cid,
            centroid_world=np.asarray(centroid_world, dtype=np.float32).copy(),
            num_points_weight=int(num_points),
            views_seen={int(view_id)},
        )
        self._clusters.append(c)
        self._by_id[cid] = c
        return cid

    def _append_points(self, cluster_id: int, pts_world: np.ndarray, view_id: int) -> None:
        c = self._by_id.get(int(cluster_id))
        if c is None:
            return

        pts = np.asarray(pts_world, dtype=np.float32).reshape((-1, 3))
        if pts.size == 0:
            return

        c.points_world = pts if c.points_world.size == 0 else np.vstack([c.points_world, pts])
        c.views_seen.add(int(view_id))

        max_n = int(self.cfg.max_points_per_cluster)
        if max_n > 0 and c.points_world.shape[0] > max_n:
            idx = np.random.choice(c.points_world.shape[0], size=max_n, replace=False)
            c.points_world = c.points_world[idx]

    def export(self, plant_dir: Path) -> None:
        if not self.cfg.enabled:
            return

        out_dir = plant_dir / self.cfg.export_dirname
        out_dir.mkdir(parents=True, exist_ok=True)

        rows: List[List[Any]] = []
        all_pts: List[np.ndarray] = []

        for c in self._clusters:
            if c.points_world.size == 0:
                continue

            ply_path = out_dir / f"cluster_{c.cluster_id:03d}.ply"
            self._write_ply_xyz(ply_path, c.points_world, ascii_mode=bool(self.cfg.ply_ascii))
            all_pts.append(c.points_world)

            rows.append(
                [
                    int(c.cluster_id),
                    float(c.centroid_world[0]),
                    float(c.centroid_world[1]),
                    float(c.centroid_world[2]),
                    int(c.points_world.shape[0]),
                    ",".join(map(str, sorted(c.views_seen))),
                    ply_path.name,
                ]
            )

        with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["cluster_id", "centroid_x", "centroid_y", "centroid_z", "num_points", "views_seen", "ply_file"])
            w.writerows(rows)

        with (out_dir / "assignments.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(
                ["plant_id", "view_id", "instance_id", "cluster_id", "num_points", "created",
                 "centroid_world_x", "centroid_world_y", "centroid_world_z"]
            )
            for a in self._assignments:
                w.writerow(
                    [a["plant_id"], a["view_id"], a["instance_id"], a["cluster_id"], a["num_points"], int(a["created"]),
                     a["centroid_world_x"], a["centroid_world_y"], a["centroid_world_z"]]
                )

        if self.cfg.export_fused_plant_cloud and all_pts:
            pts_all = np.vstack(all_pts).astype(np.float32, copy=False)

            if float(self.cfg.fused_voxel_size) > 0.0:
                pts_all = self._voxel_downsample(pts_all, float(self.cfg.fused_voxel_size))

            if int(self.cfg.fused_max_points) > 0 and pts_all.shape[0] > int(self.cfg.fused_max_points):
                idx = np.random.choice(pts_all.shape[0], size=int(self.cfg.fused_max_points), replace=False)
                pts_all = pts_all[idx]

            fused_path = out_dir / self.cfg.fused_filename
            self._write_ply_xyz(fused_path, pts_all, ascii_mode=bool(self.cfg.ply_ascii))

    @staticmethod
    def _voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
        v = float(voxel)
        if v <= 0.0 or pts.size == 0:
            return pts
        q = np.floor(pts / v).astype(np.int32)
        _, idx = np.unique(q, axis=0, return_index=True)
        return pts[np.sort(idx)]

    @staticmethod
    def _write_ply_xyz(path: Path, points_xyz: np.ndarray, ascii_mode: bool = True) -> None:
        pts = np.asarray(points_xyz, dtype=np.float32).reshape((-1, 3))
        n = int(pts.shape[0])

        if ascii_mode:
            header = (
                "ply\n"
                "format ascii 1.0\n"
                f"element vertex {n}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "end_header\n"
            )
            lines = [header]
            lines.extend([f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n" for p in pts])
            path.write_text("".join(lines), encoding="utf-8")
            return

        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        ).encode("ascii")
        with path.open("wb") as f:
            f.write(header)
            f.write(pts.astype("<f4", copy=False).tobytes())
