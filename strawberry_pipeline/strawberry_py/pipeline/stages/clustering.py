# strawberry_py/pipeline/stages/clustering.py
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from strawberry_py.config import ClusterCfg
from strawberry_py.types import InstanceFeatures, PointCloud, Pose
from strawberry_py.pipeline.stages.transforms import apply_pose, optical_to_trf


@dataclass
class Cluster:
    cluster_id: int
    centroid_world: np.ndarray  # (3,) float32
    num_points_weight: int
    views_seen: Set[int] = field(default_factory=set)
    points_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))


class StrawberryClusterer:
    """
    Offline clustering across views.

    Wichtig (und der Grund, warum es bei dir wie "3 Inseln" aussieht):
      - Die 3D-Punkte aus der Depth-Projektion sind im RealSense OPTICAL frame:
          x rechts, y runter, z vorwärts
      - Deine Roboter-Pose (GetPose/TRF) ist nicht in diesem Frame.
      - Deshalb muss man VOR dem World-Transform einen Frame-Convert machen:
          optical -> "tool/trf-like"  (optical_to_trf)
        und erst danach:
          tool/trf -> world (apply_pose)

    Dieses Modul macht genau das (inkl. flip_y-Hack als Option).
    """

    def __init__(self, cfg: ClusterCfg) -> None:
        self.cfg = cfg
        self._clusters: List[Cluster] = []
        self._by_id: Dict[int, Cluster] = {}
        self._next_id: int = 1
        self._assignments: List[Dict[str, Any]] = []

        # deterministic downsample when truncating points (optional)
        self._rng = np.random.default_rng(0)

    def reset(self) -> None:
        self._clusters.clear()
        self._by_id.clear()
        self._next_id = 1
        self._assignments.clear()

    # ---------------------------------------------------------------------
    # Transform helpers
    # ---------------------------------------------------------------------

    def _points_world_from_optical(self, pts_optical: np.ndarray, pose_world_cam: Pose) -> np.ndarray:
        """
        pts_optical: Nx3 in RealSense optical frame (x right, y down, z forward)
        pose_world_cam: Pose of camera/TRF in world (as provided by your robot config)
        """
        pts_optical = np.asarray(pts_optical, dtype=np.float32).reshape((-1, 3))
        if pts_optical.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # 1) optical -> trf/tool-like frame
        pts_trf = optical_to_trf(pts_optical, flip_y=bool(getattr(self.cfg, "flip_y", False)))

        # 2) trf/tool-like -> world
        pts_world = apply_pose(pts_trf, pose_world_cam)
        return np.asarray(pts_world, dtype=np.float32).reshape((-1, 3))

    def _centroid_world_from_optical(self, centroid_optical: np.ndarray, pose_world_cam: Pose) -> np.ndarray:
        c = np.asarray(centroid_optical, dtype=np.float32).reshape((1, 3))
        cw = self._points_world_from_optical(c, pose_world_cam)
        return cw.reshape((3,))

    # ---------------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------------

    def add_view(
        self,
        plant_id: int,
        view_id: int,
        clouds_by_instance: Dict[int, PointCloud],
        features: Dict[int, InstanceFeatures],
        pose_world_cam: Pose,
    ) -> None:
        """
        For each instance in this view:
          - transform points from optical -> trf -> world
          - compute centroid_world
          - assign to a cluster (centroid distance)
          - append world points to cluster
        """
        if not bool(self.cfg.enabled):
            return

        for inst_id, pts_optical in clouds_by_instance.items():
            pts_optical_f32 = np.asarray(pts_optical, dtype=np.float32).reshape((-1, 3))
            if pts_optical_f32.size == 0:
                continue

            # Transform points FIRST (ensures centroid and points share identical frame logic)
            pts_world = self._points_world_from_optical(pts_optical_f32, pose_world_cam)
            if pts_world.size == 0:
                continue

            # Prefer centroid from transformed points (robust even if FeatureExtractor centroid differs slightly)
            centroid_world = pts_world.mean(axis=0).astype(np.float32, copy=False)

            feat = features.get(int(inst_id))
            num_points = int(feat.num_points) if feat is not None else int(pts_world.shape[0])

            cid, created = self._assign_to_cluster(centroid_world, num_points, view_id)
            self._append_points(cid, pts_world, view_id)

            self._assignments.append(
                {
                    "plant_id": int(plant_id),
                    "view_id": int(view_id),
                    "instance_id": int(inst_id),
                    "cluster_id": int(cid),
                    "num_points": int(num_points),
                    "created": bool(created),
                    "centroid_world_x": float(centroid_world[0]),
                    "centroid_world_y": float(centroid_world[1]),
                    "centroid_world_z": float(centroid_world[2]),
                }
            )

    # ---------------------------------------------------------------------
    # Clustering logic
    # ---------------------------------------------------------------------

    def _assign_to_cluster(self, centroid_world: np.ndarray, num_points: int, view_id: int) -> Tuple[int, bool]:
        """
        Returns: (cluster_id, created_new_cluster)
        """
        centroid_world = np.asarray(centroid_world, dtype=np.float32).reshape((3,))

        if not self._clusters:
            cid = self._create_cluster(centroid_world, num_points, view_id)
            return cid, True

        # find nearest
        dists = [float(np.linalg.norm(centroid_world - c.centroid_world)) for c in self._clusters]
        min_idx = int(np.argmin(dists))
        min_dist = float(dists[min_idx])
        best = self._clusters[min_idx]

        if min_dist < float(self.cfg.distance_threshold_m):
            # weighted centroid update
            total = int(best.num_points_weight + int(num_points))
            if total <= 0:
                total = 1
            w_old = best.num_points_weight / float(total)
            w_new = int(num_points) / float(total)

            best.centroid_world = (w_old * best.centroid_world) + (w_new * centroid_world)
            best.centroid_world = best.centroid_world.astype(np.float32, copy=False)
            best.num_points_weight = total
            best.views_seen.add(int(view_id))
            return best.cluster_id, False

        if len(self._clusters) < int(self.cfg.max_clusters):
            cid = self._create_cluster(centroid_world, num_points, view_id)
            return cid, True

        # fallback: assign to closest anyway (no new cluster)
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

        pts_world = np.asarray(pts_world, dtype=np.float32).reshape((-1, 3))
        if pts_world.size == 0:
            return

        if c.points_world.size == 0:
            c.points_world = pts_world
        else:
            c.points_world = np.vstack([c.points_world, pts_world])

        c.views_seen.add(int(view_id))

        max_n = int(getattr(self.cfg, "max_points_per_cluster", 0))
        if max_n > 0 and c.points_world.shape[0] > max_n:
            idx = self._rng.choice(c.points_world.shape[0], size=max_n, replace=False)
            c.points_world = c.points_world[idx]

    # ---------------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------------

    def export(self, plant_dir: Path) -> None:
        if not bool(self.cfg.enabled):
            return

        export_dirname = str(getattr(self.cfg, "export_dirname", "clusters"))
        out_dir = Path(plant_dir) / export_dirname
        out_dir.mkdir(parents=True, exist_ok=True)

        rows_summary: List[List[Any]] = []
        all_pts: List[np.ndarray] = []

        for c in self._clusters:
            if c.points_world.size == 0:
                continue

            ply_path = out_dir / f"cluster_{c.cluster_id:03d}.ply"
            self._write_ply_xyz(ply_path, c.points_world, ascii_mode=bool(self.cfg.ply_ascii))

            all_pts.append(c.points_world)

            rows_summary.append(
                [
                    int(c.cluster_id),
                    float(c.centroid_world[0]),
                    float(c.centroid_world[1]),
                    float(c.centroid_world[2]),
                    int(c.points_world.shape[0]),
                    ",".join(map(str, sorted(list(c.views_seen)))),
                    ply_path.name,
                ]
            )

        # summary.csv
        csv_path = out_dir / "summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["cluster_id", "centroid_x", "centroid_y", "centroid_z", "num_points", "views_seen", "ply_file"])
            w.writerows(rows_summary)

        # assignments.csv
        a_path = out_dir / "assignments.csv"
        with a_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(
                [
                    "plant_id",
                    "view_id",
                    "instance_id",
                    "cluster_id",
                    "num_points",
                    "created",
                    "centroid_world_x",
                    "centroid_world_y",
                    "centroid_world_z",
                ]
            )
            for a in self._assignments:
                w.writerow(
                    [
                        a["plant_id"],
                        a["view_id"],
                        a["instance_id"],
                        a["cluster_id"],
                        a["num_points"],
                        int(bool(a["created"])),
                        a["centroid_world_x"],
                        a["centroid_world_y"],
                        a["centroid_world_z"],
                    ]
                )

        # fused plant cloud
        if bool(getattr(self.cfg, "export_fused_plant_cloud", False)) and all_pts:
            pts_all = np.vstack(all_pts).astype(np.float32, copy=False)

            voxel = float(getattr(self.cfg, "fused_voxel_size", 0.0))
            if voxel > 0.0:
                pts_all = self._voxel_downsample(pts_all, voxel)

            max_pts = int(getattr(self.cfg, "fused_max_points", 0))
            if max_pts > 0 and pts_all.shape[0] > max_pts:
                idx = self._rng.choice(pts_all.shape[0], size=max_pts, replace=False)
                pts_all = pts_all[idx]

            fused_name = str(getattr(self.cfg, "fused_filename", "plant_fused.ply"))
            fused_path = out_dir / fused_name
            self._write_ply_xyz(fused_path, pts_all, ascii_mode=bool(self.cfg.ply_ascii))

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

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

        path.parent.mkdir(parents=True, exist_ok=True)

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
