#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


def _require_open3d():
    try:
        import open3d as o3d  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Open3D ist nicht installiert. Installiere es mit:\n"
            "  pip install open3d\n"
        ) from exc
    return o3d


def list_ply_files(p: Path) -> List[Path]:
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.glob("*.ply"))
    return []


def load_cloud(o3d, ply_path: Path):
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise ValueError(f"PointCloud ist leer: {ply_path}")
    return pcd


def basic_stats(points: np.ndarray) -> dict:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extent = maxs - mins
    centroid = points.mean(axis=0)

    return {
        "n_points": int(points.shape[0]),
        "min_xyz": mins,
        "max_xyz": maxs,
        "extent_xyz": extent,
        "centroid": centroid,
    }


def nn_distance_stats(o3d, pcd) -> Optional[dict]:
    # Open3D kann die nearest-neighbor distances direkt berechnen
    try:
        dists = np.asarray(pcd.compute_nearest_neighbor_distance(), dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None

    if dists.size == 0:
        return None

    return {
        "nn_mean": float(dists.mean()),
        "nn_median": float(np.median(dists)),
        "nn_p10": float(np.percentile(dists, 10)),
        "nn_p90": float(np.percentile(dists, 90)),
    }


def maybe_downsample(o3d, pcd, voxel_size: float):
    if voxel_size <= 0.0:
        return pcd
    return pcd.voxel_down_sample(voxel_size=float(voxel_size))


def maybe_remove_outliers(
    pcd,
    sor_enable: bool,
    sor_nb_neighbors: int,
    sor_std_ratio: float,
    ror_enable: bool,
    ror_radius: float,
    ror_min_neighbors: int,
):
    if sor_enable:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=int(sor_nb_neighbors),
            std_ratio=float(sor_std_ratio),
        )

    if ror_enable:
        pcd, _ = pcd.remove_radius_outlier(
            nb_points=int(ror_min_neighbors),
            radius=float(ror_radius),
        )

    return pcd


def colorize_by_z(o3d, pcd) -> None:
    pts = np.asarray(pcd.points, dtype=np.float32)
    z = pts[:, 2]
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if z_max - z_min < 1e-9:
        colors = np.zeros((pts.shape[0], 3), dtype=np.float32)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return

    t = (z - z_min) / (z_max - z_min)
    # simple colormap: blue -> green -> red
    colors = np.zeros((pts.shape[0], 3), dtype=np.float32)
    colors[:, 0] = t
    colors[:, 1] = 1.0 - np.abs(t - 0.5) * 2.0
    colors[:, 2] = 1.0 - t
    colors = np.clip(colors, 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(colors)


def plot_nn_histogram(dists: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt  # noqa: WPS433

    plt.figure()
    plt.hist(dists, bins=60)
    plt.title(title)
    plt.xlabel("Nearest-neighbor distance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def visualize(o3d, pcd, window_title: str, show_frame: bool) -> None:
    geoms = [pcd]
    if show_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geoms.append(frame)
    o3d.visualization.draw_geometries(geoms, window_name=window_title)


def inspect_file(
    o3d,
    ply_path: Path,
    voxel: float,
    sor: bool,
    sor_nb: int,
    sor_std: float,
    ror: bool,
    ror_radius: float,
    ror_min: int,
    color_z: bool,
    show_frame: bool,
    show_hist: bool,
) -> None:
    print("\n" + "=" * 80)
    print(f"FILE: {ply_path}")

    pcd = load_cloud(o3d, ply_path)
    pcd = maybe_downsample(o3d, pcd, voxel)

    pcd = maybe_remove_outliers(
        pcd=pcd,
        sor_enable=sor,
        sor_nb_neighbors=sor_nb,
        sor_std_ratio=sor_std,
        ror_enable=ror,
        ror_radius=ror_radius,
        ror_min_neighbors=ror_min,
    )

    pts = np.asarray(pcd.points, dtype=np.float32)
    stats = basic_stats(pts)
    print(f"Points: {stats['n_points']}")
    print(f"Centroid (x,y,z): {stats['centroid']}")
    print(f"Min xyz: {stats['min_xyz']}")
    print(f"Max xyz: {stats['max_xyz']}")
    print(f"Extent xyz: {stats['extent_xyz']}")

    nn_stats = nn_distance_stats(o3d, pcd)
    if nn_stats is not None:
        print(
            "NN distance (proxy for density): "
            f"mean={nn_stats['nn_mean']:.5f}, "
            f"median={nn_stats['nn_median']:.5f}, "
            f"p10={nn_stats['nn_p10']:.5f}, "
            f"p90={nn_stats['nn_p90']:.5f}"
        )

    if color_z:
        colorize_by_z(o3d, pcd)

    if show_hist:
        dists = np.asarray(pcd.compute_nearest_neighbor_distance(), dtype=np.float32)
        plot_nn_histogram(dists, title=f"NN distances: {ply_path.name}")

    visualize(o3d, pcd, window_title=ply_path.name, show_frame=show_frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and visualize .ply point clouds (Open3D)."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a .ply file OR a directory containing *.ply files.",
    )
    parser.add_argument("--all", action="store_true", help="If path is a directory, iterate all .ply.")
    parser.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size in meters (0 disables).")

    parser.add_argument("--sor", action="store_true", help="Enable Statistical Outlier Removal.")
    parser.add_argument("--sor-nb", type=int, default=20, help="SOR nb_neighbors.")
    parser.add_argument("--sor-std", type=float, default=2.0, help="SOR std_ratio.")

    parser.add_argument("--ror", action="store_true", help="Enable Radius Outlier Removal.")
    parser.add_argument("--ror-radius", type=float, default=0.01, help="ROR radius (meters).")
    parser.add_argument("--ror-min", type=int, default=8, help="ROR min neighbors.")

    parser.add_argument("--color-z", action="store_true", help="Colorize points by z-value.")
    parser.add_argument("--frame", action="store_true", help="Show coordinate frame.")
    parser.add_argument("--hist", action="store_true", help="Show NN distance histogram.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    o3d = _require_open3d()

    p = Path(args.path).expanduser().resolve()
    files = list_ply_files(p)
    if not files:
        raise SystemExit(f"Keine .ply gefunden unter: {p}")

    if p.is_dir() and not args.all:
        print(f"Verzeichnis erkannt. Nutze --all um alle zu iterieren. Zeige nur: {files[0].name}")
        files = [files[0]]

    for f in files:
        inspect_file(
            o3d=o3d,
            ply_path=f,
            voxel=args.voxel,
            sor=args.sor,
            sor_nb=args.sor_nb,
            sor_std=args.sor_std,
            ror=args.ror,
            ror_radius=args.ror_radius,
            ror_min=args.ror_min,
            color_z=args.color_z,
            show_frame=args.frame,
            show_hist=args.hist,
        )


if __name__ == "__main__":
    main()
