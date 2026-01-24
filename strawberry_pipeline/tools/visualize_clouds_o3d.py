#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_clouds_o3d.py

Interaktive Visualisierung (Open3D) für je Plant drei Frames:
- cam
- trf
- world

Pro Frame öffnet sich ein eigenes Fenster, mit fixer Farbzuordnung:
view 0 = rot, view 1 = gelb, view 2 = grün

Pfadlayout (wie bei dir):
<out_root>/plant_007/raw_clouds/cloud_0_cam.ply
<out_root>/plant_007/raw_clouds/cloud_0_trf.ply
<out_root>/plant_007/raw_clouds/cloud_0_world.ply
etc.

Tasten:
- N  : nächstes Plant
- P  : vorheriges Plant
- Q/ESC : quit
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except Exception as e:
    raise SystemExit(
        "Open3D import failed. Install with:\n"
        "  pip install open3d\n"
        f"Error: {e}"
    )


# view 0 = rot, view 1 = gelb, view 2 = grün
VIEW_COLORS_RGB01: Dict[int, Tuple[float, float, float]] = {
    0: (1.0, 0.0, 0.0),
    1: (1.0, 1.0, 0.0),
    2: (0.0, 1.0, 0.0),
}


# ----------------------------
# PLY reader (xyz only)
# ----------------------------
def read_ply_xyz(path: Path) -> np.ndarray:
    """Minimal PLY reader for vertex x y z (ASCII or binary_little_endian). Returns (N,3) float32."""
    with path.open("rb") as f:
        fmt = None
        n = None
        props: List[str] = []
        in_vertex = False

        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY (EOF before end_header): {path}")
            s = line.decode("ascii", errors="ignore").strip()

            if s.startswith("format "):
                fmt = s.split()[1]
            elif s.startswith("element vertex "):
                n = int(s.split()[-1])
                in_vertex = True
                props = []
            elif s.startswith("element ") and not s.startswith("element vertex"):
                in_vertex = False
            elif s.startswith("property ") and in_vertex:
                props.append(s.split()[-1])
            elif s == "end_header":
                break

        if fmt is None or n is None:
            raise ValueError(f"PLY missing format/vertex count: {path}")

        try:
            ix, iy, iz = props.index("x"), props.index("y"), props.index("z")
        except ValueError as e:
            raise ValueError(f"PLY has no x/y/z properties: {path} props={props}") from e

        if fmt == "ascii":
            lines = f.read().decode("utf-8", errors="ignore").splitlines()
            pts = np.zeros((n, 3), dtype=np.float32)
            for i in range(n):
                parts = lines[i].split()
                pts[i, 0] = float(parts[ix])
                pts[i, 1] = float(parts[iy])
                pts[i, 2] = float(parts[iz])
            return pts

        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format '{fmt}': {path}")

        stride = len(props) * 4
        raw = f.read(n * stride)
        if len(raw) < n * stride:
            raise ValueError(f"PLY binary too short: {path}")

        data = np.frombuffer(raw, dtype="<f4").reshape((n, len(props)))
        return data[:, [ix, iy, iz]].astype(np.float32, copy=False)


def voxel_downsample_numpy(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Fast voxel downsample in numpy (keeps one point per voxel)."""
    pts = np.asarray(pts, dtype=np.float32).reshape((-1, 3))
    if pts.size == 0 or voxel <= 0:
        return pts
    q = np.floor(pts / float(voxel)).astype(np.int32)
    _, idx = np.unique(q, axis=0, return_index=True)
    return pts[idx]


def load_clouds(plant_dir: Path, views: List[int], frame: str) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for vid in views:
        p = plant_dir / f"cloud_{vid}_{frame}.ply"
        if not p.exists():
            continue
        try:
            out[vid] = read_ply_xyz(p)
        except Exception as e:
            print(f"[WARN] failed reading {p}: {e}")
    return out


def to_o3d_pcd(pts: np.ndarray, color_rgb01: Tuple[float, float, float]) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pts = np.asarray(pts, dtype=np.float32).reshape((-1, 3))
    pcd.points = o3d.utility.Vector3dVector(pts)
    # uniform color
    pcd.paint_uniform_color(list(color_rgb01))
    return pcd


def make_geometries(
    clouds: Dict[int, np.ndarray],
    views: List[int],
    voxel: float,
    sample: int,
) -> List[o3d.geometry.Geometry]:
    geoms: List[o3d.geometry.Geometry] = []
    for vid in views:
        if vid not in clouds:
            continue
        pts = clouds[vid]
        if pts.shape[0] == 0:
            continue

        if voxel > 0:
            pts = voxel_downsample_numpy(pts, voxel)

        if sample > 0 and pts.shape[0] > sample:
            idx = np.random.choice(pts.shape[0], size=sample, replace=False)
            pts = pts[idx]

        col = VIEW_COLORS_RGB01.get(vid, (0.7, 0.7, 0.7))
        geoms.append(to_o3d_pcd(pts, col))

    # optional coordinate frame for orientation
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))
    return geoms


def show_frame_windows_for_plant(
    pid: int,
    out_root: Path,
    views: List[int],
    frames: List[str],
    voxel: float,
    sample: int,
) -> None:
    plant_dir = out_root / f"plant_{pid:03d}" / "raw_clouds"
    if not plant_dir.exists():
        print(f"[plant_{pid:03d}] Keine Clouds gefunden unter: {plant_dir}")
        return

    print(f"\n=== plant_{pid:03d} ===")
    print("Keys: N=next plant | P=prev plant | Q/ESC=quit (in the console after closing windows)")

    # Für jedes Frame ein eigenes Fenster (nacheinander):
    for frame in frames:
        clouds = load_clouds(plant_dir, views, frame)
        if not clouds:
            print(f"[plant_{pid:03d}] frame={frame}: nichts gefunden.")
            continue

        geoms = make_geometries(clouds, views=views, voxel=voxel, sample=sample)

        title = f"plant_{pid:03d} | {frame} | view0=rot view1=gelb view2=gruen"
        try:
            # draw_geometries is simplest + interactive (mouse rotate/zoom)
            o3d.visualization.draw_geometries(
                geoms,
                window_name=title,
                width=1280,
                height=720,
            )
        except Exception as e:
            print(f"[plant_{pid:03d}] Open3D window failed for frame={frame}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, type=Path, help="e.g. configs/outputs")
    ap.add_argument("--plant_ids", nargs="+", type=int, required=True)
    ap.add_argument("--views", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--frames", nargs="+", default=["cam", "trf", "world"], choices=["cam", "trf", "world"])
    ap.add_argument("--voxel", type=float, default=0.0, help="voxel size in meters (0 disables)")
    ap.add_argument("--sample", type=int, default=60000, help="random sample per view (0 disables)")
    args = ap.parse_args()

    np.random.seed(0)

    plant_ids = list(args.plant_ids)
    idx = 0

    while 0 <= idx < len(plant_ids):
        pid = plant_ids[idx]
        show_frame_windows_for_plant(
            pid=pid,
            out_root=args.out_root,
            views=args.views,
            frames=args.frames,
            voxel=args.voxel,
            sample=args.sample,
        )

        # Navigation über Konsole (nachdem du die 3 Fenster geschlossen hast)
        cmd = input("[N]ext / [P]rev / [Q]uit? ").strip().lower()
        if cmd in ("q", "quit", "esc"):
            break
        if cmd in ("p", "prev", "previous"):
            idx = max(0, idx - 1)
        else:
            # default: next
            idx = min(len(plant_ids) - 1, idx + 1)


if __name__ == "__main__":
    main()
