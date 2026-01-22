#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import numpy as np

def Rx(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa, ca]], dtype=np.float64)

def Ry(b):
    cb, sb = np.cos(b), np.sin(b)
    return np.array([[cb, 0, sb],
                     [0, 1, 0],
                     [-sb, 0, cb]], dtype=np.float64)

def Rz(c):
    cc, sc = np.cos(c), np.sin(c)
    return np.array([[cc, -sc, 0],
                     [sc,  cc, 0],
                     [0,   0,  1]], dtype=np.float64)

def euler_to_R(alpha_deg, beta_deg, gamma_deg, variant: str):
    # Meca: "mobile XYZ (RxRyRz)" — je nach Implementierung kommt man in der Praxis
    # häufig in eine der beiden Ketten. Wir testen beide.
    a = np.deg2rad(alpha_deg)
    b = np.deg2rad(beta_deg)
    c = np.deg2rad(gamma_deg)
    if variant == "RxRyRz":
        return Rx(a) @ Ry(b) @ Rz(c)
    if variant == "RzRyRx":
        return Rz(c) @ Ry(b) @ Rx(a)
    raise ValueError(variant)

def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def load_yaml_min(path: Path):
    # ultra-min YAML reader für die paar Felder; nutzt PyYAML falls vorhanden
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        raise SystemExit("PyYAML fehlt. Install: pip install pyyaml")

def find_cam_plys(results_raw_cloud_dir: Path):
    # Erwartet .../raw_clouds/plant_XXX/*.ply
    cam = sorted(results_raw_cloud_dir.glob("*_cam.ply"))
    if cam:
        items = []
        for p in cam:
            view_id = p.stem.replace("_cam", "")
            items.append((view_id, p))
        return items
    # Fallback: .../raw_clouds/plant_XXX/view_XX/cloud_cam.ply
    cam = sorted(results_raw_cloud_dir.glob("**/cloud_cam.ply"))
    items = []
    for p in cam:
        view_id = p.parent.name
        items.append((view_id, p))
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="deine pipeline config.yaml")
    ap.add_argument("--raw-cloud-dir", type=Path, required=True,
                    help=".../raw_clouds/plant_XXX (wo *_cam.ply liegt)")
    ap.add_argument("--poses-json", type=Path, default=None,
                    help="optional: poses.json; sonst versucht aus config.dataset.poses_json oder dataset-root zu finden")
    ap.add_argument("--voxel", type=float, default=0.004, help="Downsample voxel size [m]")
    ap.add_argument("--maxcorr", type=float, default=0.02, help="ICP max correspondence [m]")
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args()

    try:
        import open3d as o3d
    except ImportError:
        raise SystemExit("open3d fehlt. Install: pip install open3d")

    cfg = load_yaml_min(args.config)

    # Extrinsics cam->trf
    ext = cfg.get("extrinsics_to_trf", {})
    R_trf_cam = np.array(ext["R_trf_cam"], dtype=np.float64).reshape(3, 3)
    t_trf_cam_mm = np.array(ext["t_trf_cam_mm"], dtype=np.float64).reshape(3)
    t_trf_cam_m = t_trf_cam_mm * 1e-3
    T_trf_cam = make_T(R_trf_cam, t_trf_cam_m)

    # Poses: TRF wrt WORLD/WRF (mm/deg)
    poses_path = args.poses_json
    if poses_path is None:
        ds = cfg.get("dataset", {})
        if "poses_json" in ds:
            poses_path = Path(ds["poses_json"])
    if poses_path is None or not poses_path.exists():
        raise SystemExit("poses.json nicht gefunden. Übergib --poses-json <path> oder setz dataset.poses_json in config.")

    with open(poses_path, "r", encoding="utf-8") as f:
        poses = json.load(f)

    # poses kann dict oder list sein -> wir bauen lookup view_id -> pose
    pose_by_view = {}
    if isinstance(poses, dict):
        pose_by_view = poses
    else:
        for it in poses:
            pose_by_view[it["view_id"]] = it

    cam_items = find_cam_plys(args.raw_cloud_dir)
    if len(cam_items) < 2:
        raise SystemExit(f"Zu wenige CAM-PLYs gefunden in {args.raw_cloud_dir} (mind. 2).")

    # Load + downsample
    pcd_cam = {}
    for view_id, ply in cam_items:
        pc = o3d.io.read_point_cloud(str(ply))
        if args.voxel > 0:
            pc = pc.voxel_down_sample(args.voxel)
        pcd_cam[view_id] = pc

    view_ids = [vid for vid, _ in cam_items if vid in pcd_cam and vid in pose_by_view]

    def T_world_trf_from_pose(view_id, euler_variant):
        p = pose_by_view[view_id]
        # akzeptiere verschiedene key-namen
        x = p.get("x_mm", p.get("x", p["pose_mm_deg"][0]))
        y = p.get("y_mm", p.get("y", p["pose_mm_deg"][1]))
        z = p.get("z_mm", p.get("z", p["pose_mm_deg"][2]))
        a = p.get("alpha_deg", p.get("alpha", p["pose_mm_deg"][3]))
        b = p.get("beta_deg",  p.get("beta",  p["pose_mm_deg"][4]))
        g = p.get("gamma_deg", p.get("gamma", p["pose_mm_deg"][5]))
        t = np.array([x, y, z], dtype=np.float64) * 1e-3
        R = euler_to_R(a, b, g, euler_variant)
        return make_T(R, t)

    def build_world_clouds(euler_variant, invert_pose: bool, invert_extr: bool):
        world = {}
        Tex = inv_T(T_trf_cam) if invert_extr else T_trf_cam
        for vid in view_ids:
            Twt = T_world_trf_from_pose(vid, euler_variant)
            Twt = inv_T(Twt) if invert_pose else Twt
            Twc = Twt @ Tex
            pc = pcd_cam[vid].transform(Twc.copy())  # open3d mutiert in-place
            world[vid] = pc
        return world

    def icp_score(world_clouds):
        # Pairwise ICP, init=I, kleine maxcorr -> wenn global falsch, gibt's miese fitness
        vids = list(world_clouds.keys())
        rmses, fits = [], []
        for i in range(len(vids)):
            for j in range(i + 1, len(vids)):
                src = world_clouds[vids[i]]
                tgt = world_clouds[vids[j]]
                reg = o3d.pipelines.registration.registration_icp(
                    src, tgt,
                    args.maxcorr,
                    np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )
                rmses.append(reg.inlier_rmse)
                fits.append(reg.fitness)
        return float(np.mean(fits)), float(np.mean(rmses))

    hypotheses = []
    for euler_variant in ["RxRyRz", "RzRyRx"]:
        for invert_pose in [False, True]:
            for invert_extr in [False, True]:
                world = build_world_clouds(euler_variant, invert_pose, invert_extr)
                fit, rmse = icp_score(world)
                hypotheses.append((fit, rmse, euler_variant, invert_pose, invert_extr, world))

    # best: hohe fitness, niedrige rmse
    hypotheses.sort(key=lambda x: (-x[0], x[1]))
    best = hypotheses[0]
    fit, rmse, euler_variant, invert_pose, invert_extr, world = best

    print("\n=== Ranking (top 5) ===")
    for k, (f, r, ev, ip, ie, _) in enumerate(hypotheses[:5], 1):
        print(f"{k:>2}. fitness={f:.3f}  rmse={r*1000:.2f} mm   euler={ev}  invert_pose={ip}  invert_extr={ie}")

    print("\n=== Diagnose ===")
    if invert_extr and not invert_pose:
        print("→ Sieht stark nach: EXTRINSIC invertiert (cam↔trf) aus.")
    elif invert_pose and not invert_extr:
        print("→ Sieht stark nach: POSE invertiert (world↔trf) aus.")
    elif invert_pose and invert_extr:
        print("→ Beide Inversionen zusammen liefern die beste Übereinstimmung (prüf Pose & Extrinsic).")
    else:
        print("→ Pose/Extrinsic wirken konsistent; falls es trotzdem 'komisch' aussieht: Euler-Konvention prüfen.")

    if args.no_viz:
        return

    # Visualisierung: Views unterschiedlich färben + Welt-Frames
    colors = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1],
        [0.8, 0.5, 0.2], [0.5, 0.2, 0.8]
    ]

    geoms = []
    for idx, vid in enumerate(sorted(world.keys())):
        pc = world[vid]
        c = colors[idx % len(colors)]
        pc.paint_uniform_color(c)
        geoms.append(pc)

        # optional: Koordinatenframe am jeweiligen Kamera-Frame in WORLD
        # Wir rekonstruieren Twc wie oben (fürs Frame)
        Tex = inv_T(T_trf_cam) if invert_extr else T_trf_cam
        Twt = T_world_trf_from_pose(vid, euler_variant)
        Twt = inv_T(Twt) if invert_pose else Twt
        Twc = Twt @ Tex

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame.transform(Twc)
        geoms.append(frame)

    o3d.visualization.draw_geometries(geoms)

if __name__ == "__main__":
    main()
