#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

from strawberry_py.config import load_config
from strawberry_py.pipeline.stages.transforms import quaternion_to_rotation_matrix


def load_centroids(features_csv: Path):
    rows = []
    with features_csv.open("r", encoding="utf-8", newline="") as f:
        # features.csv ist bei dir ";"
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            rows.append(row)
    out = {}
    for row in rows:
        vid = int(row["view_id"])
        cx, cy, cz = float(row["cx"]), float(row["cy"]), float(row["cz"])
        out[vid] = np.array([cx, cy, cz], dtype=np.float64)
    return out


def pose_to_Rt(pose):
    qx, qy, qz, qw = pose.q_xyzw
    R = quaternion_to_rotation_matrix(float(qx), float(qy), float(qz), float(qw)).astype(np.float64)
    t = np.array(pose.t_xyz, dtype=np.float64).reshape(3,)
    return R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--view-ids", default="0,1,2")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    cents_cam = load_centroids(Path(args.features))
    view_ids = [int(x.strip()) for x in args.view_ids.split(",") if x.strip()]

    R_trf_cam = np.array(cfg.robot.cam_axes_correction_R_trf_cam_row_major_3x3, dtype=np.float64).reshape(3, 3)
    t_trf_cam = np.array(cfg.robot.camera_in_trf_translation_mm, dtype=np.float64).reshape(3,) / 1000.0

    print("Using R_trf_cam:\n", R_trf_cam.astype(int))
    print("Using t_trf_cam [m]:", t_trf_cam)

    world = {}
    for vid in view_ids:
        if vid not in cents_cam or vid not in cfg.robot.views:
            continue
        p_cam = cents_cam[vid]
        p_trf = (R_trf_cam @ p_cam) + t_trf_cam

        pose_world_trf = cfg.robot.views[vid].pose_world
        Rw, tw = pose_to_Rt(pose_world_trf)

        p_world = (Rw @ p_trf) + tw
        world[vid] = p_world

    print("\nCentroids in WORLD (m):")
    for vid in view_ids:
        if vid in world:
            print(f"  view {vid}: {world[vid]}")

    print("\nPairwise distances in WORLD:")
    vids = sorted(world.keys())
    for i in range(len(vids)):
        for j in range(i + 1, len(vids)):
            a, b = vids[i], vids[j]
            d = np.linalg.norm(world[a] - world[b])
            print(f"  dist view{a}-view{b} = {d*1000:.1f} mm")


if __name__ == "__main__":
    main()
