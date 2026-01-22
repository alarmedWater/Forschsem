#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import csv
import argparse

def load_features(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--view-ids", default="0,1,2")
    ap.add_argument("--instance-id", type=int, default=1)
    args = ap.parse_args()

    rows = load_features(Path(args.features))
    vids = [int(x) for x in args.view_ids.split(",") if x.strip()]

    cents = {}
    for row in rows:
        vid = int(row["view_id"])
        iid = int(row["instance_id"])
        if vid not in vids or iid != args.instance_id:
            continue
        c = np.array([float(row["cx"]), float(row["cy"]), float(row["cz"])], dtype=float)
        cents[vid] = c

    print("Centroids in CAM (m):")
    for vid in vids:
        print(vid, cents.get(vid))

    # Distanzen im CAM (nur um zu sehen, ob das Objekt sehr unterschiedlich segmentiert ist)
    for i in range(len(vids)):
        for j in range(i+1, len(vids)):
            a = cents.get(vids[i]); b = cents.get(vids[j])
            if a is None or b is None: continue
            print(f"dist CAM view{vids[i]}-view{vids[j]} = {np.linalg.norm(a-b)*1000:.1f} mm")

if __name__ == "__main__":
    main()
