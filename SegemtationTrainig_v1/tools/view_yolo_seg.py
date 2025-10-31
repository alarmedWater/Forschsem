#!/usr/bin/env python3
import os
import glob
import random
import cv2
import numpy as np
from pathlib import Path

# ---- Pfade robust setzen ----
# Basis ist dein Projektordner:
PROJECT_ROOT = Path("SegemtationTrainig_v1").resolve()

# Primär: kompletter YOLO-Export
YOLO_BASE = PROJECT_ROOT / "converted" / "yolo" / "val"

# Fallback: CPU-Sanity-Subset
YOLO_SANITY_BASE = PROJECT_ROOT / "converted" / "yolo" / "_cpu_sanity" / "val"

def pick_split_base() -> Path:
    # Nimm val/, wenn vorhanden, sonst _cpu_sanity/val
    cand1 = YOLO_BASE
    cand2 = YOLO_SANITY_BASE
    if (cand1 / "images").exists() and (cand1 / "labels").exists():
        return cand1
    if (cand2 / "images").exists() and (cand2 / "labels").exists():
        print("[INFO] Using CPU-sanity subset.")
        return cand2
    raise FileNotFoundError(
        f"Kein val-Split gefunden unter:\n  {cand1}\n  {cand2}\n"
        "Bitte Converter laufen lassen oder Pfade anpassen."
    )

def main():
    base = pick_split_base()
    img_dir = base / "images"
    lbl_dir = base / "labels"
    out_dir = PROJECT_ROOT / "debug_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Wie viele zufällig?
    N = 3
    random.seed(42)

    # Alle Bilder (auch Großbuchstaben)
    exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    img_paths = []
    for e in exts:
        img_paths.extend(sorted(img_dir.glob(e)))

    # Nur Bilder mit passendem Label
    img_paths_with_lbl = []
    for p in img_paths:
        name = p.stem
        if (lbl_dir / f"{name}.txt").exists():
            img_paths_with_lbl.append(p)

    print(f"[INFO] img_dir: {img_dir}")
    print(f"[INFO] lbl_dir: {lbl_dir}")
    print(f"[INFO] Bilder insgesamt: {len(img_paths)}")
    print(f"[INFO] mit Label:          {len(img_paths_with_lbl)}")

    if not img_paths_with_lbl:
        print("[WARN] Keine Bild/Label-Paare gefunden. Prüfe, ob die Dateinamen exakt matchen.")
        return

    pick = random.sample(img_paths_with_lbl, k=min(N, len(img_paths_with_lbl)))

    for img_path in pick:
        name = img_path.stem
        lbl_path = lbl_dir / f"{name}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Konnte Bild nicht lesen: {img_path}")
            continue
        H, W = img.shape[:2]

        # Labels parsen (YOLO-Seg: class x1 y1 x2 y2 ...)
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                vals = [float(v) for v in line.strip().split()]
                if not vals:
                    continue
                cls = int(vals[0])
                coords = vals[1:]
                if len(coords) < 6 or len(coords) % 2 != 0:
                    # mind. 3 Punkte und gerade Anzahl
                    continue
                pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] *= W
                pts[:, 1] *= H
                pts = pts.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        out_path = out_dir / f"{name}.png"
        ok = cv2.imwrite(str(out_path), img)
        if ok:
            print("wrote:", out_path)
        else:
            print("[WARN] Konnte nicht schreiben:", out_path)

if __name__ == "__main__":
    main()
