#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_indexed_folder_to_plants.py

Input:
  Ein Ordner mit indexierten RGB/Depth-PNGs, z.B.:
    color_0_.png, color_1_.png, color_2_.png, color_3_.png, ...
    depth_0_.png, depth_1_.png, depth_2_.png, depth_3_.png, ...

  Akzeptierte Namensvarianten (robust):
    color0.png, color_0.png, color_0_.png, color-0.png, color__0__.png
    depth0.png, depth_0.png, depth_0_.png, depth_aligned0.png, depth_aligned_0.png, ...

Mapping:
  idx 0,1,2 => plant_000 views 0,1,2 (links/mittig/rechts)
  idx 3,4,5 => plant_001 views 0,1,2
  ...

Output:
  strawberry_pipeline/data/<dataset_name>/plant_000/
    color_0.png color_1.png color_2.png
    depth_0.png depth_1.png depth_2.png

Beispiel:
  cd strawberry_pipeline
  python3 tools/convert_indexed_folder_to_plants.py \
    --src ../Forschsem/Datensatz/massivokkludiert \
    --dst_root data \
    --link_mode symlink

Oder (wenn du von irgendwo startest):
  python3 strawberry_pipeline/tools/convert_indexed_folder_to_plants.py \
    --src Forschsem/Datensatz/massivokkludiert \
    --dst_root strawberry_pipeline/data \
    --link_mode copy
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# --- Regex: tolerant gegenüber _, -, mehrfachen _ und optionalem trailing "_" ---
COLOR_RE = re.compile(r"^color[\W_]*([0-9]+)[\W_]*\.png$", re.IGNORECASE)
# depth oder depth_aligned (aligned wird bevorzugt, falls beides existiert)
DEPTH_RE = re.compile(r"^depth(?:_aligned)?[\W_]*([0-9]+)[\W_]*\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class Pair:
    color: Path
    depth: Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=str, help="Quellordner mit color/depth PNGs (relativ oder absolut)")
    ap.add_argument(
        "--dst_root",
        default="strawberry_pipeline/data",
        type=str,
        help="Ziel-Root. Default: strawberry_pipeline/data (relativ zum cwd)",
    )
    ap.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        help="Optional: Name des Dataset-Ordners unter dst_root. Default: src-Ordnername",
    )
    ap.add_argument("--start_id", default=0, type=int, help="Start plant id (Default: 0 => plant_000)")
    ap.add_argument("--views_per_plant", default=3, type=int, help="Default 3 (links/mittig/rechts)")
    ap.add_argument("--link_mode", default="copy", choices=["copy", "symlink", "hardlink"])
    ap.add_argument("--dry_run", action="store_true", help="Nur ausgeben, nichts schreiben")
    return ap.parse_args()


def _ensure_dir(p: Path, dry_run: bool) -> None:
    if dry_run:
        return
    p.mkdir(parents=True, exist_ok=True)


def _place(src: Path, dst: Path, mode: str, dry_run: bool) -> None:
    if dry_run:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    # existierende Datei/Link entfernen
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        dst.hardlink_to(src)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(mode)


def _scan_indexed_files(src_dir: Path) -> Tuple[Dict[int, Path], Dict[int, Path], Dict[int, Path]]:
    """
    Returns:
      color_by_idx
      depth_by_idx
      depth_aligned_by_idx
    (aligned wird später bevorzugt, falls vorhanden)
    """
    color_by_idx: Dict[int, Path] = {}
    depth_by_idx: Dict[int, Path] = {}
    depth_aligned_by_idx: Dict[int, Path] = {}

    for p in src_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ".png":
            continue

        m = COLOR_RE.match(p.name)
        if m:
            idx = int(m.group(1))
            # bei doppelten Kandidaten: nimm "kürzeren" Namen als stabilen Default
            if idx not in color_by_idx or len(p.name) < len(color_by_idx[idx].name):
                color_by_idx[idx] = p
            continue

        # depth / depth_aligned
        m2 = DEPTH_RE.match(p.name)
        if m2:
            idx = int(m2.group(1))
            if "aligned" in p.name.lower():
                if idx not in depth_aligned_by_idx or len(p.name) < len(depth_aligned_by_idx[idx].name):
                    depth_aligned_by_idx[idx] = p
            else:
                if idx not in depth_by_idx or len(p.name) < len(depth_by_idx[idx].name):
                    depth_by_idx[idx] = p

    return color_by_idx, depth_by_idx, depth_aligned_by_idx


def _build_pairs(
    color_by_idx: Dict[int, Path],
    depth_by_idx: Dict[int, Path],
    depth_aligned_by_idx: Dict[int, Path],
) -> Dict[int, Pair]:
    """
    Für jedes idx muss es color und depth geben.
    Depth bevorzugt: depth_aligned[idx] > depth[idx]
    """
    pairs: Dict[int, Pair] = {}

    all_color = set(color_by_idx.keys())
    all_depth = set(depth_by_idx.keys()) | set(depth_aligned_by_idx.keys())
    all_idx = sorted(all_color | all_depth)

    for idx in all_idx:
        c = color_by_idx.get(idx)
        d = depth_aligned_by_idx.get(idx) or depth_by_idx.get(idx)
        if c is None or d is None:
            continue
        pairs[idx] = Pair(color=c, depth=d)

    return pairs


def main() -> None:
    args = parse_args()

    src_dir = Path(args.src).expanduser().resolve()
    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"--src is not a directory: {src_dir}")

    dst_root = Path(args.dst_root).expanduser().resolve()
    dataset_name = args.dataset_name or src_dir.name
    out_dataset_dir = dst_root / dataset_name

    views_per_plant = int(args.views_per_plant)
    if views_per_plant <= 0:
        raise ValueError("--views_per_plant must be > 0")
    if views_per_plant != 3:
        print(f"[WARN] views_per_plant={views_per_plant}. Dein Use-Case ist 3 (links/mittig/rechts).")

    print("[CONVERT] src:", src_dir)
    print("[CONVERT] dst_root:", dst_root)
    print("[CONVERT] dataset_name:", dataset_name)
    print("[CONVERT] out_dataset_dir:", out_dataset_dir)
    print("[CONVERT] link_mode:", args.link_mode, "| dry_run:", bool(args.dry_run))
    print("[CONVERT] views_per_plant:", views_per_plant, "| start_id:", int(args.start_id))

    _ensure_dir(out_dataset_dir, args.dry_run)

    color_by_idx, depth_by_idx, depth_aligned_by_idx = _scan_indexed_files(src_dir)
    pairs = _build_pairs(color_by_idx, depth_by_idx, depth_aligned_by_idx)

    if not color_by_idx and not depth_by_idx and not depth_aligned_by_idx:
        raise FileNotFoundError(
            f"Keine passenden PNGs gefunden in {src_dir}.\n"
            "Erwartet sowas wie color_0_.png / depth_0_.png (oder depth_aligned...)."
        )

    # Prüfe fehlende Indizes explizit
    all_seen = sorted(set(color_by_idx.keys()) | set(depth_by_idx.keys()) | set(depth_aligned_by_idx.keys()))
    if not all_seen:
        raise FileNotFoundError("Keine indexierten Dateien gefunden.")

    max_idx = max(all_seen)
    n_plants_expected = (max_idx // views_per_plant) + 1

    errors: List[str] = []
    written = 0

    for g in range(n_plants_expected):
        plant_id = int(args.start_id) + g
        plant_dir = out_dataset_dir / f"plant_{plant_id:03d}"

        # Prüfe alle Views in der Gruppe
        missing_this: List[str] = []
        view_items: List[Tuple[int, int, Pair]] = []  # (view, idx, pair)

        for view in range(views_per_plant):
            idx = g * views_per_plant + view
            c = color_by_idx.get(idx)
            d = (depth_aligned_by_idx.get(idx) or depth_by_idx.get(idx))

            if c is None:
                missing_this.append(f"color idx={idx}")
            if d is None:
                missing_this.append(f"depth idx={idx}")

            if c is not None and d is not None:
                view_items.append((view, idx, Pair(color=c, depth=d)))

        # Wenn komplett leer (z.B. am Ende max_idx nicht genau endet), dann überspringen
        if len(view_items) == 0 and len(missing_this) == views_per_plant * 2:
            continue

        if missing_this:
            errors.append(
                f"[plant_{plant_id:03d}] fehlt: {', '.join(missing_this)} "
                f"(erwartet Indizes {g*views_per_plant}..{g*views_per_plant + views_per_plant - 1})"
            )
            continue

        # Schreiben
        _ensure_dir(plant_dir, args.dry_run)
        print(f"\n[PLANT] group={g} -> {plant_dir}")

        for view, idx, pair in view_items:
            rgb_dst = plant_dir / f"color_{view}.png"
            dep_dst = plant_dir / f"depth_{view}.png"

            _place(pair.color, rgb_dst, args.link_mode, args.dry_run)
            _place(pair.depth, dep_dst, args.link_mode, args.dry_run)

            print(
                f"  view {view} (idx={idx}): "
                f"{pair.color.name} -> {rgb_dst.name} | {pair.depth.name} -> {dep_dst.name}"
            )

        written += 1

    print("\n" + "=" * 80)
    print(f"[DONE] wrote {written} plants into: {out_dataset_dir}")
    if errors:
        print("\n[WARN] Es gab Probleme (Views/Indizes fehlen). Details:")
        for e in errors:
            print(" ", e)
        raise SystemExit(2)


if __name__ == "__main__":
    main()