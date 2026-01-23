#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert_legacy_dataset_to_plant_views.py

Konvertiert dein Legacy-Format in das Pipeline-Format.

Unterstützte Inputs:

(A) Root mit mehreren Runs:
  Datensatz/
    Aufnahme001/
    Aufnahme002/
    ...

(B) Ein einzelner Run-Ordner direkt:
  Datensatz/Aufnahme001/
    color0.png
    color1.png
    color2.png
    depth_aligned0.png
    depth_aligned1.png
    depth_aligned2.png

Output (pro Run -> ein plant_###):
  plant_views_new/plant_000/
    color_0.png, color_1.png, color_2.png
    depth_0.png, depth_1.png, depth_2.png   (inhaltlich = depth_aligned*)

Optional zusätzlich:
    color0.png, color1.png, color2.png
    depth0.png, depth1.png, depth2.png

Beispiel:
  python strawberry_pipeline/tools/convert_to_datasetformat.py \
    --src Datensatz \
    --dst /home/parallels/Documents/Forschsem/strawberry_pipeline/data/plant_views_new \
    --link_mode copy

Oder nur ein Run:
  python strawberry_pipeline/tools/convert_to_datasetformat.py \
    --src Datensatz/Aufnahme001 \
    --dst /home/parallels/Documents/Forschsem/strawberry_pipeline/data/plant_views_new \
    --start_id 0
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="Quelle: Datensatz ODER Datensatz/Aufnahme001")
    ap.add_argument("--dst", type=str, required=True, help="Ziel: .../plant_views_new")
    ap.add_argument("--link_mode", type=str, default="copy", choices=["copy", "symlink", "hardlink"])
    ap.add_argument(
        "--with_no_underscore",
        action="store_true",
        help="zusätzlich color0.png/depth0.png erzeugen (neben color_0.png/depth_0.png)",
    )
    ap.add_argument("--start_id", type=int, default=0, help="Start plant id (default 0 => plant_000)")
    ap.add_argument("--views", type=int, nargs="+", default=[0, 1, 2], help="view ids (default: 0 1 2)")
    ap.add_argument("--dry_run", action="store_true", help="nur ausgeben, nichts schreiben")
    return ap.parse_args()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _is_run_dir(p: Path, views: List[int]) -> bool:
    """Heuristik: enthält mind. color{v}.png und irgendein depth*{v}.png für alle v."""
    for v in views:
        if not (p / f"color{v}.png").exists():
            return False
        if _pick_depth_path(p, v).exists() is False:
            return False
    return True


def _sorted_aufnahme_dirs(src_root: Path) -> List[Path]:
    pat = re.compile(r"^Aufnahme(\d+)$")
    items: List[Tuple[int, Path]] = []
    for p in src_root.iterdir():
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        items.append((int(m.group(1)), p))
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]


def _pick_depth_path(run_dir: Path, view: int) -> Path:
    # 1) bevorzugt aligned (das willst du für eure Pipeline!)
    cand = run_dir / f"depth_aligned{view}.png"
    if cand.exists():
        return cand

    # 2) weitere sinnvolle Fallbacks (falls jemand umbenannt hat)
    for name in (
        f"depth_{view}.png",
        f"depth{view}.png",
        f"depth_aligned_{view}.png",
        f"depth_raw{view}.png",  # letzter Fallback (eigentlich nicht empfohlen)
        f"depth_raw_{view}.png",
    ):
        p = run_dir / name
        if p.exists():
            return p

    return cand  # wird später als missing gemeldet


def _place(src: Path, dst: Path, mode: str, dry_run: bool) -> None:
    if dry_run:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
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


def main() -> None:
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()
    views = [int(v) for v in args.views]

    if not src.exists():
        raise FileNotFoundError(f"src not found: {src}")

    _ensure_dir(dst_root)

    # --- Determine runs ---
    runs: List[Path] = []

    # Case (B): src ist direkt eine Aufnahme### (oder generell ein run-dir)
    if src.is_dir() and _is_run_dir(src, views):
        runs = [src]
    else:
        # Case (A): src ist root mit Aufnahme### Unterordnern
        if not src.is_dir():
            raise ValueError(f"--src must be a directory, got: {src}")
        runs = _sorted_aufnahme_dirs(src)

        # Falls keine Aufnahme### gefunden, evtl. ist src schon ein run-dir, aber Heuristik oben schlug fehl
        if not runs:
            raise FileNotFoundError(
                f"Keine Aufnahme### Ordner gefunden in: {src}\n"
                f"Und src sieht nicht aus wie ein Run-Ordner mit color{views[0]}.png/depth_aligned{views[0]}.png."
            )

    plant_id = int(args.start_id)

    print(f"[CONVERT] src={src}")
    print(f"[CONVERT] dst={dst_root}")
    print(f"[CONVERT] link_mode={args.link_mode}  dry_run={bool(args.dry_run)}")
    print(f"[CONVERT] views={views}")
    print(f"[CONVERT] n_runs={len(runs)}  start_id={plant_id}")

    for run_dir in runs:
        plant_dir = dst_root / f"plant_{plant_id:03d}"
        if not args.dry_run:
            _ensure_dir(plant_dir)

        print(f"\n[RUN] {run_dir}  ->  {plant_dir}")

        for v in views:
            rgb_src = run_dir / f"color{v}.png"
            depth_src = _pick_depth_path(run_dir, v)

            if not rgb_src.exists():
                raise FileNotFoundError(f"Missing RGB: {rgb_src}")
            if not depth_src.exists():
                raise FileNotFoundError(f"Missing DEPTH: {depth_src}")

            # Ziel: underscore-Variante (wie eure plant_views Struktur)
            rgb_dst_u = plant_dir / f"color_{v}.png"
            dep_dst_u = plant_dir / f"depth_{v}.png"

            _place(rgb_src, rgb_dst_u, args.link_mode, args.dry_run)
            _place(depth_src, dep_dst_u, args.link_mode, args.dry_run)

            # Optional zusätzlich ohne underscore
            if args.with_no_underscore:
                rgb_dst_n = plant_dir / f"color{v}.png"
                dep_dst_n = plant_dir / f"depth{v}.png"
                _place(rgb_src, rgb_dst_n, args.link_mode, args.dry_run)
                _place(depth_src, dep_dst_n, args.link_mode, args.dry_run)

            print(f"  view {v}: {rgb_src.name} -> {rgb_dst_u.name} | {depth_src.name} -> {dep_dst_u.name}")

        plant_id += 1

    print(f"\n[DONE] wrote plants: plant_{args.start_id:03d} .. plant_{plant_id-1:03d}")
    print(f"[DONE] output root: {dst_root}")


if __name__ == "__main__":
    main()
