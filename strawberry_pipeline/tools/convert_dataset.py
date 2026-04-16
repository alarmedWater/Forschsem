from __future__ import annotations

import re
import shutil
import argparse
from pathlib import Path


SAMPLE_DIR_RE = re.compile(r"^\d+(?:_(?:hok|lok))?$")
COLOR_RE = re.compile(r"^color_(\d+)_(\d+)\.png$")
DEPTH_ALIGNED_RE = re.compile(r"^depth_aligned_(\d+)_(\d+)\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Konvertiert alten Datensatz in neues Format "
                    "(color_<image>_<view>.png und depth_<image>_<view>.png)."
    )
    parser.add_argument("src", type=Path, help="Quellordner mit alten Unterordnern (z.B. 45, 45_hok, ...)")
    parser.add_argument("dst", type=Path, help="Zielordner für neue Struktur")
    parser.add_argument(
        "--move",
        action="store_true",
        help="Dateien verschieben statt kopieren (Standard: kopieren)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Vorhandene Zieldateien überschreiben"
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_or_move(src: Path, dst: Path, move: bool, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Zieldatei existiert bereits: {dst}")
        dst.unlink()

    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def validate_view_id(view_id: int, folder: Path, file_name: str) -> None:
    if view_id not in (0, 1, 2):
        raise ValueError(f"Ungültige view_id {view_id} in {folder / file_name} (erwartet: 0,1,2)")


def process_sample_dir(src_dir: Path, dst_dir: Path, move: bool, overwrite: bool) -> None:
    ensure_dir(dst_dir)

    colors_by_image: dict[int, set[int]] = {}
    depths_by_image: dict[int, set[int]] = {}

    for file_path in sorted(src_dir.iterdir()):
        if not file_path.is_file():
            continue

        name = file_path.name

        m_color = COLOR_RE.fullmatch(name)
        if m_color:
            image_id = int(m_color.group(1))
            view_id = int(m_color.group(2))
            validate_view_id(view_id, src_dir, name)

            colors_by_image.setdefault(image_id, set()).add(view_id)
            dst_path = dst_dir / f"color_{image_id}_{view_id}.png"
            copy_or_move(file_path, dst_path, move=move, overwrite=overwrite)
            continue

        m_depth = DEPTH_ALIGNED_RE.fullmatch(name)
        if m_depth:
            image_id = int(m_depth.group(1))
            view_id = int(m_depth.group(2))
            validate_view_id(view_id, src_dir, name)

            depths_by_image.setdefault(image_id, set()).add(view_id)
            dst_path = dst_dir / f"depth_{image_id}_{view_id}.png"
            copy_or_move(file_path, dst_path, move=move, overwrite=overwrite)
            continue

    if not colors_by_image and not depths_by_image:
        print(f"[WARN] Keine passenden color_/depth_aligned_-Dateien in {src_dir}")
        return

    image_ids = sorted(set(colors_by_image.keys()) | set(depths_by_image.keys()))
    for image_id in image_ids:
        color_views = colors_by_image.get(image_id, set())
        depth_views = depths_by_image.get(image_id, set())

        missing_colors = [v for v in (0, 1, 2) if v not in color_views]
        missing_depths = [v for v in (0, 1, 2) if v not in depth_views]

        if missing_colors:
            print(f"[WARN] {src_dir.name}: image_id={image_id} hat fehlende color-Views: {missing_colors}")
        if missing_depths:
            print(f"[WARN] {src_dir.name}: image_id={image_id} hat fehlende depth_aligned-Views: {missing_depths}")

        if not missing_colors and not missing_depths:
            print(f"[OK]   {src_dir.name}: image_id={image_id} vollständig")


def main() -> None:
    args = parse_args()

    src_root: Path = args.src.resolve()
    dst_root: Path = args.dst.resolve()

    if not src_root.exists() or not src_root.is_dir():
        raise FileNotFoundError(f"Quellordner existiert nicht oder ist kein Verzeichnis: {src_root}")

    ensure_dir(dst_root)

    sample_dirs = [p for p in sorted(src_root.iterdir()) if p.is_dir() and SAMPLE_DIR_RE.fullmatch(p.name)]

    if not sample_dirs:
        raise FileNotFoundError(
            f"Keine passenden Unterordner gefunden in {src_root} "
            f"(erwartet z.B. '45', '45_hok', '45_lok')"
        )

    for src_dir in sample_dirs:
        dst_dir = dst_root / src_dir.name
        print(f"\n[INFO] Verarbeite {src_dir.name}")
        process_sample_dir(src_dir, dst_dir, move=args.move, overwrite=args.overwrite)

    print("\nFertig.")


if __name__ == "__main__":
    main()