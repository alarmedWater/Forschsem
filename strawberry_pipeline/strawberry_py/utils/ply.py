from __future__ import annotations

from pathlib import Path
import numpy as np


def write_ply_xyz(path: Path, points_xyz: np.ndarray, ascii_mode: bool = True) -> None:
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
