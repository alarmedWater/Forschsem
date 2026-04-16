from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List

import cv2
import numpy as np

from strawberry_py.st_types import (
    PlantId,
    ViewId,
    FrameIndex,
    RGBImage,
    DepthU16,
    FrameInfo,
    ViewFrame,
    PlantSample,
    assert_rgb,
    assert_depth_u16,
)


@dataclass(frozen=True)
class PlantCapturePaths:
    plant_id: PlantId
    sample_name: str              # "1", "1_hok", "1_lok"
    variant: str                  # "base", "hok", "lok"
    capture_id: int               # 1, 2, 3, ...
    source_dir: Path
    rgb_paths: Dict[int, Path]    # view_id -> path
    depth_paths: Dict[int, Path]  # view_id -> path


class PlantViewsDataset:
    def __init__(
        self,
        root: str | Path,
        plant_glob: str = "*",
        rgb_pattern: str = "color_{capture}_{view}.png",
        depth_pattern: str = "depth_{capture}_{view}.png",
        view_ids: List[int] | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.plant_glob = str(plant_glob)
        self.rgb_pattern = str(rgb_pattern)
        self.depth_pattern = str(depth_pattern)
        self.view_ids = list(view_ids) if view_ids is not None else [0, 1, 2]

        self.rgb_prefix = self._extract_prefix(self.rgb_pattern, default="color")
        self.depth_prefix = self._extract_prefix(self.depth_pattern, default="depth")

        self._captures: List[PlantCapturePaths] = self._index_captures()

    @staticmethod
    def _extract_prefix(pattern: str, default: str) -> str:
        token = str(pattern).split("_", 1)[0].strip()
        return token if token else default

    @staticmethod
    def _parse_sample_dir(folder_name: str) -> tuple[PlantId, str, str]:
        """
        Erwartet:
          - 1
          - 1_hok
          - 1_lok
        """
        m = re.fullmatch(r"(\d+)(?:_(hok|lok))?", folder_name)
        if not m:
            raise ValueError(
                f"Cannot parse sample folder '{folder_name}' "
                f"(expected '1', '1_hok' or '1_lok')"
            )

        plant_id = PlantId(int(m.group(1)))
        suffix = m.group(2)
        variant = "base" if suffix is None else suffix
        sample_name = folder_name
        return plant_id, sample_name, variant

    @staticmethod
    def _parse_image_name(file_name: str, prefix: str) -> tuple[int, int] | None:
        """
        Erwartet:
          color_1_0.png
          depth_2_1.png
        Gibt zurück:
          (capture_id, view_id)
        """
        m = re.fullmatch(rf"{re.escape(prefix)}_(\d+)_(\d+)\.png", file_name)
        if not m:
            return None
        capture_id = int(m.group(1))
        view_id = int(m.group(2))
        return capture_id, view_id

    def _index_captures(self) -> List[PlantCapturePaths]:
        candidate_dirs = [p for p in self.root.glob(self.plant_glob) if p.is_dir()]
        out: List[PlantCapturePaths] = []

        variant_order = {"base": 0, "hok": 1, "lok": 2}
        indexed_rows: list[tuple[int, int, int, PlantCapturePaths]] = []

        for sample_dir in candidate_dirs:
            try:
                plant_id, sample_name, variant = self._parse_sample_dir(sample_dir.name)
            except ValueError:
                continue

            rgb_by_capture: Dict[int, Dict[int, Path]] = {}
            depth_by_capture: Dict[int, Dict[int, Path]] = {}

            for file_path in sorted(sample_dir.glob("*.png")):
                rgb_hit = self._parse_image_name(file_path.name, self.rgb_prefix)
                if rgb_hit is not None:
                    capture_id, view_id = rgb_hit
                    rgb_by_capture.setdefault(capture_id, {})
                    if view_id in rgb_by_capture[capture_id]:
                        raise ValueError(f"Duplicate RGB file for capture={capture_id}, view={view_id} in {sample_dir}")
                    rgb_by_capture[capture_id][view_id] = file_path
                    continue

                depth_hit = self._parse_image_name(file_path.name, self.depth_prefix)
                if depth_hit is not None:
                    capture_id, view_id = depth_hit
                    depth_by_capture.setdefault(capture_id, {})
                    if view_id in depth_by_capture[capture_id]:
                        raise ValueError(f"Duplicate depth file for capture={capture_id}, view={view_id} in {sample_dir}")
                    depth_by_capture[capture_id][view_id] = file_path
                    continue

            capture_ids = sorted(set(rgb_by_capture.keys()) | set(depth_by_capture.keys()))
            if not capture_ids:
                raise FileNotFoundError(f"No valid RGB/depth files found in {sample_dir}")

            for capture_id in capture_ids:
                rgb_views = rgb_by_capture.get(capture_id, {})
                depth_views = depth_by_capture.get(capture_id, {})

                missing_rgb = [vid for vid in self.view_ids if vid not in rgb_views]
                missing_depth = [vid for vid in self.view_ids if vid not in depth_views]

                if missing_rgb:
                    raise FileNotFoundError(
                        f"Missing RGB views in {sample_dir} for capture {capture_id}: {missing_rgb}"
                    )

                if missing_depth:
                    raise FileNotFoundError(
                        f"Missing depth views in {sample_dir} for capture {capture_id}: {missing_depth}"
                    )

                capture = PlantCapturePaths(
                    plant_id=plant_id,
                    sample_name=sample_name,
                    variant=variant,
                    capture_id=capture_id,
                    source_dir=sample_dir,
                    rgb_paths={int(vid): rgb_views[int(vid)] for vid in self.view_ids},
                    depth_paths={int(vid): depth_views[int(vid)] for vid in self.view_ids},
                )

                indexed_rows.append(
                    (
                        int(plant_id),
                        variant_order[variant],
                        int(capture_id),
                        capture,
                    )
                )

        indexed_rows.sort(key=lambda x: (x[0], x[1], x[2]))
        out = [row[3] for row in indexed_rows]

        if not out:
            raise FileNotFoundError(
                f"No valid sample folders found in {self.root} "
                f"(expected folders like '1', '1_hok', '1_lok')"
            )

        return out

    @staticmethod
    def _load_rgb(path: Path) -> RGBImage:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Failed to read RGB image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)
        assert_rgb(rgb)
        return rgb

    @staticmethod
    def _load_depth(path: Path) -> DepthU16:
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise IOError(f"Failed to read depth image: {path}")
        if d.dtype != np.uint16:
            raise TypeError(f"Depth must be uint16 PNG, got dtype={d.dtype} at {path}")
        d = d.astype(np.uint16, copy=False)
        assert_depth_u16(d)
        return d

    def iter_plants(self) -> Iterator[PlantSample]:
        frame_index = 0

        for capture in self._captures:
            views: List[ViewFrame] = []

            for vid in self.view_ids:
                info = FrameInfo(
                    plant_id=capture.plant_id,
                    sample_name=capture.sample_name,   # braucht Update in st_types.py
                    variant=capture.variant,           # braucht Update in st_types.py
                    capture_id=capture.capture_id,     # braucht Update in st_types.py
                    view_id=ViewId(int(vid)),
                    frame_index=FrameIndex(int(frame_index)),
                    rgb_path=str(capture.rgb_paths[int(vid)]),
                    depth_path=str(capture.depth_paths[int(vid)]),
                )

                rgb = self._load_rgb(capture.rgb_paths[int(vid)])
                depth = self._load_depth(capture.depth_paths[int(vid)])

                views.append(ViewFrame(info=info, rgb=rgb, depth=depth))
                frame_index += 1

            yield PlantSample(
                plant_id=capture.plant_id,
                sample_name=capture.sample_name,   # braucht Update in st_types.py
                variant=capture.variant,           # braucht Update in st_types.py
                capture_id=capture.capture_id,     # braucht Update in st_types.py
                views=tuple(views),
            )