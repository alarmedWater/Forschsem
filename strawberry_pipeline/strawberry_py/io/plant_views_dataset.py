from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import cv2
import numpy as np

from strawberry_py.st_types import (
    PlantId, ViewId, FrameIndex,
    RGBImage, DepthU16,
    FrameInfo, ViewFrame, PlantSample,
    assert_rgb, assert_depth_u16,
)


@dataclass(frozen=True)
class PlantPaths:
    plant_id: PlantId
    rgb_paths: Dict[int, Path]    # view_id -> path
    depth_paths: Dict[int, Path]  # view_id -> path


class PlantViewsDataset:
    def __init__(
        self,
        root: str | Path,
        plant_glob: str = "plant_*",
        rgb_pattern: str = "color_{view}.png",
        depth_pattern: str = "depth_{view}.png",
        view_ids: List[int] | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.plant_glob = str(plant_glob)
        self.rgb_pattern = str(rgb_pattern)
        self.depth_pattern = str(depth_pattern)
        self.view_ids = list(view_ids) if view_ids is not None else [0, 1, 2]

        self._plants: List[PlantPaths] = self._index_plants()

    @staticmethod
    def _parse_plant_id(folder_name: str) -> PlantId:
        m = re.match(r"^plant_(\d+)$", folder_name)
        if not m:
            raise ValueError(f"Cannot parse plant id from '{folder_name}' (expected plant_###)")
        return PlantId(int(m.group(1)))

    def _resolve_view_path(self, plant_dir: Path, pattern: str, view_id: int) -> Path:
        # preferred: template pattern like "color_{view}.png"
        if "{view}" in pattern:
            return plant_dir / pattern.format(view=int(view_id))

        # fallback: glob pattern like "color_*.png" -> try to pick matching view_id by digits in filename
        candidates = sorted(plant_dir.glob(pattern))
        if not candidates:
            return plant_dir / pattern  # will fail later with clearer error

        # map by last number in name
        by_num: Dict[int, Path] = {}
        for p in candidates:
            nums = re.findall(r"\d+", p.stem)
            if nums:
                by_num[int(nums[-1])] = p

        if int(view_id) in by_num:
            return by_num[int(view_id)]

        # final fallback: order-based
        idx = self.view_ids.index(int(view_id)) if int(view_id) in self.view_ids else 0
        idx = max(0, min(idx, len(candidates) - 1))
        return candidates[idx]

    def _index_plants(self) -> List[PlantPaths]:
        plant_dirs = sorted([p for p in self.root.glob(self.plant_glob) if p.is_dir()])
        out: List[PlantPaths] = []

        for pd in plant_dirs:
            pid = self._parse_plant_id(pd.name)

            rgb_paths: Dict[int, Path] = {}
            depth_paths: Dict[int, Path] = {}

            for vid in self.view_ids:
                rgb_path = self._resolve_view_path(pd, self.rgb_pattern, vid)
                dep_path = self._resolve_view_path(pd, self.depth_pattern, vid)
                if not rgb_path.exists():
                    raise FileNotFoundError(f"Missing {rgb_path}")
                if not dep_path.exists():
                    raise FileNotFoundError(f"Missing {dep_path}")
                rgb_paths[int(vid)] = rgb_path
                depth_paths[int(vid)] = dep_path

            out.append(PlantPaths(pid, rgb_paths, depth_paths))

        if not out:
            raise FileNotFoundError(f"No plant folders found in {self.root} (glob='{self.plant_glob}')")
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
        for plant in self._plants:
            views: List[ViewFrame] = []
            for vid in self.view_ids:
                info = FrameInfo(
                    plant_id=plant.plant_id,
                    view_id=ViewId(int(vid)),
                    frame_index=FrameIndex(int(frame_index)),
                    rgb_path=str(plant.rgb_paths[int(vid)]),
                    depth_path=str(plant.depth_paths[int(vid)]),
                )
                rgb = self._load_rgb(plant.rgb_paths[int(vid)])
                depth = self._load_depth(plant.depth_paths[int(vid)])
                views.append(ViewFrame(info=info, rgb=rgb, depth=depth))
                frame_index += 1

            yield PlantSample(plant_id=plant.plant_id, views=tuple(views))
