from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, NewType, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# ============================================================
# Strong-ish ids
# ============================================================

PlantId = NewType("PlantId", int)
ViewId = NewType("ViewId", int)       # typically 0,1,2 (but can be more)
FrameIndex = NewType("FrameIndex", int)

# ============================================================
# Core array types
# ============================================================

RGBImage = NDArray[np.uint8]          # (H,W,3) RGB
DepthU16 = NDArray[np.uint16]         # (H,W) uint16 (RealSense z16 raw units OR mm)
DepthF32 = NDArray[np.float32]        # (H,W) meters
LabelImage = NDArray[np.uint16]       # (H,W) instance ids (0=background)
Mask = NDArray[np.bool_]              # (H,W)
PointCloud = NDArray[np.float32]      # (N,3) meters


class DepthUnit(str, Enum):
    MM = "mm"
    REALSENSE_UNITS = "realsense_units"


# ============================================================
# Camera / pose
# ============================================================

@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class Pose:
    """
    Pose of camera in world: T_world_cam.

    - t_xyz: translation (meters)
    - q_xyzw: quaternion (x,y,z,w)
    """
    t_xyz: Tuple[float, float, float]
    q_xyzw: Tuple[float, float, float, float]


# ============================================================
# Dataset frames
# ============================================================

@dataclass(frozen=True)
class FrameInfo:
    plant_id: PlantId
    view_id: ViewId
    frame_index: FrameIndex
    rgb_path: str
    depth_path: str

    # Optional metadata, useful for exports / future ROS compatibility
    world_frame_id: str = "world"
    camera_pose_world: Optional[Pose] = None  # T_world_cam

    # Optional "token" timestamp (nanoseconds) for exact sync semantics
    token_ns: Optional[int] = None


@dataclass(frozen=True)
class ViewFrame:
    info: FrameInfo
    rgb: RGBImage
    depth: DepthU16


@dataclass(frozen=True)
class PlantSample:
    """
    One plant with N views (usually N=3 for l/m/r).
    Using a variadic tuple keeps this compatible with cfg.dataset.view_ids.
    """
    plant_id: PlantId
    views: Tuple[ViewFrame, ...]


# ============================================================
# Stage outputs
# ============================================================

@dataclass(frozen=True)
class SegmentationResult:
    label: LabelImage
    overlay_rgb: Optional[RGBImage]
    # Visualization for debugging (mono8 0/255 or colored BGR/RGB depending on implementation)
    label_vis: Optional[NDArray[np.uint8]]


@dataclass(frozen=True)
class DepthMaskResult:
    depth_masked: DepthU16
    valid_range_mask: Mask


@dataclass(frozen=True)
class InstanceFeatures:
    instance_id: int
    num_points: int
    centroid_m: Tuple[float, float, float]
    extent_m: Tuple[float, float, float]
    box_volume_m3: float


@dataclass(frozen=True)
class FeaturesResult:
    # inst_id -> Nx3 (camera frame, meters)
    clouds_by_instance: Dict[int, PointCloud]
    # concatenated points across all kept instances (camera frame, meters)
    all_points: PointCloud
    features: Dict[int, InstanceFeatures]


# ============================================================
# Validation helpers
# ============================================================

def assert_rgb(img: RGBImage) -> None:
    if not isinstance(img, np.ndarray):
        raise TypeError("RGB must be a numpy array")
    if img.dtype != np.uint8:
        raise TypeError(f"RGB dtype must be uint8, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"RGB shape must be (H,W,3), got {img.shape}")


def assert_depth_u16(depth: DepthU16) -> None:
    if not isinstance(depth, np.ndarray):
        raise TypeError("Depth must be a numpy array")
    if depth.dtype != np.uint16:
        raise TypeError(f"Depth dtype must be uint16, got {depth.dtype}")
    if depth.ndim != 2:
        raise ValueError(f"Depth shape must be (H,W), got {depth.shape}")


def assert_label_u16(lbl: LabelImage) -> None:
    if not isinstance(lbl, np.ndarray):
        raise TypeError("Label must be a numpy array")
    if lbl.dtype != np.uint16:
        raise TypeError(f"Label dtype must be uint16, got {lbl.dtype}")
    if lbl.ndim != 2:
        raise ValueError(f"Label shape must be (H,W), got {lbl.shape}")


def assert_mask(mask: Mask) -> None:
    if not isinstance(mask, np.ndarray):
        raise TypeError("Mask must be a numpy array")
    if mask.dtype != np.bool_:
        raise TypeError(f"Mask dtype must be bool, got {mask.dtype}")
    if mask.ndim != 2:
        raise ValueError(f"Mask shape must be (H,W), got {mask.shape}")


def assert_point_cloud(pc: PointCloud) -> None:
    if not isinstance(pc, np.ndarray):
        raise TypeError("Point cloud must be a numpy array")
    if pc.dtype != np.float32:
        raise TypeError(f"Point cloud dtype must be float32, got {pc.dtype}")
    if pc.ndim != 2 or pc.shape[1] != 3:
        raise ValueError(f"Point cloud shape must be (N,3), got {pc.shape}")
