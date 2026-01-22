# strawberry_py/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import numpy as np
import yaml

from strawberry_py.types import CameraIntrinsics, DepthUnit, Pose


# ============================================================
# Typed config model
# ============================================================

@dataclass(frozen=True)
class DepthRangeFilterCfg:
    enabled: bool = True
    min_m: float = 0.05
    max_m: float = 0.60


@dataclass(frozen=True)
class DepthCfg:
    unit: DepthUnit = DepthUnit.REALSENSE_UNITS
    scale_m_per_unit: float = 9.999999747378752e-05
    treat_65535_as_invalid: bool = True
    range_filter: DepthRangeFilterCfg = field(default_factory=DepthRangeFilterCfg)


@dataclass(frozen=True)
class SegmentationCfg:
    model_path: Path
    device: str = "auto"
    imgsz: int = 640
    conf: float = 0.65
    iou: float = 0.50
    max_det: int = 100
    min_mask_area_px: int = 1500
    classes: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class FeaturesCfg:
    downsample_step: int = 1
    min_points: int = 50
    log_features: bool = True


@dataclass(frozen=True)
class SelectedCfg:
    enabled: bool = True
    instance_id: int = 1
    min_pixels: int = 50
    darken_factor: float = 0.3
    draw_bbox: bool = True


@dataclass(frozen=True)
class RawCloudCfg:
    enabled: bool = True
    export_frame: str = "camera"  # "camera" | "trf" | "world" | "all"
    save_once_per_view: bool = True
    overwrite: bool = False
    ply_ascii: bool = True


@dataclass(frozen=True)
class ClusterCfg:
    enabled: bool = True
    distance_threshold_m: float = 0.05
    max_clusters: int = 50
    max_points_per_cluster: int = 200_000

    export_dirname: str = "clusters"
    ply_ascii: bool = True

    export_fused_plant_cloud: bool = True
    fused_filename: str = "plant_fused.ply"
    fused_voxel_size: float = 0.0
    fused_max_points: int = 0  # 0 => unlimited

    # Optional debug knob (your YAML uses it)
    flip_y: bool = False


@dataclass(frozen=True)
class OutputsCfg:
    out_root: Path = Path("outputs")
    save_overlay: bool = True
    save_label_vis: bool = True
    save_depth_mask_preview: bool = True

    raw_cloud: RawCloudCfg = field(default_factory=RawCloudCfg)
    cluster: ClusterCfg = field(default_factory=ClusterCfg)


@dataclass(frozen=True)
class DatasetCfg:
    root: Path
    plant_glob: str = "plant_*"
    rgb_pattern: str = "color_{view}.png"
    depth_pattern: str = "depth_{view}.png"
    expected_views_per_plant: int = 3
    view_ids: List[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass(frozen=True)
class PosesCfg:
    pose_file: Path = Path("")
    default_pose: Pose = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


# ---- Camera meta (optional, informational) ----

@dataclass(frozen=True)
class CameraDeviceCfg:
    name: str = ""
    serial_number: str = ""
    firmware_version: str = ""
    product_line: str = ""


@dataclass(frozen=True)
class StreamModeCfg:
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass(frozen=True)
class IntrinsicsCfg:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str = ""
    distortion_coeffs: List[float] = field(default_factory=list)


@dataclass(frozen=True)
class DepthToColorExtrinsicsCfg:
    rotation_row_major_3x3: Tuple[float, ...] = (
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )
    translation_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CameraMetaCfg:
    timestamp_utc: str = ""
    device: CameraDeviceCfg = field(default_factory=CameraDeviceCfg)
    stream_mode: StreamModeCfg = field(default_factory=StreamModeCfg)
    intrinsics_color: Optional[IntrinsicsCfg] = None
    intrinsics_depth: Optional[IntrinsicsCfg] = None
    extrinsics_depth_to_color: Optional[DepthToColorExtrinsicsCfg] = None


# ---- Robot meta (Meca500 poses) ----

@dataclass(frozen=True)
class RobotViewPoseCfg:
    pose_wrf_mm_deg: Tuple[float, float, float, float, float, float]
    key: str = ""
    pose_world: Pose = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))  # WORLD <- TRF (GetPose)


@dataclass(frozen=True)
class RobotCfg:
    model: str = "Meca500"
    ip: str = ""
    wrf_equals_brf: bool = True

    # purely informational (optional)
    trf_set_during_capture_mm_deg: Tuple[float, float, float, float, float, float] = (
        0.0, 0.0, 36.0, 0.0, 0.0, 45.0
    )

    # IMPORTANT: we use this to interpret pose_wrf_mm_deg
    euler_convention: str = "RzRyRx_deg"

    views: Dict[int, RobotViewPoseCfg] = field(default_factory=dict)

    # CAM->TRF fixed correction (used by runner/clusterer)
    cam_axes_correction_R_trf_cam_row_major_3x3: Optional[Tuple[float, ...]] = None  # len=9
    camera_in_trf_translation_mm: Optional[Tuple[float, float, float]] = None       # len=3


@dataclass(frozen=True)
class AppCfg:
    dataset: DatasetCfg
    camera: CameraIntrinsics
    camera_meta: CameraMetaCfg
    depth: DepthCfg
    robot: RobotCfg

    segmentation: SegmentationCfg
    features: FeaturesCfg
    selected: SelectedCfg
    outputs: OutputsCfg
    poses: PosesCfg


# ============================================================
# Helpers
# ============================================================

def _get(d: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, Mapping) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _as_path(p: str | Path, base_dir: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base_dir / pp).resolve()


def _rotx(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float64)


def _roty(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)


def _rotz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float64)


def _rotmat_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n > 0:
        q /= n
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def _rot_from_euler_convention(rx: float, ry: float, rz: float, conv: str) -> np.ndarray:
    """
    conv examples:
      - "RzRyRx_deg" (your BEST)
      - "RxRyRz_deg"
    We interpret it literally as matrix multiplication order:
      RzRyRx => R = Rz(rz) @ Ry(ry) @ Rx(rx)
    """
    base = conv.strip()
    if base.endswith("_deg"):
        base = base[:-4]

    base = base.replace(" ", "")
    if not base.startswith("R") or len(base) != 6:
        raise ValueError(f"Unsupported euler_convention '{conv}'. Expected e.g. 'RzRyRx_deg'")

    order = [base[1:2], base[3:4], base[5:6]]  # e.g. ["z","y","x"]
    mats: Dict[str, np.ndarray] = {
        "x": _rotx(rx),
        "y": _roty(ry),
        "z": _rotz(rz),
    }

    R = np.eye(3, dtype=np.float64)
    # multiply in the string order: RzRyRx => Rz @ Ry @ Rx
    for ax in order:
        if ax not in mats:
            raise ValueError(f"Invalid axis '{ax}' in euler_convention '{conv}'")
        R = mats[ax] @ R  # left-multiply (keeps the written order correct)
    return R


def _pose_from_meca_pose_mm_deg(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    euler_convention: str,
) -> Pose:
    """
    Pose from Mecademic pose_wrf_mm_deg using configured convention.

    For your case (BEST):
      euler_convention = "RzRyRx_deg"
      => R = Rz(rz) @ Ry(ry) @ Rx(rx)
    """
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    # FIXED: actually use the convention
    R = _rot_from_euler_convention(rx, ry, rz, euler_convention)

    qx, qy, qz, qw = _rotmat_to_quat_xyzw(R)
    t_xyz = (x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0)
    return Pose(t_xyz=t_xyz, q_xyzw=(qx, qy, qz, qw))


def _parse_tuple_floats(v: Any, n: int, name: str) -> Optional[Tuple[float, ...]]:
    if v is None:
        return None
    if not (isinstance(v, list) and len(v) == n):
        raise ValueError(f"{name} must be a list of {n} numbers")
    return tuple(float(x) for x in v)


# ============================================================
# Loader
# ============================================================

def load_config(path: str | Path) -> AppCfg:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    base_dir = cfg_path.parent.resolve()
    raw_any = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw_any, dict):
        raise ValueError("Config YAML must be a mapping at the top level.")
    raw: Mapping[str, Any] = cast(Mapping[str, Any], raw_any)

    # ---------------- Dataset ----------------
    ds_raw = cast(Mapping[str, Any], _get(raw, "dataset", {}) or {})
    root_str = str(ds_raw.get("root", ds_raw.get("plants_root_dir", ""))).strip()
    if not root_str:
        raise ValueError("dataset.root (or dataset.plants_root_dir) must be set.")

    dataset = DatasetCfg(
        root=_as_path(root_str, base_dir),
        plant_glob=str(ds_raw.get("plant_glob", "plant_*")),
        rgb_pattern=str(ds_raw.get("rgb_pattern", "color_{view}.png")),
        depth_pattern=str(ds_raw.get("depth_pattern", "depth_{view}.png")),
        expected_views_per_plant=int(ds_raw.get("expected_views_per_plant", 3)),
        view_ids=list(ds_raw.get("view_ids", [0, 1, 2])),
    )

    # ---------------- Camera meta (optional) ----------------
    cam_candidate = raw_any.get("camera") if isinstance(raw_any, dict) else None
    cam_root: Mapping[str, Any] = cast(Mapping[str, Any], cam_candidate) if isinstance(cam_candidate, dict) else raw

    device_raw = cast(Mapping[str, Any], _get(cam_root, "device", {}) or {})
    stream_raw = cast(Mapping[str, Any], _get(cam_root, "stream_mode", {}) or {})
    intr_raw = cast(Mapping[str, Any], _get(cam_root, "intrinsics", {}) or {})
    extr_raw = cast(Mapping[str, Any], _get(cam_root, "extrinsics.depth_to_color", {}) or {})

    def _parse_intr(one: Any) -> Optional[IntrinsicsCfg]:
        if not isinstance(one, Mapping):
            return None
        try:
            return IntrinsicsCfg(
                width=int(one["width"]),
                height=int(one["height"]),
                fx=float(one["fx"]),
                fy=float(one["fy"]),
                cx=float(one["cx"]),
                cy=float(one["cy"]),
                distortion_model=str(one.get("distortion_model", "")),
                distortion_coeffs=list(one.get("distortion_coeffs", [])) or [],
            )
        except Exception:
            return None

    # support both:
    # camera.intrinsics: {fx,fy,cx,cy,width,height}  (your current YAML)
    # camera.intrinsics.color / depth (legacy)
    intr_color = _parse_intr(_get(intr_raw, "color", None))
    intr_depth = _parse_intr(_get(intr_raw, "depth", None))
    intr_flat = _parse_intr(intr_raw) if (intr_color is None and intr_depth is None) else None

    extr_d2c: Optional[DepthToColorExtrinsicsCfg] = None
    try:
        rot = extr_raw.get("rotation_row_major_3x3", None)
        trans = extr_raw.get("translation_m", None)
        if isinstance(rot, list) and len(rot) == 9 and isinstance(trans, list) and len(trans) == 3:
            extr_d2c = DepthToColorExtrinsicsCfg(
                rotation_row_major_3x3=tuple(float(x) for x in rot),
                translation_m=(float(trans[0]), float(trans[1]), float(trans[2])),
            )
    except Exception:
        extr_d2c = None

    camera_meta = CameraMetaCfg(
        timestamp_utc=str(_get(cam_root, "timestamp_utc", "")) or "",
        device=CameraDeviceCfg(
            name=str(device_raw.get("name", "")),
            serial_number=str(device_raw.get("serial_number", "")),
            firmware_version=str(device_raw.get("firmware_version", "")),
            product_line=str(device_raw.get("product_line", "")),
        ),
        stream_mode=StreamModeCfg(
            width=int(stream_raw.get("width", 640)),
            height=int(stream_raw.get("height", 480)),
            fps=int(stream_raw.get("fps", 30)),
        ),
        intrinsics_color=intr_color,
        intrinsics_depth=intr_depth,
        extrinsics_depth_to_color=extr_d2c,
    )

    active_intr = (
        intr_flat
        or intr_color
        or intr_depth
        or IntrinsicsCfg(width=640, height=480, fx=392.0, fy=392.0, cx=320.0, cy=240.0)
    )
    camera = CameraIntrinsics(
        fx=float(active_intr.fx),
        fy=float(active_intr.fy),
        cx=float(active_intr.cx),
        cy=float(active_intr.cy),
        width=int(active_intr.width),
        height=int(active_intr.height),
    )

    # ---------------- Depth ----------------
    depth_raw = cast(Mapping[str, Any], _get(raw, "depth", {}) or {})
    depth_scale_default = float(_get(cam_root, "depth_scale_m_per_unit", 9.999999747378752e-05))
    dr = cast(Mapping[str, Any], _get(depth_raw, "range_filter", {}) or {})

    unit_str = str(depth_raw.get("unit", "realsense_units")).strip().lower()
    try:
        unit = DepthUnit(unit_str)
    except Exception as exc:
        raise ValueError(f"depth.unit must be one of {[u.value for u in DepthUnit]} (got '{unit_str}')") from exc

    depth = DepthCfg(
        unit=unit,
        scale_m_per_unit=float(depth_raw.get("scale_m_per_unit", depth_scale_default)),
        treat_65535_as_invalid=bool(depth_raw.get("treat_65535_as_invalid", True)),
        range_filter=DepthRangeFilterCfg(
            enabled=bool(dr.get("enabled", True)),
            min_m=float(dr.get("min_m", 0.05)),
            max_m=float(dr.get("max_m", 0.60)),
        ),
    )

    # ---------------- Robot ----------------
    robot_raw = cast(Mapping[str, Any], _get(raw, "robot", {}) or {})
    views_raw = _get(robot_raw, "views", {}) or {}

    # euler convention (THIS is what you want pinned)
    euler_convention = str(robot_raw.get("euler_convention", "RzRyRx_deg")).strip()

    # Parse optional fixed CAM->TRF correction (what runner expects)
    R_trf_cam = _parse_tuple_floats(
        robot_raw.get("cam_axes_correction_R_trf_cam_row_major_3x3", None),
        9,
        "robot.cam_axes_correction_R_trf_cam_row_major_3x3",
    )
    t_trf_cam_mm = _parse_tuple_floats(
        robot_raw.get("camera_in_trf_translation_mm", None),
        3,
        "robot.camera_in_trf_translation_mm",
    )

    trf_tuple = (
        0.0, 0.0, 36.0, 0.0, 0.0, 45.0
    )

    views: Dict[int, RobotViewPoseCfg] = {}
    if isinstance(views_raw, Mapping):
        for vid_str, v in views_raw.items():
            try:
                vid = int(vid_str)
            except Exception:
                continue
            if not isinstance(v, Mapping):
                continue

            pose_list = v.get("pose_wrf_mm_deg", None)
            if not (isinstance(pose_list, list) and len(pose_list) == 6):
                continue

            x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = (float(pose_list[i]) for i in range(6))
            pose_world = _pose_from_meca_pose_mm_deg(
                x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg,
                euler_convention=euler_convention,
            )
            views[vid] = RobotViewPoseCfg(
                pose_wrf_mm_deg=(x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg),
                key=str(v.get("key", "")),
                pose_world=pose_world,
            )

    robot = RobotCfg(
        model=str(robot_raw.get("model", "Meca500")),
        ip=str(robot_raw.get("ip", "")),
        wrf_equals_brf=bool(robot_raw.get("wrf_equals_brf", True)),
        trf_set_during_capture_mm_deg=trf_tuple,
        euler_convention=euler_convention,
        views=views,
        cam_axes_correction_R_trf_cam_row_major_3x3=cast(Optional[Tuple[float, ...]], R_trf_cam),
        camera_in_trf_translation_mm=cast(Optional[Tuple[float, float, float]], t_trf_cam_mm),
    )

    # ---------------- Segmentation ----------------
    seg_raw = cast(Mapping[str, Any], _get(raw, "segmentation", {}) or {})
    model_path_str = str(seg_raw.get("model_path", "")).strip()
    if not model_path_str:
        raise ValueError("segmentation.model_path must be set (e.g. models/best.pt).")
    model_path = _as_path(model_path_str, base_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    segmentation = SegmentationCfg(
        model_path=model_path,
        device=str(seg_raw.get("device", "auto")),
        imgsz=int(seg_raw.get("imgsz", 640)),
        conf=float(seg_raw.get("conf", 0.65)),
        iou=float(seg_raw.get("iou", 0.50)),
        max_det=int(seg_raw.get("max_det", 100)),
        min_mask_area_px=int(seg_raw.get("min_mask_area_px", 1500)),
        classes=list(seg_raw.get("classes", [])) or [],
    )

    # ---------------- Features ----------------
    feat_raw = cast(Mapping[str, Any], _get(raw, "features", {}) or {})
    features = FeaturesCfg(
        downsample_step=int(feat_raw.get("downsample_step", 1)),
        min_points=int(feat_raw.get("min_points", 50)),
        log_features=bool(feat_raw.get("log_features", True)),
    )

    # ---------------- Selected ----------------
    sel_raw = cast(Mapping[str, Any], _get(raw, "selected", {}) or {})
    selected = SelectedCfg(
        enabled=bool(sel_raw.get("enabled", True)),
        instance_id=int(sel_raw.get("instance_id", 1)),
        min_pixels=int(sel_raw.get("min_pixels", 50)),
        darken_factor=float(sel_raw.get("darken_factor", 0.3)),
        draw_bbox=bool(sel_raw.get("draw_bbox", True)),
    )

    # ---------------- Outputs ----------------
    out_raw = cast(Mapping[str, Any], _get(raw, "outputs", {}) or {})

    raw_cloud_raw = cast(Mapping[str, Any], _get(out_raw, "raw_cloud", {}) or {})
    raw_cloud = RawCloudCfg(
        enabled=bool(raw_cloud_raw.get("enabled", True)),
        export_frame=str(raw_cloud_raw.get("export_frame", "camera")).strip().lower(),
        save_once_per_view=bool(raw_cloud_raw.get("save_once_per_view", True)),
        overwrite=bool(raw_cloud_raw.get("overwrite", False)),
        ply_ascii=bool(raw_cloud_raw.get("ply_ascii", True)),
    )
    if raw_cloud.export_frame not in ("camera", "trf", "world", "all"):
        raise ValueError("outputs.raw_cloud.export_frame must be 'camera', 'trf', 'world' or 'all'.")



    cluster_raw = cast(Mapping[str, Any], _get(out_raw, "cluster", {}) or {})
    cluster = ClusterCfg(
        enabled=bool(cluster_raw.get("enabled", True)),
        distance_threshold_m=float(cluster_raw.get("distance_threshold_m", 0.05)),
        max_clusters=int(cluster_raw.get("max_clusters", 50)),
        max_points_per_cluster=int(cluster_raw.get("max_points_per_cluster", 200_000)),
        export_dirname=str(cluster_raw.get("export_dirname", "clusters")),
        ply_ascii=bool(cluster_raw.get("ply_ascii", True)),
        export_fused_plant_cloud=bool(cluster_raw.get("export_fused_plant_cloud", True)),
        fused_filename=str(cluster_raw.get("fused_filename", "plant_fused.ply")),
        fused_voxel_size=float(cluster_raw.get("fused_voxel_size", 0.0)),
        fused_max_points=int(cluster_raw.get("fused_max_points", 0)),
        flip_y=bool(cluster_raw.get("flip_y", False)),
    )

    out_root = _as_path(str(out_raw.get("out_root", "outputs")), base_dir)
    outputs = OutputsCfg(
        out_root=out_root,
        save_overlay=bool(out_raw.get("save_overlay", True)),
        save_label_vis=bool(out_raw.get("save_label_vis", True)),
        save_depth_mask_preview=bool(out_raw.get("save_depth_mask_preview", True)),
        raw_cloud=raw_cloud,
        cluster=cluster,
    )

    # ---------------- Poses (optional) ----------------
    poses_raw = cast(Mapping[str, Any], _get(raw, "poses", {}) or {})
    pose_file_str = str(poses_raw.get("pose_file", "")).strip()
    dp = poses_raw.get("default_pose", {}) if isinstance(poses_raw.get("default_pose", {}), Mapping) else {}
    poses = PosesCfg(
        pose_file=_as_path(pose_file_str, base_dir) if pose_file_str else Path(""),
        default_pose=Pose(
            t_xyz=tuple(dp.get("t_xyz", [0.0, 0.0, 0.0])),
            q_xyzw=tuple(dp.get("q_xyzw", [0.0, 0.0, 0.0, 1.0])),
        ),
    )

    return AppCfg(
        dataset=dataset,
        camera=camera,
        camera_meta=camera_meta,
        depth=depth,
        robot=robot,
        segmentation=segmentation,
        features=features,
        selected=selected,
        outputs=outputs,
        poses=poses,
    )
