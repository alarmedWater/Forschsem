# strawberry_py/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import numpy as np
import yaml

from strawberry_py.st_types import CameraIntrinsics, DepthUnit, Pose


# ============================================================
# Typed config model (pragmatic: defaults for non-essential fields)
# ============================================================

@dataclass(frozen=True)
class DatasetCfg:
    root: Path
    plant_glob: str
    rgb_pattern: str
    depth_pattern: str
    expected_views_per_plant: int
    view_ids: List[int]


@dataclass(frozen=True)
class DepthRangeFilterCfg:
    enabled: bool
    min_m: float
    max_m: float


@dataclass(frozen=True)
class DepthCfg:
    unit: DepthUnit
    scale_m_per_unit: float
    treat_65535_as_invalid: bool
    range_filter: DepthRangeFilterCfg


@dataclass(frozen=True)
class MaskPostprocessCfg:
    enabled: bool
    keep_largest_cc: bool
    morph_open: bool
    morph_close: bool
    kernel_size: int
    open_iters: int
    close_iters: int
    fill_holes: bool


@dataclass(frozen=True)
class SegmentationCfg:
    model_path: Path
    device: str
    imgsz: int
    conf: float
    iou: float
    max_det: int
    min_mask_area_px: int
    classes: List[int]

    border_relax_factor: float
    fallback_enabled: bool
    fallback_conf: float
    fallback_imgsz: Optional[int]
    fallback_min_mask_area_px: Optional[int]

    postprocess: MaskPostprocessCfg


@dataclass(frozen=True)
class FeaturesCfg:
    downsample_step: int
    min_points: int
    log_features: bool


@dataclass(frozen=True)
class SelectedCfg:
    enabled: bool
    instance_id: int
    min_pixels: int
    darken_factor: float
    draw_bbox: bool


@dataclass(frozen=True)
class RawCloudCfg:
    enabled: bool
    export_frame: str  # camera|trf|world|all
    save_once_per_view: bool
    overwrite: bool
    ply_ascii: bool


@dataclass(frozen=True)
class ClusterCfg:
    enabled: bool
    distance_threshold_m: float
    max_clusters: int
    max_points_per_cluster: int

    export_dirname: str
    ply_ascii: bool

    export_fused_plant_cloud: bool
    fused_filename: str
    fused_voxel_size: float
    fused_max_points: int

    flip_y: bool


@dataclass(frozen=True)
class OutputsCfg:
    out_root: Path
    save_overlay: bool
    save_label_vis: bool
    save_depth_mask_preview: bool
    raw_cloud: RawCloudCfg
    cluster: ClusterCfg


@dataclass(frozen=True)
class PosesCfg:
    # pose_file is optional now ("" or null -> None)
    pose_file: Optional[Path]
    default_pose: Pose


@dataclass(frozen=True)
class RobotViewPoseCfg:
    pose_wrf_mm_deg: Tuple[float, float, float, float, float, float]
    key: str
    pose_world: Pose  # WORLD <- TRF (aus GetPose / pose_wrf_mm_deg)


@dataclass(frozen=True)
class RobotCfg:
    # minimal, but keeps optional metadata fields without forcing non-empty strings
    model: str
    ip: str
    wrf_equals_brf: bool

    # optional: for documentation only; not required by the pipeline
    trf_set_during_capture_mm_deg: Tuple[float, float, float, float, float, float]

    euler_convention: str
    pose_negate_angles: bool
    pose_negate_translation: bool

    cam_axes_correction_R_trf_cam_row_major_3x3: Tuple[float, ...]  # len=9
    camera_in_trf_translation_mm: Tuple[float, float, float]        # len=3

    views: Dict[int, RobotViewPoseCfg]


@dataclass(frozen=True)
class AppCfg:
    dataset: DatasetCfg
    camera: CameraIntrinsics
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

def _as_path(p: str | Path, base_dir: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base_dir / pp).resolve()


def _odd_kernel(k: int) -> int:
    k = int(k)
    if k < 1:
        k = 1
    if (k % 2) == 0:
        k += 1
    return k


def _check_keys(d: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(d.keys()) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys in '{where}': {unknown}")


def _req_map(d: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    if key not in d:
        raise ValueError(f"Missing required key: {where}.{key}")
    v = d[key]
    if not isinstance(v, Mapping):
        raise ValueError(f"{where}.{key} must be a mapping")
    return cast(Mapping[str, Any], v)


def _req_str(d: Mapping[str, Any], key: str, where: str) -> str:
    if key not in d:
        raise ValueError(f"Missing required key: {where}.{key}")
    v = d[key]
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{where}.{key} must be a non-empty string")
    return v.strip()


def _opt_str(
    d: Mapping[str, Any],
    key: str,
    where: str,
    default: str,
    *,
    allow_empty: bool = True,
) -> str:
    if key not in d:
        return default
    v = d[key]
    if v is None:
        return default
    if not isinstance(v, str):
        raise ValueError(f"{where}.{key} must be a string or null")
    s = v.strip()
    if (not s) and (not allow_empty):
        return default
    return s


def _req_bool(d: Mapping[str, Any], key: str, where: str) -> bool:
    if key not in d:
        raise ValueError(f"Missing required key: {where}.{key}")
    v = d[key]
    if not isinstance(v, bool):
        raise ValueError(f"{where}.{key} must be boolean")
    return bool(v)


def _opt_bool(d: Mapping[str, Any], key: str, where: str, default: bool) -> bool:
    if key not in d:
        return bool(default)
    v = d[key]
    if v is None:
        return bool(default)
    if not isinstance(v, bool):
        raise ValueError(f"{where}.{key} must be boolean or null")
    return bool(v)


def _req_int(d: Mapping[str, Any], key: str, where: str) -> int:
    if key not in d:
        raise ValueError(f"Missing required key: {where}.{key}")
    v = d[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{where}.{key} must be int")
    return int(v)


def _opt_int(d: Mapping[str, Any], key: str, where: str) -> Optional[int]:
    # keep old behavior: key must exist; use null for None
    if key not in d:
        raise ValueError(f"Missing required key: {where}.{key} (use null if you want None)")
    v = d[key]
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{where}.{key} must be int or null")
    return int(v)


def _req_float(d: Mapping[str, Any], key: str, where: str) -> float:
    if key not in d:
        raise ValueError(f"Missing required key: {where}.{key}")
    v = d[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{where}.{key} must be float")
    return float(v)


def _req_list_len(d: Mapping[str, Any], key: str, n: int, where: str) -> List[Any]:
    if key not in d:
        raise ValueError(f"Missing required key: {where}.{key}")
    v = d[key]
    if not isinstance(v, list) or len(v) != n:
        raise ValueError(f"{where}.{key} must be a list of length {n}")
    return v


def _opt_list_len(
    d: Mapping[str, Any],
    key: str,
    n: int,
    where: str,
    default: List[float],
) -> List[float]:
    if key not in d or d[key] is None:
        if len(default) != n:
            raise ValueError(f"Internal default for {where}.{key} must have length {n}")
        return [float(x) for x in default]
    v = d[key]
    if not isinstance(v, list) or len(v) != n:
        raise ValueError(f"{where}.{key} must be a list of length {n} or null")
    return [float(x) for x in v]


def _opt_path_allow_empty(
    d: Mapping[str, Any],
    key: str,
    where: str,
    base_dir: Path,
) -> Optional[Path]:
    # accepts: missing -> None, null -> None, "" -> None, "relative/or/abs" -> Path
    if key not in d:
        return None
    v = d[key]
    if v is None:
        return None
    if not isinstance(v, str):
        raise ValueError(f"{where}.{key} must be string|null")
    s = v.strip()
    if not s:
        return None
    return _as_path(s, base_dir)


def _validate_rotation_matrix(R: np.ndarray, name: str) -> None:
    R = np.asarray(R, dtype=np.float64).reshape((3, 3))
    I = np.eye(3, dtype=np.float64)
    err = float(np.linalg.norm(R.T @ R - I, ord="fro"))
    det = float(np.linalg.det(R))
    if err > 1e-6 or abs(det - 1.0) > 1e-6:
        raise ValueError(f"Invalid rotation matrix {name}: ortho_err={err:.3e}, det={det:.6f}")


def _rotx(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _roty(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rotz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


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
    base = conv.strip()
    if base.endswith("_deg"):
        base = base[:-4]
    base = base.replace(" ", "")
    if not base.startswith("R") or len(base) != 6:
        raise ValueError(f"Unsupported euler_convention '{conv}' (expected e.g. 'RzRyRx_deg')")
    order = [base[1:2], base[3:4], base[5:6]]  # e.g. ["z","y","x"]
    mats: Dict[str, np.ndarray] = {"x": _rotx(rx), "y": _roty(ry), "z": _rotz(rz)}
    try:
        return mats[order[0]] @ mats[order[1]] @ mats[order[2]]
    except KeyError as exc:
        raise ValueError(f"Invalid axis in euler_convention '{conv}'") from exc


def _pose_from_meca_pose_mm_deg(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    euler_convention: str,
    negate_angles: bool,
    negate_translation: bool,
) -> Pose:
    if negate_angles:
        rx_deg, ry_deg, rz_deg = -rx_deg, -ry_deg, -rz_deg
    if negate_translation:
        x_mm, y_mm, z_mm = -x_mm, -y_mm, -z_mm

    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    R = _rot_from_euler_convention(rx, ry, rz, euler_convention)
    qx, qy, qz, qw = _rotmat_to_quat_xyzw(R)
    t_xyz = (x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0)
    return Pose(t_xyz=t_xyz, q_xyzw=(qx, qy, qz, qw))


# ============================================================
# Loader
# ============================================================

def load_config(path: str | Path) -> AppCfg:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    base_dir = cfg_path.parent.resolve()
    raw_any = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw_any, Mapping):
        raise ValueError("Config YAML must be a mapping at the top level.")
    raw = cast(Mapping[str, Any], raw_any)

    _check_keys(
        raw,
        allowed={"dataset", "camera", "depth", "segmentation", "features", "selected", "poses", "robot", "outputs"},
        where="root",
    )

    # -------- dataset (required) --------
    ds = _req_map(raw, "dataset", "root")
    _check_keys(ds, {"root", "plant_glob", "rgb_pattern", "depth_pattern", "expected_views_per_plant", "view_ids"}, "dataset")

    view_ids_raw = ds.get("view_ids", None)
    if not isinstance(view_ids_raw, list) or len(view_ids_raw) == 0:
        raise ValueError("dataset.view_ids must be a non-empty list of ints")

    dataset = DatasetCfg(
        root=_as_path(_req_str(ds, "root", "dataset"), base_dir),
        plant_glob=_req_str(ds, "plant_glob", "dataset"),
        rgb_pattern=_req_str(ds, "rgb_pattern", "dataset"),
        depth_pattern=_req_str(ds, "depth_pattern", "dataset"),
        expected_views_per_plant=_req_int(ds, "expected_views_per_plant", "dataset"),
        view_ids=[int(x) for x in view_ids_raw],
    )

    # -------- camera intrinsics (required) --------
    cam = _req_map(raw, "camera", "root")
    _check_keys(cam, {"intrinsics"}, "camera")
    intr = _req_map(cam, "intrinsics", "camera")
    _check_keys(intr, {"width", "height", "fx", "fy", "cx", "cy"}, "camera.intrinsics")

    camera = CameraIntrinsics(
        width=_req_int(intr, "width", "camera.intrinsics"),
        height=_req_int(intr, "height", "camera.intrinsics"),
        fx=_req_float(intr, "fx", "camera.intrinsics"),
        fy=_req_float(intr, "fy", "camera.intrinsics"),
        cx=_req_float(intr, "cx", "camera.intrinsics"),
        cy=_req_float(intr, "cy", "camera.intrinsics"),
    )

    # -------- depth (required) --------
    dep = _req_map(raw, "depth", "root")
    _check_keys(dep, {"unit", "scale_m_per_unit", "treat_65535_as_invalid", "range_filter"}, "depth")
    rf = _req_map(dep, "range_filter", "depth")
    _check_keys(rf, {"enabled", "min_m", "max_m"}, "depth.range_filter")

    unit_str = _req_str(dep, "unit", "depth").lower()
    try:
        unit = DepthUnit(unit_str)
    except Exception as exc:
        raise ValueError(f"depth.unit must be one of {[u.value for u in DepthUnit]} (got '{unit_str}')") from exc

    depth = DepthCfg(
        unit=unit,
        scale_m_per_unit=_req_float(dep, "scale_m_per_unit", "depth"),
        treat_65535_as_invalid=_req_bool(dep, "treat_65535_as_invalid", "depth"),
        range_filter=DepthRangeFilterCfg(
            enabled=_req_bool(rf, "enabled", "depth.range_filter"),
            min_m=_req_float(rf, "min_m", "depth.range_filter"),
            max_m=_req_float(rf, "max_m", "depth.range_filter"),
        ),
    )

    # -------- segmentation (required) --------
    seg = _req_map(raw, "segmentation", "root")
    _check_keys(
        seg,
        {
            "model_path",
            "device",
            "imgsz",
            "conf",
            "iou",
            "max_det",
            "min_mask_area_px",
            "classes",
            "border_relax_factor",
            "fallback_enabled",
            "fallback_conf",
            "fallback_imgsz",
            "fallback_min_mask_area_px",
            "postprocess",
        },
        "segmentation",
    )

    model_path = _as_path(_req_str(seg, "model_path", "segmentation"), base_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    classes_v = seg["classes"]
    if not isinstance(classes_v, list):
        raise ValueError("segmentation.classes must be a list (can be empty: [])")
    classes = [int(x) for x in classes_v]

    pp = _req_map(seg, "postprocess", "segmentation")
    _check_keys(
        pp,
        {"enabled", "keep_largest_cc", "morph_open", "morph_close", "kernel_size", "open_iters", "close_iters", "fill_holes"},
        "segmentation.postprocess",
    )

    postprocess = MaskPostprocessCfg(
        enabled=_req_bool(pp, "enabled", "segmentation.postprocess"),
        keep_largest_cc=_req_bool(pp, "keep_largest_cc", "segmentation.postprocess"),
        morph_open=_req_bool(pp, "morph_open", "segmentation.postprocess"),
        morph_close=_req_bool(pp, "morph_close", "segmentation.postprocess"),
        kernel_size=_odd_kernel(_req_int(pp, "kernel_size", "segmentation.postprocess")),
        open_iters=_req_int(pp, "open_iters", "segmentation.postprocess"),
        close_iters=_req_int(pp, "close_iters", "segmentation.postprocess"),
        fill_holes=_req_bool(pp, "fill_holes", "segmentation.postprocess"),
    )

    segmentation = SegmentationCfg(
        model_path=model_path,
        device=_req_str(seg, "device", "segmentation"),
        imgsz=_req_int(seg, "imgsz", "segmentation"),
        conf=_req_float(seg, "conf", "segmentation"),
        iou=_req_float(seg, "iou", "segmentation"),
        max_det=_req_int(seg, "max_det", "segmentation"),
        min_mask_area_px=_req_int(seg, "min_mask_area_px", "segmentation"),
        classes=classes,
        border_relax_factor=_req_float(seg, "border_relax_factor", "segmentation"),
        fallback_enabled=_req_bool(seg, "fallback_enabled", "segmentation"),
        fallback_conf=_req_float(seg, "fallback_conf", "segmentation"),
        fallback_imgsz=_opt_int(seg, "fallback_imgsz", "segmentation"),
        fallback_min_mask_area_px=_opt_int(seg, "fallback_min_mask_area_px", "segmentation"),
        postprocess=postprocess,
    )

    # -------- features (required) --------
    feat = _req_map(raw, "features", "root")
    _check_keys(feat, {"downsample_step", "min_points", "log_features"}, "features")

    features = FeaturesCfg(
        downsample_step=_req_int(feat, "downsample_step", "features"),
        min_points=_req_int(feat, "min_points", "features"),
        log_features=_req_bool(feat, "log_features", "features"),
    )

    # -------- selected (required) --------
    sel = _req_map(raw, "selected", "root")
    _check_keys(sel, {"enabled", "instance_id", "min_pixels", "darken_factor", "draw_bbox"}, "selected")

    selected = SelectedCfg(
        enabled=_req_bool(sel, "enabled", "selected"),
        instance_id=_req_int(sel, "instance_id", "selected"),
        min_pixels=_req_int(sel, "min_pixels", "selected"),
        darken_factor=_req_float(sel, "darken_factor", "selected"),
        draw_bbox=_req_bool(sel, "draw_bbox", "selected"),
    )

    # -------- poses (optional) --------
    if "poses" in raw and raw["poses"] is not None:
        po = cast(Mapping[str, Any], _req_map(raw, "poses", "root"))
        _check_keys(po, {"pose_file", "default_pose"}, "poses")
        dp = _req_map(po, "default_pose", "poses")
        _check_keys(dp, {"t_xyz", "q_xyzw"}, "poses.default_pose")

        t_xyz_list = _req_list_len(dp, "t_xyz", 3, "poses.default_pose")
        q_list = _req_list_len(dp, "q_xyzw", 4, "poses.default_pose")

        pose_file = _opt_path_allow_empty(po, "pose_file", "poses", base_dir)
        poses = PosesCfg(
            pose_file=pose_file,
            default_pose=Pose(
                t_xyz=(float(t_xyz_list[0]), float(t_xyz_list[1]), float(t_xyz_list[2])),
                q_xyzw=(float(q_list[0]), float(q_list[1]), float(q_list[2]), float(q_list[3])),
            ),
        )
    else:
        poses = PosesCfg(
            pose_file=None,
            default_pose=Pose(t_xyz=(0.0, 0.0, 0.0), q_xyzw=(0.0, 0.0, 0.0, 1.0)),
        )

    # -------- robot (required, but with defaults for non-essential fields) --------
    rb = _req_map(raw, "robot", "root")
    _check_keys(
        rb,
        {
            "model",
            "ip",
            "wrf_equals_brf",
            "trf_set_during_capture_mm_deg",
            "euler_convention",
            "pose_negate_angles",
            "pose_negate_translation",
            "cam_axes_correction_R_trf_cam_row_major_3x3",
            "camera_in_trf_translation_mm",
            "views",
        },
        "robot",
    )

    # optional metadata (allow empty)
    model = _opt_str(rb, "model", "robot", default="", allow_empty=True)
    ip = _opt_str(rb, "ip", "robot", default="", allow_empty=True)
    wrf_equals_brf = _opt_bool(rb, "wrf_equals_brf", "robot", default=True)

    # defaults (if missing): not required by pipeline, but keep recorded if present
    trf_list = _opt_list_len(rb, "trf_set_during_capture_mm_deg", 6, "robot", default=[0, 0, 0, 0, 0, 0])

    euler_convention = _opt_str(rb, "euler_convention", "robot", default="RzRyRx_deg", allow_empty=False)
    pose_negate_angles = _opt_bool(rb, "pose_negate_angles", "robot", default=False)
    pose_negate_translation = _opt_bool(rb, "pose_negate_translation", "robot", default=False)

    # cam in trf: default identity/zero if omitted
    R_list = _opt_list_len(rb, "cam_axes_correction_R_trf_cam_row_major_3x3", 9, "robot", default=[1, 0, 0, 0, 1, 0, 0, 0, 1])
    t_list = _opt_list_len(rb, "camera_in_trf_translation_mm", 3, "robot", default=[0, 0, 0])

    R_trf_cam = tuple(float(x) for x in R_list)
    _validate_rotation_matrix(
        np.asarray(R_trf_cam, dtype=np.float64).reshape((3, 3)),
        "robot.cam_axes_correction_R_trf_cam_row_major_3x3",
    )

    views_raw = _req_map(rb, "views", "robot")
    views: Dict[int, RobotViewPoseCfg] = {}

    for vid in dataset.view_ids:
        vid_s = str(int(vid))
        if vid_s not in views_raw:
            raise ValueError(f"Missing robot.views['{vid_s}'] for dataset.view_ids={dataset.view_ids}")
        v_any = views_raw[vid_s]
        if not isinstance(v_any, Mapping):
            raise ValueError(f"robot.views['{vid_s}'] must be a mapping")
        v = cast(Mapping[str, Any], v_any)
        _check_keys(v, {"key", "pose_wrf_mm_deg"}, f"robot.views['{vid_s}']")

        pose_list = v.get("pose_wrf_mm_deg", None)
        if not (isinstance(pose_list, list) and len(pose_list) == 6):
            raise ValueError(f"robot.views['{vid_s}'].pose_wrf_mm_deg must be list length 6")

        x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = (float(pose_list[i]) for i in range(6))
        pose_world = _pose_from_meca_pose_mm_deg(
            x_mm,
            y_mm,
            z_mm,
            rx_deg,
            ry_deg,
            rz_deg,
            euler_convention=euler_convention,
            negate_angles=pose_negate_angles,
            negate_translation=pose_negate_translation,
        )

        views[int(vid)] = RobotViewPoseCfg(
            pose_wrf_mm_deg=(x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg),
            key=_req_str(v, "key", f"robot.views['{vid_s}']"),
            pose_world=pose_world,
        )

    robot = RobotCfg(
        model=model,
        ip=ip,
        wrf_equals_brf=wrf_equals_brf,
        trf_set_during_capture_mm_deg=tuple(float(x) for x in trf_list),  # type: ignore[arg-type]
        euler_convention=euler_convention,
        pose_negate_angles=pose_negate_angles,
        pose_negate_translation=pose_negate_translation,
        cam_axes_correction_R_trf_cam_row_major_3x3=R_trf_cam,
        camera_in_trf_translation_mm=(float(t_list[0]), float(t_list[1]), float(t_list[2])),
        views=views,
    )

    # -------- outputs (required) --------
    out = _req_map(raw, "outputs", "root")
    _check_keys(out, {"out_root", "save_overlay", "save_label_vis", "save_depth_mask_preview", "raw_cloud", "cluster"}, "outputs")

    rc = _req_map(out, "raw_cloud", "outputs")
    _check_keys(rc, {"enabled", "export_frame", "save_once_per_view", "overwrite", "ply_ascii"}, "outputs.raw_cloud")

    export_frame = _req_str(rc, "export_frame", "outputs.raw_cloud").lower()
    if export_frame not in ("camera", "trf", "world", "all"):
        raise ValueError("outputs.raw_cloud.export_frame must be one of: camera|trf|world|all")

    raw_cloud = RawCloudCfg(
        enabled=_req_bool(rc, "enabled", "outputs.raw_cloud"),
        export_frame=export_frame,
        save_once_per_view=_req_bool(rc, "save_once_per_view", "outputs.raw_cloud"),
        overwrite=_req_bool(rc, "overwrite", "outputs.raw_cloud"),
        ply_ascii=_req_bool(rc, "ply_ascii", "outputs.raw_cloud"),
    )

    cl = _req_map(out, "cluster", "outputs")
    _check_keys(
        cl,
        {
            "enabled",
            "distance_threshold_m",
            "max_clusters",
            "max_points_per_cluster",
            "export_dirname",
            "ply_ascii",
            "export_fused_plant_cloud",
            "fused_filename",
            "fused_voxel_size",
            "fused_max_points",
            "flip_y",
        },
        "outputs.cluster",
    )

    export_dirname = _opt_str(cl, "export_dirname", "outputs.cluster", default="clusters", allow_empty=False)

    cluster = ClusterCfg(
        enabled=_req_bool(cl, "enabled", "outputs.cluster"),
        distance_threshold_m=_req_float(cl, "distance_threshold_m", "outputs.cluster"),
        max_clusters=_req_int(cl, "max_clusters", "outputs.cluster"),
        max_points_per_cluster=_req_int(cl, "max_points_per_cluster", "outputs.cluster"),
        export_dirname=export_dirname,
        ply_ascii=_req_bool(cl, "ply_ascii", "outputs.cluster"),
        export_fused_plant_cloud=_req_bool(cl, "export_fused_plant_cloud", "outputs.cluster"),
        fused_filename=_req_str(cl, "fused_filename", "outputs.cluster"),
        fused_voxel_size=_req_float(cl, "fused_voxel_size", "outputs.cluster"),
        fused_max_points=_req_int(cl, "fused_max_points", "outputs.cluster"),
        flip_y=_req_bool(cl, "flip_y", "outputs.cluster"),
    )

    outputs = OutputsCfg(
        out_root=_as_path(_req_str(out, "out_root", "outputs"), base_dir),
        save_overlay=_req_bool(out, "save_overlay", "outputs"),
        save_label_vis=_req_bool(out, "save_label_vis", "outputs"),
        save_depth_mask_preview=_req_bool(out, "save_depth_mask_preview", "outputs"),
        raw_cloud=raw_cloud,
        cluster=cluster,
    )

    return AppCfg(
        dataset=dataset,
        camera=camera,
        depth=depth,
        robot=robot,
        segmentation=segmentation,
        features=features,
        selected=selected,
        outputs=outputs,
        poses=poses,
    )
