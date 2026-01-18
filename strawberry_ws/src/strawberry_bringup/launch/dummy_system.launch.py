#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch file for the strawberry demo pipeline (dataset playback).

Konvention (Pipeline):
- QoS: qos_profile_sensor_data (BEST_EFFORT)
- Sync: TimeSynchronizer (exakt gleiche stamps) in seg_ultra + depth_mask
- Token stamp: FrameInfo.source_stamp (fallback: FrameInfo.header.stamp)
- Outputs übernehmen immer token_stamp

Nodes:
- camera_folder: plays RGB + depth images from folders as a fake camera
- seg_ultra: YOLOv8 instance segmentation on RGB images + FrameInfo passthrough
- depth_mask: applies instance mask to depth image + FrameInfo passthrough
- strawberry_features: per-instance 3D features + point clouds
- strawberry_selected_overlay (optional)
- raw_cloud_saver (optional, default on)
- strawberry_cluster (optional, default off)
"""

from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    home = os.path.expanduser("~")

    # ---------------- Launch arguments (raw) ----------------
    plants_root_dir = LaunchConfiguration("plants_root_dir")
    plant_glob = LaunchConfiguration("plant_glob")
    use_plants_root = LaunchConfiguration("use_plants_root")
    rgb_pattern = LaunchConfiguration("rgb_pattern")
    depth_pattern = LaunchConfiguration("depth_pattern")

    rgb_dir = LaunchConfiguration("rgb_dir")
    depth_dir = LaunchConfiguration("depth_dir")

    fps = LaunchConfiguration("fps")
    loop = LaunchConfiguration("loop")

    publish_depth = LaunchConfiguration("publish_depth")
    publish_pose = LaunchConfiguration("publish_pose")

    pose_topic = LaunchConfiguration("pose_topic")
    world_frame_id = LaunchConfiguration("world_frame_id")

    publish_frame_info = LaunchConfiguration("publish_frame_info")
    frame_info_topic = LaunchConfiguration("frame_info_topic")

    model_path = LaunchConfiguration("model_path")
    publish_overlay = LaunchConfiguration("publish_overlay")

    sync_queue_size = LaunchConfiguration("sync_queue_size")
    sync_slop = LaunchConfiguration("sync_slop")  # kept for backward compat (unused by exact sync nodes)

    depth_unit = LaunchConfiguration("depth_unit")
    depth_scale_m_per_unit = LaunchConfiguration("depth_scale_m_per_unit")

    selected_instance_id = LaunchConfiguration("selected_instance_id")
    enable_selected_overlay = LaunchConfiguration("enable_selected_overlay")

    # Output dirs
    cluster_output_dir = LaunchConfiguration("cluster_output_dir")
    raw_output_dir = LaunchConfiguration("raw_output_dir")

    # raw_cloud_saver behavior
    raw_save_once_per_view = LaunchConfiguration("raw_save_once_per_view")
    raw_overwrite = LaunchConfiguration("raw_overwrite")
    raw_assume_cloud_in = LaunchConfiguration("raw_assume_cloud_in")   # "camera"|"world"
    raw_export_cloud_in = LaunchConfiguration("raw_export_cloud_in")   # "world"|"camera"
    raw_ply_ascii = LaunchConfiguration("raw_ply_ascii")

    # NEW toggles
    enable_raw_cloud_saver = LaunchConfiguration("enable_raw_cloud_saver")
    enable_cluster = LaunchConfiguration("enable_cluster")

    # ---------------- Typed launch configs (IMPORTANT) ----------------
    use_plants_root_t = ParameterValue(use_plants_root, value_type=bool)

    fps_t = ParameterValue(fps, value_type=float)
    loop_t = ParameterValue(loop, value_type=bool)

    publish_depth_t = ParameterValue(publish_depth, value_type=bool)
    publish_pose_t = ParameterValue(publish_pose, value_type=bool)

    publish_frame_info_t = ParameterValue(publish_frame_info, value_type=bool)
    publish_overlay_t = ParameterValue(publish_overlay, value_type=bool)

    sync_queue_size_t = ParameterValue(sync_queue_size, value_type=int)
    sync_slop_t = ParameterValue(sync_slop, value_type=float)

    depth_scale_t = ParameterValue(depth_scale_m_per_unit, value_type=float)
    selected_instance_id_t = ParameterValue(selected_instance_id, value_type=int)

    raw_save_once_per_view_t = ParameterValue(raw_save_once_per_view, value_type=bool)
    raw_overwrite_t = ParameterValue(raw_overwrite, value_type=bool)
    raw_ply_ascii_t = ParameterValue(raw_ply_ascii, value_type=bool)

    return LaunchDescription(
        [
            # ---------------- Declare launch arguments ----------------
            DeclareLaunchArgument(
                "plants_root_dir",
                default_value="/home/parallels/Forschsemrep/strawberry_ws/data/plant_views",
                description="Root folder containing plant subfolders (recommended mode).",
            ),
            DeclareLaunchArgument(
                "plant_glob",
                default_value="plant_*",
                description="Glob for plant directories inside plants_root_dir.",
            ),
            DeclareLaunchArgument(
                "use_plants_root",
                default_value="true",
                description="Use plants_root_dir mode (true/false).",
            ),
            DeclareLaunchArgument(
                "rgb_pattern",
                default_value="color_*.png",
                description="RGB filename pattern inside each plant folder.",
            ),
            DeclareLaunchArgument(
                "depth_pattern",
                default_value="depth_*.png",
                description="Depth filename pattern inside each plant folder.",
            ),
            DeclareLaunchArgument(
                "rgb_dir",
                default_value="",
                description="(Legacy) Folder containing RGB images.",
            ),
            DeclareLaunchArgument(
                "depth_dir",
                default_value="",
                description="(Legacy) Folder containing depth images.",
            ),
            DeclareLaunchArgument(
                "fps",
                default_value="0.05",
                description="Playback FPS for camera_folder.",
            ),
            DeclareLaunchArgument(
                "loop",
                default_value="true",
                description="Loop dataset playback (true/false).",
            ),
            DeclareLaunchArgument(
                "publish_depth",
                default_value="true",
                description="Publish depth images (true/false).",
            ),
            DeclareLaunchArgument(
                "publish_pose",
                default_value="true",
                description="Publish /camera_pose_world PoseStamped (true/false).",
            ),
            DeclareLaunchArgument(
                "pose_topic",
                default_value="/camera_pose_world",
                description="PoseStamped topic name.",
            ),
            DeclareLaunchArgument(
                "world_frame_id",
                default_value="world",
                description="Frame id for published camera pose + frame info header.",
            ),
            DeclareLaunchArgument(
                "publish_frame_info",
                default_value="true",
                description="Publish FrameInfo on /camera/frame_info (true/false).",
            ),
            DeclareLaunchArgument(
                "frame_info_topic",
                default_value="/camera/frame_info",
                description="FrameInfo topic name published by camera_folder.",
            ),
            DeclareLaunchArgument(
                "model_path",
                default_value="",
                description="Path to best.pt (empty = use package share model).",
            ),
            DeclareLaunchArgument(
                "publish_overlay",
                default_value="true",
                description="Publish /seg/overlay from seg_ultra (true/false).",
            ),
            DeclareLaunchArgument(
                "sync_queue_size",
                default_value="200",
                description="TimeSynchronizer queue size (exact sync nodes).",
            ),
            DeclareLaunchArgument(
                "sync_slop",
                default_value="0.2",
                description="Kept for compatibility (unused by exact sync nodes).",
            ),
            DeclareLaunchArgument(
                "depth_unit",
                default_value="realsense_units",
                description="Depth unit: 'mm' or 'realsense_units'.",
            ),
            DeclareLaunchArgument(
                "depth_scale_m_per_unit",
                default_value="9.999999747378752e-05",
                description="Depth scale in meters/unit for realsense_units.",
            ),
            DeclareLaunchArgument(
                "selected_instance_id",
                default_value="1",
                description="Selected instance id for selected cloud/overlay.",
            ),
            DeclareLaunchArgument(
                "enable_selected_overlay",
                default_value="true",
                description="Enable strawberry_selected_overlay node (true/false).",
            ),

            # Output dirs
            DeclareLaunchArgument(
                "cluster_output_dir",
                default_value=os.path.join(home, "strawberry_ply"),
                description="Output root for clustered PLYs.",
            ),
            DeclareLaunchArgument(
                "raw_output_dir",
                default_value=os.path.join(home, "strawberry_raw_ply"),
                description="Output root for raw per-view PLYs.",
            ),

            # raw_cloud_saver params
            DeclareLaunchArgument(
                "raw_save_once_per_view",
                default_value="true",
                description="Save exactly one raw cloud per (plant_id, view_id).",
            ),
            DeclareLaunchArgument(
                "raw_overwrite",
                default_value="false",
                description="Overwrite raw cloud files if they already exist.",
            ),
            DeclareLaunchArgument(
                "raw_assume_cloud_in",
                default_value="camera",
                description="Interpret incoming cloud coordinates as 'camera' or 'world'.",
            ),
            DeclareLaunchArgument(
                "raw_export_cloud_in",
                default_value="world",
                description="Export raw clouds in 'world' or 'camera' coordinates.",
            ),
            DeclareLaunchArgument(
                "raw_ply_ascii",
                default_value="true",
                description="Write raw PLY in ASCII (true) or binary (false).",
            ),

            # NEW toggles
            DeclareLaunchArgument(
                "enable_raw_cloud_saver",
                default_value="true",
                description="Enable raw_cloud_saver (true/false).",
            ),
            DeclareLaunchArgument(
                "enable_cluster",
                default_value="true",
                description="Enable strawberry_cluster (true/false).",
            ),

            # ---------------- Camera from folder ----------------
            Node(
                package="strawberry_camera",
                executable="camera_folder",
                name="camera_folder",
                output="screen",
                parameters=[
                    {
                        "use_plants_root": use_plants_root_t,
                        "plants_root_dir": plants_root_dir,
                        "plant_glob": plant_glob,
                        "rgb_pattern": rgb_pattern,
                        "depth_pattern": depth_pattern,
                        "rgb_dir": rgb_dir,
                        "depth_dir": depth_dir,
                        "fps": fps_t,
                        "loop": loop_t,
                        "publish_depth": publish_depth_t,
                        "publish_pose": publish_pose_t,
                        "pose_topic": pose_topic,
                        "world_frame_id": world_frame_id,
                        "publish_frame_info": publish_frame_info_t,
                        "frame_info_topic": frame_info_topic,
                        "startup_log": True,
                        "startup_preview_n": 6,
                        "runtime_log_level": "warn",
                        "warn_slow_tick": False,
                    }
                ],
            ),

            # ---------------- YOLOv8 segmentation ----------------
            Node(
                package="strawberry_segmentation",
                executable="seg_ultra",
                name="strawberry_seg_ultra",
                output="screen",
                parameters=[
                    {
                        "model_path": model_path,
                        "topic_in": "/camera/color/image_raw",
                        "frame_info_topic": frame_info_topic,
                        "publish_frame_info": publish_frame_info_t,
                        "frame_info_out_topic": "/seg/frame_info",
                        "publish_overlay": publish_overlay_t,
                        "sync_queue_size": sync_queue_size_t,
                        "sync_slop": sync_slop_t,  # unused by exact sync (kept for compat)

                        # Explicit output topics (safe, matches new node defaults)
                        "label_topic": "/seg/label_image",
                        "label_vis_topic": "/seg/label_image_vis",
                        "overlay_topic": "/seg/overlay",
                    }
                ],
            ),

            # ---------------- Depth masking ----------------
            Node(
                package="strawberry_segmentation",
                executable="depth_mask",
                name="strawberry_depth_mask",
                output="screen",
                parameters=[
                    {
                        "depth_topic": "/camera/aligned_depth_to_color/image_raw",
                        "label_topic": "/seg/label_image",
                        "output_topic": "/seg/depth_masked",
                        "frame_info_topic": "/seg/frame_info",
                        "publish_frame_info": publish_frame_info_t,
                        "frame_info_out_topic": "/seg/frame_info_depth_masked",
                        "sync_queue_size": sync_queue_size_t,
                        "sync_slop": sync_slop_t,  # unused by exact sync (kept for compat)
                        "zero_background": True,
                        "depth_unit": depth_unit,
                        "depth_scale_m_per_unit": depth_scale_t,
                        "range_filter_enable": True,
                        "min_depth_m": 0.05,
                        "max_depth_m": 0.60,
                        "treat_65535_as_invalid": True,
                    }
                ],
            ),

            # ---------------- Features + point clouds ----------------
            Node(
                package="strawberry_segmentation",
                executable="strawberry_features",
                name="strawberry_features",
                output="screen",
                parameters=[
                    {
                        "depth_topic": "/seg/depth_masked",
                        "label_topic": "/seg/label_image",
                        "camera_info_topic": "/camera/color/camera_info",
                        "frame_info_topic": "/seg/frame_info_depth_masked",
                        "downsample_step": 1,
                        "min_points": 50,
                        "profile": False,
                        "depth_unit": depth_unit,
                        "depth_scale_m_per_unit": depth_scale_t,
                        "publish_all_cloud": True,
                        "cloud_topic_all": "/seg/strawberry_cloud",
                        "publish_selected_cloud": True,
                        "cloud_topic_selected": "/seg/strawberry_cloud_selected",
                        "selected_instance_id": selected_instance_id_t,
                        "log_features": True,
                        "sync_queue_size": sync_queue_size_t,
                        "sync_slop": sync_slop_t,
                    }
                ],
            ),

            # ---------------- Selected instance overlay (optional) ----------------
            Node(
                package="strawberry_segmentation",
                executable="strawberry_selected_overlay",
                name="strawberry_selected_overlay",
                output="screen",
                condition=IfCondition(enable_selected_overlay),
                parameters=[
                    {
                        "image_topic": "/camera/color/image_raw",
                        "label_topic": "/seg/label_image",
                        "output_topic": "/seg/selected_overlay",
                        "frame_info_topic": "/seg/frame_info",
                        "publish_frame_info": publish_frame_info_t,
                        "frame_info_out_topic": "/seg/frame_info_selected_overlay",
                        "selected_instance_id": selected_instance_id_t,
                        "min_pixels": 50,
                        "darken_factor": 0.3,
                        "draw_bbox": True,
                        "sync_queue_size": sync_queue_size_t,
                        "sync_slop": sync_slop_t,
                    }
                ],
            ),

            # ---------------- Raw cloud saver (optional, default ON) ----------------
            Node(
                package="strawberry_cluster",
                executable="raw_cloud_saver",
                name="raw_cloud_saver",
                output="screen",
                condition=IfCondition(enable_raw_cloud_saver),
                parameters=[
                    {
                        "cloud_topic": "/seg/strawberry_cloud",
                        "frame_info_topic": "/seg/frame_info_depth_masked",
                        "output_dir_raw": raw_output_dir,   # <-- FIX (war output_dir)
                        "save_once_per_view": raw_save_once_per_view_t,
                        "overwrite": raw_overwrite_t,
                        "assume_cloud_in": raw_assume_cloud_in,
                        "export_cloud_in": raw_export_cloud_in,
                        "ply_ascii": raw_ply_ascii_t,
                    }
                    ],
            ),

            # ---------------- Cluster node (optional, default OFF) ----------------
            Node(
                package="strawberry_cluster",
                executable="strawberry_cluster",
                name="strawberry_cluster",
                output="screen",
                condition=IfCondition(enable_cluster),
                parameters=[
                    {
                        "depth_topic": "/seg/depth_masked",
                        "label_topic": "/seg/label_image",
                        "frame_info_topic": "/seg/frame_info_depth_masked",
                        "camera_info_topic": "/camera/color/camera_info",
                        "depth_unit": depth_unit,
                        "depth_scale_m_per_unit": depth_scale_t,
                        "sync_queue_size": sync_queue_size_t,
                        "sync_slop": sync_slop_t,
                        "reset_on_new_plant": True,
                        "log_assignments": True,
                        "output_dir": cluster_output_dir,
                        "write_ply_on_plant_change": True,
                        "write_ply_on_shutdown": True,
                        "ply_ascii": True,
                        "export_fused_plant_cloud": True,
                        "fused_filename": "plant_fused.ply",
                    }
                ],
            ),
        ]
    )
