#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch file for the strawberry demo pipeline (dataset playback).

Nodes:
- camera_folder: plays RGB + depth images from folders as a fake camera
- seg_ultra: YOLOv8 instance segmentation on RGB images + FrameInfo passthrough
- depth_mask: applies instance mask to depth image + FrameInfo passthrough
- strawberry_features: per-instance 3D features + point clouds
- strawberry_selected_overlay: highlights one selected instance in RGB image (optional)
- strawberry_cluster: clusters instances across views using pose (optional)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # ---------------- Launch arguments ----------------
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
    sync_slop = LaunchConfiguration("sync_slop")

    depth_unit = LaunchConfiguration("depth_unit")
    depth_scale_m_per_unit = LaunchConfiguration("depth_scale_m_per_unit")

    selected_instance_id = LaunchConfiguration("selected_instance_id")

    enable_selected_overlay = LaunchConfiguration("enable_selected_overlay")
    enable_cluster = LaunchConfiguration("enable_cluster")

    return LaunchDescription(
        [
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
                default_value="2.0",
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
                description="ApproximateTimeSynchronizer queue size.",
            ),
            DeclareLaunchArgument(
                "sync_slop",
                default_value="0.2",
                description="ApproximateTimeSynchronizer slop (seconds).",
            ),
            DeclareLaunchArgument(
                "depth_unit",
                default_value="realsense_units",
                description="Depth unit for features/cluster: 'mm' or 'realsense_units'.",
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
            DeclareLaunchArgument(
                "enable_cluster",
                default_value="true",
                description="Enable strawberry_cluster node (true/false).",
            ),

            # ---------------- Camera from folder ----------------
            Node(
                package="strawberry_camera",
                executable="camera_folder",
                name="camera_folder",
                output="screen",
                parameters=[
                    {
                        "use_plants_root": use_plants_root,
                        "plants_root_dir": plants_root_dir,
                        "plant_glob": plant_glob,
                        "rgb_pattern": rgb_pattern,
                        "depth_pattern": depth_pattern,
                        "rgb_dir": rgb_dir,
                        "depth_dir": depth_dir,
                        "fps": fps,
                        "loop": loop,
                        "publish_depth": publish_depth,
                        "publish_pose": publish_pose,
                        "pose_topic": pose_topic,
                        "world_frame_id": world_frame_id,
                        "publish_frame_info": publish_frame_info,
                        "frame_info_topic": frame_info_topic,
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
                        "publish_frame_info": True,
                        "frame_info_out_topic": "/seg/frame_info",
                        "publish_overlay": publish_overlay,
                        "sync_queue_size": sync_queue_size,
                        "sync_slop": sync_slop,
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
                        "publish_frame_info": True,
                        "frame_info_out_topic": "/seg/frame_info_depth_masked",
                        "sync_queue_size": sync_queue_size,
                        "sync_slop": sync_slop,
                        "zero_background": True,

                        # --- WICHTIG für realsense_units + range gating ---
                        "depth_unit": depth_unit,
                        "depth_scale_m_per_unit": depth_scale_m_per_unit,

                        # optional (wenn du meinen Nahbereichsfilter eingebaut hast)
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
                        "depth_scale_m_per_unit": depth_scale_m_per_unit,
                        "publish_all_cloud": True,
                        "cloud_topic_all": "/seg/strawberry_cloud",
                        "publish_selected_cloud": True,
                        "cloud_topic_selected": "/seg/strawberry_cloud_selected",
                        "selected_instance_id": selected_instance_id,
                        "log_features": True,
                        "sync_queue_size": sync_queue_size,
                        "sync_slop": sync_slop,
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
                        "publish_frame_info": True,
                        "frame_info_out_topic": "/seg/frame_info_selected_overlay",
                        "selected_instance_id": selected_instance_id,
                        "min_pixels": 50,
                        "darken_factor": 0.3,
                        "draw_bbox": True,
                        "sync_queue_size": sync_queue_size,
                        "sync_slop": sync_slop,

                    }
                ],
            ),

            # ---------------- Cluster node (optional) ----------------
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
                        "camera_pose_topic": pose_topic,
                        "frame_info_topic": "/seg/frame_info_depth_masked",
                        "camera_info_topic": "/camera/color/camera_info",
                        "depth_unit": depth_unit,
                        "depth_scale_m_per_unit": depth_scale_m_per_unit,
                        "sync_queue_size": sync_queue_size,
                        "sync_slop": sync_slop,
                        "reset_on_new_plant": True,
                        "log_assignments": True,
                    }
                ],
            ),
        ]
    )
