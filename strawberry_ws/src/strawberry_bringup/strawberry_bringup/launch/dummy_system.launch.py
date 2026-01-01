#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch file for the strawberry demo pipeline (dummy dataset playback).

Nodes:
- camera_folder: plays RGB + depth images from folders as a fake camera
- seg_ultra: YOLOv8 instance segmentation on RGB images
- depth_mask: applies instance mask to depth image
- strawberry_features: per-instance 3D features + point clouds
- strawberry_selected_overlay: highlights one selected instance in RGB image
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Generate the launch description for the strawberry pipeline."""
    # Dataset mode A (recommended): plants root directory
    plants_root_dir = LaunchConfiguration("plants_root_dir")
    plant_glob = LaunchConfiguration("plant_glob")
    use_plants_root = LaunchConfiguration("use_plants_root")

    rgb_pattern = LaunchConfiguration("rgb_pattern")
    depth_pattern = LaunchConfiguration("depth_pattern")

    # Legacy mode B (flat folders) kept for fallback
    rgb_dir = LaunchConfiguration("rgb_dir")
    depth_dir = LaunchConfiguration("depth_dir")

    # General
    fps = LaunchConfiguration("fps")
    loop = LaunchConfiguration("loop")
    model_path = LaunchConfiguration("model_path")

    # Pose + FrameInfo topics
    publish_pose = LaunchConfiguration("publish_pose")
    pose_topic = LaunchConfiguration("pose_topic")
    world_frame_id = LaunchConfiguration("world_frame_id")

    publish_frame_info = LaunchConfiguration("publish_frame_info")
    frame_info_topic = LaunchConfiguration("frame_info_topic")

    return LaunchDescription(
        [
            # ---------------- Launch arguments ----------------
            DeclareLaunchArgument(
                "plants_root_dir",
                default_value="/home/parallels/Forschsemrep/strawberry_ws/data/plant_views",
                description=(
                    "Root folder containing plant subfolders (recommended mode). "
                    "Example: plant_000/, plant_001/, ..."
                ),
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
            # Legacy flat-folder args (still available)
            DeclareLaunchArgument(
                "rgb_dir",
                default_value="/home/parallels/Forschsemrep/strawberry_ws/data/test_rgb",
                description="(Legacy) Folder containing RGB images.",
            ),
            DeclareLaunchArgument(
                "depth_dir",
                default_value="/home/parallels/Forschsemrep/strawberry_ws/data/test_depth",
                description="(Legacy) Folder containing depth images.",
            ),
            DeclareLaunchArgument(
                "fps",
                default_value="2.0",
                description="Playback FPS for camera_folder.",
            ),
            DeclareLaunchArgument(
                "loop",
                default_value="false",
                description="Whether to loop the dataset (true/false).",
            ),
            DeclareLaunchArgument(
                "model_path",
                default_value="",
                description=(
                    "Path to best.pt (empty = use model from strawberry_segmentation/models in share)."
                ),
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
                description="FrameInfo topic name.",
            ),
            # ---------------- Camera from folder ----------------
            Node(
                package="strawberry_camera",
                executable="camera_folder",
                name="camera_folder",
                parameters=[
                    {
                        # Mode A (recommended)
                        "use_plants_root": use_plants_root,
                        "plants_root_dir": plants_root_dir,
                        "plant_glob": plant_glob,
                        "rgb_pattern": rgb_pattern,
                        "depth_pattern": depth_pattern,
                        # Mode B (legacy fallback)
                        "rgb_dir": rgb_dir,
                        "depth_dir": depth_dir,
                        # Playback
                        "fps": fps,
                        "loop": loop,
                        # Pose + frame info
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
                parameters=[
                    {
                        "model_path": model_path,
                    }
                ],
            ),
            # ---------------- Depth masking ----------------
            Node(
                package="strawberry_segmentation",
                executable="depth_mask",
                name="strawberry_depth_mask",
            ),
            # ---------------- Features + point clouds ----------------
            Node(
                package="strawberry_segmentation",
                executable="strawberry_features",
                name="strawberry_features",
                parameters=[
                    {
                        "depth_topic": "/seg/depth_masked",
                        "label_topic": "/seg/label_image",
                        "camera_info_topic": "/camera/color/camera_info",
                        "downsample_step": 1,
                        "min_points": 50,
                        "profile": False,
                        "publish_all_cloud": True,
                        "cloud_topic_all": "/seg/strawberry_cloud",
                        "publish_selected_cloud": True,
                        "cloud_topic_selected": "/seg/strawberry_cloud_selected",
                        "selected_instance_id": 1,
                    }
                ],
            ),
            # ---------------- Selected instance overlay ----------------
            Node(
                package="strawberry_segmentation",
                executable="strawberry_selected_overlay",
                name="strawberry_selected_overlay",
                parameters=[
                    {
                        "image_topic": "/camera/color/image_raw",
                        "label_topic": "/seg/label_image",
                        "output_topic": "/seg/selected_overlay",
                        "selected_instance_id": 1,
                        "min_pixels": 50,
                        "darken_factor": 0.3,
                    }
                ],
            ),
        ]
    )
