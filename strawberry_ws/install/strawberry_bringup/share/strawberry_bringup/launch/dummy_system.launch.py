#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Launch file for the strawberry demo pipeline.

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
    rgb_dir = LaunchConfiguration("rgb_dir")
    depth_dir = LaunchConfiguration("depth_dir")
    fps = LaunchConfiguration("fps")
    loop = LaunchConfiguration("loop")
    model_path = LaunchConfiguration("model_path")

    return LaunchDescription(
        [
            # ---------------- Launch arguments ----------------
            DeclareLaunchArgument(
                "rgb_dir",
                default_value="/home/parallels/Forschsemrep/strawberry_ws/data/test_rgb",
                description="Folder containing RGB images.",
            ),
            DeclareLaunchArgument(
                "depth_dir",
                default_value="/home/parallels/Forschsemrep/strawberry_ws/data/test_depth",
                description="Folder containing depth images.",
            ),
            DeclareLaunchArgument(
                "fps",
                default_value="2.0",
                description="Playback FPS for camera_folder.",
            ),
            DeclareLaunchArgument(
                "loop",
                default_value="true",
                description="Whether to loop the images (true/false).",
            ),
            DeclareLaunchArgument(
                "model_path",
                default_value="",
                description=(
                    "Path to best.pt "
                    "(empty = use model from strawberry_segmentation/models in share)."
                ),
            ),

            # ---------------- Camera from folder ----------------
            Node(
                package="strawberry_camera",
                executable="camera_folder",
                name="camera_folder",
                parameters=[
                    {
                        "rgb_dir": rgb_dir,
                        "depth_dir": depth_dir,
                        "fps": fps,
                        "loop": loop,
                        # fx, fy, cx, cy have defaults inside the node
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
                        # empty -> use share/strawberry_segmentation/models/best.pt
                        "model_path": model_path,
                        # "topic_in": "/camera/color/image_raw",  # default in node
                    }
                ],
            ),

            # ---------------- Depth masking ----------------
            Node(
                package="strawberry_segmentation",
                executable="depth_mask",
                name="strawberry_depth_mask",
                # defaults in node:
                # depth_topic:  /camera/aligned_depth_to_color/image_raw
                # label_topic:  /seg/label_image
                # output_topic: /seg/depth_masked
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
                        # point cloud publishing
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
