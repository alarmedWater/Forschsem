from glob import glob
import os

from setuptools import setup

package_name = "strawberry_segmentation"

# Collect all model files in the models folder (e.g. best.onnx, best.pt)
model_files = [
    f
    for f in glob(os.path.join(package_name, "models", "*"))
    if os.path.isfile(f)
]

# Collect all launch files (e.g. strawberry_pipeline.launch.py)
launch_files = [
    f
    for f in glob(os.path.join(package_name, "launch", "*.launch.py"))
    if os.path.isfile(f)
]

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        # ROS 2 package index
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        # package.xml
        ("share/" + package_name, ["package.xml"]),
        # Install models (ONNX / PT) into the share/models directory
        ("share/" + package_name + "/models", model_files),
        # Install launch files into the share/launch directory
        ("share/" + package_name + "/launch", launch_files),
    ],
    install_requires=[
        "setuptools",
        # Python dependencies for ROS are usually handled via package.xml.
        # Add entries here only if you want to pip-install this package.
        # "ultralytics",
        # "onnxruntime",
        # "opencv-python",
        # "numpy",
    ],
    zip_safe=False,
    maintainer="Julian Schrenk",
    maintainer_email="julian_martin.schrenk@smail.th-koeln.de",
    description=(
        "ROS2 strawberry segmentation using YOLOv8 (Ultralytics), "
        "depth masking, and point cloud processing."
    ),
    license="MIT",
    entry_points={
        "console_scripts": [
            # YOLOv8 segmentation node
            "seg_ultra = strawberry_segmentation.seg_ultra_node:main",
            # Depth masking node
            "depth_mask = strawberry_segmentation.depth_mask_node:main",
            # Per-instance features + selected cloud
            (
                "strawberry_features = "
                "strawberry_segmentation.strawberry_features_node:main"
            ),
            # RGB overlay of the selected instance
            (
                "strawberry_selected_overlay = "
                "strawberry_segmentation.strawberry_selected_overlay_node:main"
            ),
        ],
    },
)
