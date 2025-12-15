from __future__ import annotations

import os
from glob import glob

from setuptools import setup

package_name = "strawberry_camera"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        # ament package index marker file (MUSS existieren!):
        # strawberry_ws/src/strawberry_camera/resource/strawberry_camera
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),

        # Install calibration YAML(s) into:
        #   <install>/share/strawberry_camera/config/*.yml
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yml") + glob("config/*.yaml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Julian Schrenk",
    maintainer_email="julian.schrenk@stud.hs-hannover.de",
    description=(
        "Dummy / folder-based camera publishers (RGB, aligned depth, camera_info)"
    ),
    license="MIT",
    entry_points={
        "console_scripts": [
            "camera_dummy = strawberry_camera.camera_dummy_node:main",
            "camera_folder = strawberry_camera.camera_folder_node:main",
        ],
    },
)
