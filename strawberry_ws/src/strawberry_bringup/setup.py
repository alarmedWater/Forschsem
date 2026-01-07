from __future__ import annotations

import os
from glob import glob

from setuptools import setup

package_name = "strawberry_bringup"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        # Install ALL launch files from launch/
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Julian Schrenk",
    maintainer_email="julian.schrenk@stud.hs-hannover.de",
    description="Bringup package for the strawberry perception pipeline.",
    license="MIT",
    entry_points={"console_scripts": []},
)
