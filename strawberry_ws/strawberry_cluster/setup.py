from setuptools import setup

package_name = "strawberry_cluster"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        # ROS 2 package index
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        # package.xml
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Julian Schrenk",
    maintainer_email="julian_martin.schrenk@smail.th-koeln.de",
    description=(
        "ROS2 package for clustering strawberry instances across "
        "multiple RGB-D views using 3D centroids in world coordinates."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # strawberry_cluster_node.py: main()
            "strawberry_cluster = "
            "strawberry_cluster.strawberry_cluster_node:main",
        ],
    },
)
