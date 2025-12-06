from setuptools import setup

package_name = "strawberry_bringup"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (
            f"share/{package_name}",
            ["package.xml"],
        ),
        # Install launch files
        (
            f"share/{package_name}/launch",
            ["strawberry_bringup/launch/dummy_system.launch.py"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Julian Schrenk",
    maintainer_email="julian_martin.schrenk@smail.th-koeln.de",
    description="Bringup package for the strawberry perception pipeline.",
    license="MIT",
    entry_points={
        "console_scripts": [
            # Add future Python nodes here if needed.
        ],
    },
)
