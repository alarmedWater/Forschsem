from setuptools import find_packages, setup

package_name = "strawberry_tools"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="parallels",
    maintainer_email="julian.schrenk@stud.hs-hannover.de",
    description="Tools for strawberry pipeline debugging and auditing",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sync_audit = strawberry_tools.sync_audit_node:main",
        ],
    },
)
