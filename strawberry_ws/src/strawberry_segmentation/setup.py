from setuptools import setup
from glob import glob
import os

package_name = 'strawberry_segmentation'

# Alle Modelle im models-Ordner (z.B. best.onnx, best.pt) automatisch mitnehmen
model_files = [
    f for f in glob(os.path.join(package_name, 'models', '*'))
    if os.path.isfile(f)
]

# Launch-Dateien (z.B. strawberry_pipeline.launch.py) mitnehmen
launch_files = [
    f for f in glob(os.path.join(package_name, 'launch', '*.launch.py'))
    if os.path.isfile(f)
]

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        # ROS 2 Package-Index
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        # package.xml
        ('share/' + package_name, ['package.xml']),
        # Modelle (ONNX + PT) ins Share-Verzeichnis kopieren
        ('share/' + package_name + '/models', model_files),
        # 🔥 Launch-Dateien ins Share-Launch-Verzeichnis kopieren
        ('share/' + package_name + '/launch', launch_files),
    ],
    install_requires=[
        'setuptools',
        # Python-Abhängigkeiten für ROS kommen primär aus package.xml,
        # hier ist nur relevant, falls du das Paket mal per pip installierst.
        # 'ultralytics',
        # 'onnxruntime',
        # 'opencv-python',
        # 'numpy',
    ],
    zip_safe=False,
    maintainer='Julian Schrenk',
    maintainer_email='julian.schrenk@stud.hs-hannover.de',
    description='ROS2 strawberry segmentation using YOLOv8 (Ultralytics) and depth masking.',
    license='MIT',
    entry_points={
        'console_scripts': [
            # ✅ Nur noch der Ultralytics-Knoten (da du den ONNX-Knoten ja entfernt hast)
            'seg_ultra = strawberry_segmentation.seg_ultra_node:main',
            # ✅ Depth-Mask-Knoten
            'depth_mask = strawberry_segmentation.depth_mask_node:main',
            # ✅ PointCloud-Knoten
            'strawberry_cloud = strawberry_segmentation.strawberry_pointcloud_node:main',
            # ✅ Features Knoten, einzelen erdbeer punktewolken
            'strawberry_features = strawberry_segmentation.strawberry_features_node:main',
            'strawberry_selected_overlay = strawberry_segmentation.strawberry_selected_overlay_node:main'
        ],
    },
)
