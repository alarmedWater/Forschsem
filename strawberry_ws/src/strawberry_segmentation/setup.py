from setuptools import setup
from glob import glob
import os

package_name = 'strawberry_segmentation'

# Alle Dateien im models-Ordner (z.B. best.onnx, best.pt) automatisch mitnehmen
model_files = [
    f for f in glob(os.path.join(package_name, 'models', '*'))
    if os.path.isfile(f)
]

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        # ROS 2 Package-Index
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # package.xml
        ('share/' + package_name, ['package.xml']),
        # Modelle (ONNX + PT) ins Share-Verzeichnis kopieren
        ('share/' + package_name + '/models', model_files),
    ],
    install_requires=[
        'setuptools',
        # Optional: nur relevant, wenn du das Paket per pip installierst.
        # Für ROS-Abhängigkeiten ist eigentlich package.xml zuständig.
        # 'ultralytics',
        # 'onnxruntime',
        # 'opencv-python',
        # 'numpy',
    ],
    zip_safe=False,
    maintainer='Julian Schrenk',
    maintainer_email='julian.schrenk@stud.hs-hannover.de',
    description='ROS2 strawberry segmentation using YOLOv8 (ONNXRuntime and Ultralytics).',
    license='MIT',
    entry_points={
        'console_scripts': [
            # ONNX-Variante
            'seg_onnx  = strawberry_segmentation.seg_onnx_node:main',
            # Ultralytics-.pt-Variante (wie yolo predict)
            'seg_ultra = strawberry_segmentation.seg_ultra_node:main',
        ],
    },
)
