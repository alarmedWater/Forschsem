from setuptools import setup

package_name = 'strawberry_segmentation'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Julian Schrenk',
    maintainer_email='julian.schrenk@stud.hs-hannover.de',
    description='ONNXRuntime-based YOLOv8 segmentation for strawberries.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'seg_onnx = strawberry_segmentation.seg_onnx_node:main',
        ],
    },
)
