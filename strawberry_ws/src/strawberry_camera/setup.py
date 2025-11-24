from setuptools import setup

package_name = 'strawberry_camera'

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
    maintainer='you',
    maintainer_email='you@example.com',
    description='Dummy / folder-based camera publishers (RGB, aligned depth, camera_info)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_dummy = strawberry_camera.camera_dummy_node:main',
            'camera_folder = strawberry_camera.camera_folder_node:main',  # <-- neuer Folder-Player
        ],
    },
)
