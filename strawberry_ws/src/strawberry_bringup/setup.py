# strawberry_bringup/setup.py
from setuptools import setup
from glob import glob

package_name = 'strawberry_bringup'

setup(
    name=package_name,
    version='0.0.1',
    packages=[],  # bringup carries only launch files
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Julian Schrenk',
    maintainer_email='julian.schrenk@stud.hs-hannover.de',
    description='Launch files for the strawberry pipeline (camera + robot dummies).',
    license='MIT',
)
