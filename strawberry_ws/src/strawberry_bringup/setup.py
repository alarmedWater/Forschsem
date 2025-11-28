from setuptools import setup

package_name = 'strawberry_bringup'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name,
         ['package.xml']),
        # ✅ KORRIGIERTER PFAD:
        ('share/' + package_name + '/launch', [
            'strawberry_bringup/launch/dummy_system.launch.py'
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Julian Schrenk',
    maintainer_email='julian.schrenk@stud.hs-hannover.de',
    description='Bringup für das Strawberry-System.',
    license='MIT',
    entry_points={
        'console_scripts': [
            # optional: eigene Nodes
        ],
    },
)
