from setuptools import setup
package_name = 'meca500_ros2'
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
    description='Dummy Meca500 publishers (joint_states + TF)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'robot_dummy = meca500_ros2.robot_dummy_node:main',
        ],
    },
)
