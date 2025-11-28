from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Launch-Argumente durchreichen
    rgb_dir   = LaunchConfiguration('rgb_dir')
    depth_dir = LaunchConfiguration('depth_dir')
    fps       = LaunchConfiguration('fps')
    loop      = LaunchConfiguration('loop')
    model_path = LaunchConfiguration('model_path')

    # Pfad zu unserem Pipeline-Launch im strawberry_segmentation-Paket
    seg_share = get_package_share_directory('strawberry_segmentation')
    pipeline_launch = os.path.join(seg_share, 'launch', 'strawberry_pipeline.launch.py')

    return LaunchDescription([
        DeclareLaunchArgument(
            'rgb_dir',
            default_value='/home/parallels/Forschsemrep/strawberry_ws/data/test_rgb',
            description='Ordner mit RGB-Bildern'
        ),
        DeclareLaunchArgument(
            'depth_dir',
            default_value='/home/parallels/Forschsemrep/strawberry_ws/data/test_rgb',
            description='Ordner mit Depth-Bildern'
        ),
        DeclareLaunchArgument(
            'fps',
            default_value='2.0',
            description='FPS für camera_folder'
        ),
        DeclareLaunchArgument(
            'loop',
            default_value='true',
            description='Bilder in Schleife abspielen (true/false)'
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value='',
            description='Pfad zu best.pt (leer = Modell aus strawberry_segmentation/models)'
        ),

        # 👉 Hier binden wir den Pipeline-Launch ein
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(pipeline_launch),
            launch_arguments={
                'rgb_dir': rgb_dir,
                'depth_dir': depth_dir,
                'fps': fps,
                'loop': loop,
                'model_path': model_path,
            }.items()
        )
    ])
