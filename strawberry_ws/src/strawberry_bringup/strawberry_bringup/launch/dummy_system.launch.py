from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch-Argumente definieren
    rgb_dir    = LaunchConfiguration('rgb_dir')
    depth_dir  = LaunchConfiguration('depth_dir')
    fps        = LaunchConfiguration('fps')
    loop       = LaunchConfiguration('loop')
    model_path = LaunchConfiguration('model_path')

    return LaunchDescription([
        # ---------- Launch-Argumente ----------
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

        # ---------- Kamera aus Ordner (strawberry_camera) ----------
        Node(
            package='strawberry_camera',
            executable='camera_folder',
            name='camera_folder',
            parameters=[{
                'rgb_dir': rgb_dir,
                'depth_dir': depth_dir,
                'fps': fps,
                'loop': loop,
                # weitere Parameter wie fx, fy, cx, cy haben Defaultwerte im Node
            }],
        ),

        # ---------- YOLOv8-Segmentation (strawberry_segmentation) ----------
        Node(
            package='strawberry_segmentation',
            executable='seg_ultra',
            name='strawberry_seg_ultra',
            parameters=[{
                # leer = nimmt share/strawberry_segmentation/models/best.pt
                'model_path': model_path,
                # 'topic_in': '/camera/color/image_raw',  # ist im Node schon Default
            }],
        ),

        # ---------- Depth-Mask-Node (strawberry_segmentation) ----------
        Node(
            package='strawberry_segmentation',
            executable='depth_mask',
            name='strawberry_depth_mask',
            # Defaults:
            # depth_topic: /camera/aligned_depth_to_color/image_raw
            # label_topic: /seg/label_image
            # output_topic: /seg/depth_masked
        ),
        
        # --- Strawberry-PointCloud-Node ---
        Node(
            package='strawberry_segmentation',
            executable='strawberry_cloud',
            name='strawberry_pointcloud',
            parameters=[{
                'depth_topic': '/seg/depth_masked',
                'camera_info_topic': '/camera/color/camera_info',
                'cloud_topic': '/seg/strawberry_cloud',
                'downsample_step': 1,   # ggf. auf 2/4 setzen, wenn es zu viele Punkte werden
            }],
        ),
         # --- Feature-Node (pro Erdbeere) ---
        Node(
            package='strawberry_segmentation',
            executable='strawberry_features',
            name='strawberry_features',
            parameters=[{
                'depth_topic': '/seg/depth_masked',
                'label_topic': '/seg/label_image',
                'camera_info_topic': '/camera/color/camera_info',
                'downsample_step': 1,
                'min_points': 50,
                'profile': False,
            }],
        ),
        
         Node(
            package='strawberry_segmentation',
            executable='strawberry_selected_overlay',
            name='strawberry_selected_overlay',
            parameters=[{
                'image_topic': '/camera/color/image_raw',
                'label_topic': '/seg/label_image',
                'output_topic': '/seg/selected_overlay',
                'selected_instance_id': 1,
                'min_pixels': 50,
                'darken_factor': 0.3,
            }],
        ),
    ])

    

