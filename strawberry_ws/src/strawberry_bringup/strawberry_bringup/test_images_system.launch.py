from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='strawberry_camera', executable='camera_folder', name='camera_folder',
            parameters=[{
                'image_dir': '/home/parallels/Forschsemrep/strawberry_ws/data/test_images',
                'fps': 5.0,
                'frame_color': 'camera_color_optical_frame',
                # Intrinsics grob; wenn du echte Kennwerte hast, hier einsetzen:
                'fx': 900.0, 'fy': 900.0, 'cx': 640.0, 'cy': 360.0,
                'publish_info': True, 'loop': True,
            }],
            output='screen'
        ),
        Node(
            package='strawberry_segmentation', executable='seg_onnx', name='seg_onnx',
            parameters=[{
                'model_path': '/home/parallels/Forschsemrep/best-seg.onnx',  # <— ANPASSEN!
                'imgsz': 1024, 'conf_thres': 0.25, 'iou_thres': 0.5,
                'num_classes': 1, 'mask_dim': 32,
                'providers': ['CPUExecutionProvider'],
            }],
            output='screen'
        ),
    ])
