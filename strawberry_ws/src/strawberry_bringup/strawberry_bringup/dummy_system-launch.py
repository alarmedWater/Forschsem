from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='strawberry_camera', executable='camera_dummy', name='camera_dummy',
            parameters=[{
                'color_width': 1280, 'color_height': 720,
                'depth_width': 640,  'depth_height': 480,
                'fps': 30.0,
                'frame_color': 'camera_color_optical_frame',
                'frame_depth': 'camera_color_optical_frame',
                'fx': 900.0, 'fy': 900.0, 'cx': 640.0, 'cy': 360.0,  # Platzhalter!
            }],
            output='screen'
        ),
        Node(
            package='meca500_ros2', executable='robot_dummy', name='meca500_dummy',
            parameters=[{
                'world_frame': 'world',
                'base_frame': 'base_link',
                'tool_frame': 'tool0',
                'camera_link': 'camera_link',
                'camera_color_optical': 'camera_color_optical_frame',
            }],
            output='screen'
        ),
        Node(
            package='strawberry_segmentation', executable='seg_onnx', name='seg_onnx',
            parameters=[{
                'model_path': '/ABS/PATH/to/best.onnx',
                'imgsz': 1024, 'conf_thres': 0.25, 'iou_thres': 0.5,
                'num_classes': 1, 'mask_dim': 32,
                'providers': ['CPUExecutionProvider'],  # or ['CUDAExecutionProvider','CPUExecutionProvider']
            }],
            output='screen'
        ),

    ])
