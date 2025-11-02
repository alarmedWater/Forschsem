#!/usr/bin/env python3
import rclpy, numpy as np, time
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge

class CameraDummy(Node):
    def __init__(self):
        super().__init__('camera_dummy')

        # Params (anpassbar via Launch/YAML)
        self.declare_parameter('color_width', 1280)
        self.declare_parameter('color_height', 720)
        self.declare_parameter('depth_width', 640)
        self.declare_parameter('depth_height', 480)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('fx', 600.0)   # Platzhalter
        self.declare_parameter('fy', 600.0)
        self.declare_parameter('cx', 640.0)
        self.declare_parameter('cy', 360.0)
        self.declare_parameter('frame_color', 'camera_color_optical_frame')
        self.declare_parameter('frame_depth', 'camera_color_optical_frame')  # aligned to color!
        self.declare_parameter('distortion_model', 'plumb_bob')
        self.declare_parameter('D', [0.0, 0.0, 0.0, 0.0, 0.0])               # Dummy

        self.bridge = CvBridge()
        self.pub_rgb   = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.pub_depth = self.create_publisher(Image, '/camera/aligned_depth_to_color/image_raw', 10)
        self.pub_info  = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)

        period = 1.0 / float(self.get_parameter('fps').get_parameter_value().double_value)
        self.timer = self.create_timer(period, self.tick)
        self.t0 = time.time()
        self.i  = 0

    def make_cam_info(self, stamp):
        w = self.get_parameter('color_width').value
        h = self.get_parameter('color_height').value
        fx = self.get_parameter('fx').value
        fy = self.get_parameter('fy').value
        cx = self.get_parameter('cx').value
        cy = self.get_parameter('cy').value
        frame = self.get_parameter('frame_color').value

        info = CameraInfo()
        info.header = Header(stamp=stamp, frame_id=frame)
        info.width, info.height = w, h
        info.distortion_model = self.get_parameter('distortion_model').value
        info.d = self.get_parameter('D').value
        info.k = [fx, 0.0, cx,
                  0.0, fy, cy,
                  0.0, 0.0, 1.0]
        info.r = [1,0,0, 0,1,0, 0,0,1]
        # Projektionsmatrix (mono)
        info.p = [fx, 0.0, cx, 0.0,
                  0.0, fy, cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        return info

    def tick(self):
        # Einheitlicher Zeitstempel für RGB/Depth/Info
        now = self.get_clock().now().to_msg()

        # Dummy RGB: horizontales Farbverlauf-Pattern
        w = self.get_parameter('color_width').value
        h = self.get_parameter('color_height').value
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)
        rgb[..., 1] = (self.i * 5) % 255
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
        rgb_msg.header.stamp = now
        rgb_msg.header.frame_id = self.get_parameter('frame_color').value

        # Dummy Depth (16UC1 mm): geneigte Ebene bei ~300–500 mm
        wd = self.get_parameter('depth_width').value
        hd = self.get_parameter('depth_height').value
        xv = np.linspace(0, 1, wd, dtype=np.float32)
        yv = np.linspace(0, 1, hd, dtype=np.float32)[:, None]
        depth_mm = (300.0 + 200.0*(xv + yv))  # 300..700 mm
        depth = depth_mm.astype(np.uint16)
        depth_msg = Image()
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = self.get_parameter('frame_depth').value  # aligned to color
        depth_msg.height = hd
        depth_msg.width  = wd
        depth_msg.encoding = '16UC1'
        depth_msg.is_bigendian = 0
        depth_msg.step = wd * 2
        depth_msg.data = depth.tobytes()

        info = self.make_cam_info(now)

        self.pub_rgb.publish(rgb_msg)
        self.pub_depth.publish(depth_msg)
        self.pub_info.publish(info)
        self.i += 1

def main():
    rclpy.init()
    node = CameraDummy()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
