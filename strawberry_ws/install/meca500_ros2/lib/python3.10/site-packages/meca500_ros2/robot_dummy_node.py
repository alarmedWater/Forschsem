#!/usr/bin/env python3
import math, rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

def quat_from_euler(roll, pitch, yaw):
    cr = math.cos(roll * 0.5);   sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5);  sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5);    sy = math.sin(yaw * 0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return (x, y, z, w)

def qmul(a,b):
    ax,ay,az,aw = a; bx,by,bz,bw = b
    return (
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz
    )

class Meca500Dummy(Node):
    def __init__(self):
        super().__init__('meca500_dummy')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'tool0')
        self.declare_parameter('camera_link', 'camera_link')
        self.declare_parameter('camera_color_optical', 'camera_color_optical_frame')

        self.pub_js = self.create_publisher(JointState, '/joint_states', 10)
        self.br = TransformBroadcaster(self)
        self.sbr = StaticTransformBroadcaster(self)

        self.publish_static_tfs()
        self.timer = self.create_timer(0.05, self.tick)
        self.t = 0.0

    def publish_static_tfs(self):
        tool0   = self.get_parameter('tool_frame').value
        cam     = self.get_parameter('camera_link').value
        cam_opt = self.get_parameter('camera_color_optical').value

        # tool0 -> camera_link
        t1 = TransformStamped()
        t1.header.stamp = self.get_clock().now().to_msg()
        t1.header.frame_id = tool0
        t1.child_frame_id  = cam
        t1.transform.translation.x = 0.05
        t1.transform.translation.y = 0.00
        t1.transform.translation.z = 0.10
        t1.transform.rotation.w = 1.0
        self.sbr.sendTransform(t1)

        # camera_link -> camera_color_optical_frame
        # optical: x right, y down, z forward  => R = R_x(-90°) * R_z(90°)
        qx = quat_from_euler(-math.pi/2, 0.0, 0.0)
        qz = quat_from_euler(0.0, 0.0, math.pi/2)
        q_final = qmul(qz, qx)

        t2 = TransformStamped()
        t2.header.stamp = t1.header.stamp
        t2.header.frame_id = cam
        t2.child_frame_id  = cam_opt
        t2.transform.rotation.x, t2.transform.rotation.y, t2.transform.rotation.z, t2.transform.rotation.w = q_final
        self.sbr.sendTransform(t2)

    def tick(self):
        now = self.get_clock().now().to_msg()
        world = self.get_parameter('world_frame').value
        base  = self.get_parameter('base_frame').value
        tool0 = self.get_parameter('tool_frame').value

        # world -> base_link (circle)
        self.t += 0.05
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = world
        t.child_frame_id  = base
        R = 0.1
        t.transform.translation.x = R*math.cos(self.t)
        t.transform.translation.y = R*math.sin(self.t)
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0
        self.br.sendTransform(t)

        # base_link -> tool0 (fixed)
        t2 = TransformStamped()
        t2.header.stamp = now
        t2.header.frame_id = base
        t2.child_frame_id  = tool0
        t2.transform.translation.x = 0.3
        t2.transform.translation.y = 0.0
        t2.transform.translation.z = 0.4
        t2.transform.rotation.w = 1.0
        self.br.sendTransform(t2)

        # JointState
        js = JointState()
        js.header = Header(stamp=now)
        js.name = [f'joint_{i+1}' for i in range(6)]
        js.position = [0.0, 0.0, 0.0, 0.0, 0.0, self.t % (2*math.pi)]
        self.pub_js.publish(js)

def main():
    rclpy.init()
    node = Meca500Dummy()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
