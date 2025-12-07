#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Simple Meca500 controller as a ROS 2 node.

- Connects to the robot via mecademicpy.
- Moves to predefined joint positions: 'links', 'mitte', 'rechts'.
- After each motion:
    * reads the TCP pose
    * publishes it as PoseStamped on /camera_pose_world (world <- camera)
    * calls the CaptureSnapshot service with (plant_id, view_id)
      so that RGB+D images are saved automatically.

Interaction:
    - Node reads from stdin (links/mitte/rechts/exit).
    - No subscriptions, only publishers + a service client.
"""

from __future__ import annotations

import math
import sys
from typing import Dict, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from mecademicpy import robot as mdr
from mecademicpy.robot_classes import CommunicationError
from rclpy.node import Node

# Import your custom service type
# Make sure the srv file exists at:
#   strawberry_msgs/srv/CaptureSnapshot.srv
from strawberry_msgs.srv import CaptureSnapshot


# --------------------------------------------------------------------------- #
# Helper functions for rotations                                             #
# --------------------------------------------------------------------------- #


def rotx(theta: float) -> np.ndarray:
    """Rotation around x-axis by angle theta [rad]."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )


def roty(theta: float) -> np.ndarray:
    """Rotation around y-axis by angle theta [rad]."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def rotz(theta: float) -> np.ndarray:
    """Rotation around z-axis by angle theta [rad]."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def rotation_matrix_to_quaternion(
    R: np.ndarray,
) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a (x, y, z, w) quaternion.

    Uses a numerically stable algorithm. Assumes R is a proper rotation matrix.
    """
    t = np.trace(R)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    return float(qx), float(qy), float(qz), float(qw)


# --------------------------------------------------------------------------- #
# Node                                                                        #
# --------------------------------------------------------------------------- #


class Meca500ViewNode(Node):
    """ROS 2 node controlling the Meca500 and publishing camera poses.

    After each move, it also calls the CaptureSnapshot service with:
        plant_id (parameter) and view_id (depends on 'links/mitte/rechts').
    """

    def __init__(self) -> None:
        super().__init__("meca500_view_node")

        # Parameters
        self.declare_parameter("robot_ip", "192.168.0.100")
        self.declare_parameter("velocity", 20.0)   # joint vel
        self.declare_parameter("acceleration", 20.0)
        self.declare_parameter("world_frame_id", "world")

        # Optional experiment metadata (plant we are looking at)
        self.declare_parameter("plant_id", 1)
        self.declare_parameter("view_id", 0)

        # Predefined joint positions (deg)
        self._positions: Dict[str, Tuple[float, float, float, float, float, float]] = {
            "links": (70.0, 14.0, 16.0, 80.0, -100.0, 75.0),
            "mitte": (0.0, -35.0, 35.0, 0.0, 0.0, 135.0),
            "rechts": (-70.0, 14.0, 16.0, 80.0, 100.0, 20.0),
        }

        # Mapping from view name to view_id
        self._view_ids: Dict[str, int] = {
            "links": 0,
            "mitte": 1,
            "rechts": 2,
        }

        # Publisher for camera pose in world frame
        self.pose_pub = self.create_publisher(
            PoseStamped,
            "/camera_pose_world",
            10,
        )

        # Service client for snapshot capture
        self.snapshot_client = self.create_client(
            CaptureSnapshot,
            "capture_snapshot",
        )
        self.get_logger().info(
            "Waiting for 'capture_snapshot' service..."
        )
        self.snapshot_client.wait_for_service()
        self.get_logger().info("Snapshot service is available.")

        # Connect to the robot
        self._connect_robot()

    # ------------------------------------------------------------------ #
    # Robot setup                                                        #
    # ------------------------------------------------------------------ #

    def _connect_robot(self) -> None:
        """Connect to the Meca500 and prepare it for motion."""
        robot_ip = str(self.get_parameter("robot_ip").value)
        vel = float(self.get_parameter("velocity").value)
        acc = float(self.get_parameter("acceleration").value)

        self.robot = mdr.Robot()

        try:
            self.get_logger().info(f"[ROBOT] Connecting to {robot_ip} ...")
            self.robot.Connect(robot_ip)
            self.get_logger().info("[ROBOT] Connected.")

            # Reset errors, clear old motions, home
            self._ensure_robot_ready()

            # Set speed
            self.robot.SetJointVel(vel)
            self.robot.SetJointAcc(acc)

            self.get_logger().info(
                f"[ROBOT] Ready. JointVel={vel}, JointAcc={acc}"
            )

        except (TimeoutError, CommunicationError) as exc:  # type: ignore[name-defined]
            self.get_logger().error(f"[ROBOT] Communication error: {exc}")
            raise

    def _ensure_robot_ready(self) -> None:
        """Clear errors/motions and home the robot."""
        try:
            self.robot.ResetError()
        except Exception:  # noqa: BLE001
            pass

        try:
            self.robot.ClearMotion()
        except Exception:  # noqa: BLE001
            pass

        self.robot.ActivateAndHome()
        self.robot.WaitHomed()

    # ------------------------------------------------------------------ #
    # Main interaction loop                                              #
    # ------------------------------------------------------------------ #

    def run_interactive(self) -> None:
        """Interactive loop using stdin to select positions.

        This function blocks and does not use rclpy.spin().
        Publishers and the service client still work.
        """
        self.get_logger().info(
            "Enter a position name: 'links', 'mitte', 'rechts'.\n"
            "Type 'exit' to quit."
        )

        world_frame = str(self.get_parameter("world_frame_id").value)

        try:
            while rclpy.ok():
                try:
                    user_input = input(
                        "Position (links/mitte/rechts/exit): "
                    ).strip().lower()
                except EOFError:
                    break

                if user_input == "exit":
                    break

                if user_input not in self._positions:
                    self.get_logger().warn(
                        f"Invalid input: '{user_input}'. "
                        "Valid: links, mitte, rechts."
                    )
                    continue

                joints = self._positions[user_input]
                self.get_logger().info(
                    f"[ROBOT] Moving to '{user_input}' "
                    f"with joints {joints} ..."
                )

                self.robot.MoveJoints(*joints)
                self.robot.WaitIdle()
                self.get_logger().info(
                    f"[ROBOT] Position '{user_input}' reached."
                )

                # Read TCP pose (x, y, z [mm], rx, ry, rz [deg])
                x, y, z, rx, ry, rz = self.robot.GetPose()

                # Convert to meters
                t_tcp_world = np.array(
                    [x, y, z],
                    dtype=np.float32,
                ) / 1000.0

                # Euler ZYX -> rotation matrix
                R_tcp_world = (
                    rotz(np.deg2rad(rz))
                    @ roty(np.deg2rad(ry))
                    @ rotx(np.deg2rad(rx))
                )

                qx, qy, qz, qw = rotation_matrix_to_quaternion(
                    R_tcp_world
                )

                # Update view_id parameter based on selected position (for logging)
                view_id = self._view_ids.get(user_input, 0)
                self._set_parameters_from_list(
                    [
                        ("view_id", view_id),
                    ]
                )

                # Log pose
                self._log_pose(x, y, z, R_tcp_world, view_id)

                # Publish PoseStamped for the camera (treating TCP as camera)
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = world_frame

                pose_msg.pose.position.x = float(t_tcp_world[0])
                pose_msg.pose.position.y = float(t_tcp_world[1])
                pose_msg.pose.position.z = float(t_tcp_world[2])

                pose_msg.pose.orientation.x = qx
                pose_msg.pose.orientation.y = qy
                pose_msg.pose.orientation.z = qz
                pose_msg.pose.orientation.w = qw

                self.pose_pub.publish(pose_msg)
                self.get_logger().info(
                    "Published /camera_pose_world for view "
                    f"'{user_input}' (view_id={view_id})."
                )

                # ---------------------------------------------------------- #
                # Call snapshot service with plant_id and view_id           #
                # ---------------------------------------------------------- #
                plant_id = int(self.get_parameter("plant_id").value)

                self.get_logger().info(
                    f"Calling capture_snapshot "
                    f"(plant_id={plant_id}, view_id={view_id}) ..."
                )

                req = CaptureSnapshot.Request()
                req.plant_id = plant_id
                req.view_id = view_id

                future = self.snapshot_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)

                if future.result() is not None:
                    if future.result().success:
                        self.get_logger().info(
                            f"Snapshot OK: {future.result().message}"
                        )
                    else:
                        self.get_logger().warn(
                            f"Snapshot failed: {future.result().message}"
                        )
                else:
                    self.get_logger().warn(
                        "Snapshot service call returned no result."
                    )

            self.get_logger().info(
                "[ROBOT] Returning to idle / shutting down loop."
            )

        except (TimeoutError, CommunicationError) as exc:  # type: ignore[name-defined]
            self.get_logger().error(f"[ROBOT] Communication error: {exc}")

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _set_parameters_from_list(self, items):
        """Helper to set parameters from a list of (name, value)."""
        from rclpy.parameter import Parameter  # local import

        params = []
        for name, value in items:
            if isinstance(value, bool):
                p_type = Parameter.Type.BOOL
            elif isinstance(value, int):
                p_type = Parameter.Type.INTEGER
            elif isinstance(value, float):
                p_type = Parameter.Type.DOUBLE
            else:
                p_type = Parameter.Type.STRING
            params.append(
                Parameter(
                    name=name,
                    value=value,
                    type=p_type,
                )
            )
        self.set_parameters(params)

    def _log_pose(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        R_tcp_world: np.ndarray,
        view_id: int,
    ) -> None:
        """Pretty-print the robot/camera pose to console."""
        self.get_logger().info("\n==============================")
        self.get_logger().info("      CAMERA / TCP POSE")
        self.get_logger().info("==============================")
        self.get_logger().info(
            "Camera position (WORLD) [mm]: "
            f"x={x_mm:.3f}, y={y_mm:.3f}, z={z_mm:.3f}"
        )

        cam_x = R_tcp_world[:, 0]
        cam_y = R_tcp_world[:, 1]
        cam_z = R_tcp_world[:, 2]

        self.get_logger().info("Camera orientation (WORLD axes):")
        self.get_logger().info(f"   X-axis: {cam_x}")
        self.get_logger().info(f"   Y-axis: {cam_y}")
        self.get_logger().info(f"   Z-axis: {cam_z}")
        self.get_logger().info(f"Current view_id: {view_id}")
        self.get_logger().info("==============================\n")


def main() -> None:
    rclpy.init()
    node: Meca500ViewNode | None = None

    try:
        node = Meca500ViewNode()
        node.run_interactive()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.get_logger().info(
                    "[ROBOT] Deactivating robot and disconnecting ..."
                )
                try:
                    node.robot.DeactivateRobot()
                    node.robot.WaitDeactivated()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    node.robot.Disconnect()
                except Exception:  # noqa: BLE001
                    pass
                node.get_logger().info(
                    "[ROBOT] Robot deactivated and disconnected."
                )
            except Exception:  # noqa: BLE001
                pass

            node.destroy_node()

        rclpy.shutdown()
        print("[ROS2] Shutdown complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
