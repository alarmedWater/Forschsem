#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from typing import Optional, Tuple

import message_filters
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from strawberry_msgs.msg import FrameInfo


def stamp_tuple(stamp) -> Tuple[int, int]:
    return int(stamp.sec), int(stamp.nanosec)


def fmt_stamp(sec: int, nsec: int) -> str:
    return f"{sec}.{nsec:09d}"


class StampMonitor(Node):
    """
    Exact stamp monitor for:
      /camera/frame_info (FrameInfo)
      /camera/color/image_raw (Image)
      /camera/aligned_depth_to_color/image_raw (Image)

    Uses message_filters.TimeSynchronizer -> checks only synchronized triples.

    Also tracks per-topic last-seen wall time (watchdog) to reveal stalls/drops.
    """

    def __init__(self) -> None:
        super().__init__("stamp_monitor")

        # -------- Params (optional) --------
        self.declare_parameter("frame_info_topic", "/camera/frame_info")
        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")

        self.declare_parameter("queue_size", 200)
        self.declare_parameter("watchdog_sec", 2.0)      # warn if topic hasn't produced for this many seconds
        self.declare_parameter("log_every_n_ok", 50)      # print OK line every N synced triples

        self.fi_topic = str(self.get_parameter("frame_info_topic").value)
        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)

        self.queue_size = max(1, int(self.get_parameter("queue_size").value))
        self.watchdog_sec = float(self.get_parameter("watchdog_sec").value)
        self.log_every_n_ok = max(1, int(self.get_parameter("log_every_n_ok").value))

        # -------- Stats --------
        self.synced_count = 0
        self.mismatch_count = 0

        # last seen (ROS stamps + wall time) for each topic (independent of sync)
        self._last_fi_stamp: Optional[Tuple[int, int]] = None
        self._last_rgb_stamp: Optional[Tuple[int, int]] = None
        self._last_depth_stamp: Optional[Tuple[int, int]] = None

        self._last_fi_wall: float = 0.0
        self._last_rgb_wall: float = 0.0
        self._last_depth_wall: float = 0.0

        # -------- Subscriptions (raw) for watchdog bookkeeping --------
        # We subscribe twice: once via message_filters for sync, once lightweight for "last seen".
        # Using qos_profile_sensor_data everywhere to match your pipeline.
        self.create_subscription(FrameInfo, self.fi_topic, self._fi_watch_cb, qos_profile_sensor_data)
        self.create_subscription(Image, self.rgb_topic, self._rgb_watch_cb, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self._depth_watch_cb, qos_profile_sensor_data)

        # -------- message_filters exact sync --------
        self._sub_fi = message_filters.Subscriber(self, FrameInfo, self.fi_topic, qos_profile=qos_profile_sensor_data)
        self._sub_rgb = message_filters.Subscriber(self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        self._sub_depth = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)

        self._ts = message_filters.TimeSynchronizer(
            [self._sub_fi, self._sub_rgb, self._sub_depth],
            queue_size=self.queue_size,
        )
        self._ts.registerCallback(self._sync_cb)

        # watchdog timer
        self._watchdog_timer = self.create_timer(0.5, self._watchdog_tick)

        self.get_logger().info(
            "StampMonitor running (EXACT sync):\n"
            f"  frame_info_topic = {self.fi_topic}\n"
            f"  rgb_topic        = {self.rgb_topic}\n"
            f"  depth_topic      = {self.depth_topic}\n"
            f"  queue_size       = {self.queue_size}\n"
            f"  watchdog_sec     = {self.watchdog_sec}\n"
            f"  log_every_n_ok   = {self.log_every_n_ok}"
        )

    # ---------------- Watch callbacks (independent) ----------------

    def _fi_watch_cb(self, m: FrameInfo) -> None:
        self._last_fi_stamp = stamp_tuple(m.header.stamp)
        self._last_fi_wall = time.time()

    def _rgb_watch_cb(self, m: Image) -> None:
        self._last_rgb_stamp = stamp_tuple(m.header.stamp)
        self._last_rgb_wall = time.time()

    def _depth_watch_cb(self, m: Image) -> None:
        self._last_depth_stamp = stamp_tuple(m.header.stamp)
        self._last_depth_wall = time.time()

    # ---------------- Exact sync callback ----------------

    def _sync_cb(self, fi: FrameInfo, rgb: Image, depth: Image) -> None:
        self.synced_count += 1

        fi_s = stamp_tuple(fi.header.stamp)
        rgb_s = stamp_tuple(rgb.header.stamp)
        depth_s = stamp_tuple(depth.header.stamp)

        ok = (fi_s == rgb_s == depth_s)

        if not ok:
            self.mismatch_count += 1
            self.get_logger().error(
                "STAMP MISMATCH (SYNCED TRIPLE) "
                f"fi={fmt_stamp(*fi_s)} rgb={fmt_stamp(*rgb_s)} depth={fmt_stamp(*depth_s)} "
                f"| uid={getattr(fi, 'frame_uid', '')} plant={int(fi.plant_id)} view={int(fi.view_id)} frame={int(fi.frame_index)}"
            )
            return

        # print periodic OK so you know it's alive
        if (self.synced_count % self.log_every_n_ok) == 0:
            self.get_logger().info(
                "STAMP OK "
                f"{fmt_stamp(*fi_s)} "
                f"| uid={getattr(fi, 'frame_uid', '')} plant={int(fi.plant_id)} view={int(fi.view_id)} frame={int(fi.frame_index)} "
                f"| synced={self.synced_count} mismatches={self.mismatch_count}"
            )

    # ---------------- Watchdog ----------------

    def _watchdog_tick(self) -> None:
        now = time.time()

        def warn_if_stale(name: str, last_wall: float, last_stamp: Optional[Tuple[int, int]]) -> None:
            if last_wall <= 0.0:
                return
            dt = now - last_wall
            if dt > self.watchdog_sec:
                stamp_str = "n/a" if last_stamp is None else fmt_stamp(*last_stamp)
                self.get_logger().warning(f"Watchdog: no '{name}' messages for {dt:.2f}s (last stamp {stamp_str})")

        warn_if_stale("frame_info", self._last_fi_wall, self._last_fi_stamp)
        warn_if_stale("rgb", self._last_rgb_wall, self._last_rgb_stamp)
        warn_if_stale("depth", self._last_depth_wall, self._last_depth_stamp)


def main() -> None:
    rclpy.init()
    node = StampMonitor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
