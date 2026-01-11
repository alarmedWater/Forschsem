#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Set, Tuple

import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2  # noqa: F401  (optional, for deep checks)

from strawberry_msgs.msg import FrameInfo


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def ns_to_str(ns: int) -> str:
    sec = ns // 1_000_000_000
    nsec = ns % 1_000_000_000
    return f"{sec}.{nsec:09d}"


@dataclass
class FrameRecord:
    ns: int
    cam: Optional[Tuple[int, int, int]] = None   # (frame_index, plant_id, view_id)
    seg: Optional[Tuple[int, int, int]] = None
    dm: Optional[Tuple[int, int, int]] = None
    cloud: Optional[Tuple[int, int, int]] = None

    # diagnostics
    seg_instances: Optional[int] = None
    cloud_points: Optional[int] = None


class SyncAuditNode(Node):
    """
    Audit synchronization and metadata consistency across the pipeline.

    Checks:
    - camera FrameInfo matches seg FrameInfo / depth_mask FrameInfo (by stamp)
    - label/depth/cloud stamps match their corresponding FrameInfo
    - per-plant view coverage (expected 0,1,2)
    """

    def __init__(self) -> None:
        super().__init__("strawberry_sync_audit")

        # -------- Parameters --------
        self.declare_parameter("expected_views_per_plant", 3)
        self.declare_parameter("sync_queue_size", 200)
        self.declare_parameter("sync_slop", 0.02)  # tighter default for safety
        self.declare_parameter("print_every_n_frames", 20)
        self.declare_parameter("record_ttl_s", 10.0)

        # topics
        self.declare_parameter("cam_frame_info_topic", "/camera/frame_info")
        self.declare_parameter("seg_frame_info_topic", "/seg/frame_info")
        self.declare_parameter("dm_frame_info_topic", "/seg/frame_info_depth_masked")

        self.declare_parameter("label_topic", "/seg/label_image")
        self.declare_parameter("depth_masked_topic", "/seg/depth_masked")
        self.declare_parameter("cloud_topic", "/seg/strawberry_cloud")

        self._expected_views = int(self.get_parameter("expected_views_per_plant").value)
        self._queue = max(10, int(self.get_parameter("sync_queue_size").value))
        self._slop = float(self.get_parameter("sync_slop").value)
        self._print_every = max(1, int(self.get_parameter("print_every_n_frames").value))
        self._ttl_s = float(self.get_parameter("record_ttl_s").value)

        cam_fi_t = str(self.get_parameter("cam_frame_info_topic").value)
        seg_fi_t = str(self.get_parameter("seg_frame_info_topic").value)
        dm_fi_t = str(self.get_parameter("dm_frame_info_topic").value)

        label_t = str(self.get_parameter("label_topic").value)
        dm_depth_t = str(self.get_parameter("depth_masked_topic").value)
        cloud_t = str(self.get_parameter("cloud_topic").value)

        self.get_logger().info(
            "SyncAuditNode starting:\n"
            f"  cam_frame_info_topic = {cam_fi_t}\n"
            f"  seg_frame_info_topic = {seg_fi_t}\n"
            f"  dm_frame_info_topic  = {dm_fi_t}\n"
            f"  label_topic          = {label_t}\n"
            f"  depth_masked_topic   = {dm_depth_t}\n"
            f"  cloud_topic          = {cloud_t}\n"
            f"  sync_queue_size      = {self._queue}\n"
            f"  sync_slop            = {self._slop}\n"
            f"  expected_views/plant = {self._expected_views}\n"
            f"  record_ttl_s         = {self._ttl_s}"
        )

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
        )

        # -------- Internal state --------
        self._records: Dict[int, FrameRecord] = {}
        self._order: Deque[int] = deque()  # stamp ns ordering for TTL cleanup

        self._views_cam: Dict[int, Set[int]] = defaultdict(set)
        self._views_cloud: Dict[int, Set[int]] = defaultdict(set)

        self._last_cam_plant: Optional[int] = None

        self._n_cam = 0
        self._n_errors = 0

        # -------- Subscribers + synchronizers --------
        # We sync "data topic" with its corresponding FrameInfo to verify stamp equality
        self._sub_cam_fi = message_filters.Subscriber(self, FrameInfo, cam_fi_t, qos_profile=qos)
        self._sub_seg_fi = message_filters.Subscriber(self, FrameInfo, seg_fi_t, qos_profile=qos)
        self._sub_dm_fi_a = message_filters.Subscriber(self, FrameInfo, dm_fi_t, qos_profile=qos)
        self._sub_dm_fi_b = message_filters.Subscriber(self, FrameInfo, dm_fi_t, qos_profile=qos)

        self._sub_label = message_filters.Subscriber(self, Image, label_t, qos_profile=qos)
        self._sub_dm_depth = message_filters.Subscriber(self, Image, dm_depth_t, qos_profile=qos)
        self._sub_cloud = message_filters.Subscriber(self, PointCloud2, cloud_t, qos_profile=qos)

        # label + seg frameinfo
        self._ts_label = message_filters.ApproximateTimeSynchronizer(
            [self._sub_label, self._sub_seg_fi],
            queue_size=self._queue,
            slop=self._slop,
        )
        self._ts_label.registerCallback(self._cb_label_seg)

        # depth_masked + dm frameinfo
        self._ts_dm = message_filters.ApproximateTimeSynchronizer(
            [self._sub_dm_depth, self._sub_dm_fi_a],
            queue_size=self._queue,
            slop=self._slop,
        )
        self._ts_dm.registerCallback(self._cb_depthmask_dmfi)

        # cloud + dm frameinfo (separate subscriber instance)
        self._ts_cloud = message_filters.ApproximateTimeSynchronizer(
            [self._sub_cloud, self._sub_dm_fi_b],
            queue_size=self._queue,
            slop=self._slop,
        )
        self._ts_cloud.registerCallback(self._cb_cloud_dmfi)

        # cam frameinfo alone (source of truth)
        # (No message_filters needed; we just cache it)
        self.create_subscription(FrameInfo, cam_fi_t, self._cb_cam_fi, 50)

        # periodic status
        self.create_timer(1.0, self._tick_status)

    # ---------------- Record helpers ----------------

    def _get_rec(self, ns: int) -> FrameRecord:
        rec = self._records.get(ns)
        if rec is None:
            rec = FrameRecord(ns=ns)
            self._records[ns] = rec
            self._order.append(ns)
        return rec

    def _cleanup_old(self) -> None:
        now_ns = stamp_to_ns(self.get_clock().now().to_msg())
        ttl_ns = int(self._ttl_s * 1_000_000_000)
        while self._order:
            ns0 = self._order[0]
            if now_ns - ns0 <= ttl_ns:
                break
            self._order.popleft()
            self._records.pop(ns0, None)

    # ---------------- Callbacks ----------------

    def _cb_cam_fi(self, fi: FrameInfo) -> None:
        ns = stamp_to_ns(fi.header.stamp)
        rec = self._get_rec(ns)

        meta = (int(fi.frame_index), int(fi.plant_id), int(fi.view_id))
        rec.cam = meta

        frame_index, plant_id, view_id = meta

        # plant change summary
        if self._last_cam_plant is None:
            self._last_cam_plant = plant_id
        elif plant_id != self._last_cam_plant:
            self._report_plant_summary(self._last_cam_plant)
            self._last_cam_plant = plant_id

        self._views_cam[plant_id].add(view_id)
        self._n_cam += 1

        if self._n_cam % self._print_every == 0:
            self.get_logger().info(
                f"[CAM] frames={self._n_cam} last={frame_index} plant={plant_id} view={view_id} "
                f"stamp={ns_to_str(ns)} errors={self._n_errors}"
            )

    def _cb_label_seg(self, label_msg: Image, fi_seg: FrameInfo) -> None:
        ns_l = stamp_to_ns(label_msg.header.stamp)
        ns_f = stamp_to_ns(fi_seg.header.stamp)
        if abs(ns_l - ns_f) > int(self._slop * 1e9):
            self._n_errors += 1
            self.get_logger().error(
                f"[SEG] stamp mismatch label={ns_to_str(ns_l)} fi={ns_to_str(ns_f)}"
            )

        ns = ns_l
        rec = self._get_rec(ns)
        meta = (int(fi_seg.frame_index), int(fi_seg.plant_id), int(fi_seg.view_id))
        rec.seg = meta

        # count instances quickly
        try:
            # mono16 label image
            data = np.frombuffer(label_msg.data, dtype=np.uint16)
            if data.size > 0:
                rec.seg_instances = int(data.max())
        except Exception:
            rec.seg_instances = None

        self._check_against_cam(ns, "SEG", meta)

    def _cb_depthmask_dmfi(self, depth_msg: Image, fi_dm: FrameInfo) -> None:
        ns_d = stamp_to_ns(depth_msg.header.stamp)
        ns_f = stamp_to_ns(fi_dm.header.stamp)
        if abs(ns_d - ns_f) > int(self._slop * 1e9):
            self._n_errors += 1
            self.get_logger().error(
                f"[DM] stamp mismatch depth_masked={ns_to_str(ns_d)} fi={ns_to_str(ns_f)}"
            )

        ns = ns_d
        rec = self._get_rec(ns)
        meta = (int(fi_dm.frame_index), int(fi_dm.plant_id), int(fi_dm.view_id))
        rec.dm = meta

        self._check_against_cam(ns, "DM", meta)

    def _cb_cloud_dmfi(self, cloud_msg: PointCloud2, fi_dm: FrameInfo) -> None:
        ns_c = stamp_to_ns(cloud_msg.header.stamp)
        ns_f = stamp_to_ns(fi_dm.header.stamp)
        if abs(ns_c - ns_f) > int(self._slop * 1e9):
            self._n_errors += 1
            self.get_logger().error(
                f"[CLOUD] stamp mismatch cloud={ns_to_str(ns_c)} fi={ns_to_str(ns_f)}"
            )

        ns = ns_c
        rec = self._get_rec(ns)
        meta = (int(fi_dm.frame_index), int(fi_dm.plant_id), int(fi_dm.view_id))
        rec.cloud = meta

        # point count (works for create_cloud_xyz32: height=1, width=N)
        try:
            rec.cloud_points = int(cloud_msg.width) * int(cloud_msg.height)
        except Exception:
            rec.cloud_points = None

        _, plant_id, view_id = meta
        self._views_cloud[plant_id].add(view_id)

        self._check_against_cam(ns, "CLOUD", meta)

    def _check_against_cam(self, ns: int, stage: str, meta: Tuple[int, int, int]) -> None:
        rec = self._records.get(ns)
        if rec is None or rec.cam is None:
            # cam fi maybe not received yet; not fatal
            return

        if rec.cam != meta:
            self._n_errors += 1
            self.get_logger().error(
                f"[{stage}] META mismatch at stamp={ns_to_str(ns)} "
                f"cam={rec.cam} vs {stage.lower()}={meta}"
            )

    # ---------------- Reporting ----------------

    def _report_plant_summary(self, plant_id: int) -> None:
        cam_views = sorted(list(self._views_cam.get(plant_id, set())))
        cloud_views = sorted(list(self._views_cloud.get(plant_id, set())))

        expected = {0, 1, 2} if self._expected_views == 3 else set(range(self._expected_views))
        missing_cam = sorted(list(expected - set(cam_views)))
        missing_cloud = sorted(list(expected - set(cloud_views)))

        self.get_logger().info(
            "----- PLANT SUMMARY -----\n"
            f"plant={plant_id}\n"
            f"  cam_views   = {cam_views} missing={missing_cam}\n"
            f"  cloud_views = {cloud_views} missing={missing_cloud}\n"
            "-------------------------"
        )

    def _tick_status(self) -> None:
        self._cleanup_old()

        # simple “pipeline health”
        n = len(self._records)
        if n == 0:
            self.get_logger().warning("No records yet (are topics publishing?).")
            return

        # count how many have each stage
        cam_ok = sum(1 for r in self._records.values() if r.cam is not None)
        seg_ok = sum(1 for r in self._records.values() if r.seg is not None)
        dm_ok = sum(1 for r in self._records.values() if r.dm is not None)
        cloud_ok = sum(1 for r in self._records.values() if r.cloud is not None)

        self.get_logger().info(
            f"[STATUS] records={n} cam={cam_ok} seg={seg_ok} dm={dm_ok} cloud={cloud_ok} "
            f"errors={self._n_errors}"
        )


def main() -> None:
    rclpy.init()
    node = SyncAuditNode()
    try:
        rclpy.spin(node)
    finally:
        # last plant summary (if we have one)
        if node._last_cam_plant is not None:
            node._report_plant_summary(node._last_cam_plant)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
