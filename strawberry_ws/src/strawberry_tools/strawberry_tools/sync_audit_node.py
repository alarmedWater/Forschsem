#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exact sync audit tool for the strawberry pipeline (ROS2).

Checks the pipeline conventions:

- QoS: qos_profile_sensor_data (BEST_EFFORT)
- Token stamp: FrameInfo.source_stamp (fallback: FrameInfo.header.stamp)
- "Exact sync" expectation:
    label.header.stamp      == token_stamp (from /seg/frame_info)
    depth_masked.header.stamp == token_stamp (from /seg/frame_info_depth_masked)
    cloud.header.stamp        == token_stamp (from /seg/frame_info_depth_masked)

It logs mismatches and per-plant view completeness.
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, PointCloud2
from strawberry_msgs.msg import FrameInfo  # type: ignore


def _time_to_ns(t: Any) -> Optional[int]:
    if t is None:
        return None
    if hasattr(t, "sec") and hasattr(t, "nanosec"):
        return int(t.sec) * 1_000_000_000 + int(t.nanosec)
    return None


def _stamp_ns_from_header(msg: Any) -> Optional[int]:
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        return _time_to_ns(msg.header.stamp)
    return None


def token_stamp_ns(fi: FrameInfo) -> Optional[int]:
    # Convention: FrameInfo.source_stamp if exists, else header.stamp
    if hasattr(fi, "source_stamp"):
        ns = _time_to_ns(getattr(fi, "source_stamp"))
        if ns is not None:
            return ns
    return _stamp_ns_from_header(fi)


def frame_uid(fi: FrameInfo) -> str:
    return str(getattr(fi, "frame_uid", ""))


@dataclass
class TopicStats:
    recv_count: int = 0
    last_recv_ns: Optional[int] = None
    gaps_s: Deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def on_msg(self, now_ns: int) -> None:
        self.recv_count += 1
        if self.last_recv_ns is not None:
            gap = (now_ns - self.last_recv_ns) / 1e9
            if gap >= 0.0:
                self.gaps_s.append(gap)
        self.last_recv_ns = now_ns

    def rate_hz(self) -> Optional[float]:
        if not self.gaps_s:
            return None
        m = sum(self.gaps_s) / len(self.gaps_s)
        return (1.0 / m) if m > 0.0 else None


@dataclass
class PlantState:
    plant: int
    last_seen_ns: int
    cam_views: set = field(default_factory=set)
    seg_views: set = field(default_factory=set)
    dm_views: set = field(default_factory=set)
    label_views: set = field(default_factory=set)
    depth_views: set = field(default_factory=set)
    cloud_views: set = field(default_factory=set)


class SyncAuditExactNode(Node):
    def __init__(self) -> None:
        super().__init__("strawberry_sync_audit_exact")

        # Topics
        self.declare_parameter("cam_frame_info_topic", "/camera/frame_info")
        self.declare_parameter("seg_frame_info_topic", "/seg/frame_info")
        self.declare_parameter("dm_frame_info_topic", "/seg/frame_info_depth_masked")
        self.declare_parameter("label_topic", "/seg/label_image")
        self.declare_parameter("depth_masked_topic", "/seg/depth_masked")
        self.declare_parameter("cloud_topic", "/seg/strawberry_cloud")

        # Audit behavior
        self.declare_parameter("expected_views_per_plant", 3)
        self.declare_parameter("ttl_s", 10.0)
        self.declare_parameter("status_period_s", 1.0)
        self.declare_parameter("topic_stats_period_s", 2.0)

        # If True: require exact match. If False: allow slop (nanoseconds).
        self.declare_parameter("exact", True)
        self.declare_parameter("slop_s", 0.02)

        # Logging
        self.declare_parameter("log_dir", os.path.expanduser("~/strawberry_audit_logs"))
        self.declare_parameter("log_basename", "sync_audit_exact")

        self.cam_fi_t = str(self.get_parameter("cam_frame_info_topic").value)
        self.seg_fi_t = str(self.get_parameter("seg_frame_info_topic").value)
        self.dm_fi_t = str(self.get_parameter("dm_frame_info_topic").value)
        self.label_t = str(self.get_parameter("label_topic").value)
        self.depth_t = str(self.get_parameter("depth_masked_topic").value)
        self.cloud_t = str(self.get_parameter("cloud_topic").value)

        self.expected_views = int(self.get_parameter("expected_views_per_plant").value)
        self.ttl_ns = int(float(self.get_parameter("ttl_s").value) * 1e9)

        self.exact = bool(self.get_parameter("exact").value)
        self.slop_ns = int(float(self.get_parameter("slop_s").value) * 1e9)

        # File logger
        log_dir = str(self.get_parameter("log_dir").value)
        base = str(self.get_parameter("log_basename").value)
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{base}_{ts}.txt")

        self._pylog = logging.getLogger("strawberry_sync_audit_exact")
        self._pylog.setLevel(logging.INFO)
        fh = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self._pylog.handlers = [fh]

        # Indices: token_stamp_ns -> (plant, view, frame, uid)
        self._idx_cam: Dict[int, Tuple[int, int, int, str]] = {}
        self._idx_seg: Dict[int, Tuple[int, int, int, str]] = {}
        self._idx_dm: Dict[int, Tuple[int, int, int, str]] = {}

        # For TTL cleanup, keep insertion order
        self._idx_cam_q: Deque[int] = deque(maxlen=5000)
        self._idx_seg_q: Deque[int] = deque(maxlen=5000)
        self._idx_dm_q: Deque[int] = deque(maxlen=5000)

        self._plants: Dict[int, PlantState] = {}
        self._errors = 0

        self._topic_stats: Dict[str, TopicStats] = {}

        # Subscriptions (QoS sensor)
        self.create_subscription(FrameInfo, self.cam_fi_t, self._on_cam_fi, qos_profile_sensor_data)
        self.create_subscription(FrameInfo, self.seg_fi_t, self._on_seg_fi, qos_profile_sensor_data)
        self.create_subscription(FrameInfo, self.dm_fi_t, self._on_dm_fi, qos_profile_sensor_data)

        self.create_subscription(Image, self.label_t, self._on_label, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_t, self._on_depth, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, self.cloud_t, self._on_cloud, qos_profile_sensor_data)

        # Timers
        self.create_timer(float(self.get_parameter("status_period_s").value), self._on_status)
        self.create_timer(float(self.get_parameter("topic_stats_period_s").value), self._on_topic_stats)

        self._logi("SyncAuditExactNode starting:")
        self._logi(f"  cam_frame_info_topic = {self.cam_fi_t}")
        self._logi(f"  seg_frame_info_topic = {self.seg_fi_t}")
        self._logi(f"  dm_frame_info_topic  = {self.dm_fi_t}")
        self._logi(f"  label_topic          = {self.label_t}")
        self._logi(f"  depth_masked_topic   = {self.depth_t}")
        self._logi(f"  cloud_topic          = {self.cloud_t}")
        self._logi(f"  exact                = {self.exact}")
        self._logi(f"  slop_s               = {self.slop_ns/1e9:.3f}")
        self._logi(f"  log_path             = {self.log_path}")

    # ---------- logging ----------
    def _logi(self, msg: str) -> None:
        self.get_logger().info(msg)
        self._pylog.info(msg)

    def _logw(self, msg: str) -> None:
        self.get_logger().warning(msg)
        self._pylog.warning(msg)

    # ---------- small helpers ----------
    def _stats(self, topic: str) -> None:
        now_ns = self.get_clock().now().nanoseconds
        st = self._topic_stats.get(topic)
        if st is None:
            st = TopicStats()
            self._topic_stats[topic] = st
        st.on_msg(now_ns)

    def _upsert_plant(self, plant: int) -> PlantState:
        now_ns = self.get_clock().now().nanoseconds
        ps = self._plants.get(plant)
        if ps is None:
            ps = PlantState(plant=plant, last_seen_ns=now_ns)
            self._plants[plant] = ps
        ps.last_seen_ns = now_ns
        return ps

    def _cleanup_ttl(self) -> None:
        now_ns = self.get_clock().now().nanoseconds

        # plant ttl summaries
        dead = [p for p, ps in self._plants.items() if (now_ns - ps.last_seen_ns) > self.ttl_ns]
        for p in sorted(dead):
            self._finalize_plant(p, reason="ttl")

        # index cleanup: we just drop older than ttl from the left
        def _drop_old(q: Deque[int], idx: Dict[int, Tuple[int, int, int, str]]) -> None:
            while q:
                s = q[0]
                # We don't store recv time; good-enough: keep bounded by deque maxlen + TTL handled by overwrite.
                # If you want strict TTL, store recv times too.
                if s in idx:
                    break
                q.popleft()

        _drop_old(self._idx_cam_q, self._idx_cam)
        _drop_old(self._idx_seg_q, self._idx_seg)
        _drop_old(self._idx_dm_q, self._idx_dm)

    def _finalize_plant(self, plant: int, reason: str) -> None:
        ps = self._plants.get(plant)
        if ps is None:
            return

        exp = set(range(self.expected_views))
        self._logi("----- PLANT SUMMARY -----")
        self._logi(f"plant={plant} reason={reason}")
        self._logi(f"  cam   ={sorted(ps.cam_views)}   missing={sorted(exp-ps.cam_views)}")
        self._logi(f"  seg   ={sorted(ps.seg_views)}   missing={sorted(exp-ps.seg_views)}")
        self._logi(f"  dm    ={sorted(ps.dm_views)}    missing={sorted(exp-ps.dm_views)}")
        self._logi(f"  label ={sorted(ps.label_views)} missing={sorted(exp-ps.label_views)}")
        self._logi(f"  depth ={sorted(ps.depth_views)} missing={sorted(exp-ps.depth_views)}")
        self._logi(f"  cloud ={sorted(ps.cloud_views)} missing={sorted(exp-ps.cloud_views)}")
        self._logi("-------------------------")

        del self._plants[plant]

    def _match_idx(
        self,
        idx: Dict[int, Tuple[int, int, int, str]],
        stamp_ns: int,
    ) -> Optional[Tuple[int, int, int, str, int]]:
        """
        Return (plant, view, frame, uid, matched_stamp_ns)
        Exact: direct lookup only.
        Non-exact: nearest within slop.
        """
        if self.exact:
            v = idx.get(stamp_ns)
            return (v[0], v[1], v[2], v[3], stamp_ns) if v is not None else None

        # slop match (nearest)
        best = None
        best_dt = None
        for s, v in idx.items():
            dt = abs(s - stamp_ns)
            if dt <= self.slop_ns and (best_dt is None or dt < best_dt):
                best_dt = dt
                best = (v[0], v[1], v[2], v[3], s)
        return best

    # ---------- FrameInfo handlers ----------
    def _on_cam_fi(self, fi: FrameInfo) -> None:
        self._stats(self.cam_fi_t)
        s = token_stamp_ns(fi)
        if s is None:
            return
        plant = int(fi.plant_id)
        view = int(fi.view_id)
        frame = int(fi.frame_index)
        uid = frame_uid(fi)

        self._idx_cam[s] = (plant, view, frame, uid)
        self._idx_cam_q.append(s)
        self._upsert_plant(plant).cam_views.add(view)

    def _on_seg_fi(self, fi: FrameInfo) -> None:
        self._stats(self.seg_fi_t)
        s = token_stamp_ns(fi)
        if s is None:
            return
        plant = int(fi.plant_id)
        view = int(fi.view_id)
        frame = int(fi.frame_index)
        uid = frame_uid(fi)

        self._idx_seg[s] = (plant, view, frame, uid)
        self._idx_seg_q.append(s)
        self._upsert_plant(plant).seg_views.add(view)

    def _on_dm_fi(self, fi: FrameInfo) -> None:
        self._stats(self.dm_fi_t)
        s = token_stamp_ns(fi)
        if s is None:
            return
        plant = int(fi.plant_id)
        view = int(fi.view_id)
        frame = int(fi.frame_index)
        uid = frame_uid(fi)

        self._idx_dm[s] = (plant, view, frame, uid)
        self._idx_dm_q.append(s)
        self._upsert_plant(plant).dm_views.add(view)

    # ---------- Data topic handlers ----------
    def _on_label(self, msg: Image) -> None:
        self._stats(self.label_t)
        stamp = _stamp_ns_from_header(msg)
        if stamp is None:
            return

        m = self._match_idx(self._idx_seg, stamp)
        if m is None:
            # fallback: sometimes label might align to cam stamp
            m = self._match_idx(self._idx_cam, stamp)
            if m is None:
                return

        plant, view, frame, uid, matched_stamp = m
        self._upsert_plant(plant).label_views.add(view)

        if self.exact and matched_stamp != stamp:
            self._errors += 1
            self._logw(f"[LABEL] stamp mismatch?! plant={plant} view={view} uid={uid}")

    def _on_depth(self, msg: Image) -> None:
        self._stats(self.depth_t)
        stamp = _stamp_ns_from_header(msg)
        if stamp is None:
            return

        m = self._match_idx(self._idx_dm, stamp)
        if m is None:
            # fallback chain
            m = self._match_idx(self._idx_seg, stamp) or self._match_idx(self._idx_cam, stamp)
            if m is None:
                return

        plant, view, frame, uid, matched_stamp = m
        self._upsert_plant(plant).depth_views.add(view)

        if self.exact and matched_stamp != stamp:
            self._errors += 1
            self._logw(f"[DEPTH] stamp mismatch?! plant={plant} view={view} uid={uid}")

    def _on_cloud(self, msg: PointCloud2) -> None:
        self._stats(self.cloud_t)
        stamp = _stamp_ns_from_header(msg)
        if stamp is None:
            return

        m = self._match_idx(self._idx_dm, stamp)
        if m is None:
            m = self._match_idx(self._idx_seg, stamp) or self._match_idx(self._idx_cam, stamp)
            if m is None:
                return

        plant, view, frame, uid, matched_stamp = m
        self._upsert_plant(plant).cloud_views.add(view)

        if self.exact and matched_stamp != stamp:
            self._errors += 1
            self._logw(f"[CLOUD] stamp mismatch?! plant={plant} view={view} uid={uid}")

    # ---------- timers ----------
    def _on_status(self) -> None:
        self._cleanup_ttl()
        self._logi(
            f"[STATUS] plants={len(self._plants)} "
            f"idx(cam/seg/dm)={len(self._idx_cam)}/{len(self._idx_seg)}/{len(self._idx_dm)} "
            f"errors={self._errors}"
        )

    def _on_topic_stats(self) -> None:
        if not self._topic_stats:
            self._logw("[STATS] no messages yet")
            return

        lines = ["----- TOPIC STATS -----"]
        for t, st in sorted(self._topic_stats.items()):
            hz = st.rate_hz()
            hz_s = f"{hz:.2f}Hz" if hz is not None else "n/a"
            lines.append(f"{t}: count={st.recv_count} rate={hz_s}")
        lines.append("-----------------------")
        for ln in lines:
            self._logi(ln)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = SyncAuditExactNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt - shutting down.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
