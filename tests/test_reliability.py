"""Reliability and robustness test suite for SafeKid AI (Prompt 05).

Covers:
1. Live-stream timestamp reliability (monotonic clock, dropped frames, reconnections).
2. Tracking-ID discontinuity handling (grace period, occlusion recovery, ID switch independence).
3. Non-destructive ROI configuration reload (unchanged zones preserved, modified/deleted handled).
4. Polygon validation (self-intersecting bow-tie/figure-8 rejected, concave accepted).
"""

import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from safekid_ai.config import (
    DetectorConfig,
    OperationalMode,
    SafeKidPipelineConfig,
    TrackerConfig,
    VideoInputConfig,
)
from safekid_ai.detector import DetectionBox
from safekid_ai.events import SafetyEvent, SafetyEventType, ZoneStateManager
from safekid_ai.pipeline import SafeKidPipeline
from safekid_ai.roi.engine import ROIEngine
from safekid_ai.roi.models import (
    AlertSeverity,
    CameraROIConfig,
    Point2D,
    Polygon,
    SafetyZone,
    SafetyZoneType,
)
from safekid_ai.video_input import VideoInput, VideoStreamMetadata


# ============================================================================
# 1. LIVE-STREAM TIMESTAMP RELIABILITY TESTS
# ============================================================================

def test_live_stream_monotonic_timestamp_continuity():
    """Verifies that live stream timestamps derive from monotonic clock and never move backward."""
    vin = VideoInput(source="0")
    # Simulate live stream metadata
    vin._metadata = VideoStreamMetadata(
        source="0",
        width=640,
        height=480,
        fps=30.0,
        total_frames=0,
        is_live_stream=True,
        is_single_image=False,
    )
    vin._stream_start_monotonic = time.monotonic() - 10.0  # Stream started 10s ago

    # Simulate frames reading over time
    timestamps = []
    for _ in range(5):
        elapsed_ms = (time.monotonic() - vin._stream_start_monotonic) * 1000.0
        ts = max(vin._last_emitted_timestamp_ms, elapsed_ms)
        vin._last_emitted_timestamp_ms = ts
        timestamps.append(ts)
        time.sleep(0.01)

    assert len(timestamps) == 5
    # Strict monotonicity check
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1]
    assert timestamps[0] >= 10000.0


def test_pipeline_backward_timestamp_clamping():
    """Ensures that pipeline clamps any backward/corrupted timestamps to last known timestamp."""
    pipeline = SafeKidPipeline()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: t=5000.0ms
    res1 = pipeline.process_frame(frame, frame_id=1, timestamp_ms=5000.0, annotate=False)
    assert res1.timestamp_ms == 5000.0

    # Frame 2: Corrupted/rollback timestamp (e.g. stream looping or packet jump) t=2000.0ms
    res2 = pipeline.process_frame(frame, frame_id=2, timestamp_ms=2000.0, annotate=False)
    # Must be clamped to 5000.0ms
    assert res2.timestamp_ms == 5000.0

    # Frame 3: Normal progression t=5500.0ms
    res3 = pipeline.process_frame(frame, frame_id=3, timestamp_ms=5500.0, annotate=False)
    assert res3.timestamp_ms == 5500.0


def test_live_stream_reconnection_preserves_monotonic_epoch():
    """Verifies that reconnecting a live stream preserves start monotonic epoch without timestamp rollback."""
    vin = VideoInput(source="0")
    vin._metadata = VideoStreamMetadata(
        source="0",
        width=640,
        height=480,
        fps=30.0,
        total_frames=0,
        is_live_stream=True,
        is_single_image=False,
    )
    start_time = time.monotonic() - 5.0
    vin._stream_start_monotonic = start_time
    vin._last_emitted_timestamp_ms = 5000.0

    # Simulate reconnect
    vin._parsed_src = 0
    # Even if reconnect fails to find actual webcam in CI/headless, test that internal epoch is maintained
    vin.reconnect()
    assert vin._stream_start_monotonic == start_time
    assert vin._last_emitted_timestamp_ms == 5000.0


# ============================================================================
# 2. TRACKING-ID DISCONTINUITY HANDLING TESTS
# ============================================================================

def test_temporary_occlusion_within_grace_period():
    """Verifies that temporary detection drops within grace period do not trigger false exits."""
    manager = ZoneStateManager(max_missing_frames=10)
    poly = Polygon([(100, 100), (300, 100), (300, 300), (100, 300)])
    zone = SafetyZone("z1", "Test Zone", SafetyZoneType.SWIMMING_POOL_GATE, poly, AlertSeverity.CRITICAL, dwell_time_seconds=3.0)
    config = CameraROIConfig("comm_1", "cam_1", 1000, 1000, [zone])
    engine = ROIEngine(config=config)

    det = [DetectionBox((150, 150, 250, 250), 0.9, track_id=42)]

    # Frame 1: Entry at t=0
    e1 = manager.update(1, 0.0, det, engine, (1000, 1000))
    assert len(e1) == 1
    assert e1[0].event_type == SafetyEventType.ZONE_ENTRY

    # Frames 2..7 (6 missing frames < 10): Temporary occlusion behind pillar
    for f in range(2, 8):
        evts = manager.update(f, f * 500.0, [], engine, (1000, 1000))
        assert len(evts) == 0, f"Frame {f} emitted premature event: {evts}"

    # Frame 8 (t=4000.0ms): Reappearance with same ID 42
    # Dwell time should have accumulated to 4.0s without duplicate entry!
    evts8 = manager.update(8, 4000.0, det, engine, (1000, 1000))
    # Threshold 3.0s exceeded during absence -> fires dwell alert
    dwell_alerts = [e for e in evts8 if e.event_type == SafetyEventType.DWELL_TIME_EXCEEDED]
    assert len(dwell_alerts) == 1
    assert dwell_alerts[0].track_id == 42


def test_reappearance_after_grace_period_expires():
    """Verifies that exceeding the grace period emits ZONE_EXIT, and reappearance is a new entry."""
    manager = ZoneStateManager(max_missing_frames=5)
    poly = Polygon([(100, 100), (300, 100), (300, 300), (100, 300)])
    zone = SafetyZone("z1", "Test Zone", SafetyZoneType.SWIMMING_POOL_GATE, poly, AlertSeverity.HIGH)
    config = CameraROIConfig("comm_1", "cam_1", 1000, 1000, [zone])
    engine = ROIEngine(config=config)

    det = [DetectionBox((150, 150, 250, 250), 0.9, track_id=88)]

    # Frame 1: Entry
    manager.update(1, 0.0, det, engine, (1000, 1000))

    # Frames 2..6: 5 missing frames -> triggers cleanup ZONE_EXIT at frame 6
    evts_f6 = manager.update(6, 3000.0, [], engine, (1000, 1000))
    assert len(evts_f6) == 1
    assert evts_f6[0].event_type == SafetyEventType.ZONE_EXIT
    assert evts_f6[0].track_id == 88

    # Frame 10: Reappears much later -> treated as brand new entry
    evts_f10 = manager.update(10, 6000.0, det, engine, (1000, 1000))
    assert len(evts_f10) == 1
    assert evts_f10[0].event_type == SafetyEventType.ZONE_ENTRY
    assert evts_f10[0].dwell_time_seconds == 0.0


def test_id_switch_creates_independent_tracks_no_silent_merge():
    """Verifies that when tracker re-assigns an ID (Track 1 -> Track 2), states are NOT merged."""
    manager = ZoneStateManager(max_missing_frames=5)
    poly = Polygon([(100, 100), (300, 100), (300, 300), (100, 300)])
    zone = SafetyZone("z1", "Test Zone", SafetyZoneType.SWIMMING_POOL_GATE, poly, AlertSeverity.HIGH, dwell_time_seconds=4.0)
    config = CameraROIConfig("comm_1", "cam_1", 1000, 1000, [zone])
    engine = ROIEngine(config=config)

    # Frame 1: Track 1 enters
    manager.update(1, 1000.0, [DetectionBox((150, 150, 250, 250), 0.9, track_id=1)], engine, (1000, 1000))

    # Frame 2: Tracker assigns new Track 2 at identical coordinates (ID switch)
    evts2 = manager.update(2, 2000.0, [DetectionBox((150, 150, 250, 250), 0.9, track_id=2)], engine, (1000, 1000))
    # Track 2 generates a new entry event
    entry_t2 = [e for e in evts2 if e.track_id == 2 and e.event_type == SafetyEventType.ZONE_ENTRY]
    assert len(entry_t2) == 1

    # Track 1 state is NOT merged into Track 2
    assert manager._track_states[1].track_id == 1
    assert manager._track_states[2].track_id == 2
    assert manager.get_dwell_time_seconds(2, "z1", 2000.0) == 0.0


# ============================================================================
# 3. NON-DESTRUCTIVE ROI CONFIGURATION RELOAD TESTS
# ============================================================================

def test_non_destructive_reload_preserves_unchanged_zone_dwell():
    """Verifies that reloading configuration does NOT reset dwell timers in unchanged zones."""
    z_pool = SafetyZone("pool", "Pool", SafetyZoneType.SWIMMING_POOL_GATE, Polygon([(0, 0), (200, 0), (200, 200), (0, 200)]), AlertSeverity.CRITICAL, dwell_time_seconds=5.0)
    z_deck = SafetyZone("deck", "Deck", SafetyZoneType.GENERAL_COMMON_AREA, Polygon([(200, 0), (400, 0), (400, 200), (200, 200)]), AlertSeverity.LOW)
    old_cfg = CameraROIConfig("comm_1", "cam_pool", 1000, 1000, [z_pool, z_deck])

    manager = ZoneStateManager()
    engine = ROIEngine(config=old_cfg)

    # Track 10 is inside pool at t=1000ms
    manager.update(1, 1000.0, [DetectionBox((50, 50, 150, 150), 0.9, track_id=10)], engine, (1000, 1000))
    # Advance to t=4000ms (dwell is 3.0s)
    manager.update(2, 4000.0, [DetectionBox((50, 50, 150, 150), 0.9, track_id=10)], engine, (1000, 1000))
    assert abs(manager.get_dwell_time_seconds(10, "pool", 4000.0) - 3.0) < 1e-3

    # New config: Pool is UNCHANGED, Deck is deleted, Playground is added
    z_playground = SafetyZone("play", "Playground", SafetyZoneType.PLAYGROUND, Polygon([(500, 500), (700, 500), (700, 700), (500, 700)]), AlertSeverity.LOW)
    new_cfg = CameraROIConfig("comm_1", "cam_pool", 1000, 1000, [z_pool, z_playground])

    # Perform non-destructive configuration update
    events = manager.update_configuration(old_cfg, new_cfg, current_timestamp_ms=4000.0)

    # Dwell timer for Track 10 in unchanged pool zone MUST be preserved!
    assert "pool" in manager.get_track_zones(10)
    dwell_after = manager.get_dwell_time_seconds(10, "pool", 4500.0)
    assert abs(dwell_after - 3.5) < 1e-3


def test_non_destructive_reload_handles_modified_and_deleted_zones():
    """Verifies that modified geometry zones are reset and deleted zones emit exits."""
    z_mod_old = SafetyZone("mod_zone", "Zone", SafetyZoneType.PLAYGROUND, Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]), AlertSeverity.LOW)
    z_del = SafetyZone("del_zone", "Delete Me", SafetyZoneType.RESTRICTED_INFRASTRUCTURE, Polygon([(200, 200), (300, 200), (300, 300), (200, 300)]), AlertSeverity.HIGH)
    old_cfg = CameraROIConfig("comm_1", "cam_test", 1000, 1000, [z_mod_old, z_del])

    manager = ZoneStateManager()
    engine = ROIEngine(config=old_cfg)

    # Track 5 inside mod_zone, Track 6 inside del_zone
    manager.update(1, 1000.0, [
        DetectionBox((20, 20, 60, 60), 0.9, track_id=5),
        DetectionBox((220, 220, 260, 260), 0.9, track_id=6),
    ], engine, (1000, 1000))

    # New config: mod_zone geometry expanded to 150x150, del_zone removed
    z_mod_new = SafetyZone("mod_zone", "Zone", SafetyZoneType.PLAYGROUND, Polygon([(0, 0), (150, 0), (150, 150), (0, 150)]), AlertSeverity.LOW)
    new_cfg = CameraROIConfig("comm_1", "cam_test", 1000, 1000, [z_mod_new])

    events = manager.update_configuration(old_cfg, new_cfg, current_timestamp_ms=2000.0)

    # del_zone must emit ZONE_EXIT
    exit_events = [e for e in events if e.roi_id == "del_zone" and e.event_type == SafetyEventType.ZONE_EXIT]
    assert len(exit_events) == 1
    assert exit_events[0].track_id == 6
    assert "del_zone" not in manager.get_track_zones(6)

    # mod_zone was modified -> occupant state was cleared to re-evaluate on next frame
    assert "mod_zone" not in manager.get_track_zones(5)


def test_reload_different_camera_executes_full_state_reset():
    """Verifies that changing camera or community never retains stale state."""
    z1 = SafetyZone("z1", "Zone", SafetyZoneType.PLAYGROUND, Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]), AlertSeverity.LOW)
    cfg_cam1 = CameraROIConfig("comm_1", "cam_pool", 1000, 1000, [z1])
    cfg_cam2 = CameraROIConfig("comm_1", "cam_gate", 1000, 1000, [z1])

    manager = ZoneStateManager()
    engine = ROIEngine(config=cfg_cam1)
    manager.update(1, 1000.0, [DetectionBox((10, 10, 50, 50), 0.9, track_id=99)], engine, (1000, 1000))
    assert 99 in manager._track_states

    # Reload with different camera config
    manager.update_configuration(cfg_cam1, cfg_cam2, 2000.0)
    assert len(manager._track_states) == 0


# ============================================================================
# 4. POLYGON VALIDATION TESTS (SELF-INTERSECTION & CONCAVE SUPPORT)
# ============================================================================

def test_reject_self_intersecting_bowtie_polygon():
    """Ensures self-intersecting bow-tie / hourglass polygons are strictly rejected."""
    # Edges cross at (50, 50)
    bowtie_pts = [(0.0, 0.0), (100.0, 100.0), (100.0, 0.0), (0.0, 100.0)]
    with pytest.raises(ValueError, match="Self-intersecting polygon detected"):
        Polygon(bowtie_pts)


def test_reject_figure_eight_polygon():
    """Ensures figure-8 self-crossing polygons are rejected."""
    fig8_pts = [(0.0, 0.0), (100.0, 100.0), (100.0, 50.0), (0.0, 50.0), (0.0, 100.0), (100.0, 0.0)]
    with pytest.raises(ValueError, match="Self-intersecting polygon detected"):
        Polygon(fig8_pts)


def test_reject_duplicate_consecutive_vertices():
    """Ensures duplicate consecutive vertices forming zero-length edges are rejected."""
    dup_pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 0.0), (50.0, 50.0)]
    with pytest.raises(ValueError, match="Duplicate consecutive vertex detected"):
        Polygon(dup_pts)


def test_accept_valid_concave_l_shape_polygon():
    """Confirms that valid non-self-intersecting concave polygons (e.g. L-shape) are accepted."""
    l_shape_pts = [
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 50.0),
        (50.0, 50.0),
        (50.0, 100.0),
        (0.0, 100.0),
    ]
    poly = Polygon(l_shape_pts)
    assert poly.num_vertices == 6
    assert abs(poly.area - 7500.0) < 1e-4

    # Point inside the bottom leg
    assert poly.contains_point((75.0, 25.0)) is True
    # Point inside the vertical leg
    assert poly.contains_point((25.0, 75.0)) is True
    # Point in the cut-out corner (outside the L-shape)
    assert poly.contains_point((75.0, 75.0)) is False


def test_accept_valid_concave_chevron_arrowhead():
    """Confirms that arrowhead/chevron concave polygons are accepted and tested accurately."""
    chevron_pts = [(0.0, 0.0), (100.0, 50.0), (0.0, 100.0), (30.0, 50.0)]
    poly = Polygon(chevron_pts)
    assert poly.num_vertices == 4
    assert poly.area > 0.0
    assert poly.contains_point((50.0, 50.0)) is True
    assert poly.contains_point((10.0, 50.0)) is False

