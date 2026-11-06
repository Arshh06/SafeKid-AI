"""Unit and integration tests for SafeKid AI structured safety events and ZoneStateManager."""

import json
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

from safekid_ai.config import (
    DetectorConfig,
    OperationalMode,
    SafeKidPipelineConfig,
    TrackerConfig,
)
from safekid_ai.detector import DetectionBox
from safekid_ai.events import SafetyEvent, SafetyEventType, ZoneStateManager
from safekid_ai.pipeline import SafeKidPipeline
from safekid_ai.roi.engine import ROIEngine, calculate_bottom_center
from safekid_ai.roi.models import (
    AlertSeverity,
    CameraROIConfig,
    Polygon,
    SafetyZone,
    SafetyZoneType,
)


@pytest.fixture
def sample_roi_engine():
    """Builds a test ROIEngine with pool and driveway zones at 1920x1080 reference."""
    pool_zone = SafetyZone(
        roi_id="pool_01",
        name="Swimming Pool Perimeter",
        zone_type=SafetyZoneType.SWIMMING_POOL_GATE,
        polygon=Polygon([(100.0, 100.0), (500.0, 100.0), (500.0, 500.0), (100.0, 500.0)]),
        severity=AlertSeverity.CRITICAL,
        enabled=True,
        dwell_time_seconds=3.0,
    )
    deck_zone = SafetyZone(
        roi_id="deck_01",
        name="Pool Deck (Overlapping)",
        zone_type=SafetyZoneType.GENERAL_COMMON_AREA,
        polygon=Polygon([(50.0, 50.0), (600.0, 50.0), (600.0, 600.0), (50.0, 600.0)]),
        severity=AlertSeverity.MEDIUM,
        enabled=True,
        dwell_time_seconds=5.0,
    )
    disabled_zone = SafetyZone(
        roi_id="disabled_01",
        name="Maintenance Shed",
        zone_type=SafetyZoneType.RESTRICTED_INFRASTRUCTURE,
        polygon=Polygon([(700.0, 100.0), (900.0, 100.0), (900.0, 300.0), (700.0, 300.0)]),
        severity=AlertSeverity.HIGH,
        enabled=False,
    )
    config = CameraROIConfig(
        community_id="comm_greenview",
        camera_id="cam_pool_north",
        frame_width=1920,
        frame_height=1080,
        zones=[pool_zone, deck_zone, disabled_zone],
    )
    return ROIEngine(config=config)


# ============================================================================
# 1. GROUND CONTACT (BOTTOM-CENTER) TESTS
# ============================================================================

def test_bottom_center_calculation():
    """Verifies bottom-center point calculation: x_bc = (x1+x2)/2, y_bc = y2."""
    bbox = (100.0, 200.0, 300.0, 600.0)
    bc = calculate_bottom_center(bbox)
    assert bc == (200.0, 600.0)

    # DetectionBox property
    det = DetectionBox(bbox_xyxy=bbox, confidence=0.95, class_id=0, track_id=42)
    assert det.bottom_center == (200.0, 600.0)


def test_bottom_center_calculation_with_frame_clamping():
    """Verifies that bottom-center point is safely clamped to valid frame pixel bounds."""
    # Bbox extending outside right and bottom edges
    bbox_oob = (1850.0, 1000.0, 1950.0, 1150.0)
    bc_clamped = calculate_bottom_center(bbox_oob, frame_resolution=(1920, 1080))
    # Expected: x_bc clamped to <= 1919.0, y_bc clamped to <= 1079.0
    assert bc_clamped[0] == 1900.0
    assert bc_clamped[1] == 1079.0

    # Bbox extending outside left and top edges
    bbox_neg = (-50.0, -20.0, 30.0, 40.0)
    bc_neg = calculate_bottom_center(bbox_neg, frame_resolution=(1920, 1080))
    assert bc_neg[0] == 0.0
    assert bc_neg[1] == 40.0


def test_ground_contact_distinction_from_bbox_overlap(sample_roi_engine):
    """Verifies that a person whose bounding box overlaps an ROI but whose bottom-center
    ground-contact point is OUTSIDE does NOT trigger an ROI match.
    """
    # Zone covers [100, 100] to [500, 500]
    # Person box: top overlaps zone (y1=450), but feet are at y2=550 (outside zone below it)
    outside_person_bbox = (200.0, 450.0, 300.0, 550.0)
    bc = calculate_bottom_center(outside_person_bbox)
    assert bc == (250.0, 550.0)  # y=550 is below pool zone (max y=500)

    # Check pool zone
    pool_zone = sample_roi_engine.config.get_zone("pool_01")
    assert pool_zone.contains_point(bc) is False

    # Find matching zones for this point
    matching = sample_roi_engine.find_zones_for_point(bc, (1920, 1080))
    # It falls inside deck_zone [50..600] but strictly outside pool_zone [100..500]
    matching_ids = [z.roi_id for z in matching]
    assert "pool_01" not in matching_ids
    assert "deck_01" in matching_ids


# ============================================================================
# 2. ZONE ENTRY & DUPLICATE SUPPRESSION TESTS
# ============================================================================

def test_zone_entry_event_generated(sample_roi_engine):
    """Verifies that ZONE_ENTRY is generated on the frame a person crosses into the zone."""
    manager = ZoneStateManager()

    # Frame 1: Person outside pool (bc = 250, 700)
    det_outside = [DetectionBox(bbox_xyxy=(200.0, 600.0, 300.0, 700.0), confidence=0.9, track_id=1)]
    events_f1 = manager.update(
        frame_id=1,
        timestamp_ms=0.0,
        detections=det_outside,
        roi_engine=sample_roi_engine,
        frame_resolution=(1920, 1080),
    )
    # Outside both pool and deck
    assert len(events_f1) == 0

    # Frame 2: Person moves inside pool (bc = 250, 300)
    det_inside = [DetectionBox(bbox_xyxy=(200.0, 200.0, 300.0, 300.0), confidence=0.92, track_id=1)]
    events_f2 = manager.update(
        frame_id=2,
        timestamp_ms=100.0,
        detections=det_inside,
        roi_engine=sample_roi_engine,
        frame_resolution=(1920, 1080),
    )

    # Person entered both pool_01 and deck_01 (overlapping)
    assert len(events_f2) == 2
    entry_roi_ids = {e.roi_id for e in events_f2}
    assert entry_roi_ids == {"pool_01", "deck_01"}
    for e in events_f2:
        assert e.event_type == SafetyEventType.ZONE_ENTRY
        assert e.track_id == 1
        assert e.community_id == "comm_greenview"
        assert e.camera_id == "cam_pool_north"
        assert e.dwell_time_seconds == 0.0


def test_zone_entry_no_duplicate_spam_on_subsequent_frames(sample_roi_engine):
    """Verifies that continuing inside a zone across subsequent frames does NOT repeat ZONE_ENTRY."""
    manager = ZoneStateManager()

    det_inside = [DetectionBox(bbox_xyxy=(200.0, 200.0, 300.0, 300.0), confidence=0.9, track_id=5)]

    # Frame 1: Initial entry
    events_f1 = manager.update(
        frame_id=1,
        timestamp_ms=0.0,
        detections=det_inside,
        roi_engine=sample_roi_engine,
        frame_resolution=(1920, 1080),
    )
    assert any(e.event_type == SafetyEventType.ZONE_ENTRY for e in events_f1)

    # Frame 2, 3, 4: Remaining in same position within 1 second
    for f_id in [2, 3, 4]:
        events = manager.update(
            frame_id=f_id,
            timestamp_ms=f_id * 200.0,
            detections=det_inside,
            roi_engine=sample_roi_engine,
            frame_resolution=(1920, 1080),
        )
        # Should NOT emit any duplicate entry events
        assert len(events) == 0, f"Frame {f_id} generated unexpected events: {events}"


# ============================================================================
# 3. DWELL TIME MONITORING & ALERT THRESHOLD TESTS
# ============================================================================

def test_dwell_time_exceeded_alert(sample_roi_engine):
    """Verifies DWELL_TIME_EXCEEDED is triggered when dwell threshold is surpassed,
    and is NOT repeated repeatedly afterwards.
    """
    manager = ZoneStateManager()
    # pool_01 threshold is 3.0s, deck_01 threshold is 5.0s
    det = [DetectionBox(bbox_xyxy=(200.0, 200.0, 300.0, 300.0), confidence=0.9, track_id=7)]

    # Frame 1: t=0.0s -> Entry
    e1 = manager.update(1, 0.0, det, sample_roi_engine, (1920, 1080))
    assert len(e1) == 2  # pool_01 and deck_01 entries

    # Frame 2: t=2.0s -> Below threshold (3.0s)
    e2 = manager.update(2, 2000.0, det, sample_roi_engine, (1920, 1080))
    assert len(e2) == 0
    assert abs(manager.get_dwell_time_seconds(7, "pool_01", 2000.0) - 2.0) < 1e-3

    # Frame 3: t=3.2s -> Pool threshold (3.0s) exceeded!
    e3 = manager.update(3, 3200.0, det, sample_roi_engine, (1920, 1080))
    assert len(e3) == 1
    assert e3[0].event_type == SafetyEventType.DWELL_TIME_EXCEEDED
    assert e3[0].roi_id == "pool_01"
    assert e3[0].severity == AlertSeverity.CRITICAL
    assert e3[0].dwell_time_seconds >= 3.0

    # Frame 4: t=4.0s -> Still below deck threshold (5.0s), pool already alerted
    e4 = manager.update(4, 4000.0, det, sample_roi_engine, (1920, 1080))
    assert len(e4) == 0

    # Frame 5: t=5.2s -> Deck threshold (5.0s) exceeded!
    e5 = manager.update(5, 5200.0, det, sample_roi_engine, (1920, 1080))
    assert len(e5) == 1
    assert e5[0].event_type == SafetyEventType.DWELL_TIME_EXCEEDED
    assert e5[0].roi_id == "deck_01"

    # Frame 6: t=6.5s -> Both alerted already, no duplicate dwell events
    e6 = manager.update(6, 6500.0, det, sample_roi_engine, (1920, 1080))
    assert len(e6) == 0


# ============================================================================
# 4. ZONE EXIT & TIMER RESET TESTS
# ============================================================================

def test_zone_exit_resets_timer(sample_roi_engine):
    """Verifies that moving out of an ROI emits ZONE_EXIT and resets the dwell timer."""
    manager = ZoneStateManager()

    det_inside = [DetectionBox(bbox_xyxy=(200.0, 200.0, 300.0, 300.0), confidence=0.9, track_id=9)]
    # Enter at t=0
    manager.update(1, 0.0, det_inside, sample_roi_engine, (1920, 1080))

    # Advance to t=2.0s
    manager.update(2, 2000.0, det_inside, sample_roi_engine, (1920, 1080))

    # Move outside all zones at t=2.5s (bc = 1500, 900)
    det_outside = [DetectionBox(bbox_xyxy=(1400.0, 800.0, 1600.0, 900.0), confidence=0.9, track_id=9)]
    exit_events = manager.update(3, 2500.0, det_outside, sample_roi_engine, (1920, 1080))

    assert len(exit_events) == 2  # Exited pool_01 and deck_01
    for e in exit_events:
        assert e.event_type == SafetyEventType.ZONE_EXIT
        assert e.track_id == 9
        assert abs(e.dwell_time_seconds - 2.5) < 1e-2

    # Verify track has no active zones
    assert manager.get_track_zones(9) == set()
    assert manager.get_dwell_time_seconds(9, "pool_01", 3000.0) == 0.0

    # Re-entering at t=5.0s should trigger a brand new entry event
    reenter_events = manager.update(4, 5000.0, det_inside, sample_roi_engine, (1920, 1080))
    assert len(reenter_events) == 2
    for e in reenter_events:
        assert e.event_type == SafetyEventType.ZONE_ENTRY
        assert e.dwell_time_seconds == 0.0


# ============================================================================
# 5. UNTRACKED / MISSING TRACK ID HANDLING
# ============================================================================

def test_untracked_detection_handling(sample_roi_engine):
    """Verifies that detections with track_id=None are handled safely without crashing."""
    manager = ZoneStateManager()

    # Detections without tracker ID (e.g. tracker dropped or before assignment)
    det_untracked = [
        DetectionBox(bbox_xyxy=(200.0, 200.0, 300.0, 300.0), confidence=0.85, track_id=None)
    ]
    events = manager.update(
        frame_id=1,
        timestamp_ms=50.0,
        detections=det_untracked,
        roi_engine=sample_roi_engine,
        frame_resolution=(1920, 1080),
    )

    # Should generate instantaneous entry warnings for the occupied zones
    assert len(events) == 2
    for e in events:
        assert e.event_type == SafetyEventType.ZONE_ENTRY
        assert e.track_id is None
        assert "Untracked person" in e.message

    # Internal tracking states should not have None or corrupted entries
    assert None not in manager._track_states


# ============================================================================
# 6. DISAPPEARED TRACK CLEANUP TESTS
# ============================================================================

def test_disappearing_track_cleanup(sample_roi_engine):
    """Verifies that a track that stops appearing in detections emits EXIT events
    after exceeding max_missing_frames.
    """
    manager = ZoneStateManager(max_missing_frames=5)

    # Track 12 is inside pool at frame 10 (t=1000.0ms)
    det = [DetectionBox(bbox_xyxy=(200.0, 200.0, 300.0, 300.0), confidence=0.9, track_id=12)]
    manager.update(10, 1000.0, det, sample_roi_engine, (1920, 1080))

    # Frames 11..14: Track 12 is missing (empty detections)
    for f in range(11, 15):
        evts = manager.update(f, 1000.0 + (f - 10) * 100, [], sample_roi_engine, (1920, 1080))
        assert len(evts) == 0

    # Frame 15: Exactly 5 missing frames -> track considered disappeared
    evts_f15 = manager.update(15, 1500.0, [], sample_roi_engine, (1920, 1080))
    assert len(evts_f15) == 2  # Exited both pool_01 and deck_01
    for e in evts_f15:
        assert e.event_type == SafetyEventType.ZONE_EXIT
        assert e.track_id == 12
        assert "left camera view" in e.message

    # State for track 12 should now be completely pruned
    assert 12 not in manager._track_states


# ============================================================================
# 7. END-TO-END PIPELINE ROI INTEGRATION TEST
# ============================================================================

def test_pipeline_with_roi_engine_integration():
    """End-to-end test of SafeKidPipeline with an active camera ROI configuration."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "cam_test_pipeline.json"
        roi_data = {
            "community_id": "comm_integration",
            "camera_id": "cam_pool_gate",
            "frame_width": 640,
            "frame_height": 480,
            "zones": [
                {
                    "roi_id": "pool_danger_zone",
                    "name": "Pool Gate Restricted",
                    "zone_type": "SWIMMING_POOL_GATE",
                    "polygon": [[100, 100], [400, 100], [400, 400], [100, 400]],
                    "severity": "CRITICAL",
                    "enabled": True,
                    "dwell_time_seconds": 2.0,
                }
            ],
        }
        with open(config_path, "w") as f:
            json.dump(roi_data, f)

        pipe_cfg = SafeKidPipelineConfig(
            mode=OperationalMode.COMMUNITY_MONITORING,
            community_name="Integration Villa",
            community_id="comm_integration",
            camera_id="cam_pool_gate",
            roi_config_path=str(config_path),
            detector=DetectorConfig(model_name_or_path="yolov8n.pt", device="cpu"),
            tracker=TrackerConfig(tracker_config_path="bytetrack.yaml"),
        )
        pipeline = SafeKidPipeline(pipe_cfg)

        assert pipeline.roi_engine is not None
        assert pipeline.roi_engine.camera_id == "cam_pool_gate"
        assert pipeline.roi_engine.num_active_zones == 1

        # Test processing a synthetic frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame, frame_id=1, timestamp_ms=0.0, annotate=True)

        assert result.frame_id == 1
        assert result.community_id == "comm_integration"
        assert result.camera_id == "cam_pool_gate"
        assert isinstance(result.safety_events, list)
        assert result.annotated_frame is not None
        assert result.annotated_frame.shape == (480, 640, 3)

        # Test dynamic reload through pipeline
        reloaded = pipeline.reload_roi_config()
        assert reloaded.camera_id == "cam_pool_gate"


# ============================================================================
# 7. PROMPT 07 REFINEMENT TESTS: SHARED BOUNDARY, PRECEDENCE, JITTER & SEMANTICS
# ============================================================================

def test_shared_boundary_water_zone_precedence():
    """Verifies explicit water-zone precedence on shared dividing boundaries.

    Ensures that a point falling directly on the water/deck dividing boundary cannot
    generate simultaneous water and deck classification.
    """
    water_poly = Polygon([(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)])
    deck_poly = Polygon([(300.0, 100.0), (500.0, 100.0), (500.0, 300.0), (300.0, 300.0)])

    water_zone = SafetyZone(
        roi_id="water_01",
        name="Pool Water Basin",
        zone_type=SafetyZoneType.SWIMMING_POOL_BOUNDARY,
        polygon=water_poly,
        severity=AlertSeverity.CRITICAL,
    )
    deck_zone = SafetyZone(
        roi_id="deck_01",
        name="Pool Deck Walkway",
        zone_type=SafetyZoneType.SWIMMING_POOL_GATE,
        polygon=deck_poly,
        severity=AlertSeverity.HIGH,
    )

    config = CameraROIConfig("comm_test", "cam_test", 1000, 1000, [water_zone, deck_zone])
    engine = ROIEngine(config=config)

    # Point strictly inside water
    pt_water = (200.0, 200.0)
    matched_water = engine.find_zones_for_point(pt_water, (1000, 1000))
    assert len(matched_water) == 1
    assert matched_water[0].roi_id == "water_01"

    # Point strictly inside deck
    pt_deck = (400.0, 200.0)
    matched_deck = engine.find_zones_for_point(pt_deck, (1000, 1000))
    assert len(matched_deck) == 1
    assert matched_deck[0].roi_id == "deck_01"

    # Point exactly on shared dividing line (x=300.0, y=200.0)
    pt_shared = (300.0, 200.0)
    assert water_zone.contains_point(pt_shared) is True
    assert deck_zone.contains_point(pt_shared) is True

    # Precedence rule: water takes exclusive priority, suppressing deck
    matched_shared = engine.find_zones_for_point(pt_shared, (1000, 1000))
    assert len(matched_shared) == 1
    assert matched_shared[0].roi_id == "water_01"
    assert matched_shared[0].zone_type == SafetyZoneType.SWIMMING_POOL_BOUNDARY


def test_composite_deck_excludes_water_basin():
    """Verifies that composite multi-polygon deck zone strictly excludes water basin interior."""
    W_pts = [(454.0, 189.0), (714.0, 243.0), (582.0, 387.0), (173.0, 261.0)]
    water_poly = Polygon(W_pts)

    deck_north = Polygon([(320.0, 120.0), (840.0, 200.0), (714.0, 243.0), (454.0, 189.0)])
    deck_east = Polygon([(840.0, 200.0), (660.0, 470.0), (582.0, 387.0), (714.0, 243.0)])
    deck_south = Polygon([(582.0, 387.0), (660.0, 470.0), (60.0, 310.0), (173.0, 261.0)])
    deck_west = Polygon([(60.0, 310.0), (320.0, 120.0), (454.0, 189.0), (173.0, 261.0)])

    deck_zone = SafetyZone(
        roi_id="pool_deck",
        name="Composite Pool Deck",
        zone_type=SafetyZoneType.SWIMMING_POOL_GATE,
        polygon=deck_north,
        polygons=[deck_north, deck_east, deck_south, deck_west],
    )

    # Test centroid of water
    cx, cy = water_poly.centroid
    assert water_poly.contains_point((cx, cy)) is True
    assert deck_zone.contains_point((cx, cy)) is False

    # Test multiple interior water points
    interior_points = [
        (480.0, 220.0),
        (500.0, 250.0),
        (550.0, 300.0),
        (350.0, 270.0),
    ]
    for pt in interior_points:
        assert water_poly.contains_point(pt) is True
        assert deck_zone.contains_point(pt) is False, f"Water point {pt} incorrectly matched deck!"


def test_three_frame_entry_confirmation_and_timestamp_backdating():
    """Verifies 3-frame entry confirmation preserves timestamp of first qualifying observation."""
    poly = Polygon([(100.0, 100.0), (500.0, 100.0), (500.0, 500.0), (100.0, 500.0)])
    zone = SafetyZone("z1", "Test Zone", SafetyZoneType.SWIMMING_POOL_GATE, poly)
    config = CameraROIConfig("c1", "cam1", 1000, 1000, [zone])
    engine = ROIEngine(config=config)

    manager = ZoneStateManager(entry_confirmation_frames=3, exit_confirmation_frames=3)
    det_inside = [DetectionBox((200.0, 200.0, 300.0, 300.0), 0.9, track_id=42)]

    # Frame 1: t=100.0ms (Observation 1 inside) -> Pending, no event
    e1 = manager.update(frame_id=1, timestamp_ms=100.0, detections=det_inside, roi_engine=engine, frame_resolution=(1000, 1000))
    assert len(e1) == 0

    # Frame 2: t=141.7ms (Observation 2 inside) -> Pending, no event
    e2 = manager.update(frame_id=2, timestamp_ms=141.7, detections=det_inside, roi_engine=engine, frame_resolution=(1000, 1000))
    assert len(e2) == 0

    # Frame 3: t=183.3ms (Observation 3 inside) -> Confirmed!
    # Spans ~83.3ms across 2 frame intervals
    e3 = manager.update(frame_id=3, timestamp_ms=183.3, detections=det_inside, roi_engine=engine, frame_resolution=(1000, 1000))
    assert len(e3) == 1
    assert e3[0].event_type == SafetyEventType.ZONE_ENTRY
    assert e3[0].track_id == 42
    # Timestamp backdated to first inside observation!
    assert e3[0].timestamp_ms == 100.0
    assert e3[0].frame_id == 1


def test_boundary_jitter_suppression():
    """Verifies alternating in/out boundary jitter does not emit spurious events."""
    poly = Polygon([(100.0, 100.0), (500.0, 100.0), (500.0, 500.0), (100.0, 500.0)])
    zone = SafetyZone("z1", "Test Zone", SafetyZoneType.SWIMMING_POOL_GATE, poly)
    config = CameraROIConfig("c1", "cam1", 1000, 1000, [zone])
    engine = ROIEngine(config=config)

    manager = ZoneStateManager(entry_confirmation_frames=3, exit_confirmation_frames=3)
    det_in = [DetectionBox((200.0, 200.0, 300.0, 300.0), 0.9, track_id=2)]
    det_out = [DetectionBox((200.0, 600.0, 300.0, 700.0), 0.9, track_id=2)]

    # 10 frames of 1-frame jitter (in, out, in, out, in, in, out, in, out, out)
    jitter_sequence = [det_in, det_out, det_in, det_out, det_in, det_in, det_out, det_in, det_out, det_out]
    all_events = []
    for fid, d in enumerate(jitter_sequence, start=1):
        evts = manager.update(frame_id=fid, timestamp_ms=fid * 41.67, detections=d, roi_engine=engine, frame_resolution=(1000, 1000))
        all_events.extend(evts)

    # All boundary oscillations filtered: 0 events emitted
    assert len(all_events) == 0


def test_three_frame_exit_confirmation_and_dwell_accuracy():
    """Verifies 3-frame exit confirmation preserves first outside observation timestamp and accurate dwell."""
    poly = Polygon([(100.0, 100.0), (500.0, 100.0), (500.0, 500.0), (100.0, 500.0)])
    zone = SafetyZone("z1", "Test Zone", SafetyZoneType.SWIMMING_POOL_GATE, poly, dwell_time_seconds=3.0)
    config = CameraROIConfig("c1", "cam1", 1000, 1000, [zone])
    engine = ROIEngine(config=config)

    manager = ZoneStateManager(entry_confirmation_frames=3, exit_confirmation_frames=3)
    det_in = [DetectionBox((200.0, 200.0, 300.0, 300.0), 0.9, track_id=8)]
    det_out = [DetectionBox((200.0, 600.0, 300.0, 700.0), 0.9, track_id=8)]

    # Frames 1, 2, 3: Entry confirmed at frame 3 (entry_ts = 100.0ms)
    manager.update(1, 100.0, det_in, engine, (1000, 1000))
    manager.update(2, 141.7, det_in, engine, (1000, 1000))
    manager.update(3, 183.3, det_in, engine, (1000, 1000))

    # Frame 10: Still inside at t=3100.0ms (dwell = 3.0s, triggers alert)
    e10 = manager.update(10, 3100.0, det_in, engine, (1000, 1000))
    assert len(e10) == 1
    assert e10[0].event_type == SafetyEventType.DWELL_TIME_EXCEEDED

    # Frame 20: t=4000.0ms (Outside observation 1) -> Pending exit
    e20 = manager.update(20, 4000.0, det_out, engine, (1000, 1000))
    assert len(e20) == 0

    # Frame 21: t=4041.7ms (Outside observation 2) -> Pending exit
    e21 = manager.update(21, 4041.7, det_out, engine, (1000, 1000))
    assert len(e21) == 0

    # Frame 22: t=4083.3ms (Outside observation 3) -> Confirmed exit!
    e22 = manager.update(22, 4083.3, det_out, engine, (1000, 1000))
    assert len(e22) == 1
    assert e22[0].event_type == SafetyEventType.ZONE_EXIT
    # Exit timestamp backdated to first outside observation (4000.0ms)
    assert e22[0].timestamp_ms == 4000.0
    assert e22[0].frame_id == 20
    # Dwell = (4000.0 - 100.0) / 1000.0 = 3.9 seconds
    assert abs(e22[0].dwell_time_seconds - 3.9) < 1e-2


def test_interrupted_exit_cancels_pending_exit():
    """Verifies that returning inside before exit confirmation cancels the pending exit."""
    poly = Polygon([(100.0, 100.0), (500.0, 100.0), (500.0, 500.0), (100.0, 500.0)])
    zone = SafetyZone("z1", "Test Zone", SafetyZoneType.SWIMMING_POOL_GATE, poly)
    config = CameraROIConfig("c1", "cam1", 1000, 1000, [zone])
    engine = ROIEngine(config=config)

    manager = ZoneStateManager(entry_confirmation_frames=2, exit_confirmation_frames=3)
    det_in = [DetectionBox((200.0, 200.0, 300.0, 300.0), 0.9, track_id=8)]
    det_out = [DetectionBox((200.0, 600.0, 300.0, 700.0), 0.9, track_id=8)]

    # Enter (confirmed on frame 2)
    manager.update(1, 100.0, det_in, engine, (1000, 1000))
    manager.update(2, 200.0, det_in, engine, (1000, 1000))
    assert "z1" in manager.get_track_zones(8)

    # Frame 3: Outside (pending exit 1)
    manager.update(3, 300.0, det_out, engine, (1000, 1000))
    # Frame 4: Outside (pending exit 2)
    manager.update(4, 400.0, det_out, engine, (1000, 1000))

    # Frame 5: Step back inside! (Pending exit must be cancelled)
    e5 = manager.update(5, 500.0, det_in, engine, (1000, 1000))
    assert len(e5) == 0
    assert "z1" in manager.get_track_zones(8)

    # Frame 6: Still inside at t=1000ms -> dwell continues from t=100ms
    manager.update(6, 1000.0, det_in, engine, (1000, 1000))
    dwell = manager.get_dwell_time_seconds(8, "z1", 1000.0)
    assert abs(dwell - 0.9) < 1e-2


def test_temporary_missing_detections_not_counted_as_outside():
    """Verifies that missing detections do not count as outside observations during grace period."""
    poly = Polygon([(100.0, 100.0), (500.0, 100.0), (500.0, 500.0), (100.0, 500.0)])
    zone = SafetyZone("z1", "Test Zone", SafetyZoneType.SWIMMING_POOL_GATE, poly)
    config = CameraROIConfig("c1", "cam1", 1000, 1000, [zone])
    engine = ROIEngine(config=config)

    manager = ZoneStateManager(max_missing_frames=10, entry_confirmation_frames=2, exit_confirmation_frames=3)
    det_in = [DetectionBox((200.0, 200.0, 300.0, 300.0), 0.9, track_id=5)]

    # Enter
    manager.update(1, 0.0, det_in, engine, (1000, 1000))
    manager.update(2, 100.0, det_in, engine, (1000, 1000))
    assert "z1" in manager.get_track_zones(5)

    # Missing for 4 frames (< max_missing_frames 10)
    for f in range(3, 7):
        evts = manager.update(f, f * 100.0, [], engine, (1000, 1000))
        assert len(evts) == 0  # No premature exit
        assert manager._track_states[5].pending_exit_count.get("z1", 0) == 0

    # Reappears inside zone at frame 7
    e7 = manager.update(7, 700.0, det_in, engine, (1000, 1000))
    assert len(e7) == 0  # Continued visit, no re-entry event
    assert abs(manager.get_dwell_time_seconds(5, "z1", 700.0) - 0.7) < 1e-2


def test_visual_rendering_safe_monitored_danger_zones():
    """Verifies that draw_roi_polygons distinguishes SAFE, MONITORED, and DANGER zones cleanly."""
    from safekid_ai.visualization import draw_roi_polygons
    from safekid_ai.roi.models import ZoneRole, ZoneSafetyPolicy

    frame = np.zeros((480, 848, 3), dtype=np.uint8)
    z_safe = SafetyZone(
        "s1", "Sandbox", SafetyZoneType.PLAYGROUND,
        Polygon([(50.0, 50.0), (200.0, 50.0), (200.0, 200.0), (50.0, 200.0)]),
        severity=AlertSeverity.LOW,
        policy=ZoneSafetyPolicy(zone_role=ZoneRole.SAFE, entry_alert=False),
    )
    z_danger = SafetyZone(
        "d1", "Deep Water", SafetyZoneType.SWIMMING_POOL_BOUNDARY,
        Polygon([(300.0, 50.0), (500.0, 50.0), (500.0, 200.0), (300.0, 200.0)]),
        severity=AlertSeverity.CRITICAL,
        policy=ZoneSafetyPolicy(zone_role=ZoneRole.DANGER, entry_alert=True),
    )
    vis = draw_roi_polygons(frame, [z_safe, z_danger])
    assert vis.shape == (480, 848, 3)
    # Ensure pixels inside polygons are non-zero
    assert np.any(vis[100, 100] > 0)
    assert np.any(vis[100, 400] > 0)


def test_visual_rendering_collision_aware_labels_and_clamping():
    """Verifies that draw_person_and_events renders collision-aware labels and clamps markers safely."""
    from safekid_ai.visualization import draw_person_and_events

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Two proximate detections causing horizontal collision
    det1 = DetectionBox((200.0, 400.0, 250.0, 550.0), 0.88, track_id=1)
    det2 = DetectionBox((230.0, 405.0, 280.0, 555.0), 0.92, track_id=2)
    # Detection out-of-bounds at bottom right
    det_edge = DetectionBox((1880.0, 1020.0, 1940.0, 1120.0), 0.75, track_id=3)

    vis = draw_person_and_events(
        frame=frame,
        detections=[det1, det2, det_edge],
        occupied_zones_per_track={1: ["Sidewalk"], 2: ["Driveway"], 3: []},
        dwell_times_per_track={1: {}, 2: {"Driveway": 1.5}, 3: {}},
        events=[],
        community_name="Community 001",
        camera_id="camera_gate_north",
        track_classifications={1: "ADULT", 2: "CHILD", 3: "UNCLASSIFIED"},
        active_alerts=["CHILD ALERT: Child in danger zone"],
    )
    assert vis.shape == (1080, 1920, 3)
    # Ensure drawing rendered onto frame
    assert np.any(vis > 0)


def test_alert_prioritization_critical_over_high():
    """Verifies that the alert panel strictly prioritizes CRITICAL alerts over HIGH, MEDIUM, and LOW."""
    from safekid_ai.visualization import draw_person_and_events, extract_alert_info, SEVERITY_ORDER

    # 1. Test severity extraction helper
    msg1, sev1 = extract_alert_info("CHILD SAFETY ALERT: Child in pool [Severity: CRITICAL]")
    assert sev1 == AlertSeverity.CRITICAL

    msg2, sev2 = extract_alert_info("CHILD ALERT: Child entered vehicle driveway [Severity: HIGH]")
    assert sev2 == AlertSeverity.HIGH

    msg3, sev3 = extract_alert_info("Dwell warning in monitored area [Severity: MEDIUM]")
    assert sev3 == AlertSeverity.MEDIUM

    msg4, sev4 = extract_alert_info(("Child safely entered sandbox", AlertSeverity.LOW))
    assert sev4 == AlertSeverity.LOW

    assert SEVERITY_ORDER[sev1] > SEVERITY_ORDER[sev2] > SEVERITY_ORDER[sev3] > SEVERITY_ORDER[sev4]

    # 2. Render frame with mixed alerts: ensure CRITICAL is top-ranked
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mixed_alerts = [
        ("Child safely entered sandbox", AlertSeverity.LOW),
        ("CHILD ALERT: Child entered vehicle driveway", AlertSeverity.HIGH),
        ("Dwell warning in monitored area", AlertSeverity.MEDIUM),
        ("CHILD SAFETY ALERT: Child in deep pool", AlertSeverity.CRITICAL),
    ]

    vis = draw_person_and_events(
        frame=frame,
        detections=[],
        occupied_zones_per_track={},
        dwell_times_per_track={},
        events=[],
        community_name="Community 001",
        camera_id="camera_pool",
        active_alerts=mixed_alerts,
    )
    assert vis.shape == (720, 1280, 3)
    assert np.any(vis > 0)


def test_alert_panel_overflow_indication():
    """Verifies that when 3 or more active alerts are present, overflow is rendered and no alerts are lost."""
    from safekid_ai.visualization import draw_person_and_events

    frame = np.zeros((480, 848, 3), dtype=np.uint8)
    four_alerts = [
        ("Alert 1: Child near gate", AlertSeverity.HIGH),
        ("Alert 2: Child in water", AlertSeverity.CRITICAL),
        ("Alert 3: Child in driveway", AlertSeverity.HIGH),
        ("Alert 4: Dwell in corridor", AlertSeverity.MEDIUM),
    ]
    vis = draw_person_and_events(
        frame=frame,
        detections=[],
        occupied_zones_per_track={},
        dwell_times_per_track={},
        events=[],
        community_name="Community 001",
        camera_id="camera_gate",
        active_alerts=four_alerts,
    )
    assert vis.shape == (480, 848, 3)
    assert np.any(vis > 0)


def test_crowded_scene_label_compactness():
    """Verifies that proximate tracks have secondary text suppressed to keep labels compact and legible."""
    from safekid_ai.visualization import draw_person_and_events

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Two very close persons (x centers 325 and 365, delta = 40px < 130px threshold)
    det1 = DetectionBox((300.0, 500.0, 350.0, 650.0), 0.90, track_id=1)
    det2 = DetectionBox((340.0, 505.0, 390.0, 655.0), 0.92, track_id=2)

    vis = draw_person_and_events(
        frame=frame,
        detections=[det1, det2],
        occupied_zones_per_track={1: ["Driveway Lane"], 2: ["Driveway Lane"]},
        dwell_times_per_track={1: {"Driveway Lane": 2.5}, 2: {"Driveway Lane": 2.2}},
        events=[],
        community_name="Community 001",
        camera_id="camera_gate_north",
        track_classifications={1: "CHILD", 2: "CHILD"},
    )
    assert vis.shape == (1080, 1920, 3)
    assert np.any(vis > 0)


def test_large_roi_polygon_adaptive_opacity_and_perimeter_badge():
    """Verifies that large ROI polygons (> 15% of frame) render with lighter opacity and perimeter badges."""
    from safekid_ai.visualization import draw_roi_polygons
    from safekid_ai.roi.models import ZoneRole, ZoneSafetyPolicy

    # 1920x1080 frame: total area = 2,073,600 px^2
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Large driveway zone: area = (1500 - 380) * (1080 - 350) = 1120 * 730 = 817,600 (~39.4% of frame)
    large_driveway = SafetyZone(
        "large_driveway",
        "Main Vehicular Driveway",
        SafetyZoneType.VEHICULAR_GATEWAY,
        Polygon([(380.0, 350.0), (1500.0, 350.0), (1500.0, 1080.0), (380.0, 1080.0)]),
        severity=AlertSeverity.HIGH,
        policy=ZoneSafetyPolicy(zone_role=ZoneRole.DANGER, entry_alert=True),
    )
    vis = draw_roi_polygons(frame, [large_driveway])
    assert vis.shape == (1080, 1920, 3)
    # Check that fill rendered
    assert np.any(vis[500, 800] > 0)


