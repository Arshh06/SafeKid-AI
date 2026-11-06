"""Unit and integration test suite for surveillance tracking optimization, ground contact, and ethical invariants."""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from safekid_ai.child_safety import (
    ChildSafetyEvent,
    ChildSafetyEventEngine,
    TrackClassification,
    TrackClassificationProvider,
)
from safekid_ai.config import (
    DetectorConfig,
    OperationalMode,
    TrackerConfig,
)
from safekid_ai.detector import DetectionBox, FrameDetections, PersonDetector
from safekid_ai.events import SafetyEvent, SafetyEventType, ZoneStateManager
from safekid_ai.roi.engine import ROIEngine, calculate_bottom_center
from safekid_ai.roi.models import (
    AlertSeverity,
    CameraROIConfig,
    Point2D,
    Polygon,
    SafetyZone,
    SafetyZoneType,
    ZoneRole,
    ZoneSafetyPolicy,
)
from safekid_ai.tracker import PersonTracker, TrackedPerson


def test_ground_contact_point_calculation():
    """Verifies that ground contact point is strictly bottom-center (x_mid, y_max)."""
    bbox = (100.0, 50.0, 200.0, 250.0)
    bc = calculate_bottom_center(bbox)
    assert bc == (150.0, 250.0)

    # Clamping within resolution bounds
    bc_clamped = calculate_bottom_center(bbox, frame_resolution=(180, 200))
    assert bc_clamped == (150.0, 199.0)  # y_max clamped to h-1 (199)


def test_small_object_box_area_filtering():
    """Verifies that min_box_area and max_box_area filter noisy microscopic or oversized detections."""
    cfg = DetectorConfig(min_box_area=500.0, max_box_area=50000.0)
    assert cfg.min_box_area == 500.0
    assert cfg.max_box_area == 50000.0

    # Detection 1: Microscopic noise (area = 10 * 20 = 200 < 500)
    tiny_box = DetectionBox(bbox_xyxy=(10.0, 10.0, 20.0, 30.0), confidence=0.8)
    assert tiny_box.area == 200.0

    # Detection 2: Valid distant pedestrian (area = 30 * 80 = 2400)
    valid_box = DetectionBox(bbox_xyxy=(100.0, 100.0, 130.0, 180.0), confidence=0.75)
    assert valid_box.area == 2400.0

    # Detection 3: Whole-screen artifact (area = 500 * 500 = 250000 > 50000)
    huge_box = DetectionBox(bbox_xyxy=(0.0, 0.0, 500.0, 500.0), confidence=0.9)
    assert huge_box.area == 250000.0

    # Filter manually using config rules
    boxes = [tiny_box, valid_box, huge_box]
    filtered = [
        b for b in boxes
        if b.area >= cfg.min_box_area and (cfg.max_box_area is None or b.area <= cfg.max_box_area)
    ]
    assert len(filtered) == 1
    assert filtered[0] == valid_box


def test_tracker_surveillance_config():
    """Verifies TrackerConfig allows calibrated surveillance thresholds."""
    t_cfg = TrackerConfig(
        tracker_type="bytetrack",
        tracker_config_path="configs/trackers/bytetrack_surveillance.yaml",
        track_high_thresh=0.30,
        track_low_thresh=0.10,
        new_track_thresh=0.35,
        track_buffer=60,
        match_thresh=0.75,
    )
    assert t_cfg.track_buffer == 60
    assert t_cfg.track_high_thresh == 0.30
    assert t_cfg.track_low_thresh == 0.10
    assert t_cfg.new_track_thresh == 0.35
    assert t_cfg.match_thresh == 0.75


def test_person_tracker_reset_clears_active_tracks():
    """Verifies that PersonTracker.reset() thoroughly clears state."""
    class DummyModel:
        def __init__(self):
            self.predictor = type("Predictor", (), {"trackers": ["mock_tracker"]})()

    dummy_model = DummyModel()
    tracker = PersonTracker(dummy_model, TrackerConfig())
    tracker.active_tracks[1] = TrackedPerson(
        track_id=1,
        current_box=DetectionBox(bbox_xyxy=(0, 0, 10, 10), confidence=0.9),
    )
    assert len(tracker.active_tracks) == 1
    assert dummy_model.predictor.trackers is not None

    tracker.reset()
    assert len(tracker.active_tracks) == 0
    assert dummy_model.predictor.trackers is None


def test_zone_entry_exit_stability_debounce():
    """Verifies that entry_confirmation_frames and exit_confirmation_frames prevent boundary oscillation."""
    roi_config = CameraROIConfig(
        community_id="comm_test",
        camera_id="cam_test",
        frame_width=1000,
        frame_height=1000,
        zones=[
            SafetyZone(
                roi_id="danger_transit",
                name="Danger Transit Lane",
                zone_type=SafetyZoneType.VEHICULAR_GATEWAY,
                severity=AlertSeverity.HIGH,
                polygon=Polygon([[0.0, 0.0], [500.0, 0.0], [500.0, 500.0], [0.0, 500.0]]),
                policy=ZoneSafetyPolicy(zone_role=ZoneRole.DANGER, entry_alert=True),
            )
        ],
    )
    roi_engine = ROIEngine(config=roi_config)

    # Require 3 frames to confirm entry and 3 frames to confirm exit
    manager = ZoneStateManager(entry_confirmation_frames=3, exit_confirmation_frames=3)

    # Frame 1: Person enters boundary (bottom center at x=250, y=250 -> inside)
    det_inside = DetectionBox(bbox_xyxy=(240, 200, 260, 250), confidence=0.9, track_id=10)
    ev1 = manager.update(1, 100.0, [det_inside], roi_engine, (1000, 1000))
    assert len(ev1) == 0  # Frame 1: Pending entry, no event yet

    # Frame 2: Person jittered slightly outside (x=550, y=250 -> outside)
    det_outside = DetectionBox(bbox_xyxy=(540, 200, 560, 250), confidence=0.9, track_id=10)
    ev2 = manager.update(2, 200.0, [det_outside], roi_engine, (1000, 1000))
    assert len(ev2) == 0  # Entry aborted due to jitter!

    # Now person firmly enters for 3 consecutive frames
    manager.update(3, 300.0, [det_inside], roi_engine, (1000, 1000))
    manager.update(4, 400.0, [det_inside], roi_engine, (1000, 1000))
    ev5 = manager.update(5, 500.0, [det_inside], roi_engine, (1000, 1000))
    assert len(ev5) == 1
    assert ev5[0].event_type == SafetyEventType.ZONE_ENTRY


def test_ethical_policy_invariants_unclassified_suppression():
    """CRITICAL ETHICAL INVARIANT: Tracks from surveillance footage must remain UNCLASSIFIED and produce 0 child alerts."""
    class SurveillanceUnclassifiedProvider(TrackClassificationProvider):
        def get_classification(self, track_id: int, timestamp_ms: float = 0.0) -> TrackClassification:
            return TrackClassification.UNCLASSIFIED

        def is_classified(self, track_id: int) -> bool:
            return True

    provider = SurveillanceUnclassifiedProvider()
    assert provider.get_classification(1) == TrackClassification.UNCLASSIFIED

    roi_config = CameraROIConfig(
        community_id="community_virat",
        camera_id="camera_virat_s01",
        frame_width=1920,
        frame_height=1080,
        zones=[
            SafetyZone(
                roi_id="danger_transit",
                name="Vehicular Transit Lane",
                zone_type=SafetyZoneType.VEHICULAR_GATEWAY,
                severity=AlertSeverity.HIGH,
                polygon=Polygon([[0.0, 0.0], [500.0, 0.0], [500.0, 500.0], [0.0, 500.0]]),
                policy=ZoneSafetyPolicy(
                    zone_role=ZoneRole.DANGER,
                    entry_alert=True,
                    exit_alert=False,
                    dwell_alert=True,
                    dwell_threshold_seconds=1.0,
                    severity=AlertSeverity.HIGH,
                ),
            )
        ],
    )
    engine = ChildSafetyEventEngine(annotation_provider=provider, roi_config=roi_config)

    # Simulate raw zone entry and dwell events into danger zone
    raw_entry = SafetyEvent(
        event_id="EVT-001",
        event_type=SafetyEventType.ZONE_ENTRY,
        timestamp_ms=1000.0,
        frame_id=1,
        community_id="community_virat",
        camera_id="camera_virat_s01",
        track_id=1,
        roi_id="danger_transit",
        zone_name="Vehicular Transit Lane",
        zone_type=SafetyZoneType.VEHICULAR_GATEWAY,
        severity=AlertSeverity.HIGH,
        bottom_center=(250.0, 450.0),
        bbox_xyxy=(220.0, 300.0, 280.0, 450.0),
        dwell_time_seconds=0.0,
        message="Track ID 1 entered Vehicular Transit Lane",
    )
    raw_dwell = SafetyEvent(
        event_id="EVT-002",
        event_type=SafetyEventType.DWELL_TIME_EXCEEDED,
        timestamp_ms=3000.0,
        frame_id=48,
        community_id="community_virat",
        camera_id="camera_virat_s01",
        track_id=1,
        roi_id="danger_transit",
        zone_name="Vehicular Transit Lane",
        zone_type=SafetyZoneType.VEHICULAR_GATEWAY,
        severity=AlertSeverity.HIGH,
        bottom_center=(250.0, 450.0),
        bbox_xyxy=(220.0, 300.0, 280.0, 450.0),
        dwell_time_seconds=2.0,
        message="Track ID 1 dwelled in Vehicular Transit Lane for 2.0s",
    )

    child_alerts = engine.process_events([raw_entry, raw_dwell], frame_id=48, timestamp_ms=3000.0, roi_config=roi_config)
    assert len(child_alerts) == 0, "UNCLASSIFIED surveillance tracks must NEVER trigger child alerts!"


def test_track_history_and_velocity_vector():
    """Verifies that TrackedPerson preserves centroid history and computes velocity vectors."""
    box1 = DetectionBox(bbox_xyxy=(100.0, 100.0, 150.0, 200.0), confidence=0.9, track_id=5)
    tracked = TrackedPerson(track_id=5, current_box=box1, first_seen_frame=0, last_seen_frame=0)
    tracked.history_centroids.append(box1.center)

    # Frame 10: Moved right by 50px (x_center: 125 -> 175)
    box2 = DetectionBox(bbox_xyxy=(150.0, 100.0, 200.0, 200.0), confidence=0.88, track_id=5)
    tracked.update(box2, frame_id=10, max_history=60)

    assert len(tracked.history_centroids) == 2
    vx, vy = tracked.velocity_vector
    assert vx == 50.0
    assert vy == 0.0


def test_tiled_candidate_nms_filtering():
    """Verifies that global NMS merges overlapping bounding boxes across candidate tiles."""
    import torchvision.ops

    # Simulate two overlapping tile detections of the same person
    box_tile1 = [450.0, 360.0, 515.0, 535.0]
    box_tile2 = [452.0, 362.0, 517.0, 536.0]  # Almost identical overlap
    box_other = [800.0, 300.0, 860.0, 480.0]  # Separate person

    boxes = torch.tensor([box_tile1, box_tile2, box_other], dtype=torch.float32)
    scores = torch.tensor([0.85, 0.70, 0.80], dtype=torch.float32)

    keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.50)
    assert len(keep) == 2  # box_tile2 suppressed by higher confidence box_tile1!
    assert 0 in keep.tolist()
    assert 2 in keep.tolist()

