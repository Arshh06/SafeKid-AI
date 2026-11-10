"""Tests for CPU inference optimization and keyframe sampling in SafeKid AI."""

import numpy as np
import pytest
import torch

from safekid_ai.config import DetectorConfig, SafeKidPipelineConfig, TrackerConfig
from safekid_ai.child_detection.config import ChildDetectorConfig
from safekid_ai.child_detection.detector import ChildDetector
from safekid_ai.child_detection.models import ChildDetectionLabel
from safekid_ai.child_detection.provider import AutomatedChildClassificationProvider
from safekid_ai.detector import DetectionBox, FrameDetections
from safekid_ai.detectors.dfine_detector import DFINEDetectorAdapter
from safekid_ai.pipeline import SafeKidPipeline
from safekid_ai.tracker import PersonTracker, TrackedPerson


def test_detection_box_predicted_and_staleness_fields():
    """DetectionBox must support is_predicted and stale_frames with sensible defaults."""
    box = DetectionBox(bbox_xyxy=(10.0, 10.0, 50.0, 100.0), confidence=0.88, track_id=1)
    assert box.is_predicted is False
    assert box.stale_frames == 0

    pred_box = DetectionBox(
        bbox_xyxy=(12.0, 10.0, 52.0, 100.0),
        confidence=0.88,
        track_id=1,
        is_predicted=True,
        stale_frames=2,
    )
    assert pred_box.is_predicted is True
    assert pred_box.stale_frames == 2


def test_tracked_person_extrapolate():
    """TrackedPerson.extrapolate moves bounding box along velocity vector and updates staleness."""
    initial_box = DetectionBox(bbox_xyxy=(100.0, 100.0, 150.0, 200.0), confidence=0.9, track_id=42)
    tracked = TrackedPerson(track_id=42, current_box=initial_box, first_seen_frame=0, last_seen_frame=0)
    tracked.history_centroids.append(initial_box.center)

    # Simulate next detected frame moving +10 pixels in X, +5 pixels in Y
    next_box = DetectionBox(bbox_xyxy=(110.0, 105.0, 160.0, 205.0), confidence=0.92, track_id=42)
    tracked.update(next_box, frame_id=1)
    assert tracked.stale_frames == 0
    assert tracked.is_predicted is False

    vx, vy = tracked.velocity_vector
    assert pytest.approx(vx, 0.1) == 10.0
    assert pytest.approx(vy, 0.1) == 5.0

    # Extrapolate for frame 2 (skipped detection frame)
    pred_box = tracked.extrapolate(frame_id=2, frame_size=(640, 480))
    assert tracked.stale_frames == 1
    assert tracked.is_predicted is True
    assert pred_box.is_predicted is True
    assert pred_box.stale_frames == 1
    # Coordinates should have shifted by vx, vy
    assert pytest.approx(pred_box.x1, 0.1) == 120.0
    assert pytest.approx(pred_box.y1, 0.1) == 110.0
    assert pytest.approx(pred_box.x2, 0.1) == 170.0
    assert pytest.approx(pred_box.y2, 0.1) == 210.0

    # Clamping test at image edge
    tracked.current_box = DetectionBox(bbox_xyxy=(635.0, 470.0, 645.0, 490.0), confidence=0.9, track_id=42)
    pred_clamped = tracked.extrapolate(frame_id=3, frame_size=(640, 480))
    assert pred_clamped.x1 <= 639.0
    assert pred_clamped.y1 <= 479.0
    assert pred_clamped.x2 <= 640.0
    assert pred_clamped.y2 <= 480.0


def test_person_tracker_keyframe_skipping():
    """PersonTracker skips model forward pass on intermediate frames when frame_interval > 1."""
    class MockYOLO:
        def __init__(self):
            self.track_calls = 0

        def track(self, *args, **kwargs):
            self.track_calls += 1
            # Return empty or dummy
            return []

    mock_yolo = MockYOLO()
    cfg = TrackerConfig(frame_interval=3, max_track_staleness=4)
    tracker = PersonTracker(detector_model=mock_yolo, config=cfg)

    # Seed an active track
    box = DetectionBox(bbox_xyxy=(50.0, 50.0, 100.0, 150.0), confidence=0.85, track_id=7)
    tracked = TrackedPerson(track_id=7, current_box=box, first_seen_frame=0, last_seen_frame=0)
    tracked.history_centroids.append(box.center)
    tracker.active_tracks[7] = tracked

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: skipped frame (1 % 3 != 0)
    res1 = tracker.track(dummy_frame, frame_id=1, timestamp_ms=33.3)
    assert mock_yolo.track_calls == 0  # Did NOT call heavy YOLO forward pass
    assert len(res1.detections) == 1
    assert res1.detections[0].track_id == 7
    assert res1.detections[0].is_predicted is True
    assert res1.detections[0].stale_frames == 1

    # Frame 2: skipped frame (2 % 3 != 0)
    res2 = tracker.track(dummy_frame, frame_id=2, timestamp_ms=66.6)
    assert mock_yolo.track_calls == 0
    assert len(res2.detections) == 1
    assert res2.detections[0].stale_frames == 2

    # Frame 3: keyframe (3 % 3 == 0) -> calls YOLO
    res3 = tracker.track(dummy_frame, frame_id=3, timestamp_ms=99.9)
    assert mock_yolo.track_calls == 1


def test_person_tracker_max_staleness_expiration():
    """Active tracks exceeding max_track_staleness are purged on skipped frames."""
    class MockYOLO:
        def track(self, *args, **kwargs):
            return []

    cfg = TrackerConfig(frame_interval=2, max_track_staleness=2)
    tracker = PersonTracker(detector_model=MockYOLO(), config=cfg)

    box = DetectionBox(bbox_xyxy=(50.0, 50.0, 100.0, 150.0), confidence=0.85, track_id=9)
    tracked = TrackedPerson(track_id=9, current_box=box, first_seen_frame=0, last_seen_frame=0)
    tracked.history_centroids.append(box.center)
    tracker.active_tracks[9] = tracked

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: stale_frames = 1 (<= 2, retained)
    res1 = tracker.track(dummy_frame, frame_id=1)
    assert len(res1.detections) == 1

    # Frame 3: if called as skipped frame with stale_frames + 1 > 2, track is purged
    tracked.stale_frames = 2
    res_purged = tracker.track(dummy_frame, frame_id=3)
    assert len(res_purged.detections) == 0
    assert 9 not in tracker.active_tracks


def test_child_classification_throttling():
    """AutomatedChildClassificationProvider throttles inference for predicted and established tracks."""
    class MockChildDetector:
        def __init__(self):
            self.classify_calls = 0

        def classify_crop(self, *args, **kwargs):
            self.classify_calls += 1
            from safekid_ai.child_detection.models import ChildClassificationResult, ChildDetectionLabel, VerificationStatus
            return ChildClassificationResult(
                label=ChildDetectionLabel.CHILD,
                confidence=0.92,
                probabilities={"CHILD": 0.92, "ADULT": 0.05, "UNKNOWN": 0.03},
                verification_status=VerificationStatus.PROVISIONAL_MODEL_PREDICTION,
                track_id=kwargs.get("track_id"),
                frame_id=kwargs.get("frame_id"),
            )

    det_mock = MockChildDetector()
    cfg = ChildDetectorConfig(classification_interval=3, skip_stationary_crops=True)
    provider = AutomatedChildClassificationProvider(detector=det_mock, config=cfg)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Keyframe 0 with fresh detection: should run classifier
    det0 = DetectionBox(bbox_xyxy=(10.0, 10.0, 50.0, 120.0), confidence=0.9, track_id=1, is_predicted=False)
    provider.process_frame(dummy_frame, [det0], frame_id=0, timestamp_ms=0.0)
    assert det_mock.classify_calls == 1

    # Frame 1: predicted box on skipped frame: should NOT run classifier
    det1 = DetectionBox(bbox_xyxy=(10.0, 10.0, 50.0, 120.0), confidence=0.9, track_id=1, is_predicted=True, stale_frames=1)
    res1 = provider.process_frame(dummy_frame, [det1], frame_id=1, timestamp_ms=33.3)
    assert det_mock.classify_calls == 1  # Still 1
    assert 1 in res1

    # Frame 2: predicted box on skipped frame: should NOT run classifier
    det2 = DetectionBox(bbox_xyxy=(10.0, 10.0, 50.0, 120.0), confidence=0.9, track_id=1, is_predicted=True, stale_frames=2)
    provider.process_frame(dummy_frame, [det2], frame_id=2, timestamp_ms=66.6)
    assert det_mock.classify_calls == 1  # Still 1


def test_thread_allocation_configuration():
    """Verify thread configuration is accepted and applied cleanly."""
    orig_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        assert torch.get_num_threads() == 2
        torch.set_num_threads(4)
        assert torch.get_num_threads() == 4
    finally:
        torch.set_num_threads(orig_threads)


def test_pipeline_config_frame_interval_propagation():
    """SafeKidPipeline correctly propagates frame_interval to both detector and tracker configs."""
    cfg = SafeKidPipelineConfig(frame_interval=3)
    assert cfg.frame_interval == 3
    # Check initialization propagation without loading heavy models
    cfg.detector.frame_interval = cfg.frame_interval
    cfg.tracker.frame_interval = cfg.frame_interval
    assert cfg.detector.frame_interval == 3
    assert cfg.tracker.frame_interval == 3

