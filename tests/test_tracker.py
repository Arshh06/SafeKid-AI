"""Unit tests for ByteTrack PersonTracker."""

import numpy as np
import pytest

from safekid_ai.config import DetectorConfig, TrackerConfig
from safekid_ai.detector import DetectionBox, PersonDetector
from safekid_ai.tracker import PersonTracker, TrackedPerson


@pytest.fixture(scope="module")
def tracker():
    """Initializes PersonTracker with YOLO and ByteTrack."""
    det_config = DetectorConfig(model_name_or_path="yolov8n.pt", device="cpu")
    detector = PersonDetector(det_config)
    tracker_config = TrackerConfig(tracker_config_path="bytetrack.yaml", persist=True)
    return PersonTracker(detector.model, tracker_config)


def test_tracker_initialization(tracker):
    """Verifies that PersonTracker initializes successfully."""
    assert tracker.yolo is not None
    assert tracker.active_tracks == {}


def test_track_frame_execution(tracker):
    """Verifies executing tracker on test frames."""
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    res = tracker.track(frame, frame_id=0, timestamp_ms=0.0)
    assert res.frame_id == 0
    assert isinstance(res.detections, list)


def test_tracker_reset(tracker):
    """Verifies tracker reset cleans active tracks dictionary."""
    dummy_box = DetectionBox((10.0, 10.0, 40.0, 40.0), 0.9, 0, "person", 99)
    tracked = TrackedPerson(99, dummy_box)
    tracker.active_tracks[99] = tracked
    assert len(tracker.active_tracks) == 1

    tracker.reset()
    assert len(tracker.active_tracks) == 0


def test_tracked_person_velocity():
    """Verifies velocity calculation over successive centroids."""
    box1 = DetectionBox((10.0, 10.0, 30.0, 30.0), 0.9, 0, "person", 1)  # center (20, 20)
    tracked = TrackedPerson(1, box1)
    tracked.history_centroids.append((20.0, 20.0))

    box2 = DetectionBox((30.0, 40.0, 50.0, 60.0), 0.9, 0, "person", 1)  # center (40, 50)
    tracked.update(box2, frame_id=1)

    # dx = 40 - 20 = 20, dy = 50 - 20 = 30 over 1 interval
    vx, vy = tracked.velocity_vector
    assert pytest.approx(vx, 0.1) == 20.0
    assert pytest.approx(vy, 0.1) == 30.0

