"""Unit tests for YOLO PersonDetector."""

import numpy as np
import pytest

from safekid_ai.config import DetectorConfig
from safekid_ai.detector import DetectionBox, PersonDetector


@pytest.fixture(scope="module")
def detector():
    """Initializes YOLO person detector with official pretrained weights."""
    config = DetectorConfig(
        model_name_or_path="yolov8n.pt",
        device="cpu",
        confidence_threshold=0.25,
    )
    return PersonDetector(config)


def test_detector_initialization(detector):
    """Verifies successful loading of the official pretrained YOLO checkpoint."""
    assert detector.model is not None


def test_detect_blank_frame(detector):
    """Verifies that empty/blank frame produces no false person detections."""
    blank = np.zeros((320, 320, 3), dtype=np.uint8)
    detections = detector.detect(blank, frame_id=1, timestamp_ms=100.0)
    assert detections.frame_id == 1
    assert detections.timestamp_ms == 100.0
    assert len(detections) == 0


def test_detect_empty_frame_raises(detector):
    """Verifies ValueError when passed None or empty numpy array."""
    with pytest.raises(ValueError):
        detector.detect(None)
    with pytest.raises(ValueError):
        detector.detect(np.array([]))


def test_detection_box_properties_and_crop():
    """Tests DetectionBox geometric properties and crop extraction."""
    box = DetectionBox(
        bbox_xyxy=(10.0, 20.0, 50.0, 80.0),
        confidence=0.88,
        class_id=0,
        class_name="person",
        track_id=1,
    )
    assert box.width == 40.0
    assert box.height == 60.0
    assert box.area == 2400.0
    assert box.center == (30.0, 50.0)

    # Test crop extraction
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 50
    crop = box.extract_crop(frame)
    assert crop.shape == (60, 40, 3)

