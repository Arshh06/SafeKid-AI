"""Automated test suite for RT-DETRv2 detector adapter in SafeKid AI."""

from pathlib import Path
import numpy as np
import pytest
import torch

from safekid_ai.config import DetectorConfig, SafeKidPipelineConfig, TrackerConfig
from safekid_ai.detector import DetectionBox, FrameDetections
from safekid_ai.detectors import (
    BasePersonDetector,
    RTDETRv2DetectorAdapter,
    YOLOv8DetectorAdapter,
    create_detector,
)
from safekid_ai.pipeline import SafeKidPipeline


class TestRTDETRv2DetectorAdapter:
    """Tests for RTDETRv2DetectorAdapter interface and functionality."""

    @pytest.fixture(scope="class")
    def rtdetr_adapter(self):
        """Loads RT-DETRv2 detector adapter once for the test class."""
        cfg = DetectorConfig(
            backend="rtdetrv2",
            model_name_or_path="rtdetr-l.pt",
            confidence_threshold=0.35,
            iou_threshold=0.45,
            device="cpu",
            num_threads=4,
        )
        tracker_cfg = TrackerConfig(
            tracker_type="bytetrack",
            tracker_config_path="bytetrack.yaml",
        )
        return RTDETRv2DetectorAdapter(
            config=cfg,
            tracker_config=tracker_cfg,
            checkpoint_path="rtdetr-l.pt",
        )

    def test_adapter_inheritance_and_properties(self, rtdetr_adapter):
        """Verifies that RTDETRv2DetectorAdapter adheres to BasePersonDetector contract."""
        assert isinstance(rtdetr_adapter, BasePersonDetector)
        assert rtdetr_adapter.backend_name == "rtdetrv2"
        assert rtdetr_adapter.model_name == "rtdetr-l"
        assert rtdetr_adapter.parameter_count > 30_000_000  # ~32.97M parameters
        assert rtdetr_adapter.model_size_mb > 50.0  # ~63.4 MB

    def test_invalid_input_handling(self, rtdetr_adapter):
        """Ensures detector safely raises ValueError on None or empty frames."""
        with pytest.raises(ValueError, match="Input frame is empty or None"):
            rtdetr_adapter.detect(None)

        with pytest.raises(ValueError, match="Input frame is empty or None"):
            rtdetr_adapter.detect(np.empty((0, 0, 3), dtype=np.uint8))

        with pytest.raises(ValueError, match="Input frame is empty or None"):
            rtdetr_adapter.track(None)

    def test_detect_dummy_frame_returns_framedetections(self, rtdetr_adapter):
        """Verifies detect execution on a synthetic blank frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = rtdetr_adapter.detect(
            frame=frame,
            frame_id=1,
            timestamp_ms=41.6,
        )
        assert isinstance(result, FrameDetections)
        assert result.frame_id == 1
        assert result.timestamp_ms == 41.6
        assert isinstance(result.detections, list)

    def test_track_dummy_frame_returns_framedetections(self, rtdetr_adapter):
        """Verifies track execution with ByteTrack on a synthetic frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = rtdetr_adapter.track(
            frame=frame,
            frame_id=1,
            timestamp_ms=41.6,
        )
        assert isinstance(result, FrameDetections)
        assert result.frame_id == 1
        assert result.timestamp_ms == 41.6
        assert isinstance(result.detections, list)

    def test_detection_box_attributes_and_bounds(self, rtdetr_adapter):
        """Tests that detections on a frame with synthetic person-like patterns satisfy schema."""
        # Create a frame with random texture so it passes through inference safely
        frame = np.ones((640, 640, 3), dtype=np.uint8) * 128
        result = rtdetr_adapter.detect(frame=frame, frame_id=5, timestamp_ms=208.3, confidence_thresh=0.01)
        
        for det in result.detections:
            assert isinstance(det, DetectionBox)
            assert det.class_id == 0
            assert det.class_name == "person"
            assert 0.0 <= det.confidence <= 1.0
            x1, y1, x2, y2 = det.bbox_xyxy
            assert 0.0 <= x1 < x2 <= 640.0
            assert 0.0 <= y1 < y2 <= 640.0
            assert det.width > 0.0
            assert det.height > 0.0
            assert det.area > 0.0


class TestDetectorFactoryAndPipeline:
    """Tests for detector factory creation and end-to-end pipeline wiring."""

    def test_create_detector_factory_yolo(self):
        """Verifies create_detector instantiates YOLOv8 adapter."""
        cfg = DetectorConfig(backend="yolov8", model_name_or_path="yolov8n.pt")
        detector = create_detector(cfg)
        assert isinstance(detector, YOLOv8DetectorAdapter)
        assert detector.backend_name == "yolov8"

    def test_create_detector_factory_rtdetr(self):
        """Verifies create_detector instantiates RTDETRv2 adapter."""
        cfg = DetectorConfig(backend="rtdetrv2", model_name_or_path="rtdetr-l.pt")
        detector = create_detector(cfg)
        assert isinstance(detector, RTDETRv2DetectorAdapter)
        assert detector.backend_name == "rtdetrv2"

    def test_create_detector_factory_invalid_backend(self):
        """Verifies create_detector raises ValueError for unknown backends."""
        cfg = DetectorConfig(backend="invalid_detector_xyz")
        with pytest.raises(ValueError, match="Unsupported detector backend"):
            create_detector(cfg)

    def test_pipeline_with_rtdetrv2_backend(self):
        """Verifies SafeKidPipeline seamlessly runs with RT-DETRv2 configured."""
        pipe_cfg = SafeKidPipelineConfig(
            detector=DetectorConfig(
                backend="rtdetrv2",
                model_name_or_path="rtdetr-l.pt",
                confidence_threshold=0.35,
            )
        )
        pipeline = SafeKidPipeline(pipe_cfg)
        assert hasattr(pipeline, "detector_adapter")
        assert isinstance(pipeline.detector_adapter, RTDETRv2DetectorAdapter)

        # Process a frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame, frame_id=0, timestamp_ms=0.0)
        assert result.frame_id == 0
        assert result.timestamp_ms == 0.0
        assert hasattr(result, "events")
        assert hasattr(result, "person_states")

