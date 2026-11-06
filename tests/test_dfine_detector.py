"""Automated test suite for D-FINE detector adapter in SafeKid AI."""

from pathlib import Path
import numpy as np
import pytest
import torch

from safekid_ai.config import DetectorConfig, SafeKidPipelineConfig, TrackerConfig
from safekid_ai.detector import DetectionBox, FrameDetections
from safekid_ai.detectors import (
    BasePersonDetector,
    DFINEDetectorAdapter,
    create_detector,
)
from safekid_ai.pipeline import SafeKidPipeline


class TestDFINEDetectorAdapter:
    """Tests for DFINEDetectorAdapter interface and operational correctness."""

    @pytest.fixture(scope="class")
    def dfine_n_adapter(self):
        """Loads D-FINE-N detector adapter once for the test class."""
        cfg = DetectorConfig(
            backend="dfine",
            model_variant="dfine_n",
            confidence_threshold=0.35,
            iou_threshold=0.45,
            device="cpu",
            num_threads=4,
        )
        tracker_cfg = TrackerConfig(
            tracker_type="bytetrack",
            confidence_threshold=0.35,
        )
        return DFINEDetectorAdapter(
            config=cfg,
            tracker_config=tracker_cfg,
            variant="dfine_n",
        )

    def test_adapter_inheritance_and_properties(self, dfine_n_adapter):
        """Verifies that DFINEDetectorAdapter adheres to BasePersonDetector contract."""
        assert isinstance(dfine_n_adapter, BasePersonDetector)
        assert dfine_n_adapter.backend_name == "dfine"
        assert dfine_n_adapter.model_name == "dfine_n"
        assert dfine_n_adapter.parameter_count > 3_500_000  # ~3.81M parameters
        assert dfine_n_adapter.model_size_mb > 10.0  # ~14.77 MB

    def test_invalid_input_handling(self, dfine_n_adapter):
        """Ensures detector safely raises ValueError on None or empty frames."""
        with pytest.raises(ValueError, match="Input frame is empty or None"):
            dfine_n_adapter.detect(None)

        with pytest.raises(ValueError, match="Input frame is empty or None"):
            dfine_n_adapter.detect(np.empty((0, 0, 3), dtype=np.uint8))

        with pytest.raises(ValueError, match="Input frame is empty or None"):
            dfine_n_adapter.track(None)

    def test_detect_dummy_frame_returns_framedetections(self, dfine_n_adapter):
        """Verifies detect execution on a synthetic blank frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = dfine_n_adapter.detect(
            frame=frame,
            frame_id=1,
            timestamp_ms=41.6,
        )
        assert isinstance(result, FrameDetections)
        assert result.frame_id == 1
        assert result.timestamp_ms == 41.6
        assert isinstance(result.detections, list)

    def test_track_dummy_frame_returns_framedetections(self, dfine_n_adapter):
        """Verifies track execution with ByteTrack on a synthetic frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = dfine_n_adapter.track(
            frame=frame,
            frame_id=1,
            timestamp_ms=41.6,
        )
        assert isinstance(result, FrameDetections)
        assert result.frame_id == 1
        assert result.timestamp_ms == 41.6
        assert isinstance(result.detections, list)

    def test_detection_box_attributes_and_bounds(self, dfine_n_adapter):
        """Tests that detections on a synthetic textured frame satisfy boundary constraints."""
        frame = np.ones((640, 640, 3), dtype=np.uint8) * 128
        result = dfine_n_adapter.detect(frame=frame, frame_id=5, timestamp_ms=208.3, confidence_thresh=0.01)

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


class TestDFINEFactoryAndPipeline:
    """Tests for D-FINE factory creation and end-to-end SafeKidPipeline integration."""

    def test_create_detector_factory_dfine_n(self):
        """Verifies create_detector instantiates D-FINE-N adapter."""
        cfg = DetectorConfig(backend="dfine", model_variant="dfine_n")
        detector = create_detector(cfg)
        assert isinstance(detector, DFINEDetectorAdapter)
        assert detector.backend_name == "dfine"
        assert detector.model_name == "dfine_n"

    def test_create_detector_factory_dfine_s(self):
        """Verifies create_detector instantiates D-FINE-S adapter."""
        cfg = DetectorConfig(backend="dfine_s", model_variant="dfine_s")
        detector = create_detector(cfg)
        assert isinstance(detector, DFINEDetectorAdapter)
        assert detector.backend_name == "dfine"
        assert detector.model_name == "dfine_s"
        assert detector.parameter_count > 9_000_000  # ~10.39M parameters

    def test_pipeline_with_dfine_backend(self):
        """Verifies SafeKidPipeline seamlessly runs with D-FINE configured."""
        pipe_cfg = SafeKidPipelineConfig(
            detector=DetectorConfig(
                backend="dfine",
                model_variant="dfine_n",
                confidence_threshold=0.35,
            )
        )
        pipeline = SafeKidPipeline(pipe_cfg)
        assert hasattr(pipeline, "detector_adapter")
        assert isinstance(pipeline.detector_adapter, DFINEDetectorAdapter)

        # Process a frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame, frame_id=0, timestamp_ms=0.0)
        assert result.frame_id == 0
        assert result.timestamp_ms == 0.0
        assert hasattr(result, "events")
        assert hasattr(result, "person_states")

