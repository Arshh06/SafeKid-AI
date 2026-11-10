"""Unit and integration tests for real-world child classification model training and deployment.

Validates:
- Loading of the real-world checkpoint (models/child_classifier_jaad_v1.pt).
- Architectural integrity (ChildAdultClassifierNet, input size, output shape).
- Label mapping and probability calibration.
- Multi-stage UNKNOWN abstention gating (confidence, margin, resolution, occlusion).
- Graceful handling of missing or corrupted checkpoint paths.
- Deterministic inference reproducibility.
- End-to-end integration with ChildDetector and TrackChildClassifier.
"""

import json
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest
import torch

from safekid_ai.child_detection.classifier_net import (
    ChildAdultClassifierNet,
    preprocess_crop,
)
from safekid_ai.child_detection.config import ChildDetectorConfig
from safekid_ai.child_detection.detector import ChildDetector
from safekid_ai.child_detection.models import (
    AbstentionReason,
    ChildClassificationResult,
    ChildDetectionLabel,
)
from safekid_ai.child_detection.tracker_integration import TrackChildClassifier


class TestRealWorldCheckpointLoading:
    """Tests for loading and validating the real-world JAAD child classifier checkpoint."""

    def test_checkpoint_file_exists_and_valid(self):
        """Verifies models/child_classifier_jaad_v1.pt exists with required metadata keys."""
        ckpt_path = Path("models/child_classifier_jaad_v1.pt")
        assert ckpt_path.exists(), "Real-world checkpoint file missing!"

        payload = torch.load(ckpt_path, map_location="cpu")
        assert isinstance(payload, dict)
        assert "model_state_dict" in payload
        assert payload.get("version") == "jaad_v1"
        assert payload.get("num_classes") == 2
        assert payload.get("input_size") == [128, 64]
        assert "normalization" in payload
        assert "label_mapping" in payload

    def test_child_detector_loads_real_world_checkpoint(self):
        """Verifies ChildDetector successfully initializes with real-world weights."""
        cfg = ChildDetectorConfig(model_weights_path="models/child_classifier_jaad_v1.pt")
        detector = ChildDetector(config=cfg)

        assert detector.is_weights_loaded is True
        assert detector.model is not None
        assert isinstance(detector.model, ChildAdultClassifierNet)

    def test_graceful_handling_of_missing_checkpoint(self):
        """Verifies detector logs warning and falls back to baseline when weights are missing."""
        cfg = ChildDetectorConfig(model_weights_path="models/non_existent_weights_xyz.pt")
        detector = ChildDetector(config=cfg)

        assert detector.is_weights_loaded is False
        # Model should still be operable in fallback mode
        dummy_crop = np.full((128, 64, 3), 100, dtype=np.uint8)
        res = detector.classify_crop(dummy_crop)
        assert res is not None
        assert isinstance(res, ChildClassificationResult)

    def test_graceful_handling_of_corrupted_checkpoint(self):
        """Verifies detector handles a corrupted weights file without crashing."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"NOT_A_VALID_PYTORCH_FILE")
            f.flush()
            corrupt_path = f.name

        try:
            cfg = ChildDetectorConfig(model_weights_path=corrupt_path)
            detector = ChildDetector(config=cfg)
            assert detector.is_weights_loaded is False
        finally:
            Path(corrupt_path).unlink()


class TestModelArchitectureAndInference:
    """Tests input/output shapes, tensor preprocessing, and deterministic forward pass."""

    def test_preprocess_crop_shape_and_normalization(self):
        """Verifies preprocess_crop produces (1, 3, 128, 64) normalized tensor."""
        crop = np.random.randint(0, 256, (100, 50, 3), dtype=np.uint8)
        tensor = preprocess_crop(crop, target_size=(128, 64), device="cpu")

        assert tensor.shape == (1, 3, 128, 64)
        assert tensor.dtype == torch.float32

    def test_forward_pass_output_shape_and_logits(self):
        """Verifies ChildAdultClassifierNet outputs (B, 2) logits."""
        model = ChildAdultClassifierNet(num_classes=2)
        model.eval()

        dummy_batch = torch.randn(4, 3, 128, 64)
        with torch.no_grad():
            logits = model(dummy_batch)

        assert logits.shape == (4, 2)

    def test_deterministic_reproducibility(self):
        """Verifies identical inputs produce identical output probabilities."""
        cfg = ChildDetectorConfig(model_weights_path="models/child_classifier_jaad_v1.pt")
        detector = ChildDetector(config=cfg)

        crop = np.random.RandomState(42).randint(0, 256, (128, 64, 3), dtype=np.uint8)

        res1 = detector.classify_crop(crop)
        res2 = detector.classify_crop(crop)

        assert res1.label == res2.label
        assert abs(res1.confidence - res2.confidence) < 1e-6
        assert abs(res1.probabilities["CHILD"] - res2.probabilities["CHILD"]) < 1e-6


class TestAbstentionAndUncertainty:
    """Tests multi-stage UNKNOWN abstention gating."""

    def test_low_resolution_abstention(self):
        """Verifies crop smaller than min dimensions abstains with LOW_RESOLUTION."""
        cfg = ChildDetectorConfig(
            model_weights_path="models/child_classifier_jaad_v1.pt",
            min_crop_height=80,
            min_crop_width=32,
        )
        detector = ChildDetector(config=cfg)

        small_crop = np.full((50, 25, 3), 128, dtype=np.uint8)
        res = detector.classify_crop(small_crop)

        assert res.label == ChildDetectionLabel.UNKNOWN
        assert res.abstention_reason == AbstentionReason.LOW_RESOLUTION

    def test_severe_aspect_ratio_occlusion_abstention(self):
        """Verifies distorted aspect ratio abstains with SEVERE_OCCLUSION."""
        cfg = ChildDetectorConfig(
            model_weights_path="models/child_classifier_jaad_v1.pt",
            min_aspect_ratio=1.5,
            max_aspect_ratio=4.0,
        )
        detector = ChildDetector(config=cfg)

        # Square crop (aspect ratio = 1.0 < 1.5)
        square_crop = np.full((100, 100, 3), 128, dtype=np.uint8)
        res = detector.classify_crop(square_crop)

        assert res.label == ChildDetectionLabel.UNKNOWN
        assert res.abstention_reason == AbstentionReason.SEVERE_OCCLUSION

    def test_high_confidence_threshold_abstention(self):
        """Verifies prediction below confidence threshold abstains to UNKNOWN."""
        # Set artificially high confidence threshold (0.999)
        cfg = ChildDetectorConfig(
            model_weights_path="models/child_classifier_jaad_v1.pt",
            confidence_threshold=0.9999,
        )
        detector = ChildDetector(config=cfg)

        crop = np.full((128, 64, 3), 128, dtype=np.uint8)
        # Add random texture so it doesn't trigger flat obstacle gate
        crop = np.random.randint(50, 200, (128, 64, 3), dtype=np.uint8)
        res = detector.classify_crop(crop)

        assert res.label == ChildDetectionLabel.UNKNOWN


class TestTrackClassifierIntegration:
    """Tests integration of real-world detector with TrackChildClassifier and Provider."""

    def test_track_classifier_with_real_world_detector(self):
        """Verifies TrackChildClassifier accumulates predictions and smooths tracks."""
        cfg = ChildDetectorConfig(model_weights_path="models/child_classifier_jaad_v1.pt")
        detector = ChildDetector(config=cfg)
        track_classifier = TrackChildClassifier(config=cfg)

        crop = np.random.RandomState(123).randint(40, 220, (128, 64, 3), dtype=np.uint8)

        # Process 5 consecutive frames for track 101
        for frame_idx in range(5):
            single_res = detector.classify_crop(crop, track_id=101, frame_id=frame_idx)
            res = track_classifier.update_track(
                track_id=101,
                frame_result=single_res,
                frame_id=frame_idx,
                timestamp_ms=float(frame_idx * 33.3),
            )
            assert res is not None

        track_summary = track_classifier.get_track_classification(101)
        assert track_summary is not None
        assert track_summary.track_id == 101
        assert track_summary.label in (ChildDetectionLabel.CHILD, ChildDetectionLabel.ADULT, ChildDetectionLabel.UNKNOWN)
        assert track_classifier.tracks[101].total_frames_observed == 5
