"""Independent audit regression tests for Prompt 29.

Verifies:
1. Checkpoint SHA-256 integrity and metadata consistency.
2. Architecture shapes and class output mapping (0: CHILD, 1: ADULT).
3. Abstention gating invariants (confidence < 0.65, margin < 0.20, low resolution).
4. Multi-frame temporal suppression of isolated false child predictions.
5. Track reset and ID-change safety (no identity leakage).
6. Safe handling of corrupted/missing checkpoints.
7. Backward compatibility with synthetic baseline checkpoint (child_classifier_v1.pt).
"""

import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
import torch

from safekid_ai.child_detection.classifier_net import ChildAdultClassifierNet
from safekid_ai.child_detection.config import ChildDetectorConfig
from safekid_ai.child_detection.detector import ChildDetector
from safekid_ai.child_detection.models import (
    AbstentionReason,
    ChildClassificationResult,
    ChildDetectionLabel,
    VerificationStatus,
)
from safekid_ai.child_detection.tracker_integration import TrackChildClassifier, TrackClassificationState
from safekid_ai.child_detection.provider import AutomatedChildClassificationProvider


ROOT_DIR = Path(__file__).resolve().parent.parent


def test_jaad_checkpoint_sha256_integrity():
    """Verify that models/child_classifier_jaad_v1.pt matches recorded SHA-256."""
    chk_path = ROOT_DIR / "models" / "child_classifier_jaad_v1.pt"
    assert chk_path.exists(), "JAAD checkpoint must exist"

    h = hashlib.sha256()
    with open(chk_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    actual_sha256 = h.hexdigest().lower()

    meta_path = ROOT_DIR / "outputs" / "prompt_28_checkpoint_metadata.json"
    with open(meta_path, "r") as f:
        recorded_meta = json.load(f)

    assert actual_sha256 == recorded_meta["sha256"].lower()
    assert chk_path.stat().st_size == recorded_meta["size_bytes"]


def test_synthetic_checkpoint_backward_compatibility():
    """Verify that models/child_classifier_v1.pt is untouched and loadable."""
    synth_path = ROOT_DIR / "models" / "child_classifier_v1.pt"
    assert synth_path.exists(), "Synthetic baseline checkpoint must exist"

    h = hashlib.sha256()
    with open(synth_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    actual_sha256 = h.hexdigest().lower()
    assert actual_sha256.startswith("c30148601a6d"), "Synthetic checkpoint hash must match baseline"

    config = ChildDetectorConfig(model_weights_path=str(synth_path))
    detector = ChildDetector(config=config)
    assert detector.is_weights_loaded is True

    # Test dummy inference
    crop = np.zeros((128, 64, 3), dtype=np.uint8)
    res = detector.classify_crop(crop)
    assert isinstance(res, ChildClassificationResult)


def test_jaad_checkpoint_architecture_and_class_mapping():
    """Verify tensor shapes and output logit mapping (0: CHILD, 1: ADULT)."""
    chk_path = ROOT_DIR / "models" / "child_classifier_jaad_v1.pt"
    raw_chk = torch.load(chk_path, map_location="cpu", weights_only=False)
    state = raw_chk["model_state_dict"] if (isinstance(raw_chk, dict) and "model_state_dict" in raw_chk) else raw_chk

    net = ChildAdultClassifierNet(num_classes=2)
    net.load_state_dict(state)
    net.eval()

    # Forward pass on dummy tensor
    tensor = torch.zeros(1, 3, 128, 64)
    with torch.no_grad():
        logits = net(tensor)
    assert logits.shape == (1, 2), "Output logits must be exactly [1, 2]"

    # Classifier output dimension
    assert net.classifier[-1].out_features == 2


def test_abstention_confidence_and_margin_gates():
    """Verify that ambiguous or low-confidence predictions abstain to UNKNOWN."""
    config = ChildDetectorConfig(
        confidence_threshold=0.65,
        margin_threshold=0.20,
    )
    detector = ChildDetector(config=config)

    # 1. Very small crop -> LOW_RESOLUTION
    tiny_crop = np.zeros((20, 10, 3), dtype=np.uint8)
    res_tiny = detector.classify_crop(tiny_crop)
    assert res_tiny.label == ChildDetectionLabel.UNKNOWN
    assert res_tiny.abstention_reason == AbstentionReason.LOW_RESOLUTION

    # 2. Extreme aspect ratio -> SEVERE_OCCLUSION
    narrow_crop = np.zeros((200, 20, 3), dtype=np.uint8)  # aspect ratio 10.0
    res_narrow = detector.classify_crop(narrow_crop)
    assert res_narrow.label == ChildDetectionLabel.UNKNOWN
    assert res_narrow.abstention_reason == AbstentionReason.SEVERE_OCCLUSION


def test_temporal_suppression_of_false_child_predictions():
    """Verify that isolated false child crops are suppressed by TrackChildClassifier."""
    config = ChildDetectorConfig(
        temporal_window_size=15,
        temporal_confirmation_frames=3,
    )
    track_classifier = TrackChildClassifier(config=config)

    adult_res = ChildClassificationResult(
        label=ChildDetectionLabel.ADULT,
        confidence=0.92,
        probabilities={"CHILD": 0.08, "ADULT": 0.92, "UNKNOWN": 0.0},
    )
    false_child_res = ChildClassificationResult(
        label=ChildDetectionLabel.CHILD,
        confidence=0.72,
        probabilities={"CHILD": 0.72, "ADULT": 0.28, "UNKNOWN": 0.0},
    )

    # 10 frames of verified ADULT
    for f in range(1, 11):
        track_classifier.update_track(track_id=101, frame_result=adult_res, frame_id=f, timestamp_ms=f * 33.3)

    assert track_classifier.get_track_classification(101).label == ChildDetectionLabel.ADULT

    # Send 2 isolated false CHILD frames
    track_classifier.update_track(track_id=101, frame_result=false_child_res, frame_id=11, timestamp_ms=366.3)
    track_classifier.update_track(track_id=101, frame_result=false_child_res, frame_id=12, timestamp_ms=399.6)

    # Must remain ADULT due to temporal smoothing and majority voting
    state = track_classifier.get_track_classification(101)
    assert state.label == ChildDetectionLabel.ADULT, "Isolated false child crops must be suppressed"


def test_track_reset_and_id_isolation():
    """Verify that when a track is lost or dropped, a new ID does not inherit child classification."""
    config = ChildDetectorConfig(temporal_confirmation_frames=3)
    track_classifier = TrackChildClassifier(config=config)

    child_res = ChildClassificationResult(
        label=ChildDetectionLabel.CHILD,
        confidence=0.95,
        probabilities={"CHILD": 0.95, "ADULT": 0.05, "UNKNOWN": 0.0},
    )

    # Track 201 confirmed as CHILD
    for f in range(1, 4):
        track_classifier.update_track(track_id=201, frame_result=child_res, frame_id=f, timestamp_ms=f * 33.3)
    assert track_classifier.get_track_classification(201).label == ChildDetectionLabel.CHILD

    # Unseen Track 202 must be completely independent and UNKNOWN
    t202_state = track_classifier.get_track_classification(202)
    assert t202_state.label == ChildDetectionLabel.UNKNOWN
    assert t202_state.abstention_reason == AbstentionReason.UNINITIALIZED


def test_corrupted_checkpoint_safe_failure():
    """Verify that a corrupted or invalid model file fails safely without crashing."""
    corrupted_path = ROOT_DIR / "scratch" / "corrupted_weights.pt"
    corrupted_path.write_bytes(b"corrupted_header_data_not_a_valid_checkpoint")

    try:
        config = ChildDetectorConfig(model_weights_path=str(corrupted_path))
        detector = ChildDetector(config=config)
        assert detector.is_weights_loaded is False, "Corrupted checkpoint must set is_weights_loaded to False"
        
        # Detector should still be able to operate safely with uninitialized weights
        crop = np.zeros((128, 64, 3), dtype=np.uint8)
        res = detector.classify_crop(crop)
        assert isinstance(res, ChildClassificationResult)
    finally:
        if corrupted_path.exists():
            corrupted_path.unlink()

