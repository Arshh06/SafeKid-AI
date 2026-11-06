"""Unit and integration test suite for genuine dataset validation, audit protocols, and metric correctness."""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import cv2

from safekid_ai.child_safety import TrackClassification
from safekid_ai.datasets.virat_annotations import (
    VIRATAnnotationParser,
    VIRATObjectDetection,
    VIRATObjectType,
)
from safekid_ai.datasets.virat_evaluator import (
    VIRATDetectionEvaluator,
    VIRATDetectionMetrics,
    VIRATTrackingEvaluator,
    calculate_iou,
)
from safekid_ai.datasets.virat_loader import VIRATDatasetLoader, VIRATSequenceMetadata
from safekid_ai.datasets.virat_validator import SequenceValidationResult, VIRATDatasetValidator
from safekid_ai.detector import DetectionBox


def test_validator_detects_corrupted_or_missing_video(tmp_path: Path):
    """Verifies that VIRATDatasetValidator correctly flags missing video files as errors."""
    fake_ann = tmp_path / "VIRAT_S_fake.viratdata.objects.txt"
    fake_ann.write_text("1 10 0 100 100 50 100 1\n", encoding="utf-8")

    meta = VIRATSequenceMetadata(
        sequence_id="VIRAT_S_fake",
        video_path=tmp_path / "VIRAT_S_fake.mp4",  # does not exist
        objects_path=fake_ann,
        has_video=False,
        has_annotations=True,
    )
    validator = VIRATDatasetValidator()
    result = validator.validate_sequence(meta)

    assert not result.is_valid
    assert len(result.errors) >= 1
    assert any("missing or unreadable" in err.lower() for err in result.errors)


def test_validator_detects_invalid_bounding_box_coordinates(tmp_path: Path):
    """Verifies that bounding boxes with negative coordinates or non-positive dimensions are flagged as errors."""
    ann_file = tmp_path / "VIRAT_S_invalid.viratdata.objects.txt"
    # Line 1: valid
    # Line 2: negative width (-20)
    # Line 3: negative y (-50)
    ann_content = (
        "1 10 0 100 100 50 100 1\n"
        "2 5 0 200 200 -20 100 1\n"
        "3 5 1 300 -50 50 100 1\n"
    )
    ann_file.write_text(ann_content, encoding="utf-8")

    meta = VIRATSequenceMetadata(
        sequence_id="VIRAT_S_invalid",
        video_path=None,
        objects_path=ann_file,
        has_video=False,
        has_annotations=True,
    )
    validator = VIRATDatasetValidator()
    result = validator.validate_sequence(meta)

    assert not result.is_valid
    assert result.annotation_summary["invalid_bbox_count"] == 2
    assert any("non-positive dimensions or negative coordinates" in err for err in result.errors)


def test_validator_detects_duplicate_annotations(tmp_path: Path):
    """Verifies that duplicate (track_id, frame_id) annotations are detected and reported as warnings."""
    ann_file = tmp_path / "VIRAT_S_dup.viratdata.objects.txt"
    # Duplicate track_id 1 on frame 0
    ann_content = (
        "1 10 0 100 100 50 100 1\n"
        "1 10 0 105 105 50 100 1\n"
        "2 10 0 300 300 60 120 1\n"
    )
    ann_file.write_text(ann_content, encoding="utf-8")

    meta = VIRATSequenceMetadata(
        sequence_id="VIRAT_S_dup",
        video_path=None,
        objects_path=ann_file,
        has_video=False,
        has_annotations=True,
    )
    validator = VIRATDatasetValidator()
    result = validator.validate_sequence(meta)

    assert result.annotation_summary["duplicate_records"] == 1
    assert any("duplicate" in warn.lower() for warn in result.warnings)


def test_validator_detects_temporal_desynchronization(tmp_path: Path):
    """Verifies that annotations referencing frames beyond the video length trigger synchronization warnings."""
    video_path = tmp_path / "test_sync.mp4"
    # Create 10-frame dummy video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (100, 100))
    for _ in range(10):
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()

    ann_file = tmp_path / "test_sync.viratdata.objects.txt"
    # Annotation references frame 50 (exceeding 10 frames!)
    ann_file.write_text("1 5 50 10 10 20 40 1\n", encoding="utf-8")

    meta = VIRATSequenceMetadata(
        sequence_id="test_sync",
        video_path=video_path,
        objects_path=ann_file,
        has_video=True,
        has_annotations=True,
        total_frames=10,
    )
    validator = VIRATDatasetValidator()
    result = validator.validate_sequence(meta)

    assert not result.synchronization_summary["is_synchronized"]
    assert any("exceeds video frame count" in warn for warn in result.warnings)


def test_dataset_tier_separation():
    """Verifies that synthetic fixtures and genuine footage are categorized cleanly and never conflated."""
    meta_fixture = VIRATSequenceMetadata(
        sequence_id="VIRAT_S_000001_sample",
        video_path=Path("data/virat/fixtures/videos/VIRAT_S_000001_sample.mp4"),
        objects_path=Path("data/virat/fixtures/annotations/VIRAT_S_000001_sample.viratdata.objects.txt"),
        is_genuine=False,
        tier="fixture",
    )
    meta_genuine = VIRATSequenceMetadata(
        sequence_id="VIRAT_S_000002",
        video_path=Path("data/virat/genuine/videos/VIRAT_S_000002.mp4"),
        objects_path=Path("data/virat/genuine/annotations/VIRAT_S_000002.viratdata.objects.txt"),
        is_genuine=True,
        tier="genuine",
    )

    validator = VIRATDatasetValidator()
    res_fix = validator.validate_sequence(meta_fixture)
    res_gen = validator.validate_sequence(meta_genuine)

    assert res_fix.tier == "fixture"
    assert not res_fix.is_genuine

    assert res_gen.tier == "genuine"
    assert res_gen.is_genuine


def test_evaluator_empty_detection_frames_contribute_false_negatives():
    """Verifies that when ground-truth exists but predictions are empty, false negatives are accurately recorded."""
    evaluator = VIRATDetectionEvaluator(iou_threshold=0.50)
    gt_person = VIRATObjectDetection(track_id=1, frame_id=0, bbox_xywh=(100, 100, 50, 100), object_type=VIRATObjectType.PERSON)

    # Frame 0: 1 GT person, 0 predictions
    evaluator.record_frame(frame_id=0, ground_truth=[gt_person], predictions=[])

    metrics = evaluator.compute_metrics()
    assert metrics.total_ground_truth_persons == 1
    assert metrics.total_predicted_persons == 0
    assert metrics.true_positives == 0
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 1
    assert metrics.recall == 0.0
    assert metrics.precision == 0.0


def test_evaluator_hungarian_idf1_independent_manual_example():
    """Independently verifies Hungarian IDF1 bipartite matching on a manually calculable example.

    Ground Truth:
      Track 1 present for 10 frames (F0..F9)
      Track 2 present for 10 frames (F0..F9)
      Total GT instances = 20

    Predictions:
      Pred 10 matches GT 1 on F0..F9 (10 overlapping frames)
      Pred 20 matches GT 2 on F0..F4 (5 overlapping frames)
      Pred 21 matches GT 2 on F5..F9 (5 overlapping frames)
      Total Pred instances = 20

    Hungarian Bipartite Assignment:
      GT 1 is assigned Pred 10 -> IDTP = 10
      GT 2 is assigned Pred 20 (or Pred 21) -> IDTP = 5
      Total IDTP = 15

    CLEAR MOT IDF1 Formula:
      IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
           = 2 * 15 / (2 * 15 + (20 - 15) + (20 - 15))
           = 30 / (30 + 5 + 5) = 30 / 40 = 0.75 (75.0%)
    """
    tracker_eval = VIRATTrackingEvaluator(iou_threshold=0.50)

    for fid in range(10):
        # GT boxes
        gt1 = VIRATObjectDetection(track_id=1, frame_id=fid, bbox_xywh=(100, 100, 50, 100), object_type=VIRATObjectType.PERSON)
        gt2 = VIRATObjectDetection(track_id=2, frame_id=fid, bbox_xywh=(300, 300, 50, 100), object_type=VIRATObjectType.PERSON)

        # Pred boxes
        p10 = DetectionBox(bbox_xyxy=(100, 100, 150, 200), confidence=0.9, track_id=10)
        p2_id = 20 if fid < 5 else 21  # ID switch at frame 5!
        p2 = DetectionBox(bbox_xyxy=(300, 300, 350, 400), confidence=0.9, track_id=p2_id)

        tracker_eval.record_frame(fid, [gt1, gt2], [p10, p2])

    metrics = tracker_eval.compute_metrics()
    assert metrics.evaluated_frames == 10
    assert metrics.total_gt_tracks == 2
    assert metrics.total_predicted_tracks == 3
    assert metrics.id_switches == 1

    # Exactly 0.75 (75.0% IDF1) as calculated manually!
    assert abs(metrics.idf1 - 0.75) < 1e-4
    assert abs(metrics.id_precision - 0.75) < 1e-4
    assert abs(metrics.id_recall - 0.75) < 1e-4


def test_evaluator_ap_independent_manual_example():
    """Independently verifies AP calculation against a manually calculable Precision-Recall step function."""
    eval = VIRATDetectionEvaluator(iou_threshold=0.50)

    # Frame 0: 1 GT, 1 Pred (Matching, conf=0.9) -> TP
    gt0 = VIRATObjectDetection(track_id=1, frame_id=0, bbox_xywh=(0, 0, 10, 10), object_type=VIRATObjectType.PERSON)
    p0 = DetectionBox(bbox_xyxy=(0, 0, 10, 10), confidence=0.9)
    eval.record_frame(0, [gt0], [p0])

    # Frame 1: 1 GT, 1 Pred (Non-matching, conf=0.8) -> FP + FN
    gt1 = VIRATObjectDetection(track_id=2, frame_id=1, bbox_xywh=(50, 50, 10, 10), object_type=VIRATObjectType.PERSON)
    p1 = DetectionBox(bbox_xyxy=(200, 200, 10, 10), confidence=0.8)
    eval.record_frame(1, [gt1], [p1])

    metrics = eval.compute_metrics()
    assert metrics.total_ground_truth_persons == 2
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5

    # With 1 TP at conf 0.9 (Rec=0.5, Prec=1.0) and 1 FP at conf 0.8 (Rec=0.5, Prec=0.5):
    # AP = 0.5 * 1.0 = 0.50
    assert abs(metrics.ap_50 - 0.50) < 1e-4


def test_privacy_invariants_no_demographics_on_detections():
    """Verifies that DetectionBox and FrameDetections strictly avoid demographic or persistent biometric attributes."""
    box = DetectionBox(bbox_xyxy=(10.0, 20.0, 50.0, 80.0), confidence=0.85, track_id=42)

    # Ensure no demographic or biometric fields exist
    assert not hasattr(box, "age")
    assert not hasattr(box, "gender")
    assert not hasattr(box, "race")
    assert not hasattr(box, "face_embedding")
    assert not hasattr(box, "biometric_id")

    # Only ephemeral spatial/temporal tracking data is stored
    assert box.track_id == 42
    assert box.bottom_center == (30.0, 80.0)

