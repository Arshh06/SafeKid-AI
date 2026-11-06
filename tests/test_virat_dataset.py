"""Unit and integration test suite for VIRAT surveillance dataset integration and evaluation."""

import json
from pathlib import Path
import tempfile
import pytest

from safekid_ai.child_safety import (
    ChildSafetyEvent,
    ChildSafetyEventEngine,
    TrackClassification,
)
from safekid_ai.datasets.virat_annotations import (
    VIRATAnnotationParser,
    VIRATEvent,
    VIRATFrameAnnotation,
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
from safekid_ai.detector import DetectionBox
from safekid_ai.events import SafetyEvent, SafetyEventType, ZoneStateManager
from safekid_ai.roi.models import (
    AlertSeverity,
    CameraROIConfig,
    SafetyZone,
    SafetyZoneType,
    ZoneRole,
    ZoneSafetyPolicy,
)


def test_virat_annotation_parser_release2_format(tmp_path: Path):
    """Verifies that VIRATAnnotationParser correctly parses official Release 2.0 8-column objects.txt."""
    ann_file = tmp_path / "test.viratdata.objects.txt"
    content = (
        "# VIRAT Release 2.0 Objects\n"
        "1 10 0 100.0 200.0 50.0 120.0 1\n"
        "1 10 1 105.0 200.0 50.0 120.0 1\n"
        "2 5 0 300.0 400.0 200.0 150.0 2\n"
        "3 2 1 500.0 100.0 40.0 40.0 4\n"
    )
    ann_file.write_text(content, encoding="utf-8")

    frame_map = VIRATAnnotationParser.parse_objects_file(ann_file)
    assert 0 in frame_map
    assert 1 in frame_map

    frame_0_objs = frame_map[0]
    assert len(frame_0_objs) == 2
    p1 = frame_0_objs[0]
    assert p1.track_id == 1
    assert p1.frame_id == 0
    assert p1.bbox_xywh == (100.0, 200.0, 50.0, 120.0)
    assert p1.bbox_xyxy == (100.0, 200.0, 150.0, 320.0)
    assert p1.object_type == VIRATObjectType.PERSON
    assert p1.is_person is True
    assert p1.bottom_center == (125.0, 320.0)

    car = frame_0_objs[1]
    assert car.track_id == 2
    assert car.object_type == VIRATObjectType.CAR
    assert car.is_person is False


def test_virat_annotation_parser_fallback_formats(tmp_path: Path):
    """Verifies that 7-token and 6-token variants and malformed rows are handled safely."""
    ann_file = tmp_path / "test_fallback.objects.txt"
    content = (
        "1 0 100 200 50 120 1\n"         # 7 tokens: obj_id, frame_id, x, y, w, h, obj_type
        "2 0 300 400 60 140\n"           # 6 tokens: obj_id, frame_id, x, y, w, h (defaults to person)
        "corrupted line with words\n"    # Malformed line (must be skipped without crashing)
        "3 0 10 20 -5 10 1\n"            # Negative dimension (must be skipped)
    )
    ann_file.write_text(content, encoding="utf-8")

    frame_map = VIRATAnnotationParser.parse_objects_file(ann_file)
    assert 0 in frame_map
    assert len(frame_map[0]) == 2
    assert frame_map[0][0].track_id == 1
    assert frame_map[0][1].track_id == 2
    assert frame_map[0][1].is_person is True


def test_virat_annotation_parser_missing_file_raises_not_found():
    """Verifies that attempting to parse a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        VIRATAnnotationParser.parse_objects_file("non_existent_file.txt")


def test_virat_annotation_parser_events_and_mapping(tmp_path: Path):
    """Verifies parsing of VIRAT events and mapping files."""
    evt_file = tmp_path / "test.events.txt"
    evt_file.write_text("1 5 10 0 9 0 100 200 50 50\n", encoding="utf-8")

    map_file = tmp_path / "test.mapping.txt"
    map_file.write_text("1 2 101 102\n", encoding="utf-8")

    events = VIRATAnnotationParser.parse_events_file(evt_file)
    assert 1 in events
    evt = events[1]
    assert evt.event_id == 1
    assert evt.start_frame == 0
    assert evt.end_frame == 9

    mapping = VIRATAnnotationParser.parse_mapping_file(map_file)
    assert 1 in mapping
    assert mapping[1] == [101, 102]


def test_calculate_iou():
    """Verifies precision of IoU calculation."""
    box_a = (100.0, 100.0, 200.0, 200.0)  # Area 100x100 = 10,000
    box_b = (150.0, 100.0, 250.0, 200.0)  # Overlap 50x100 = 5,000; Union 15,000; IoU = 1/3
    iou = calculate_iou(box_a, box_b)
    assert abs(iou - 0.3333333) < 1e-4

    # Non-overlapping boxes
    box_c = (300.0, 300.0, 400.0, 400.0)
    assert calculate_iou(box_a, box_c) == 0.0

    # Identical boxes
    assert calculate_iou(box_a, box_a) == 1.0


def test_virat_loader_discovers_sample_sequence():
    """Verifies that VIRATDatasetLoader correctly scans data/virat and reports sequence metadata."""
    loader = VIRATDatasetLoader()
    sequences = loader.list_sequences()
    assert len(sequences) >= 1
    assert "VIRAT_S_000001_sample" in sequences

    meta = loader.get_sequence_metadata("VIRAT_S_000001_sample")
    assert meta is not None
    assert meta.is_complete is True
    assert meta.width == 1920
    assert meta.height == 1080
    assert meta.person_instance_count == 144
    assert meta.unique_person_tracks == 2


def test_virat_loader_sequence_annotations():
    """Verifies that loading a sequence exposes frame-level person annotations."""
    loader = VIRATDatasetLoader()
    seq = loader.load_sequence("VIRAT_S_000001_sample")
    frame_0 = seq.get_frame_annotation(0)
    assert frame_0 is not None
    assert len(frame_0.detections) == 3  # 2 persons + 1 car
    assert len(frame_0.person_detections) == 2
    assert len(frame_0.vehicle_detections) == 1


def test_virat_detection_evaluator_perfect_match():
    """Verifies that perfect predictions achieve 100% Precision, Recall, F1, and mAP@0.50."""
    evaluator = VIRATDetectionEvaluator(iou_threshold=0.50)

    gt_person = VIRATObjectDetection(
        track_id=1,
        frame_id=0,
        bbox_xywh=(100.0, 200.0, 60.0, 150.0),
        object_type=VIRATObjectType.PERSON,
    )
    pred_box = DetectionBox(
        bbox_xyxy=(100.0, 200.0, 160.0, 350.0),
        confidence=0.95,
        class_name="person",
    )

    evaluator.record_frame(frame_id=0, ground_truth=[gt_person], predictions=[pred_box], inference_time_ms=10.0)
    metrics: VIRATDetectionMetrics = evaluator.compute_metrics()

    assert metrics.evaluated_frames == 1
    assert metrics.total_ground_truth_persons == 1
    assert metrics.total_predicted_persons == 1
    assert metrics.true_positives == 1
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.ap_50 == 1.0
    assert metrics.mean_iou == 1.0


def test_virat_detection_evaluator_partial_match():
    """Verifies that TP, FP, and FN are counted accurately with degraded precision/recall."""
    evaluator = VIRATDetectionEvaluator(iou_threshold=0.50)

    # Frame 0: 1 GT person, 1 matching prediction, 1 spurious prediction (FP)
    gt_person = VIRATObjectDetection(track_id=1, frame_id=0, bbox_xywh=(100, 100, 50, 100), object_type=VIRATObjectType.PERSON)
    pred_tp = DetectionBox(bbox_xyxy=(102, 100, 150, 200), confidence=0.90)  # Matching
    pred_fp = DetectionBox(bbox_xyxy=(500, 500, 550, 600), confidence=0.80)  # Spurious

    evaluator.record_frame(0, [gt_person], [pred_tp, pred_fp], inference_time_ms=15.0)

    # Frame 1: 1 GT person, 0 predictions (FN)
    gt_person_2 = VIRATObjectDetection(track_id=2, frame_id=1, bbox_xywh=(200, 200, 50, 100), object_type=VIRATObjectType.PERSON)
    evaluator.record_frame(1, [gt_person_2], [], inference_time_ms=15.0)

    metrics = evaluator.compute_metrics()
    assert metrics.evaluated_frames == 2
    assert metrics.total_ground_truth_persons == 2
    assert metrics.total_predicted_persons == 2
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1_score == 0.5


def test_virat_tracking_evaluator_mota_and_id_switches():
    """Verifies tracking evaluation calculates MOTA, IDF1, and detects ID switches."""
    tracker_eval = VIRATTrackingEvaluator(iou_threshold=0.50)

    # Frame 0: GT 1 matched to Pred 10
    gt_1_f0 = VIRATObjectDetection(track_id=1, frame_id=0, bbox_xywh=(100, 100, 50, 100), object_type=VIRATObjectType.PERSON)
    p_10_f0 = DetectionBox(bbox_xyxy=(100, 100, 150, 200), confidence=0.9, track_id=10)
    tracker_eval.record_frame(0, [gt_1_f0], [p_10_f0])

    # Frame 1: GT 1 matched to Pred 11 (ID Switch!)
    gt_1_f1 = VIRATObjectDetection(track_id=1, frame_id=1, bbox_xywh=(110, 100, 50, 100), object_type=VIRATObjectType.PERSON)
    p_11_f1 = DetectionBox(bbox_xyxy=(110, 100, 160, 200), confidence=0.9, track_id=11)
    tracker_eval.record_frame(1, [gt_1_f1], [p_11_f1])

    metrics = tracker_eval.compute_metrics()
    assert metrics.evaluated_frames == 2
    assert metrics.total_gt_tracks == 1
    assert metrics.total_predicted_tracks == 2
    assert metrics.id_switches == 1
    # MOTA = 1 - (FN(0) + FP(0) + IDSW(1)) / Total_GT(2) = 1 - 0.5 = 0.5
    assert abs(metrics.mota - 0.5) < 1e-4


def test_virat_unclassified_safety_policy_blocks_child_alerts():
    """CRITICAL ETHICAL SAFEGUARD: Verifies that surveillance tracks in danger zones trigger ZERO child alerts."""
    from scripts.run_virat_benchmark import VIRATUnclassifiedAnnotationProvider

    provider = VIRATUnclassifiedAnnotationProvider()
    assert provider.get_classification(track_id=1) == TrackClassification.UNCLASSIFIED
    assert provider.get_classification(track_id=99) == TrackClassification.UNCLASSIFIED

    # Set up child safety engine with a DANGER zone
    roi_config = CameraROIConfig(
        community_id="community_virat",
        camera_id="camera_virat_s01",
        frame_width=1920,
        frame_height=1080,
        zones=[
            SafetyZone(
                roi_id="danger_transit",
                name="Danger Transit Lane",
                zone_type="vehicular_gateway",
                severity=AlertSeverity.HIGH,
                polygon=[[0.0, 0.0], [500.0, 0.0], [500.0, 500.0], [0.0, 500.0]],
                policy=ZoneSafetyPolicy(
                    zone_role=ZoneRole.DANGER,
                    entry_alert=True,
                    exit_alert=False,
                    dwell_alert=True,
                    dwell_threshold_seconds=2.0,
                    severity=AlertSeverity.HIGH,
                ),
            )
        ],
    )

    engine = ChildSafetyEventEngine(annotation_provider=provider, roi_config=roi_config)

    # Simulate raw zone entry event into danger zone
    raw_event = SafetyEvent(
        event_id="EVT-001",
        event_type=SafetyEventType.ZONE_ENTRY,
        timestamp_ms=1000.0,
        frame_id=1,
        community_id="community_virat",
        camera_id="camera_virat_s01",
        track_id=1,
        roi_id="danger_transit",
        zone_name="Danger Transit Lane",
        zone_type=SafetyZoneType.VEHICULAR_GATEWAY,
        severity=AlertSeverity.HIGH,
        bottom_center=(250.0, 450.0),
        bbox_xyxy=(220.0, 300.0, 280.0, 450.0),
        dwell_time_seconds=0.0,
        message="Track ID 1 entered Danger Transit Lane",
    )

    child_alerts = engine.process_events([raw_event], frame_id=1, timestamp_ms=1000.0, roi_config=roi_config)

    # UNCLASSIFIED pedestrian must NEVER produce a child-specific danger alert
    assert len(child_alerts) == 0
