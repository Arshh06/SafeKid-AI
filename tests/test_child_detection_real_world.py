"""Unit and integration tests for real-world child detection validation, provenance audit, and leakage prevention.

Validates:
- Dataset provenance validation and tier separation.
- Perceptual duplicate detection and deduplication.
- Split leakage prevention via group-aware partitioning.
- Semantic attribute mapping for academic surveillance datasets (RAP v2, PETA).
- Invalid bounding-box and corrupted image rejection.
- Perspective scale-mismatch gating in ChildDetector.
- Wilson score confidence interval mathematical soundness.
- Full CCTV pipeline integration with automated child detection.
- Ethical safeguards: exclusion of adults and unknown pedestrians from child alerts.
"""

import json
import math
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
import pytest

from safekid_ai.child_detection.classifier_net import ChildAdultClassifierNet
from safekid_ai.child_detection.config import ChildDetectorConfig
from safekid_ai.child_detection.dataset_ingestion import (
    ChildDatasetIngestionEngine,
    DatasetSource,
    compute_dhash,
    hamming_distance,
)
from safekid_ai.child_detection.detector import ChildDetector
from safekid_ai.child_detection.models import (
    AbstentionReason,
    ChildClassificationResult,
    ChildDetectionLabel,
    VerificationStatus,
)
from safekid_ai.child_detection.provider import AutomatedChildClassificationProvider
from safekid_ai.child_safety import ChildSafetyEventEngine, TrackClassification
from safekid_ai.config import SafeKidPipelineConfig, SafetyZoneType
from safekid_ai.detector import DetectionBox
from safekid_ai.events import SafetyEvent, SafetyEventType
from safekid_ai.pipeline import SafeKidPipeline
from safekid_ai.roi.models import (
    AlertSeverity,
    Polygon,
    SafetyZone,
    ZoneRole,
    ZoneSafetyPolicy,
)
from scripts.run_prompt_22_evaluation import wilson_score_interval


class TestDatasetProvenanceAndIngestion:
    """Tests for dataset provenance audit, duplicate detection, and leakage prevention."""

    def test_provenance_manifest_and_tier_separation(self):
        """Verifies provenance inventory clearly separates synthetic fixtures from real surveillance."""
        inv_path = Path("outputs/child_detection_dataset_inventory.json")
        assert inv_path.exists(), "Dataset inventory JSON missing"
        with open(inv_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "genuine_academic_datasets" in data
        assert "reference_benchmark_fixtures" in data
        # Ensure fixtures are explicitly labeled as local fixture/synthetic
        fix_entry = data["reference_benchmark_fixtures"][0]
        assert fix_entry["status"] == "LOCAL_FIXTURE_GENERATED"

    def test_perceptual_hash_and_duplicate_detection(self, tmp_path):
        """Tests perceptual hashing (dHash) and deduplication logic."""
        img1 = np.full((100, 40, 3), 120, dtype=np.uint8)
        cv2.rectangle(img1, (10, 20), (30, 80), (200, 50, 50), -1)

        # Identical image with 1-pixel shift (near-duplicate)
        img2 = img1.copy()
        img2[21:81, 10:30] = (200, 50, 50)

        # Completely different image (dark)
        img3 = np.full((100, 40, 3), 30, dtype=np.uint8)
        cv2.rectangle(img3, (5, 5), (35, 95), (20, 200, 20), -1)

        h1 = compute_dhash(img1)
        h2 = compute_dhash(img2)
        h3 = compute_dhash(img3)

        assert hamming_distance(h1, h2) <= 2, "Near-duplicates should have low Hamming distance"
        assert hamming_distance(h1, h3) > 5, "Distinct images should have larger Hamming distance"

    def test_split_leakage_prevention_by_group(self, tmp_path):
        """Verifies that all crops from the same video/camera group remain in the SAME split."""
        engine = ChildDatasetIngestionEngine(target_root=tmp_path / "real_world")

        # Create 6 mock image files across 2 camera groups
        test_images_dir = tmp_path / "raw_images"
        test_images_dir.mkdir()

        candidates = []
        for i in range(6):
            img_p = test_images_dir / f"img_{i}.png"
            # Render visually distinct patterns to prevent dHash deduplication
            np.random.seed(100 + i)
            img = np.random.randint(0, 255, (120, 50, 3), dtype=np.uint8)
            cv2.rectangle(img, (5 + i * 4, 10 + i * 8), (25 + i * 3, 50 + i * 8), (220, 40, 40), -1)
            cv2.imwrite(str(img_p), img)
            
            # Groups: 0, 1, 2 share group_cam_01; 3, 4, 5 share group_cam_02
            gid = "group_cam_01" if i < 3 else "group_cam_02"
            candidates.append({
                "sample_id": i,
                "image_path": str(img_p),
                "bbox_xyxy": [0, 0, 50, 120],
                "group_id": gid,
                "camera_id": gid,
                "raw_attributes": {"Age31_45": 1},
            })

        record = engine.ingest_candidate_records(
            raw_candidates=candidates,
            source=DatasetSource.RAP_V2,
            subfolder_name="test_leakage_group",
        )

        assert record.valid_samples == 6

        # Read generated manifest
        with open(record.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        samples = manifest["samples"]
        # Verify that all group_cam_01 samples belong to the SAME split
        cam1_splits = {s["split"] for s in samples if s["group_id"] == "group_cam_01"}
        assert len(cam1_splits) == 1, f"Split leakage detected: group_cam_01 spans {cam1_splits}"

        cam2_splits = {s["split"] for s in samples if s["group_id"] == "group_cam_02"}
        assert len(cam2_splits) == 1, f"Split leakage detected: group_cam_02 spans {cam2_splits}"

    def test_rap_v2_attribute_mapping_semantic_uncertainty(self):
        """RAP v2 AgeLess16 must map conservatively to UNKNOWN to avoid false child intrusion alarms."""
        engine = ChildDatasetIngestionEngine()
        
        # Candidate with AgeLess16 = 1
        label, conf, reason = engine.map_attribute_to_operational_class(
            raw_attributes={"AgeLess16": 1},
            source=DatasetSource.RAP_V2,
        )
        assert label == ChildDetectionLabel.UNKNOWN
        assert conf < 0.65
        assert "bundles 13-15yo teens" in reason

        # Mature adult
        label_ad, conf_ad, _ = engine.map_attribute_to_operational_class(
            raw_attributes={"Age31_45": 1},
            source=DatasetSource.RAP_V2,
        )
        assert label_ad == ChildDetectionLabel.ADULT
        assert conf_ad >= 0.90

    def test_invalid_bbox_and_corrupted_image_rejection(self, tmp_path):
        """Invalid bounding boxes and corrupted/missing images must be rejected cleanly."""
        engine = ChildDatasetIngestionEngine(target_root=tmp_path / "real_world")

        # 1. Non-existent image
        cands = [
            {
                "sample_id": 1,
                "image_path": str(tmp_path / "non_existent.jpg"),
                "bbox_xyxy": [0, 0, 50, 100],
            },
            {
                "sample_id": 2,
                "image_path": str(tmp_path / "valid.jpg"),
                "bbox_xyxy": [50, 50, 10, 10],  # Inverted coordinates (x2 <= x1)
            }
        ]
        # Create valid image for sample 2
        valid_p = tmp_path / "valid.jpg"
        cv2.imwrite(str(valid_p), np.full((100, 100, 3), 120, dtype=np.uint8))

        record = engine.ingest_candidate_records(cands, source=DatasetSource.RAP_V2, subfolder_name="test_rejections")
        assert record.valid_samples == 0
        assert record.corrupted_images == 1
        assert record.invalid_bboxes == 1


class TestClassifierGeneralizationAndGating:
    """Tests for perspective scale sanity cross-check, checkpoint integrity, and confidence intervals."""

    def test_perspective_scale_mismatch_gate(self):
        """Entity occupying > 45% of frame in immediate foreground cannot be a child."""
        detector = ChildDetector(ChildDetectorConfig(model_weights_path="models/child_classifier_v1.pt"))
        
        # Synthetic child crop
        crop = np.full((250, 80, 3), 125, dtype=np.uint8)
        cv2.circle(crop, (40, 30), 20, (160, 190, 230), -1)
        cv2.rectangle(crop, (15, 55), (65, 170), (220, 60, 40), -1)

        # Huge bbox in immediate foreground: y1=200, y2=1000 in 1080p frame (height=800px > 74% frame height, bottom_y=0.92)
        res = detector.classify_crop(
            crop=crop,
            bbox_xyxy=(100.0, 200.0, 300.0, 1000.0),
            frame_size=(1920, 1080),
        )

        assert res.label == ChildDetectionLabel.UNKNOWN
        assert res.abstention_reason == AbstentionReason.SCALE_MISMATCH
        assert res.features.get("rel_height", 0.0) > 0.45

    def test_checkpoint_integrity_and_forward(self):
        """Verifies models/child_classifier_v1.pt loads cleanly and produces finite probabilities."""
        ckpt_path = Path("models/child_classifier_v1.pt")
        assert ckpt_path.exists(), "Model checkpoint missing"
        
        detector = ChildDetector(ChildDetectorConfig(model_weights_path=str(ckpt_path)))
        assert detector.is_weights_loaded is True

        dummy_crop = np.full((100, 40, 3), 120, dtype=np.uint8)
        res = detector.classify_crop(dummy_crop)
        assert res.confidence >= 0.0
        assert math.isfinite(res.inference_time_ms)
        assert res.inference_time_ms > 0.0

    def test_wilson_score_confidence_intervals(self):
        """Verifies statistical validity of Wilson score confidence interval calculation."""
        # 0 successes out of 10
        low0, up0 = wilson_score_interval(0, 10)
        assert low0 == 0.0
        assert up0 > 0.0  # Wilson interval properly reflects positive upper bound for 0/10

        # 10 successes out of 10 (100%)
        low10, up10 = wilson_score_interval(10, 10)
        assert low10 < 1.0  # Proper lower bound reflects uncertainty with small N
        assert up10 == 1.0

        # As N increases, margin narrows
        low_large, up_large = wilson_score_interval(1000, 1000)
        assert (up_large - low_large) < (up10 - low10)


class TestFullCCTVPipelineEndToEnd:
    """Validates complete CCTV pipeline: YOLO -> ByteTrack -> ChildDetector -> ChildSafetyEventEngine."""

    def test_full_pipeline_adult_exclusion_in_danger_zone(self):
        """Verifies that an adult tracked through a playground perimeter fence never fires a child alert."""
        pipeline_cfg = SafeKidPipelineConfig(
            community_id="community_greenfield",
            community_name="Greenfield Enclave",
            camera_id="cam_playground_north",
            zone_type=SafetyZoneType.PLAYGROUND,
            enable_automated_child_detection=True,
            child_detector_config=ChildDetectorConfig(
                model_weights_path="models/child_classifier_v1.pt",
                confidence_threshold=0.65,
            ),
            roi_config_path="configs/cameras/community_002/camera_playground_01.json",
        )
        pipeline = SafeKidPipeline(pipeline_cfg)

        # Create mock frame with an adult silhouette entering the zone
        frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
        # Adult at (500, 100) -> (560, 340) (height 240, width 60, aspect ratio 4.0)
        cv2.circle(frame, (530, 130), 16, (140, 170, 210), -1)
        cv2.rectangle(frame, (510, 150), (550, 270), (45, 45, 50), -1)
        cv2.rectangle(frame, (515, 270), (525, 338), (40, 45, 55), -1)
        cv2.rectangle(frame, (535, 270), (545, 338), (40, 45, 55), -1)

        # Run 3 frames
        child_alerts = []
        for fid in range(1, 4):
            res = pipeline.process_frame(frame, frame_id=fid, timestamp_ms=fid * 40.0, annotate=False)
            if res.child_events:
                child_alerts.extend(res.child_events)

        # Zero child alerts must be generated for adult
        assert len(child_alerts) == 0

    def test_unclassified_track_danger_zone_suppression(self):
        """Uncertain/unclassified detections must be excluded from child safety alert streams."""
        provider = AutomatedChildClassificationProvider()
        zone = SafetyZone(
            roi_id="zone_danger",
            name="Restricted Zone",
            zone_type=SafetyZoneType.VEHICULAR_GATEWAY,
            polygon=Polygon([[0.0, 0.0], [400.0, 0.0], [400.0, 400.0], [0.0, 400.0]]),
            severity=AlertSeverity.CRITICAL,
            policy=ZoneSafetyPolicy(zone_role=ZoneRole.DANGER, entry_alert=True),
        )
        engine = ChildSafetyEventEngine(annotation_provider=provider)

        raw_evt = SafetyEvent(
            event_id="evt_test_unclassified",
            event_type=SafetyEventType.ZONE_ENTRY,
            timestamp_ms=1000.0,
            frame_id=1,
            community_id="comm_001",
            camera_id="cam_01",
            track_id=99,
            roi_id=zone.roi_id,
            zone_name=zone.name,
            zone_type=zone.zone_type,
            severity=zone.severity,
            bottom_center=(200.0, 200.0),
            bbox_xyxy=(180.0, 150.0, 220.0, 250.0),
            dwell_time_seconds=0.0,
            message="Unclassified entity in restricted zone",
            policy=zone.policy,
        )

        child_events = engine.process_events([raw_evt], frame_id=1, timestamp_ms=1000.0)
        assert len(child_events) == 0
        assert len(engine.excluded_events_history) == 1
        assert engine.excluded_events_history[0].classification == TrackClassification.UNCLASSIFIED
