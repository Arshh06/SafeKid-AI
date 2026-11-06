"""Comprehensive test suite for SafeKid AI dedicated ChildDetector subsystem.

Validates:
- Supervised child detection on pedestrian silhouette crops.
- Supervised adult detection.
- Abstention gating yielding UNKNOWN on low resolution, severe occlusion, and degraded crops.
- Configurable confidence and decision margin thresholds.
- Anti-flickering hysteresis and temporal confidence aggregation.
- Track-level classification persistence through temporary occlusions (grace window).
- Graceful temporal decay to UNKNOWN on prolonged signal loss.
- Track ID discontinuity isolation (no silent label transfer across identities).
- Integration with ChildSafetyEventEngine: child danger zone entry fires child safety alerts.
- Adult danger zone entry: child alerts suppressed.
- Unknown pedestrian danger zone entry: child alerts suppressed.
- Parental consent and staff review authorization safeguards.
"""

from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import pytest

from safekid_ai.child_detection.classifier_net import ChildAdultClassifierNet
from safekid_ai.child_detection.config import ChildDetectorConfig
from safekid_ai.child_detection.detector import ChildDetector
from safekid_ai.child_detection.models import (
    AbstentionReason,
    ChildClassificationResult,
    ChildDetectionLabel,
    VerificationStatus,
)
from safekid_ai.child_detection.provider import AutomatedChildClassificationProvider
from safekid_ai.child_detection.tracker_integration import TrackChildClassifier
from safekid_ai.child_safety import (
    ChildSafetyEventEngine,
    ChildSafetyEventType,
    TrackClassification,
)
from safekid_ai.detector import DetectionBox
from safekid_ai.events import SafetyEvent, SafetyEventType
from safekid_ai.roi.models import (
    AlertSeverity,
    CameraROIConfig,
    Polygon,
    SafetyZone,
    SafetyZoneType,
    ZoneRole,
    ZoneSafetyPolicy,
)


def _create_mock_pedestrian_crop(is_child: bool, height: int = 100, width: int = 36) -> np.ndarray:
    """Helper rendering a synthetic pedestrian crop with child or adult proportions."""
    from scripts.generate_child_dataset_fixture import render_pedestrian_crop
    h = height if height else (90 if is_child else 170)
    w = width if width else (32 if is_child else 50)
    head_r = 0.25 if is_child else 0.13
    color = (220, 60, 40) if is_child else (45, 45, 50)
    skin = (160, 190, 230) if is_child else (130, 160, 210)
    return render_pedestrian_crop(
        is_child=is_child,
        height=h,
        width=w,
        head_ratio=head_r,
        clothing_color=color,
        skin_tone=skin,
    )


def _create_danger_zone(roi_id: str = "zone_gate", name: str = "Vehicular Gateway") -> SafetyZone:
    """Helper creating a danger-role safety zone."""
    return SafetyZone(
        roi_id=roi_id,
        name=name,
        zone_type=SafetyZoneType.VEHICULAR_GATEWAY,
        polygon=Polygon([[0.0, 0.0], [300.0, 0.0], [300.0, 300.0], [0.0, 300.0]]),
        severity=AlertSeverity.HIGH,
        policy=ZoneSafetyPolicy(
            zone_role=ZoneRole.DANGER,
            entry_alert=True,
            exit_alert=False,
            severity=AlertSeverity.HIGH,
        ),
    )


def _create_zone_event(track_id: int, zone: SafetyZone, frame_id: int = 10, ts_ms: float = 1000.0) -> SafetyEvent:
    return SafetyEvent(
        event_id=f"evt_{track_id}_{frame_id}",
        event_type=SafetyEventType.ZONE_ENTRY,
        timestamp_ms=ts_ms,
        frame_id=frame_id,
        community_id="comm_001",
        camera_id="cam_gate_01",
        track_id=track_id,
        roi_id=zone.roi_id,
        zone_name=zone.name,
        zone_type=zone.zone_type,
        severity=zone.severity,
        bottom_center=(150.0, 150.0),
        bbox_xyxy=(130.0, 100.0, 170.0, 150.0),
        dwell_time_seconds=0.0,
        message=f"Track ID {track_id} entered {zone.name}",
        policy=zone.policy,
    )


class TestChildDetectorCore:
    """Tests for single-crop classification and abstention gating in ChildDetector."""

    def test_child_detection_crop(self):
        """Verifies child crop produces CHILD label with confidence >= threshold."""
        detector = ChildDetector(ChildDetectorConfig(model_weights_path="models/child_classifier_v1.pt"))
        crop = _create_mock_pedestrian_crop(is_child=True, height=90, width=32)
        res = detector.classify_crop(crop)
        
        assert res.label == ChildDetectionLabel.CHILD
        assert res.is_child is True
        assert res.is_adult is False
        assert res.confidence >= 0.65
        assert res.abstention_reason == AbstentionReason.NONE
        assert res.inference_time_ms > 0.0

    def test_adult_detection_crop(self):
        """Verifies adult crop produces ADULT label."""
        detector = ChildDetector(ChildDetectorConfig(model_weights_path="models/child_classifier_v1.pt"))
        crop = _create_mock_pedestrian_crop(is_child=False, height=170, width=48)
        res = detector.classify_crop(crop)
        
        assert res.label == ChildDetectionLabel.ADULT
        assert res.is_adult is True
        assert res.is_child is False
        assert res.confidence >= 0.65
        assert res.abstention_reason == AbstentionReason.NONE

    def test_unknown_classification_low_resolution(self):
        """Small or distant pedestrians below min_crop_height must abstain to UNKNOWN."""
        detector = ChildDetector(ChildDetectorConfig(min_crop_height=36))
        # Height 24px is below 36px threshold
        crop = _create_mock_pedestrian_crop(is_child=True, height=24, width=12)
        res = detector.classify_crop(crop)
        
        assert res.label == ChildDetectionLabel.UNKNOWN
        assert res.is_unknown is True
        assert res.abstention_reason == AbstentionReason.LOW_RESOLUTION
        assert res.confidence == 0.0

    def test_unknown_classification_severe_occlusion(self):
        """Severe aspect ratio distortion or solid barrier must abstain to UNKNOWN."""
        detector = ChildDetector(ChildDetectorConfig(min_aspect_ratio=1.0, max_aspect_ratio=4.8))
        
        # Abnormally wide/short crop (e.g. height 25, width 50, aspect_ratio = 0.5)
        distorted_crop = np.full((25, 50, 3), 120, dtype=np.uint8)
        res = detector.classify_crop(distorted_crop)
        assert res.label == ChildDetectionLabel.UNKNOWN
        assert res.abstention_reason in (AbstentionReason.SEVERE_OCCLUSION, AbstentionReason.LOW_RESOLUTION)

    def test_unknown_classification_degraded_crop(self):
        """Empty or None crops must abstain with DEGRADED_CROP."""
        detector = ChildDetector()
        res_none = detector.classify_crop(None)
        assert res_none.label == ChildDetectionLabel.UNKNOWN
        assert res_none.abstention_reason == AbstentionReason.DEGRADED_CROP

        res_empty = detector.classify_crop(np.zeros((0, 0, 3), dtype=np.uint8))
        assert res_empty.label == ChildDetectionLabel.UNKNOWN
        assert res_empty.abstention_reason == AbstentionReason.DEGRADED_CROP

    def test_classification_confidence_threshold_gating(self):
        """Strict confidence threshold forces abstention when evidence is insufficient."""
        # Force unattainable confidence threshold 1.0
        strict_detector = ChildDetector(ChildDetectorConfig(
            model_weights_path="models/child_classifier_v1.pt",
            confidence_threshold=1.0,
        ))
        crop = _create_mock_pedestrian_crop(is_child=True, height=90, width=32)
        res = strict_detector.classify_crop(crop)
        
        assert res.label == ChildDetectionLabel.UNKNOWN
        assert res.abstention_reason == AbstentionReason.LOW_CONFIDENCE


class TestTrackClassificationAggregation:
    """Tests for temporal aggregation, flickering suppression, and occlusion handling in ByteTrack."""

    def test_classification_flickering_prevention(self):
        """A single noisy child frame must not immediately trigger CHILD track classification."""
        aggregator = TrackChildClassifier(ChildDetectorConfig(temporal_confirmation_frames=3))
        
        # Frame 1: Glitch frame predicts CHILD
        glitch_result = ChildClassificationResult(
            label=ChildDetectionLabel.CHILD,
            confidence=0.85,
            probabilities={"CHILD": 0.85, "ADULT": 0.15, "UNKNOWN": 0.0},
        )
        res1 = aggregator.update_track(track_id=1, frame_result=glitch_result, frame_id=1, timestamp_ms=40.0)
        # Should NOT be promoted to CHILD yet on just 1 frame!
        assert res1.label == ChildDetectionLabel.UNKNOWN

        # Frame 2: Another CHILD frame (consecutive = 2)
        res2 = aggregator.update_track(track_id=1, frame_result=glitch_result, frame_id=2, timestamp_ms=80.0)
        assert res2.label == ChildDetectionLabel.UNKNOWN

        # Frame 3: Third consecutive CHILD frame (consecutive = 3 == temporal_confirmation_frames)
        res3 = aggregator.update_track(track_id=1, frame_result=glitch_result, frame_id=3, timestamp_ms=120.0)
        # Now confirmed!
        assert res3.label == ChildDetectionLabel.CHILD

    def test_track_persistence_through_temporary_occlusion(self):
        """Confirmed child track preserves status during short occlusions (grace period)."""
        aggregator = TrackChildClassifier(ChildDetectorConfig(
            temporal_confirmation_frames=2,
            temporal_decay_frames=5,
        ))
        
        child_res = ChildClassificationResult(
            label=ChildDetectionLabel.CHILD,
            confidence=0.90,
            probabilities={"CHILD": 0.90, "ADULT": 0.10, "UNKNOWN": 0.0},
        )
        
        # Establish track as CHILD
        aggregator.update_track(track_id=5, frame_result=child_res, frame_id=1, timestamp_ms=40.0)
        aggregator.update_track(track_id=5, frame_result=child_res, frame_id=2, timestamp_ms=80.0)
        assert aggregator.get_track_classification(5).label == ChildDetectionLabel.CHILD

        # Frames 3, 4: Pedestrian temporarily occluded by car (UNKNOWN with LOW_RESOLUTION)
        occ_res = ChildClassificationResult(
            label=ChildDetectionLabel.UNKNOWN,
            confidence=0.0,
            abstention_reason=AbstentionReason.LOW_RESOLUTION,
        )
        res3 = aggregator.update_track(track_id=5, frame_result=occ_res, frame_id=3, timestamp_ms=120.0)
        res4 = aggregator.update_track(track_id=5, frame_result=occ_res, frame_id=4, timestamp_ms=160.0)
        
        # Must retain CHILD classification during short 2-frame grace period!
        assert res3.label == ChildDetectionLabel.CHILD
        assert res4.label == ChildDetectionLabel.CHILD

    def test_graceful_temporal_decay_to_unknown(self):
        """Prolonged occlusion exceeding temporal_decay_frames must safely revert to UNKNOWN."""
        aggregator = TrackChildClassifier(ChildDetectorConfig(
            temporal_confirmation_frames=2,
            temporal_decay_frames=3,
        ))
        child_res = ChildClassificationResult(
            label=ChildDetectionLabel.CHILD,
            confidence=0.90,
            probabilities={"CHILD": 0.90, "ADULT": 0.10, "UNKNOWN": 0.0},
        )
        aggregator.update_track(track_id=7, frame_result=child_res, frame_id=1, timestamp_ms=40.0)
        aggregator.update_track(track_id=7, frame_result=child_res, frame_id=2, timestamp_ms=80.0)
        assert aggregator.get_track_classification(7).label == ChildDetectionLabel.CHILD

        occ_res = ChildClassificationResult(label=ChildDetectionLabel.UNKNOWN, confidence=0.0)
        # Feed 4 consecutive unknown frames (> temporal_decay_frames = 3)
        for f in range(3, 7):
            aggregator.update_track(track_id=7, frame_result=occ_res, frame_id=f, timestamp_ms=f * 40.0)

        # Must have decayed safely to UNKNOWN
        assert aggregator.get_track_classification(7).label == ChildDetectionLabel.UNKNOWN
        assert aggregator.get_track_classification(7).abstention_reason == AbstentionReason.TEMPORAL_DECAY

    def test_track_id_discontinuity_isolation(self):
        """New track identity starting after track dropout must not inherit previous track label."""
        aggregator = TrackChildClassifier(ChildDetectorConfig(temporal_confirmation_frames=2))
        child_res = ChildClassificationResult(
            label=ChildDetectionLabel.CHILD,
            confidence=0.90,
            probabilities={"CHILD": 0.90, "ADULT": 0.10, "UNKNOWN": 0.0},
        )
        aggregator.update_track(track_id=10, frame_result=child_res, frame_id=1, timestamp_ms=40.0)
        aggregator.update_track(track_id=10, frame_result=child_res, frame_id=2, timestamp_ms=80.0)
        assert aggregator.get_track_classification(10).label == ChildDetectionLabel.CHILD

        # Track 10 is lost and purged
        aggregator.purge_lost_tracks(active_track_ids=set())

        # New track 11 appears
        res_new = aggregator.get_track_classification(11)
        assert res_new.label == ChildDetectionLabel.UNKNOWN
        assert res_new.abstention_reason == AbstentionReason.UNINITIALIZED


class TestChildSafetyEngineIntegration:
    """Tests for integration between AutomatedChildClassificationProvider and ChildSafetyEventEngine."""

    def test_child_danger_zone_entry_scenario(self):
        """Child entering danger zone triggers CHILD_ZONE_ENTRY safety alert."""
        provider = AutomatedChildClassificationProvider()
        zone = _create_danger_zone()
        engine = ChildSafetyEventEngine(annotation_provider=provider)

        # Simulate frame where track 1 is classified as CHILD
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw child crop inside frame at (100, 100) -> (132, 190)
        crop = _create_mock_pedestrian_crop(is_child=True, height=90, width=32)
        frame[100:190, 100:132] = crop
        
        det = DetectionBox(bbox_xyxy=(100.0, 100.0, 132.0, 190.0), confidence=0.95, track_id=1)
        
        # Process 3 frames to establish confirmed child state
        for fid in range(1, 4):
            provider.process_frame(frame, [det], frame_id=fid, timestamp_ms=fid * 40.0)

        # Trigger zone entry event
        raw_event = _create_zone_event(track_id=1, zone=zone)
        child_events = engine.process_events(safety_events=[raw_event], frame_id=4, timestamp_ms=160.0)

        assert len(child_events) == 1
        assert child_events[0].event_type == ChildSafetyEventType.CHILD_ZONE_ENTRY
        assert child_events[0].track_id == 1
        assert child_events[0].severity == AlertSeverity.HIGH
        assert "CHILD ALERT" in child_events[0].message or "Child" in child_events[0].message

    def test_adult_danger_zone_entry_suppression(self):
        """Adult entering danger zone generates raw event but is excluded from child safety alerts."""
        provider = AutomatedChildClassificationProvider()
        zone = _create_danger_zone()
        engine = ChildSafetyEventEngine(annotation_provider=provider)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        crop = _create_mock_pedestrian_crop(is_child=False, height=180, width=50)
        frame[50:230, 100:150] = crop
        det = DetectionBox(bbox_xyxy=(100.0, 50.0, 150.0, 230.0), confidence=0.95, track_id=2)

        for fid in range(1, 4):
            provider.process_frame(frame, [det], frame_id=fid, timestamp_ms=fid * 40.0)

        raw_event = _create_zone_event(track_id=2, zone=zone)
        child_events = engine.process_events(safety_events=[raw_event], frame_id=4, timestamp_ms=160.0)

        # Must NOT generate child safety event!
        assert len(child_events) == 0
        # Audit log must record exclusion
        assert len(engine.excluded_events_history) == 1
        assert engine.excluded_events_history[0].track_id == 2
        assert engine.excluded_events_history[0].classification == TrackClassification.ADULT

    def test_unknown_pedestrian_danger_zone_suppression(self):
        """Uncertain/unknown pedestrian entering danger zone must not generate false child alarm."""
        provider = AutomatedChildClassificationProvider()
        zone = _create_danger_zone()
        engine = ChildSafetyEventEngine(annotation_provider=provider)

        # Distant figure yielding UNKNOWN
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        crop = _create_mock_pedestrian_crop(is_child=True, height=20, width=8) # < 36px
        frame[100:120, 100:108] = crop
        det = DetectionBox(bbox_xyxy=(100.0, 100.0, 108.0, 120.0), confidence=0.80, track_id=3)

        for fid in range(1, 4):
            provider.process_frame(frame, [det], frame_id=fid, timestamp_ms=fid * 40.0)

        raw_event = _create_zone_event(track_id=3, zone=zone)
        child_events = engine.process_events(safety_events=[raw_event], frame_id=4, timestamp_ms=160.0)

        assert len(child_events) == 0
        assert len(engine.excluded_events_history) == 1
        assert engine.excluded_events_history[0].classification == TrackClassification.UNCLASSIFIED

    def test_consent_and_staff_review_safeguards(self):
        """Verifies staff confirmation and dismissal controls."""
        provider = AutomatedChildClassificationProvider(allow_provisional_alerts=False)
        zone = _create_danger_zone()
        engine = ChildSafetyEventEngine(annotation_provider=provider)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        crop = _create_mock_pedestrian_crop(is_child=True, height=90, width=32)
        frame[100:190, 100:132] = crop
        det = DetectionBox(bbox_xyxy=(100.0, 100.0, 132.0, 190.0), confidence=0.95, track_id=4)

        for fid in range(1, 4):
            provider.process_frame(frame, [det], frame_id=fid, timestamp_ms=fid * 40.0)

        # Since allow_provisional_alerts=False and track 4 not enrolled, classification is UNCLASSIFIED
        raw_event = _create_zone_event(track_id=4, zone=zone)
        child_events = engine.process_events(safety_events=[raw_event], frame_id=4, timestamp_ms=160.0)
        assert len(child_events) == 0

        # Staff confirms candidate after reviewing consent
        provider.confirm_child_candidate(track_id=4)
        assert provider.get_classification(4) == TrackClassification.CHILD

        # Now zone entry correctly generates child safety alert
        child_events_confirmed = engine.process_events(safety_events=[raw_event], frame_id=5, timestamp_ms=200.0)
        assert len(child_events_confirmed) == 1
        assert child_events_confirmed[0].track_id == 4

        # Staff later dismisses track
        provider.dismiss_child_candidate(track_id=4)
        assert provider.get_classification(4) == TrackClassification.UNCLASSIFIED
        child_events_dismissed = engine.process_events(safety_events=[raw_event], frame_id=6, timestamp_ms=240.0)
        assert len(child_events_dismissed) == 0

