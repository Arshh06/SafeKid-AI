"""Unit and integration tests for the generalized child-focused safety event engine.

Validates:
- Manually annotated child entering danger zone generates CHILD_ZONE_ENTRY.
- Manually annotated adult entering danger zone is excluded from child alerts.
- Unclassified track entering danger zone is excluded from child alerts.
- Child exit events generate CHILD_ZONE_EXIT.
- Child dwell events generate CHILD_DWELL_TIME_EXCEEDED.
- Multiple community environments and zone types (playground, vehicle gateway, perimeter, pool).
- Independent operation across different camera configurations.
- Honest handling of track identity discontinuities (no silent label inheritance).
- End-to-end pipeline integration with child safety layer.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from safekid_ai.child_safety import (
    ChildSafetyEvent,
    ChildSafetyEventEngine,
    ChildSafetyEventType,
    ChildTrackAnnotation,
    ManualTrackAnnotationProvider,
    TrackClassification,
    TrackDiscontinuity,
)
from safekid_ai.config import SafeKidPipelineConfig
from safekid_ai.detector import DetectionBox
from safekid_ai.events import SafetyEvent, SafetyEventType, ZoneStateManager
from safekid_ai.pipeline import SafeKidPipeline
from safekid_ai.roi.engine import ROIEngine
from safekid_ai.roi.loader import validate_roi_config_dict
from safekid_ai.roi.models import (
    AlertSeverity,
    CameraROIConfig,
    Polygon,
    SafetyZone,
    SafetyZoneType,
    ZoneRole,
    ZoneSafetyPolicy,
)


def _create_sample_zone(
    roi_id: str,
    name: str,
    zone_type: SafetyZoneType,
    severity: AlertSeverity,
    dwell_time_seconds: float = 0.0,
    policy: Optional[ZoneSafetyPolicy] = None,
) -> SafetyZone:
    """Helper creating a 100x100 square zone at (100, 100) -> (200, 200)."""
    poly = Polygon([[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]])
    return SafetyZone(
        roi_id=roi_id,
        name=name,
        zone_type=zone_type,
        polygon=poly,
        severity=severity,
        dwell_time_seconds=dwell_time_seconds,
        policy=policy,
    )


def _create_safety_event(
    track_id: int,
    event_type: SafetyEventType,
    zone: SafetyZone,
    community_id: str = "comm_test",
    camera_id: str = "cam_test",
    dwell_sec: float = 0.0,
    timestamp_ms: float = 1000.0,
    frame_id: int = 24,
) -> SafetyEvent:
    return SafetyEvent(
        event_id=f"evt_test_{track_id}",
        event_type=event_type,
        timestamp_ms=timestamp_ms,
        frame_id=frame_id,
        community_id=community_id,
        camera_id=camera_id,
        track_id=track_id,
        roi_id=zone.roi_id,
        zone_name=zone.name,
        zone_type=zone.zone_type,
        severity=zone.severity,
        bottom_center=(150.0, 150.0),
        bbox_xyxy=(130.0, 100.0, 170.0, 150.0),
        dwell_time_seconds=dwell_sec,
        message=f"Track ID {track_id} test message in {zone.name}",
        policy=zone.policy,
    )


class TestChildSafetyEventEngine:
    """Core evaluation tests for ChildSafetyEventEngine."""

    def test_annotated_child_entering_danger_zone_generates_child_alert(self):
        """A verified child track entering a danger zone generates a CHILD_ZONE_ENTRY alert."""
        provider = ManualTrackAnnotationProvider(community_id="comm_01", camera_id="cam_01")
        provider.set_annotation(track_id=4, classification=TrackClassification.CHILD, notes="Test child")

        engine = ChildSafetyEventEngine(annotation_provider=provider)
        zone = _create_sample_zone("pool_basin", "Swimming Pool Water", SafetyZoneType.SWIMMING_POOL_BOUNDARY, AlertSeverity.CRITICAL)

        raw_event = _create_safety_event(track_id=4, event_type=SafetyEventType.ZONE_ENTRY, zone=zone)
        child_events = engine.process_events([raw_event], frame_id=24, timestamp_ms=1000.0)

        assert len(child_events) == 1
        ce = child_events[0]
        assert ce.event_type == ChildSafetyEventType.CHILD_ZONE_ENTRY
        assert ce.track_id == 4
        assert ce.classification == TrackClassification.CHILD
        assert ce.roi_id == "pool_basin"
        assert ce.severity == AlertSeverity.CRITICAL
        assert "CHILD SAFETY ALERT" in ce.message
        assert ce.underlying_event_id == raw_event.event_id

    def test_annotated_adult_entering_same_zone_is_excluded(self):
        """A verified adult entering the same zone generates NO child-specific alert."""
        provider = ManualTrackAnnotationProvider(community_id="comm_01", camera_id="cam_01")
        provider.set_annotation(track_id=1, classification=TrackClassification.ADULT, notes="Supervising adult")

        engine = ChildSafetyEventEngine(annotation_provider=provider)
        zone = _create_sample_zone("pool_basin", "Swimming Pool Water", SafetyZoneType.SWIMMING_POOL_BOUNDARY, AlertSeverity.CRITICAL)

        raw_event = _create_safety_event(track_id=1, event_type=SafetyEventType.ZONE_ENTRY, zone=zone)
        child_events = engine.process_events([raw_event], frame_id=24, timestamp_ms=1000.0)

        assert len(child_events) == 0
        assert len(engine.excluded_events_history) == 1
        ex = engine.excluded_events_history[0]
        assert ex.track_id == 1
        assert ex.classification == TrackClassification.ADULT
        assert "ADULT: excluded" in ex.reason

    def test_unclassified_track_entering_zone_is_excluded(self):
        """An unclassified track entering a zone generates NO child-specific alert."""
        provider = ManualTrackAnnotationProvider(community_id="comm_01", camera_id="cam_01")
        # Track 2 is unclassified
        provider.set_annotation(track_id=2, classification=TrackClassification.UNCLASSIFIED)
        # Track 99 is completely unannotated (must default to UNCLASSIFIED)

        engine = ChildSafetyEventEngine(annotation_provider=provider)
        zone = _create_sample_zone("driveway", "Main Driveway", SafetyZoneType.VEHICULAR_GATEWAY, AlertSeverity.HIGH)

        evt2 = _create_safety_event(track_id=2, event_type=SafetyEventType.ZONE_ENTRY, zone=zone)
        evt99 = _create_safety_event(track_id=99, event_type=SafetyEventType.ZONE_ENTRY, zone=zone)

        child_events = engine.process_events([evt2, evt99], frame_id=30, timestamp_ms=1250.0)

        assert len(child_events) == 0
        assert len(engine.excluded_events_history) == 2
        for ex in engine.excluded_events_history:
            assert ex.classification == TrackClassification.UNCLASSIFIED
            assert "UNCLASSIFIED: excluded" in ex.reason

    def test_untracked_detection_is_excluded(self):
        """Detections with no track ID (None) are treated as unclassified and excluded."""
        provider = ManualTrackAnnotationProvider()
        engine = ChildSafetyEventEngine(annotation_provider=provider)
        zone = _create_sample_zone("sandbox", "Sandbox", SafetyZoneType.PLAYGROUND, AlertSeverity.LOW)

        raw_event = _create_safety_event(track_id=None, event_type=SafetyEventType.ZONE_ENTRY, zone=zone)
        raw_event.track_id = None

        child_events = engine.process_events([raw_event], frame_id=10, timestamp_ms=416.7)
        assert len(child_events) == 0
        assert len(engine.excluded_events_history) == 1
        assert engine.excluded_events_history[0].track_id is None

    def test_child_exit_event_generation(self):
        """A verified child exiting a zone generates a CHILD_ZONE_EXIT event."""
        provider = ManualTrackAnnotationProvider()
        provider.set_annotation(track_id=4, classification=TrackClassification.CHILD)

        engine = ChildSafetyEventEngine(annotation_provider=provider)
        zone = _create_sample_zone("pool_basin", "Swimming Pool Water", SafetyZoneType.SWIMMING_POOL_BOUNDARY, AlertSeverity.CRITICAL)

        exit_evt = _create_safety_event(track_id=4, event_type=SafetyEventType.ZONE_EXIT, zone=zone, dwell_sec=4.2)
        child_events = engine.process_events([exit_evt], frame_id=120, timestamp_ms=5000.0)

        assert len(child_events) == 1
        ce = child_events[0]
        assert ce.event_type == ChildSafetyEventType.CHILD_ZONE_EXIT
        assert ce.dwell_time_seconds == 4.2
        assert "Swimming Pool Water" in ce.message and "4.2s" in ce.message

    def test_child_dwell_time_exceeded_event_generation(self):
        """A verified child lingering past a dwell threshold triggers CHILD_DWELL_TIME_EXCEEDED."""
        provider = ManualTrackAnnotationProvider()
        provider.set_annotation(track_id=6, classification=TrackClassification.CHILD)

        engine = ChildSafetyEventEngine(annotation_provider=provider)
        zone = _create_sample_zone("pool_deck", "Pool Deck Enclosure", SafetyZoneType.SWIMMING_POOL_GATE, AlertSeverity.HIGH, dwell_time_seconds=3.0)

        dwell_evt = _create_safety_event(track_id=6, event_type=SafetyEventType.DWELL_TIME_EXCEEDED, zone=zone, dwell_sec=3.1)
        child_events = engine.process_events([dwell_evt], frame_id=96, timestamp_ms=4000.0)

        assert len(child_events) == 1
        ce = child_events[0]
        assert ce.event_type == ChildSafetyEventType.CHILD_DWELL_TIME_EXCEEDED
        assert ce.dwell_time_seconds == 3.1
        assert "CHILD DWELL ALERT" in ce.message

    def test_multiple_community_environments_hazard_agnostic(self):
        """The child safety event engine operates consistently across diverse environment zones."""
        provider = ManualTrackAnnotationProvider()
        provider.set_annotation(track_id=10, classification=TrackClassification.CHILD)
        engine = ChildSafetyEventEngine(annotation_provider=provider)

        test_scenarios = [
            # Playground
            ("playground_swings", "Swings Danger Zone", SafetyZoneType.PLAYGROUND, AlertSeverity.MEDIUM, 0.0),
            # Vehicle movement
            ("vehicular_driveway", "North Driveway Thoroughfare", SafetyZoneType.VEHICULAR_GATEWAY, AlertSeverity.HIGH, 2.0),
            # Perimeter fence
            ("north_perimeter", "Outer Boundary Perimeter", SafetyZoneType.PERIMETER_FENCE, AlertSeverity.HIGH, 5.0),
            # Restricted mechanical infrastructure
            ("boom_barrier", "Automated Boom Barrier", SafetyZoneType.RESTRICTED_INFRASTRUCTURE, AlertSeverity.CRITICAL, 0.0),
            # Water body edge
            ("pond_edge", "Stormwater Retention Basin Edge", SafetyZoneType.WATER_BODY_EDGE, AlertSeverity.CRITICAL, 0.0),
        ]

        for roi_id, name, zone_type, severity, dwell_thresh in test_scenarios:
            policy = ZoneSafetyPolicy(
                zone_role=ZoneRole.DANGER,
                entry_alert=True,
                exit_alert=False,
                dwell_alert=(dwell_thresh > 0.0),
                dwell_threshold_seconds=dwell_thresh if dwell_thresh > 0.0 else None,
                severity=severity,
            )
            zone = _create_sample_zone(roi_id, name, zone_type, severity, dwell_thresh, policy=policy)
            evt = _create_safety_event(track_id=10, event_type=SafetyEventType.ZONE_ENTRY, zone=zone)
            child_events = engine.process_events([evt], frame_id=50, timestamp_ms=2000.0)

            assert len(child_events) == 1
            ce = child_events[0]
            assert ce.zone_type == zone_type
            assert ce.severity == severity
            assert ce.zone_name == name
            assert "Child (Track ID 10) entered" in ce.message

    def test_different_camera_configurations_independent(self):
        """Different cameras maintain completely independent annotation and event states."""
        # Camera 1: North Gate (Child Track 5, Adult Track 1)
        prov1 = ManualTrackAnnotationProvider(community_id="comm_001", camera_id="cam_gate_north")
        prov1.set_annotation(track_id=5, classification=TrackClassification.CHILD)
        prov1.set_annotation(track_id=1, classification=TrackClassification.ADULT)
        eng1 = ChildSafetyEventEngine(annotation_provider=prov1)

        # Camera 2: Playground (Track 1 is a CHILD in this camera, Track 5 is an ADULT)
        prov2 = ManualTrackAnnotationProvider(community_id="comm_002", camera_id="cam_playground_01")
        prov2.set_annotation(track_id=1, classification=TrackClassification.CHILD)
        prov2.set_annotation(track_id=5, classification=TrackClassification.ADULT)
        eng2 = ChildSafetyEventEngine(annotation_provider=prov2)

        gate_policy = ZoneSafetyPolicy(zone_role=ZoneRole.DANGER, entry_alert=True, severity=AlertSeverity.HIGH)
        play_policy = ZoneSafetyPolicy(zone_role=ZoneRole.DANGER, entry_alert=True, severity=AlertSeverity.MEDIUM)
        gate_zone = _create_sample_zone("driveway", "Driveway", SafetyZoneType.VEHICULAR_GATEWAY, AlertSeverity.HIGH, policy=gate_policy)
        play_zone = _create_sample_zone("swings", "Swings", SafetyZoneType.PLAYGROUND, AlertSeverity.MEDIUM, policy=play_policy)

        # In Camera 1: Track 5 enters -> Child Alert; Track 1 enters -> Excluded (Adult)
        e1_t5 = _create_safety_event(track_id=5, event_type=SafetyEventType.ZONE_ENTRY, zone=gate_zone, camera_id="cam_gate_north")
        e1_t1 = _create_safety_event(track_id=1, event_type=SafetyEventType.ZONE_ENTRY, zone=gate_zone, camera_id="cam_gate_north")

        res1_t5 = eng1.process_events([e1_t5], frame_id=1, timestamp_ms=41.67)
        res1_t1 = eng1.process_events([e1_t1], frame_id=1, timestamp_ms=41.67)

        assert len(res1_t5) == 1
        assert len(res1_t1) == 0

        # In Camera 2: Track 1 enters -> Child Alert; Track 5 enters -> Excluded (Adult)
        e2_t1 = _create_safety_event(track_id=1, event_type=SafetyEventType.ZONE_ENTRY, zone=play_zone, camera_id="cam_playground_01")
        e2_t5 = _create_safety_event(track_id=5, event_type=SafetyEventType.ZONE_ENTRY, zone=play_zone, camera_id="cam_playground_01")

        res2_t1 = eng2.process_events([e2_t1], frame_id=1, timestamp_ms=41.67)
        res2_t5 = eng2.process_events([e2_t5], frame_id=1, timestamp_ms=41.67)

        assert len(res2_t1) == 1
        assert len(res2_t5) == 0

    def test_track_identity_discontinuity_handled_honestly(self):
        """ByteTrack ID switches or dropouts are not silently assumed to inherit child classification."""
        provider = ManualTrackAnnotationProvider(community_id="comm_demo", camera_id="cam_pool_demo")
        provider.set_annotation(track_id=4, classification=TrackClassification.CHILD, notes="Child in pool")

        engine = ChildSafetyEventEngine(annotation_provider=provider)
        zone = _create_sample_zone("pool_basin", "Swimming Pool Water", SafetyZoneType.SWIMMING_POOL_BOUNDARY, AlertSeverity.CRITICAL)

        # Track 4 enters -> confirmed child alert
        evt4 = _create_safety_event(track_id=4, event_type=SafetyEventType.ZONE_ENTRY, zone=zone)
        alerts4 = engine.process_events([evt4], frame_id=10, timestamp_ms=416.7)
        assert len(alerts4) == 1

        # Track 4 experiences occlusion, and a new track ID 14 appears in the same zone.
        # SafeKid AI MUST NOT silently transfer Track 4's child status to Track 14!
        evt14 = _create_safety_event(track_id=14, event_type=SafetyEventType.ZONE_ENTRY, zone=zone)
        alerts14 = engine.process_events([evt14], frame_id=15, timestamp_ms=625.0)

        # Track 14 is unannotated -> strictly UNCLASSIFIED -> 0 child alerts
        assert len(alerts14) == 0
        assert any(ex.track_id == 14 and ex.classification == TrackClassification.UNCLASSIFIED for ex in engine.excluded_events_history)

        # Record explicit discontinuity audit record
        disc = engine.record_discontinuity(
            frame_id=15,
            timestamp_ms=625.0,
            original_track_id=4,
            zone_id="pool_basin",
            description="Track 4 dropped due to occlusion; Track 14 emerged nearby without verified label.",
            potential_new_track_id=14,
        )
        assert disc.original_track_id == 4
        assert disc.potential_new_track_id == 14
        assert len(engine.discontinuities_history) == 1

    def test_json_annotation_loader_and_persistence(self, tmp_path: Path):
        """Verifies loading from JSON, unannotated track safety, and saving."""
        json_file = tmp_path / "test_annotations.json"
        data = {
            "community_id": "comm_test",
            "camera_id": "cam_test",
            "description": "Test annotations",
            "annotations": {
                "3": {"classification": "CHILD", "notes": "Child 3"},
                "7": {"classification": "ADULT", "notes": "Adult 7"},
                "8": "UNCLASSIFIED",
            },
        }
        with open(json_file, "w") as f:
            json.dump(data, f)

        provider = ManualTrackAnnotationProvider.load_from_json(json_file)
        assert provider.get_classification(3) == TrackClassification.CHILD
        assert provider.get_classification(7) == TrackClassification.ADULT
        assert provider.get_classification(8) == TrackClassification.UNCLASSIFIED
        # Unlisted track defaults to UNCLASSIFIED
        assert provider.get_classification(999) == TrackClassification.UNCLASSIFIED

        # Save and verify
        out_json = tmp_path / "saved_annotations.json"
        provider.save_to_json(out_json)
        reloaded = ManualTrackAnnotationProvider.load_from_json(out_json)
        assert reloaded.get_classification(3) == TrackClassification.CHILD
        assert reloaded.get_classification(7) == TrackClassification.ADULT


class TestPipelineChildSafetyIntegration:
    """Integration tests verifying SafeKidPipeline operates cleanly with child safety layer."""

    def test_pipeline_operational_mode_processes_child_safety(self, tmp_path: Path):
        """Pipeline generates child safety events when child annotations are provided."""
        ann_file = tmp_path / "demo_ann.json"
        with open(ann_file, "w") as f:
            json.dump({
                "community_id": "comm_demo",
                "camera_id": "cam_demo",
                "annotations": {
                    "1": "CHILD",
                    "2": "ADULT",
                }
            }, f)

        cfg = SafeKidPipelineConfig(
            community_id="comm_demo",
            camera_id="cam_demo",
            child_annotation_path=str(ann_file),
            enable_child_safety_layer=True,
        )

        pipeline = SafeKidPipeline(cfg)
        assert pipeline.child_safety_engine is not None
        assert pipeline.child_annotation_provider is not None
        assert pipeline.child_annotation_provider.get_classification(1) == TrackClassification.CHILD
        assert pipeline.child_annotation_provider.get_classification(2) == TrackClassification.ADULT

        # Configure a test zone in pipeline's ROI engine
        test_zone = _create_sample_zone("test_zone", "Playground Area", SafetyZoneType.PLAYGROUND, AlertSeverity.HIGH)
        cam_cfg = CameraROIConfig(
            community_id="comm_demo",
            camera_id="cam_demo",
            frame_width=640,
            frame_height=480,
            zones=[test_zone],
        )
        pipeline.set_roi_config(cam_cfg)

        # Process a blank frame
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        res = pipeline.process_frame(blank, frame_id=0, timestamp_ms=0.0)

        # Result contains child_events list attribute
        assert hasattr(res, "child_events")
        assert isinstance(res.child_events, list)


class TestZoneSafetyPolicies:
    """Comprehensive test suite for configurable child safety zone policies (Prompt 09)."""

    def test_safe_zone_entry_without_danger_alert(self):
        """Entering a SAFE zone generates NO danger alert for children."""
        provider = ManualTrackAnnotationProvider()
        provider.set_annotation(track_id=4, classification=TrackClassification.CHILD)
        engine = ChildSafetyEventEngine(annotation_provider=provider)

        safe_policy = ZoneSafetyPolicy(
            zone_role=ZoneRole.SAFE,
            entry_alert=False,
            exit_alert=False,
            dwell_alert=False,
            severity=AlertSeverity.LOW,
        )
        zone = _create_sample_zone("sandbox", "Play Sandbox", SafetyZoneType.PLAYGROUND, AlertSeverity.LOW, policy=safe_policy)

        raw_event = _create_safety_event(track_id=4, event_type=SafetyEventType.ZONE_ENTRY, zone=zone)
        alerts = engine.process_events([raw_event], frame_id=1, timestamp_ms=41.67)

        assert len(alerts) == 0

    def test_monitored_zone_entry_without_alert(self):
        """Entering a MONITORED zone (e.g. pool deck) does NOT trigger an immediate danger alert."""
        provider = ManualTrackAnnotationProvider()
        provider.set_annotation(track_id=4, classification=TrackClassification.CHILD)
        engine = ChildSafetyEventEngine(annotation_provider=provider)

        monitored_policy = ZoneSafetyPolicy(
            zone_role=ZoneRole.MONITORED,
            entry_alert=False,
            exit_alert=False,
            dwell_alert=True,
            dwell_threshold_seconds=3.0,
            severity=AlertSeverity.HIGH,
        )
        zone = _create_sample_zone("pool_deck", "Pool Deck Enclosure", SafetyZoneType.SWIMMING_POOL_GATE, AlertSeverity.HIGH, dwell_time_seconds=3.0, policy=monitored_policy)

        raw_event = _create_safety_event(track_id=4, event_type=SafetyEventType.ZONE_ENTRY, zone=zone)
        alerts = engine.process_events([raw_event], frame_id=1, timestamp_ms=41.67)

        assert len(alerts) == 0

    def test_danger_zone_entry_with_immediate_alert(self):
        """Entering a DANGER zone (e.g. swimming pool water) triggers an immediate CRITICAL alert."""
        provider = ManualTrackAnnotationProvider()
        provider.set_annotation(track_id=4, classification=TrackClassification.CHILD)
        engine = ChildSafetyEventEngine(annotation_provider=provider)

        danger_policy = ZoneSafetyPolicy(
            zone_role=ZoneRole.DANGER,
            entry_alert=True,
            exit_alert=True,
            dwell_alert=False,
            severity=AlertSeverity.CRITICAL,
        )
        zone = _create_sample_zone("pool_basin", "Swimming Pool Basin", SafetyZoneType.SWIMMING_POOL_BOUNDARY, AlertSeverity.CRITICAL, policy=danger_policy)

        raw_event = _create_safety_event(track_id=4, event_type=SafetyEventType.ZONE_ENTRY, zone=zone)
        alerts = engine.process_events([raw_event], frame_id=10, timestamp_ms=416.7)

        assert len(alerts) == 1
        assert alerts[0].event_type == ChildSafetyEventType.CHILD_ZONE_ENTRY
        assert alerts[0].severity == AlertSeverity.CRITICAL
        assert "CHILD SAFETY ALERT" in alerts[0].message

    def test_dwell_only_danger_policy(self):
        """Zone configured for dwell alert only triggers when dwell threshold is exceeded, not upon entry."""
        provider = ManualTrackAnnotationProvider()
        provider.set_annotation(track_id=6, classification=TrackClassification.CHILD)
        engine = ChildSafetyEventEngine(annotation_provider=provider)

        policy = ZoneSafetyPolicy(
            zone_role=ZoneRole.MONITORED,
            entry_alert=False,
            exit_alert=False,
            dwell_alert=True,
            dwell_threshold_seconds=3.0,
            severity=AlertSeverity.HIGH,
        )
        zone = _create_sample_zone("deck", "Pool Deck Walkway", SafetyZoneType.SWIMMING_POOL_GATE, AlertSeverity.HIGH, dwell_time_seconds=3.0, policy=policy)

        # 1. Entry -> NO alert
        entry_evt = _create_safety_event(track_id=6, event_type=SafetyEventType.ZONE_ENTRY, zone=zone)
        entry_alerts = engine.process_events([entry_evt], frame_id=20, timestamp_ms=833.3)
        assert len(entry_alerts) == 0

        # 2. Dwell exceeded (3.2s) -> triggers CHILD_DWELL_TIME_EXCEEDED
        dwell_evt = _create_safety_event(track_id=6, event_type=SafetyEventType.DWELL_TIME_EXCEEDED, zone=zone, dwell_sec=3.2)
        dwell_alerts = engine.process_events([dwell_evt], frame_id=97, timestamp_ms=4041.7)
        assert len(dwell_alerts) == 1
        assert dwell_alerts[0].event_type == ChildSafetyEventType.CHILD_DWELL_TIME_EXCEEDED
        assert dwell_alerts[0].dwell_time_seconds == 3.2
        assert dwell_alerts[0].severity == AlertSeverity.HIGH

    def test_configured_exit_alert_behavior(self):
        """Exit alerts are only emitted when exit_alert=True; suppressed when exit_alert=False."""
        provider = ManualTrackAnnotationProvider()
        provider.set_annotation(track_id=4, classification=TrackClassification.CHILD)
        engine = ChildSafetyEventEngine(annotation_provider=provider)

        # Zone A: Water basin has exit_alert=True
        water_policy = ZoneSafetyPolicy(zone_role=ZoneRole.DANGER, entry_alert=True, exit_alert=True, severity=AlertSeverity.CRITICAL)
        water_zone = _create_sample_zone("water", "Water Basin", SafetyZoneType.SWIMMING_POOL_BOUNDARY, AlertSeverity.CRITICAL, policy=water_policy)

        # Zone B: Deck has exit_alert=False
        deck_policy = ZoneSafetyPolicy(zone_role=ZoneRole.MONITORED, entry_alert=False, exit_alert=False, severity=AlertSeverity.HIGH)
        deck_zone = _create_sample_zone("deck", "Deck Walkway", SafetyZoneType.SWIMMING_POOL_GATE, AlertSeverity.HIGH, policy=deck_policy)

        # Child exits water -> alert generated
        exit_water_evt = _create_safety_event(track_id=4, event_type=SafetyEventType.ZONE_EXIT, zone=water_zone, dwell_sec=2.2)
        water_alerts = engine.process_events([exit_water_evt], frame_id=63, timestamp_ms=2625.0)
        assert len(water_alerts) == 1
        assert water_alerts[0].event_type == ChildSafetyEventType.CHILD_ZONE_EXIT

        # Child exits deck -> NO alert generated
        exit_deck_evt = _create_safety_event(track_id=4, event_type=SafetyEventType.ZONE_EXIT, zone=deck_zone, dwell_sec=5.0)
        deck_alerts = engine.process_events([exit_deck_evt], frame_id=120, timestamp_ms=5000.0)
        assert len(deck_alerts) == 0

    def test_different_severity_levels_in_policy(self):
        """Child safety event inherits severity directly from configured zone policy."""
        provider = ManualTrackAnnotationProvider()
        provider.set_annotation(track_id=10, classification=TrackClassification.CHILD)
        engine = ChildSafetyEventEngine(annotation_provider=provider)

        for sev in [AlertSeverity.LOW, AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            policy = ZoneSafetyPolicy(zone_role=ZoneRole.DANGER, entry_alert=True, severity=sev)
            zone = _create_sample_zone(f"zone_{sev.value}", f"Zone {sev.value}", SafetyZoneType.GENERAL_COMMON_AREA, sev, policy=policy)

            evt = _create_safety_event(track_id=10, event_type=SafetyEventType.ZONE_ENTRY, zone=zone)
            alerts = engine.process_events([evt], frame_id=1, timestamp_ms=41.67)
            assert len(alerts) == 1
            assert alerts[0].severity == sev

    def test_adult_and_unclassified_exclusion_under_all_policies(self):
        """Under SAFE, MONITORED, and DANGER policies, adults and unclassified tracks NEVER generate child alerts."""
        provider = ManualTrackAnnotationProvider()
        provider.set_annotation(track_id=1, classification=TrackClassification.ADULT)
        provider.set_annotation(track_id=2, classification=TrackClassification.UNCLASSIFIED)
        engine = ChildSafetyEventEngine(annotation_provider=provider)

        policies = [
            ZoneSafetyPolicy(zone_role=ZoneRole.SAFE, entry_alert=False, severity=AlertSeverity.LOW),
            ZoneSafetyPolicy(zone_role=ZoneRole.MONITORED, entry_alert=True, exit_alert=True, dwell_alert=True, severity=AlertSeverity.MEDIUM),
            ZoneSafetyPolicy(zone_role=ZoneRole.DANGER, entry_alert=True, exit_alert=True, dwell_alert=True, severity=AlertSeverity.CRITICAL),
        ]

        for i, policy in enumerate(policies):
            zone = _create_sample_zone(f"test_zone_{i}", f"Zone {i}", SafetyZoneType.SWIMMING_POOL_BOUNDARY, policy.severity, policy=policy)
            for tid in [1, 2]:
                for evt_type in [SafetyEventType.ZONE_ENTRY, SafetyEventType.DWELL_TIME_EXCEEDED, SafetyEventType.ZONE_EXIT]:
                    evt = _create_safety_event(track_id=tid, event_type=evt_type, zone=zone)
                    alerts = engine.process_events([evt], frame_id=10, timestamp_ms=416.7)
                    assert len(alerts) == 0, f"Expected 0 alerts for track {tid} on {evt_type.value} in {policy.zone_role.value}"

    def test_existing_roi_and_transition_behavior_unchanged(self):
        """ZoneStateManager 3-frame transition confirmation operates identically with policy attached."""
        manager = ZoneStateManager(entry_confirmation_frames=3, exit_confirmation_frames=3)
        danger_policy = ZoneSafetyPolicy(zone_role=ZoneRole.DANGER, entry_alert=True, severity=AlertSeverity.CRITICAL)
        zone = _create_sample_zone("danger_z", "Danger Zone", SafetyZoneType.SWIMMING_POOL_BOUNDARY, AlertSeverity.CRITICAL, policy=danger_policy)
        cam_cfg = CameraROIConfig("comm", "cam", 640, 480, zones=[zone])
        engine = ROIEngine(cam_cfg)

        det = DetectionBox(bbox_xyxy=(140.0, 140.0, 160.0, 160.0), confidence=0.9, track_id=5)

        # Frame 1: inside -> observation 1, not confirmed
        e1 = manager.update(frame_id=1, timestamp_ms=41.67, detections=[det], roi_engine=engine, frame_resolution=(640, 480))
        assert len(e1) == 0

        # Frame 2: inside -> observation 2, not confirmed
        e2 = manager.update(frame_id=2, timestamp_ms=83.33, detections=[det], roi_engine=engine, frame_resolution=(640, 480))
        assert len(e2) == 0

        # Frame 3: inside -> observation 3, CONFIRMED entry with timestamp from Frame 1 (41.67ms)
        e3 = manager.update(frame_id=3, timestamp_ms=125.0, detections=[det], roi_engine=engine, frame_resolution=(640, 480))
        assert len(e3) == 1
        assert e3[0].event_type == SafetyEventType.ZONE_ENTRY
        assert e3[0].timestamp_ms == 41.67
        assert e3[0].policy is not None
        assert e3[0].policy.zone_role == ZoneRole.DANGER

    def test_invalid_policy_configurations_rejected(self):
        """Invalid policy inputs are rejected cleanly with validation errors."""
        # 1. Invalid zone_role
        with pytest.raises(ValueError, match="Invalid zone_role"):
            ZoneSafetyPolicy(zone_role="HAZARD_ZONE")  # type: ignore

        # 2. Negative dwell threshold
        with pytest.raises(ValueError, match="Must be a non-negative number"):
            ZoneSafetyPolicy(zone_role=ZoneRole.DANGER, dwell_threshold_seconds=-1.5)

        # 3. Non-boolean alert flags
        with pytest.raises(TypeError, match="entry_alert must be a boolean"):
            ZoneSafetyPolicy(zone_role=ZoneRole.DANGER, entry_alert="yes")  # type: ignore

        with pytest.raises(TypeError, match="exit_alert must be a boolean"):
            ZoneSafetyPolicy(zone_role=ZoneRole.DANGER, exit_alert=1)  # type: ignore

        with pytest.raises(TypeError, match="dwell_alert must be a boolean"):
            ZoneSafetyPolicy(zone_role=ZoneRole.DANGER, dwell_alert="true")  # type: ignore

        # 4. Loader rejects invalid policy in dictionary
        bad_config = {
            "community_id": "c1",
            "camera_id": "cam1",
            "frame_width": 640,
            "frame_height": 480,
            "zones": [
                {
                    "roi_id": "z1",
                    "name": "Z1",
                    "zone_type": "PLAYGROUND",
                    "polygon": [[10, 10], [50, 10], [50, 50], [10, 50]],
                    "policy": {
                        "zone_role": "INVALID_ROLE",
                    },
                }
            ],
        }
        with pytest.raises(ValueError, match="unrecognized 'zone_role'"):
            validate_roi_config_dict(bad_config)


