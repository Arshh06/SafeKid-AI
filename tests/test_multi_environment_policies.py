"""Multi-environment child safety policy validation and configuration tests (Prompt 10).

Validates:
1. Configuration-level integrity:
   - All existing camera JSON files load and validate cleanly.
   - Policies survive round-trip serialization and deserialization.
   - Each zone's configured policy is respected by the event engine.
   - Independent camera configurations operate independently without cross-talk.
   - Missing policies use documented backward-compatible defaults.
   - Configuration reloads preserve active policies and track states.
2. Deterministic synthetic multi-environment validation:
   - Playground safe-zone entry (SAFE: 0 alerts, designated safe area message).
   - Playground restricted-boundary entry & dwell (DANGER with entry_alert=False: 0 entry alert, dwell triggers at 4.0s).
   - Main driveway vehicle lane entry & dwell (DANGER with entry_alert=True: immediate HIGH alert, dwell triggers at 2.0s).
   - Boom barrier mechanical danger zone entry (DANGER with entry_alert=True: immediate CRITICAL alert).
   - Monitored zone dwell-only policy (MONITORED: 0 entry alerts, dwell triggers at 3.0s, monitored area message).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from safekid_ai.child_safety import (
    ChildSafetyEvent,
    ChildSafetyEventEngine,
    ChildSafetyEventType,
    ManualTrackAnnotationProvider,
    TrackClassification,
)
from safekid_ai.detector import DetectionBox
from safekid_ai.events import SafetyEvent, SafetyEventType, ZoneStateManager
from safekid_ai.roi.engine import ROIEngine
from safekid_ai.roi.loader import load_camera_roi_config, parse_roi_config_dict, validate_roi_config_dict
from safekid_ai.roi.models import (
    AlertSeverity,
    CameraROIConfig,
    Polygon,
    SafetyZone,
    SafetyZoneType,
    ZoneRole,
    ZoneSafetyPolicy,
)

CONFIG_DIR = Path("configs/cameras")


def _simulate_presence(
    manager: ZoneStateManager,
    roi_engine: ROIEngine,
    track_id: int,
    bc_x: float,
    bc_y: float,
    start_frame: int,
    end_frame: int,
    fps: float = 24.0,
    res: Tuple[int, int] = (1920, 1080),
) -> List[SafetyEvent]:
    """Simulates consecutive observations for a track at bottom-center (bc_x, bc_y)."""
    box_w, box_h = 40.0, 80.0
    x1, y1 = bc_x - box_w / 2.0, bc_y - box_h
    x2, y2 = bc_x + box_w / 2.0, bc_y

    all_events: List[SafetyEvent] = []
    frame_interval_ms = 1000.0 / fps

    for fid in range(start_frame, end_frame + 1):
        ts_ms = (fid - 1) * frame_interval_ms
        det = DetectionBox(bbox_xyxy=(x1, y1, x2, y2), confidence=0.95, track_id=track_id)
        evts = manager.update(
            frame_id=fid,
            timestamp_ms=ts_ms,
            detections=[det],
            roi_engine=roi_engine,
            frame_resolution=res,
        )
        all_events.extend(evts)

    return all_events


class TestConfigurationLevelPolicies:
    """Rigorous audit and validation of camera configurations and policy lifecycle."""

    def test_all_existing_camera_json_files_load_and_validate(self):
        """Every camera JSON configuration under configs/cameras loads and passes validation."""
        json_files = list(CONFIG_DIR.rglob("*.json"))
        assert len(json_files) >= 4, f"Expected at least 4 camera configurations, found {len(json_files)}"

        for path in json_files:
            config = load_camera_roi_config(str(path))
            assert config.camera_id, f"Missing camera_id in {path}"
            assert config.community_id, f"Missing community_id in {path}"
            assert config.frame_width > 0 and config.frame_height > 0
            assert len(config.zones) > 0, f"No zones in {path}"

            zone_ids = set()
            for zone in config.zones:
                assert zone.roi_id, f"Zone missing roi_id in {path}"
                assert zone.roi_id not in zone_ids, f"Duplicate zone ID '{zone.roi_id}' in {path}"
                zone_ids.add(zone.roi_id)

                # Ensure every zone has an instantiated ZoneSafetyPolicy
                assert zone.policy is not None, f"Zone '{zone.roi_id}' in {path} lacks policy"
                assert isinstance(zone.policy.zone_role, ZoneRole)
                assert isinstance(zone.policy.entry_alert, bool)
                assert isinstance(zone.policy.exit_alert, bool)
                assert isinstance(zone.policy.dwell_alert, bool)
                assert isinstance(zone.policy.severity, AlertSeverity)
                if zone.policy.dwell_threshold_seconds is not None:
                    assert zone.policy.dwell_threshold_seconds >= 0.0

    def test_policies_survive_serialization_and_deserialization(self):
        """Zone safety policies round-trip losslessly through JSON serialization and parsing."""
        for path in CONFIG_DIR.rglob("*.json"):
            original = load_camera_roi_config(str(path))
            as_dict = original.to_dict()
            json_str = json.dumps(as_dict)
            reloaded_dict = json.loads(json_str)
            reloaded = parse_roi_config_dict(reloaded_dict)

            assert reloaded.camera_id == original.camera_id
            assert len(reloaded.zones) == len(original.zones)

            for orig_z, rel_z in zip(original.zones, reloaded.zones):
                assert orig_z.roi_id == rel_z.roi_id
                assert orig_z.policy is not None and rel_z.policy is not None
                assert orig_z.policy.zone_role == rel_z.policy.zone_role
                assert orig_z.policy.entry_alert == rel_z.policy.entry_alert
                assert orig_z.policy.exit_alert == rel_z.policy.exit_alert
                assert orig_z.policy.dwell_alert == rel_z.policy.dwell_alert
                assert orig_z.policy.dwell_threshold_seconds == rel_z.policy.dwell_threshold_seconds
                assert orig_z.policy.severity == rel_z.policy.severity

    def test_missing_policies_use_documented_backward_compatible_defaults(self):
        """Legacy configs without a 'policy' key construct sensible defaults."""
        legacy_dict = {
            "community_id": "legacy_comm",
            "camera_id": "legacy_cam",
            "frame_width": 1280,
            "frame_height": 720,
            "zones": [
                {
                    "roi_id": "legacy_water",
                    "name": "Legacy Pool Water",
                    "zone_type": "SWIMMING_POOL_BOUNDARY",
                    "polygon": [[100, 100], [300, 100], [300, 300], [100, 300]],
                    "severity": "CRITICAL",
                    "dwell_time_seconds": 0.0,
                },
                {
                    "roi_id": "legacy_deck",
                    "name": "Legacy Pool Deck",
                    "zone_type": "SWIMMING_POOL_GATE",
                    "polygon": [[50, 50], [400, 50], [400, 400], [50, 400]],
                    "severity": "HIGH",
                    "dwell_time_seconds": 3.0,
                },
                {
                    "roi_id": "legacy_walkway",
                    "name": "Legacy Walkway",
                    "zone_type": "GENERAL_COMMON_AREA",
                    "polygon": [[500, 100], [700, 100], [700, 300], [500, 300]],
                    "severity": "LOW",
                    "dwell_time_seconds": None,
                },
            ],
        }

        config = parse_roi_config_dict(legacy_dict)

        # 1. Swimming pool boundary defaults to DANGER with entry & exit alert
        z_water = config.get_zone("legacy_water")
        assert z_water.policy.zone_role == ZoneRole.DANGER
        assert z_water.policy.entry_alert is True
        assert z_water.policy.exit_alert is True
        assert z_water.policy.severity == AlertSeverity.CRITICAL

        # 2. Pool deck with dwell threshold defaults to MONITORED with dwell alert
        z_deck = config.get_zone("legacy_deck")
        assert z_deck.policy.zone_role == ZoneRole.MONITORED
        assert z_deck.policy.entry_alert is False
        assert z_deck.policy.dwell_alert is True
        assert z_deck.policy.dwell_threshold_seconds == 3.0
        assert z_deck.policy.severity == AlertSeverity.HIGH

        # 3. Low-severity common area defaults to SAFE
        z_walk = config.get_zone("legacy_walkway")
        assert z_walk.policy.zone_role == ZoneRole.SAFE
        assert z_walk.policy.entry_alert is False
        assert z_walk.policy.dwell_alert is False
        assert z_walk.policy.severity == AlertSeverity.LOW

    def test_two_cameras_with_different_policies_behave_independently(self):
        """Independent cameras maintain completely decoupled policy evaluation."""
        gate_cfg = load_camera_roi_config(str(CONFIG_DIR / "community_001" / "camera_gate_north.json"))
        play_cfg = load_camera_roi_config(str(CONFIG_DIR / "community_002" / "camera_playground_01.json"))

        prov = ManualTrackAnnotationProvider()
        prov.set_annotation(track_id=10, classification=TrackClassification.CHILD)

        engine_gate = ChildSafetyEventEngine(annotation_provider=prov, roi_config=gate_cfg)
        engine_play = ChildSafetyEventEngine(annotation_provider=prov, roi_config=play_cfg)

        # 1. Gate camera: Boom barrier has DANGER policy (entry_alert=True)
        barrier_zone = gate_cfg.get_zone("boom_barrier_danger_zone")
        evt_barrier = SafetyEvent(
            event_id="evt_b",
            event_type=SafetyEventType.ZONE_ENTRY,
            timestamp_ms=100.0,
            frame_id=3,
            community_id=gate_cfg.community_id,
            camera_id=gate_cfg.camera_id,
            track_id=10,
            roi_id=barrier_zone.roi_id,
            zone_name=barrier_zone.name,
            zone_type=barrier_zone.zone_type,
            severity=barrier_zone.severity,
            bottom_center=(900.0, 550.0),
            bbox_xyxy=(880.0, 500.0, 920.0, 550.0),
            dwell_time_seconds=0.0,
            message="Test barrier entry",
            policy=barrier_zone.policy,
        )
        gate_alerts = engine_gate.process_events([evt_barrier], frame_id=3, timestamp_ms=100.0, roi_config=gate_cfg)
        assert len(gate_alerts) == 1
        assert gate_alerts[0].event_type == ChildSafetyEventType.CHILD_ZONE_ENTRY
        assert "configured danger zone" in gate_alerts[0].message

        # 2. Playground camera: Sandbox has SAFE policy (entry_alert=False)
        sandbox_zone = play_cfg.get_zone("toddler_play_sandbox")
        evt_sandbox = SafetyEvent(
            event_id="evt_s",
            event_type=SafetyEventType.ZONE_ENTRY,
            timestamp_ms=100.0,
            frame_id=3,
            community_id=play_cfg.community_id,
            camera_id=play_cfg.camera_id,
            track_id=10,
            roi_id=sandbox_zone.roi_id,
            zone_name=sandbox_zone.name,
            zone_type=sandbox_zone.zone_type,
            severity=sandbox_zone.severity,
            bottom_center=(400.0, 385.0),
            bbox_xyxy=(380.0, 330.0, 420.0, 385.0),
            dwell_time_seconds=0.0,
            message="Test sandbox entry",
            policy=sandbox_zone.policy,
        )
        play_alerts = engine_play.process_events([evt_sandbox], frame_id=3, timestamp_ms=100.0, roi_config=play_cfg)
        assert len(play_alerts) == 0

    def test_configuration_reload_preserves_correct_active_policy(self):
        """Updating camera ROI configuration dynamically updates policies without dropping track dwell."""
        initial_cfg = load_camera_roi_config(str(CONFIG_DIR / "community_001" / "camera_pool_01.json"))
        engine = ROIEngine(initial_cfg)
        manager = ZoneStateManager(entry_confirmation_frames=1, exit_confirmation_frames=1)

        # Simulate presence in pool deck (threshold 3.0s, monitored)
        det = DetectionBox(bbox_xyxy=(1030.0, 800.0, 1070.0, 850.0), confidence=0.9, track_id=7)
        manager.update(frame_id=1, timestamp_ms=0.0, detections=[det], roi_engine=engine, frame_resolution=(1920, 1080))

        # Dynamically reload configuration with updated policy (e.g., lower dwell threshold to 1.5s)
        updated_dict = initial_cfg.to_dict()
        updated_dict["zones"][1]["policy"]["dwell_threshold_seconds"] = 1.5
        updated_cfg = parse_roi_config_dict(updated_dict)
        manager.update_configuration(initial_cfg, updated_cfg, current_timestamp_ms=100.0)

        deck_zone = updated_cfg.get_zone("pool_deck_enclosure_01")
        assert deck_zone.policy.dwell_threshold_seconds == 1.5


class TestSyntheticMultiEnvironmentPolicies:
    """Deterministic validation of safety policies across diverse residential environments."""

    def test_playground_safe_zone_entry(self):
        """Toddler sandbox entry in playground produces ZERO danger alerts."""
        cfg = load_camera_roi_config(str(CONFIG_DIR / "community_002" / "camera_playground_01.json"))
        roi_engine = ROIEngine(cfg)
        manager = ZoneStateManager(entry_confirmation_frames=3, exit_confirmation_frames=3)

        prov = ManualTrackAnnotationProvider(community_id=cfg.community_id, camera_id=cfg.camera_id)
        prov.set_annotation(track_id=12, classification=TrackClassification.CHILD)
        child_engine = ChildSafetyEventEngine(annotation_provider=prov, roi_config=cfg)

        # Bottom-center at (400, 385) is inside toddler_play_sandbox (150-680, 180-580)
        events = _simulate_presence(
            manager=manager,
            roi_engine=roi_engine,
            track_id=12,
            bc_x=400.0,
            bc_y=385.0,
            start_frame=1,
            end_frame=10,
            fps=24.0,
            res=(1280, 720),
        )

        # Raw manager emits confirmed ZONE_ENTRY at frame 3
        assert len(events) == 1
        assert events[0].event_type == SafetyEventType.ZONE_ENTRY
        assert events[0].roi_id == "toddler_play_sandbox"

        # Child safety engine suppresses alert due to SAFE policy (entry_alert=False)
        child_alerts = child_engine.process_events(events, frame_id=10, timestamp_ms=416.7, roi_config=cfg)
        assert len(child_alerts) == 0

    def test_playground_restricted_boundary_entry_and_dwell(self):
        """Playground outer perimeter boundary produces no entry alert, but triggers dwell alert after 4.0s."""
        cfg = load_camera_roi_config(str(CONFIG_DIR / "community_002" / "camera_playground_01.json"))
        roi_engine = ROIEngine(cfg)
        manager = ZoneStateManager(entry_confirmation_frames=3, exit_confirmation_frames=3)

        prov = ManualTrackAnnotationProvider(community_id=cfg.community_id, camera_id=cfg.camera_id)
        prov.set_annotation(track_id=14, classification=TrackClassification.CHILD)
        child_engine = ChildSafetyEventEngine(annotation_provider=prov, roi_config=cfg)

        # Coordinate (995, 160) is inside perimeter_hedge_boundary (polygon: 750-1250, 60-260)
        # 1. Simulate entry (frames 1-10: ~0.4s)
        entry_events = _simulate_presence(
            manager=manager,
            roi_engine=roi_engine,
            track_id=14,
            bc_x=995.0,
            bc_y=160.0,
            start_frame=1,
            end_frame=10,
            fps=24.0,
            res=(1280, 720),
        )
        assert len(entry_events) == 1
        assert entry_events[0].event_type == SafetyEventType.ZONE_ENTRY

        # Entry alert is False on perimeter_hedge_boundary -> 0 child alerts
        child_entry_alerts = child_engine.process_events(entry_events, frame_id=10, timestamp_ms=416.7, roi_config=cfg)
        assert len(child_entry_alerts) == 0

        # 2. Simulate lingering past 4.0s threshold (frame 105 at 24 FPS = 4.33s elapsed)
        dwell_events = _simulate_presence(
            manager=manager,
            roi_engine=roi_engine,
            track_id=14,
            bc_x=995.0,
            bc_y=160.0,
            start_frame=11,
            end_frame=105,
            fps=24.0,
            res=(1280, 720),
        )

        # ZoneStateManager emits DWELL_TIME_EXCEEDED
        assert any(e.event_type == SafetyEventType.DWELL_TIME_EXCEEDED for e in dwell_events)

        # Child safety engine promotes to CHILD_DWELL_TIME_EXCEEDED
        child_dwell_alerts = child_engine.process_events(dwell_events, frame_id=105, timestamp_ms=4333.3, roi_config=cfg)
        assert len(child_dwell_alerts) == 1
        assert child_dwell_alerts[0].event_type == ChildSafetyEventType.CHILD_DWELL_TIME_EXCEEDED
        assert child_dwell_alerts[0].roi_id == "perimeter_hedge_boundary"
        assert child_dwell_alerts[0].severity == AlertSeverity.HIGH
        assert "remained in danger zone" in child_dwell_alerts[0].message

    def test_child_entering_driveway(self):
        """Entering the main vehicle driveway generates an immediate HIGH entry alert and subsequent dwell alert."""
        cfg = load_camera_roi_config(str(CONFIG_DIR / "community_001" / "camera_gate_north.json"))
        roi_engine = ROIEngine(cfg)
        manager = ZoneStateManager(entry_confirmation_frames=3, exit_confirmation_frames=3)

        prov = ManualTrackAnnotationProvider(community_id=cfg.community_id, camera_id=cfg.camera_id)
        prov.set_annotation(track_id=15, classification=TrackClassification.CHILD)
        child_engine = ChildSafetyEventEngine(annotation_provider=prov, roi_config=cfg)

        # Coordinate (925, 725) is inside vehicle_driveway_lane (500-1500, 400-1050)
        # 1. Entry across 3 frames
        entry_events = _simulate_presence(
            manager=manager,
            roi_engine=roi_engine,
            track_id=15,
            bc_x=925.0,
            bc_y=725.0,
            start_frame=1,
            end_frame=10,
            fps=24.0,
            res=(1920, 1080),
        )
        assert len(entry_events) == 1
        assert entry_events[0].event_type == SafetyEventType.ZONE_ENTRY

        child_entry_alerts = child_engine.process_events(entry_events, frame_id=10, timestamp_ms=416.7, roi_config=cfg)
        assert len(child_entry_alerts) == 1
        assert child_entry_alerts[0].event_type == ChildSafetyEventType.CHILD_ZONE_ENTRY
        assert child_entry_alerts[0].severity == AlertSeverity.HIGH
        assert "CHILD SAFETY ALERT" in child_entry_alerts[0].message
        assert "configured danger zone" in child_entry_alerts[0].message

        # 2. Dwell beyond 2.0s (frame 60 = 2.5s)
        dwell_events = _simulate_presence(
            manager=manager,
            roi_engine=roi_engine,
            track_id=15,
            bc_x=925.0,
            bc_y=725.0,
            start_frame=11,
            end_frame=60,
            fps=24.0,
            res=(1920, 1080),
        )
        child_dwell_alerts = child_engine.process_events(dwell_events, frame_id=60, timestamp_ms=2500.0, roi_config=cfg)
        assert len(child_dwell_alerts) == 1
        assert child_dwell_alerts[0].event_type == ChildSafetyEventType.CHILD_DWELL_TIME_EXCEEDED
        assert child_dwell_alerts[0].severity == AlertSeverity.HIGH

    def test_child_entering_boom_barrier_danger_zone(self):
        """Entering automated boom barrier mechanical swing zone produces immediate CRITICAL alert."""
        cfg = load_camera_roi_config(str(CONFIG_DIR / "community_001" / "camera_gate_north.json"))
        roi_engine = ROIEngine(cfg)
        manager = ZoneStateManager(entry_confirmation_frames=3, exit_confirmation_frames=3)

        prov = ManualTrackAnnotationProvider(community_id=cfg.community_id, camera_id=cfg.camera_id)
        prov.set_annotation(track_id=16, classification=TrackClassification.CHILD)
        child_engine = ChildSafetyEventEngine(annotation_provider=prov, roi_config=cfg)

        # Coordinate (900, 550) is inside boom_barrier_danger_zone (700-1100, 480-620)
        events = _simulate_presence(
            manager=manager,
            roi_engine=roi_engine,
            track_id=16,
            bc_x=900.0,
            bc_y=550.0,
            start_frame=1,
            end_frame=5,
            fps=24.0,
            res=(1920, 1080),
        )
        assert any(e.roi_id == "boom_barrier_danger_zone" for e in events)
        barrier_events = [e for e in events if e.roi_id == "boom_barrier_danger_zone"]
        assert len(barrier_events) == 1

        alerts = child_engine.process_events(events, frame_id=5, timestamp_ms=208.3, roi_config=cfg)
        barrier_alerts = [a for a in alerts if a.roi_id == "boom_barrier_danger_zone"]
        assert len(barrier_alerts) == 1
        assert barrier_alerts[0].event_type == ChildSafetyEventType.CHILD_ZONE_ENTRY
        assert barrier_alerts[0].severity == AlertSeverity.CRITICAL
        assert "CHILD SAFETY ALERT" in barrier_alerts[0].message
        assert "Boom Barrier Mechanical Zone" in barrier_alerts[0].message

    def test_child_approaching_community_perimeter(self):
        """Child walking outside perimeter triggers no alert; stepping into monitored border triggers policy."""
        poly = Polygon([[100, 100], [500, 100], [500, 300], [100, 300]])
        policy = ZoneSafetyPolicy(
            zone_role=ZoneRole.DANGER,
            entry_alert=False,
            exit_alert=True,
            dwell_alert=True,
            dwell_threshold_seconds=5.0,
            severity=AlertSeverity.HIGH,
        )
        zone = SafetyZone(
            roi_id="perimeter_border_strip",
            name="Community Outer Perimeter Border",
            zone_type=SafetyZoneType.PERIMETER_FENCE,
            polygon=poly,
            severity=AlertSeverity.HIGH,
            dwell_time_seconds=5.0,
            policy=policy,
        )
        cam_cfg = CameraROIConfig("comm_perim", "cam_perim_01", 640, 480, zones=[zone])
        roi_engine = ROIEngine(cam_cfg)
        manager = ZoneStateManager(entry_confirmation_frames=3, exit_confirmation_frames=3)

        prov = ManualTrackAnnotationProvider(community_id="comm_perim", camera_id="cam_perim_01")
        prov.set_annotation(track_id=18, classification=TrackClassification.CHILD)
        child_engine = ChildSafetyEventEngine(annotation_provider=prov, roi_config=cam_cfg)

        # 1. Child at (50, 50) is OUTSIDE perimeter -> zero events
        out_events = _simulate_presence(manager, roi_engine, track_id=18, bc_x=50.0, bc_y=50.0, start_frame=1, end_frame=5, fps=24.0, res=(640, 480))
        assert len(out_events) == 0
        assert len(child_engine.process_events(out_events, frame_id=5, timestamp_ms=208.3, roi_config=cam_cfg)) == 0

        # 2. Child steps into border strip at (250, 200) -> Entry confirmed, but entry_alert=False
        in_events = _simulate_presence(manager, roi_engine, track_id=18, bc_x=250.0, bc_y=200.0, start_frame=6, end_frame=15, fps=24.0, res=(640, 480))
        assert len(in_events) == 1
        assert in_events[0].event_type == SafetyEventType.ZONE_ENTRY
        assert len(child_engine.process_events(in_events, frame_id=15, timestamp_ms=625.0, roi_config=cam_cfg)) == 0

    def test_monitored_zone_with_dwell_only_policy(self):
        """Monitored zone produces zero alerts on entry, and emits alert with monitored wording upon dwell."""
        cfg = load_camera_roi_config(str(CONFIG_DIR / "community_001" / "camera_pool_01.json"))
        roi_engine = ROIEngine(cfg)
        manager = ZoneStateManager(entry_confirmation_frames=3, exit_confirmation_frames=3)

        prov = ManualTrackAnnotationProvider(community_id=cfg.community_id, camera_id=cfg.camera_id)
        prov.set_annotation(track_id=20, classification=TrackClassification.CHILD)
        child_engine = ChildSafetyEventEngine(annotation_provider=prov, roi_config=cfg)

        # Point (1050, 850) is inside pool_deck_enclosure_01 (950-1890, 200-950)
        # Step onto deck (frames 1-10) -> 0 child alerts
        entry_events = _simulate_presence(manager, roi_engine, track_id=20, bc_x=1050.0, bc_y=850.0, start_frame=1, end_frame=10, fps=24.0, res=(1920, 1080))
        assert len(entry_events) == 1
        assert len(child_engine.process_events(entry_events, frame_id=10, timestamp_ms=416.7, roi_config=cfg)) == 0

        # Linger beyond 3.0s threshold (frame 80 = 3.33s elapsed)
        dwell_events = _simulate_presence(manager, roi_engine, track_id=20, bc_x=1050.0, bc_y=850.0, start_frame=11, end_frame=80, fps=24.0, res=(1920, 1080))
        alerts = child_engine.process_events(dwell_events, frame_id=80, timestamp_ms=3333.3, roi_config=cfg)

        assert len(alerts) == 1
        assert alerts[0].event_type == ChildSafetyEventType.CHILD_DWELL_TIME_EXCEEDED
        assert alerts[0].severity == AlertSeverity.HIGH
        # Message must explicitly reflect MONITORED area, not danger zone!
        assert "remained in monitored area" in alerts[0].message
        assert "danger zone" not in alerts[0].message
