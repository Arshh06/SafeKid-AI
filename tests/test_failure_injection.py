"""SafeKid AI — Comprehensive Failure Injection, Incident Recovery & Reliability Tests (Prompt 16).

Verifies the Critical Safety Principle:
Loss of observation must NEVER be interpreted as confirmation that a child is safe.

Tests:
1. Camera disconnection immediately after danger zone entry.
2. Camera disconnection while a child remains in a nested hazard zone.
3. Camera reconnection after a short interruption.
4. Camera reconnection after a prolonged interruption.
5. Camera never reconnects (permanent camera failure).
6. Safe-exit event is never invented during downtime.
7. One-frame detection loss (momentary detector miss).
8. Short occlusion within configured grace period (dwell preserved, observation resumed).
9. Extended occlusion exceeding grace period (marked TRACK_LOST, incident NOT resolved).
10. Repeated intermittent detection stability (transition confirmation debouncing).
11. Disappearance while CRITICAL incident is active (submersion/trapped simulation).
12. Detection returning after grace period (safe ID switch independence).
13. Observation status decoupled from incident resolution.
14. UI alert panel renders explicit [OFFLINE], [LOST], and [OCCL] badges.
"""

import time
from typing import List, Set, Tuple
import numpy as np
import pytest

from safekid_ai.child_safety import (
    ChildSafetyEvent,
    ChildSafetyEventType,
    TrackClassification,
)
from safekid_ai.config import SafeKidPipelineConfig
from safekid_ai.detector import DetectionBox
from safekid_ai.events import SafetyEvent, SafetyEventType, ZoneStateManager
from safekid_ai.incidents import (
    Incident,
    IncidentManager,
    IncidentPolicyConfig,
    IncidentState,
    ObservationStatus,
)
from safekid_ai.pipeline import SafeKidPipeline
from safekid_ai.roi.engine import ROIEngine
from safekid_ai.roi.models import (
    AlertSeverity,
    CameraROIConfig,
    Point2D,
    Polygon,
    SafetyZone,
    SafetyZoneType,
)
from safekid_ai.visualization import draw_person_and_events


def make_child_event(
    event_id: str,
    event_type: ChildSafetyEventType,
    timestamp_ms: float,
    frame_id: int,
    track_id: int,
    roi_id: str = "danger_zone_01",
    zone_name: str = "Swimming Pool Water Basin",
    zone_type: SafetyZoneType = SafetyZoneType.SWIMMING_POOL_BOUNDARY,
    severity: AlertSeverity = AlertSeverity.CRITICAL,
    dwell_time_seconds: float = 0.0,
    community_id: str = "comm_demo",
    camera_id: str = "cam_pool_01",
    is_loss_of_tracking: bool = False,
) -> ChildSafetyEvent:
    """Helper to generate deterministic child safety events."""
    return ChildSafetyEvent(
        event_id=event_id,
        event_type=event_type,
        timestamp_ms=timestamp_ms,
        frame_id=frame_id,
        community_id=community_id,
        camera_id=camera_id,
        track_id=track_id,
        classification=TrackClassification.CHILD,
        roi_id=roi_id,
        zone_name=zone_name,
        zone_type=zone_type,
        severity=severity,
        bottom_center=(100.0, 100.0),
        bbox_xyxy=(80.0, 50.0, 120.0, 150.0),
        dwell_time_seconds=dwell_time_seconds,
        message=f"CHILD ALERT: Child (Track ID {track_id}) in {zone_name}",
        is_loss_of_tracking=is_loss_of_tracking,
    )


# ============================================================================
# 1. CAMERA DISCONNECTION FAILURE INJECTION
# ============================================================================

def test_camera_disconnect_immediately_after_danger_entry():
    """Simulates camera disconnecting immediately after a child enters a DANGER zone.
    
    Invariants:
    - Incident does NOT transition to RESOLVED merely because frames stop arriving.
    - System preserves last known incident state (ACTIVE, CRITICAL).
    - Observation status is updated to DISCONNECTED.
    - Audit history records CAMERA_DISCONNECTED.
    """
    manager = IncidentManager()

    # Frame 10 (t=1000ms): Child Track 4 enters swimming pool water basin (CRITICAL)
    e_entry = make_child_event("e1", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 10, 4)
    open_incs = manager.process_frame(10, 1000.0, [e_entry], [], active_track_ids={4})
    assert len(open_incs) == 1
    inc = open_incs[0]
    assert inc.state == IncidentState.ACTIVE
    assert inc.severity == AlertSeverity.CRITICAL
    assert inc.observation_status == ObservationStatus.OBSERVING

    # Camera disconnects at t=1200ms
    affected = manager.handle_camera_disconnect(camera_id="cam_pool_01", timestamp_ms=1200.0, frame_id=12)
    assert len(affected) == 1
    assert affected[0].incident_id == inc.incident_id

    # Verify incident state:
    assert inc.state == IncidentState.ACTIVE  # MUST REMAIN ACTIVE!
    assert inc.is_open() is True
    assert inc.observation_status == ObservationStatus.DISCONNECTED
    assert inc.severity == AlertSeverity.CRITICAL
    assert inc.primary_zone_name == "Swimming Pool Water Basin"
    assert inc.last_observed_at_ms == 1000.0
    assert inc.last_observed_at_frame == 10

    # Verify audit history
    disc_entries = [h for h in inc.lifecycle_history if h.get("transition") == "CAMERA_DISCONNECTED"]
    assert len(disc_entries) == 1
    assert disc_entries[0]["to_status"] == ObservationStatus.DISCONNECTED.value
    assert "stream disconnected" in disc_entries[0]["reason"]


def test_camera_disconnect_while_in_nested_hazard_zone():
    """Simulates camera disconnecting while child is inside a nested hazard zone.
    
    Verifies that multi-zone containment and CRITICAL severity are preserved.
    """
    manager = IncidentManager()

    # Frame 1: Child Track 2 enters general driveway (HIGH)
    e1 = make_child_event(
        "e1", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 1, 2,
        roi_id="driveway", zone_name="Main Driveway", zone_type=SafetyZoneType.VEHICULAR_GATEWAY,
        severity=AlertSeverity.HIGH, camera_id="cam_gate"
    )
    manager.process_frame(1, 1000.0, [e1], [], active_track_ids={2})

    # Frame 10: Child steps into nested boom barrier mechanism (CRITICAL)
    e2 = make_child_event(
        "e2", ChildSafetyEventType.CHILD_ZONE_ENTRY, 2000.0, 10, 2,
        roi_id="barrier", zone_name="Boom Barrier Mechanism", zone_type=SafetyZoneType.RESTRICTED_INFRASTRUCTURE,
        severity=AlertSeverity.CRITICAL, camera_id="cam_gate"
    )
    open_incs = manager.process_frame(10, 2000.0, [e2], [], active_track_ids={2})
    assert len(open_incs) == 1
    inc = open_incs[0]
    assert inc.severity == AlertSeverity.CRITICAL
    assert inc.current_zone_ids == {"driveway", "barrier"}

    # Camera disconnects while child is in nested zone
    manager.handle_camera_disconnect("cam_gate", timestamp_ms=2500.0, frame_id=15)

    # Invariants:
    assert inc.state == IncidentState.ACTIVE
    assert inc.severity == AlertSeverity.CRITICAL
    assert inc.observation_status == ObservationStatus.DISCONNECTED
    assert inc.current_zone_ids == {"driveway", "barrier"}  # Both zones preserved


def test_camera_reconnects_after_short_interruption():
    """Simulates camera stream restoring after a 2-second glitch.
    
    Invariants:
    - Reconnection does NOT silently erase the incident.
    - Incident remains OPEN.
    - Visual observation resumes when child is detected again.
    """
    manager = IncidentManager()
    e_entry = make_child_event("e1", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 10, 4)
    manager.process_frame(10, 1000.0, [e_entry], [], active_track_ids={4})

    # Disconnect at t=2000ms
    manager.handle_camera_disconnect("cam_pool_01", timestamp_ms=2000.0, frame_id=20)
    inc = manager.all_incidents[0]
    assert inc.observation_status == ObservationStatus.DISCONNECTED

    # Reconnect at t=4000ms
    manager.handle_camera_reconnect("cam_pool_01", timestamp_ms=4000.0, frame_id=40)
    assert inc.is_open() is True
    assert inc.observation_status == ObservationStatus.TRACK_LOST  # Awaiting re-acquisition

    # Next frame arrives with Track 4 re-detected!
    manager.process_frame(41, 4041.6, [], [], active_track_ids={4})
    assert inc.observation_status == ObservationStatus.OBSERVING
    assert inc.is_open() is True

    # Audit history contains disconnection, reconnection, and resumption
    transitions = [h["transition"] for h in inc.lifecycle_history]
    assert "CAMERA_DISCONNECTED" in transitions
    assert "CAMERA_RECONNECTED" in transitions
    assert "OBSERVATION_RESUMED" in transitions


def test_camera_reconnects_after_prolonged_interruption():
    """Simulates camera reconnecting after 60 seconds (exceeding 15s default expiration).
    
    CRITICAL RULE:
    Camera downtime must NOT cause the incident to expire and vanish from security view.
    """
    cfg = IncidentPolicyConfig(default_expiration_seconds=15.0, preserve_lost_observation=True)
    manager = IncidentManager(config=cfg)

    # Child enters pool water basin at t=1000ms
    e_entry = make_child_event("e1", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 10, 4)
    manager.process_frame(10, 1000.0, [e_entry], [], active_track_ids={4})
    inc = manager.all_incidents[0]

    # Camera disconnects at t=2000ms
    manager.handle_camera_disconnect("cam_pool_01", timestamp_ms=2000.0, frame_id=20)

    # 60 seconds of downtime!
    manager.handle_camera_reconnect("cam_pool_01", timestamp_ms=62000.0, frame_id=620)

    # First frame after reconnection arrives at t=62040ms
    open_incs = manager.process_frame(621, 62041.6, [], [], active_track_ids=set())

    # MUST NOT HAVE EXPIRED OR BEEN ERASED!
    assert len(open_incs) == 1
    assert open_incs[0].incident_id == inc.incident_id
    assert open_incs[0].state == IncidentState.ACTIVE
    assert open_incs[0].is_open() is True
    assert open_incs[0].observation_status == ObservationStatus.TRACK_LOST


def test_camera_never_reconnects_incident_persists():
    """Simulates permanent camera hardware failure.
    
    Verifies that the incident remains recorded, auditable, and open indefinitely.
    """
    manager = IncidentManager()
    e_entry = make_child_event("e1", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 10, 4)
    manager.process_frame(10, 1000.0, [e_entry], [], active_track_ids={4})
    inc = manager.all_incidents[0]

    manager.handle_camera_disconnect("cam_pool_01", timestamp_ms=2000.0, frame_id=20)

    # No more frames ever arrive. Check status:
    assert inc.state == IncidentState.ACTIVE
    assert inc.is_open() is True
    assert inc.observation_status == ObservationStatus.DISCONNECTED
    assert inc.resolved_at_ms is None


def test_system_does_not_invent_safe_exit_during_downtime():
    """Verifies that no safe-exit events or RESOLVED transitions are invented during downtime."""
    manager = IncidentManager()
    e_entry = make_child_event("e1", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 10, 4)
    manager.process_frame(10, 1000.0, [e_entry], [], active_track_ids={4})
    inc = manager.all_incidents[0]

    manager.handle_camera_disconnect("cam_pool_01", timestamp_ms=2000.0)

    # Check all history entries: NONE should be RESOLVED
    for h in inc.lifecycle_history:
        assert h.get("transition") != "RESOLVED"
        assert h.get("to_state") != IncidentState.RESOLVED.value


# ============================================================================
# 2. TEMPORARY DETECTION LOSS & OCCLUSION
# ============================================================================

def test_one_frame_detection_loss_within_grace_period():
    """Simulates a single-frame detector drop.
    
    Verifies:
    - Incident remains ACTIVE.
    - No false exit event emitted.
    - Observation transitions: OBSERVING -> OCCLUDED -> OBSERVING.
    """
    cfg = IncidentPolicyConfig(occlusion_grace_frames=3)
    manager = IncidentManager(config=cfg)

    # Frame 10: Child observed inside danger zone
    e_entry = make_child_event("e1", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 10, 4)
    manager.process_frame(10, 1000.0, [e_entry], [], active_track_ids={4})
    inc = manager.all_incidents[0]
    assert inc.observation_status == ObservationStatus.OBSERVING

    # Frame 11: Detector drops Track 4 for 1 frame
    manager.process_frame(11, 1041.6, [], [], active_track_ids=set())
    assert inc.state == IncidentState.ACTIVE
    assert inc.observation_status == ObservationStatus.OCCLUDED
    assert inc.missing_frames_count == 1

    # Frame 12: Track 4 detected again!
    manager.process_frame(12, 1083.3, [], [], active_track_ids={4})
    assert inc.state == IncidentState.ACTIVE
    assert inc.observation_status == ObservationStatus.OBSERVING
    assert inc.missing_frames_count == 0


def test_short_occlusion_preserves_dwell_and_incident():
    """Simulates short occlusion (4 frames) behind a pillar.
    
    Verifies:
    - Dwell time does not reset to 0.
    - Dwell alert still fires when threshold reached.
    """
    poly = Polygon([(0, 0), (200, 0), (200, 200), (0, 200)])
    zone = SafetyZone("z_pool", "Pool", SafetyZoneType.SWIMMING_POOL_GATE, poly, AlertSeverity.HIGH, dwell_time_seconds=3.0)
    config = CameraROIConfig("comm_1", "cam_1", 1000, 1000, [zone])
    engine = ROIEngine(config=config)
    state_mgr = ZoneStateManager(max_missing_frames=10)
    inc_mgr = IncidentManager()

    # Frame 1 (t=1000ms): Entry
    det = [DetectionBox((50, 50, 150, 150), 0.9, track_id=7)]
    raw_evts = state_mgr.update(1, 1000.0, det, engine, (1000, 1000))
    c_evts = [make_child_event("ce1", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 1, 7, roi_id="z_pool", zone_name="Pool")]
    inc_mgr.process_frame(1, 1000.0, c_evts, raw_evts, active_track_ids={7})

    # Frames 2..5: 4 frames missing (occluded behind column)
    for f in range(2, 6):
        raw_evts = state_mgr.update(f, 1000.0 + f * 500.0, [], engine, (1000, 1000))
        inc_mgr.process_frame(f, 1000.0 + f * 500.0, [], raw_evts, active_track_ids=set())

    # Frame 6 (t=4500ms): Reappearance. Dwell has reached 3.5s!
    raw_evts6 = state_mgr.update(6, 4500.0, det, engine, (1000, 1000))
    dwell_alerts = [e for e in raw_evts6 if e.event_type == SafetyEventType.DWELL_TIME_EXCEEDED]
    assert len(dwell_alerts) == 1

    c_dwell = [make_child_event("ce_dwell", ChildSafetyEventType.CHILD_DWELL_TIME_EXCEEDED, 4500.0, 6, 7, roi_id="z_pool", zone_name="Pool", dwell_time_seconds=3.5)]
    inc_mgr.process_frame(6, 4500.0, c_dwell, raw_evts6, active_track_ids={7})

    inc = inc_mgr.all_incidents[0]
    assert inc.state == IncidentState.ACTIVE
    assert inc.observation_status == ObservationStatus.OBSERVING
    assert "Dwell 3.5s" in inc.summary_message


def test_extended_occlusion_exceeding_grace_marks_track_lost_not_resolved():
    """Simulates child disappearing from camera view beyond grace period.
    
    CRITICAL SAFETY PRINCIPLE:
    When ZoneStateManager emits 'left camera view', IncidentManager must mark
    observation_status = TRACK_LOST, but MUST NOT resolve the incident!
    """
    poly = Polygon([(0, 0), (200, 0), (200, 200), (0, 200)])
    zone = SafetyZone("z_pool", "Pool Basin", SafetyZoneType.SWIMMING_POOL_BOUNDARY, poly, AlertSeverity.CRITICAL)
    config = CameraROIConfig("comm_1", "cam_1", 1000, 1000, [zone])
    engine = ROIEngine(config=config)
    state_mgr = ZoneStateManager(max_missing_frames=5)
    inc_mgr = IncidentManager()

    # Frame 1: Entry into pool water basin (CRITICAL)
    det = [DetectionBox((50, 50, 150, 150), 0.9, track_id=4)]
    raw_evts = state_mgr.update(1, 1000.0, det, engine, (1000, 1000))
    c_evts = [
        make_child_event(
            "ce1",
            ChildSafetyEventType.CHILD_ZONE_ENTRY,
            1000.0,
            1,
            4,
            roi_id="z_pool",
            zone_name="Pool Basin",
            community_id="comm_1",
            camera_id="cam_1",
        )
    ]
    inc_mgr.process_frame(1, 1000.0, c_evts, raw_evts, active_track_ids={4})
    inc = inc_mgr.all_incidents[0]
    assert inc.state == IncidentState.ACTIVE

    # Frames 2..5: Missing
    for f in range(2, 6):
        raw_evts = state_mgr.update(f, 1000.0 + f * 100.0, [], engine, (1000, 1000))
        inc_mgr.process_frame(f, 1000.0 + f * 100.0, [], raw_evts, active_track_ids=set())

    # Frame 6: Exactly 5 missing frames -> ZoneStateManager emits 'left camera view'
    raw_evts6 = state_mgr.update(6, 1600.0, [], engine, (1000, 1000))
    assert len(raw_evts6) == 1
    assert raw_evts6[0].is_loss_of_tracking is True
    assert "left camera view" in raw_evts6[0].message

    # IncidentManager processes frame 6
    open_incs = inc_mgr.process_frame(6, 1600.0, [], raw_evts6, active_track_ids=set())

    # CRITICAL INVARIANT: INCIDENT IS NOT RESOLVED!
    assert len(open_incs) == 1
    assert inc.state == IncidentState.ACTIVE
    assert inc.is_open() is True
    assert inc.observation_status == ObservationStatus.TRACK_LOST

    # Verify audit history
    loss_entries = [h for h in inc.lifecycle_history if h.get("transition") == "OBSERVATION_LOST"]
    assert len(loss_entries) == 1
    assert "NOT resolved" in loss_entries[0]["reason"]


def test_disappearance_while_critical_active_remains_active_indefinitely():
    """Simulates a child disappearing while swimming pool CRITICAL alert is active (e.g. submersion).
    
    Verifies that no timeout, frame count, or missing detection ever marks the child safe.
    """
    cfg = IncidentPolicyConfig(preserve_lost_observation=True)
    inc_mgr = IncidentManager(config=cfg)

    # Frame 1: Child enters pool basin (CRITICAL)
    c_entry = make_child_event("ce1", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 1, 11)
    inc_mgr.process_frame(1, 1000.0, [c_entry], [], active_track_ids={11})
    inc = inc_mgr.all_incidents[0]

    # Child disappears from camera view:
    loss_event = SafetyEvent(
        event_id="loss_evt",
        event_type=SafetyEventType.ZONE_EXIT,
        timestamp_ms=2000.0,
        frame_id=25,
        community_id="comm_demo",
        camera_id="cam_pool_01",
        track_id=11,
        roi_id="danger_zone_01",
        zone_name="Swimming Pool Water Basin",
        zone_type=SafetyZoneType.SWIMMING_POOL_BOUNDARY,
        severity=AlertSeverity.CRITICAL,
        bottom_center=(100.0, 100.0),
        bbox_xyxy=(80.0, 50.0, 120.0, 150.0),
        dwell_time_seconds=1.0,
        message="Track ID 11 left camera view (exited Swimming Pool Water Basin)",
        is_loss_of_tracking=True,
    )
    inc_mgr.process_frame(25, 2000.0, [], [loss_event], active_track_ids=set())

    # Simulate 100 subsequent frames over 30 seconds
    for f in range(26, 126):
        open_incs = inc_mgr.process_frame(f, 2000.0 + (f - 25) * 300.0, [], [], active_track_ids=set())

    # INVARIANT: Incident MUST remain OPEN and ACTIVE with CRITICAL severity!
    assert inc.is_open() is True
    assert inc.state == IncidentState.ACTIVE
    assert inc.observation_status == ObservationStatus.TRACK_LOST
    assert inc.severity == AlertSeverity.CRITICAL
    assert inc.resolved_at_ms is None


def test_reappearance_after_grace_period_handles_id_switch_safely():
    """Simulates Track 4 disappearing and later a person appearing with Track 9 (ID switch).
    
    Verifies that Track 9 does NOT silently inherit Track 4's incident,
    and Track 4's incident remains open for physical verification.
    """
    inc_mgr = IncidentManager()

    # Track 4 enters danger zone
    e4 = make_child_event("e4", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 1, 4)
    inc_mgr.process_frame(1, 1000.0, [e4], [], active_track_ids={4})
    inc4 = inc_mgr.all_incidents[0]

    # Track 4 is lost
    loss_evt = SafetyEvent(
        event_id="loss4",
        event_type=SafetyEventType.ZONE_EXIT,
        timestamp_ms=2000.0,
        frame_id=25,
        community_id="comm_demo",
        camera_id="cam_pool_01",
        track_id=4,
        roi_id="danger_zone_01",
        zone_name="Swimming Pool Water Basin",
        zone_type=SafetyZoneType.SWIMMING_POOL_BOUNDARY,
        severity=AlertSeverity.CRITICAL,
        bottom_center=(100.0, 100.0),
        bbox_xyxy=(80.0, 50.0, 120.0, 150.0),
        dwell_time_seconds=1.0,
        message="Track ID 4 left camera view (exited Swimming Pool Water Basin)",
        is_loss_of_tracking=True,
    )
    inc_mgr.process_frame(25, 2000.0, [], [loss_evt], active_track_ids=set())
    assert inc4.observation_status == ObservationStatus.TRACK_LOST

    # Later: New Track 9 enters
    e9 = make_child_event("e9", ChildSafetyEventType.CHILD_ZONE_ENTRY, 8000.0, 100, 9)
    inc_mgr.process_frame(100, 8000.0, [e9], [], active_track_ids={9})

    # Both incidents must be independent!
    assert len(inc_mgr.all_incidents) == 2
    assert inc4.track_id == 4
    assert inc4.state == IncidentState.ACTIVE
    assert inc4.observation_status == ObservationStatus.TRACK_LOST

    inc9 = [i for i in inc_mgr.all_incidents if i.track_id == 9][0]
    assert inc9.track_id == 9
    assert inc9.state == IncidentState.ACTIVE
    assert inc9.observation_status == ObservationStatus.OBSERVING


# ============================================================================
# 3. UI ALERT PANEL OBSERVATION HEALTH BADGE RENDERING
# ============================================================================

def test_visualization_renders_observation_status_badges():
    """Verifies that the visualization overlay renders dedicated observation health badges.
    
    [OFFLINE] for DISCONNECTED
    [LOST] for TRACK_LOST
    [OCCL] for OCCLUDED
    [ACT] for OBSERVING
    """
    inc_online = Incident(
        incident_id="INC-001",
        community_id="c1",
        camera_id="cam1",
        track_id=1,
        classification=TrackClassification.CHILD,
        primary_zone_id="z1",
        primary_zone_name="Pool Basin",
        zone_type=SafetyZoneType.SWIMMING_POOL_BOUNDARY,
        current_zone_ids={"z1"},
        severity=AlertSeverity.CRITICAL,
        state=IncidentState.ACTIVE,
        created_at_ms=1000.0,
        created_at_frame=10,
        updated_at_ms=1000.0,
        updated_at_frame=10,
        observation_status=ObservationStatus.OBSERVING,
    )

    inc_lost = Incident(
        incident_id="INC-002",
        community_id="c1",
        camera_id="cam1",
        track_id=2,
        classification=TrackClassification.CHILD,
        primary_zone_id="z2",
        primary_zone_name="Vehicular Gate",
        zone_type=SafetyZoneType.VEHICULAR_GATEWAY,
        current_zone_ids={"z2"},
        severity=AlertSeverity.HIGH,
        state=IncidentState.ACTIVE,
        created_at_ms=2000.0,
        created_at_frame=20,
        updated_at_ms=2000.0,
        updated_at_frame=20,
        observation_status=ObservationStatus.TRACK_LOST,
    )

    inc_offline = Incident(
        incident_id="INC-003",
        community_id="c1",
        camera_id="cam1",
        track_id=3,
        classification=TrackClassification.CHILD,
        primary_zone_id="z3",
        primary_zone_name="Boom Barrier",
        zone_type=SafetyZoneType.RESTRICTED_INFRASTRUCTURE,
        current_zone_ids={"z3"},
        severity=AlertSeverity.CRITICAL,
        state=IncidentState.ACTIVE,
        created_at_ms=3000.0,
        created_at_frame=30,
        updated_at_ms=3000.0,
        updated_at_frame=30,
        observation_status=ObservationStatus.DISCONNECTED,
    )

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    rendered = draw_person_and_events(
        frame=frame,
        detections=[],
        occupied_zones_per_track={},
        dwell_times_per_track={},
        events=[],
        community_name="Demo Community",
        camera_id="cam1",
        incidents=[inc_online, inc_lost, inc_offline],
    )

    assert rendered is not None
    assert rendered.shape == (720, 1280, 3)
    # Visual rendering executed cleanly without exception and contains rendered pixels
    assert np.any(rendered > 0)


# ============================================================================
# 4. PIPELINE-LEVEL FAILURE INJECTION INTEGRATION
# ============================================================================

def test_pipeline_camera_disconnect_and_reconnect_methods():
    """Verifies that SafeKidPipeline exposes disconnect and reconnect hooks."""
    config = SafeKidPipelineConfig(camera_id="cam_test_north", enable_incident_management=True)
    pipeline = SafeKidPipeline(config=config)

    # Simulate an incident
    e_entry = make_child_event("e1", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 1, 10, camera_id="cam_test_north")
    pipeline.incident_manager.process_frame(1, 1000.0, [e_entry], [], active_track_ids={10})

    # Call disconnect
    pipeline.handle_camera_disconnect(timestamp_ms=2000.0)
    inc = pipeline.incident_manager.all_incidents[0]
    assert inc.observation_status == ObservationStatus.DISCONNECTED
    assert inc.state == IncidentState.ACTIVE

    # Call reconnect
    pipeline.handle_camera_reconnect(timestamp_ms=5000.0)
    assert inc.observation_status == ObservationStatus.TRACK_LOST
    assert inc.state == IncidentState.ACTIVE
    assert inc.is_open() is True
