"""Deterministic tests for SafeKid AI Alert Lifecycle Management and Incident Correlation.

Covers:
1. Entry creates an active incident.
2. Dwell escalates an existing incident without duplicate notification spam.
3. Exit resolves only the relevant hazard.
4. Acknowledgement does not resolve an incident.
5. Expiration works according to configuration.
6. HIGH and CRITICAL events correlate correctly into a single incident with CRITICAL precedence.
7. Nested zones do not cause duplicate active notifications.
8. Different children never get merged.
9. Different cameras never get merged.
10. Track-ID discontinuity prevents silent identity transfer.
11. Multiple simultaneous incidents are prioritized correctly (CRITICAL > HIGH > MEDIUM > LOW).
12. Historical event logs and audit history remain complete.
"""

import pytest

from safekid_ai.child_safety import ChildSafetyEvent, ChildSafetyEventType, TrackClassification
from safekid_ai.events import SafetyEvent, SafetyEventType
from safekid_ai.incidents import (
    Incident,
    IncidentManager,
    IncidentPolicyConfig,
    IncidentState,
    SEVERITY_ORDER,
)
from safekid_ai.roi.models import AlertSeverity, SafetyZoneType


def make_child_event(
    event_id: str,
    event_type: ChildSafetyEventType,
    timestamp_ms: float,
    frame_id: int,
    track_id: int,
    roi_id: str = "driveway_01",
    zone_name: str = "Main Vehicular Driveway",
    severity: AlertSeverity = AlertSeverity.HIGH,
    community_id: str = "comm_01",
    camera_id: str = "cam_01",
    dwell_time_seconds: float = 0.0,
    zone_type: SafetyZoneType = SafetyZoneType.VEHICULAR_GATEWAY,
) -> ChildSafetyEvent:
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
        bottom_center=(500.0, 500.0),
        bbox_xyxy=(480.0, 400.0, 520.0, 500.0),
        dwell_time_seconds=dwell_time_seconds,
        message=f"Child {track_id} in {zone_name}",
    )


# ============================================================================
# 1. ENTRY CREATES ACTIVE INCIDENT
# ============================================================================

def test_entry_creates_active_incident():
    """Verifies that an entry into a danger zone creates an ACTIVE incident."""
    manager = IncidentManager()
    evt = make_child_event(
        event_id="evt_01",
        event_type=ChildSafetyEventType.CHILD_ZONE_ENTRY,
        timestamp_ms=1000.0,
        frame_id=24,
        track_id=2,
        roi_id="driveway_lane",
        zone_name="Main Vehicular Driveway",
        severity=AlertSeverity.HIGH,
    )

    open_incidents = manager.process_frame(
        frame_id=24,
        timestamp_ms=1000.0,
        child_events=[evt],
        raw_safety_events=[],
    )

    assert len(open_incidents) == 1
    inc = open_incidents[0]
    assert inc.state == IncidentState.ACTIVE
    assert inc.track_id == 2
    assert inc.severity == AlertSeverity.HIGH
    assert inc.primary_zone_id == "driveway_lane"
    assert "driveway_lane" in inc.current_zone_ids
    assert "evt_01" in inc.event_ids
    assert inc.created_at_ms == 1000.0
    assert inc.is_open() is True
    assert len(inc.lifecycle_history) >= 1
    assert inc.lifecycle_history[0]["transition"] == "CREATED"


# ============================================================================
# 2. DWELL ESCALATES EXISTING INCIDENT
# ============================================================================

def test_dwell_escalates_existing_incident_without_duplicates():
    """Verifies that a dwell event updates and escalates the existing incident rather than duplicating."""
    manager = IncidentManager()

    # Frame 24: Entry
    e_entry = make_child_event("evt_entry", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 24, 2)
    manager.process_frame(24, 1000.0, [e_entry], [])

    # Frame 72: Dwell (2.0s exceeded)
    e_dwell = make_child_event(
        "evt_dwell",
        ChildSafetyEventType.CHILD_DWELL_TIME_EXCEEDED,
        3000.0,
        72,
        2,
        dwell_time_seconds=2.0,
    )
    open_incidents = manager.process_frame(72, 3000.0, [e_dwell], [])

    # Must still be ONE incident, not two!
    assert len(open_incidents) == 1
    assert len(manager.all_incidents) == 1
    inc = open_incidents[0]
    assert inc.state == IncidentState.ACTIVE
    assert "evt_dwell" in inc.event_ids
    assert inc.updated_at_ms == 3000.0
    assert "2.0s" in inc.summary_message
    assert any(h["transition"] == "DWELL_UPDATE" for h in inc.lifecycle_history)


# ============================================================================
# 3. EXIT RESOLVES ONLY RELEVANT HAZARD
# ============================================================================

def test_exit_resolves_incident_when_all_hazard_zones_exited():
    """Verifies that exiting the hazard zone transitions incident to RESOLVED."""
    manager = IncidentManager()

    # Enter
    e_in = make_child_event("e1", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 24, 2)
    manager.process_frame(24, 1000.0, [e_in], [])
    assert len(manager.get_open_incidents()) == 1

    # Exit
    e_out = make_child_event("e2", ChildSafetyEventType.CHILD_ZONE_EXIT, 5000.0, 120, 2)
    open_incidents = manager.process_frame(120, 5000.0, [e_out], [])

    # Incident must be resolved and no longer open
    assert len(open_incidents) == 0
    assert len(manager.all_incidents) == 1
    inc = manager.all_incidents[0]
    assert inc.state == IncidentState.RESOLVED
    assert inc.resolved_at_ms == 5000.0
    assert inc.is_open() is False
    assert any(h["transition"] == "RESOLVED" for h in inc.lifecycle_history)


# ============================================================================
# 4. ACKNOWLEDGEMENT DOES NOT RESOLVE AN INCIDENT
# ============================================================================

def test_acknowledgement_does_not_resolve_incident():
    """Verifies that an acknowledged incident remains open and is NOT resolved."""
    manager = IncidentManager()
    e_in = make_child_event("e1", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 24, 2)
    open_incs = manager.process_frame(24, 1000.0, [e_in], [])
    inc_id = open_incs[0].incident_id

    # Acknowledge incident
    success = manager.acknowledge_incident(inc_id, acknowledged_by="guard_station_01", timestamp_ms=2000.0)
    assert success is True

    inc = manager.get_incident(inc_id)
    assert inc.state == IncidentState.ACKNOWLEDGED
    assert inc.acknowledged_by == "guard_station_01"
    assert inc.acknowledged_at_ms == 2000.0
    # Must STILL be open!
    assert inc.is_open() is True
    assert len(manager.get_open_incidents()) == 1
    assert manager.get_open_incidents()[0].state == IncidentState.ACKNOWLEDGED


# ============================================================================
# 5. EXPIRATION WORKS ACCORDING TO CONFIGURATION
# ============================================================================

def test_expiration_policy_transitions_stale_incidents():
    """Verifies that an incident with no activity past the expiration window transitions to EXPIRED."""
    # Configure 5-second expiration
    cfg = IncidentPolicyConfig(default_expiration_seconds=5.0)
    manager = IncidentManager(config=cfg)

    # Enter at t=1000ms
    e_in = make_child_event("e1", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 24, 2)
    manager.process_frame(24, 1000.0, [e_in], [])
    assert len(manager.get_open_incidents()) == 1

    # At t=4000ms (3s elapsed < 5s threshold): still active
    open_incs = manager.process_frame(96, 4000.0, [], [])
    assert len(open_incs) == 1

    # At t=7000ms (6s elapsed > 5s threshold): incident expires
    open_incs = manager.process_frame(168, 7000.0, [], [])
    assert len(open_incs) == 0

    inc = manager.all_incidents[0]
    assert inc.state == IncidentState.EXPIRED
    assert inc.expired_at_ms == 7000.0
    assert inc.is_open() is False
    assert any(h["transition"] == "EXPIRED" for h in inc.lifecycle_history)


# ============================================================================
# 6. HIGH AND CRITICAL EVENTS CORRELATE CORRECTLY (CRITICAL PRECEDENCE)
# ============================================================================

def test_high_and_critical_events_correlate_with_critical_precedence():
    """Verifies that a child entering driveway (HIGH) and boom barrier (CRITICAL) correlates to 1 incident with CRITICAL severity."""
    manager = IncidentManager()

    # Event 1: Child enters Driveway (HIGH)
    e_driveway = make_child_event(
        "evt_dw",
        ChildSafetyEventType.CHILD_ZONE_ENTRY,
        1000.0,
        24,
        2,
        roi_id="driveway_lane",
        zone_name="Main Vehicular Driveway",
        severity=AlertSeverity.HIGH,
    )
    open_incs = manager.process_frame(24, 1000.0, [e_driveway], [])
    assert len(open_incs) == 1
    assert open_incs[0].severity == AlertSeverity.HIGH

    # Event 2: Child progresses into Boom Barrier Mechanical Zone (CRITICAL) 800ms later
    e_barrier = make_child_event(
        "evt_bar",
        ChildSafetyEventType.CHILD_ZONE_ENTRY,
        1800.0,
        43,
        2,
        roi_id="boom_barrier_danger",
        zone_name="Boom Barrier Mechanical Zone",
        severity=AlertSeverity.CRITICAL,
    )
    open_incs = manager.process_frame(43, 1800.0, [e_barrier], [])

    # Must correlate into the SAME incident with CRITICAL taking precedence!
    assert len(open_incs) == 1
    inc = open_incs[0]
    assert inc.severity == AlertSeverity.CRITICAL
    assert "driveway_lane" in inc.current_zone_ids
    assert "boom_barrier_danger" in inc.current_zone_ids
    assert inc.event_ids == ["evt_dw", "evt_bar"]
    assert any(h["transition"] == "ESCALATED" for h in inc.lifecycle_history)


# ============================================================================
# 7. NESTED ZONES DO NOT CAUSE DUPLICATE NOTIFICATIONS OR PREMATURE RESOLUTION
# ============================================================================

def test_nested_zone_exit_does_not_resolve_wider_hazard():
    """Verifies that exiting a nested zone (boom barrier) does not resolve the wider hazard (driveway lane)."""
    manager = IncidentManager()

    # Enter driveway (HIGH)
    e1 = make_child_event("e1", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 24, 2, roi_id="driveway_lane")
    manager.process_frame(24, 1000.0, [e1], [])

    # Enter nested boom barrier (CRITICAL)
    e2 = make_child_event("e2", ChildSafetyEventType.CHILD_ZONE_ENTRY, 2000.0, 48, 2, roi_id="boom_barrier_danger", severity=AlertSeverity.CRITICAL)
    manager.process_frame(48, 2000.0, [e2], [])
    assert len(manager.get_open_incidents()) == 1

    # Child steps out of boom barrier into driveway (exits boom_barrier_danger, still in driveway_lane)
    e_exit_bar = make_child_event("e3", ChildSafetyEventType.CHILD_ZONE_EXIT, 5000.0, 120, 2, roi_id="boom_barrier_danger")
    open_incs = manager.process_frame(120, 5000.0, [e_exit_bar], [])

    # Must NOT resolve! Incident must remain ACTIVE because child is still in driveway_lane!
    assert len(open_incs) == 1
    inc = open_incs[0]
    assert inc.state == IncidentState.ACTIVE
    assert "driveway_lane" in inc.current_zone_ids
    assert "boom_barrier_danger" not in inc.current_zone_ids
    assert any(h["transition"] == "NESTED_ZONE_EXIT" for h in inc.lifecycle_history)

    # Finally child exits driveway onto sidewalk
    e_exit_dw = make_child_event("e4", ChildSafetyEventType.CHILD_ZONE_EXIT, 6500.0, 156, 2, roi_id="driveway_lane")
    open_incs2 = manager.process_frame(156, 6500.0, [e_exit_dw], [])

    # Now all hazard zones are exited -> RESOLVED!
    assert len(open_incs2) == 0
    assert inc.state == IncidentState.RESOLVED


# ============================================================================
# 8. DIFFERENT CHILDREN NEVER GET MERGED
# ============================================================================

def test_different_children_never_get_merged():
    """Verifies that separate children in the same zone create separate independent incidents."""
    manager = IncidentManager()

    # Child 1 (Track 2) enters driveway
    e_child1 = make_child_event("e1", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 24, track_id=2)
    # Child 2 (Track 4) enters driveway
    e_child2 = make_child_event("e2", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1200.0, 29, track_id=4)

    open_incs = manager.process_frame(29, 1200.0, [e_child1, e_child2], [])

    assert len(open_incs) == 2
    tids = {inc.track_id for inc in open_incs}
    assert tids == {2, 4}
    assert open_incs[0].incident_id != open_incs[1].incident_id


# ============================================================================
# 9. DIFFERENT CAMERAS NEVER GET MERGED
# ============================================================================

def test_different_cameras_never_get_merged():
    """Verifies that events from different cameras create separate incidents."""
    manager = IncidentManager()

    # Camera 1 event
    e_cam1 = make_child_event("e1", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 24, track_id=2, camera_id="camera_gate_north")
    # Camera 2 event with same track ID
    e_cam2 = make_child_event("e2", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1050.0, 25, track_id=2, camera_id="camera_playground_01")

    open_incs = manager.process_frame(25, 1050.0, [e_cam1, e_cam2], [])

    assert len(open_incs) == 2
    cam_ids = {inc.camera_id for inc in open_incs}
    assert cam_ids == {"camera_gate_north", "camera_playground_01"}


# ============================================================================
# 10. TRACK-ID DISCONTINUITY PREVENTS SILENT IDENTITY TRANSFER
# ============================================================================

def test_track_discontinuity_prevents_silent_identity_transfer():
    """Verifies that when a track ID disappears and a new ID appears, incidents do not transfer."""
    manager = IncidentManager()

    # Track 2 has active incident
    e_tr2 = make_child_event("e1", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 24, track_id=2)
    manager.process_frame(24, 1000.0, [e_tr2], [])

    # Track 2 becomes occluded / discontinuous; Track 5 appears at frame 80
    e_tr5 = make_child_event("e2", ChildSafetyEventType.CHILD_ZONE_ENTRY, 3300.0, 80, track_id=5)
    open_incs = manager.process_frame(80, 3300.0, [e_tr5], [])

    # Track 2 and Track 5 must remain completely distinct incidents
    assert len(open_incs) == 2
    t2_inc = next(i for i in open_incs if i.track_id == 2)
    t5_inc = next(i for i in open_incs if i.track_id == 5)
    assert t2_inc.incident_id != t5_inc.incident_id
    assert t2_inc.event_ids == ["e1"]
    assert t5_inc.event_ids == ["e2"]


# ============================================================================
# 11. MULTIPLE SIMULTANEOUS INCIDENTS ARE PRIORITIZED CORRECTLY
# ============================================================================

def test_multiple_simultaneous_incidents_prioritization():
    """Verifies deterministic sorting: CRITICAL > HIGH > MEDIUM > LOW with FIFO secondary."""
    manager = IncidentManager()

    # 4 simultaneous incidents with different severities and creation times
    e_low = make_child_event("e_low", ChildSafetyEventType.CHILD_ZONE_ENTRY, 100.0, 2, 1, severity=AlertSeverity.LOW)
    e_high = make_child_event("e_high", ChildSafetyEventType.CHILD_ZONE_ENTRY, 200.0, 4, 2, severity=AlertSeverity.HIGH)
    e_crit = make_child_event("e_crit", ChildSafetyEventType.CHILD_ZONE_ENTRY, 300.0, 6, 3, severity=AlertSeverity.CRITICAL)
    e_med = make_child_event("e_med", ChildSafetyEventType.CHILD_ZONE_ENTRY, 400.0, 8, 4, severity=AlertSeverity.MEDIUM)

    manager.process_frame(8, 400.0, [e_low, e_high, e_crit, e_med], [])
    open_incs = manager.get_open_incidents()

    assert len(open_incs) == 4
    severities = [inc.severity for inc in open_incs]
    assert severities == [
        AlertSeverity.CRITICAL,
        AlertSeverity.HIGH,
        AlertSeverity.MEDIUM,
        AlertSeverity.LOW,
    ]


# ============================================================================
# 12. HISTORICAL EVENT LOGS REMAIN COMPLETE
# ============================================================================

def test_historical_event_logs_remain_complete(tmp_path):
    """Verifies that resolved and expired incidents are preserved in full with all event IDs and audit records."""
    manager = IncidentManager()

    # Create and resolve incident
    e_in = make_child_event("evt_start", ChildSafetyEventType.CHILD_ZONE_ENTRY, 1000.0, 24, 2)
    e_dwell = make_child_event("evt_mid", ChildSafetyEventType.CHILD_DWELL_TIME_EXCEEDED, 3000.0, 72, 2, dwell_time_seconds=2.0)
    e_out = make_child_event("evt_end", ChildSafetyEventType.CHILD_ZONE_EXIT, 5000.0, 120, 2)

    manager.process_frame(24, 1000.0, [e_in], [])
    manager.process_frame(72, 3000.0, [e_dwell], [])
    manager.process_frame(120, 5000.0, [e_out], [])

    # No open incidents
    assert len(manager.get_open_incidents()) == 0
    # But all_incidents preserves the complete record!
    assert len(manager.all_incidents) == 1
    inc = manager.all_incidents[0]
    assert inc.event_ids == ["evt_start", "evt_mid", "evt_end"]
    assert len(inc.lifecycle_history) == 3
    transitions = [h["transition"] for h in inc.lifecycle_history]
    assert transitions == ["CREATED", "DWELL_UPDATE", "RESOLVED"]

    # Test serialization to JSON
    out_file = tmp_path / "test_incidents.json"
    manager.export_to_json(out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0

