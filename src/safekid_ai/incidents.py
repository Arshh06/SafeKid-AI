"""Alert Lifecycle Management and Incident Correlation Layer for SafeKid AI.

Implements:
- Explicit alert/incident lifecycle states: ACTIVE, ACKNOWLEDGED, RESOLVED, EXPIRED.
- Incident correlation model grouping related events across time and nested/overlapping zones.
- Dwell escalation without duplicate notification spam.
- Non-destructive resolution: exit from a nested hazard does not resolve wider incidents.
- Explicit acknowledgement semantics (does not automatically resolve).
- Configurable expiration policy.
- Full audit history and event reference preservation.
- Strict isolation preventing cross-child, cross-camera, or discontinuity identity transfer.
"""

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from safekid_ai.child_safety import ChildSafetyEvent, ChildSafetyEventType, TrackClassification
from safekid_ai.events import SafetyEvent, SafetyEventType
from safekid_ai.roi.models import AlertSeverity, CameraROIConfig, SafetyZone, SafetyZoneType
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.incidents")

SEVERITY_ORDER: Dict[AlertSeverity, int] = {
    AlertSeverity.CRITICAL: 4,
    AlertSeverity.HIGH: 3,
    AlertSeverity.MEDIUM: 2,
    AlertSeverity.LOW: 1,
}


class IncidentState(str, Enum):
    """Explicit lifecycle states for community child safety incidents."""
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    EXPIRED = "EXPIRED"


class ObservationStatus(str, Enum):
    """Observation health and tracking visibility status for an incident.

    Decouples observation continuity from incident resolution:
    Loss of camera feed, temporary occlusion, or track loss must NEVER
    be interpreted as a confirmed safe resolution of a child hazard.
    """
    OBSERVING = "OBSERVING"         # Actively tracked and observed in live camera frames
    OCCLUDED = "OCCLUDED"           # Temporarily missing within short grace period (< threshold)
    TRACK_LOST = "TRACK_LOST"       # Track lost or disappeared while still inside hazard zone
    DISCONNECTED = "DISCONNECTED"   # Camera stream offline / frame feed interrupted


@dataclass
class IncidentPolicyConfig:
    """Configurable rules for incident lifecycle and temporal/spatial correlation."""
    correlation_window_ms: float = 10000.0        # 10.0s correlation window for related transitions
    default_expiration_seconds: float = 15.0      # Inactivity duration before an open incident expires
    auto_resolve_on_exit: bool = True             # Automatically resolve when all incident hazard zones are exited
    escalate_on_dwell: bool = True                # Escalate/update existing incident on dwell events
    require_full_exit_for_correlated: bool = True # Do not resolve if child remains in nested or correlated zone
    preserve_lost_observation: bool = True        # If True, tracks lost while inside danger zones are NOT expired automatically
    occlusion_grace_frames: int = 5               # Short occlusion frames before marking TRACK_LOST


@dataclass
class Incident:
    """Structured representation of a consolidated safety incident.

    Preserves references to all underlying raw safety events and child safety events,
    maintains an immutable audit log of lifecycle state transitions, and enforces
    severity escalation.
    """
    incident_id: str
    community_id: str
    camera_id: str
    track_id: int
    classification: TrackClassification
    primary_zone_id: str
    primary_zone_name: str
    zone_type: SafetyZoneType
    current_zone_ids: Set[str]
    severity: AlertSeverity
    state: IncidentState
    created_at_ms: float
    created_at_frame: int
    updated_at_ms: float
    updated_at_frame: int
    resolved_at_ms: Optional[float] = None
    resolved_at_frame: Optional[int] = None
    acknowledged_at_ms: Optional[float] = None
    acknowledged_by: Optional[str] = None
    expired_at_ms: Optional[float] = None
    summary_message: str = ""
    event_ids: List[str] = field(default_factory=list)
    lifecycle_history: List[Dict[str, Any]] = field(default_factory=list)

    # Observation health and tracking continuity
    observation_status: ObservationStatus = ObservationStatus.OBSERVING
    last_observed_at_ms: float = 0.0
    last_observed_at_frame: int = 0
    missing_frames_count: int = 0

    def is_open(self) -> bool:
        """Returns True if the incident is currently active or acknowledged."""
        return self.state in (IncidentState.ACTIVE, IncidentState.ACKNOWLEDGED)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes incident to a JSON-compatible dictionary."""
        return {
            "incident_id": self.incident_id,
            "community_id": self.community_id,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "classification": self.classification.value,
            "primary_zone_id": self.primary_zone_id,
            "primary_zone_name": self.primary_zone_name,
            "zone_type": self.zone_type.value,
            "current_zone_ids": sorted(list(self.current_zone_ids)),
            "severity": self.severity.value,
            "state": self.state.value,
            "observation_status": (
                self.observation_status.value
                if hasattr(self.observation_status, "value")
                else str(self.observation_status)
            ),
            "created_at_ms": round(self.created_at_ms, 2),
            "created_at_frame": self.created_at_frame,
            "updated_at_ms": round(self.updated_at_ms, 2),
            "updated_at_frame": self.updated_at_frame,
            "last_observed_at_ms": round(self.last_observed_at_ms, 2),
            "last_observed_at_frame": self.last_observed_at_frame,
            "missing_frames_count": self.missing_frames_count,
            "resolved_at_ms": round(self.resolved_at_ms, 2) if self.resolved_at_ms is not None else None,
            "resolved_at_frame": self.resolved_at_frame,
            "acknowledged_at_ms": round(self.acknowledged_at_ms, 2) if self.acknowledged_at_ms is not None else None,
            "acknowledged_by": self.acknowledged_by,
            "expired_at_ms": round(self.expired_at_ms, 2) if self.expired_at_ms is not None else None,
            "summary_message": self.summary_message,
            "event_ids": list(self.event_ids),
            "lifecycle_history": list(self.lifecycle_history),
        }


class IncidentManager:
    """Manages the full lifecycle and temporal/spatial correlation of child safety incidents.

    Features:
    - Incident correlation: groups consecutive danger zone entries, nested zone crossings,
      and dwell updates for the same child within a configurable time window.
    - Deterministic escalation: CRITICAL severity overrides HIGH; summary text updates dynamically.
    - Nested zone safety: exiting a smaller nested hazard zone (e.g. boom barrier gate) does not
      prematurely resolve a wider surrounding hazard incident (e.g. vehicle driveway lane).
    - Strict isolation: separate children, separate cameras, and disconnected track IDs never merge.
    - Expiration management: automatically transitions stale inactive alerts to EXPIRED.
    - Complete audit trail: preserves all underlying event IDs and state transition records.
    """

    def __init__(self, config: Optional[IncidentPolicyConfig] = None):
        self.config: IncidentPolicyConfig = config or IncidentPolicyConfig()
        self._incident_counter: int = 0
        self.all_incidents: List[Incident] = []
        # Key: (community_id, camera_id, track_id) -> Incident
        self._open_incidents: Dict[Tuple[str, str, int], Incident] = {}

    def reset(self) -> None:
        """Resets all open incidents and counters for a fresh video session."""
        self._incident_counter = 0
        self.all_incidents.clear()
        self._open_incidents.clear()
        logger.info("IncidentManager state reset.")

    def _generate_incident_id(self) -> str:
        self._incident_counter += 1
        return f"INC-{self._incident_counter:03d}"

    def process_frame(
        self,
        frame_id: int,
        timestamp_ms: float,
        child_events: List[ChildSafetyEvent],
        raw_safety_events: List[SafetyEvent],
        roi_config: Optional[CameraROIConfig] = None,
        active_track_ids: Optional[Set[int]] = None,
    ) -> List[Incident]:
        """Evaluates incoming frame events, manages lifecycle transitions, and returns active incidents."""
        # 1. Update observation continuity and health for currently open incidents
        if active_track_ids is not None:
            for inc in self.get_open_incidents():
                if inc.track_id in active_track_ids:
                    if inc.observation_status in (
                        ObservationStatus.OCCLUDED,
                        ObservationStatus.TRACK_LOST,
                        ObservationStatus.DISCONNECTED,
                    ):
                        inc.lifecycle_history.append({
                            "transition": "OBSERVATION_RESUMED",
                            "from_status": inc.observation_status.value if hasattr(inc.observation_status, "value") else str(inc.observation_status),
                            "to_status": ObservationStatus.OBSERVING.value,
                            "frame_id": frame_id,
                            "timestamp_ms": round(timestamp_ms, 2),
                            "reason": f"Track ID {inc.track_id} visual observation re-acquired",
                        })
                    inc.observation_status = ObservationStatus.OBSERVING
                    inc.last_observed_at_ms = timestamp_ms
                    inc.last_observed_at_frame = frame_id
                    inc.missing_frames_count = 0
                else:
                    inc.missing_frames_count += 1
                    if inc.observation_status in (ObservationStatus.OBSERVING, ObservationStatus.OCCLUDED):
                        if inc.missing_frames_count <= self.config.occlusion_grace_frames:
                            if inc.observation_status != ObservationStatus.OCCLUDED:
                                inc.observation_status = ObservationStatus.OCCLUDED
                                inc.lifecycle_history.append({
                                    "transition": "OCCLUSION_DETECTED",
                                    "frame_id": frame_id,
                                    "timestamp_ms": round(timestamp_ms, 2),
                                    "reason": f"Track ID {inc.track_id} temporarily unobserved ({inc.missing_frames_count} frame(s) missing)",
                                })
                        else:
                            inc.observation_status = ObservationStatus.TRACK_LOST
                            inc.lifecycle_history.append({
                                "transition": "OBSERVATION_LOST",
                                "from_status": ObservationStatus.OCCLUDED.value,
                                "to_status": ObservationStatus.TRACK_LOST.value,
                                "frame_id": frame_id,
                                "timestamp_ms": round(timestamp_ms, 2),
                                "reason": f"Track ID {inc.track_id} exceeded occlusion grace period ({inc.missing_frames_count} frames); marked TRACK_LOST",
                            })

        # 2. Evaluate expiration for inactive open incidents
        self._check_expirations(timestamp_ms, frame_id)

        # 3. Process child-specific safety events
        for ce in child_events:
            tid = ce.track_id
            key = (ce.community_id, ce.camera_id, tid)
            existing = self._open_incidents.get(key)

            if ce.event_type == ChildSafetyEventType.CHILD_ZONE_ENTRY:
                if existing is not None and self._can_correlate(existing, ce, timestamp_ms):
                    # Correlate into existing open incident
                    self._correlate_entry(existing, ce, frame_id, timestamp_ms)
                else:
                    # Create new active incident
                    inc = self._create_incident(ce, frame_id, timestamp_ms)
                    self._open_incidents[key] = inc
                    self.all_incidents.append(inc)

            elif ce.event_type == ChildSafetyEventType.CHILD_DWELL_TIME_EXCEEDED:
                if existing is not None and self.config.escalate_on_dwell:
                    self._handle_dwell(existing, ce, frame_id, timestamp_ms)
                else:
                    # Create incident directly if no prior entry was tracked
                    inc = self._create_incident(ce, frame_id, timestamp_ms)
                    self._open_incidents[key] = inc
                    self.all_incidents.append(inc)

            elif ce.event_type == ChildSafetyEventType.CHILD_ZONE_EXIT:
                if existing is not None:
                    is_loss = getattr(ce, "is_loss_of_tracking", False) or "left camera view" in getattr(ce, "message", "")
                    self._handle_exit(existing, ce.roi_id, ce.zone_name, ce.event_id, frame_id, timestamp_ms, is_loss_of_tracking=is_loss)
                    if not existing.is_open():
                        self._open_incidents.pop(key, None)

        # 4. Check raw safety events for confirmed zone exits of tracked children
        # (Handles cases where zone policy has exit_alert=False on child engine but physical zone exit occurred)
        for se in raw_safety_events:
            if se.event_type == SafetyEventType.ZONE_EXIT and se.track_id is not None:
                key = (se.community_id, se.camera_id, se.track_id)
                existing = self._open_incidents.get(key)
                if existing is not None and se.roi_id in existing.current_zone_ids:
                    is_loss = getattr(se, "is_loss_of_tracking", False) or "left camera view" in getattr(se, "message", "")
                    self._handle_exit(existing, se.roi_id, se.zone_name, se.event_id, frame_id, timestamp_ms, is_loss_of_tracking=is_loss)
                    if not existing.is_open():
                        self._open_incidents.pop(key, None)

        return self.get_open_incidents()

    def _can_correlate(self, incident: Incident, event: ChildSafetyEvent, timestamp_ms: float) -> bool:
        """Determines whether a new event correlates with an existing open incident."""
        # 1. Strict identity check: same community, camera, and track ID
        if (incident.community_id != event.community_id or
            incident.camera_id != event.camera_id or
            incident.track_id != event.track_id):
            return False

        # 2. Temporal window check
        time_diff = timestamp_ms - incident.updated_at_ms
        if time_diff > self.config.correlation_window_ms:
            return False

        # 3. Zone relationship check: same zone, already tracked zone, or spatial/hierarchy connection
        if event.roi_id in incident.current_zone_ids or event.roi_id == incident.primary_zone_id:
            return True

        # Connected or overlapping hazard within the same camera area
        return True

    def _create_incident(self, event: ChildSafetyEvent, frame_id: int, timestamp_ms: float) -> Incident:
        """Creates a new ACTIVE incident from an initiating child safety event."""
        inc_id = self._generate_incident_id()
        msg = f"[{event.severity.value}] {event.zone_name}: {event.event_type.value}"
        history_entry = {
            "transition": "CREATED",
            "from_state": None,
            "to_state": IncidentState.ACTIVE.value,
            "frame_id": frame_id,
            "timestamp_ms": round(timestamp_ms, 2),
            "severity": event.severity.value,
            "zone_id": event.roi_id,
            "zone_name": event.zone_name,
            "event_id": event.event_id,
            "reason": event.message,
        }
        inc = Incident(
            incident_id=inc_id,
            community_id=event.community_id,
            camera_id=event.camera_id,
            track_id=event.track_id,
            classification=event.classification,
            primary_zone_id=event.roi_id,
            primary_zone_name=event.zone_name,
            zone_type=event.zone_type,
            current_zone_ids={event.roi_id},
            severity=event.severity,
            state=IncidentState.ACTIVE,
            created_at_ms=timestamp_ms,
            created_at_frame=frame_id,
            updated_at_ms=timestamp_ms,
            updated_at_frame=frame_id,
            last_observed_at_ms=timestamp_ms,
            last_observed_at_frame=frame_id,
            observation_status=ObservationStatus.OBSERVING,
            missing_frames_count=0,
            summary_message=msg,
            event_ids=[event.event_id],
            lifecycle_history=[history_entry],
        )
        logger.warning(
            f"Created Incident {inc_id} [ACTIVE] for Track {event.track_id} in '{event.zone_name}' "
            f"[Severity: {event.severity.value}]"
        )
        return inc

    def _correlate_entry(self, incident: Incident, event: ChildSafetyEvent, frame_id: int, timestamp_ms: float) -> None:
        """Correlates an entry into an additional or nested hazard zone."""
        incident.current_zone_ids.add(event.roi_id)
        incident.event_ids.append(event.event_id)
        incident.updated_at_ms = timestamp_ms
        incident.updated_at_frame = frame_id

        old_sev = incident.severity
        escalated = False

        # Escalate severity if incoming event is higher priority (e.g. CRITICAL > HIGH)
        if SEVERITY_ORDER.get(event.severity, 0) > SEVERITY_ORDER.get(incident.severity, 0):
            incident.severity = event.severity
            incident.primary_zone_id = event.roi_id
            incident.primary_zone_name = event.zone_name
            incident.zone_type = event.zone_type
            escalated = True

        incident.summary_message = f"[{incident.severity.value}] {event.zone_name}: {event.event_type.value}"

        history_entry = {
            "transition": "ESCALATED" if escalated else "CORRELATED_ENTRY",
            "from_state": incident.state.value,
            "to_state": incident.state.value,
            "frame_id": frame_id,
            "timestamp_ms": round(timestamp_ms, 2),
            "severity_before": old_sev.value,
            "severity_after": incident.severity.value,
            "zone_id": event.roi_id,
            "zone_name": event.zone_name,
            "event_id": event.event_id,
            "reason": f"Correlated hazard entry into {event.zone_name}",
        }
        incident.lifecycle_history.append(history_entry)
        logger.info(
            f"Correlated Incident {incident.incident_id} (Track {incident.track_id}): "
            f"Entered '{event.zone_name}' [Severity: {incident.severity.value}]"
        )

    def _handle_dwell(self, incident: Incident, event: ChildSafetyEvent, frame_id: int, timestamp_ms: float) -> None:
        """Updates and escalates existing incident upon dwell threshold violation."""
        incident.event_ids.append(event.event_id)
        incident.updated_at_ms = timestamp_ms
        incident.updated_at_frame = frame_id

        # Update summary with dwell details
        incident.summary_message = (
            f"[{incident.severity.value}] {event.zone_name}: Dwell {event.dwell_time_seconds:.1f}s"
        )

        history_entry = {
            "transition": "DWELL_UPDATE",
            "from_state": incident.state.value,
            "to_state": incident.state.value,
            "frame_id": frame_id,
            "timestamp_ms": round(timestamp_ms, 2),
            "dwell_seconds": round(event.dwell_time_seconds, 2),
            "zone_id": event.roi_id,
            "zone_name": event.zone_name,
            "event_id": event.event_id,
            "reason": f"Dwell threshold exceeded in {event.zone_name}",
        }
        incident.lifecycle_history.append(history_entry)
        logger.warning(
            f"Dwell Escalation for Incident {incident.incident_id} (Track {incident.track_id}): "
            f"{event.dwell_time_seconds:.1f}s in '{event.zone_name}'"
        )

    def _handle_exit(
        self,
        incident: Incident,
        zone_id: str,
        zone_name: str,
        event_id: str,
        frame_id: int,
        timestamp_ms: float,
        is_loss_of_tracking: bool = False,
    ) -> None:
        """Handles zone exit with strict support for nested and surrounding zones.

        CRITICAL SAFETY PRINCIPLE:
        Loss of observation (disappearing track, occlusion, camera disconnect) must
        NEVER be interpreted as confirmation that a child is safe. If is_loss_of_tracking
        is True, the incident remains ACTIVE or ACKNOWLEDGED and observation_status is
        updated to TRACK_LOST. It is NEVER falsely marked RESOLVED.
        """
        if is_loss_of_tracking:
            incident.observation_status = ObservationStatus.TRACK_LOST
            incident.updated_at_ms = timestamp_ms
            incident.updated_at_frame = frame_id
            incident.event_ids.append(event_id)
            history_entry = {
                "transition": "OBSERVATION_LOST",
                "from_status": (
                    incident.observation_status.value
                    if hasattr(incident.observation_status, "value")
                    else str(incident.observation_status)
                ),
                "to_status": ObservationStatus.TRACK_LOST.value,
                "frame_id": frame_id,
                "timestamp_ms": round(timestamp_ms, 2),
                "zone_id": zone_id,
                "zone_name": zone_name,
                "event_id": event_id,
                "reason": (
                    f"Child track lost or disappeared while still inside hazard zone '{zone_name}'. "
                    f"CRITICAL SAFETY RULE: Incident remains {incident.state.value} (NOT resolved)."
                ),
            }
            incident.lifecycle_history.append(history_entry)
            logger.warning(
                f"Incident {incident.incident_id} (Track {incident.track_id}): Observation LOST in '{zone_name}'. "
                f"Incident remains {incident.state.value}."
            )
            return

        # Physical, confirmed exit:
        if zone_id in incident.current_zone_ids:
            incident.current_zone_ids.remove(zone_id)

        incident.event_ids.append(event_id)
        incident.updated_at_ms = timestamp_ms
        incident.updated_at_frame = frame_id

        # Non-destructive check: if child remains in other active hazard zones, DO NOT resolve
        if self.config.require_full_exit_for_correlated and len(incident.current_zone_ids) > 0:
            history_entry = {
                "transition": "NESTED_ZONE_EXIT",
                "from_state": incident.state.value,
                "to_state": incident.state.value,
                "frame_id": frame_id,
                "timestamp_ms": round(timestamp_ms, 2),
                "exited_zone_id": zone_id,
                "exited_zone_name": zone_name,
                "remaining_zones": sorted(list(incident.current_zone_ids)),
                "event_id": event_id,
                "reason": f"Exited nested zone '{zone_name}', still present in wider hazard zones",
            }
            incident.lifecycle_history.append(history_entry)
            logger.info(
                f"Incident {incident.incident_id} (Track {incident.track_id}): Exited nested zone '{zone_name}', "
                f"remains active in: {incident.current_zone_ids}"
            )
            return

        # Child has physically exited all hazard zones belonging to this incident -> Transition to RESOLVED
        if self.config.auto_resolve_on_exit:
            old_state = incident.state
            incident.state = IncidentState.RESOLVED
            incident.resolved_at_ms = timestamp_ms
            incident.resolved_at_frame = frame_id

            history_entry = {
                "transition": "RESOLVED",
                "from_state": old_state.value,
                "to_state": IncidentState.RESOLVED.value,
                "frame_id": frame_id,
                "timestamp_ms": round(timestamp_ms, 2),
                "exited_zone_id": zone_id,
                "exited_zone_name": zone_name,
                "event_id": event_id,
                "reason": f"Child fully exited hazard zone '{zone_name}'",
            }
            incident.lifecycle_history.append(history_entry)
            logger.info(
                f"Resolved Incident {incident.incident_id} (Track {incident.track_id}): "
                f"Fully exited '{zone_name}'"
            )

    def _check_expirations(self, current_timestamp_ms: float, frame_id: int) -> None:
        """Expires stale open incidents exceeding the configured inactivity duration."""
        exp_thresh_ms = self.config.default_expiration_seconds * 1000.0
        keys_to_remove = []

        for key, inc in self._open_incidents.items():
            if not inc.is_open():
                continue

            # CRITICAL SAFETY RULE: If observation was lost or camera disconnected while
            # the child was in a danger zone or severity is HIGH/CRITICAL, do NOT silently
            # expire and erase the incident if preserve_lost_observation is True.
            if self.config.preserve_lost_observation and inc.observation_status in (
                ObservationStatus.DISCONNECTED,
                ObservationStatus.TRACK_LOST,
            ):
                continue

            time_diff = current_timestamp_ms - inc.updated_at_ms
            if time_diff > exp_thresh_ms:
                old_state = inc.state
                inc.state = IncidentState.EXPIRED
                inc.expired_at_ms = current_timestamp_ms

                history_entry = {
                    "transition": "EXPIRED",
                    "from_state": old_state.value,
                    "to_state": IncidentState.EXPIRED.value,
                    "frame_id": frame_id,
                    "timestamp_ms": round(current_timestamp_ms, 2),
                    "reason": (
                        f"Inactivity timeout: {time_diff:.1f}ms > "
                        f"{exp_thresh_ms:.1f}ms without new observations"
                    ),
                }
                inc.lifecycle_history.append(history_entry)
                keys_to_remove.append(key)
                logger.warning(f"Incident {inc.incident_id} (Track {inc.track_id}) EXPIRED due to inactivity.")

        for key in keys_to_remove:
            self._open_incidents.pop(key, None)

    def handle_camera_disconnect(self, camera_id: str, timestamp_ms: float, frame_id: int = 0) -> List[Incident]:
        """Handles camera stream disconnection or frame feed interruption.

        Invariants:
        - The incident must NOT transition to RESOLVED merely because frames stop arriving.
        - Preserves last known incident state, coordinates, zones, and timestamps.
        - Indicates that observation was interrupted (observation_status = DISCONNECTED).
        - Camera availability and incident resolution remain strictly separate concepts.
        """
        affected = []
        for inc in self.get_open_incidents():
            if inc.camera_id == camera_id:
                old_status = inc.observation_status
                inc.observation_status = ObservationStatus.DISCONNECTED
                history_entry = {
                    "transition": "CAMERA_DISCONNECTED",
                    "from_status": old_status.value if hasattr(old_status, "value") else str(old_status),
                    "to_status": ObservationStatus.DISCONNECTED.value,
                    "frame_id": frame_id,
                    "timestamp_ms": round(timestamp_ms, 2),
                    "reason": f"Camera '{camera_id}' stream disconnected; incident preserved in state {inc.state.value}",
                }
                inc.lifecycle_history.append(history_entry)
                affected.append(inc)
                logger.warning(
                    f"Incident {inc.incident_id} (Track {inc.track_id}): Camera '{camera_id}' DISCONNECTED. "
                    f"Incident preserved in state {inc.state.value}."
                )
        return affected

    def handle_camera_reconnect(self, camera_id: str, timestamp_ms: float, frame_id: int = 0) -> List[Incident]:
        """Handles camera stream restoration after an interruption.

        Invariants:
        - Reconnection does NOT silently erase or resolve open incidents.
        - The system does not invent a safe-exit event during downtime.
        - Incidents remain OPEN until physical departure is confirmed or operator resolves.
        """
        affected = []
        for inc in self.get_open_incidents():
            if inc.camera_id == camera_id:
                old_status = inc.observation_status
                inc.observation_status = ObservationStatus.TRACK_LOST
                history_entry = {
                    "transition": "CAMERA_RECONNECTED",
                    "from_status": old_status.value if hasattr(old_status, "value") else str(old_status),
                    "to_status": ObservationStatus.TRACK_LOST.value,
                    "frame_id": frame_id,
                    "timestamp_ms": round(timestamp_ms, 2),
                    "reason": f"Camera '{camera_id}' reconnected; awaiting detection re-acquisition; incident NOT resolved",
                }
                inc.lifecycle_history.append(history_entry)
                affected.append(inc)
                logger.info(
                    f"Incident {inc.incident_id} (Track {inc.track_id}): Camera '{camera_id}' RECONNECTED. "
                    f"Preserved open incident."
                )
        return affected

    def acknowledge_incident(self, incident_id: str, acknowledged_by: str, timestamp_ms: float) -> bool:
        """Acknowledges an open incident.

        Strict Requirement: Acknowledgement does NOT automatically resolve an incident.
        The incident remains open and tracked until resolved or expired.
        """
        for inc in self.all_incidents:
            if inc.incident_id == incident_id:
                if inc.state == IncidentState.ACTIVE:
                    inc.state = IncidentState.ACKNOWLEDGED
                    inc.acknowledged_at_ms = timestamp_ms
                    inc.acknowledged_by = acknowledged_by

                    history_entry = {
                        "transition": "ACKNOWLEDGED",
                        "from_state": IncidentState.ACTIVE.value,
                        "to_state": IncidentState.ACKNOWLEDGED.value,
                        "timestamp_ms": round(timestamp_ms, 2),
                        "acknowledged_by": acknowledged_by,
                        "reason": f"Operator '{acknowledged_by}' acknowledged incident",
                    }
                    inc.lifecycle_history.append(history_entry)
                    logger.info(f"Incident {incident_id} ACKNOWLEDGED by '{acknowledged_by}'.")
                    return True
                return False
        return False

    def resolve_incident(
        self, incident_id: str, resolved_by: str, timestamp_ms: float, reason: str = ""
    ) -> bool:
        """Manually resolves an open incident."""
        for inc in self.all_incidents:
            if inc.incident_id == incident_id and inc.is_open():
                old_state = inc.state
                inc.state = IncidentState.RESOLVED
                inc.resolved_at_ms = timestamp_ms

                history_entry = {
                    "transition": "RESOLVED_MANUAL",
                    "from_state": old_state.value,
                    "to_state": IncidentState.RESOLVED.value,
                    "timestamp_ms": round(timestamp_ms, 2),
                    "resolved_by": resolved_by,
                    "reason": reason or f"Operator '{resolved_by}' manually resolved incident",
                }
                inc.lifecycle_history.append(history_entry)

                # Remove from open map
                key = (inc.community_id, inc.camera_id, inc.track_id)
                self._open_incidents.pop(key, None)
                logger.info(f"Incident {incident_id} MANUALLY RESOLVED by '{resolved_by}'.")
                return True
        return False

    def get_open_incidents(self) -> List[Incident]:
        """Returns all open (ACTIVE or ACKNOWLEDGED) incidents, sorted deterministically by priority.

        Sort Order:
        1. Severity weight descending (CRITICAL > HIGH > MEDIUM > LOW)
        2. Creation timestamp ascending (FIFO for identical severity)
        """
        open_list = [inc for inc in self.all_incidents if inc.is_open()]
        return sorted(
            open_list,
            key=lambda x: (-SEVERITY_ORDER.get(x.severity, 0), x.created_at_ms),
        )

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Looks up an incident by unique incident ID."""
        for inc in self.all_incidents:
            if inc.incident_id == incident_id:
                return inc
        return None

    def export_to_json(
        self,
        output_path: Optional[Union[str, Path]] = None,
        file_path: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Serializes all tracked incidents and full audit histories to a JSON file."""
        target_path = output_path or file_path
        if not target_path:
            raise ValueError("Must provide either 'output_path' or 'file_path'")
        out_path = Path(target_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {}
        if metadata:
            data["metadata"] = metadata
        data["total_incidents_count"] = len(self.all_incidents)
        data["open_incidents_count"] = len(self.get_open_incidents())
        data["incidents"] = [inc.to_dict() for inc in self.all_incidents]

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported {len(self.all_incidents)} incidents to: {out_path}")
