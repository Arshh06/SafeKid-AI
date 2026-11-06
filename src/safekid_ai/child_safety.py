"""Generalized Child-Focused Safety Event Layer for SafeKid AI.

Hazard-agnostic, configurable safety event processing for gated residential communities.
Separates person detection/tracking from child-specific safety alerts.

Supports:
- Manual ground-truth track annotations for controlled validation and testing.
- Child-specific zone transitions (entry, exit, dwell threshold exceeded).
- Exclusion of verified adults and unclassified persons from child alerts.
- Honest handling of ByteTrack track identity discontinuities (no silent ID transfer).
- Multi-environment hazard support (playgrounds, vehicle gateways, pools, perimeters).
"""

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from safekid_ai.events import SafetyEvent, SafetyEventType
from safekid_ai.roi.models import (
    AlertSeverity,
    CameraROIConfig,
    SafetyZone,
    SafetyZoneType,
    ZoneRole,
    ZoneSafetyPolicy,
)
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.child_safety")


class TrackClassification(str, Enum):
    """Classification of tracked persons for controlled testing and evaluation.

    NOTE: Manual ground-truth classifications are for controlled validation only.
    They do not represent automated real-time age classification in operational mode.
    """
    CHILD = "CHILD"
    ADULT = "ADULT"
    UNCLASSIFIED = "UNCLASSIFIED"


class ChildSafetyEventType(str, Enum):
    """Specific categories of child-focused safety events."""
    CHILD_ZONE_ENTRY = "CHILD_ZONE_ENTRY"
    CHILD_ZONE_EXIT = "CHILD_ZONE_EXIT"
    CHILD_DWELL_TIME_EXCEEDED = "CHILD_DWELL_TIME_EXCEEDED"


@dataclass
class ChildTrackAnnotation:
    """Ground-truth annotation for a specific track ID in a controlled evaluation sequence."""
    track_id: int
    classification: TrackClassification
    notes: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "classification": self.classification.value,
            "notes": self.notes,
            "confidence": self.confidence,
        }


@dataclass
class TrackDiscontinuity:
    """Record of a potential or observed tracking-ID discontinuity.

    ByteTrack assigns ephemeral IDs that may switch or be re-created upon occlusion.
    SafeKid AI does NOT silently transfer child classifications across discontinuities.
    """
    discontinuity_id: str
    frame_id: int
    timestamp_ms: float
    original_track_id: int
    zone_id: Optional[str]
    description: str
    potential_new_track_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "discontinuity_id": self.discontinuity_id,
            "frame_id": self.frame_id,
            "timestamp_ms": round(self.timestamp_ms, 2),
            "original_track_id": self.original_track_id,
            "zone_id": self.zone_id,
            "description": self.description,
            "potential_new_track_id": self.potential_new_track_id,
        }


@dataclass
class ChildSafetyEvent:
    """Structured representation of a child-specific safety hazard event."""
    event_id: str
    event_type: ChildSafetyEventType
    timestamp_ms: float
    frame_id: int
    community_id: str
    camera_id: str
    track_id: int
    classification: TrackClassification
    roi_id: str
    zone_name: str
    zone_type: SafetyZoneType
    severity: AlertSeverity
    bottom_center: Tuple[float, float]
    bbox_xyxy: Tuple[float, float, float, float]
    dwell_time_seconds: float
    message: str
    underlying_event_id: Optional[str] = None
    is_loss_of_tracking: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes child safety event to a structured dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp_ms": round(self.timestamp_ms, 2),
            "timestamp_seconds": round(self.timestamp_ms / 1000.0, 3),
            "frame_id": self.frame_id,
            "community_id": self.community_id,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "classification": self.classification.value,
            "roi_id": self.roi_id,
            "zone_name": self.zone_name,
            "zone_type": self.zone_type.value,
            "severity": self.severity.value,
            "bottom_center": [round(self.bottom_center[0], 2), round(self.bottom_center[1], 2)],
            "bbox_xyxy": [round(c, 2) for c in self.bbox_xyxy],
            "dwell_time_seconds": round(self.dwell_time_seconds, 2),
            "message": self.message,
            "underlying_event_id": self.underlying_event_id,
            "is_loss_of_tracking": self.is_loss_of_tracking,
        }


class TrackClassificationProvider:
    """Abstract interface for querying person track classifications."""

    def get_classification(self, track_id: Optional[int], timestamp_ms: float = 0.0) -> TrackClassification:
        raise NotImplementedError

    def get_annotation(self, track_id: Optional[int]) -> Optional[ChildTrackAnnotation]:
        return None

    def list_annotations(self) -> Dict[int, ChildTrackAnnotation]:
        return {}


class ManualTrackAnnotationProvider(TrackClassificationProvider):
    """Ground-truth track annotation provider for controlled testing and audits.

    Enforces strict isolation:
    - Any track ID not explicitly listed in annotations defaults to UNCLASSIFIED.
    - No automatic inference from bounding box dimensions, aspect ratio, or velocity.
    - No automatic transfer of child labels to newly created or reassigned track IDs.
    """

    def __init__(
        self,
        community_id: str = "",
        camera_id: str = "",
        annotations: Optional[Dict[int, ChildTrackAnnotation]] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.community_id = community_id
        self.camera_id = camera_id
        self._annotations: Dict[int, ChildTrackAnnotation] = annotations or {}
        self.description = description
        self.metadata = metadata or {}

    def set_annotation(
        self,
        track_id: int,
        classification: Union[TrackClassification, str],
        notes: str = "",
        confidence: float = 1.0,
    ) -> None:
        """Sets or updates manual ground-truth annotation for a track ID."""
        if isinstance(classification, str):
            classification = TrackClassification(classification.upper())
        self._annotations[track_id] = ChildTrackAnnotation(
            track_id=track_id,
            classification=classification,
            notes=notes,
            confidence=confidence,
        )

    def get_classification(self, track_id: Optional[int], timestamp_ms: float = 0.0) -> TrackClassification:
        """Returns the verified classification for a track ID, defaulting to UNCLASSIFIED."""
        if track_id is None:
            return TrackClassification.UNCLASSIFIED
        ann = self._annotations.get(track_id)
        if ann is not None:
            return ann.classification
        return TrackClassification.UNCLASSIFIED

    def get_annotation(self, track_id: Optional[int]) -> Optional[ChildTrackAnnotation]:
        if track_id is None:
            return None
        return self._annotations.get(track_id)

    def list_annotations(self) -> Dict[int, ChildTrackAnnotation]:
        return dict(self._annotations)

    @classmethod
    def load_from_json(cls, json_path: Union[str, Path]) -> "ManualTrackAnnotationProvider":
        """Loads manual track annotations from a JSON file."""
        path = Path(json_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Track annotation file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        community_id = data.get("community_id", "")
        camera_id = data.get("camera_id", "")
        description = data.get("description", "")
        metadata = {k: v for k, v in data.items() if k not in ("annotations", "community_id", "camera_id", "description")}

        annotations_dict: Dict[int, ChildTrackAnnotation] = {}
        raw_annotations = data.get("annotations", {})

        for tid_str, item in raw_annotations.items():
            try:
                tid = int(tid_str)
            except ValueError:
                logger.warning(f"Invalid non-integer track ID in annotation JSON: '{tid_str}'")
                continue

            if isinstance(item, str):
                cls_val = TrackClassification(item.upper())
                notes = ""
                conf = 1.0
            elif isinstance(item, dict):
                cls_str = item.get("classification", "UNCLASSIFIED").upper()
                cls_val = TrackClassification(cls_str)
                notes = item.get("notes", "")
                conf = float(item.get("confidence", 1.0))
            else:
                continue

            annotations_dict[tid] = ChildTrackAnnotation(
                track_id=tid,
                classification=cls_val,
                notes=notes,
                confidence=conf,
            )

        logger.info(
            f"Loaded {len(annotations_dict)} track annotations for camera '{camera_id}' ({community_id}) from {path.name}"
        )
        return cls(
            community_id=community_id,
            camera_id=camera_id,
            annotations=annotations_dict,
            description=description,
            metadata=metadata,
        )

    def save_to_json(self, json_path: Union[str, Path]) -> None:
        """Serializes manual track annotations to a JSON file."""
        path = Path(json_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "community_id": self.community_id,
            "camera_id": self.camera_id,
            "description": self.description,
            "annotations": {
                str(tid): ann.to_dict() for tid, ann in self._annotations.items()
            },
        }
        payload.update(self.metadata)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


@dataclass
class ExcludedTrackEvent:
    """Record of a person safety event excluded from child-specific alerts."""
    event_id: str
    track_id: Optional[int]
    classification: TrackClassification
    event_type: SafetyEventType
    zone_name: str
    zone_type: SafetyZoneType
    timestamp_ms: float
    frame_id: int
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "track_id": self.track_id,
            "classification": self.classification.value,
            "event_type": self.event_type.value,
            "zone_name": self.zone_name,
            "zone_type": self.zone_type.value,
            "timestamp_ms": round(self.timestamp_ms, 2),
            "frame_id": self.frame_id,
            "reason": self.reason,
        }


class ChildSafetyEventEngine:
    """Generalized, hazard-agnostic child-focused safety event engine.

    Responsibilities:
    - Filters general person safety events into child-specific alerts based on track classification.
    - Generates child events (CHILD_ZONE_ENTRY, CHILD_ZONE_EXIT, CHILD_DWELL_TIME_EXCEEDED).
    - Excludes verified adults and unclassified persons from child alert streams.
    - Maintains comprehensive audit logs of all child alerts, excluded events, and discontinuities.
    - Operates consistently across any community environment (playgrounds, vehicle gates, pools, perimeters).
    """

    def __init__(
        self,
        annotation_provider: Optional[TrackClassificationProvider] = None,
        roi_config: Optional[CameraROIConfig] = None,
        zone_policies: Optional[Dict[str, ZoneSafetyPolicy]] = None,
    ):
        self.annotation_provider: TrackClassificationProvider = (
            annotation_provider or ManualTrackAnnotationProvider()
        )
        self._roi_config: Optional[CameraROIConfig] = roi_config
        self.zone_policies: Dict[str, ZoneSafetyPolicy] = zone_policies or {}
        self.child_events_history: List[ChildSafetyEvent] = []
        self.excluded_events_history: List[ExcludedTrackEvent] = []
        self.discontinuities_history: List[TrackDiscontinuity] = []

        # Internal tracking for discontinuity detection
        self._last_seen_child_tracks: Dict[int, Tuple[int, float, str]] = {}  # tid -> (frame_id, timestamp_ms, zone_id)

    def reset(self) -> None:
        """Resets engine history and audit records."""
        self.child_events_history.clear()
        self.excluded_events_history.clear()
        self.discontinuities_history.clear()
        self._last_seen_child_tracks.clear()
        logger.info("ChildSafetyEventEngine state reset.")

    def set_annotation_provider(self, provider: TrackClassificationProvider) -> None:
        """Updates the track classification provider."""
        self.annotation_provider = provider

    def set_roi_config(self, roi_config: Optional[CameraROIConfig]) -> None:
        """Updates the camera ROI configuration for policy lookups."""
        self._roi_config = roi_config

    def set_zone_policy(self, roi_id: str, policy: ZoneSafetyPolicy) -> None:
        """Explicitly assigns or overrides a safety policy for a zone ID."""
        self.zone_policies[roi_id] = policy

    def get_zone_policy(
        self,
        roi_id: str,
        roi_config: Optional[CameraROIConfig] = None,
        event: Optional[SafetyEvent] = None,
    ) -> ZoneSafetyPolicy:
        """Retrieves the configured safety policy for a zone ID, falling back to sensible defaults."""
        if roi_id in self.zone_policies:
            return self.zone_policies[roi_id]

        active_cfg = roi_config or self._roi_config
        if active_cfg is not None:
            zone = active_cfg.get_zone(roi_id)
            if zone is not None and zone.policy is not None:
                return zone.policy
            elif zone is not None:
                return ZoneSafetyPolicy.create_default(
                    severity=zone.severity,
                    dwell_threshold_seconds=zone.dwell_time_seconds,
                )

        if event is not None and getattr(event, "policy", None) is not None:
            return event.policy

        # Fallback based on event metadata if available
        if event is not None:
            if (
                event.zone_type in (SafetyZoneType.SWIMMING_POOL_BOUNDARY, SafetyZoneType.RESTRICTED_INFRASTRUCTURE)
                or event.severity == AlertSeverity.CRITICAL
            ):
                role = ZoneRole.DANGER
                exit_alert = True if event.zone_type == SafetyZoneType.SWIMMING_POOL_BOUNDARY else False
            elif event.zone_type == SafetyZoneType.GENERAL_COMMON_AREA and event.severity == AlertSeverity.LOW:
                role = ZoneRole.SAFE
                exit_alert = False
            else:
                role = ZoneRole.MONITORED
                exit_alert = False

            return ZoneSafetyPolicy.create_default(
                zone_role=role,
                severity=event.severity,
                dwell_threshold_seconds=event.dwell_time_seconds,
                exit_alert=exit_alert,
            )

        return ZoneSafetyPolicy.create_default(zone_role=ZoneRole.MONITORED)

    def process_events(
        self,
        safety_events: List[SafetyEvent],
        frame_id: int,
        timestamp_ms: float,
        roi_config: Optional[CameraROIConfig] = None,
    ) -> List[ChildSafetyEvent]:
        """Processes raw person safety events and generates policy-driven child-focused alerts.

        Args:
            safety_events: List of confirmed SafetyEvents from ZoneStateManager.
            frame_id: Current frame counter.
            timestamp_ms: Current stream elapsed timestamp in milliseconds.
            roi_config: Optional active CameraROIConfig for dynamic policy lookup.

        Returns:
            List of newly emitted ChildSafetyEvent objects for this frame.
        """
        child_events: List[ChildSafetyEvent] = []

        for evt in safety_events:
            tid = evt.track_id

            # Untracked person detections cannot be verified as children
            if tid is None:
                excluded = ExcludedTrackEvent(
                    event_id=evt.event_id,
                    track_id=None,
                    classification=TrackClassification.UNCLASSIFIED,
                    event_type=evt.event_type,
                    zone_name=evt.zone_name,
                    zone_type=evt.zone_type,
                    timestamp_ms=evt.timestamp_ms,
                    frame_id=evt.frame_id,
                    reason="Untracked person detection: excluded from child-specific safety alerts.",
                )
                self.excluded_events_history.append(excluded)
                logger.debug(f"Untracked detection excluded from child alerts: {evt.message}")
                continue

            classification = self.annotation_provider.get_classification(tid, timestamp_ms)

            # Adult tracks: preserve in underlying pipeline, but strictly exclude from child alerts
            if classification == TrackClassification.ADULT:
                excluded = ExcludedTrackEvent(
                    event_id=evt.event_id,
                    track_id=tid,
                    classification=TrackClassification.ADULT,
                    event_type=evt.event_type,
                    zone_name=evt.zone_name,
                    zone_type=evt.zone_type,
                    timestamp_ms=evt.timestamp_ms,
                    frame_id=evt.frame_id,
                    reason=f"Track ID {tid} is verified as ADULT: excluded from child-specific safety alerts.",
                )
                self.excluded_events_history.append(excluded)
                logger.info(f"Adult track ID {tid} excluded from child safety alerts in '{evt.zone_name}'")
                continue

            # Unclassified tracks: strictly exclude from child alerts in controlled testing
            if classification == TrackClassification.UNCLASSIFIED:
                excluded = ExcludedTrackEvent(
                    event_id=evt.event_id,
                    track_id=tid,
                    classification=TrackClassification.UNCLASSIFIED,
                    event_type=evt.event_type,
                    zone_name=evt.zone_name,
                    zone_type=evt.zone_type,
                    timestamp_ms=evt.timestamp_ms,
                    frame_id=evt.frame_id,
                    reason=f"Track ID {tid} is UNCLASSIFIED: excluded from child-specific safety alerts.",
                )
                self.excluded_events_history.append(excluded)
                logger.info(f"Unclassified track ID {tid} excluded from child safety alerts in '{evt.zone_name}'")
                continue

            # Verified CHILD track: evaluate against the configured ZoneSafetyPolicy
            if classification == TrackClassification.CHILD:
                policy = self.get_zone_policy(evt.roi_id, roi_config, event=evt)

                # Check policy rules for this event type
                if evt.event_type == SafetyEventType.ZONE_ENTRY:
                    self._last_seen_child_tracks[tid] = (evt.frame_id, evt.timestamp_ms, evt.roi_id)
                    if not policy.entry_alert:
                        # Entry alert disabled by policy (e.g. MONITORED or SAFE zone)
                        logger.info(
                            f"Child Track ID {tid} entered '{evt.zone_name}' (role={policy.zone_role.value}, "
                            "entry_alert=False; no danger alert generated)"
                        )
                        continue

                elif evt.event_type == SafetyEventType.DWELL_TIME_EXCEEDED:
                    if not policy.dwell_alert:
                        logger.info(
                            f"Child Track ID {tid} dwelling in '{evt.zone_name}' (role={policy.zone_role.value}, "
                            "dwell_alert=False; no dwell alert generated)"
                        )
                        continue

                elif evt.event_type == SafetyEventType.ZONE_EXIT:
                    self._last_seen_child_tracks.pop(tid, None)
                    if not policy.exit_alert:
                        logger.info(
                            f"Child Track ID {tid} exited '{evt.zone_name}' (role={policy.zone_role.value}, "
                            "exit_alert=False; no exit alert generated)"
                        )
                        continue

                # Policy permits this event -> Generate child-focused safety event
                child_event_type, msg = self._format_child_event(evt, tid, policy)

                child_evt = ChildSafetyEvent(
                    event_id=f"cevt_{uuid.uuid4().hex[:10]}",
                    event_type=child_event_type,
                    timestamp_ms=evt.timestamp_ms,
                    frame_id=evt.frame_id,
                    community_id=evt.community_id,
                    camera_id=evt.camera_id,
                    track_id=tid,
                    classification=TrackClassification.CHILD,
                    roi_id=evt.roi_id,
                    zone_name=evt.zone_name,
                    zone_type=evt.zone_type,
                    severity=policy.severity,
                    bottom_center=evt.bottom_center,
                    bbox_xyxy=evt.bbox_xyxy,
                    dwell_time_seconds=evt.dwell_time_seconds,
                    message=msg,
                    underlying_event_id=evt.event_id,
                    is_loss_of_tracking=getattr(evt, "is_loss_of_tracking", False),
                )
                child_events.append(child_evt)
                self.child_events_history.append(child_evt)

                logger.warning(
                    f"Generated {child_event_type.value}: {child_evt.message} [Severity: {policy.severity.value}]"
                )

        return child_events

    def record_discontinuity(
        self,
        frame_id: int,
        timestamp_ms: float,
        original_track_id: int,
        zone_id: Optional[str],
        description: str,
        potential_new_track_id: Optional[int] = None,
    ) -> TrackDiscontinuity:
        """Records an explicit track discontinuity without transferring labels."""
        discontinuity = TrackDiscontinuity(
            discontinuity_id=f"disc_{uuid.uuid4().hex[:8]}",
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            original_track_id=original_track_id,
            zone_id=zone_id,
            description=description,
            potential_new_track_id=potential_new_track_id,
        )
        self.discontinuities_history.append(discontinuity)
        logger.warning(f"Recorded track discontinuity: {description}")
        return discontinuity

    @staticmethod
    def _format_child_event(
        evt: SafetyEvent, track_id: int, policy: Optional[ZoneSafetyPolicy] = None
    ) -> Tuple[ChildSafetyEventType, str]:
        """Formats hazard-agnostic child safety event message based on zone, transition type, and policy."""
        sev = policy.severity if policy else evt.severity
        role = policy.zone_role if policy else ZoneRole.MONITORED

        if evt.event_type == SafetyEventType.ZONE_ENTRY:
            event_type = ChildSafetyEventType.CHILD_ZONE_ENTRY
            if role == ZoneRole.DANGER:
                msg = (
                    f"CHILD SAFETY ALERT: Child (Track ID {track_id}) entered configured danger zone "
                    f"'{evt.zone_name}' ({evt.zone_type.value}) [Severity: {sev.value}]"
                )
            elif role == ZoneRole.SAFE:
                msg = (
                    f"Child (Track ID {track_id}) entered designated safe area "
                    f"'{evt.zone_name}' ({evt.zone_type.value}) [Severity: {sev.value}]"
                )
            else:  # MONITORED
                msg = (
                    f"Child (Track ID {track_id}) entered monitored area "
                    f"'{evt.zone_name}' ({evt.zone_type.value}) [Severity: {sev.value}]"
                )
        elif evt.event_type == SafetyEventType.DWELL_TIME_EXCEEDED:
            event_type = ChildSafetyEventType.CHILD_DWELL_TIME_EXCEEDED
            if role == ZoneRole.DANGER:
                msg = (
                    f"CHILD DWELL ALERT: Child (Track ID {track_id}) remained in danger zone "
                    f"'{evt.zone_name}' beyond configured threshold ({evt.dwell_time_seconds:.1f}s) "
                    f"[Severity: {sev.value}]"
                )
            elif role == ZoneRole.MONITORED:
                msg = (
                    f"CHILD DWELL ALERT: Child (Track ID {track_id}) remained in monitored area "
                    f"'{evt.zone_name}' beyond configured threshold ({evt.dwell_time_seconds:.1f}s) "
                    f"[Severity: {sev.value}]"
                )
            else:  # SAFE
                msg = (
                    f"CHILD DWELL ALERT: Child (Track ID {track_id}) remained in safe area "
                    f"'{evt.zone_name}' beyond configured threshold ({evt.dwell_time_seconds:.1f}s) "
                    f"[Severity: {sev.value}]"
                )
        elif evt.event_type == SafetyEventType.ZONE_EXIT:
            event_type = ChildSafetyEventType.CHILD_ZONE_EXIT
            if role == ZoneRole.DANGER:
                msg = f"Child (Track ID {track_id}) exited configured danger zone '{evt.zone_name}' after {evt.dwell_time_seconds:.1f}s"
            elif role == ZoneRole.MONITORED:
                msg = f"Child (Track ID {track_id}) exited monitored area '{evt.zone_name}' after {evt.dwell_time_seconds:.1f}s"
            else:  # SAFE
                msg = f"Child (Track ID {track_id}) exited safe area '{evt.zone_name}' after {evt.dwell_time_seconds:.1f}s"
        else:
            event_type = ChildSafetyEventType.CHILD_ZONE_ENTRY
            msg = f"Child event for Track ID {track_id} in {evt.zone_name}"

        return event_type, msg

    def get_summary(self) -> Dict[str, Any]:
        """Returns consolidated audit statistics for child safety event evaluation."""
        child_entries = sum(1 for e in self.child_events_history if e.event_type == ChildSafetyEventType.CHILD_ZONE_ENTRY)
        child_dwells = sum(1 for e in self.child_events_history if e.event_type == ChildSafetyEventType.CHILD_DWELL_TIME_EXCEEDED)
        child_exits = sum(1 for e in self.child_events_history if e.event_type == ChildSafetyEventType.CHILD_ZONE_EXIT)

        adult_exclusions = sum(1 for e in self.excluded_events_history if e.classification == TrackClassification.ADULT)
        unclass_exclusions = sum(1 for e in self.excluded_events_history if e.classification == TrackClassification.UNCLASSIFIED)

        child_tracks = sorted(list(set(e.track_id for e in self.child_events_history)))
        excluded_adult_tracks = sorted(list(set(e.track_id for e in self.excluded_events_history if e.classification == TrackClassification.ADULT and e.track_id is not None)))
        excluded_unclass_tracks = sorted(list(set(e.track_id for e in self.excluded_events_history if e.classification == TrackClassification.UNCLASSIFIED and e.track_id is not None)))

        return {
            "total_child_events": len(self.child_events_history),
            "child_zone_entries": child_entries,
            "child_dwell_alerts": child_dwells,
            "child_zone_exits": child_exits,
            "active_child_track_ids": child_tracks,
            "total_excluded_events": len(self.excluded_events_history),
            "excluded_adult_events": adult_exclusions,
            "excluded_unclassified_events": unclass_exclusions,
            "excluded_adult_track_ids": excluded_adult_tracks,
            "excluded_unclassified_track_ids": excluded_unclass_tracks,
            "total_discontinuities_recorded": len(self.discontinuities_history),
        }

    def export_events_json(
        self,
        output_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Exports child-specific safety events and audit records to JSON."""
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "metadata": metadata or {},
            "summary": self.get_summary(),
            "child_events": [e.to_dict() for e in self.child_events_history],
            "excluded_events": [e.to_dict() for e in self.excluded_events_history],
            "discontinuities": [d.to_dict() for d in self.discontinuities_history],
            "annotations_used": {
                str(tid): ann.to_dict()
                for tid, ann in self.annotation_provider.list_annotations().items()
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Exported child safety events ({len(self.child_events_history)} alerts) to {path}")

