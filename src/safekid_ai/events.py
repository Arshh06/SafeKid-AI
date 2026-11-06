"""Structured safety events and zone state management for SafeKid AI."""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from safekid_ai.detector import DetectionBox
from safekid_ai.roi.engine import ROIEngine, calculate_bottom_center
from safekid_ai.roi.models import AlertSeverity, CameraROIConfig, SafetyZone, SafetyZoneType, ZoneSafetyPolicy
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.events")


class SafetyEventType(str, Enum):
    """Specific categories of safety events triggered by zone boundary interactions."""
    ZONE_ENTRY = "ZONE_ENTRY"
    ZONE_EXIT = "ZONE_EXIT"
    DWELL_TIME_EXCEEDED = "DWELL_TIME_EXCEEDED"


@dataclass
class SafetyEvent:
    """Structured representation of a safety-critical occurrence in a residential community."""
    event_id: str
    event_type: SafetyEventType
    timestamp_ms: float
    frame_id: int
    community_id: str
    camera_id: str
    track_id: Optional[int]
    roi_id: str
    zone_name: str
    zone_type: SafetyZoneType
    severity: AlertSeverity
    bottom_center: Tuple[float, float]
    bbox_xyxy: Tuple[float, float, float, float]
    dwell_time_seconds: float
    message: str
    policy: Optional[ZoneSafetyPolicy] = None
    is_loss_of_tracking: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Converts safety event to dictionary for logging, persistence, or telemetry dispatch."""
        data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp_ms": round(self.timestamp_ms, 2),
            "frame_id": self.frame_id,
            "community_id": self.community_id,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "roi_id": self.roi_id,
            "zone_name": self.zone_name,
            "zone_type": self.zone_type.value,
            "severity": self.severity.value,
            "bottom_center": [round(self.bottom_center[0], 2), round(self.bottom_center[1], 2)],
            "bbox_xyxy": [round(c, 2) for c in self.bbox_xyxy],
            "dwell_time_seconds": round(self.dwell_time_seconds, 2),
            "message": self.message,
            "is_loss_of_tracking": self.is_loss_of_tracking,
        }
        if self.policy is not None:
            data["policy"] = self.policy.to_dict()
        return data


@dataclass
class TrackZonePresence:
    """Internal state tracking a person's presence within safety zones."""
    track_id: int
    active_zones: Dict[str, float] = field(default_factory=dict)  # roi_id -> entry_timestamp_ms
    dwell_alerted: Dict[str, bool] = field(default_factory=dict)  # roi_id -> bool
    last_seen_frame: int = 0
    last_seen_timestamp_ms: float = 0.0
    last_bottom_center: Tuple[float, float] = (0.0, 0.0)
    last_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    # Transition confirmation tracking (zone-specific and track-specific)
    pending_entry_count: Dict[str, int] = field(default_factory=dict)
    pending_entry_first_ts: Dict[str, float] = field(default_factory=dict)
    pending_entry_first_frame: Dict[str, int] = field(default_factory=dict)
    pending_entry_bc: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    pending_entry_bbox: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)

    pending_exit_count: Dict[str, int] = field(default_factory=dict)
    pending_exit_first_ts: Dict[str, float] = field(default_factory=dict)
    pending_exit_first_frame: Dict[str, int] = field(default_factory=dict)
    pending_exit_bc: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    pending_exit_bbox: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)

    initial_frame_presence: Dict[str, bool] = field(default_factory=dict)


class ZoneStateManager:
    """Manages zone occupancy states, transition events, and dwell timers.

    Separates event generation from object detection and tracking.
    Guarantees:
    - Zone entry event is generated ONLY on confirmed boundary transition.
    - Transition confirmation suppresses boundary jitter from momentary 1-frame crossings.
    - Preserves timestamp of first qualifying observation upon confirmation.
    - Dwell time alert is triggered when threshold is exceeded without duplicate spam.
    - Zone exit event is generated after confirmed departure, accurately calculating dwell.
    - Temporary missed detections do not count as outside observations.
    - Handles missing or unassigned tracking IDs safely.
    """

    def __init__(
        self,
        max_missing_frames: int = 15,
        track_loss_grace_ms: float = 0.0,
        entry_confirmation_frames: int = 1,
        exit_confirmation_frames: int = 1,
    ):
        """Initializes ZoneStateManager.

        Args:
            max_missing_frames: Number of consecutive frames a track can be missing
                                before being considered to have left all active zones.
            track_loss_grace_ms: Optional duration grace period in milliseconds.
                                If > 0, tracks are retained until both frame and duration
                                thresholds are reached.
            entry_confirmation_frames: Number of consecutive inside observations required
                                      to confirm a zone entry (default: 1 for backward compat).
            exit_confirmation_frames: Number of consecutive outside observations required
                                     to confirm a zone exit (default: 1 for backward compat).
        """
        self.max_missing_frames = max_missing_frames
        self.track_loss_grace_ms = track_loss_grace_ms
        self.entry_confirmation_frames = max(1, entry_confirmation_frames)
        self.exit_confirmation_frames = max(1, exit_confirmation_frames)
        self._track_states: Dict[int, TrackZonePresence] = {}

    def reset(self) -> None:
        """Resets all active tracking presence states."""
        self._track_states.clear()
        logger.info("ZoneStateManager state reset.")

    def update_configuration(
        self,
        old_config: Optional[CameraROIConfig],
        new_config: CameraROIConfig,
        current_timestamp_ms: float = 0.0,
    ) -> List[SafetyEvent]:
        """Updates zone state non-destructively when camera configuration is reloaded.

        Requirements:
        - Reloading configuration does NOT unnecessarily reset dwell timers for occupants in unchanged zones.
        - Preserves state only when community ID, camera ID and zone identity remain compatible.
        - If a zone's geometry or relevant safety settings change, explicitly resets state for that specific zone.
        - Removes state for deleted/disabled zones safely and generates ZONE_EXIT.
        - Newly added zones will begin tracking occupants on the next frame naturally.
        - Never retains stale state across different cameras or communities.
        """
        if (
            old_config is None
            or old_config.community_id != new_config.community_id
            or old_config.camera_id != new_config.camera_id
        ):
            logger.info(
                f"Camera or community changed (old='{getattr(old_config, 'community_id', None)}:"
                f"{getattr(old_config, 'camera_id', None)}', new='{new_config.community_id}:"
                f"{new_config.camera_id}'); executing full state reset."
            )
            self.reset()
            return []

        old_zones: Dict[str, SafetyZone] = {z.roi_id: z for z in old_config.zones}
        new_zones: Dict[str, SafetyZone] = {z.roi_id: z for z in new_config.zones}

        events: List[SafetyEvent] = []

        # 1. Identify deleted or disabled zones
        deleted_zone_ids: Set[str] = set()
        for zid, old_z in old_zones.items():
            if zid not in new_zones or not new_zones[zid].enabled:
                deleted_zone_ids.add(zid)

        # 2. Identify modified zones (geometry or dwell threshold changed)
        modified_zone_ids: Set[str] = set()
        for zid, new_z in new_zones.items():
            if zid in old_zones:
                old_z = old_zones[zid]
                if old_z.polygon != new_z.polygon or old_z.dwell_time_seconds != new_z.dwell_time_seconds:
                    modified_zone_ids.add(zid)

        # 3. Apply non-destructive updates to active occupants
        for tid, presence in list(self._track_states.items()):
            # Handle deleted / disabled zones: safely remove and emit ZONE_EXIT
            for zid in deleted_zone_ids:
                if zid in presence.active_zones:
                    entry_ts = presence.active_zones.pop(zid, current_timestamp_ms)
                    presence.dwell_alerted.pop(zid, None)
                    dwell_sec = max(0.0, (current_timestamp_ms - entry_ts) / 1000.0)
                    old_z = old_zones.get(zid)
                    z_name = old_z.name if old_z else zid
                    z_type = old_z.zone_type if old_z else SafetyZoneType.GENERAL_COMMON_AREA
                    z_sev = old_z.severity if old_z else AlertSeverity.LOW

                    event = SafetyEvent(
                        event_id=f"evt_{uuid.uuid4().hex[:10]}",
                        event_type=SafetyEventType.ZONE_EXIT,
                        timestamp_ms=current_timestamp_ms,
                        frame_id=presence.last_seen_frame,
                        community_id=new_config.community_id,
                        camera_id=new_config.camera_id,
                        track_id=tid,
                        roi_id=zid,
                        zone_name=z_name,
                        zone_type=z_type,
                        severity=z_sev,
                        bottom_center=presence.last_bottom_center,
                        bbox_xyxy=presence.last_bbox,
                        dwell_time_seconds=dwell_sec,
                        message=f"Track ID {tid} exited {z_name} (zone removed or disabled in configuration reload)",
                        policy=old_z.policy if old_z else None,
                    )
                    events.append(event)
                    logger.info(f"Generated ZONE_EXIT on config reload: {event.message}")

            # Handle modified zones: reset occupant state for this specific zone so they are re-evaluated
            for zid in modified_zone_ids:
                if zid in presence.active_zones:
                    presence.active_zones.pop(zid, None)
                    presence.dwell_alerted.pop(zid, None)
                    logger.info(
                        f"Reset state in zone '{zid}' for Track ID {tid} due to geometry/threshold change."
                    )

        logger.info(
            f"Non-destructive configuration update applied for camera '{new_config.camera_id}'. "
            f"Preserved dwell tracking for {len(self._track_states)} active tracks in unchanged zones."
        )
        return events

    def get_track_zones(self, track_id: int) -> Set[str]:
        """Returns the set of active roi_ids currently occupied by a track."""
        if track_id in self._track_states:
            return set(self._track_states[track_id].active_zones.keys())
        return set()

    def get_dwell_time_seconds(self, track_id: int, roi_id: str, current_timestamp_ms: float) -> float:
        """Calculates current dwell time in seconds for a specific track and zone."""
        if track_id in self._track_states:
            entry_ts = self._track_states[track_id].active_zones.get(roi_id)
            if entry_ts is not None:
                return max(0.0, (current_timestamp_ms - entry_ts) / 1000.0)
        return 0.0

    def update(
        self,
        frame_id: int,
        timestamp_ms: float,
        detections: List[DetectionBox],
        roi_engine: ROIEngine,
        frame_resolution: Tuple[int, int],
    ) -> List[SafetyEvent]:
        """Updates zone occupancies and returns safety events for the current frame.

        Args:
            frame_id: Monotonic frame counter.
            timestamp_ms: Current stream timestamp in milliseconds.
            detections: List of detected/tracked person boxes in current frame.
            roi_engine: Initialized and calibrated ROIEngine.
            frame_resolution: (width, height) of the current frame.

        Returns:
            List of newly generated SafetyEvent objects for this frame.
        """
        new_events: List[SafetyEvent] = []
        community_id = roi_engine.community_id
        camera_id = roi_engine.camera_id

        # Track which track IDs were seen in this frame
        seen_track_ids: Set[int] = set()

        for det in detections:
            bbox = det.bbox_xyxy
            bottom_center = calculate_bottom_center(bbox, frame_resolution)
            occupied_zones = roi_engine.find_zones_for_point(bottom_center, frame_resolution)

            # Map matching zone IDs
            current_zone_map: Dict[str, SafetyZone] = {z.roi_id: z for z in occupied_zones}

            if det.track_id is None:
                # Handle untracked detection safely:
                # Emit instantaneous entry event for high-severity or pool zones without polluting state
                for z in occupied_zones:
                    event = SafetyEvent(
                        event_id=f"evt_{uuid.uuid4().hex[:10]}",
                        event_type=SafetyEventType.ZONE_ENTRY,
                        timestamp_ms=timestamp_ms,
                        frame_id=frame_id,
                        community_id=community_id,
                        camera_id=camera_id,
                        track_id=None,
                        roi_id=z.roi_id,
                        zone_name=z.name,
                        zone_type=z.zone_type,
                        severity=z.severity,
                        bottom_center=bottom_center,
                        bbox_xyxy=bbox,
                        dwell_time_seconds=0.0,
                        message=f"Untracked person entered {z.name} ({z.zone_type.value})",
                    )
                    new_events.append(event)
                continue

            tid = det.track_id
            seen_track_ids.add(tid)

            # Retrieve or create track state
            if tid not in self._track_states:
                presence = TrackZonePresence(
                    track_id=tid,
                    last_seen_frame=frame_id,
                    last_seen_timestamp_ms=timestamp_ms,
                    last_bottom_center=bottom_center,
                    last_bbox=bbox,
                )
                self._track_states[tid] = presence
            else:
                presence = self._track_states[tid]
                presence.last_seen_frame = frame_id
                presence.last_seen_timestamp_ms = timestamp_ms
                presence.last_bottom_center = bottom_center
                presence.last_bbox = bbox

            current_zone_ids = set(current_zone_map.keys())

            # 1. ZONE ENTRY EVALUATION: Transition confirmation for inside observations
            for zid, z in current_zone_map.items():
                if zid in presence.active_zones:
                    # Confirmed occupant inside: cancel pending exit if track returned before confirmation!
                    presence.pending_exit_count.pop(zid, None)
                    presence.pending_exit_first_ts.pop(zid, None)
                    presence.pending_exit_first_frame.pop(zid, None)
                    presence.pending_exit_bc.pop(zid, None)
                    presence.pending_exit_bbox.pop(zid, None)
                else:
                    # Unconfirmed inside observation
                    if zid not in presence.pending_entry_first_ts:
                        presence.pending_entry_first_ts[zid] = timestamp_ms
                        presence.pending_entry_first_frame[zid] = frame_id
                        presence.pending_entry_bc[zid] = bottom_center
                        presence.pending_entry_bbox[zid] = bbox
                        presence.pending_entry_count[zid] = 0

                    presence.pending_entry_count[zid] += 1

                    # Check entry confirmation (immediate for stream start frame 0)
                    is_initial = (frame_id == 0)
                    is_confirmed = (presence.pending_entry_count[zid] >= self.entry_confirmation_frames) or is_initial

                    if is_confirmed:
                        # Preserve the timestamp of the first qualifying observation!
                        entry_ts = presence.pending_entry_first_ts.pop(zid)
                        entry_fid = presence.pending_entry_first_frame.pop(zid)
                        entry_bc = presence.pending_entry_bc.pop(zid, bottom_center)
                        entry_box = presence.pending_entry_bbox.pop(zid, bbox)
                        presence.pending_entry_count.pop(zid, None)

                        presence.active_zones[zid] = entry_ts
                        presence.dwell_alerted[zid] = False

                        if is_initial:
                            presence.initial_frame_presence[zid] = True
                            msg = f"Track ID {tid} entered {z.name} ({z.zone_type.value}; presence observed at clip start)"
                        else:
                            msg = f"Track ID {tid} entered {z.name} ({z.zone_type.value})"

                        event = SafetyEvent(
                            event_id=f"evt_{uuid.uuid4().hex[:10]}",
                            event_type=SafetyEventType.ZONE_ENTRY,
                            timestamp_ms=entry_ts,
                            frame_id=entry_fid,
                            community_id=community_id,
                            camera_id=camera_id,
                            track_id=tid,
                            roi_id=z.roi_id,
                            zone_name=z.name,
                            zone_type=z.zone_type,
                            severity=z.severity,
                            bottom_center=entry_bc,
                            bbox_xyxy=entry_box,
                            dwell_time_seconds=0.0,
                            message=msg,
                            policy=z.policy,
                        )
                        new_events.append(event)
                        logger.info(f"Generated ZONE_ENTRY event: {event.message} [Severity: {z.severity.value}]")

            # Reset pending entries if track fluctuated outside before confirmation (boundary jitter)
            for zid in list(presence.pending_entry_count.keys()):
                if zid not in current_zone_ids:
                    presence.pending_entry_count.pop(zid, None)
                    presence.pending_entry_first_ts.pop(zid, None)
                    presence.pending_entry_first_frame.pop(zid, None)
                    presence.pending_entry_bc.pop(zid, None)
                    presence.pending_entry_bbox.pop(zid, None)

            # 2. DWELL TIME MONITORING: Remaining inside confirmed zone
            for zid in current_zone_ids & set(presence.active_zones.keys()):
                z = current_zone_map[zid]
                entry_ts = presence.active_zones[zid]
                dwell_sec = max(0.0, (timestamp_ms - entry_ts) / 1000.0)

                # Check dwell threshold if configured
                if z.dwell_time_seconds is not None and z.dwell_time_seconds > 0:
                    if dwell_sec >= z.dwell_time_seconds and not presence.dwell_alerted.get(zid, False):
                        presence.dwell_alerted[zid] = True
                        event = SafetyEvent(
                            event_id=f"evt_{uuid.uuid4().hex[:10]}",
                            event_type=SafetyEventType.DWELL_TIME_EXCEEDED,
                            timestamp_ms=timestamp_ms,
                            frame_id=frame_id,
                            community_id=community_id,
                            camera_id=camera_id,
                            track_id=tid,
                            roi_id=z.roi_id,
                            zone_name=z.name,
                            zone_type=z.zone_type,
                            severity=z.severity,
                            bottom_center=bottom_center,
                            bbox_xyxy=bbox,
                            dwell_time_seconds=dwell_sec,
                            message=(
                                f"Track ID {tid} dwell time ({dwell_sec:.1f}s) exceeded threshold "
                                f"({z.dwell_time_seconds:.1f}s) in {z.name}"
                            ),
                            policy=z.policy,
                        )
                        new_events.append(event)
                        logger.warning(f"Generated DWELL_TIME_EXCEEDED event: {event.message}")

            # 3. ZONE EXIT EVALUATION: Transition confirmation for outside observations
            for zid in list(presence.active_zones.keys()):
                if zid not in current_zone_ids:
                    if zid not in presence.pending_exit_first_ts:
                        presence.pending_exit_first_ts[zid] = timestamp_ms
                        presence.pending_exit_first_frame[zid] = frame_id
                        presence.pending_exit_bc[zid] = bottom_center
                        presence.pending_exit_bbox[zid] = bbox
                        presence.pending_exit_count[zid] = 0

                    presence.pending_exit_count[zid] += 1

                    # Check exit confirmation:
                    if presence.pending_exit_count[zid] >= self.exit_confirmation_frames:
                        exit_ts = presence.pending_exit_first_ts.pop(zid)
                        exit_fid = presence.pending_exit_first_frame.pop(zid)
                        exit_bc = presence.pending_exit_bc.pop(zid, bottom_center)
                        exit_box = presence.pending_exit_bbox.pop(zid, bbox)
                        presence.pending_exit_count.pop(zid, None)

                        entry_ts = presence.active_zones.pop(zid, exit_ts)
                        presence.dwell_alerted.pop(zid, None)
                        presence.initial_frame_presence.pop(zid, None)
                        dwell_sec = max(0.0, (exit_ts - entry_ts) / 1000.0)

                        # Find zone metadata from config
                        z_cfg = roi_engine.config.get_zone(zid) if roi_engine.config else None
                        z_name = z_cfg.name if z_cfg else zid
                        z_type = z_cfg.zone_type if z_cfg else SafetyZoneType.GENERAL_COMMON_AREA
                        z_sev = z_cfg.severity if z_cfg else AlertSeverity.LOW

                        event = SafetyEvent(
                            event_id=f"evt_{uuid.uuid4().hex[:10]}",
                            event_type=SafetyEventType.ZONE_EXIT,
                            timestamp_ms=exit_ts,
                            frame_id=exit_fid,
                            community_id=community_id,
                            camera_id=camera_id,
                            track_id=tid,
                            roi_id=zid,
                            zone_name=z_name,
                            zone_type=z_type,
                            severity=z_sev,
                            bottom_center=exit_bc,
                            bbox_xyxy=exit_box,
                            dwell_time_seconds=dwell_sec,
                            message=f"Track ID {tid} exited {z_name} after {dwell_sec:.1f}s",
                            policy=z_cfg.policy if z_cfg else None,
                            is_loss_of_tracking=False,
                        )
                        new_events.append(event)
                        logger.info(f"Generated ZONE_EXIT event: {event.message}")

        # 4. HANDLE DISAPPEARED TRACKS: Persons that completely left camera field of view
        # Respects configurable grace period (max_missing_frames and track_loss_grace_ms).
        # Temporary detection drops (< threshold) do NOT generate premature exits.
        stale_tids: List[int] = []
        for tid, presence in self._track_states.items():
            if tid not in seen_track_ids:
                frames_missing = frame_id - presence.last_seen_frame
                time_missing_ms = max(0.0, timestamp_ms - presence.last_seen_timestamp_ms)
                is_expired = frames_missing >= self.max_missing_frames
                if self.track_loss_grace_ms > 0.0:
                    is_expired = is_expired and (time_missing_ms >= self.track_loss_grace_ms)

                if is_expired:
                    # Emit exit events for all zones this disappearing track was in
                    for zid, entry_ts in list(presence.active_zones.items()):
                        dwell_sec = max(0.0, (presence.last_seen_timestamp_ms - entry_ts) / 1000.0)
                        z_cfg = roi_engine.config.get_zone(zid) if roi_engine.config else None
                        z_name = z_cfg.name if z_cfg else zid
                        z_type = z_cfg.zone_type if z_cfg else SafetyZoneType.GENERAL_COMMON_AREA
                        z_sev = z_cfg.severity if z_cfg else AlertSeverity.LOW

                        event = SafetyEvent(
                            event_id=f"evt_{uuid.uuid4().hex[:10]}",
                            event_type=SafetyEventType.ZONE_EXIT,
                            timestamp_ms=timestamp_ms,
                            frame_id=frame_id,
                            community_id=community_id,
                            camera_id=camera_id,
                            track_id=tid,
                            roi_id=zid,
                            zone_name=z_name,
                            zone_type=z_type,
                            severity=z_sev,
                            bottom_center=presence.last_bottom_center,
                            bbox_xyxy=presence.last_bbox,
                            dwell_time_seconds=dwell_sec,
                            message=f"Track ID {tid} left camera view (exited {z_name})",
                            policy=z_cfg.policy if z_cfg else None,
                            is_loss_of_tracking=True,
                        )
                        new_events.append(event)
                    stale_tids.append(tid)

        for tid in stale_tids:
            del self._track_states[tid]

        return new_events

