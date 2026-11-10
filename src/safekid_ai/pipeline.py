"""Unified SafeKid AI pipeline for gated residential communities."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np

from safekid_ai.age_estimator import (
    AgeEstimationResult,
    AgeEstimationStatus,
    MiVOLOAgeEstimator,
)
from safekid_ai.config import (
    OperationalMode,
    SafeKidPipelineConfig,
    SafetyZoneType,
)
from safekid_ai.detector import DetectionBox, FrameDetections, PersonDetector
from safekid_ai.events import SafetyEvent, SafetyEventType, ZoneStateManager
from safekid_ai.child_safety import (
    ChildSafetyEvent,
    ChildSafetyEventEngine,
    ManualTrackAnnotationProvider,
    TrackClassification,
    TrackClassificationProvider,
)
from safekid_ai.incidents import Incident, IncidentManager, IncidentPolicyConfig
from safekid_ai.roi.engine import ROIEngine, calculate_bottom_center
from safekid_ai.roi.models import AlertSeverity, CameraROIConfig
from safekid_ai.tracker import PersonTracker, TrackedPerson
from safekid_ai.utils.logger import get_logger
from safekid_ai.video_input import VideoInput
from safekid_ai.visualization import draw_person_and_events, draw_roi_polygons

logger = get_logger("safekid_ai.pipeline")


@dataclass
class SafeKidPersonEvent:
    """Individual person tracking and zone occupancy status within a residential community."""
    track_id: Optional[int]
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    zone_type: SafetyZoneType
    age_estimation: AgeEstimationResult
    velocity: Tuple[float, float] = (0.0, 0.0)
    is_hazard_alert: bool = False
    alert_reason: str = ""
    bottom_center: Tuple[float, float] = (0.0, 0.0)
    occupied_zones: List[str] = field(default_factory=list)
    dwell_times: Dict[str, float] = field(default_factory=dict)


@dataclass
class SafeKidFrameResult:
    """Consolidated output for a single processed video or stream frame."""
    frame_id: int
    timestamp_ms: float
    community_name: str
    camera_id: str
    operational_mode: OperationalMode
    events: List[SafetyEvent] = field(default_factory=list)
    child_events: List[ChildSafetyEvent] = field(default_factory=list)
    person_states: List[SafeKidPersonEvent] = field(default_factory=list)
    zone_type: Optional[SafetyZoneType] = None
    community_id: str = ""
    annotated_frame: Optional[np.ndarray] = None
    incidents: List[Incident] = field(default_factory=list)

    @property
    def safety_events(self) -> List[SafetyEvent]:
        """Convenience property for accessing generated safety events."""
        return self.events


class SafeKidPipeline:
    """Unified safety monitoring pipeline specifically designed for gated residential communities."""

    def __init__(self, config: Optional[SafeKidPipelineConfig] = None):
        self.config = config or SafeKidPipelineConfig()
        logger.info(
            f"Initializing SafeKid AI Pipeline for community '{self.config.community_name}', "
            f"Camera: {self.config.camera_id}, Mode: {self.config.mode.value}"
        )

        # Propagate unified frame_interval override if specified
        if getattr(self.config, "frame_interval", None) is not None:
            self.config.detector.frame_interval = self.config.frame_interval
            self.config.tracker.frame_interval = self.config.frame_interval

        # 1. Initialize Person Detector & Tracker
        backend = getattr(self.config.detector, "backend", "yolov8").lower()
        if backend in ("dfine", "dfine_n", "dfine_s", "dfine-n", "dfine-s"):
            from safekid_ai.detectors import create_detector
            self.detector_adapter = create_detector(self.config.detector, self.config.tracker)
            self.detector = self.detector_adapter
            self.tracker = self.detector_adapter.tracker
        elif backend in ("rtdetr", "rtdetrv2", "transformer"):
            from safekid_ai.detectors.rtdetrv2_detector import RTDETRv2DetectorAdapter
            ckpt = self.config.detector.model_name_or_path
            if not ckpt or ckpt.startswith("yolo"):
                ckpt = "rtdetr-l.pt"
            self.detector_adapter = RTDETRv2DetectorAdapter(
                config=self.config.detector,
                tracker_config=self.config.tracker,
                checkpoint_path=ckpt,
            )
            self.detector = self.detector_adapter
            self.tracker = self.detector_adapter.tracker
        else:
            self.detector = PersonDetector(self.config.detector)
            self.tracker = PersonTracker(self.detector.model, self.config.tracker)

        # 3. Initialize MiVOLO Age Estimator (Research Module Only)
        self.age_estimator = MiVOLOAgeEstimator(self.config.age_estimator)

        # 4. Initialize Configurable Camera-Specific ROI Engine
        self.roi_engine = ROIEngine(
            config=getattr(self.config, "roi_config", None),
            config_path=self.config.roi_config_path,
        )

        # 5. Initialize Zone State & Event Manager with pipeline-configured transition confirmation
        self.zone_manager = ZoneStateManager(
            entry_confirmation_frames=getattr(self.config, "entry_confirmation_frames", 3),
            exit_confirmation_frames=getattr(self.config, "exit_confirmation_frames", 3),
        )

        # 6. Initialize Generalized Child-Focused Safety Event Layer
        self.child_annotation_provider: Optional[TrackClassificationProvider] = None
        if getattr(self.config, "child_annotation_path", None):
            self.child_annotation_provider = ManualTrackAnnotationProvider.load_from_json(
                self.config.child_annotation_path
            )
        elif getattr(self.config, "enable_automated_child_detection", False):
            from safekid_ai.child_detection import AutomatedChildClassificationProvider, ChildDetectorConfig
            det_cfg = getattr(self.config, "child_detector_config", None)
            if isinstance(det_cfg, dict):
                det_cfg = ChildDetectorConfig(**det_cfg)
            self.child_annotation_provider = AutomatedChildClassificationProvider(
                config=det_cfg,
                community_id=self.config.community_id,
                camera_id=self.config.camera_id,
            )
        self.child_safety_engine = ChildSafetyEventEngine(
            annotation_provider=self.child_annotation_provider,
            roi_config=self.roi_engine.config,
        )

        # 7. Initialize Alert Lifecycle & Incident Correlation Manager
        self.incident_manager: Optional[IncidentManager] = None
        if getattr(self.config, "enable_incident_management", True):
            inc_cfg = getattr(self.config, "incident_policy", None)
            if isinstance(inc_cfg, dict):
                inc_cfg = IncidentPolicyConfig(**inc_cfg)
            self.incident_manager = IncidentManager(config=inc_cfg)

        self._last_timestamp_ms: float = 0.0
        self._active_alerts: List[Tuple[float, str, AlertSeverity]] = []

    def set_child_annotation_provider(self, provider: TrackClassificationProvider) -> None:
        """Sets or replaces the track classification provider for child-focused events."""
        self.child_annotation_provider = provider
        self.child_safety_engine.set_annotation_provider(provider)

    def reload_roi_config(self, json_path: Optional[Union[str, Path]] = None) -> CameraROIConfig:
        """Dynamically reloads camera-specific ROI configuration non-destructively.

        Preserves active dwell timers for occupants in unchanged zones while safely
        handling modified, deleted, and newly added zones.
        """
        old_config = self.roi_engine.config
        if json_path is not None:
            config = self.roi_engine.load_config(json_path)
        else:
            config = self.roi_engine.reload_config()
        self.zone_manager.update_configuration(
            old_config=old_config,
            new_config=config,
            current_timestamp_ms=self._last_timestamp_ms,
        )
        if self.child_safety_engine:
            self.child_safety_engine.set_roi_config(config)
        logger.info(f"Non-destructively reloaded ROI configuration for camera '{config.camera_id}'.")
        return config

    def set_roi_config(self, config: CameraROIConfig) -> None:
        """Sets a new CameraROIConfig in memory and updates zone state non-destructively."""
        old_config = self.roi_engine.config
        self.roi_engine.set_config(config)
        self.zone_manager.update_configuration(
            old_config=old_config,
            new_config=config,
            current_timestamp_ms=self._last_timestamp_ms,
        )
        if self.child_safety_engine:
            self.child_safety_engine.set_roi_config(config)

    def handle_camera_disconnect(self, timestamp_ms: Optional[float] = None, frame_id: int = 0) -> List[Incident]:
        """Handles camera stream disconnection or signal loss.

        Ensures active incidents are preserved with observation_status = DISCONNECTED,
        and never falsely resolved merely because frames stop arriving.
        """
        ts = timestamp_ms if timestamp_ms is not None else self._last_timestamp_ms
        if self.incident_manager:
            return self.incident_manager.handle_camera_disconnect(
                camera_id=self.config.camera_id,
                timestamp_ms=ts,
                frame_id=frame_id,
            )
        return []

    def handle_camera_reconnect(self, timestamp_ms: Optional[float] = None, frame_id: int = 0) -> List[Incident]:
        """Handles camera stream restoration.

        Ensures active incidents are not erased and continue monitoring for the tracked person.
        """
        ts = timestamp_ms if timestamp_ms is not None else self._last_timestamp_ms
        if self.incident_manager:
            return self.incident_manager.handle_camera_reconnect(
                camera_id=self.config.camera_id,
                timestamp_ms=ts,
                frame_id=frame_id,
            )
        return []

    def reset(self) -> None:
        """Resets all internal pipeline states for a clean video session."""
        self._last_timestamp_ms = 0.0
        self.tracker.reset()
        if self.child_annotation_provider and hasattr(self.child_annotation_provider, "reset"):
            self.child_annotation_provider.reset()
        self.zone_manager.reset()
        if self.child_safety_engine:
            self.child_safety_engine.reset()
        if self.incident_manager:
            self.incident_manager.reset()
        logger.info("SafeKidPipeline internal session state reset.")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp_ms: float = 0.0,
        annotate: bool = True,
    ) -> SafeKidFrameResult:
        """Processes a single frame through detection, tracking, ROI engine, safety events, and visualization.

        Args:
            frame: OpenCV BGR image array.
            frame_id: Frame sequence index.
            timestamp_ms: Milliseconds elapsed in stream.
            annotate: Whether to render visual overlays.

        Returns:
            SafeKidFrameResult containing active safety events, person states, and annotated frame.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Provided frame is empty or invalid.")

        # Enforce strict timestamp monotonicity within a pipeline stream session
        if timestamp_ms < self._last_timestamp_ms:
            logger.warning(
                f"Non-monotonic timestamp detected: {timestamp_ms:.2f}ms < last {self._last_timestamp_ms:.2f}ms. "
                f"Clamping to {self._last_timestamp_ms:.2f}ms to prevent negative dwell durations."
            )
            timestamp_ms = self._last_timestamp_ms
        else:
            self._last_timestamp_ms = timestamp_ms

        h, w = frame.shape[:2]

        # 1. Person Detection & Tracking via ByteTrack
        detections: FrameDetections = self.tracker.track(
            frame=frame,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            confidence_thresh=self.config.detector.confidence_threshold,
            iou_thresh=self.config.detector.iou_threshold,
        )

        # 1b. Dynamic Child Detection & Track Aggregation (if provider is automated)
        if self.child_annotation_provider is not None and hasattr(self.child_annotation_provider, "process_frame"):
            self.child_annotation_provider.process_frame(
                frame=frame,
                detections=detections.detections,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
            )

        # 2. ROI Engine Evaluation & Safety Event Generation
        # Only actual zone boundary transitions or dwell thresholds emit events
        safety_events: List[SafetyEvent] = self.zone_manager.update(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            detections=detections.detections,
            roi_engine=self.roi_engine,
            frame_resolution=(w, h),
        )

        # 3. Generalized Child-Focused Safety Event Processing
        child_events: List[ChildSafetyEvent] = []
        if getattr(self.config, "enable_child_safety_layer", True) and self.child_safety_engine:
            child_events = self.child_safety_engine.process_events(
                safety_events=safety_events,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                roi_config=self.roi_engine.config,
            )

        # 3b. Alert Lifecycle Management & Incident Correlation
        active_incidents: List[Incident] = []
        if self.incident_manager:
            active_incidents = self.incident_manager.process_frame(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                child_events=child_events,
                raw_safety_events=safety_events,
                roi_config=self.roi_engine.config,
                active_track_ids={det.track_id for det in detections.detections if det.track_id is not None},
            )

        person_states: List[SafeKidPersonEvent] = []
        is_research = self.config.mode == OperationalMode.RESEARCH_BENCHMARK
        occupied_zones_per_track: Dict[Optional[int], List[str]] = {}
        dwell_times_per_track: Dict[Optional[int], Dict[str, float]] = {}
        track_classifications: Dict[int, str] = {}

        for det in detections.detections:
            tid = det.track_id
            if tid is not None and self.child_safety_engine and self.child_safety_engine.annotation_provider:
                cls_val = self.child_safety_engine.annotation_provider.get_classification(tid, timestamp_ms)
                track_classifications[tid] = cls_val.value

            bottom_center = calculate_bottom_center(det.bbox_xyxy, (w, h))
            matching_zones = self.roi_engine.find_zones_for_point(bottom_center, (w, h))
            zone_names = [z.name for z in matching_zones]
            occupied_zones_per_track[tid] = zone_names

            # Dwell times
            dwell_map: Dict[str, float] = {}
            if tid is not None:
                for z in matching_zones:
                    dwell_map[z.roi_id] = self.zone_manager.get_dwell_time_seconds(tid, z.roi_id, timestamp_ms)
            dwell_times_per_track[tid] = dwell_map

            # Age estimation execution:
            # - In operational monitoring mode, strictly preserves 'UNKNOWN' / restricted status
            # - In research benchmark mode, runs official MiVOLO if consent is verified
            if is_research and self.age_estimator.is_available:
                person_crop = det.extract_crop(frame)
                age_res = self.age_estimator.estimate(
                    face_crop=None,
                    person_crop=person_crop,
                    consent_override=self.config.age_estimator.consent_verified,
                    research_mode_override=True,
                )
            else:
                age_res = AgeEstimationResult(
                    status=AgeEstimationStatus.RESTRICTED_OPERATIONAL_MODE,
                    estimated_age=None,
                    age_group="UNKNOWN",
                    reason=(
                        "Age estimation disabled in operational residential monitoring mode. "
                        "Per community ethics policy, age estimation is restricted to consented research datasets."
                    ),
                )

            # Retrieve tracking dynamics
            vel = (0.0, 0.0)
            if tid is not None and tid in self.tracker.active_tracks:
                vel = self.tracker.active_tracks[tid].velocity_vector

            # Check if this person triggered an alert in this frame
            has_alert = any(
                e.track_id == tid and e.event_type in (SafetyEventType.ZONE_ENTRY, SafetyEventType.DWELL_TIME_EXCEEDED)
                for e in safety_events
            )
            alert_reason = "; ".join([e.message for e in safety_events if e.track_id == tid])

            # Fallback zone type representation
            primary_zone_type = matching_zones[0].zone_type if matching_zones else self.config.zone_type

            person_event = SafeKidPersonEvent(
                track_id=tid,
                bbox_xyxy=det.bbox_xyxy,
                confidence=det.confidence,
                zone_type=primary_zone_type,
                age_estimation=age_res,
                velocity=vel,
                is_hazard_alert=has_alert,
                alert_reason=alert_reason,
                bottom_center=bottom_center,
                occupied_zones=zone_names,
                dwell_times=dwell_map,
            )
            person_states.append(person_event)

        # Prune expired alert banners and register new safety alerts (2.5s duration)
        self._active_alerts = [(exp, msg, sev) for exp, msg, sev in self._active_alerts if exp > timestamp_ms]
        for ce in child_events:
            self._active_alerts.append((timestamp_ms + 2500.0, f"CHILD ALERT: {ce.message}", ce.severity))
        for e in safety_events:
            # If not already covered by child alert
            if not any(ce.underlying_event_id == e.event_id for ce in child_events):
                self._active_alerts.append((timestamp_ms + 2500.0, f"{e.event_type.value}: {e.message}", e.severity))

        # 4. Visual Annotations (ROIs, Detections, Ground Contact, Dwell Times & Alerts)
        annotated: Optional[np.ndarray] = None
        if annotate:
            calibrated_zones = self.roi_engine.calibrate_for_resolution(w, h)
            vis = draw_roi_polygons(frame, calibrated_zones, alpha=0.22, detections=detections.detections)
            active_alert_items = [(msg, sev) for exp, msg, sev in self._active_alerts]
            annotated = draw_person_and_events(
                frame=vis,
                detections=detections.detections,
                occupied_zones_per_track=occupied_zones_per_track,
                dwell_times_per_track=dwell_times_per_track,
                events=safety_events,
                community_name=self.config.community_name,
                camera_id=self.config.camera_id,
                active_alerts=active_alert_items,
                track_classifications=track_classifications,
                child_events=child_events,
                incidents=active_incidents,
                copy=False,
            )

        return SafeKidFrameResult(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            community_name=self.config.community_name,
            camera_id=self.config.camera_id,
            operational_mode=self.config.mode,
            events=safety_events,
            child_events=child_events,
            person_states=person_states,
            zone_type=self.config.zone_type,
            community_id=self.config.community_id,
            annotated_frame=annotated,
            incidents=active_incidents,
        )

    def process_video(
        self,
        video_input: VideoInput,
        max_frames: Optional[int] = None,
        annotate: bool = True,
    ) -> Generator[SafeKidFrameResult, None, None]:
        """Runs the pipeline over a video or stream source."""
        for frame_id, ts_ms, frame in video_input.frames():
            if max_frames is not None and frame_id >= max_frames:
                break
            yield self.process_frame(frame, frame_id=frame_id, timestamp_ms=ts_ms, annotate=annotate)
