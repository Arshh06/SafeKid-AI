"""SafeKid AI: AI-powered child safety monitoring system specifically for gated residential communities."""

from safekid_ai.config import (
    AgeEstimatorConfig,
    DetectorConfig,
    OperationalMode,
    SafeKidPipelineConfig,
    SafetyZoneType,
    TrackerConfig,
    VideoInputConfig,
)
from safekid_ai.detector import DetectionBox, FrameDetections, PersonDetector
from safekid_ai.tracker import PersonTracker, TrackedPerson
from safekid_ai.age_estimator import (
    AgeEstimationResult,
    AgeEstimationStatus,
    MiVOLOAgeEstimator,
    classify_age_group,
)
from safekid_ai.video_input import VideoInput
from safekid_ai.roi import (
    AlertSeverity,
    CameraROIConfig,
    Point2D,
    Polygon,
    ROIEngine,
    SafetyZone,
    SafetyZoneType,
    ZoneRole,
    ZoneSafetyPolicy,
    calculate_bottom_center,
    load_camera_roi_config,
    save_camera_roi_config,
)
from safekid_ai.events import SafetyEvent, SafetyEventType, ZoneStateManager
from safekid_ai.child_safety import (
    ChildSafetyEvent,
    ChildSafetyEventEngine,
    ChildSafetyEventType,
    ChildTrackAnnotation,
    ExcludedTrackEvent,
    ManualTrackAnnotationProvider,
    TrackClassification,
    TrackClassificationProvider,
    TrackDiscontinuity,
)
from safekid_ai.incidents import (
    Incident,
    IncidentManager,
    IncidentPolicyConfig,
    IncidentState,
    ObservationStatus,
)
from safekid_ai.pipeline import SafeKidPipeline, SafeKidPersonEvent, SafeKidFrameResult

__version__ = "0.3.0"
__all__ = [
    "__version__",
    "SafeKidPipelineConfig",
    "OperationalMode",
    "SafetyZoneType",
    "AlertSeverity",
    "VideoInputConfig",
    "DetectorConfig",
    "TrackerConfig",
    "AgeEstimatorConfig",
    "PersonDetector",
    "DetectionBox",
    "FrameDetections",
    "PersonTracker",
    "TrackedPerson",
    "MiVOLOAgeEstimator",
    "AgeEstimationResult",
    "AgeEstimationStatus",
    "classify_age_group",
    "VideoInput",
    "Point2D",
    "Polygon",
    "SafetyZone",
    "ZoneRole",
    "ZoneSafetyPolicy",
    "CameraROIConfig",
    "ROIEngine",
    "calculate_bottom_center",
    "load_camera_roi_config",
    "save_camera_roi_config",
    "SafetyEvent",
    "SafetyEventType",
    "ZoneStateManager",
    "TrackClassification",
    "ChildSafetyEventType",
    "ChildTrackAnnotation",
    "TrackDiscontinuity",
    "ChildSafetyEvent",
    "TrackClassificationProvider",
    "ManualTrackAnnotationProvider",
    "ChildSafetyEventEngine",
    "ExcludedTrackEvent",
    "SafeKidPipeline",
    "SafeKidPersonEvent",
    "SafeKidFrameResult",
    "Incident",
    "IncidentState",
    "ObservationStatus",
    "IncidentManager",
    "IncidentPolicyConfig",
]
