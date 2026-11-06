"""SafeKid AI Dedicated Child Detection & Safety Intelligence Subsystem."""

from safekid_ai.child_detection.classifier_net import (
    ChildAdultClassifierNet,
    preprocess_crop,
)
from safekid_ai.child_detection.config import ChildDetectorConfig
from safekid_ai.child_detection.detector import ChildDetector
from safekid_ai.child_detection.models import (
    AbstentionReason,
    ChildClassificationResult,
    ChildDetectionLabel,
    VerificationStatus,
)
from safekid_ai.child_detection.provider import AutomatedChildClassificationProvider
from safekid_ai.child_detection.tracker_integration import (
    TrackChildClassifier,
    TrackClassificationState,
)

__all__ = [
    "AbstentionReason",
    "AutomatedChildClassificationProvider",
    "ChildAdultClassifierNet",
    "ChildClassificationResult",
    "ChildDetectionLabel",
    "ChildDetector",
    "ChildDetectorConfig",
    "TrackChildClassifier",
    "TrackClassificationState",
    "VerificationStatus",
    "preprocess_crop",
]

