"""Data models and representations for the dedicated ChildDetector subsystem."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class ChildDetectionLabel(str, Enum):
    """Core child detection classification labels."""
    CHILD = "CHILD"
    ADULT = "ADULT"
    UNKNOWN = "UNKNOWN"


class AbstentionReason(str, Enum):
    """Explicit rationale when classification abstains and yields UNKNOWN."""
    NONE = "NONE"
    LOW_RESOLUTION = "LOW_RESOLUTION"              # Pedestrian too small or distant
    LOW_CONFIDENCE = "LOW_CONFIDENCE"              # Below minimum prediction threshold
    AMBIGUOUS_MARGIN = "AMBIGUOUS_MARGIN"          # P(Child) and P(Adult) too close to call
    SEVERE_OCCLUSION = "SEVERE_OCCLUSION"          # Bounding box or aspect ratio indicates partial body
    DEGRADED_CROP = "DEGRADED_CROP"                # Blurred, invalid, or empty crop
    TEMPORAL_DECAY = "TEMPORAL_DECAY"              # Track lost evidence over time
    UNINITIALIZED = "UNINITIALIZED"                # New track without sufficient frames
    SCALE_MISMATCH = "SCALE_MISMATCH"              # Height/scale inconsistent with camera perspective
    PERSPECTIVE_ANOMALY = "PERSPECTIVE_ANOMALY"    # Impossible physical scale relative to ground plane


class VerificationStatus(str, Enum):
    """Governance and consent status for child safety intelligence."""
    PROVISIONAL_MODEL_PREDICTION = "PROVISIONAL_MODEL_PREDICTION"  # Heuristic model candidate
    VERIFIED_CHILD_PROFILE = "VERIFIED_CHILD_PROFILE"              # Enrolled with verified parental consent
    UNVERIFIED = "UNVERIFIED"                                      # General unverified pedestrian


@dataclass
class ChildClassificationResult:
    """Structured output of the dedicated child detection subsystem."""
    label: ChildDetectionLabel
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=lambda: {"CHILD": 0.0, "ADULT": 0.0, "UNKNOWN": 1.0})
    abstention_reason: AbstentionReason = AbstentionReason.NONE
    verification_status: VerificationStatus = VerificationStatus.PROVISIONAL_MODEL_PREDICTION
    features: Dict[str, float] = field(default_factory=dict)
    inference_time_ms: float = 0.0
    track_id: Optional[int] = None
    frame_id: Optional[int] = None

    @property
    def is_child(self) -> bool:
        """Convenience property returning True if confidently classified as CHILD."""
        return self.label == ChildDetectionLabel.CHILD

    @property
    def is_adult(self) -> bool:
        """Convenience property returning True if confidently classified as ADULT."""
        return self.label == ChildDetectionLabel.ADULT

    @property
    def is_unknown(self) -> bool:
        """Convenience property returning True if classification abstained."""
        return self.label == ChildDetectionLabel.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result to a dictionary."""
        return {
            "label": self.label.value,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "abstention_reason": self.abstention_reason.value,
            "verification_status": self.verification_status.value,
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "inference_time_ms": round(self.inference_time_ms, 2),
            "track_id": self.track_id,
            "frame_id": self.frame_id,
        }

