"""VIRAT dataset integration and benchmarking package for SafeKid AI."""

from safekid_ai.datasets.jaad_parser import (
    JAADAnnotationParser,
    JAADBoundingBox,
    JAADValidationReport,
)
from safekid_ai.datasets.virat_annotations import (
    VIRATAnnotationParser,
    VIRATEvent,
    VIRATFrameAnnotation,
    VIRATObjectDetection,
    VIRATObjectType,
)
from safekid_ai.datasets.virat_evaluator import (
    VIRATDetectionEvaluator,
    VIRATDetectionMetrics,
    VIRATTrackingEvaluator,
    VIRATTrackingMetrics,
    calculate_iou,
)
from safekid_ai.datasets.virat_loader import (
    VIRATDatasetLoader,
    VIRATSequence,
    VIRATSequenceMetadata,
)
from safekid_ai.datasets.virat_validator import (
    SequenceValidationResult,
    VIRATDatasetValidator,
)

__all__ = [
    "JAADAnnotationParser",
    "JAADBoundingBox",
    "JAADValidationReport",
    "VIRATObjectType",
    "VIRATObjectDetection",
    "VIRATEvent",
    "VIRATFrameAnnotation",
    "VIRATAnnotationParser",
    "calculate_iou",
    "VIRATDetectionMetrics",
    "VIRATDetectionEvaluator",
    "VIRATTrackingMetrics",
    "VIRATTrackingEvaluator",
    "VIRATSequenceMetadata",
    "VIRATSequence",
    "VIRATDatasetLoader",
    "SequenceValidationResult",
    "VIRATDatasetValidator",
]

