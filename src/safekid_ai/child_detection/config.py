"""Configuration for the dedicated ChildDetector and tracking aggregator."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class ChildDetectorConfig:
    """Configurable thresholds and operational parameters for child detection."""
    # Classification confidence thresholds
    confidence_threshold: float = 0.65       # Minimum softmax probability to accept CHILD or ADULT
    margin_threshold: float = 0.20           # Minimum separation between top class and second class
    
    # Scale and aspect ratio validation
    min_crop_height: int = 36                # Minimum pixel height; smaller figures yield UNKNOWN (LOW_RESOLUTION)
    min_crop_width: int = 16                 # Minimum pixel width
    min_aspect_ratio: float = 1.0            # Minimum height/width ratio (standing pedestrian)
    max_aspect_ratio: float = 4.8            # Maximum height/width ratio (beyond this suggests vertical slice/occlusion)
    
    # Model architecture & input sizing
    input_size: Tuple[int, int] = (128, 64)  # (height, width) for pedestrian crop normalization
    model_weights_path: Optional[str] = "models/child_classifier_v1.pt"
    device: str = "cpu"                      # "cpu" optimized for Intel i3, "cuda" if GPU
    
    # Temporal track aggregation & smoothing
    temporal_window_size: int = 15           # Number of frames to maintain in sliding history
    temporal_confirmation_frames: int = 3    # Frames of consistent evidence required to promote to CHILD
    temporal_decay_frames: int = 10          # Uncertain frames before a track reverts to UNKNOWN
    smoothing_alpha: float = 0.35            # Exponential smoothing factor for class probabilities
    
    # Consent and residential safety governance
    require_parental_consent: bool = True    # Strictly enforce consent policy for residential child alert streams
    provisional_review_required: bool = True # Flag candidate child alerts as provisional until reviewed

