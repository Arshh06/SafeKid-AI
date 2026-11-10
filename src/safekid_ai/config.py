"""Configuration classes and dataclasses for SafeKid AI."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class OperationalMode(str, Enum):
    """Operational mode for SafeKid AI pipeline."""
    COMMUNITY_MONITORING = "community_monitoring"  # Real-time safety monitoring in residential premises
    RESEARCH_BENCHMARK = "research_benchmark"     # Offline evaluation on consented benchmark datasets


class SafetyZoneType(str, Enum):
    """Safety zone definitions in a gated residential community."""
    PLAYGROUND = "playground"
    SWIMMING_POOL_GATE = "swimming_pool_gate"
    VEHICULAR_GATEWAY = "vehicular_gateway"
    PERIMETER_FENCE = "perimeter_fence"
    GENERAL_COMMON_AREA = "general_common_area"


@dataclass
class VideoInputConfig:
    """Configuration for video or image input handling."""
    source: str = "0"  # Camera index, video filepath, image filepath, or RTSP stream URL
    target_fps: Optional[int] = None  # None preserves native FPS
    frame_skip: int = 0  # Process every (frame_skip + 1)-th frame
    resize_dims: Optional[Tuple[int, int]] = None  # (width, height) or None
    loop_video: bool = False


@dataclass
class DetectorConfig:
    """Configuration for person detection."""
    backend: str = "yolov8"  # "yolov8", "rtdetrv2", "dfine"
    model_name_or_path: str = "yolov8n.pt"
    model_variant: str = "dfine_n"  # "dfine_n", "dfine_s" for D-FINE backend
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str = "cpu"  # "cpu", "cuda", etc.
    person_class_id: int = 0  # COCO person class is 0
    verbose: bool = False
    imgsz: int = 640
    fuse_model: bool = True
    num_threads: Optional[int] = 4
    frame_interval: int = 1  # 1 = detect every frame; 2, 3, 5 = skip intermediate frames
    min_box_area: float = 0.0
    max_box_area: Optional[float] = None
    enable_tiling: bool = False
    tile_size: Tuple[int, int] = (640, 640)
    tile_overlap: float = 0.20


@dataclass
class TrackerConfig:
    """Configuration for ByteTrack person tracking."""
    tracker_type: str = "bytetrack"
    tracker_config_path: str = "bytetrack.yaml"
    persist: bool = True
    track_history_length: int = 60  # Number of frames to retain for trajectory smoothing
    imgsz: int = 640
    confidence_threshold: float = 0.35
    track_high_thresh: float = 0.30
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.35
    track_buffer: int = 60
    match_thresh: float = 0.75
    frame_interval: int = 1  # Keyframe sampling interval
    max_track_staleness: int = 5  # Max consecutive frames a track may be extrapolated
    enable_velocity_extrapolation: bool = True  # Extrapolate track boxes on skipped frames


@dataclass
class AgeEstimatorConfig:
    """Configuration for the MiVOLO age estimation research module."""
    # Official pretrained checkpoint from the MiVOLO project
    checkpoint_repo_or_path: str = "typorch/mivolo-imdb_cross_person-yolov8x_person_face"
    checkpoint_filename: str = "model_imdb_cross_person_4.22_99.46.pth.tar"
    local_checkpoint_path: Optional[str] = None
    device: str = "cpu"
    half: bool = False  # Set to False on CPU for float32 execution
    disable_faces: bool = False
    use_persons: bool = True
    verbose: bool = False
    
    # Strict Governance / Ethics:
    # Must only be enabled when evaluating consented research images or labeled benchmark datasets.
    allow_research_inference: bool = False
    consent_verified: bool = False


@dataclass
class SafeKidPipelineConfig:
    """Unified configuration for the SafeKid AI gated community pipeline."""
    mode: OperationalMode = OperationalMode.COMMUNITY_MONITORING
    community_id: str = "community_001"
    community_name: str = "Greenfield Residential Enclave"
    camera_id: str = "camera_pool_01"
    zone_type: SafetyZoneType = SafetyZoneType.GENERAL_COMMON_AREA
    video: VideoInputConfig = field(default_factory=VideoInputConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    frame_interval: Optional[int] = None  # Unified keyframe sampling interval override
    age_estimator: AgeEstimatorConfig = field(default_factory=AgeEstimatorConfig)
    roi_config_path: Optional[str] = None
    output_dir: Path = field(default_factory=lambda: Path("runs/safekid"))
    entry_confirmation_frames: int = 3
    exit_confirmation_frames: int = 3
    child_annotation_path: Optional[str] = None
    enable_child_safety_layer: bool = True
    enable_incident_management: bool = True
    incident_policy: Optional[object] = None
    enable_automated_child_detection: bool = False
    child_detector_config: Optional[object] = None

