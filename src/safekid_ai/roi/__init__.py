"""Camera-specific configurable Region of Interest (ROI) engine package."""

from .engine import ROIEngine, calculate_bottom_center
from .loader import load_camera_roi_config, parse_roi_config_dict, save_camera_roi_config, validate_roi_config_dict
from .models import (
    AlertSeverity,
    CameraROIConfig,
    Point2D,
    Polygon,
    SafetyZone,
    SafetyZoneType,
    ZoneRole,
    ZoneSafetyPolicy,
)

__all__ = [
    "AlertSeverity",
    "SafetyZoneType",
    "ZoneRole",
    "ZoneSafetyPolicy",
    "Point2D",
    "Polygon",
    "SafetyZone",
    "CameraROIConfig",
    "ROIEngine",
    "calculate_bottom_center",
    "load_camera_roi_config",
    "save_camera_roi_config",
    "validate_roi_config_dict",
    "parse_roi_config_dict",
]

