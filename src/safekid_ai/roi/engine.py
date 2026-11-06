"""Runtime Region of Interest (ROI) engine for camera-specific zone evaluation."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from safekid_ai.roi.loader import load_camera_roi_config
from safekid_ai.roi.models import (
    AlertSeverity,
    CameraROIConfig,
    Point2D,
    Polygon,
    SafetyZone,
    SafetyZoneType,
)
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.roi.engine")


def calculate_bottom_center(
    bbox_xyxy: Tuple[float, float, float, float],
    frame_resolution: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float]:
    """Calculates bottom-centre ground contact point from a bounding box.

    Design Rule: Use bottom-centre (feet contact) rather than bounding box overlap
    to prevent false positives when a person's upper body or head protrudes over a boundary.

    Args:
        bbox_xyxy: (x1, y1, x2, y2) coordinates.
        frame_resolution: Optional (width, height) to clamp point within valid frame bounds.

    Returns:
        (x_bottom_center, y_bottom_center) as (float, float).
    """
    x1, y1, x2, y2 = bbox_xyxy
    x_bc = (x1 + x2) / 2.0
    y_bc = float(y2)
    if frame_resolution is not None:
        w, h = frame_resolution
        if w > 0 and h > 0:
            x_bc = max(0.0, min(float(w - 1), x_bc))
            y_bc = max(0.0, min(float(h - 1), y_bc))
    return (x_bc, y_bc)


class ROIEngine:
    """Configurable, camera-specific Region of Interest (ROI) engine.

    Manages independent zone configurations per camera, resolution calibration,
    point-in-polygon queries, and dynamic configuration reloads.
    """

    def __init__(
        self,
        config: Optional[CameraROIConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """Initializes ROIEngine with either a pre-built config or path to JSON."""
        self._config_path: Optional[Path] = Path(config_path) if config_path else None
        self._config: Optional[CameraROIConfig] = None
        # Cache calibrated zones per resolution: (width, height) -> List[SafetyZone]
        self._calibrated_cache: Dict[Tuple[int, int], List[SafetyZone]] = {}

        if config is not None:
            self._config = config
        elif self._config_path is not None:
            self.load_config(self._config_path)

    @property
    def config(self) -> Optional[CameraROIConfig]:
        """Currently active camera ROI configuration."""
        return self._config

    @property
    def camera_id(self) -> str:
        return self._config.camera_id if self._config else "unknown_camera"

    @property
    def community_id(self) -> str:
        return self._config.community_id if self._config else "unknown_community"

    @property
    def num_zones(self) -> int:
        return self._config.num_zones if self._config else 0

    @property
    def num_active_zones(self) -> int:
        return self._config.num_active_zones if self._config else 0

    def load_config(self, config_path: Union[str, Path]) -> CameraROIConfig:
        """Loads camera configuration from an external JSON file."""
        path = Path(config_path)
        self._config = load_camera_roi_config(path)
        self._config_path = path
        self._calibrated_cache.clear()
        logger.info(f"ROIEngine loaded configuration for camera '{self.camera_id}' from: {path}")
        return self._config

    def reload_config(self) -> CameraROIConfig:
        """Dynamically reloads the configuration from disk without restarting the pipeline."""
        if self._config_path is None:
            raise ValueError("Cannot reload configuration: no configuration filepath was provided.")
        logger.info(f"Reloading camera ROI configuration from: {self._config_path}")
        return self.load_config(self._config_path)

    def set_config(self, config: CameraROIConfig) -> None:
        """Sets active camera configuration programmatically."""
        self._config = config
        self._calibrated_cache.clear()
        logger.info(f"ROIEngine set new in-memory configuration for camera '{config.camera_id}'.")

    def calibrate_for_resolution(self, width: int, height: int) -> List[SafetyZone]:
        """Returns active safety zones scaled to the current camera stream resolution.

        Args:
            width: Current video frame width in pixels.
            height: Current video frame height in pixels.

        Returns:
            List of calibrated active SafetyZone instances.
        """
        if self._config is None:
            return []

        res_key = (width, height)
        if res_key in self._calibrated_cache:
            return self._calibrated_cache[res_key]

        ref_w = self._config.frame_width
        ref_h = self._config.frame_height

        active_zones = self._config.get_active_zones()
        if width == ref_w and height == ref_h and not any(z.is_normalized for z in active_zones):
            calibrated = active_zones
        else:
            logger.debug(f"Calibrating ROI coordinates from reference {ref_w}x{ref_h} to runtime {width}x{height}")
            calibrated = [
                z.scale_to_resolution(
                    target_width=width,
                    target_height=height,
                    ref_width=ref_w,
                    ref_height=ref_h,
                )
                for z in active_zones
            ]

        self._calibrated_cache[res_key] = calibrated
        return calibrated

    def find_zones_for_point(
        self, point: Union[Point2D, Tuple[float, float]], frame_resolution: Tuple[int, int]
    ) -> List[SafetyZone]:
        """Finds all enabled safety zones that contain the specified coordinate point.

        Args:
            point: (x, y) coordinates.
            frame_resolution: (width, height) of the image frame.

        Returns:
            List of matching SafetyZone objects (supports multiple overlapping zones).
        """
        if self._config is None:
            return []

        w, h = frame_resolution
        calibrated_zones = self.calibrate_for_resolution(w, h)
        matched: List[SafetyZone] = []

        for zone in calibrated_zones:
            if zone.contains_point(point, include_boundary=True):
                matched.append(zone)

        # Enforce explicit shared-boundary precedence:
        # Water hazard zones take exclusive precedence over adjoining deck/walkway zones on shared boundaries.
        return self.resolve_boundary_precedence(matched, point=point)

    @staticmethod
    def resolve_boundary_precedence(
        matched_zones: List[SafetyZone],
        point: Optional[Union[Point2D, Tuple[float, float]]] = None,
    ) -> List[SafetyZone]:
        """Resolves zone classification conflicts when a point falls on a shared dividing boundary.

        Policy:
        Water hazard zones (SWIMMING_POOL_BOUNDARY, WATER_BODY_EDGE) take exclusive precedence
        over adjoining deck/walkway zones (SWIMMING_POOL_GATE, GENERAL_COMMON_AREA).
        A point on the water/deck dividing boundary cannot generate simultaneous water and deck classification.
        """
        has_water_hazard = any(
            z.zone_type in (SafetyZoneType.SWIMMING_POOL_BOUNDARY, SafetyZoneType.WATER_BODY_EDGE)
            for z in matched_zones
        )
        if not has_water_hazard:
            return matched_zones

        has_adjoining_deck = any(
            z.zone_type in (SafetyZoneType.SWIMMING_POOL_GATE, SafetyZoneType.GENERAL_COMMON_AREA)
            for z in matched_zones
        )
        if not has_adjoining_deck:
            return matched_zones

        # If point is provided, verify whether it falls on the boundary of the water hazard zone
        if point is not None:
            water_zones = [
                z for z in matched_zones
                if z.zone_type in (SafetyZoneType.SWIMMING_POOL_BOUNDARY, SafetyZoneType.WATER_BODY_EDGE)
            ]
            is_on_boundary = any(not wz.contains_point(point, include_boundary=False) for wz in water_zones)
            if not is_on_boundary:
                # Interior point in a generic overlapping zone test
                return matched_zones

        # Filter out deck/common area zones that share dividing boundaries with the water hazard zone
        return [
            z for z in matched_zones
            if z.zone_type in (SafetyZoneType.SWIMMING_POOL_BOUNDARY, SafetyZoneType.WATER_BODY_EDGE)
            or z.zone_type not in (SafetyZoneType.SWIMMING_POOL_GATE, SafetyZoneType.GENERAL_COMMON_AREA)
        ]

    def find_zones_for_person(
        self, bbox_xyxy: Tuple[float, float, float, float], frame_resolution: Tuple[int, int]
    ) -> List[SafetyZone]:
        """Determines which safety zones contain a person using their ground contact point.

        Critical Constraint: Evaluates bottom-centre point (feet position).
        Does NOT trigger zone presence merely because a bounding box overlaps the zone boundary.

        Args:
            bbox_xyxy: Person bounding box (x1, y1, x2, y2).
            frame_resolution: (width, height) of the frame.

        Returns:
            List of SafetyZone objects containing the person's ground contact point.
        """
        bottom_center = calculate_bottom_center(bbox_xyxy)
        return self.find_zones_for_point(bottom_center, frame_resolution)
