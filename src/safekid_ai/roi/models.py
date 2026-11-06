"""Data models, geometry, and representations for the SafeKid AI Region of Interest (ROI) engine."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


class AlertSeverity(str, Enum):
    """Severity levels for safety zone alerts in residential communities."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SafetyZoneType(str, Enum):
    """Standardized safety zone types across gated residential layouts."""
    SWIMMING_POOL_GATE = "SWIMMING_POOL_GATE"
    SWIMMING_POOL_BOUNDARY = "SWIMMING_POOL_BOUNDARY"
    VEHICULAR_GATEWAY = "VEHICULAR_GATEWAY"
    PERIMETER_FENCE = "PERIMETER_FENCE"
    PLAYGROUND = "PLAYGROUND"
    GENERAL_COMMON_AREA = "GENERAL_COMMON_AREA"
    RESTRICTED_INFRASTRUCTURE = "RESTRICTED_INFRASTRUCTURE"
    WATER_BODY_EDGE = "WATER_BODY_EDGE"


class ZoneRole(str, Enum):
    """Functional safety role of a configured region of interest."""
    SAFE = "SAFE"                # Designated safe areas (e.g. toddler sandbox, internal walkway); no danger alerts
    MONITORED = "MONITORED"      # Monitored zones (e.g. pool deck pavement, lawn); entry is normal, alert on excessive dwell
    DANGER = "DANGER"            # Immediate hazard zones (e.g. pool water basin, vehicle driveway, barrier); immediate entry alert


@dataclass
class ZoneSafetyPolicy:
    """Configurable safety alert policy for a specific zone.

    Controls whether entering, dwelling, or exiting generates child safety alerts.
    """
    zone_role: ZoneRole = ZoneRole.MONITORED
    entry_alert: bool = False
    exit_alert: bool = False
    dwell_threshold_seconds: Optional[float] = None
    dwell_alert: bool = False
    severity: AlertSeverity = AlertSeverity.HIGH

    def __post_init__(self) -> None:
        """Validates policy fields and ensures consistent types."""
        if isinstance(self.zone_role, str):
            try:
                self.zone_role = ZoneRole(self.zone_role.upper())
            except ValueError:
                raise ValueError(
                    f"Invalid zone_role: '{self.zone_role}'. Must be one of {[r.value for r in ZoneRole]}"
                )

        if isinstance(self.severity, str):
            try:
                self.severity = AlertSeverity(self.severity.upper())
            except ValueError:
                raise ValueError(
                    f"Invalid severity: '{self.severity}'. Must be one of {[s.value for s in AlertSeverity]}"
                )

        for attr in ("entry_alert", "exit_alert", "dwell_alert"):
            val = getattr(self, attr)
            if not isinstance(val, bool):
                raise TypeError(f"{attr} must be a boolean (True or False), got {type(val).__name__}: {val!r}")

        if self.dwell_threshold_seconds is not None:
            try:
                self.dwell_threshold_seconds = float(self.dwell_threshold_seconds)
                if self.dwell_threshold_seconds < 0.0:
                    raise ValueError
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid dwell_threshold_seconds: {self.dwell_threshold_seconds}. Must be a non-negative number."
                )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes zone safety policy to a dictionary representation."""
        data: Dict[str, Any] = {
            "zone_role": self.zone_role.value,
            "entry_alert": bool(self.entry_alert),
            "exit_alert": bool(self.exit_alert),
            "dwell_alert": bool(self.dwell_alert),
            "severity": self.severity.value,
        }
        if self.dwell_threshold_seconds is not None:
            data["dwell_threshold_seconds"] = self.dwell_threshold_seconds
        return data

    @classmethod
    def create_default(
        cls,
        zone_role: Union[ZoneRole, str] = ZoneRole.MONITORED,
        severity: AlertSeverity = AlertSeverity.HIGH,
        dwell_threshold_seconds: Optional[float] = None,
        entry_alert: Optional[bool] = None,
        exit_alert: Optional[bool] = None,
        dwell_alert: Optional[bool] = None,
    ) -> "ZoneSafetyPolicy":
        """Creates sensible defaults based on zone_role with optional explicit overrides."""
        if isinstance(zone_role, str):
            zone_role = ZoneRole(zone_role.upper())

        if zone_role == ZoneRole.SAFE:
            def_entry = False
            def_exit = False
            def_dwell = False
            def_sev = severity if severity != AlertSeverity.HIGH else AlertSeverity.LOW
        elif zone_role == ZoneRole.DANGER:
            def_entry = True
            def_exit = False
            def_dwell = True if (dwell_threshold_seconds is not None and dwell_threshold_seconds > 0) else False
            def_sev = severity if severity != AlertSeverity.HIGH else AlertSeverity.CRITICAL
        else:  # MONITORED
            def_entry = False
            def_exit = False
            def_dwell = True if (dwell_threshold_seconds is not None and dwell_threshold_seconds > 0) else False
            def_sev = severity

        return cls(
            zone_role=zone_role,
            entry_alert=entry_alert if entry_alert is not None else def_entry,
            exit_alert=exit_alert if exit_alert is not None else def_exit,
            dwell_threshold_seconds=dwell_threshold_seconds,
            dwell_alert=dwell_alert if dwell_alert is not None else def_dwell,
            severity=severity or def_sev,
        )


@dataclass(frozen=True)
class Point2D:
    """Represents a 2D coordinate (x, y)."""
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_list(self) -> List[float]:
        return [self.x, self.y]


class Polygon:
    """A 2D geometric polygon representing a camera-specific safety zone.

    Supports validation, point-in-polygon testing with boundary detection,
    coordinate normalization, and resolution scaling.
    """

    def __init__(self, points: Sequence[Union[Tuple[float, float], List[float], Point2D]]):
        """Initializes and validates a Polygon.

        Args:
            points: Sequence of at least 3 (x, y) coordinate pairs.

        Raises:
            ValueError: If points format is invalid, fewer than 3 points,
                        or coordinates contain NaN/infinity.
        """
        self.vertices: List[Point2D] = self._validate_and_parse(points)

    def _validate_and_parse(
        self, points: Sequence[Union[Tuple[float, float], List[float], Point2D]]
    ) -> List[Point2D]:
        """Validates polygon coordinates and returns parsed Point2D list."""
        if points is None or not isinstance(points, (list, tuple)):
            raise ValueError(f"Polygon points must be a list or tuple of coordinate pairs, got: {type(points)}")

        if len(points) < 3:
            raise ValueError(
                f"A valid polygon requires at least 3 vertices to enclose an area, got {len(points)} points: {points}"
            )

        parsed: List[Point2D] = []
        for i, pt in enumerate(points):
            if isinstance(pt, Point2D):
                x, y = pt.x, pt.y
            elif isinstance(pt, (list, tuple)) and len(pt) == 2:
                try:
                    x, y = float(pt[0]), float(pt[1])
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid non-numeric coordinate at vertex {i}: {pt}") from e
            else:
                raise ValueError(f"Invalid coordinate format at vertex {i}: expected [x, y], got {pt}")

            if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
                raise ValueError(f"Coordinate at vertex {i} contains NaN or infinite value: ({x}, {y})")

            parsed.append(Point2D(x, y))

        # Check for duplicate consecutive vertices
        n = len(parsed)
        for i in range(n):
            next_p = parsed[(i + 1) % n]
            if abs(parsed[i].x - next_p.x) < 1e-7 and abs(parsed[i].y - next_p.y) < 1e-7:
                raise ValueError(
                    f"Duplicate consecutive vertex detected at index {i}: ({parsed[i].x}, {parsed[i].y}). "
                    "Ensure consecutive polygon vertices are distinct."
                )

        # Validate that edges do not self-intersect (e.g. bow-tie or figure-8 shapes)
        self._check_self_intersection(parsed)

        # Check for non-zero enclosed area (Shoelace formula)
        area = self._compute_area(parsed)
        if area <= 1e-9:
            raise ValueError(
                f"Degenerate polygon with approximately zero enclosed area ({area:.2e}). "
                "Ensure points are not all collinear or duplicated."
            )

        return parsed

    @staticmethod
    def _check_self_intersection(vertices: List[Point2D]) -> None:
        """Validates that polygon edges do not self-intersect or cross each other.

        Permits valid convex and concave polygons while strictly rejecting
        self-crossing geometries such as bow-tie or figure-8 shapes.
        """
        n = len(vertices)
        if n < 4:
            return

        def ccw(a: Point2D, b: Point2D, c: Point2D) -> float:
            return (c.y - a.y) * (b.x - a.x) - (b.y - a.y) * (c.x - a.x)

        def on_segment(p: Point2D, a: Point2D, b: Point2D) -> bool:
            return (
                min(a.x, b.x) - 1e-7 <= p.x <= max(a.x, b.x) + 1e-7
                and min(a.y, b.y) - 1e-7 <= p.y <= max(a.y, b.y) + 1e-7
            )

        def segments_cross(p1: Point2D, p2: Point2D, p3: Point2D, p4: Point2D) -> bool:
            d1 = ccw(p3, p4, p1)
            d2 = ccw(p3, p4, p2)
            d3 = ccw(p1, p2, p3)
            d4 = ccw(p1, p2, p4)

            # Strict interior crossing
            if ((d1 > 1e-7 and d2 < -1e-7) or (d1 < -1e-7 and d2 > 1e-7)) and \
               ((d3 > 1e-7 and d4 < -1e-7) or (d3 < -1e-7 and d4 > 1e-7)):
                return True

            # Collinear / point-on-segment touching between non-consecutive edges
            if abs(d1) <= 1e-7 and on_segment(p1, p3, p4):
                return True
            if abs(d2) <= 1e-7 and on_segment(p2, p3, p4):
                return True
            if abs(d3) <= 1e-7 and on_segment(p3, p1, p2):
                return True
            if abs(d4) <= 1e-7 and on_segment(p4, p1, p2):
                return True

            return False

        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]

            for j in range(i + 1, n):
                if j == (i + 1) % n or i == (j + 1) % n:
                    continue
                if i == 0 and j == n - 1:
                    continue

                p3 = vertices[j]
                p4 = vertices[(j + 1) % n]

                if segments_cross(p1, p2, p3, p4):
                    raise ValueError(
                        f"Self-intersecting polygon detected: edge ({p1.x}, {p1.y})->({p2.x}, {p2.y}) "
                        f"crosses non-consecutive edge ({p3.x}, {p3.y})->({p4.x}, {p4.y}). "
                        "Polygons must be simple, non-self-crossing boundaries (concave polygons are supported)."
                    )

    @staticmethod
    def _compute_area(vertices: List[Point2D]) -> float:
        """Calculates unsigned area using Shoelace algorithm."""
        n = len(vertices)
        area2 = 0.0
        for i in range(n):
            j = (i + 1) % n
            area2 += vertices[i].x * vertices[j].y
            area2 -= vertices[j].x * vertices[i].y
        return abs(area2) / 2.0

    @property
    def area(self) -> float:
        """Enclosed area of the polygon."""
        return self._compute_area(self.vertices)

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Returns (min_x, min_y, max_x, max_y) bounding box."""
        xs = [p.x for p in self.vertices]
        ys = [p.y for p in self.vertices]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def centroid(self) -> Tuple[float, float]:
        """Calculates polygon centroid (cx, cy)."""
        xs = [p.x for p in self.vertices]
        ys = [p.y for p in self.vertices]
        return (float(np.mean(xs)), float(np.mean(ys)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Polygon):
            return False
        if len(self.vertices) != len(other.vertices):
            return False
        return all(
            abs(v1.x - v2.x) < 1e-4 and abs(v1.y - v2.y) < 1e-4
            for v1, v2 in zip(self.vertices, other.vertices)
        )

    def to_coordinates(self) -> List[List[float]]:
        """Returns list of [x, y] coordinates."""
        return [[p.x, p.y] for p in self.vertices]

    def to_numpy(self) -> np.ndarray:
        """Returns (N, 2) numpy array of vertex coordinates."""
        return np.array([[p.x, p.y] for p in self.vertices], dtype=np.float32)

    def contains_point(
        self, point: Union[Point2D, Tuple[float, float], List[float]], include_boundary: bool = True
    ) -> bool:
        """Determines if a 2D point lies inside the polygon using ray casting and boundary detection.

        Args:
            point: (x, y) coordinate to test.
            include_boundary: Whether points exactly on edges/vertices count as inside.

        Returns:
            True if point is inside (or on boundary when include_boundary=True), False otherwise.
        """
        if isinstance(point, Point2D):
            px, py = point.x, point.y
        else:
            px, py = float(point[0]), float(point[1])

        # Fast bounding box rejection check
        min_x, min_y, max_x, max_y = self.bounding_box
        eps = 1e-7
        if px < min_x - eps or px > max_x + eps or py < min_y - eps or py > max_y + eps:
            return False

        # Check if point lies directly on an edge or vertex
        n = len(self.vertices)
        for i in range(n):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % n]

            # Check vertex exact match
            if abs(px - p1.x) < eps and abs(py - p1.y) < eps:
                return include_boundary

            # Check edge collinearity and bounding segment
            cross = (py - p1.y) * (p2.x - p1.x) - (px - p1.x) * (p2.y - p1.y)
            if abs(cross) < eps:
                if min(p1.x, p2.x) - eps <= px <= max(p1.x, p2.x) + eps and \
                   min(p1.y, p2.y) - eps <= py <= max(p1.y, p2.y) + eps:
                    return include_boundary

        # Ray Casting Algorithm (crossing count along horizontal ray y = py for x >= px)
        inside = False
        for i in range(n):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % n]

            # Check if edge crosses the horizontal ray
            if ((p1.y > py) != (p2.y > py)):
                # Compute x-coordinate of intersection with line segment
                x_intersect = p1.x + (py - p1.y) * (p2.x - p1.x) / (p2.y - p1.y)
                if px < x_intersect:
                    inside = not inside

        return inside

    def scale(self, scale_x: float, scale_y: float) -> "Polygon":
        """Scales all vertices by independent horizontal and vertical factors."""
        if scale_x <= 0 or scale_y <= 0:
            raise ValueError(f"Scale factors must be positive numbers, got ({scale_x}, {scale_y})")
        new_points = [[p.x * scale_x, p.y * scale_y] for p in self.vertices]
        return Polygon(new_points)

    def to_normalized(self, ref_width: float, ref_height: float) -> "Polygon":
        """Converts pixel coordinates to normalized [0.0, 1.0] coordinates."""
        if ref_width <= 0 or ref_height <= 0:
            raise ValueError(f"Reference frame resolution must be positive: {ref_width}x{ref_height}")
        return self.scale(1.0 / ref_width, 1.0 / ref_height)

    def to_absolute(self, target_width: float, target_height: float) -> "Polygon":
        """Converts normalized [0.0, 1.0] coordinates to absolute pixel coordinates."""
        if target_width <= 0 or target_height <= 0:
            raise ValueError(f"Target frame resolution must be positive: {target_width}x{target_height}")
        return self.scale(float(target_width), float(target_height))

    def __repr__(self) -> str:
        return f"Polygon(vertices={len(self.vertices)}, area={self.area:.1f})"


@dataclass
class SafetyZone:
    """Represents a single configurable safety zone for a camera.

    Supports single-polygon zones as well as composite multi-polygon zones
    (e.g., surrounding pool decks with an interior water basin cut out).
    """
    roi_id: str
    name: str
    zone_type: SafetyZoneType
    polygon: Polygon
    severity: AlertSeverity = AlertSeverity.HIGH
    enabled: bool = True
    dwell_time_seconds: Optional[float] = None
    description: str = ""
    is_normalized: bool = False
    polygons: List[Polygon] = field(default_factory=list)
    policy: Optional[ZoneSafetyPolicy] = None

    def __post_init__(self) -> None:
        """Ensures polygon, polygons list, and safety policy are synchronized."""
        if not self.polygons and self.polygon is not None:
            self.polygons = [self.polygon]
        elif self.polygons and self.polygon is None:
            self.polygon = self.polygons[0]
        elif self.polygons and self.polygon not in self.polygons:
            self.polygons = [self.polygon] + [p for p in self.polygons if p != self.polygon]

        # Initialize default ZoneSafetyPolicy if not provided
        if self.policy is None:
            if (
                self.zone_type in (SafetyZoneType.SWIMMING_POOL_BOUNDARY, SafetyZoneType.RESTRICTED_INFRASTRUCTURE)
                or self.severity == AlertSeverity.CRITICAL
            ):
                role = ZoneRole.DANGER
            elif self.zone_type == SafetyZoneType.GENERAL_COMMON_AREA and self.severity == AlertSeverity.LOW:
                role = ZoneRole.SAFE
            else:
                role = ZoneRole.MONITORED

            exit_alert_default = True if self.zone_type == SafetyZoneType.SWIMMING_POOL_BOUNDARY else False
            self.policy = ZoneSafetyPolicy.create_default(
                zone_role=role,
                severity=self.severity,
                dwell_threshold_seconds=self.dwell_time_seconds,
                exit_alert=exit_alert_default,
            )
        else:
            if self.policy.dwell_threshold_seconds is not None and self.dwell_time_seconds is None:
                self.dwell_time_seconds = self.policy.dwell_threshold_seconds
            elif self.dwell_time_seconds is not None and self.policy.dwell_threshold_seconds is None:
                self.policy.dwell_threshold_seconds = self.dwell_time_seconds

    def contains_point(
        self, point: Union[Point2D, Tuple[float, float], List[float]], include_boundary: bool = True
    ) -> bool:
        """Tests whether a point falls within this zone (across any constituent polygon)."""
        if not self.enabled:
            return False
        if self.polygons:
            return any(p.contains_point(point, include_boundary=include_boundary) for p in self.polygons)
        return self.polygon.contains_point(point, include_boundary=include_boundary)

    def scale_to_resolution(
        self, target_width: int, target_height: int, ref_width: int, ref_height: int
    ) -> "SafetyZone":
        """Returns a copy of this zone calibrated for target resolution."""
        if self.is_normalized:
            sx = float(target_width)
            sy = float(target_height)
        elif target_width == ref_width and target_height == ref_height:
            return self
        else:
            sx = float(target_width) / float(ref_width)
            sy = float(target_height) / float(ref_height)

        active_polys = self.polygons if self.polygons else [self.polygon]
        scaled_polys = [p.scale(sx, sy) for p in active_polys]

        return SafetyZone(
            roi_id=self.roi_id,
            name=self.name,
            zone_type=self.zone_type,
            polygon=scaled_polys[0],
            polygons=scaled_polys,
            severity=self.severity,
            enabled=self.enabled,
            dwell_time_seconds=self.dwell_time_seconds,
            description=self.description,
            is_normalized=False,
            policy=self.policy,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes safety zone to a dictionary representation."""
        data: Dict[str, Any] = {
            "roi_id": self.roi_id,
            "name": self.name,
            "zone_type": self.zone_type.value,
            "severity": self.severity.value,
            "enabled": self.enabled,
        }
        if self.policy is not None:
            data["policy"] = self.policy.to_dict()
            data["zone_role"] = self.policy.zone_role.value
            data["entry_alert"] = self.policy.entry_alert
            data["exit_alert"] = self.policy.exit_alert
            data["dwell_alert"] = self.policy.dwell_alert
        if len(self.polygons) > 1:
            data["polygons"] = [p.to_coordinates() for p in self.polygons]
            data["polygon"] = self.polygon.to_coordinates()
        else:
            data["polygon"] = self.polygon.to_coordinates()

        if self.dwell_time_seconds is not None:
            data["dwell_time_seconds"] = self.dwell_time_seconds
        if self.description:
            data["description"] = self.description
        if self.is_normalized:
            data["is_normalized"] = True
        return data


@dataclass
class CameraROIConfig:
    """Complete camera-specific ROI configuration for a residential community."""
    community_id: str
    camera_id: str
    frame_width: int
    frame_height: int
    zones: List[SafetyZone] = field(default_factory=list)
    version: str = "1.0"
    calibration_notes: str = ""

    def get_zone(self, roi_id: str) -> Optional[SafetyZone]:
        """Finds zone by unique ID."""
        for z in self.zones:
            if z.roi_id == roi_id:
                return z
        return None

    def get_active_zones(self) -> List[SafetyZone]:
        """Returns list of enabled zones."""
        return [z for z in self.zones if z.enabled]

    @property
    def num_zones(self) -> int:
        return len(self.zones)

    @property
    def num_active_zones(self) -> int:
        return len(self.get_active_zones())

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to JSON-serializable dictionary."""
        return {
            "community_id": self.community_id,
            "camera_id": self.camera_id,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "version": self.version,
            "calibration_notes": self.calibration_notes,
            "zones": [z.to_dict() for z in self.zones],
        }
