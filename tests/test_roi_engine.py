"""Unit tests for the configurable camera-specific ROI engine and geometric polygon operations."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from safekid_ai.roi.engine import ROIEngine, calculate_bottom_center
from safekid_ai.roi.loader import (
    load_camera_roi_config,
    parse_roi_config_dict,
    save_camera_roi_config,
    validate_roi_config_dict,
)
from safekid_ai.roi.models import (
    AlertSeverity,
    CameraROIConfig,
    Point2D,
    Polygon,
    SafetyZone,
    SafetyZoneType,
)


# ============================================================================
# 1. POLYGON GEOMETRY & VALIDATION TESTS
# ============================================================================

def test_polygon_validation_valid():
    """Tests creating valid triangle and quadrilateral polygons."""
    tri = Polygon([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)])
    assert tri.num_vertices == 3
    assert tri.area == 50.0

    quad = Polygon([(100, 100), (200, 100), (200, 200), (100, 200)])
    assert quad.num_vertices == 4
    assert quad.area == 10000.0
    assert quad.bounding_box == (100.0, 100.0, 200.0, 200.0)
    assert quad.centroid == (150.0, 150.0)


def test_polygon_validation_invalid_few_points():
    """Ensures polygons with fewer than 3 points are strictly rejected."""
    with pytest.raises(ValueError, match="at least 3 vertices"):
        Polygon([(0, 0), (10, 10)])
    with pytest.raises(ValueError, match="at least 3 vertices"):
        Polygon([])


def test_polygon_validation_invalid_coordinates():
    """Ensures malformed coordinate pairs, NaNs, and non-numeric inputs raise ValueError."""
    with pytest.raises(ValueError, match="Invalid coordinate format"):
        Polygon([(0, 0, 0), (10, 10, 10), (20, 20, 20)])  # 3D points
    with pytest.raises(ValueError, match="non-numeric"):
        Polygon([(0, 0), ("bad", 10), (10, 0)])
    with pytest.raises(ValueError, match="NaN or infinite"):
        Polygon([(0, 0), (float("nan"), 10), (10, 0)])


def test_polygon_validation_collinear_zero_area():
    """Ensures collinear points forming zero enclosed area are rejected."""
    with pytest.raises(ValueError, match="zero enclosed area"):
        Polygon([(0, 0), (5, 5), (10, 10)])


# ============================================================================
# 2. POINT-IN-POLYGON & BOUNDARY TESTS
# ============================================================================

def test_point_inside_polygon():
    """Tests that points strictly inside the polygon return True."""
    # Unit box from (10, 10) to (50, 50)
    poly = Polygon([(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)])
    assert poly.contains_point((30.0, 30.0)) is True
    assert poly.contains_point(Point2D(15.0, 45.0)) is True
    assert poly.contains_point([20.0, 20.0]) is True


def test_point_outside_polygon():
    """Tests that points outside the polygon return False."""
    poly = Polygon([(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)])
    assert poly.contains_point((5.0, 30.0)) is False    # Left of polygon
    assert poly.contains_point((60.0, 30.0)) is False   # Right of polygon
    assert poly.contains_point((30.0, 5.0)) is False    # Above polygon
    assert poly.contains_point((30.0, 60.0)) is False   # Below polygon
    assert poly.contains_point((0.0, 0.0)) is False     # Origin


def test_point_on_polygon_boundary():
    """Tests boundary handling on vertices and edges."""
    poly = Polygon([(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)])

    # On vertex
    assert poly.contains_point((10.0, 10.0), include_boundary=True) is True
    assert poly.contains_point((50.0, 50.0), include_boundary=True) is True

    # On horizontal edge
    assert poly.contains_point((30.0, 10.0), include_boundary=True) is True
    assert poly.contains_point((30.0, 50.0), include_boundary=True) is True

    # On vertical edge
    assert poly.contains_point((10.0, 30.0), include_boundary=True) is True
    assert poly.contains_point((50.0, 30.0), include_boundary=True) is True

    # When include_boundary is False, points directly on vertices return False
    assert poly.contains_point((10.0, 10.0), include_boundary=False) is False


# ============================================================================
# 3. COORDINATE NORMALIZATION & RESOLUTION CALIBRATION
# ============================================================================

def test_normalized_coordinate_conversion():
    """Tests converting between pixel coordinates and normalized [0.0, 1.0] space."""
    poly = Polygon([(192.0, 108.0), (384.0, 108.0), (384.0, 216.0), (192.0, 216.0)])
    norm_poly = poly.to_normalized(ref_width=1920.0, ref_height=1080.0)

    # 192/1920 = 0.1, 108/1080 = 0.1; 384/1920 = 0.2, 216/1080 = 0.2
    coords = norm_poly.to_coordinates()
    assert pytest.approx(coords[0][0], 1e-4) == 0.1
    assert pytest.approx(coords[0][1], 1e-4) == 0.1
    assert pytest.approx(coords[1][0], 1e-4) == 0.2
    assert pytest.approx(coords[2][1], 1e-4) == 0.2

    # Scale back to 1280x720 target
    scaled_poly = norm_poly.to_absolute(target_width=1280.0, target_height=720.0)
    scaled_coords = scaled_poly.to_coordinates()
    assert pytest.approx(scaled_coords[0][0], 1e-3) == 128.0  # 0.1 * 1280
    assert pytest.approx(scaled_coords[0][1], 1e-3) == 72.0   # 0.1 * 720


def test_dynamic_resolution_calibration_in_engine():
    """Verifies ROIEngine automatically adapts zones when video stream resolution changes."""
    zone = SafetyZone(
        roi_id="test_zone_1",
        name="Test Pool",
        zone_type=SafetyZoneType.SWIMMING_POOL_GATE,
        polygon=Polygon([(100.0, 100.0), (500.0, 100.0), (500.0, 500.0), (100.0, 500.0)]),
        severity=AlertSeverity.HIGH,
    )
    cfg = CameraROIConfig(
        community_id="comm_a",
        camera_id="cam_1",
        frame_width=1000,
        frame_height=1000,
        zones=[zone],
    )
    engine = ROIEngine(config=cfg)

    # In native 1000x1000 resolution: (300, 300) is inside
    assert len(engine.find_zones_for_point((300.0, 300.0), (1000, 1000))) == 1
    assert len(engine.find_zones_for_point((50.0, 50.0), (1000, 1000))) == 0

    # In scaled 500x500 resolution: coordinates scaled by 0.5 -> zone is (50, 50) to (250, 250)
    # Point (150, 150) should be inside; point (300, 300) should now be outside!
    assert len(engine.find_zones_for_point((150.0, 150.0), (500, 500))) == 1
    assert len(engine.find_zones_for_point((300.0, 300.0), (500, 500))) == 0


# ============================================================================
# 4. MULTIPLE & OVERLAPPING ZONES PER CAMERA
# ============================================================================

def test_multiple_and_overlapping_zones():
    """Tests evaluating points against multiple zones, including overlapping areas."""
    outer_zone = SafetyZone(
        roi_id="pool_outer_deck",
        name="Pool Outer Deck",
        zone_type=SafetyZoneType.SWIMMING_POOL_GATE,
        polygon=Polygon([(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)]),
        severity=AlertSeverity.MEDIUM,
    )
    inner_zone = SafetyZone(
        roi_id="pool_deep_water",
        name="Pool Deep Water",
        zone_type=SafetyZoneType.SWIMMING_POOL_BOUNDARY,
        polygon=Polygon([(50.0, 50.0), (150.0, 50.0), (150.0, 150.0), (50.0, 150.0)]),
        severity=AlertSeverity.CRITICAL,
    )
    cfg = CameraROIConfig(
        community_id="comm_01",
        camera_id="cam_01",
        frame_width=300,
        frame_height=300,
        zones=[outer_zone, inner_zone],
    )
    engine = ROIEngine(config=cfg)

    # Point at (100, 100) is inside BOTH outer deck and deep water (overlap)
    matched = engine.find_zones_for_point((100.0, 100.0), (300, 300))
    assert len(matched) == 2
    matched_ids = {z.roi_id for z in matched}
    assert matched_ids == {"pool_outer_deck", "pool_deep_water"}

    # Point at (20, 20) is inside outer deck ONLY
    matched_outer = engine.find_zones_for_point((20.0, 20.0), (300, 300))
    assert len(matched_outer) == 1
    assert matched_outer[0].roi_id == "pool_outer_deck"

    # Point at (250, 250) is outside both
    matched_none = engine.find_zones_for_point((250.0, 250.0), (300, 300))
    assert len(matched_none) == 0


# ============================================================================
# 5. CONFIGURATION LOADING, PERSISTENCE & RELOADING
# ============================================================================

def test_json_config_load_and_save():
    """Verifies round-trip JSON serialization, schema validation, and file persistence."""
    zone = SafetyZone(
        roi_id="vehicular_gate_01",
        name="North Vehicular Gate",
        zone_type=SafetyZoneType.VEHICULAR_GATEWAY,
        polygon=Polygon([(100.0, 100.0), (400.0, 100.0), (400.0, 300.0), (100.0, 300.0)]),
        severity=AlertSeverity.HIGH,
        enabled=True,
        dwell_time_seconds=2.5,
        description="Main vehicle gate",
    )
    orig_cfg = CameraROIConfig(
        community_id="greenfield_01",
        camera_id="cam_north_gate",
        frame_width=1920,
        frame_height=1080,
        zones=[zone],
        version="1.0",
        calibration_notes="Initial install",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_file = Path(tmp_dir) / "cam_north_gate.json"
        saved_path = save_camera_roi_config(orig_cfg, json_file)
        assert saved_path.exists()

        # Load back
        loaded_cfg = load_camera_roi_config(json_file)
        assert loaded_cfg.community_id == "greenfield_01"
        assert loaded_cfg.camera_id == "cam_north_gate"
        assert loaded_cfg.frame_width == 1920
        assert loaded_cfg.frame_height == 1080
        assert len(loaded_cfg.zones) == 1

        lz = loaded_cfg.zones[0]
        assert lz.roi_id == "vehicular_gate_01"
        assert lz.zone_type == SafetyZoneType.VEHICULAR_GATEWAY
        assert lz.severity == AlertSeverity.HIGH
        assert lz.dwell_time_seconds == 2.5
        assert lz.polygon.area == 60000.0


def test_dynamic_config_reload():
    """Verifies hot-reloading camera configuration from disk at runtime."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "dynamic_cam.json"

        # Version 1: 1 zone
        data_v1 = {
            "community_id": "comm_test",
            "camera_id": "cam_dynamic",
            "frame_width": 640,
            "frame_height": 480,
            "zones": [
                {
                    "roi_id": "z1",
                    "name": "Zone 1",
                    "zone_type": "PLAYGROUND",
                    "polygon": [[10, 10], [100, 10], [100, 100], [10, 100]],
                    "severity": "LOW",
                    "enabled": True,
                }
            ],
        }
        with open(config_path, "w") as f:
            json.dump(data_v1, f)

        engine = ROIEngine(config_path=config_path)
        assert len(engine.config.zones) == 1

        # Version 2: Update file with a second zone
        data_v2 = dict(data_v1)
        data_v2["zones"].append({
            "roi_id": "z2",
            "name": "Zone 2",
            "zone_type": "PERIMETER_FENCE",
            "polygon": [[200, 200], [400, 200], [400, 400], [200, 400]],
            "severity": "HIGH",
            "enabled": True,
        })
        with open(config_path, "w") as f:
            json.dump(data_v2, f)

        # Hot reload
        engine.reload_config()
        assert len(engine.config.zones) == 2
        assert engine.config.get_zone("z2") is not None


def test_invalid_json_schema_rejection():
    """Verifies that schema violations and missing required keys raise informative ValueErrors."""
    # Missing required keys
    with pytest.raises(ValueError, match="Missing required top-level configuration key"):
        validate_roi_config_dict({"camera_id": "cam_1"})

    # Duplicate ROI IDs
    dup_data = {
        "community_id": "comm_1",
        "camera_id": "cam_1",
        "frame_width": 1920,
        "frame_height": 1080,
        "zones": [
            {"roi_id": "same_id", "name": "Z1", "zone_type": "PLAYGROUND", "polygon": [[0, 0], [10, 0], [10, 10]]},
            {"roi_id": "same_id", "name": "Z2", "zone_type": "PLAYGROUND", "polygon": [[0, 0], [10, 0], [10, 10]]},
        ],
    }
    with pytest.raises(ValueError, match="Duplicate 'roi_id' detected"):
        validate_roi_config_dict(dup_data)

    # Invalid zone type
    bad_type = {
        "community_id": "comm_1",
        "camera_id": "cam_1",
        "frame_width": 1920,
        "frame_height": 1080,
        "zones": [
            {"roi_id": "z1", "name": "Z1", "zone_type": "NON_EXISTENT_ZONE", "polygon": [[0, 0], [10, 0], [10, 10]]},
        ],
    }
    with pytest.raises(ValueError, match="unrecognized 'zone_type'"):
        validate_roi_config_dict(bad_type)


def test_camera_configuration_independence():
    """Verifies that configurations for different cameras and communities remain completely independent."""
    cfg_pool = load_camera_roi_config("configs/cameras/community_001/camera_pool_01.json")
    cfg_gate = load_camera_roi_config("configs/cameras/community_001/camera_gate_north.json")
    cfg_play = load_camera_roi_config("configs/cameras/community_002/camera_playground_01.json")

    assert cfg_pool.camera_id == "camera_pool_01"
    assert cfg_gate.camera_id == "camera_gate_north"
    assert cfg_play.community_id == "community_002"

    assert cfg_play.frame_width == 1280
    assert cfg_pool.frame_width == 1920

    # Pool camera has pool zones; playground has sandbox zone
    assert cfg_pool.get_zone("pool_deep_end_01") is not None
    assert cfg_gate.get_zone("pool_deep_end_01") is None
    assert cfg_play.get_zone("toddler_play_sandbox") is not None

