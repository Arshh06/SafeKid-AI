"""JSON Configuration loader, parser, and validator for camera-specific ROIs."""

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Union

from safekid_ai.roi.models import (
    AlertSeverity,
    CameraROIConfig,
    Polygon,
    SafetyZone,
    SafetyZoneType,
    ZoneRole,
    ZoneSafetyPolicy,
)
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.roi.loader")


def validate_roi_config_dict(data: Dict[str, Any]) -> None:
    """Validates raw dictionary structure against SafeKid AI ROI schema rules.

    Args:
        data: Parsed JSON dictionary.

    Raises:
        ValueError: If any required field is missing, data types are invalid,
                    or zone definitions violate constraints.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Root ROI configuration must be a JSON object, got: {type(data)}")

    required_root_keys = ["community_id", "camera_id", "frame_width", "frame_height", "zones"]
    for key in required_root_keys:
        if key not in data:
            raise ValueError(f"Missing required top-level configuration key: '{key}'")

    if not isinstance(data["community_id"], str) or not data["community_id"].strip():
        raise ValueError(f"'community_id' must be a non-empty string, got: {data['community_id']}")

    if not isinstance(data["camera_id"], str) or not data["camera_id"].strip():
        raise ValueError(f"'camera_id' must be a non-empty string, got: {data['camera_id']}")

    try:
        fw = int(data["frame_width"])
        fh = int(data["frame_height"])
        if fw <= 0 or fh <= 0:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError(
            f"'frame_width' and 'frame_height' must be positive integers, got: "
            f"width={data.get('frame_width')}, height={data.get('frame_height')}"
        )

    zones_data = data["zones"]
    if not isinstance(zones_data, list):
        raise ValueError(f"'zones' must be a list of zone definitions, got: {type(zones_data)}")

    seen_roi_ids: Set[str] = set()

    for idx, z_raw in enumerate(zones_data):
        if not isinstance(z_raw, dict):
            raise ValueError(f"Zone at index {idx} must be a dictionary, got: {type(z_raw)}")

        # Validate roi_id
        if "roi_id" not in z_raw or not isinstance(z_raw["roi_id"], str) or not z_raw["roi_id"].strip():
            raise ValueError(f"Zone at index {idx} missing valid 'roi_id' string.")

        roi_id = z_raw["roi_id"].strip()
        if roi_id in seen_roi_ids:
            raise ValueError(f"Duplicate 'roi_id' detected: '{roi_id}' appears more than once in camera configuration.")
        seen_roi_ids.add(roi_id)

        # Validate name
        if "name" not in z_raw or not isinstance(z_raw["name"], str) or not z_raw["name"].strip():
            raise ValueError(f"Zone '{roi_id}' missing valid 'name' string.")

        # Validate zone_type
        if "zone_type" not in z_raw:
            raise ValueError(f"Zone '{roi_id}' missing required 'zone_type'.")

        zt_str = str(z_raw["zone_type"]).upper()
        valid_zone_types = {e.value for e in SafetyZoneType}
        if zt_str not in valid_zone_types:
            raise ValueError(
                f"Zone '{roi_id}' has unrecognized 'zone_type': '{z_raw['zone_type']}'. "
                f"Must be one of: {sorted(list(valid_zone_types))}"
            )

        # Validate polygon or composite polygons
        if "polygon" not in z_raw and "polygons" not in z_raw:
            raise ValueError(f"Zone '{roi_id}' missing required 'polygon' or 'polygons' coordinates.")

        try:
            if "polygons" in z_raw:
                polys_raw = z_raw["polygons"]
                if not isinstance(polys_raw, list) or len(polys_raw) == 0:
                    raise ValueError("Field 'polygons' must be a non-empty list of polygon coordinates.")
                for p_idx, p_raw in enumerate(polys_raw):
                    Polygon(p_raw)
            else:
                Polygon(z_raw["polygon"])
        except ValueError as e:
            raise ValueError(f"Invalid polygon for zone '{roi_id}': {e}") from e

        # Validate severity if provided
        if "severity" in z_raw:
            sev_str = str(z_raw["severity"]).upper()
            valid_severities = {e.value for e in AlertSeverity}
            if sev_str not in valid_severities:
                raise ValueError(
                    f"Zone '{roi_id}' has unrecognized 'severity': '{z_raw['severity']}'. "
                    f"Must be one of: {sorted(list(valid_severities))}"
                )

        # Validate dwell_time_seconds if provided
        if "dwell_time_seconds" in z_raw and z_raw["dwell_time_seconds"] is not None:
            try:
                dwell = float(z_raw["dwell_time_seconds"])
                if dwell < 0.0:
                    raise ValueError()
            except (ValueError, TypeError):
                raise ValueError(
                    f"Zone '{roi_id}' has invalid 'dwell_time_seconds': {z_raw['dwell_time_seconds']}. "
                    "Must be a non-negative number."
                )

        # Validate zone safety policy if provided
        policy_raw = z_raw.get("policy")
        if policy_raw is not None and not isinstance(policy_raw, dict):
            raise ValueError(f"Zone '{roi_id}' field 'policy' must be a JSON dictionary, got: {type(policy_raw)}")

        # Check zone_role (either inside policy or flat on z_raw)
        role_val = policy_raw.get("zone_role") if isinstance(policy_raw, dict) else z_raw.get("zone_role")
        if role_val is not None:
            valid_roles = {r.value for r in ZoneRole}
            if str(role_val).upper() not in valid_roles:
                raise ValueError(
                    f"Zone '{roi_id}' has unrecognized 'zone_role': '{role_val}'. "
                    f"Must be one of: {sorted(list(valid_roles))}"
                )

        # Check boolean alerts
        for bool_field in ("entry_alert", "exit_alert", "dwell_alert"):
            val = policy_raw.get(bool_field) if isinstance(policy_raw, dict) else z_raw.get(bool_field)
            if val is not None and not isinstance(val, bool):
                raise ValueError(
                    f"Zone '{roi_id}' field '{bool_field}' must be a boolean (true/false), got: {type(val)}"
                )

        # Check dwell_threshold_seconds
        dwell_thresh = policy_raw.get("dwell_threshold_seconds") if isinstance(policy_raw, dict) else z_raw.get("dwell_threshold_seconds")
        if dwell_thresh is not None:
            try:
                dt = float(dwell_thresh)
                if dt < 0.0:
                    raise ValueError
            except (ValueError, TypeError):
                raise ValueError(
                    f"Zone '{roi_id}' has invalid 'dwell_threshold_seconds': {dwell_thresh}. "
                    "Must be a non-negative number."
                )


def parse_roi_config_dict(data: Dict[str, Any]) -> CameraROIConfig:
    """Parses a validated dictionary into a CameraROIConfig instance."""
    validate_roi_config_dict(data)

    zones: List[SafetyZone] = []
    for z_raw in data["zones"]:
        if "polygons" in z_raw:
            polys = [Polygon(p) for p in z_raw["polygons"]]
            primary_poly = polys[0]
        else:
            primary_poly = Polygon(z_raw["polygon"])
            polys = [primary_poly]

        sev_str = str(z_raw.get("severity", "HIGH")).upper()
        severity = AlertSeverity(sev_str)
        zone_type = SafetyZoneType(str(z_raw["zone_type"]).upper())
        enabled = bool(z_raw.get("enabled", True))
        dwell = float(z_raw["dwell_time_seconds"]) if z_raw.get("dwell_time_seconds") is not None else None
        desc = str(z_raw.get("description", ""))
        is_norm = bool(z_raw.get("is_normalized", False))

        # Parse ZoneSafetyPolicy
        policy_data = z_raw.get("policy") if isinstance(z_raw.get("policy"), dict) else {}
        role_str = policy_data.get("zone_role", z_raw.get("zone_role"))

        # Determine fallback role based on zone_type & severity if not specified
        if role_str is not None:
            zone_role = ZoneRole(str(role_str).upper())
        elif zone_type in (SafetyZoneType.SWIMMING_POOL_BOUNDARY, SafetyZoneType.RESTRICTED_INFRASTRUCTURE) or severity == AlertSeverity.CRITICAL:
            zone_role = ZoneRole.DANGER
        elif zone_type == SafetyZoneType.GENERAL_COMMON_AREA and severity == AlertSeverity.LOW:
            zone_role = ZoneRole.SAFE
        else:
            zone_role = ZoneRole.MONITORED

        entry_alert = policy_data.get("entry_alert", z_raw.get("entry_alert"))
        exit_alert = policy_data.get("exit_alert", z_raw.get("exit_alert"))
        if exit_alert is None and zone_type == SafetyZoneType.SWIMMING_POOL_BOUNDARY:
            exit_alert = True
        dwell_alert = policy_data.get("dwell_alert", z_raw.get("dwell_alert"))
        dwell_thresh = policy_data.get("dwell_threshold_seconds", z_raw.get("dwell_threshold_seconds", dwell))

        pol_sev_str = policy_data.get("severity", z_raw.get("severity", severity.value))
        pol_severity = AlertSeverity(str(pol_sev_str).upper())

        policy = ZoneSafetyPolicy.create_default(
            zone_role=zone_role,
            severity=pol_severity,
            dwell_threshold_seconds=dwell_thresh,
            entry_alert=entry_alert,
            exit_alert=exit_alert,
            dwell_alert=dwell_alert,
        )

        zone = SafetyZone(
            roi_id=z_raw["roi_id"].strip(),
            name=z_raw["name"].strip(),
            zone_type=zone_type,
            polygon=primary_poly,
            polygons=polys,
            severity=severity,
            enabled=enabled,
            dwell_time_seconds=dwell,
            description=desc,
            is_normalized=is_norm,
            policy=policy,
        )
        zones.append(zone)

    return CameraROIConfig(
        community_id=data["community_id"].strip(),
        camera_id=data["camera_id"].strip(),
        frame_width=int(data["frame_width"]),
        frame_height=int(data["frame_height"]),
        zones=zones,
        version=str(data.get("version", "1.0")),
        calibration_notes=str(data.get("calibration_notes", "")),
    )


def load_camera_roi_config(json_path: Union[str, Path]) -> CameraROIConfig:
    """Loads and validates a camera-specific ROI JSON configuration file.

    Args:
        json_path: Filepath to the external JSON configuration.

    Returns:
        CameraROIConfig instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If JSON is malformed or violates schema validation.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Camera ROI configuration file not found: {path.resolve()}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in ROI config file '{path}': {e}") from e

    config = parse_roi_config_dict(data)
    logger.info(
        f"Loaded ROI config for camera '{config.camera_id}' ({config.community_id}): "
        f"{len(config.zones)} zones configured ({len(config.get_active_zones())} active)."
    )
    return config


def save_camera_roi_config(config: CameraROIConfig, json_path: Union[str, Path]) -> Path:
    """Serializes and saves a CameraROIConfig to an external JSON file.

    Args:
        config: CameraROIConfig instance to persist.
        json_path: Target destination path.

    Returns:
        Resolved Path of the saved file.
    """
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict()
    validate_roi_config_dict(data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved Camera ROI config for '{config.camera_id}' to: {path.resolve()}")
    return path

