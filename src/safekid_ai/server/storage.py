"""Persistent Storage and Configuration Management for the SafeKid AI Web Server."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from safekid_ai.roi.models import (
    AlertSeverity,
    CameraROIConfig,
    Point2D,
    Polygon,
    SafetyZone,
    SafetyZoneType,
    ZoneRole,
    ZoneSafetyPolicy,
)
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.server.storage")

DEFAULT_ROI_CONFIG_PATH = Path("data/config/roi_config.json")
JOBS_REGISTRY_PATH = Path("data/jobs/jobs.json")
INCIDENTS_REGISTRY_PATH = Path("data/incidents/incidents.json")
UPLOADS_DIR = Path("data/uploads")
OUTPUTS_DIR = Path("data/outputs/jobs")


class ServerStorage:
    """Manages file storage, job records, incident history, and camera ROI configurations."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(".")
        self.roi_path = self.base_dir / DEFAULT_ROI_CONFIG_PATH
        self.jobs_path = self.base_dir / JOBS_REGISTRY_PATH
        self.incidents_path = self.base_dir / INCIDENTS_REGISTRY_PATH
        self.uploads_dir = self.base_dir / UPLOADS_DIR
        self.outputs_dir = self.base_dir / OUTPUTS_DIR

        self._ensure_directories()
        self._initialize_default_roi_if_missing()

    def _ensure_directories(self) -> None:
        self.roi_path.parent.mkdir(parents=True, exist_ok=True)
        self.jobs_path.parent.mkdir(parents=True, exist_ok=True)
        self.incidents_path.parent.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_default_roi_if_missing(self) -> None:
        """Seeds sensible default hazard zones for residential community CCTV feeds."""
        if not self.roi_path.exists():
            # Create a default configuration tailored for gated residential communities
            # (e.g. Swimming Pool Boundary and Vehicular Gateway)
            pool_poly = Polygon([
                Point2D(120.0, 180.0),
                Point2D(720.0, 180.0),
                Point2D(720.0, 460.0),
                Point2D(120.0, 460.0),
            ])
            gate_poly = Polygon([
                Point2D(50.0, 220.0),
                Point2D(350.0, 220.0),
                Point2D(350.0, 470.0),
                Point2D(50.0, 470.0),
            ])
            pool_zone = SafetyZone(
                roi_id="zone_swimming_pool",
                name="Swimming Pool Basin & Deck",
                zone_type=SafetyZoneType.SWIMMING_POOL_BOUNDARY,
                polygon=pool_poly,
                severity=AlertSeverity.CRITICAL,
                enabled=True,
                dwell_time_seconds=2.0,
                description="Hazardous water basin and perimeter apron",
                policy=ZoneSafetyPolicy(
                    zone_role=ZoneRole.DANGER,
                    entry_alert=True,
                    exit_alert=False,
                    dwell_alert=True,
                    dwell_threshold_seconds=2.0,
                    severity=AlertSeverity.CRITICAL,
                ),
            )
            gate_zone = SafetyZone(
                roi_id="zone_vehicular_gateway",
                name="Driveway & Main Vehicle Gate",
                zone_type=SafetyZoneType.VEHICULAR_GATEWAY,
                polygon=gate_poly,
                severity=AlertSeverity.HIGH,
                enabled=True,
                dwell_time_seconds=3.0,
                description="Active vehicular entrance driveway and boom barrier",
                policy=ZoneSafetyPolicy(
                    zone_role=ZoneRole.DANGER,
                    entry_alert=True,
                    exit_alert=False,
                    dwell_alert=True,
                    dwell_threshold_seconds=3.0,
                    severity=AlertSeverity.HIGH,
                ),
            )
            default_cfg = CameraROIConfig(
                community_id="palm_grove_residential",
                camera_id="cam_main_pool",
                frame_width=848,
                frame_height=480,
                zones=[pool_zone, gate_zone],
                version="1.0",
                calibration_notes="Default residential perimeter and water hazard zones",
            )
            self.save_roi_config(default_cfg.to_dict())

    # --- ROI CONFIGURATION ---

    def get_roi_config(self) -> CameraROIConfig:
        """Loads and returns active CameraROIConfig."""
        if not self.roi_path.exists():
            self._initialize_default_roi_if_missing()

        with open(self.roi_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        zones: List[SafetyZone] = []
        for zdata in data.get("zones", []):
            poly_coords = zdata.get("polygon", [])
            points = [Point2D(float(pt[0]), float(pt[1])) for pt in poly_coords]
            poly = Polygon(points)

            pol_data = zdata.get("policy", {})
            policy = ZoneSafetyPolicy(
                zone_role=ZoneRole(pol_data.get("zone_role", zdata.get("zone_role", "DANGER"))),
                entry_alert=bool(pol_data.get("entry_alert", zdata.get("entry_alert", True))),
                exit_alert=bool(pol_data.get("exit_alert", zdata.get("exit_alert", False))),
                dwell_alert=bool(pol_data.get("dwell_alert", zdata.get("dwell_alert", False))),
                dwell_threshold_seconds=pol_data.get("dwell_threshold_seconds", zdata.get("dwell_time_seconds")),
                severity=AlertSeverity(pol_data.get("severity", zdata.get("severity", "CRITICAL"))),
            )

            z = SafetyZone(
                roi_id=zdata.get("roi_id", f"zone_{len(zones)+1}"),
                name=zdata.get("name", "Configured Zone"),
                zone_type=SafetyZoneType(zdata.get("zone_type", "SWIMMING_POOL_BOUNDARY")),
                polygon=poly,
                severity=AlertSeverity(zdata.get("severity", "CRITICAL")),
                enabled=bool(zdata.get("enabled", True)),
                dwell_time_seconds=zdata.get("dwell_time_seconds"),
                description=zdata.get("description", ""),
                policy=policy,
            )
            zones.append(z)

        return CameraROIConfig(
            community_id=data.get("community_id", "palm_grove_residential"),
            camera_id=data.get("camera_id", "cam_main_pool"),
            frame_width=int(data.get("frame_width", 848)),
            frame_height=int(data.get("frame_height", 480)),
            zones=zones,
            version=data.get("version", "1.0"),
            calibration_notes=data.get("calibration_notes", ""),
        )

    def save_roi_config(self, config_dict: Dict[str, Any]) -> CameraROIConfig:
        """Persists updated CameraROIConfig from UI form submission."""
        with open(self.roi_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Updated ROI configuration saved to {self.roi_path}")
        return self.get_roi_config()

    # --- JOBS REGISTRY ---

    def get_jobs(self) -> List[Dict[str, Any]]:
        """Returns all video processing jobs sorted by creation time (newest first)."""
        if not self.jobs_path.exists():
            return []
        try:
            with open(self.jobs_path, "r", encoding="utf-8") as f:
                jobs = json.load(f)
            return sorted(jobs, key=lambda j: j.get("created_at_ms", 0.0), reverse=True)
        except Exception as e:
            logger.warning(f"Error loading jobs registry: {e}")
            return []

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single job by unique identifier."""
        jobs = self.get_jobs()
        for j in jobs:
            if j.get("job_id") == job_id:
                return j
        return None

    def save_job(self, job_data: Dict[str, Any]) -> None:
        """Creates or updates a job record in the jobs registry."""
        jobs = self.get_jobs()
        job_id = job_data.get("job_id")
        updated = False
        for idx, j in enumerate(jobs):
            if j.get("job_id") == job_id:
                jobs[idx] = {**j, **job_data, "updated_at_ms": time.time() * 1000.0}
                updated = True
                break
        if not updated:
            if "created_at_ms" not in job_data:
                job_data["created_at_ms"] = time.time() * 1000.0
            job_data["updated_at_ms"] = time.time() * 1000.0
            jobs.insert(0, job_data)

        temp_path = self.jobs_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(jobs, f, indent=2)
        temp_path.replace(self.jobs_path)

    def delete_job(self, job_id: str) -> bool:
        """Removes a job and its associated output artifacts."""
        jobs = self.get_jobs()
        initial_len = len(jobs)
        jobs = [j for j in jobs if j.get("job_id") != job_id]
        if len(jobs) == initial_len:
            return False

        with open(self.jobs_path, "w", encoding="utf-8") as f:
            json.dump(jobs, f, indent=2)

        # Remove job output directory
        job_out = self.outputs_dir / job_id
        if job_out.exists():
            shutil.rmtree(job_out, ignore_errors=True)

        return True

    # --- INCIDENTS REGISTRY ---

    def get_incidents(
        self,
        job_id: Optional[str] = None,
        severity: Optional[str] = None,
        zone: Optional[str] = None,
        status: Optional[str] = None,
        classification: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves incidents filtered by query parameters."""
        if not self.incidents_path.exists():
            return []
        try:
            with open(self.incidents_path, "r", encoding="utf-8") as f:
                incidents = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading incidents registry: {e}")
            return []

        filtered = []
        for inc in incidents:
            if job_id and inc.get("job_id") != job_id:
                continue
            if severity and inc.get("severity", "").upper() != severity.upper():
                continue
            if zone and zone.lower() not in inc.get("primary_zone_name", "").lower():
                continue
            if status and inc.get("state", "").upper() != status.upper():
                continue
            if classification and inc.get("classification", "").upper() != classification.upper():
                continue
            filtered.append(inc)

        return sorted(filtered, key=lambda x: x.get("created_at_ms", 0.0), reverse=True)

    def get_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        for inc in self.get_incidents():
            if inc.get("incident_id") == incident_id:
                return inc
        return None

    def record_incident(self, incident_dict: Dict[str, Any]) -> None:
        """Appends or updates an incident in the persistent incident registry."""
        incidents = self.get_incidents()
        inc_id = incident_dict.get("incident_id")
        found = False
        for idx, inc in enumerate(incidents):
            if inc.get("incident_id") == inc_id:
                incidents[idx] = {**inc, **incident_dict, "updated_at_ms": time.time() * 1000.0}
                found = True
                break
        if not found:
            if "created_at_ms" not in incident_dict:
                incident_dict["created_at_ms"] = time.time() * 1000.0
            incident_dict["updated_at_ms"] = time.time() * 1000.0
            incidents.insert(0, incident_dict)

        temp_path = self.incidents_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(incidents, f, indent=2)
        temp_path.replace(self.incidents_path)

    def acknowledge_incident(self, incident_id: str, operator_name: str = "Control Room Operator", notes: str = "") -> Optional[Dict[str, Any]]:
        """Transitions an incident to ACKNOWLEDGED state."""
        inc = self.get_incident(incident_id)
        if not inc:
            return None

        inc["state"] = "ACKNOWLEDGED"
        inc["acknowledged_at_ms"] = time.time() * 1000.0
        inc["acknowledged_by"] = operator_name
        history = inc.get("lifecycle_history", [])
        history.append({
            "transition": "OPERATOR_ACKNOWLEDGED",
            "operator": operator_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "notes": notes,
        })
        inc["lifecycle_history"] = history
        self.record_incident(inc)
        return inc

    def resolve_incident(self, incident_id: str, operator_name: str = "Control Room Operator", reason: str = "Resolved by operator") -> Optional[Dict[str, Any]]:
        """Transitions an incident to RESOLVED state."""
        inc = self.get_incident(incident_id)
        if not inc:
            return None

        inc["state"] = "RESOLVED"
        inc["resolved_at_ms"] = time.time() * 1000.0
        inc["resolved_by"] = operator_name
        inc["resolution_reason"] = reason
        history = inc.get("lifecycle_history", [])
        history.append({
            "transition": "OPERATOR_RESOLVED",
            "operator": operator_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
        })
        inc["lifecycle_history"] = history
        self.record_incident(inc)
        return inc

    # --- FILE UPLOADS & MEDIA ---

    def get_upload_path(self, job_id: str, raw_filename: str) -> Path:
        """Generates a safe, sanitized destination path for uploaded videos."""
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw_filename)
        return self.uploads_dir / f"{job_id}_{safe_name}"

    def get_job_output_dir(self, job_id: str) -> Path:
        """Returns and ensures the job-specific output directory exists."""
        d = self.outputs_dir / job_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "thumbnails").mkdir(parents=True, exist_ok=True)
        return d

