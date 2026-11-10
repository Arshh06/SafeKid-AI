"""Comprehensive End-to-End Tests for SafeKid AI Operational Monitoring Web Application."""

import io
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Dict

import cv2
import numpy as np
import pytest
from werkzeug.test import Client

from safekid_ai.hardware.esp32_client import ESP32BuzzerClient
from safekid_ai.notifications.provider import (
    IncidentAlert,
    MockNotificationProvider,
    NotificationDeliveryResult,
    NotificationProviderType,
    WebhookNotificationProvider,
)
from safekid_ai.notifications.service import NotificationService
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
from safekid_ai.server.app import create_app
from safekid_ai.server.job_runner import VideoJobRunner
from safekid_ai.server.storage import ServerStorage


@pytest.fixture
def test_env():
    """Sets up an isolated temporary workspace directory for test execution."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        website_dir = Path("website").resolve()

        app = create_app(base_dir=tmp_path, website_dir=website_dir)
        client = Client(app)
        yield {
            "app": app,
            "client": client,
            "base_dir": tmp_path,
            "storage": app.storage,
            "notifications": app.notifications,
            "esp32": app.esp32,
            "job_runner": app.job_runner,
        }


# =============================================================================
# 1. REST API & STATIC SERVING TESTS
# =============================================================================

def test_static_index_serving(test_env):
    """Verifies that the main landing page serves index.html with HTTP 200."""
    client = test_env["client"]
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("Content-Type", "")
    assert "SafeKid AI" in res.get_data(as_text=True)
    assert "Control Room" in res.get_data(as_text=True)


def test_api_system_stats(test_env):
    """Verifies the /api/stats endpoint returns correct system summary counts."""
    client = test_env["client"]
    res = client.get("/api/stats")
    assert res.status_code == 200
    data = json.loads(res.get_data(as_text=True))
    assert data["status"] == "success"
    assert "total_jobs" in data["data"]
    assert "open_incidents" in data["data"]
    assert data["data"]["esp32_status"] in ("NOT CONNECTED", "UNCONFIGURED", "CONNECTED")


# =============================================================================
# 2. VIDEO UPLOAD & VALIDATION TESTS
# =============================================================================

def test_upload_valid_mp4_video(test_env):
    """Verifies single and multi-file MP4 upload, format validation, and job creation."""
    client = test_env["client"]

    # Create a small valid synthetic MP4 file in memory
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tmp_file = tf.name
    writer = cv2.VideoWriter(tmp_file, fourcc, 10.0, (160, 120))
    for _ in range(5):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    try:
        with open(tmp_file, "rb") as f:
            file_bytes = f.read()

        data = {
            "video": (io.BytesIO(file_bytes), "test_cctv_recording.mp4"),
        }
        res = client.post("/api/upload", data=data, content_type="multipart/form-data")
        assert res.status_code == 200
        resp_json = json.loads(res.get_data(as_text=True))
        assert resp_json["status"] == "success"
        assert resp_json["data"]["uploaded_count"] == 1
        job = resp_json["data"]["jobs"][0]
        assert job["filename"] == "test_cctv_recording.mp4"
        assert job["status"] == "QUEUED"
        assert job["total_frames"] == 5
    finally:
        Path(tmp_file).unlink(missing_ok=True)


def test_upload_invalid_file_extension(test_env):
    """Verifies that non-video file formats (.txt, .exe, .sh) are rejected."""
    client = test_env["client"]
    data = {
        "video": (io.BytesIO(b"malicious or invalid content"), "test_payload.txt"),
    }
    res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    resp_json = json.loads(res.get_data(as_text=True))
    assert resp_json["data"]["uploaded_count"] == 0
    assert len(resp_json["data"]["errors"]) == 1
    assert "not supported" in resp_json["data"]["errors"][0]


def test_load_demo_video_fixture(test_env):
    """Verifies loading pre-installed local demo fixtures into the job queue."""
    client = test_env["client"]
    res = client.post(
        "/api/jobs/demo",
        data=json.dumps({"demo_name": "pool_safety_demo.mp4"}),
        content_type="application/json",
    )
    assert res.status_code == 200
    resp_json = json.loads(res.get_data(as_text=True))
    assert resp_json["status"] == "success"
    job = resp_json["data"]["job"]
    assert job["filename"] == "pool_safety_demo.mp4"
    assert job["status"] == "QUEUED"
    assert job["total_frames"] == 240
    assert job["fps"] == 24.0


# =============================================================================
# 3. ROI CONFIGURATION & PERSISTENCE TESTS
# =============================================================================

def test_roi_get_and_save(test_env):
    """Verifies reading and saving camera ROI polygon configurations."""
    client = test_env["client"]

    # GET default ROI
    res_get = client.get("/api/roi")
    assert res_get.status_code == 200
    cfg = json.loads(res_get.get_data(as_text=True))["data"]
    assert "zones" in cfg
    assert len(cfg["zones"]) >= 1

    # POST updated ROI
    updated_cfg = {
        "community_id": "test_community",
        "camera_id": "test_camera",
        "frame_width": 640,
        "frame_height": 480,
        "zones": [
            {
                "roi_id": "zone_test_pool",
                "name": "Test Pool Basin",
                "zone_type": "SWIMMING_POOL_BOUNDARY",
                "severity": "CRITICAL",
                "enabled": True,
                "dwell_time_seconds": 1.5,
                "polygon": [[50, 50], [250, 50], [250, 250], [50, 250]],
                "policy": {
                    "zone_role": "DANGER",
                    "entry_alert": True,
                    "exit_alert": False,
                    "dwell_alert": True,
                    "dwell_threshold_seconds": 1.5,
                    "severity": "CRITICAL",
                },
            }
        ],
    }
    res_post = client.post(
        "/api/roi",
        data=json.dumps(updated_cfg),
        content_type="application/json",
    )
    assert res_post.status_code == 200
    data_post = json.loads(res_post.get_data(as_text=True))["data"]
    assert len(data_post["zones"]) == 1
    assert data_post["zones"][0]["name"] == "Test Pool Basin"


def test_roi_frame_extraction(test_env):
    """Verifies /api/roi/frame returns an encoded image frame for drawing polygons."""
    client = test_env["client"]
    res = client.get("/api/roi/frame")
    assert res.status_code == 200
    data = json.loads(res.get_data(as_text=True))["data"]
    assert "image_data" in data
    assert data["image_data"].startswith("data:image/jpeg;base64,")
    assert data["width"] > 0
    assert data["height"] > 0


# =============================================================================
# 4. INCIDENT LIFECYCLE & RETRIEVAL TESTS
# =============================================================================

def test_incident_acknowledgement_and_resolution(test_env):
    """Verifies operator incident state transitions (ACTIVE -> ACKNOWLEDGED -> RESOLVED)."""
    storage = test_env["storage"]
    client = test_env["client"]

    # Seed an incident
    inc_data = {
        "incident_id": "inc_test_001",
        "job_id": "job_001",
        "source_video": "cctv_pool.mp4",
        "camera_id": "cam_01",
        "community_id": "palm_grove",
        "track_id": 14,
        "classification": "CHILD",
        "primary_zone_id": "zone_swimming_pool",
        "primary_zone_name": "Swimming Pool Basin",
        "zone_type": "SWIMMING_POOL_BOUNDARY",
        "severity": "CRITICAL",
        "state": "ACTIVE",
        "created_at_frame": 12,
        "created_at_ms": 500.0,
        "timestamp_str": "2026-11-08 00:00:00",
        "summary_message": "Child entered hazardous swimming pool basin",
        "lifecycle_history": [],
    }
    storage.record_incident(inc_data)

    # 1. Retrieve via GET /api/incidents
    res_list = client.get("/api/incidents")
    assert res_list.status_code == 200
    incidents = json.loads(res_list.get_data(as_text=True))["data"]
    assert len(incidents) == 1
    assert incidents[0]["incident_id"] == "inc_test_001"
    assert incidents[0]["state"] == "ACTIVE"

    # 2. Acknowledge via POST /api/incidents/<id>/acknowledge
    res_ack = client.post(
        "/api/incidents/inc_test_001/acknowledge",
        data=json.dumps({"operator_name": "Guard John", "notes": "Dispatched patrol"}),
        content_type="application/json",
    )
    assert res_ack.status_code == 200
    ack_data = json.loads(res_ack.get_data(as_text=True))["data"]
    assert ack_data["state"] == "ACKNOWLEDGED"
    assert ack_data["acknowledged_by"] == "Guard John"

    # 3. Resolve via POST /api/incidents/<id>/resolve
    res_res = client.post(
        "/api/incidents/inc_test_001/resolve",
        data=json.dumps({"operator_name": "Guard John", "reason": "Child reunited with parent"}),
        content_type="application/json",
    )
    assert res_res.status_code == 200
    res_data = json.loads(res_res.get_data(as_text=True))["data"]
    assert res_data["state"] == "RESOLVED"
    assert res_data["resolution_reason"] == "Child reunited with parent"


# =============================================================================
# 5. NOTIFICATION ADAPTER TESTS
# =============================================================================

def test_notification_mock_provider(test_env):
    """Verifies notification dispatch and deduplication through Mock provider."""
    notif_svc = test_env["notifications"]
    notif_svc.save_config("MOCK")

    alert = IncidentAlert(
        incident_id="inc_notif_001",
        hazard_zone_name="Pool Deck",
        severity="CRITICAL",
        timestamp_ms=1000.0,
        camera_id="cam_01",
        track_id=3,
        classification="CHILD",
        description="Child approached water edge",
    )

    # First dispatch should succeed
    res1 = notif_svc.dispatch_incident_alert(alert, min_interval_seconds=10.0)
    assert res1.success is True
    assert res1.provider == "MOCK"

    # Immediate second dispatch within cooldown should be suppressed (deduplication)
    res2 = notif_svc.dispatch_incident_alert(alert, min_interval_seconds=10.0)
    assert res2.success is False
    assert "cooldown" in res2.message.lower()

    # Diagnostic test message
    res_test = notif_svc.send_test_message("Test community group ping")
    assert res_test.success is True


def test_notification_unconfigured_provider(test_env):
    """Verifies that an unconfigured provider cleanly reports NOT CONFIGURED and does not claim delivery."""
    notif_svc = test_env["notifications"]
    notif_svc.save_config("NONE")

    status = notif_svc.get_status()
    assert status["status"] == "NOT CONFIGURED"
    assert status["is_configured"] is False

    res = notif_svc.send_test_message("Test")
    assert res.success is False
    assert "NOT CONFIGURED" in res.message


# =============================================================================
# 6. ESP32 PHYSICAL BUZZER HARDWARE ADAPTER TESTS
# =============================================================================

def test_esp32_disconnected_device_handling(test_env):
    """Verifies that unreachable ESP32 hardware gracefully returns NOT CONNECTED and never fakes activation."""
    esp32 = test_env["esp32"]
    # Point to an unreachable IP/port
    esp32.save_config(host="127.0.0.1", port=59999, cooldown_seconds=5.0)

    # 1. Health check should return NOT CONNECTED
    health = esp32.check_health()
    assert health.connected is False
    assert health.status_label == "NOT CONNECTED"
    assert health.error_message is not None

    # 2. Trigger should return success=False and record failure reason
    record = esp32.trigger_buzzer(duration_ms=500, severity="CRITICAL", reason="Test trigger")
    assert record.success is False
    assert "failed" in (record.error or "").lower() or "connection" in (record.error or "").lower()

    # 3. Silence command should also fail gracefully
    silence_rec = esp32.silence_buzzer()
    assert silence_rec.success is False


def test_esp32_cooldown_enforcement(test_env):
    """Verifies that rapid repeated triggers are suppressed by cooldown policy."""
    esp32 = test_env["esp32"]
    esp32.cooldown_seconds = 60.0
    esp32._last_trigger_ms = time.time() * 1000.0  # simulate recent trigger

    record = esp32.trigger_buzzer(duration_ms=1000, severity="CRITICAL", command_type="TRIGGER")
    assert record.success is False
    assert "cooldown" in record.device_response.lower()


# =============================================================================
# 7. SECURITY & PATH TRAVERSAL GUARDS
# =============================================================================

def test_path_traversal_protection(test_env):
    """Verifies that malicious directory traversal attempts (../) are rejected with HTTP 403 or 404."""
    client = test_env["client"]
    res = client.get("/media/../etc/passwd")
    assert res.status_code in (403, 404)

    res_static = client.get("/../../secret.txt")
    assert res_static.status_code in (403, 404)


# =============================================================================
# 8. BACKGROUND PIPELINE EXECUTION ON REAL VIDEO
# =============================================================================

def test_job_runner_processes_video_pipeline(test_env):
    """Verifies that VideoJobRunner processes a video through the real SafeKidPipeline."""
    storage = test_env["storage"]
    job_runner = test_env["job_runner"]

    # Create a small 6-frame video fixture
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        vid_path = tf.name
    writer = cv2.VideoWriter(vid_path, fourcc, 10.0, (320, 240))
    for i in range(6):
        # Draw a synthetic pedestrian-like rectangle
        f = np.full((240, 320, 3), 40, dtype=np.uint8)
        cv2.rectangle(f, (100, 60), (160, 200), (200, 200, 200), -1)
        writer.write(f)
    writer.release()

    job_id = "job_test_e2e_001"
    job_data = {
        "job_id": job_id,
        "filename": "synthetic_test.mp4",
        "file_path": vid_path,
        "file_size_bytes": Path(vid_path).stat().st_size,
        "file_size_mb": 0.05,
        "status": "QUEUED",
        "progress_percent": 0.0,
        "current_frame": 0,
        "total_frames": 6,
        "fps": 10.0,
        "duration_seconds": 0.6,
        "incident_count": 0,
        "incidents": [],
    }
    storage.save_job(job_data)

    try:
        # Run processing directly in synchronous test call
        job_runner._process_video_job(job_id)

        # Check job completion state
        updated_job = storage.get_job(job_id)
        assert updated_job is not None
        assert updated_job["status"] == "COMPLETED"
        assert updated_job["progress_percent"] == 100.0
        assert updated_job["current_frame"] == 6

        # Check that annotated video was generated
        out_dir = storage.get_job_output_dir(job_id)
        annotated_file = out_dir / "annotated.mp4"
        assert annotated_file.exists()
        assert annotated_file.stat().st_size > 0
    finally:
        time.sleep(0.1)
        try:
            Path(vid_path).unlink(missing_ok=True)
        except Exception:
            pass
