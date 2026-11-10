"""WSGI Application and REST API for SafeKid AI Operational Security Control Room."""

import base64
import json
import mimetypes
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

import cv2
import numpy as np
from werkzeug.exceptions import BadRequest, HTTPException, NotFound
from werkzeug.http import http_date
from werkzeug.middleware.shared_data import SharedDataMiddleware
from werkzeug.wrappers import Request, Response

from safekid_ai.hardware.esp32_client import ESP32BuzzerClient
from safekid_ai.notifications.service import NotificationService
from safekid_ai.server.job_runner import VideoJobRunner
from safekid_ai.server.storage import ServerStorage
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.server.app")

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_UPLOAD_SIZE_BYTES = 250 * 1024 * 1024  # 250 MB


class SafeKidServerApp:
    """Production-grade WSGI web application serving REST endpoints and the operator UI."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        website_dir: Optional[Path] = None,
    ):
        self.base_dir = (base_dir or Path(".")).resolve()
        self.website_dir = (website_dir or (self.base_dir / "website")).resolve()

        self.storage = ServerStorage(base_dir=self.base_dir)
        self.notifications = NotificationService()
        self.esp32 = ESP32BuzzerClient()
        self.job_runner = VideoJobRunner(
            storage=self.storage,
            notification_service=self.notifications,
            esp32_client=self.esp32,
        )

    def __call__(self, environ: Dict[str, Any], start_response: Callable) -> Any:
        request = Request(environ)
        response = self.dispatch_request(request)
        return response(environ, start_response)

    def dispatch_request(self, req: Request) -> Response:
        """Routes incoming HTTP requests to corresponding endpoint handlers."""
        path = req.path.rstrip("/")
        if not path:
            path = "/"

        # CORS preflight
        if req.method == "OPTIONS":
            return self._build_cors_response(Response("", status=204))

        try:
            # 1. REST API Endpoints
            if path.startswith("/api/"):
                return self._handle_api(req, path[5:])

            # 2. Media Artifact Serving
            if path.startswith("/media/"):
                return self._handle_media(req, path[7:])

            # 3. Static Website Serving
            return self._handle_static(req, path)

        except HTTPException as e:
            return self._json_error(str(e.description), status=e.code)
        except Exception as e:
            logger.error(f"Unhandled server error on {req.method} {path}: {e}", exc_info=True)
            return self._json_error(f"Internal server error: {str(e)}", status=500)

    # =========================================================================
    # REST API ROUTER
    # =========================================================================

    def _handle_api(self, req: Request, subpath: str) -> Response:
        # POST /api/upload
        if subpath == "upload" and req.method == "POST":
            return self._api_upload_videos(req)

        # POST /api/jobs/demo
        if subpath == "jobs/demo" and req.method == "POST":
            return self._api_load_demo_video(req)

        # GET /api/jobs
        if subpath == "jobs" and req.method == "GET":
            return self._api_list_jobs(req)

        # /api/jobs/<job_id> or /api/jobs/<job_id>/<action>
        m_job = re.match(r"^jobs/([a-zA-Z0-9_-]+)(?:/([a-zA-Z0-9_-]+))?$", subpath)
        if m_job:
            job_id, action = m_job.group(1), m_job.group(2)
            if not action:
                if req.method == "GET":
                    return self._api_get_job(req, job_id)
                elif req.method == "DELETE":
                    return self._api_delete_job(req, job_id)
            elif action == "process" and req.method == "POST":
                return self._api_process_job(req, job_id)
            elif action == "cancel" and req.method == "POST":
                return self._api_cancel_job(req, job_id)

        # GET / POST /api/roi
        if subpath == "roi":
            if req.method == "GET":
                return self._api_get_roi(req)
            elif req.method == "POST":
                return self._api_save_roi(req)

        # GET /api/roi/frame
        if subpath == "roi/frame" and req.method == "GET":
            return self._api_get_roi_frame(req)

        # GET /api/incidents
        if subpath == "incidents" and req.method == "GET":
            return self._api_list_incidents(req)

        # POST /api/incidents/<id>/acknowledge
        m_inc_ack = re.match(r"^incidents/([a-zA-Z0-9_-]+)/acknowledge$", subpath)
        if m_inc_ack and req.method == "POST":
            return self._api_acknowledge_incident(req, m_inc_ack.group(1))

        # POST /api/incidents/<id>/resolve
        m_inc_res = re.match(r"^incidents/([a-zA-Z0-9_-]+)/resolve$", subpath)
        if m_inc_res and req.method == "POST":
            return self._api_resolve_incident(req, m_inc_res.group(1))

        # Notifications
        if subpath == "notifications/config":
            if req.method == "GET":
                return self._json_success(self.notifications.get_status())
            elif req.method == "POST":
                return self._api_save_notification_config(req)
        if subpath == "notifications/test" and req.method == "POST":
            return self._api_test_notification(req)
        if subpath == "notifications/history" and req.method == "GET":
            return self._json_success(self.notifications.get_history())

        # ESP32 Buzzer
        if subpath == "esp32/status" and req.method == "GET":
            return self._json_success(self.esp32.get_status())
        if subpath == "esp32/config" and req.method == "POST":
            return self._api_save_esp32_config(req)
        if subpath == "esp32/buzzer/test" and req.method == "POST":
            return self._api_test_esp32_buzzer(req)
        if subpath == "esp32/buzzer/silence" and req.method == "POST":
            return self._api_silence_esp32_buzzer(req)
        if subpath == "esp32/history" and req.method == "GET":
            return self._json_success(self.esp32.get_status().get("recent_history", []))

        # System stats
        if subpath == "stats" and req.method == "GET":
            return self._api_get_system_stats(req)

        return self._json_error(f"Endpoint not found: /api/{subpath}", status=404)

    # =========================================================================
    # ENDPOINT IMPLEMENTATIONS
    # =========================================================================

    def _api_upload_videos(self, req: Request) -> Response:
        """Handles single or multi-file video uploads with validation and job creation."""
        files = req.files.getlist("video") or req.files.getlist("files") or req.files.getlist("file")
        if not files:
            return self._json_error("No video file provided in multipart upload.", status=400)

        created_jobs = []
        errors = []

        for f in files:
            orig_filename = f.filename or "upload.mp4"
            ext = Path(orig_filename).suffix.lower()

            if ext not in ALLOWED_VIDEO_EXTENSIONS:
                errors.append(f"File '{orig_filename}' rejected: extension '{ext}' not supported (must be MP4, AVI, or MOV).")
                continue

            job_id = f"job_{uuid.uuid4().hex[:8]}"
            save_path = self.storage.get_upload_path(job_id, orig_filename)

            try:
                f.save(str(save_path))
                file_size = save_path.stat().st_size

                if file_size > MAX_UPLOAD_SIZE_BYTES:
                    save_path.unlink(missing_ok=True)
                    errors.append(f"File '{orig_filename}' exceeds 250MB limit ({file_size / (1024*1024):.1f}MB).")
                    continue

                # Verify file can be read by OpenCV
                test_cap = cv2.VideoCapture(str(save_path))
                if not test_cap.isOpened():
                    test_cap.release()
                    save_path.unlink(missing_ok=True)
                    errors.append(f"File '{orig_filename}' is corrupted or contains an unreadable video stream.")
                    continue

                total_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(test_cap.get(cv2.CAP_PROP_FPS))
                test_cap.release()

                job_data = {
                    "job_id": job_id,
                    "filename": orig_filename,
                    "file_path": str(save_path),
                    "file_size_bytes": file_size,
                    "file_size_mb": round(file_size / (1024 * 1024), 2),
                    "status": "QUEUED",
                    "progress_percent": 0.0,
                    "current_frame": 0,
                    "total_frames": total_frames,
                    "fps": round(fps, 2) if fps > 0 else 25.0,
                    "duration_seconds": round(total_frames / fps, 1) if (fps > 0 and total_frames > 0) else 0.0,
                    "incident_count": 0,
                    "incidents": [],
                    "created_at_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                self.storage.save_job(job_data)
                created_jobs.append(job_data)

            except Exception as e:
                logger.error(f"Error saving uploaded file '{orig_filename}': {e}")
                errors.append(f"Failed to process '{orig_filename}': {str(e)}")

        return self._json_success({
            "uploaded_count": len(created_jobs),
            "jobs": created_jobs,
            "errors": errors,
        })

    def _api_load_demo_video(self, req: Request) -> Response:
        """Loads one of the local residential demo video fixtures into the processing queue."""
        data = self._parse_json_body(req)
        demo_name = data.get("demo_name", "pool_safety_demo.mp4")

        # Map to known demo files
        demo_map = {
            "pool": "data/demo/pool_safety_demo.mp4",
            "pool_safety_demo.mp4": "data/demo/pool_safety_demo.mp4",
            "driveway": "data/demo/driveway_demo.mp4",
            "driveway_demo.mp4": "data/demo/driveway_demo.mp4",
            "playground": "data/demo/playground_demo.mp4",
            "playground_demo.mp4": "data/demo/playground_demo.mp4",
            "gate": "data/demo/boom_barrier_demo.mp4",
            "boom_barrier_demo.mp4": "data/demo/boom_barrier_demo.mp4",
        }
        rel_path = demo_map.get(demo_name, "data/demo/pool_safety_demo.mp4")
        full_path = (self.base_dir / rel_path).resolve()
        if not full_path.exists():
            full_path = (Path.cwd() / rel_path).resolve()

        if not full_path.exists():
            return self._json_error(f"Demo video file '{demo_name}' not found at {full_path}", status=404)

        job_id = f"job_demo_{uuid.uuid4().hex[:6]}"
        test_cap = cv2.VideoCapture(str(full_path))
        total_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(test_cap.get(cv2.CAP_PROP_FPS))
        test_cap.release()

        job_data = {
            "job_id": job_id,
            "filename": full_path.name,
            "file_path": str(full_path),
            "file_size_bytes": full_path.stat().st_size,
            "file_size_mb": round(full_path.stat().st_size / (1024 * 1024), 2),
            "status": "QUEUED",
            "progress_percent": 0.0,
            "current_frame": 0,
            "total_frames": total_frames,
            "fps": round(fps, 2) if fps > 0 else 24.0,
            "duration_seconds": round(total_frames / fps, 1) if (fps > 0 and total_frames > 0) else 0.0,
            "incident_count": 0,
            "incidents": [],
            "created_at_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "is_demo_fixture": True,
        }
        self.storage.save_job(job_data)
        return self._json_success({"job": job_data, "message": f"Loaded demo video fixture '{full_path.name}'"})

    def _api_list_jobs(self, req: Request) -> Response:
        jobs = self.storage.get_jobs()
        return self._json_success(jobs)

    def _api_get_job(self, req: Request, job_id: str) -> Response:
        job = self.storage.get_job(job_id)
        if not job:
            return self._json_error(f"Job {job_id} not found", status=404)
        return self._json_success(job)

    def _api_delete_job(self, req: Request, job_id: str) -> Response:
        success = self.storage.delete_job(job_id)
        if not success:
            return self._json_error(f"Job {job_id} not found or could not be removed", status=404)
        return self._json_success({"deleted": True, "job_id": job_id})

    def _api_process_job(self, req: Request, job_id: str) -> Response:
        started = self.job_runner.start_job(job_id)
        if not started:
            job = self.storage.get_job(job_id)
            if not job:
                return self._json_error(f"Job {job_id} not found", status=404)
            return self._json_error(f"Job {job_id} could not be started (already running or invalid state).", status=400)
        return self._json_success({"started": True, "job_id": job_id})

    def _api_cancel_job(self, req: Request, job_id: str) -> Response:
        cancelled = self.job_runner.cancel_job(job_id)
        return self._json_success({"cancelled": cancelled, "job_id": job_id})

    def _api_get_roi(self, req: Request) -> Response:
        cfg = self.storage.get_roi_config()
        return self._json_success(cfg.to_dict())

    def _api_save_roi(self, req: Request) -> Response:
        data = self._parse_json_body(req)
        try:
            cfg = self.storage.save_roi_config(data)
            return self._json_success(cfg.to_dict())
        except Exception as e:
            return self._json_error(f"Invalid ROI configuration: {str(e)}", status=400)

    def _api_get_roi_frame(self, req: Request) -> Response:
        """Extracts a representative frame for the interactive ROI polygon editor."""
        job_id = req.args.get("job_id")
        video_path = None

        if job_id:
            job = self.storage.get_job(job_id)
            if job and Path(job.get("file_path", "")).exists():
                video_path = Path(job["file_path"])

        if not video_path:
            # Fallback to pool demo video or create a clean baseline
            default_demo = self.base_dir / "data/demo/pool_safety_demo.mp4"
            if default_demo.exists():
                video_path = default_demo

        frame = None
        if video_path and video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                ret, frame = cap.read()
            cap.release()

        if frame is None:
            # Create a synthetic 848x480 frame for drawing
            frame = np.full((480, 848, 3), 35, dtype=np.uint8)
            cv2.putText(frame, "SafeKid AI CCTV Frame (No Video Loaded)", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            return self._json_error("Could not encode frame to JPEG", status=500)

        b64_img = base64.b64encode(buffer).decode("ascii")
        return self._json_success({
            "width": frame.shape[1],
            "height": frame.shape[0],
            "image_data": f"data:image/jpeg;base64,{b64_img}",
            "source": video_path.name if video_path else "synthetic",
        })

    def _api_list_incidents(self, req: Request) -> Response:
        incidents = self.storage.get_incidents(
            job_id=req.args.get("job_id"),
            severity=req.args.get("severity"),
            zone=req.args.get("zone"),
            status=req.args.get("status"),
            classification=req.args.get("classification"),
        )
        return self._json_success(incidents)

    def _api_acknowledge_incident(self, req: Request, incident_id: str) -> Response:
        data = self._parse_json_body(req)
        operator_name = data.get("operator_name", "Control Room Operator")
        notes = data.get("notes", "")
        updated = self.storage.acknowledge_incident(incident_id, operator_name, notes)
        if not updated:
            return self._json_error(f"Incident {incident_id} not found", status=404)
        return self._json_success(updated)

    def _api_resolve_incident(self, req: Request, incident_id: str) -> Response:
        data = self._parse_json_body(req)
        operator_name = data.get("operator_name", "Control Room Operator")
        reason = data.get("reason", "Resolved by operator")
        updated = self.storage.resolve_incident(incident_id, operator_name, reason)
        if not updated:
            return self._json_error(f"Incident {incident_id} not found", status=404)
        return self._json_success(updated)

    def _api_save_notification_config(self, req: Request) -> Response:
        data = self._parse_json_body(req)
        self.notifications.save_config(
            provider_type=data.get("provider_type", "NONE"),
            webhook_url=data.get("webhook_url"),
            telegram_token=data.get("telegram_token"),
            telegram_chat_id=data.get("telegram_chat_id"),
        )
        return self._json_success(self.notifications.get_status())

    def _api_test_notification(self, req: Request) -> Response:
        data = self._parse_json_body(req)
        msg = data.get("message", "Test alert dispatched from SafeKid AI Control Room.")
        res = self.notifications.send_test_message(msg)
        return self._json_success(res.to_dict())

    def _api_save_esp32_config(self, req: Request) -> Response:
        data = self._parse_json_body(req)
        self.esp32.save_config(
            host=data.get("host", "192.168.1.120"),
            port=int(data.get("port", 80)),
            auth_token=data.get("auth_token"),
            cooldown_seconds=float(data.get("cooldown_seconds", 10.0)),
            default_duration_ms=int(data.get("default_duration_ms", 1500)),
        )
        return self._json_success(self.esp32.get_status())

    def _api_test_esp32_buzzer(self, req: Request) -> Response:
        data = self._parse_json_body(req)
        duration = int(data.get("duration_ms", 1000))
        res = self.esp32.trigger_buzzer(
            duration_ms=duration,
            severity="CRITICAL",
            reason="Manual control room test pulse",
            command_type="TEST",
        )
        return self._json_success(res.to_dict())

    def _api_silence_esp32_buzzer(self, req: Request) -> Response:
        res = self.esp32.silence_buzzer()
        return self._json_success(res.to_dict())

    def _api_get_system_stats(self, req: Request) -> Response:
        jobs = self.storage.get_jobs()
        incidents = self.storage.get_incidents()
        open_incidents = [i for i in incidents if i.get("state") in ("ACTIVE", "ACKNOWLEDGED")]
        esp32_status = self.esp32.get_status()
        notif_status = self.notifications.get_status()

        return self._json_success({
            "total_jobs": len(jobs),
            "completed_jobs": sum(1 for j in jobs if j.get("status") == "COMPLETED"),
            "processing_jobs": sum(1 for j in jobs if j.get("status") == "PROCESSING"),
            "total_incidents": len(incidents),
            "open_incidents": len(open_incidents),
            "esp32_status": esp32_status["health"]["status"],
            "esp32_connected": esp32_status["health"]["connected"],
            "notification_status": notif_status["status"],
            "notification_provider": notif_status["provider_type"],
            "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    # =========================================================================
    # STATIC & MEDIA FILE SERVING
    # =========================================================================

    def _handle_media(self, req: Request, subpath: str) -> Response:
        """Safely serves uploaded videos, annotated outputs, and incident snapshots."""
        media_root = self.base_dir / "data"
        file_path = (media_root / subpath).resolve()

        # Path traversal guard
        try:
            file_path.relative_to(media_root)
        except ValueError:
            return self._json_error("Forbidden: access outside data directory is disallowed", status=403)

        if not file_path.exists() or not file_path.is_file():
            return self._json_error(f"Media file not found: {subpath}", status=404)

        return self._serve_file(file_path)

    def _handle_static(self, req: Request, path: str) -> Response:
        """Safely serves HTML, CSS, JS, and data files from website/ directory."""
        if path == "/" or not path:
            file_path = self.website_dir / "index.html"
        else:
            rel_path = path.lstrip("/")
            file_path = (self.website_dir / rel_path).resolve()

        # Path traversal guard
        try:
            file_path.relative_to(self.website_dir)
        except ValueError:
            return self._json_error("Forbidden", status=403)

        if not file_path.exists() or not file_path.is_file():
            # If requesting a route without extension, fall back to index.html (SPA support)
            if "." not in path:
                file_path = self.website_dir / "index.html"
            else:
                return self._json_error(f"File not found: {path}", status=404)

        return self._serve_file(file_path)

    def _serve_file(self, path: Path) -> Response:
        """Constructs an HTTP response with appropriate MIME type and caching headers."""
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = "application/octet-stream"

        try:
            stat = path.stat()
            headers = {
                "Content-Type": mime_type,
                "Content-Length": str(stat.st_size),
                "Last-Modified": http_date(stat.st_mtime),
                "Cache-Control": "no-cache, must-revalidate",
                "Access-Control-Allow-Origin": "*",
            }
            # For MP4 video, support Accept-Ranges
            if mime_type == "video/mp4":
                headers["Accept-Ranges"] = "bytes"

            with open(path, "rb") as f:
                content = f.read()

            return Response(content, status=200, headers=headers)
        except Exception as e:
            logger.error(f"Error serving file {path}: {e}")
            return self._json_error(f"Error reading file: {str(e)}", status=500)

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _parse_json_body(self, req: Request) -> Dict[str, Any]:
        try:
            data = req.get_json(force=True, silent=True)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _json_success(self, data: Any, status: int = 200) -> Response:
        body = json.dumps({"status": "success", "data": data}, indent=2)
        resp = Response(body, status=status, mimetype="application/json")
        return self._build_cors_response(resp)

    def _json_error(self, message: str, status: int = 400) -> Response:
        body = json.dumps({"status": "error", "message": message}, indent=2)
        resp = Response(body, status=status, mimetype="application/json")
        return self._build_cors_response(resp)

    def _build_cors_response(self, resp: Response) -> Response:
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return resp


def create_app(base_dir: Optional[Path] = None, website_dir: Optional[Path] = None) -> SafeKidServerApp:
    return SafeKidServerApp(base_dir=base_dir, website_dir=website_dir)
