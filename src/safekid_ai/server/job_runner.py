import gc
"""Asynchronous Video Processing Job Runner for SafeKid AI."""

from dataclasses import asdict
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set

import cv2
import numpy as np

from safekid_ai.config import (
    AgeEstimatorConfig,
    DetectorConfig,
    OperationalMode,
    SafeKidPipelineConfig,
    TrackerConfig,
)
from safekid_ai.child_detection.config import ChildDetectorConfig
from safekid_ai.hardware.esp32_client import ESP32BuzzerClient
from safekid_ai.incidents import Incident, IncidentState
from safekid_ai.notifications.provider import IncidentAlert
from safekid_ai.notifications.service import NotificationService
from safekid_ai.pipeline import SafeKidPipeline
from safekid_ai.roi.models import AlertSeverity, CameraROIConfig
from safekid_ai.server.storage import ServerStorage
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.server.job_runner")


class VideoJobRunner:
    """Manages background asynchronous execution of SafeKid AI video processing jobs."""

    def __init__(
        self,
        storage: ServerStorage,
        notification_service: NotificationService,
        esp32_client: ESP32BuzzerClient,
    ):
        self.storage = storage
        self.notifications = notification_service
        self.esp32 = esp32_client

        self._active_threads: Dict[str, threading.Thread] = {}
        self._cancel_flags: Dict[str, bool] = {}
        self._lock = threading.Lock()

    def is_job_running(self, job_id: str) -> bool:
        with self._lock:
            thread = self._active_threads.get(job_id)
            return bool(thread and thread.is_alive())

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._active_threads and self._active_threads[job_id].is_alive():
                self._cancel_flags[job_id] = True
                logger.info(f"Cancellation requested for job {job_id}")
                return True
        return False

    def start_job(self, job_id: str) -> bool:
        """Starts video processing in a background worker thread."""
        with self._lock:
            if job_id in self._active_threads and self._active_threads[job_id].is_alive():
                logger.warning(f"Job {job_id} is already actively processing.")
                return False

            job_data = self.storage.get_job(job_id)
            if not job_data:
                logger.error(f"Cannot start nonexistent job {job_id}")
                return False

            self._cancel_flags[job_id] = False
            t = threading.Thread(target=self._process_video_job, args=(job_id,), daemon=True)
            self._active_threads[job_id] = t
            t.start()
            logger.info(f"Launched background worker thread for job {job_id}")
            return True

    def _process_video_job(self, job_id: str) -> None:
        """Core video processing loop executing the actual SafeKid AI vision pipeline."""
        job_data = self.storage.get_job(job_id)
        if not job_data:
            return

        video_path = Path(job_data.get("file_path", ""))
        if not video_path.exists():
            job_data["status"] = "FAILED"
            job_data["error_message"] = f"Video source file not found: {video_path}"
            self.storage.save_job(job_data)
            return

        out_dir = self.storage.get_job_output_dir(job_id)
        annotated_video_path = out_dir / "annotated.mp4"
        thumb_dir = out_dir / "thumbnails"

        # Update initial job state
        job_data["status"] = "PROCESSING"
        job_data["started_at_ms"] = time.time() * 1000.0
        job_data["progress_percent"] = 0.0
        job_data["current_frame"] = 0
        job_data["incident_count"] = 0
        job_data["incidents"] = []
        self.storage.save_job(job_data)

        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            job_data["status"] = "FAILED"
            job_data["error_message"] = f"Failed to open video file {video_path.name} (codec or format unsupported)"
            self.storage.save_job(job_data)
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0 or np.isnan(fps):
            fps = 25.0
        frame_interval_ms = (1000.0 / fps) if fps > 0 else 40.0

        job_data["total_frames"] = total_frames
        job_data["frame_width"] = width
        job_data["frame_height"] = height
        job_data["fps"] = round(fps, 2)
        job_data["duration_seconds"] = round(total_frames / fps, 1) if total_frames > 0 else 0.0
        self.storage.save_job(job_data)

        # Retrieve active ROI configuration
        roi_config = self.storage.get_roi_config()
        # Calibrate ROI resolution if needed
        roi_config.frame_width = width
        roi_config.frame_height = height

        # Build pipeline configuration
        model_path = "models/child_classifier_jaad_v1.pt"
        if not Path(model_path).exists():
            model_path = "models/child_classifier_v1.pt"

        pipeline_cfg = SafeKidPipelineConfig(
            mode=OperationalMode.COMMUNITY_MONITORING,
            community_name="Gated Residential Community",
            community_id=roi_config.community_id,
            camera_id=roi_config.camera_id,
            detector=DetectorConfig(
                model_name_or_path="yolov8n.pt",
                device="cpu",
                confidence_threshold=0.35,
                iou_threshold=0.45,
            ),
            tracker=TrackerConfig(tracker_config_path="bytetrack.yaml"),
            age_estimator=AgeEstimatorConfig(device="cpu"),
            enable_automated_child_detection=True,
            child_detector_config={
                "model_weights_path": model_path if Path(model_path).exists() else None,
                "confidence_threshold": 0.60,
                "device": "cpu",
                "provisional_review_required": True,
            },
            enable_incident_management=True,
        )

        try:
            pipeline = SafeKidPipeline(pipeline_cfg)
            pipeline.set_roi_config(roi_config)
        except Exception as e:
            logger.error(f"Failed to initialize SafeKidPipeline: {e}", exc_info=True)
            job_data["status"] = "FAILED"
            job_data["error_message"] = f"Pipeline initialization error: {str(e)}"
            self.storage.save_job(job_data)
            cap.release()
            return

        # Prepare VideoWriter for annotated output
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (width, height))

        frame_idx = 0
        seen_incident_ids: Set[str] = set()
        recorded_incidents: List[Dict[str, Any]] = []
        last_progress_save = time.time()
        start_processing_time = time.time()

        try:
            while cap.isOpened():
                # Check for cancellation
                if self._cancel_flags.get(job_id, False):
                    logger.info(f"Job {job_id} cancelled by operator.")
                    job_data["status"] = "CANCELLED"
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                timestamp_ms = frame_idx * frame_interval_ms
                res = pipeline.process_frame(
                    frame=frame,
                    frame_id=frame_idx,
                    timestamp_ms=timestamp_ms,
                    annotate=True,
                )

                # Write annotated frame
                out_frame = res.annotated_frame if res.annotated_frame is not None else frame
                writer.write(out_frame)

                # Evaluate new incidents generated by IncidentManager
                if pipeline.incident_manager:
                    for inc in pipeline.incident_manager.all_incidents:
                        if inc.incident_id not in seen_incident_ids:
                            seen_incident_ids.add(inc.incident_id)

                            # Save representative thumbnail
                            thumb_filename = f"{inc.incident_id}.jpg"
                            thumb_path = thumb_dir / thumb_filename
                            cv2.imwrite(str(thumb_path), out_frame)
                            thumb_url = f"/media/jobs/{job_id}/thumbnails/{thumb_filename}"

                            inc_dict = {
                                "incident_id": inc.incident_id,
                                "job_id": job_id,
                                "source_video": video_path.name,
                                "camera_id": inc.camera_id,
                                "community_id": inc.community_id,
                                "track_id": inc.track_id,
                                "classification": inc.classification.value if hasattr(inc.classification, "value") else str(inc.classification),
                                "primary_zone_id": inc.primary_zone_id,
                                "primary_zone_name": inc.primary_zone_name,
                                "zone_type": inc.zone_type.value if hasattr(inc.zone_type, "value") else str(inc.zone_type),
                                "severity": inc.severity.value if hasattr(inc.severity, "value") else str(inc.severity),
                                "state": inc.state.value if hasattr(inc.state, "value") else str(inc.state),
                                "created_at_frame": inc.created_at_frame,
                                "created_at_ms": inc.created_at_ms,
                                "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                                "thumbnail_url": thumb_url,
                                "summary_message": inc.summary_message,
                                "observation_status": getattr(inc, "observation_status", "OBSERVING").value if hasattr(getattr(inc, "observation_status", None), "value") else str(getattr(inc, "observation_status", "OBSERVING")),
                                "lifecycle_history": inc.lifecycle_history,
                                "notification_status": "PENDING",
                                "buzzer_status": "PENDING",
                            }

                            # Dispatch automated community alert if configured and severity is HIGH/CRITICAL
                            sev_str = str(inc.severity.value if hasattr(inc.severity, "value") else inc.severity).upper()
                            if sev_str in ("CRITICAL", "HIGH"):
                                alert = IncidentAlert(
                                    incident_id=inc.incident_id,
                                    hazard_zone_name=inc.primary_zone_name,
                                    severity=sev_str,
                                    timestamp_ms=inc.created_at_ms,
                                    camera_id=inc.camera_id,
                                    track_id=inc.track_id,
                                    classification=inc_dict["classification"],
                                    description=inc.summary_message,
                                    source_video=video_path.name,
                                    frame_id=inc.created_at_frame,
                                )
                                dispatch_res = self.notifications.dispatch_incident_alert(alert)
                                inc_dict["notification_status"] = "DELIVERED" if dispatch_res.success else (
                                    "SUPPRESSED" if "cooldown" in dispatch_res.message.lower() else "FAILED"
                                )
                                inc_dict["notification_message"] = dispatch_res.message

                            # Dispatch automated physical buzzer trigger if severity is CRITICAL
                            if sev_str == "CRITICAL":
                                buzzer_res = self.esp32.trigger_buzzer(
                                    duration_ms=1500,
                                    incident_id=inc.incident_id,
                                    severity=sev_str,
                                    reason=f"Child entered {inc.primary_zone_name}",
                                )
                                inc_dict["buzzer_status"] = "TRIGGERED" if buzzer_res.success else (
                                    "SUPPRESSED" if "cooldown" in buzzer_res.device_response.lower() else "NOT_CONNECTED"
                                )
                                inc_dict["buzzer_message"] = buzzer_res.device_response or buzzer_res.error or ""

                            self.storage.record_incident(inc_dict)
                            recorded_incidents.append(inc_dict)

                frame_idx += 1

                # Periodically update progress in storage (every ~1.0s or 25 frames)
                now = time.time()
                if (now - last_progress_save) > 1.0 or frame_idx == total_frames:
                    pct = round((frame_idx / total_frames) * 100.0, 1) if total_frames > 0 else 0.0
                    fps_proc = round(frame_idx / (now - start_processing_time), 1) if (now - start_processing_time) > 0 else 0.0
                    job_data["current_frame"] = frame_idx
                    job_data["progress_percent"] = pct
                    job_data["processing_fps"] = fps_proc
                    job_data["incident_count"] = len(recorded_incidents)
                    job_data["incidents"] = recorded_incidents
                    self.storage.save_job(job_data)
                    last_progress_save = now

            # Video finished
            writer.release()
            cap.release()
            del writer
            del cap
            gc.collect()

            if job_data.get("status") != "CANCELLED":
                job_data["status"] = "COMPLETED"
                job_data["progress_percent"] = 100.0
                job_data["current_frame"] = frame_idx
                job_data["completed_at_ms"] = time.time() * 1000.0
                job_data["annotated_video_url"] = f"/media/jobs/{job_id}/annotated.mp4"
                job_data["incident_count"] = len(recorded_incidents)
                job_data["incidents"] = recorded_incidents
                self.storage.save_job(job_data)
                logger.info(f"Job {job_id} successfully completed. {len(recorded_incidents)} incident(s) detected.")

        except Exception as e:
            logger.error(f"Error during video processing loop for job {job_id}: {e}", exc_info=True)
            writer.release()
            cap.release()
            job_data["status"] = "FAILED"
            job_data["error_message"] = f"Runtime processing exception: {str(e)}"
            self.storage.save_job(job_data)
        finally:
            with self._lock:
                self._active_threads.pop(job_id, None)
                self._cancel_flags.pop(job_id, None)
