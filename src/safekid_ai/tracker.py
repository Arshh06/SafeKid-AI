"""Person tracking using ByteTrack for SafeKid AI."""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from safekid_ai.config import TrackerConfig
from safekid_ai.detector import DetectionBox, FrameDetections
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.tracker")


@dataclass
class TrackedPerson:
    """Represents a person tracked continuously over time."""
    track_id: int
    current_box: DetectionBox
    history_centroids: deque = field(default_factory=deque)
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    time_in_zone_seconds: float = 0.0

    def update(self, box: DetectionBox, frame_id: int, max_history: int = 60) -> None:
        """Updates the track with a new detection box."""
        self.current_box = box
        self.last_seen_frame = frame_id
        self.history_centroids.append(box.center)
        while len(self.history_centroids) > max_history:
            self.history_centroids.popleft()

    @property
    def velocity_vector(self) -> Tuple[float, float]:
        """Calculates displacement vector over recent frames (dx, dy)."""
        if len(self.history_centroids) < 2:
            return (0.0, 0.0)
        c_recent = self.history_centroids[-1]
        c_old = self.history_centroids[0]
        frames_elapsed = len(self.history_centroids) - 1
        return ((c_recent[0] - c_old[0]) / frames_elapsed, (c_recent[1] - c_old[1]) / frames_elapsed)


class PersonTracker:
    """ByteTrack-based person tracker for residential community safety monitoring."""

    def __init__(self, detector_model: YOLO, config: Optional[TrackerConfig] = None):
        """Initializes ByteTrack person tracker.

        Args:
            detector_model: Initialized YOLO model instance.
            config: Tracker configuration.
        """
        self.yolo = detector_model
        self.config = config or TrackerConfig()
        self.active_tracks: Dict[int, TrackedPerson] = {}
        logger.info(f"Initialized PersonTracker with tracker config: {self.config.tracker_config_path}")

    def reset(self) -> None:
        """Resets all tracking states for a fresh video or camera stream."""
        self.active_tracks.clear()
        if hasattr(self.yolo, "predictor") and self.yolo.predictor is not None:
            if hasattr(self.yolo.predictor, "trackers"):
                self.yolo.predictor.trackers = None
        logger.info("PersonTracker state reset.")

    def track(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp_ms: float = 0.0,
        confidence_thresh: Optional[float] = None,
        iou_thresh: Optional[float] = None,
    ) -> FrameDetections:
        """Runs person detection and ByteTrack tracking on a video frame.

        Args:
            frame: OpenCV BGR image array.
            frame_id: Frame counter.
            timestamp_ms: Timestamp in milliseconds.
            confidence_thresh: Minimum detection confidence (falls back to config if None).
            iou_thresh: IoU threshold for NMS (falls back to config if None).

        Returns:
            FrameDetections with assigned track_id for each person.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None.")

        conf = confidence_thresh if confidence_thresh is not None else getattr(self.config, "confidence_threshold", 0.35)
        iou = iou_thresh if iou_thresh is not None else 0.45

        # COCO class 0 is 'person'
        imgsz = getattr(self.config, "imgsz", 640)
        results = self.yolo.track(
            source=frame,
            persist=self.config.persist,
            tracker=self.config.tracker_config_path,
            classes=[0],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
        )

        detections: List[DetectionBox] = []
        raw_res = results[0] if len(results) > 0 else None

        if raw_res is not None and raw_res.boxes is not None:
            boxes = raw_res.boxes.cpu()
            xyxy_arr = boxes.xyxy.numpy()
            conf_arr = boxes.conf.numpy()
            cls_arr = boxes.cls.numpy()
            track_ids = boxes.id.numpy() if boxes.id is not None else [None] * len(xyxy_arr)

            for i in range(len(xyxy_arr)):
                cid = int(cls_arr[i])
                cname = raw_res.names.get(cid, "person")
                tid = int(track_ids[i]) if track_ids[i] is not None else None
                bbox = (
                    float(xyxy_arr[i][0]),
                    float(xyxy_arr[i][1]),
                    float(xyxy_arr[i][2]),
                    float(xyxy_arr[i][3]),
                )

                det_box = DetectionBox(
                    bbox_xyxy=bbox,
                    confidence=float(conf_arr[i]),
                    class_id=cid,
                    class_name=cname,
                    track_id=tid,
                )
                detections.append(det_box)

                # Maintain history
                if tid is not None:
                    if tid not in self.active_tracks:
                        tracked = TrackedPerson(
                            track_id=tid,
                            current_box=det_box,
                            first_seen_frame=frame_id,
                            last_seen_frame=frame_id,
                        )
                        tracked.history_centroids.append(det_box.center)
                        self.active_tracks[tid] = tracked
                    else:
                        self.active_tracks[tid].update(
                            det_box,
                            frame_id=frame_id,
                            max_history=self.config.track_history_length,
                        )

        return FrameDetections(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            detections=detections,
            raw_results=raw_res,
        )

