"""YOLOv8 baseline detector adapter for SafeKid AI."""

from pathlib import Path
from typing import Optional
import numpy as np
import torch
from ultralytics import YOLO

from safekid_ai.config import DetectorConfig, TrackerConfig
from safekid_ai.detector import DetectionBox, FrameDetections, PersonDetector
from safekid_ai.detectors.base import BasePersonDetector
from safekid_ai.tracker import PersonTracker
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.detectors.yolov8")


class YOLOv8DetectorAdapter(BasePersonDetector):
    """Adapter wrapping YOLOv8 baseline for the comparative study."""

    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        tracker_config: Optional[TrackerConfig] = None,
    ):
        self.config = config or DetectorConfig()
        self.tracker_config = tracker_config or TrackerConfig()
        self.detector = PersonDetector(self.config)
        self.tracker = PersonTracker(self.detector.model, self.tracker_config)
        self._param_count = sum(p.numel() for p in self.detector.model.parameters())
        
        # Calculate file size
        p = Path(self.config.model_name_or_path)
        self._size_mb = (p.stat().st_size / (1024 * 1024)) if p.exists() else 6.2

    @property
    def backend_name(self) -> str:
        return "yolov8"

    @property
    def model_name(self) -> str:
        return Path(self.config.model_name_or_path).stem

    @property
    def parameter_count(self) -> int:
        return self._param_count

    @property
    def model_size_mb(self) -> float:
        return round(self._size_mb, 2)

    @property
    def model(self) -> YOLO:
        return self.detector.model

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp_ms: float = 0.0,
        confidence_thresh: Optional[float] = None,
        iou_thresh: Optional[float] = None,
    ) -> FrameDetections:
        """Runs detection on frame."""
        return self.detector.detect(
            frame=frame,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            confidence_thresh=confidence_thresh,
            iou_thresh=iou_thresh,
        )

    def track(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp_ms: float = 0.0,
        confidence_thresh: Optional[float] = None,
        iou_thresh: Optional[float] = None,
    ) -> FrameDetections:
        """Runs detection and ByteTrack tracking on frame."""
        return self.tracker.track(
            frame=frame,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            confidence_thresh=confidence_thresh,
            iou_thresh=iou_thresh,
        )

