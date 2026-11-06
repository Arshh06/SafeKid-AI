"""Abstract base class for person detector adapters in SafeKid AI."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

from safekid_ai.detector import FrameDetections


class BasePersonDetector(ABC):
    """Abstract interface defining standard contract for person detectors."""

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp_ms: float = 0.0,
        confidence_thresh: Optional[float] = None,
        iou_thresh: Optional[float] = None,
    ) -> FrameDetections:
        """Runs person detection on a single frame.
        
        Args:
            frame: OpenCV BGR image.
            frame_id: Frame sequence index.
            timestamp_ms: Milliseconds elapsed.
            confidence_thresh: Optional detection confidence threshold override.
            iou_thresh: Optional NMS IoU threshold override.
            
        Returns:
            Standardized FrameDetections container.
        """
        raise NotImplementedError

    @abstractmethod
    def track(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp_ms: float = 0.0,
        confidence_thresh: Optional[float] = None,
        iou_thresh: Optional[float] = None,
    ) -> FrameDetections:
        """Runs detection and multi-object tracking (ByteTrack) on a single frame."""
        raise NotImplementedError

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Returns the detector backend identifier (e.g. 'yolov8', 'rtdetrv2')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the specific model variant name (e.g. 'yolov8n', 'rtdetr-l')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def parameter_count(self) -> int:
        """Total trainable parameters in the model."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_size_mb(self) -> float:
        """Model weight size on disk in megabytes."""
        raise NotImplementedError

