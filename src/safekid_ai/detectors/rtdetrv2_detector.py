"""RT-DETRv2 (Real-Time Detection Transformer v2) detector adapter for SafeKid AI."""

import time
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from safekid_ai.config import DetectorConfig, TrackerConfig
from safekid_ai.detector import DetectionBox, FrameDetections
from safekid_ai.detectors.base import BasePersonDetector
from safekid_ai.tracker import PersonTracker
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.detectors.rtdetrv2")


class RTDETRv2DetectorAdapter(BasePersonDetector):
    """Adapter wrapping official pretrained RT-DETRv2 for comparative object detection."""

    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        tracker_config: Optional[TrackerConfig] = None,
        checkpoint_path: str = "rtdetr-l.pt",
    ):
        self.config = config or DetectorConfig()
        self.tracker_config = tracker_config or TrackerConfig()
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(self.config.device if torch.cuda.is_available() or self.config.device == "cpu" else "cpu")
        
        self.model = None
        self._is_ready = False
        self._param_count = 0
        self._size_mb = 0.0
        
        self._load_model()
        self.tracker = PersonTracker(self.model, self.tracker_config)

    def _load_model(self) -> None:
        """Loads official RT-DETRv2 model weights with PyTorch 2.6+ compatibility."""
        logger.info(f"Loading RT-DETRv2 detector from '{self.checkpoint_path}' on device '{self.device}'...")

        # PyTorch 2.6+ defaults to weights_only=True which blocks custom Ultralytics DetectionModel unpickling
        orig_torch_load = torch.load
        try:
            torch.load = lambda *args, **kwargs: orig_torch_load(*args, **{**kwargs, "weights_only": False})
            from ultralytics import RTDETR
            
            # Find checkpoint either in current dir, models/, or download
            ckpt = Path(self.checkpoint_path)
            if not ckpt.exists():
                models_ckpt = Path("models") / ckpt.name
                if models_ckpt.exists():
                    self.checkpoint_path = str(models_ckpt)
                    ckpt = models_ckpt

            self.model = RTDETR(self.checkpoint_path)
            self._is_ready = True
            
            # Count parameters
            if hasattr(self.model, "model") and hasattr(self.model.model, "parameters"):
                self._param_count = sum(p.numel() for p in self.model.model.parameters())
            
            if ckpt.exists():
                self._size_mb = ckpt.stat().st_size / (1024 * 1024)
            else:
                self._size_mb = 63.4

            logger.info(
                f"Successfully initialized RT-DETRv2 ({self.model_name}): "
                f"{self._param_count:,} parameters ({self._param_count/1e6:.2f}M), Size: {self._size_mb:.1f} MB."
            )
        except Exception as e:
            logger.error(f"Failed to load RT-DETRv2 from '{self.checkpoint_path}': {e}")
            raise RuntimeError(f"RT-DETRv2 initialization failed: {e}") from e
        finally:
            torch.load = orig_torch_load

    @property
    def backend_name(self) -> str:
        return "rtdetrv2"

    @property
    def model_name(self) -> str:
        return Path(self.checkpoint_path).stem

    @property
    def parameter_count(self) -> int:
        return self._param_count

    @property
    def model_size_mb(self) -> float:
        return round(self._size_mb, 2)

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp_ms: float = 0.0,
        confidence_thresh: Optional[float] = None,
        iou_thresh: Optional[float] = None,
    ) -> FrameDetections:
        """Executes RT-DETRv2 inference on a video frame and filters for COCO person class."""
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None.")

        h, w = frame.shape[:2]
        conf = confidence_thresh if confidence_thresh is not None else self.config.confidence_threshold
        iou = iou_thresh if iou_thresh is not None else self.config.iou_threshold
        imgsz = getattr(self.config, "imgsz", 640)

        t_start = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            classes=[self.config.person_class_id],  # Class 0: person
            imgsz=imgsz,
            device=str(self.device),
            verbose=False,
        )
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        boxes: List[DetectionBox] = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy_np = result.boxes.xyxy.cpu().numpy()
                conf_np = result.boxes.conf.cpu().numpy()
                cls_np = result.boxes.cls.cpu().numpy()

                for i in range(len(xyxy_np)):
                    cid = int(cls_np[i])
                    if cid != self.config.person_class_id:
                        continue

                    # Strict boundary clipping to valid frame dimensions
                    x1 = float(max(0.0, min(float(w - 1), xyxy_np[i][0])))
                    y1 = float(max(0.0, min(float(h - 1), xyxy_np[i][1])))
                    x2 = float(max(x1 + 1.0, min(float(w), xyxy_np[i][2])))
                    y2 = float(max(y1 + 1.0, min(float(h), xyxy_np[i][3])))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    boxes.append(DetectionBox(
                        bbox_xyxy=(x1, y1, x2, y2),
                        confidence=float(conf_np[i]),
                        class_id=0,
                        class_name="person",
                        track_id=None,
                    ))

        return FrameDetections(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            detections=boxes,
            raw_results=results,
        )

    def track(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp_ms: float = 0.0,
        confidence_thresh: Optional[float] = None,
        iou_thresh: Optional[float] = None,
    ) -> FrameDetections:
        """Executes RT-DETRv2 inference coupled with ByteTrack multi-object tracking."""
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None.")

        h, w = frame.shape[:2]
        conf = confidence_thresh if confidence_thresh is not None else getattr(self.tracker_config, "confidence_threshold", 0.35)
        iou = iou_thresh if iou_thresh is not None else 0.45
        imgsz = getattr(self.tracker_config, "imgsz", 640)

        results = self.model.track(
            source=frame,
            persist=self.tracker_config.persist,
            tracker=self.tracker_config.tracker_config_path,
            conf=conf,
            iou=iou,
            classes=[self.config.person_class_id],
            imgsz=imgsz,
            device=str(self.device),
            verbose=False,
        )

        boxes: List[DetectionBox] = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy_np = result.boxes.xyxy.cpu().numpy()
                conf_np = result.boxes.conf.cpu().numpy()
                cls_np = result.boxes.cls.cpu().numpy()
                
                track_ids = None
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)

                for i in range(len(xyxy_np)):
                    if int(cls_np[i]) != self.config.person_class_id:
                        continue

                    x1 = float(max(0.0, min(float(w - 1), xyxy_np[i][0])))
                    y1 = float(max(0.0, min(float(h - 1), xyxy_np[i][1])))
                    x2 = float(max(x1 + 1.0, min(float(w), xyxy_np[i][2])))
                    y2 = float(max(y1 + 1.0, min(float(h), xyxy_np[i][3])))

                    tid = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else None

                    box = DetectionBox(
                        bbox_xyxy=(x1, y1, x2, y2),
                        confidence=float(conf_np[i]),
                        class_id=0,
                        class_name="person",
                        track_id=tid,
                    )
                    boxes.append(box)
                    
                    if tid is not None:
                        if tid not in self.tracker.active_tracks:
                            from safekid_ai.tracker import TrackedPerson
                            self.tracker.active_tracks[tid] = TrackedPerson(track_id=tid, current_box=box, first_seen_frame=frame_id)
                        self.tracker.active_tracks[tid].update(box, frame_id)

        return FrameDetections(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            detections=boxes,
            raw_results=results,
        )

