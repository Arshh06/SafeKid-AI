"""Pretrained YOLO person detection module for SafeKid AI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from ultralytics import YOLO

from safekid_ai.config import DetectorConfig
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.detector")


@dataclass
class DetectionBox:
    """Represents a single detected person object."""
    bbox_xyxy: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0
    class_name: str = "person"
    track_id: Optional[int] = None

    @property
    def x1(self) -> float:
        return self.bbox_xyxy[0]

    @property
    def y1(self) -> float:
        return self.bbox_xyxy[1]

    @property
    def x2(self) -> float:
        return self.bbox_xyxy[2]

    @property
    def y2(self) -> float:
        return self.bbox_xyxy[3]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Bottom-centre ground-contact point (x_mid, y_max)."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def extract_crop(self, frame: np.ndarray) -> np.ndarray:
        """Extracts the image crop corresponding to this detection box."""
        h, w = frame.shape[:2]
        ix1 = max(0, min(w - 1, int(round(self.x1))))
        iy1 = max(0, min(h - 1, int(round(self.y1))))
        ix2 = max(ix1 + 1, min(w, int(round(self.x2))))
        iy2 = max(iy1 + 1, min(h, int(round(self.y2))))
        return frame[iy1:iy2, ix1:ix2].copy()


@dataclass
class FrameDetections:
    """Detections for a single image/video frame."""
    frame_id: int
    timestamp_ms: float
    detections: List[DetectionBox] = field(default_factory=list)
    raw_results: Optional[object] = None

    def __len__(self) -> int:
        return len(self.detections)


class PersonDetector:
    """Pretrained YOLO person detector for community perimeter and common areas."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self._load_model()

    def _load_model(self) -> None:
        """Initializes and loads the YOLO model from checkpoint."""
        model_path = self.config.model_name_or_path
        logger.info(f"Loading pretrained YOLO person detector from '{model_path}' on device '{self.config.device}'...")
        
        # Configure optimal CPU thread allocation for hybrid P/E-core CPUs (Intel i3)
        if self.config.device == "cpu" and getattr(self.config, "num_threads", None):
            try:
                torch.set_num_threads(self.config.num_threads)
                logger.info(f"Configured PyTorch CPU threads to {self.config.num_threads} for optimal P-core execution.")
            except Exception as e:
                logger.warning(f"Could not set PyTorch CPU threads: {e}")

        # Scope safe loading strictly to Ultralytics checkpoint loading without globally weakening torch.load
        import ultralytics.nn.tasks as tasks
        orig_safe_load = tasks.torch_safe_load

        def _scoped_safe_load(weight):
            return torch.load(weight, map_location="cpu", weights_only=False), weight

        tasks.torch_safe_load = _scoped_safe_load
        try:
            self.model = YOLO(model_path)
            # Ensure safe evaluation mode on device
            self.model.to(self.config.device)

            # Fuse Conv2d and BatchNorm layers for CPU acceleration
            if getattr(self.config, "fuse_model", True):
                try:
                    self.model.fuse()
                    logger.info("Successfully fused YOLO Conv2d and BatchNorm layers for CPU inference.")
                except Exception as e:
                    logger.warning(f"Model layer fusion skipped: {e}")

            # Warm-up forward pass to eliminate frame-0 cold-start latency spike
            try:
                sz = getattr(self.config, "imgsz", 640)
                dummy = np.zeros((sz, sz, 3), dtype=np.uint8)
                self.model.predict(
                    source=dummy,
                    imgsz=sz,
                    classes=[self.config.person_class_id],
                    device=self.config.device,
                    verbose=False,
                )
                logger.info("PersonDetector warmed up successfully.")
            except Exception as e:
                logger.warning(f"PersonDetector warm-up skipped: {e}")

            names = getattr(self.model, "names", {})
            logger.info(f"YOLO detector successfully initialized. Classes available: {len(names)}")
        except Exception as e:
            logger.error(f"Failed to load YOLO checkpoint '{model_path}': {e}")
            raise RuntimeError(f"Detector initialization failed: {e}") from e
        finally:
            tasks.torch_safe_load = orig_safe_load

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
            frame: OpenCV BGR image array (uint8).
            frame_id: Frame index.
            timestamp_ms: Video timestamp in milliseconds.
            confidence_thresh: Optional confidence threshold override.
            iou_thresh: Optional IoU threshold override.

        Returns:
            FrameDetections containing filtered person bounding boxes.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None.")

        # Class 0 corresponds to 'person' in standard COCO pretrained checkpoints
        person_filter = [self.config.person_class_id]
        imgsz = getattr(self.config, "imgsz", 640)
        h, w = frame.shape[:2]
        conf = confidence_thresh if confidence_thresh is not None else self.config.confidence_threshold
        iou = iou_thresh if iou_thresh is not None else self.config.iou_threshold

        if getattr(self.config, "enable_tiling", False) and (w > imgsz or h > imgsz):
            detections = self._detect_tiled(frame, person_filter, imgsz)
            raw_res = None
        else:
            results = self.model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                classes=person_filter,
                device=self.config.device,
                imgsz=imgsz,
                verbose=self.config.verbose,
            )

            detections: List[DetectionBox] = []
            raw_res = results[0] if len(results) > 0 else None

            if raw_res is not None and raw_res.boxes is not None:
                boxes = raw_res.boxes.cpu()
                xyxy_arr = boxes.xyxy.numpy()
                conf_arr = boxes.conf.numpy()
                cls_arr = boxes.cls.numpy()

                min_area = getattr(self.config, "min_box_area", 0.0)
                max_area = getattr(self.config, "max_box_area", None)

                for i in range(len(xyxy_arr)):
                    cid = int(cls_arr[i])
                    cname = raw_res.names.get(cid, "person")
                    bbox = (
                        float(xyxy_arr[i][0]),
                        float(xyxy_arr[i][1]),
                        float(xyxy_arr[i][2]),
                        float(xyxy_arr[i][3]),
                    )
                    area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
                    if min_area > 0 and area < min_area:
                        continue
                    if max_area is not None and area > max_area:
                        continue

                    detections.append(
                        DetectionBox(
                            bbox_xyxy=bbox,
                            confidence=float(conf_arr[i]),
                            class_id=cid,
                            class_name=cname,
                        )
                    )

        return FrameDetections(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            detections=detections,
            raw_results=raw_res,
        )

    def _detect_tiled(
        self,
        frame: np.ndarray,
        person_filter: List[int],
        imgsz: int,
    ) -> List[DetectionBox]:
        """Runs tiled / sliced inference for high-resolution surveillance images."""
        import torchvision.ops

        h, w = frame.shape[:2]
        tile_w, tile_h = getattr(self.config, "tile_size", (imgsz, imgsz))
        overlap = getattr(self.config, "tile_overlap", 0.20)

        step_x = max(64, int(tile_w * (1.0 - overlap)))
        step_y = max(64, int(tile_h * (1.0 - overlap)))

        x_coords = list(range(0, max(1, w - tile_w + 1), step_x))
        if len(x_coords) == 0 or x_coords[-1] + tile_w < w:
            x_coords.append(max(0, w - tile_w))

        y_coords = list(range(0, max(1, h - tile_h + 1), step_y))
        if len(y_coords) == 0 or y_coords[-1] + tile_h < h:
            y_coords.append(max(0, h - tile_h))

        candidates: List[DetectionBox] = []

        # Run global full frame at standard imgsz first
        res_global = self.model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=person_filter,
            device=self.config.device,
            imgsz=imgsz,
            verbose=self.config.verbose,
        )
        if len(res_global) > 0 and res_global[0].boxes is not None:
            boxes = res_global[0].boxes.cpu()
            for i in range(len(boxes.xyxy)):
                candidates.append(
                    DetectionBox(
                        bbox_xyxy=tuple(float(c) for c in boxes.xyxy[i]),
                        confidence=float(boxes.conf[i]),
                        class_id=int(boxes.cls[i]),
                        class_name="person",
                    )
                )

        # Run on each tile
        for y1 in y_coords:
            for x1 in x_coords:
                x2 = min(w, x1 + tile_w)
                y2 = min(h, y1 + tile_h)
                tile_crop = frame[y1:y2, x1:x2]
                res_tile = self.model.predict(
                    source=tile_crop,
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    classes=person_filter,
                    device=self.config.device,
                    imgsz=imgsz,
                    verbose=self.config.verbose,
                )
                if len(res_tile) > 0 and res_tile[0].boxes is not None:
                    t_boxes = res_tile[0].boxes.cpu()
                    for j in range(len(t_boxes.xyxy)):
                        gx1 = float(t_boxes.xyxy[j][0]) + x1
                        gy1 = float(t_boxes.xyxy[j][1]) + y1
                        gx2 = float(t_boxes.xyxy[j][2]) + x1
                        gy2 = float(t_boxes.xyxy[j][3]) + y1
                        candidates.append(
                            DetectionBox(
                                bbox_xyxy=(gx1, gy1, gx2, gy2),
                                confidence=float(t_boxes.conf[j]),
                                class_id=int(t_boxes.cls[j]),
                                class_name="person",
                            )
                        )

        if not candidates:
            return []

        # Global NMS over combined candidates
        boxes_tensor = torch.tensor([c.bbox_xyxy for c in candidates], dtype=torch.float32)
        scores_tensor = torch.tensor([c.confidence for c in candidates], dtype=torch.float32)
        keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=self.config.iou_threshold)

        min_area = getattr(self.config, "min_box_area", 0.0)
        max_area = getattr(self.config, "max_box_area", None)

        filtered: List[DetectionBox] = []
        for idx in keep_indices.tolist():
            c = candidates[idx]
            area = c.area
            if min_area > 0 and area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
            filtered.append(c)

        return filtered

