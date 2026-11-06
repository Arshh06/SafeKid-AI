"""Dedicated ChildDetector subsystem for SafeKid AI."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from safekid_ai.child_detection.classifier_net import (
    ChildAdultClassifierNet,
    preprocess_crop,
)
from safekid_ai.child_detection.config import ChildDetectorConfig
from safekid_ai.child_detection.models import (
    AbstentionReason,
    ChildClassificationResult,
    ChildDetectionLabel,
    VerificationStatus,
)
from safekid_ai.detector import DetectionBox
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.child_detector")


class ChildDetector:
    """Supervised child detection model operating on localized pedestrian crops.
    
    Adheres strictly to SafeKid AI privacy and safety governance:
    - Never performs facial recognition or biometric enrollment.
    - Never performs continuous age regression.
    - Strictly abstains to UNKNOWN when resolution, confidence, or body proportion cues are ambiguous.
    """

    def __init__(self, config: Optional[ChildDetectorConfig] = None):
        self.config = config or ChildDetectorConfig()
        self.device = torch.device(self.config.device)
        self.model = ChildAdultClassifierNet().to(self.device)
        self.is_weights_loaded = False
        
        self._load_weights()
        self.model.eval()

    def _load_weights(self) -> None:
        """Loads trained model weights if present on disk."""
        if not self.config.model_weights_path:
            logger.info("ChildDetector initialized with default baseline weights (no path specified).")
            return
            
        weights_path = Path(self.config.model_weights_path).resolve()
        if weights_path.exists():
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    self.model.load_state_dict(state_dict["model_state_dict"])
                else:
                    self.model.load_state_dict(state_dict)
                self.is_weights_loaded = True
                logger.info(f"Successfully loaded ChildDetector weights from: {weights_path}")
            except Exception as e:
                logger.warning(f"Failed to load weights from '{weights_path}': {e}. Using initialized model.")
        else:
            logger.info(f"ChildDetector weights not found at '{weights_path}'. Operating in architectural baseline mode.")

    def set_weights(self, weights_path: Union[str, Path]) -> bool:
        """Dynamically reloads weights from a new path."""
        self.config.model_weights_path = str(weights_path)
        self._load_weights()
        return self.is_weights_loaded

    def classify_crop(
        self,
        crop: Optional[np.ndarray],
        bbox_xyxy: Optional[Tuple[float, float, float, float]] = None,
        frame_size: Optional[Tuple[int, int]] = None,
        track_id: Optional[int] = None,
        frame_id: Optional[int] = None,
    ) -> ChildClassificationResult:
        """Classifies a pedestrian crop into CHILD, ADULT, or UNKNOWN with strict abstention gating.
        
        Args:
            crop: OpenCV BGR image crop of the pedestrian.
            bbox_xyxy: Bounding box coordinates (x1, y1, x2, y2).
            frame_size: Full frame dimensions (width, height).
            track_id: Tracking ID if available.
            frame_id: Video frame index if available.
            
        Returns:
            ChildClassificationResult containing label, confidence, probabilities, and abstention reason.
        """
        t_start = time.perf_counter()

        # Gate 1: Check for empty or invalid crop
        if crop is None or crop.size == 0:
            return ChildClassificationResult(
                label=ChildDetectionLabel.UNKNOWN,
                confidence=0.0,
                probabilities={"CHILD": 0.0, "ADULT": 0.0, "UNKNOWN": 1.0},
                abstention_reason=AbstentionReason.DEGRADED_CROP,
                verification_status=VerificationStatus.UNVERIFIED,
                track_id=track_id,
                frame_id=frame_id,
                inference_time_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        h, w = crop.shape[:2]

        # Gate 2: Resolution & Distance Check
        # Distant or small figures cannot be reliably classified without high false alarm rates
        if h < self.config.min_crop_height or w < self.config.min_crop_width:
            return ChildClassificationResult(
                label=ChildDetectionLabel.UNKNOWN,
                confidence=0.0,
                probabilities={"CHILD": 0.0, "ADULT": 0.0, "UNKNOWN": 1.0},
                abstention_reason=AbstentionReason.LOW_RESOLUTION,
                verification_status=VerificationStatus.UNVERIFIED,
                features={"crop_height": float(h), "crop_width": float(w)},
                track_id=track_id,
                frame_id=frame_id,
                inference_time_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # Gate 3: Aspect Ratio & Severe Occlusion Check
        aspect_ratio = float(h) / max(1.0, float(w))
        if aspect_ratio < self.config.min_aspect_ratio or aspect_ratio > self.config.max_aspect_ratio:
            return ChildClassificationResult(
                label=ChildDetectionLabel.UNKNOWN,
                confidence=0.0,
                probabilities={"CHILD": 0.0, "ADULT": 0.0, "UNKNOWN": 1.0},
                abstention_reason=AbstentionReason.SEVERE_OCCLUSION,
                verification_status=VerificationStatus.UNVERIFIED,
                features={"aspect_ratio": aspect_ratio, "crop_height": float(h), "crop_width": float(w)},
                track_id=track_id,
                frame_id=frame_id,
                inference_time_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # Gate 3b: Lower Quadrant Solid Obstacle Check (e.g. wall, vehicle bumper)
        lower_quarter = crop[int(h * 0.65):, :]
        if lower_quarter.size > 0 and h > 50:
            std_dev = float(np.std(lower_quarter))
            if std_dev < 10.0:  # Flat solid barrier covering lower body
                return ChildClassificationResult(
                    label=ChildDetectionLabel.UNKNOWN,
                    confidence=0.0,
                    probabilities={"CHILD": 0.0, "ADULT": 0.0, "UNKNOWN": 1.0},
                    abstention_reason=AbstentionReason.SEVERE_OCCLUSION,
                    verification_status=VerificationStatus.UNVERIFIED,
                    features={"aspect_ratio": aspect_ratio, "lower_std": std_dev, "crop_height": float(h)},
                    track_id=track_id,
                    frame_id=frame_id,
                    inference_time_ms=(time.perf_counter() - t_start) * 1000.0,
                )

        # Preprocess and forward pass
        try:
            tensor = preprocess_crop(
                crop=crop,
                target_size=self.config.input_size,
                device=self.config.device,
            )
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0)
                p_child = float(probs[0].item())
                p_adult = float(probs[1].item())
        except Exception as e:
            logger.error(f"Inference error in ChildDetector: {e}")
            return ChildClassificationResult(
                label=ChildDetectionLabel.UNKNOWN,
                confidence=0.0,
                probabilities={"CHILD": 0.0, "ADULT": 0.0, "UNKNOWN": 1.0},
                abstention_reason=AbstentionReason.DEGRADED_CROP,
                verification_status=VerificationStatus.UNVERIFIED,
                track_id=track_id,
                frame_id=frame_id,
                inference_time_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        p_max = max(p_child, p_adult)
        margin = abs(p_child - p_adult)

        # Gate 4: Confidence Threshold Check
        if p_max < self.config.confidence_threshold:
            return ChildClassificationResult(
                label=ChildDetectionLabel.UNKNOWN,
                confidence=p_max,
                probabilities={"CHILD": p_child, "ADULT": p_adult, "UNKNOWN": 1.0 - p_max},
                abstention_reason=AbstentionReason.LOW_CONFIDENCE,
                verification_status=VerificationStatus.PROVISIONAL_MODEL_PREDICTION,
                features={"aspect_ratio": aspect_ratio, "margin": margin, "p_max": p_max},
                track_id=track_id,
                frame_id=frame_id,
                inference_time_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # Gate 5: Decision Margin Separation Check
        if margin < self.config.margin_threshold:
            return ChildClassificationResult(
                label=ChildDetectionLabel.UNKNOWN,
                confidence=margin,
                probabilities={"CHILD": p_child, "ADULT": p_adult, "UNKNOWN": 1.0 - margin},
                abstention_reason=AbstentionReason.AMBIGUOUS_MARGIN,
                verification_status=VerificationStatus.PROVISIONAL_MODEL_PREDICTION,
                features={"aspect_ratio": aspect_ratio, "margin": margin, "p_max": p_max},
                track_id=track_id,
                frame_id=frame_id,
                inference_time_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # Gate 6: Geometric & Perspective Scale Sanity Cross-Check
        # Cross-verifies that visual classification is physically plausible given camera perspective
        if bbox_xyxy is not None and frame_size is not None:
            fw, fh = frame_size
            if fh > 0 and fw > 0:
                x1, y1, x2, y2 = bbox_xyxy
                rel_h = float(y2 - y1) / float(fh)
                bottom_y = float(y2) / float(fh)

                # In CCTV angles (2.5m - 4m mounting), an entity occupying > 45% of frame height
                # in the immediate foreground (bottom_y > 0.85) has physical stature > 1.7m.
                if p_child > p_adult and rel_h > 0.45 and bottom_y > 0.85:
                    return ChildClassificationResult(
                        label=ChildDetectionLabel.UNKNOWN,
                        confidence=p_child,
                        probabilities={"CHILD": p_child, "ADULT": p_adult, "UNKNOWN": 1.0 - p_child},
                        abstention_reason=AbstentionReason.SCALE_MISMATCH,
                        verification_status=VerificationStatus.PROVISIONAL_MODEL_PREDICTION,
                        features={"aspect_ratio": aspect_ratio, "rel_height": rel_h, "bottom_y": bottom_y},
                        track_id=track_id,
                        frame_id=frame_id,
                        inference_time_ms=(time.perf_counter() - t_start) * 1000.0,
                    )

        # Classification passed all gates
        if p_child > p_adult:
            pred_label = ChildDetectionLabel.CHILD
            conf = p_child
        else:
            pred_label = ChildDetectionLabel.ADULT
            conf = p_adult

        return ChildClassificationResult(
            label=pred_label,
            confidence=conf,
            probabilities={"CHILD": p_child, "ADULT": p_adult, "UNKNOWN": 0.0},
            abstention_reason=AbstentionReason.NONE,
            verification_status=VerificationStatus.PROVISIONAL_MODEL_PREDICTION,
            features={"aspect_ratio": aspect_ratio, "margin": margin, "crop_height": float(h)},
            track_id=track_id,
            frame_id=frame_id,
            inference_time_ms=(time.perf_counter() - t_start) * 1000.0,
        )

    def classify_detections(
        self,
        frame: np.ndarray,
        detections: List[DetectionBox],
        frame_id: Optional[int] = None,
    ) -> List[ChildClassificationResult]:
        """Classifies each detection in a frame."""
        results: List[ChildClassificationResult] = []
        fh, fw = frame.shape[:2]
        for det in detections:
            crop = det.extract_crop(frame)
            res = self.classify_crop(
                crop=crop,
                bbox_xyxy=det.bbox_xyxy,
                frame_size=(fw, fh),
                track_id=det.track_id,
                frame_id=frame_id,
            )
            results.append(res)
        return results

