"""Automated child classification provider integrating ChildDetector with ChildSafetyEventEngine."""

from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from safekid_ai.child_detection.config import ChildDetectorConfig
from safekid_ai.child_detection.detector import ChildDetector
from safekid_ai.child_detection.models import (
    ChildClassificationResult,
    ChildDetectionLabel,
    VerificationStatus,
)
from safekid_ai.child_detection.tracker_integration import TrackChildClassifier
from safekid_ai.child_safety import (
    ChildTrackAnnotation,
    TrackClassification,
    TrackClassificationProvider,
)
from safekid_ai.detector import DetectionBox
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.child_classification_provider")


class AutomatedChildClassificationProvider(TrackClassificationProvider):
    """Dynamic, automated track classification provider powered by ChildDetector and TrackChildClassifier.
    
    Seamlessly satisfies the TrackClassificationProvider interface used by ChildSafetyEventEngine:
    - Provides real-time automated child/adult/unclassified status for every tracked person.
    - Preserves strict ethical safeguards: maintains distinction between provisional model candidates
      and verified child profiles with parental consent.
    - Prevents unverified public surveillance tracks from generating automated child alerts.
    """

    def __init__(
        self,
        detector: Optional[ChildDetector] = None,
        tracker_aggregator: Optional[TrackChildClassifier] = None,
        config: Optional[ChildDetectorConfig] = None,
        community_id: str = "community_001",
        camera_id: str = "camera_01",
        allow_provisional_alerts: bool = True,
        enrolled_child_tracks: Optional[Set[int]] = None,
    ):
        self.config = config or ChildDetectorConfig()
        self.detector = detector or ChildDetector(self.config)
        self.tracker_aggregator = tracker_aggregator or TrackChildClassifier(self.config)
        self.community_id = community_id
        self.camera_id = camera_id
        self.allow_provisional_alerts = allow_provisional_alerts
        self.enrolled_child_tracks: Set[int] = enrolled_child_tracks or set()
        
        # Staff review records for residential security audits
        self.staff_confirmed_tracks: Set[int] = set()
        self.staff_dismissed_tracks: Set[int] = set()
        
        # Cache of latest results per track ID
        self._latest_results: Dict[int, ChildClassificationResult] = {}

    def reset(self) -> None:
        """Resets dynamic tracking state."""
        self.tracker_aggregator.reset()
        self.staff_confirmed_tracks.clear()
        self.staff_dismissed_tracks.clear()
        self._latest_results.clear()
        logger.info("AutomatedChildClassificationProvider state reset.")

    def process_frame(
        self,
        frame: np.ndarray,
        detections: List[DetectionBox],
        frame_id: int = 0,
        timestamp_ms: float = 0.0,
    ) -> Dict[int, ChildClassificationResult]:
        """Classifies all detections in the frame and updates track-level aggregations.
        
        Args:
            frame: OpenCV BGR image array.
            detections: Localized detection boxes from PersonTracker.
            frame_id: Current frame sequence number.
            timestamp_ms: Milliseconds elapsed.
            
        Returns:
            Dictionary mapping track_id -> aggregated ChildClassificationResult.
        """
        active_track_ids = {det.track_id for det in detections if det.track_id is not None}
        self.tracker_aggregator.purge_lost_tracks(active_track_ids)

        frame_results: Dict[int, ChildClassificationResult] = {}
        cls_interval = getattr(self.config, "classification_interval", 1) or 1
        max_staleness = getattr(self.config, "max_track_staleness", 5)

        for det in detections:
            tid = det.track_id
            if tid is None:
                continue

            # Skip stale tracks exceeding max staleness
            if getattr(det, "stale_frames", 0) > max_staleness:
                continue

            # If detection is predicted/extrapolated on a skipped frame, reuse latest result without model pass
            if getattr(det, "is_predicted", False):
                if tid in self._latest_results:
                    frame_results[tid] = self._latest_results[tid]
                continue

            # Classification interval throttling for established tracks
            if cls_interval > 1 and (frame_id % cls_interval) != 0:
                if tid in self._latest_results and self._latest_results[tid].label != ChildDetectionLabel.UNKNOWN:
                    frame_results[tid] = self._latest_results[tid]
                    continue

            # Stationary crop optimization for established tracks
            if getattr(self.config, "skip_stationary_crops", True):
                if tid in self.tracker_aggregator.tracks:
                    track_state = self.tracker_aggregator.tracks[tid]
                    if track_state.total_frames_observed >= 3 and tid in self._latest_results:
                        if len(track_state.history) >= 1:
                            prev_res = track_state.history[-1][2]
                            if prev_res.features and "bbox_center" in prev_res.features:
                                cur_c = det.center
                                prev_c = prev_res.features["bbox_center"]
                                if abs(cur_c[0] - prev_c[0]) < 2.0 and abs(cur_c[1] - prev_c[1]) < 2.0:
                                    frame_results[tid] = self._latest_results[tid]
                                    continue

            # Step 1: Single-frame supervised child classification
            crop = det.extract_crop(frame)
            fh, fw = frame.shape[:2]
            inst_result = self.detector.classify_crop(
                crop=crop,
                bbox_xyxy=det.bbox_xyxy,
                frame_size=(fw, fh),
                track_id=tid,
                frame_id=frame_id,
            )
            if inst_result.features is None:
                inst_result.features = {}
            inst_result.features["bbox_center"] = det.center

            # Step 2: Temporal aggregation and flickering suppression
            track_result = self.tracker_aggregator.update_track(
                track_id=tid,
                frame_result=inst_result,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
            )

            # Check for parental enrollment / staff verification override
            if tid in self.enrolled_child_tracks or tid in self.staff_confirmed_tracks:
                track_result.verification_status = VerificationStatus.VERIFIED_CHILD_PROFILE
                track_result.label = ChildDetectionLabel.CHILD
            elif tid in self.staff_dismissed_tracks:
                track_result.verification_status = VerificationStatus.UNVERIFIED
                track_result.label = ChildDetectionLabel.UNKNOWN

            self._latest_results[tid] = track_result
            frame_results[tid] = track_result

        return frame_results

    def get_classification(self, track_id: Optional[int], timestamp_ms: float = 0.0) -> TrackClassification:
        """Returns TrackClassification enum (CHILD, ADULT, UNCLASSIFIED) for ChildSafetyEventEngine.
        
        Invariants:
        - If track is dismissed by staff: returns UNCLASSIFIED.
        - If track is predicted as CHILD:
          - If provisional alerts are enabled: returns TrackClassification.CHILD.
          - If parental consent is required and not present: returns UNCLASSIFIED.
        - If track is predicted as ADULT: returns TrackClassification.ADULT.
        - If UNKNOWN: returns TrackClassification.UNCLASSIFIED.
        """
        if track_id is None:
            return TrackClassification.UNCLASSIFIED

        if track_id in self.staff_dismissed_tracks:
            return TrackClassification.UNCLASSIFIED

        if track_id in self.enrolled_child_tracks or track_id in self.staff_confirmed_tracks:
            return TrackClassification.CHILD

        result = self._latest_results.get(track_id)
        if result is None:
            # Check aggregator directly
            result = self.tracker_aggregator.get_track_classification(track_id)

        if result.label == ChildDetectionLabel.CHILD:
            if not self.allow_provisional_alerts and track_id not in self.enrolled_child_tracks:
                return TrackClassification.UNCLASSIFIED
            return TrackClassification.CHILD

        elif result.label == ChildDetectionLabel.ADULT:
            return TrackClassification.ADULT

        return TrackClassification.UNCLASSIFIED

    def get_detailed_result(self, track_id: Optional[int]) -> Optional[ChildClassificationResult]:
        """Retrieves full classification details including probabilities and abstention reason."""
        if track_id is None:
            return None
        return self._latest_results.get(track_id) or self.tracker_aggregator.get_track_classification(track_id)

    def confirm_child_candidate(self, track_id: int) -> None:
        """Staff action: confirms a provisional candidate as a verified child."""
        self.staff_confirmed_tracks.add(track_id)
        self.staff_dismissed_tracks.discard(track_id)
        logger.info(f"Staff confirmed track ID {track_id} as VERIFIED_CHILD_PROFILE.")

    def dismiss_child_candidate(self, track_id: int) -> None:
        """Staff action: dismisses a provisional candidate (false alarm suppression)."""
        self.staff_dismissed_tracks.add(track_id)
        self.staff_confirmed_tracks.discard(track_id)
        logger.info(f"Staff dismissed track ID {track_id} from child alert consideration.")

    def get_annotation(self, track_id: Optional[int]) -> Optional[ChildTrackAnnotation]:
        """Provides backward-compatibility with annotation provider queries."""
        if track_id is None:
            return None
        cls_val = self.get_classification(track_id)
        res = self.get_detailed_result(track_id)
        conf = res.confidence if res else 1.0
        return ChildTrackAnnotation(
            track_id=track_id,
            classification=cls_val,
            notes=f"Automated ChildDetector prediction: {res.verification_status.value if res else 'N/A'}",
            confidence=conf,
        )

    def list_annotations(self) -> Dict[int, ChildTrackAnnotation]:
        """Lists active annotations for all currently tracked entities."""
        res_dict = {}
        for tid in list(self._latest_results.keys()):
            ann = self.get_annotation(tid)
            if ann:
                res_dict[tid] = ann
        return res_dict

