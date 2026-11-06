"""Track-level child classification and temporal confidence aggregation for ByteTrack."""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

from safekid_ai.child_detection.config import ChildDetectorConfig
from safekid_ai.child_detection.models import (
    AbstentionReason,
    ChildClassificationResult,
    ChildDetectionLabel,
    VerificationStatus,
)
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.track_child_classifier")


@dataclass
class TrackClassificationState:
    """Temporal classification state maintained for an active ByteTrack ID."""
    track_id: int
    current_label: ChildDetectionLabel = ChildDetectionLabel.UNKNOWN
    confidence: float = 0.0
    smoothed_probabilities: Dict[str, float] = field(
        default_factory=lambda: {"CHILD": 0.0, "ADULT": 0.0, "UNKNOWN": 1.0}
    )
    abstention_reason: AbstentionReason = AbstentionReason.UNINITIALIZED
    verification_status: VerificationStatus = VerificationStatus.PROVISIONAL_MODEL_PREDICTION
    
    consecutive_child_frames: int = 0
    consecutive_adult_frames: int = 0
    consecutive_unknown_frames: int = 0
    total_frames_observed: int = 0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    last_seen_timestamp_ms: float = 0.0
    
    # History of (frame_id, timestamp_ms, ChildClassificationResult)
    history: Deque[Tuple[int, float, ChildClassificationResult]] = field(default_factory=deque)

    def to_result(self) -> ChildClassificationResult:
        """Converts track state to a ChildClassificationResult."""
        return ChildClassificationResult(
            label=self.current_label,
            confidence=self.confidence,
            probabilities=dict(self.smoothed_probabilities),
            abstention_reason=self.abstention_reason,
            verification_status=self.verification_status,
            features={
                "consecutive_child_frames": float(self.consecutive_child_frames),
                "consecutive_adult_frames": float(self.consecutive_adult_frames),
                "consecutive_unknown_frames": float(self.consecutive_unknown_frames),
                "total_frames_observed": float(self.total_frames_observed),
            },
            track_id=self.track_id,
            frame_id=self.last_seen_frame,
        )


class TrackChildClassifier:
    """Aggregates single-frame ChildDetector predictions across ByteTrack trajectories.
    
    Key Safety & Reliability Invariants:
    1. Flickering Prevention: Requires multiple consecutive confirmed frames before promoting to CHILD or ADULT.
    2. Occlusion Resilience: Maintains state during short occlusions (grace period) without false dropouts.
    3. Graceful Decay: Long dropouts or ambiguous evidence decay safely back to UNKNOWN.
    4. Identity Isolation: New or reassigned ByteTrack IDs start afresh as UNKNOWN; no silent label inheritance.
    """

    def __init__(self, config: Optional[ChildDetectorConfig] = None):
        self.config = config or ChildDetectorConfig()
        self.tracks: Dict[int, TrackClassificationState] = {}
        self.archived_track_ids: Set[int] = set()

    def reset(self) -> None:
        """Resets all track states (e.g. on new camera stream or video restart)."""
        self.tracks.clear()
        self.archived_track_ids.clear()
        logger.info("TrackChildClassifier state reset.")

    def update_track(
        self,
        track_id: int,
        frame_result: ChildClassificationResult,
        frame_id: int,
        timestamp_ms: float,
    ) -> ChildClassificationResult:
        """Updates classification state for a track ID with a new frame prediction.
        
        Args:
            track_id: ByteTrack identity integer.
            frame_result: Instantaneous single-frame prediction from ChildDetector.
            frame_id: Video frame index.
            timestamp_ms: Milliseconds elapsed in stream.
            
        Returns:
            Aggregated, smoothed ChildClassificationResult for the track.
        """
        # Initialize new track if not seen before
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackClassificationState(
                track_id=track_id,
                first_seen_frame=frame_id,
                last_seen_frame=frame_id,
                last_seen_timestamp_ms=timestamp_ms,
            )

        state = self.tracks[track_id]
        state.last_seen_frame = frame_id
        state.last_seen_timestamp_ms = timestamp_ms
        state.total_frames_observed += 1

        # Maintain sliding history
        state.history.append((frame_id, timestamp_ms, frame_result))
        while len(state.history) > self.config.temporal_window_size:
            state.history.popleft()

        alpha = self.config.smoothing_alpha
        p_child_inst = frame_result.probabilities.get("CHILD", 0.0)
        p_adult_inst = frame_result.probabilities.get("ADULT", 0.0)

        # 1. Update frame counters and exponential moving average of probabilities
        if frame_result.label == ChildDetectionLabel.CHILD:
            state.consecutive_child_frames += 1
            state.consecutive_adult_frames = 0
            state.consecutive_unknown_frames = 0
            
            # Smooth probabilities upward
            state.smoothed_probabilities["CHILD"] = alpha * p_child_inst + (1.0 - alpha) * state.smoothed_probabilities.get("CHILD", 0.0)
            state.smoothed_probabilities["ADULT"] = (1.0 - alpha) * state.smoothed_probabilities.get("ADULT", 0.0)
            state.smoothed_probabilities["UNKNOWN"] = max(0.0, 1.0 - state.smoothed_probabilities["CHILD"] - state.smoothed_probabilities["ADULT"])

        elif frame_result.label == ChildDetectionLabel.ADULT:
            state.consecutive_adult_frames += 1
            state.consecutive_child_frames = 0
            state.consecutive_unknown_frames = 0
            
            # Smooth probabilities upward
            state.smoothed_probabilities["ADULT"] = alpha * p_adult_inst + (1.0 - alpha) * state.smoothed_probabilities.get("ADULT", 0.0)
            state.smoothed_probabilities["CHILD"] = (1.0 - alpha) * state.smoothed_probabilities.get("CHILD", 0.0)
            state.smoothed_probabilities["UNKNOWN"] = max(0.0, 1.0 - state.smoothed_probabilities["CHILD"] - state.smoothed_probabilities["ADULT"])

        else:
            # Prediction is UNKNOWN (e.g. temporary occlusion, low resolution, distant figure)
            state.consecutive_unknown_frames += 1
            state.consecutive_child_frames = 0
            state.consecutive_adult_frames = 0
            
            # Gradual decay of confident probabilities
            state.smoothed_probabilities["CHILD"] *= 0.90
            state.smoothed_probabilities["ADULT"] *= 0.90
            state.smoothed_probabilities["UNKNOWN"] = max(0.0, 1.0 - state.smoothed_probabilities["CHILD"] - state.smoothed_probabilities["ADULT"])

        p_ch_smooth = state.smoothed_probabilities["CHILD"]
        p_ad_smooth = state.smoothed_probabilities["ADULT"]

        # 2. Hysteresis Decision Logic:
        # A. Transition to CHILD requires sustained confirmation
        if (
            state.consecutive_child_frames >= self.config.temporal_confirmation_frames
            or (p_ch_smooth >= self.config.confidence_threshold and p_ch_smooth - p_ad_smooth >= self.config.margin_threshold)
        ):
            state.current_label = ChildDetectionLabel.CHILD
            state.confidence = max(p_ch_smooth, frame_result.confidence)
            state.abstention_reason = AbstentionReason.NONE
            state.verification_status = VerificationStatus.PROVISIONAL_MODEL_PREDICTION

        # B. Transition to ADULT requires sustained confirmation
        elif (
            state.consecutive_adult_frames >= self.config.temporal_confirmation_frames
            or (p_ad_smooth >= self.config.confidence_threshold and p_ad_smooth - p_ch_smooth >= self.config.margin_threshold)
        ):
            state.current_label = ChildDetectionLabel.ADULT
            state.confidence = max(p_ad_smooth, frame_result.confidence)
            state.abstention_reason = AbstentionReason.NONE
            state.verification_status = VerificationStatus.PROVISIONAL_MODEL_PREDICTION

        # C. Temporary Occlusion Handling (Grace Window)
        elif state.consecutive_unknown_frames <= self.config.temporal_decay_frames:
            # If the track was already classified, preserve it temporarily during brief occlusions
            if state.current_label in (ChildDetectionLabel.CHILD, ChildDetectionLabel.ADULT):
                # Retain label but attenuate confidence
                state.confidence *= 0.95
                state.abstention_reason = frame_result.abstention_reason
            else:
                state.current_label = ChildDetectionLabel.UNKNOWN
                state.confidence = frame_result.confidence
                state.abstention_reason = frame_result.abstention_reason

        # D. Grace Window Exceeded: Decay to UNKNOWN
        else:
            state.current_label = ChildDetectionLabel.UNKNOWN
            state.confidence = 0.0
            state.abstention_reason = AbstentionReason.TEMPORAL_DECAY

        return state.to_result()

    def get_track_classification(self, track_id: Optional[int]) -> ChildClassificationResult:
        """Returns the current temporal classification for a track ID."""
        if track_id is None or track_id not in self.tracks:
            return ChildClassificationResult(
                label=ChildDetectionLabel.UNKNOWN,
                confidence=0.0,
                probabilities={"CHILD": 0.0, "ADULT": 0.0, "UNKNOWN": 1.0},
                abstention_reason=AbstentionReason.UNINITIALIZED,
                verification_status=VerificationStatus.UNVERIFIED,
                track_id=track_id,
            )
        return self.tracks[track_id].to_result()

    def purge_lost_tracks(self, active_track_ids: Set[int]) -> List[int]:
        """Cleans up tracks that have ended or lost observation in ByteTrack.
        
        Returns:
            List of purged track IDs.
        """
        purged = []
        for tid in list(self.tracks.keys()):
            if tid not in active_track_ids:
                purged.append(tid)
                self.archived_track_ids.add(tid)
                del self.tracks[tid]
        return purged

