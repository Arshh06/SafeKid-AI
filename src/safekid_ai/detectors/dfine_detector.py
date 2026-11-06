"""D-FINE (Fine-grained Distribution Refinement DETR) detector adapter for SafeKid AI."""

from collections import deque
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision
from torchvision.ops import nms

# Ensure dfine_repo and src are on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DFINE_DIR = ROOT_DIR / "dfine_repo"
if str(DFINE_DIR) not in sys.path:
    sys.path.insert(0, str(DFINE_DIR))

from safekid_ai.config import DetectorConfig, TrackerConfig
from safekid_ai.detector import DetectionBox, FrameDetections
from safekid_ai.detectors.base import BasePersonDetector
from safekid_ai.tracker import TrackedPerson
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.detectors.dfine")


class _DetectionContainer:
    """Internal container converting NumPy arrays for Ultralytics BYTETracker."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.xyxy)


class DFINEDetectorAdapter(BasePersonDetector):
    """Adapter wrapping official pretrained D-FINE for real-time person detection."""

    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        tracker_config: Optional[TrackerConfig] = None,
        variant: str = "dfine_n",
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """Initializes D-FINE detector adapter.

        Args:
            config: DetectorConfig instance.
            tracker_config: TrackerConfig instance.
            variant: "dfine_n" (Nano, 3.81M params) or "dfine_s" (Small, 10.39M params).
            checkpoint_path: Optional override for pretrained weights (.pth).
            config_path: Optional override for model config (.yml).
        """
        self.config = config or DetectorConfig()
        self.tracker_config = tracker_config or TrackerConfig()
        
        # Resolve model variant
        v = (variant or getattr(self.config, "model_variant", "dfine_n")).lower()
        if v.endswith("_s") or v.endswith("-s") or v == "s" or "dfine_s" in v:
            self.variant = "dfine_s"
        else:
            self.variant = "dfine_n"

        self.device = torch.device(self.config.device if torch.cuda.is_available() or self.config.device == "cpu" else "cpu")
        self.active_tracks: Dict[int, TrackedPerson] = {}

        # Configure paths
        ckpt = checkpoint_path or getattr(self.config, "model_name_or_path", None)
        if not ckpt or not str(ckpt).endswith(".pth"):
            ckpt = str(ROOT_DIR / "models" / f"{self.variant}_coco.pth")
        self.checkpoint_path = str(ckpt)

        cfg_p = config_path or getattr(self.config, "config_path", None)
        if not cfg_p or not str(cfg_p).endswith(".yml"):
            cfg_p = str(DFINE_DIR / "configs" / "dfine" / f"dfine_hgnetv2_{self.variant[-1]}_coco.yml")
        self.config_path = str(cfg_p)

        self.model = None
        self._param_count = 0
        self._size_mb = 0.0
        self.byte_tracker = None

        self._load_model()
        self._init_tracker()

    def _load_model(self) -> None:
        """Loads and deploys genuine D-FINE model architecture and weights."""
        logger.info(f"Loading D-FINE ({self.variant}) from '{self.checkpoint_path}'...")
        ckpt_file = Path(self.checkpoint_path)
        if not ckpt_file.is_file():
            # Check models/ directory fallback
            alt_ckpt = ROOT_DIR / "models" / ckpt_file.name
            if alt_ckpt.is_file():
                self.checkpoint_path = str(alt_ckpt)
                ckpt_file = alt_ckpt
            else:
                raise FileNotFoundError(f"D-FINE weights not found at '{self.checkpoint_path}'.")

        cfg_file = Path(self.config_path)
        if not cfg_file.is_file():
            alt_cfg = DFINE_DIR / "configs" / "dfine" / cfg_file.name
            if alt_cfg.is_file():
                self.config_path = str(alt_cfg)
            else:
                raise FileNotFoundError(f"D-FINE config not found at '{self.config_path}'.")

        try:
            from src.core import YAMLConfig

            cfg = YAMLConfig(self.config_path, resume=self.checkpoint_path)
            if "HGNetv2" in cfg.yaml_cfg:
                cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            state = checkpoint.get("ema", {}).get("module", checkpoint.get("model", {}))
            cfg.model.load_state_dict(state)

            class _DeployedDFINE(torch.nn.Module):
                def __init__(self, core_model, postprocessor):
                    super().__init__()
                    self.model = core_model.deploy()
                    self.postprocessor = postprocessor.deploy()

                def forward(self, images, orig_target_sizes):
                    outputs = self.model(images)
                    outputs = self.postprocessor(outputs, orig_target_sizes)
                    return outputs

            self.model = _DeployedDFINE(cfg.model, cfg.postprocessor).to(self.device)
            self.model.eval()

            # Count parameters
            self._param_count = sum(p.numel() for p in self.model.parameters())
            self._size_mb = round(ckpt_file.stat().st_size / (1024 * 1024), 2)

            logger.info(
                f"Successfully initialized D-FINE ({self.variant}): "
                f"{self._param_count:,} params ({self._param_count/1e6:.2f}M), Size: {self._size_mb:.2f} MB."
            )
        except Exception as e:
            logger.error(f"Failed to initialize D-FINE ({self.variant}): {e}")
            raise RuntimeError(f"D-FINE initialization failed: {e}") from e

    def _init_tracker(self) -> None:
        """Initializes Ultralytics BYTETracker instance with standardized configuration."""
        from ultralytics.trackers.byte_tracker import BYTETracker

        # Read parameters from TrackerConfig
        tracker_args = SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=getattr(self.tracker_config, "track_high_thresh", 0.30),
            track_low_thresh=getattr(self.tracker_config, "track_low_thresh", 0.10),
            new_track_thresh=getattr(self.tracker_config, "new_track_thresh", 0.35),
            track_buffer=getattr(self.tracker_config, "track_buffer", 60),
            match_thresh=getattr(self.tracker_config, "match_thresh", 0.75),
            min_box_area=0.0,
            mot20=False,
        )
        self.byte_tracker = BYTETracker(tracker_args, frame_rate=24)
        self.active_tracks.clear()
        logger.info(f"Initialized BYTETracker for D-FINE (track_high={tracker_args.track_high_thresh}, match={tracker_args.match_thresh}).")

    def reset(self) -> None:
        """Resets tracker state."""
        self._init_tracker()

    @property
    def tracker(self):
        """Returns self as tracker interface for SafeKidPipeline compatibility."""
        return self

    @property
    def backend_name(self) -> str:
        return "dfine"

    @property
    def model_name(self) -> str:
        return self.variant

    @property
    def parameter_count(self) -> int:
        return self._param_count

    @property
    def model_size_mb(self) -> float:
        return self._size_mb

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp_ms: float = 0.0,
        confidence_thresh: Optional[float] = None,
        iou_thresh: Optional[float] = None,
    ) -> FrameDetections:
        """Executes D-FINE person detection on a single frame."""
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None.")

        h, w = frame.shape[:2]
        conf_thresh = confidence_thresh if confidence_thresh is not None else self.config.confidence_threshold
        nms_iou_thresh = iou_thresh if iou_thresh is not None else self.config.iou_threshold
        imgsz = getattr(self.config, "imgsz", 640)

        # Preprocess frame
        im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_resized = cv2.resize(im_rgb, (imgsz, imgsz))
        im_tensor = (
            torch.from_numpy(im_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        ).to(self.device)
        orig_size = torch.tensor([[w, h]], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            labels, boxes, scores = self.model(im_tensor, orig_size)

        l_np = labels[0].cpu().numpy().astype(int)
        b_np = boxes[0].cpu().numpy()
        s_np = scores[0].cpu().numpy()

        # Filter for person class (0)
        person_mask = (l_np == self.config.person_class_id) & (s_np >= conf_thresh)
        filtered_boxes = b_np[person_mask]
        filtered_scores = s_np[person_mask]

        if len(filtered_boxes) == 0:
            return FrameDetections(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                detections=[],
            )

        # Apply NMS to remove redundant query duplicates
        b_t = torch.from_numpy(filtered_boxes).float()
        s_t = torch.from_numpy(filtered_scores).float()
        keep_indices = nms(b_t, s_t, nms_iou_thresh).numpy()

        det_boxes: List[DetectionBox] = []
        for idx in keep_indices:
            bx = filtered_boxes[idx]
            sc = float(filtered_scores[idx])

            x1 = float(max(0.0, min(float(w - 1), bx[0])))
            y1 = float(max(0.0, min(float(h - 1), bx[1])))
            x2 = float(max(x1 + 1.0, min(float(w), bx[2])))
            y2 = float(max(y1 + 1.0, min(float(h), bx[3])))

            if x2 <= x1 or y2 <= y1:
                continue

            det_boxes.append(
                DetectionBox(
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=sc,
                    class_id=0,
                    class_name="person",
                    track_id=None,
                )
            )

        return FrameDetections(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            detections=det_boxes,
        )

    def track(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp_ms: float = 0.0,
        confidence_thresh: Optional[float] = None,
        iou_thresh: Optional[float] = None,
    ) -> FrameDetections:
        """Executes D-FINE person detection coupled with ByteTrack multi-object tracking."""
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None.")

        # 1. Run detection forward pass with low association threshold to feed ByteTrack's 2-stage association
        raw_det = self.detect(
            frame=frame,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            confidence_thresh=getattr(self.tracker_config, "track_low_thresh", 0.10),
            iou_thresh=iou_thresh,
        )

        if not raw_det.detections:
            return FrameDetections(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                detections=[],
            )

        # 2. Package detections for BYTETracker
        xyxy_list = [d.bbox_xyxy for d in raw_det.detections]
        conf_list = [d.confidence for d in raw_det.detections]
        cls_list = [0 for _ in raw_det.detections]

        det_container = _DetectionContainer(
            xyxy=np.asarray(xyxy_list, dtype=np.float32).reshape(-1, 4),
            conf=np.asarray(conf_list, dtype=np.float32),
            cls=np.asarray(cls_list, dtype=np.int64),
        )

        # 3. Update ByteTrack
        tracks = self.byte_tracker.update(det_container)

        # 4. Map assigned track IDs to detections
        final_boxes: List[DetectionBox] = []
        user_conf_thresh = confidence_thresh if confidence_thresh is not None else getattr(self.tracker_config, "confidence_threshold", 0.35)

        if tracks is not None and len(tracks) > 0:
            for t in tracks:
                x1, y1, x2, y2, tid, score, cid = t[:7]
                if score < user_conf_thresh:
                    continue
                tid = int(tid)

                det_box = DetectionBox(
                    bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(score),
                    class_id=0,
                    class_name="person",
                    track_id=tid,
                )
                final_boxes.append(det_box)

                # Maintain TrackedPerson history
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
                        max_history=getattr(self.tracker_config, "track_history_length", 60),
                    )

        # Fallback: if tracking produced 0 tracks (e.g. single unconfirmed frame), return high-confidence detection boxes
        if not final_boxes:
            for d in raw_det.detections:
                if d.confidence >= user_conf_thresh:
                    final_boxes.append(d)

        return FrameDetections(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            detections=final_boxes,
        )
