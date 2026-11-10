"""Standardized pedestrian crop extraction with aspect-ratio preservation and padding.

Extracts person bounding boxes from video frames with:
- Boundary clipping to prevent out-of-frame index errors.
- Aspect-ratio preservation via symmetric letterbox padding.
- Standardized target dimensions (default 128x64 HxW).
- Low-resolution detection (<80px height) categorized for analysis.
- Detailed geometry and provenance metadata tracking.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.crop_extractor")


@dataclass
class ExtractedCropResult:
    """Encapsulates an extracted, aspect-ratio preserved pedestrian crop."""
    crop_image: np.ndarray             # Standardized padded image (target_h, target_w, 3)
    raw_crop: np.ndarray              # Unpadded original crop
    original_bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    clipped_bbox: Tuple[float, float, float, float]   # [x1, y1, x2, y2]
    original_width: int
    original_height: int
    target_width: int
    target_height: int
    scale: float
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    is_valid: bool = True
    is_low_res: bool = False
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes crop extraction metadata."""
        return {
            "original_bbox": [round(c, 1) for c in self.original_bbox],
            "clipped_bbox": [round(c, 1) for c in self.clipped_bbox],
            "original_width": self.original_width,
            "original_height": self.original_height,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "scale": round(self.scale, 4),
            "padding": {
                "top": self.pad_top,
                "bottom": self.pad_bottom,
                "left": self.pad_left,
                "right": self.pad_right,
            },
            "is_valid": self.is_valid,
            "is_low_res": self.is_low_res,
            "rejection_reason": self.rejection_reason,
            "metadata": self.metadata,
        }


class PersonCropExtractor:
    """Extracts standardized, aspect-ratio preserved pedestrian crops from video frames."""

    def __init__(
        self,
        target_height: int = 128,
        target_width: int = 64,
        min_crop_height: int = 40,
        min_crop_width: int = 16,
        low_res_height_threshold: int = 80,
        pad_color: Tuple[int, int, int] = (0, 0, 0),
    ):
        """Initializes crop extractor.
        
        Args:
            target_height: Target output image height in pixels (default 128).
            target_width: Target output image width in pixels (default 64).
            min_crop_height: Minimum allowed unclipped bounding box height (default 40).
            min_crop_width: Minimum allowed unclipped bounding box width (default 16).
            low_res_height_threshold: Height threshold below which crops are tagged low-res (default 80).
            pad_color: BGR/RGB fill color for symmetric letterbox padding (default black).
        """
        self.target_height = target_height
        self.target_width = target_width
        self.min_crop_height = min_crop_height
        self.min_crop_width = min_crop_width
        self.low_res_height_threshold = low_res_height_threshold
        self.pad_color = pad_color

    def extract_crop(
        self,
        frame: np.ndarray,
        bbox_xyxy: Tuple[float, float, float, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExtractedCropResult:
        """Extracts and pads a pedestrian crop from a full video frame.
        
        Args:
            frame: Image array of shape (H, W, 3) or (H, W).
            bbox_xyxy: Bounding box tuple (x1, y1, x2, y2).
            metadata: Optional dictionary with source provenance (video_id, frame_id, track_id).
            
        Returns:
            ExtractedCropResult containing standardized crop array and geometry stats.
        """
        meta = metadata or {}
        empty_img = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)

        if frame is None or frame.size == 0:
            return ExtractedCropResult(
                crop_image=empty_img,
                raw_crop=np.empty((0, 0, 3), dtype=np.uint8),
                original_bbox=bbox_xyxy,
                clipped_bbox=bbox_xyxy,
                original_width=0,
                original_height=0,
                target_width=self.target_width,
                target_height=self.target_height,
                scale=0.0,
                pad_top=0,
                pad_bottom=0,
                pad_left=0,
                pad_right=0,
                is_valid=False,
                is_low_res=True,
                rejection_reason="Empty or None input frame",
                metadata=meta,
            )

        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy

        # Clip coordinates to frame boundary
        cx1 = max(0.0, min(float(fw), float(x1)))
        cy1 = max(0.0, min(float(fh), float(y1)))
        cx2 = max(0.0, min(float(fw), float(x2)))
        cy2 = max(0.0, min(float(fh), float(y2)))

        orig_w = int(round(cx2 - cx1))
        orig_h = int(round(cy2 - cy1))

        clipped_bbox = (cx1, cy1, cx2, cy2)

        # Check geometry thresholds
        if orig_w < self.min_crop_width or orig_h < self.min_crop_height:
            return ExtractedCropResult(
                crop_image=empty_img,
                raw_crop=np.empty((0, 0, 3), dtype=np.uint8),
                original_bbox=bbox_xyxy,
                clipped_bbox=clipped_bbox,
                original_width=orig_w,
                original_height=orig_h,
                target_width=self.target_width,
                target_height=self.target_height,
                scale=0.0,
                pad_top=0,
                pad_bottom=0,
                pad_left=0,
                pad_right=0,
                is_valid=False,
                is_low_res=True,
                rejection_reason=(
                    f"Sub-minimum crop geometry {orig_w}x{orig_h} "
                    f"(min {self.min_crop_width}x{self.min_crop_height})"
                ),
                metadata=meta,
            )

        ix1, iy1, ix2, iy2 = int(round(cx1)), int(round(cy1)), int(round(cx2)), int(round(cy2))
        raw_crop = frame[iy1:iy2, ix1:ix2]

        if raw_crop.size == 0 or raw_crop.shape[0] == 0 or raw_crop.shape[1] == 0:
            return ExtractedCropResult(
                crop_image=empty_img,
                raw_crop=raw_crop,
                original_bbox=bbox_xyxy,
                clipped_bbox=clipped_bbox,
                original_width=orig_w,
                original_height=orig_h,
                target_width=self.target_width,
                target_height=self.target_height,
                scale=0.0,
                pad_top=0,
                pad_bottom=0,
                pad_left=0,
                pad_right=0,
                is_valid=False,
                is_low_res=True,
                rejection_reason="Empty slice from frame",
                metadata=meta,
            )

        is_low_res = orig_h < self.low_res_height_threshold

        # Compute aspect-ratio preserving scaling factor
        scale_w = self.target_width / orig_w
        scale_h = self.target_height / orig_h
        scale = min(scale_w, scale_h)

        new_w = max(1, min(self.target_width, int(round(orig_w * scale))))
        new_h = max(1, min(self.target_height, int(round(orig_h * scale))))

        # Resize preserving aspect ratio
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized_crop = cv2.resize(raw_crop, (new_w, new_h), interpolation=interp)

        # Place on canvas with symmetric letterbox padding
        canvas = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        if self.pad_color != (0, 0, 0):
            canvas[:] = self.pad_color

        pad_top = (self.target_height - new_h) // 2
        pad_bottom = self.target_height - new_h - pad_top
        pad_left = (self.target_width - new_w) // 2
        pad_right = self.target_width - new_w - pad_left

        if len(resized_crop.shape) == 2:
            resized_crop = cv2.cvtColor(resized_crop, cv2.COLOR_GRAY2BGR)

        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_crop

        return ExtractedCropResult(
            crop_image=canvas,
            raw_crop=raw_crop,
            original_bbox=bbox_xyxy,
            clipped_bbox=clipped_bbox,
            original_width=orig_w,
            original_height=orig_h,
            target_width=self.target_width,
            target_height=self.target_height,
            scale=scale,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            pad_left=pad_left,
            pad_right=pad_right,
            is_valid=True,
            is_low_res=is_low_res,
            rejection_reason=None,
            metadata=meta,
        )

