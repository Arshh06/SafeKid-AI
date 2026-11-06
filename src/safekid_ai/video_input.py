"""Video and image input module using OpenCV for SafeKid AI."""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import cv2
import numpy as np

from safekid_ai.config import VideoInputConfig
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.video_input")


@dataclass
class VideoStreamMetadata:
    """Metadata describing the video or image stream."""
    source: str
    width: int
    height: int
    fps: float
    total_frames: int
    is_live_stream: bool
    is_single_image: bool


class VideoInput:
    """Robust OpenCV video and image ingestion handler with frame extraction.

    Supports camera index, local video files, RTSP/HTTP streams, and static images.
    Provides context management, frame skipping, resizing, and clean resource release.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(self, config: Optional[VideoInputConfig] = None, source: Optional[str] = None):
        if config is None:
            self.config = VideoInputConfig(source=source if source is not None else "0")
        else:
            self.config = config
            if source is not None:
                self.config.source = source

        self.source_str = str(self.config.source)
        self.cap: Optional[cv2.VideoCapture] = None
        self._parsed_src: Optional[Union[int, str]] = None
        self._is_single_image = False
        self._single_image_cached: Optional[np.ndarray] = None
        self._metadata: Optional[VideoStreamMetadata] = None

        # Monotonic time tracking for live camera / RTSP streams
        self._stream_start_monotonic: Optional[float] = None
        self._last_emitted_timestamp_ms: float = 0.0
        self._reconnect_count: int = 0

        self._check_source_type()

    def _check_source_type(self) -> None:
        """Determines whether the source is a single image file, video file, or camera/stream."""
        path = Path(self.source_str)
        if path.exists() and path.is_file():
            if path.suffix.lower() in self.IMAGE_EXTENSIONS:
                self._is_single_image = True
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Failed to decode image from path: {self.source_str}")
                self._single_image_cached = image
                h, w = image.shape[:2]
                self._metadata = VideoStreamMetadata(
                    source=self.source_str,
                    width=w,
                    height=h,
                    fps=1.0,
                    total_frames=1,
                    is_live_stream=False,
                    is_single_image=True,
                )
                logger.info(f"Loaded single image source: {self.source_str} ({w}x{h})")
                return

    def open(self) -> "VideoInput":
        """Opens the video source if not a static image."""
        if self._is_single_image:
            return self

        # Parse source: integer device index or filepath / stream URL
        if self.source_str.isdigit():
            src: Union[int, str] = int(self.source_str)
            is_stream = True
        else:
            src = self.source_str
            is_stream = not Path(self.source_str).exists()

        self._parsed_src = src
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video stream or device at source: {self.source_str}")

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0 or np.isnan(fps):
            fps = 30.0  # Default fallback for live streams lacking header FPS
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 0:
            total_frames = 0

        self._metadata = VideoStreamMetadata(
            source=self.source_str,
            width=w,
            height=h,
            fps=fps,
            total_frames=total_frames,
            is_live_stream=is_stream,
            is_single_image=False,
        )

        # Initialize monotonic stream epoch if live stream
        if is_stream and self._stream_start_monotonic is None:
            self._stream_start_monotonic = time.monotonic()

        logger.info(
            f"Opened video source: {self.source_str} | Resolution: {w}x{h} | "
            f"FPS: {fps:.2f} | Frames: {total_frames} | Stream: {is_stream}"
        )
        return self

    def reconnect(self) -> bool:
        """Safely reconnects an open live camera or RTSP stream without resetting monotonic clock.

        Preserves _stream_start_monotonic and _last_emitted_timestamp_ms so that dwell
        calculations and timestamps never roll backwards upon stream reconnection.
        """
        if not self._metadata or not self._metadata.is_live_stream:
            logger.warning("reconnect() called on non-live or uninitialized source; ignoring.")
            return False

        logger.warning(f"Reconnecting live stream source: {self.source_str} (attempt #{self._reconnect_count + 1})")
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.debug(f"Error releasing previous capture handle during reconnect: {e}")

        src = self._parsed_src if self._parsed_src is not None else self.source_str
        self.cap = cv2.VideoCapture(src)
        if self.cap.isOpened():
            self._reconnect_count += 1
            logger.info(f"Successfully reconnected live stream (reconnect #{self._reconnect_count}).")
            return True

        logger.error(f"Failed to reconnect live stream source: {self.source_str}")
        return False

    def get_metadata(self) -> VideoStreamMetadata:
        """Returns metadata for the open stream or image."""
        if self._metadata is None:
            self.open()
        assert self._metadata is not None
        return self._metadata

    def frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yields tuples of (frame_id, timestamp_ms, frame_bgr).

        Dwell Time Semantics:
        - For recorded files (is_live_stream=False):
          timestamp_ms represents media presentation time (container PTS or frame_idx/fps).
          Dwell time measures elapsed media playback duration.
        - For live camera / RTSP streams (is_live_stream=True):
          timestamp_ms represents monotonic stream elapsed duration via time.monotonic().
          Dwell time measures true real-world elapsed time, immune to dropped frames,
          irregular capture intervals, NTP corrections, and stream reconnections.

        Yields:
            frame_id: 0-indexed frame count.
            timestamp_ms: Monotonic timestamp in milliseconds.
            frame_bgr: OpenCV BGR image array (uint8).
        """
        if self._is_single_image:
            assert self._single_image_cached is not None
            frame = self._single_image_cached.copy()
            if self.config.resize_dims:
                frame = cv2.resize(frame, self.config.resize_dims, interpolation=cv2.INTER_LINEAR)
            yield 0, 0.0, frame
            return

        if self.cap is None or not self.cap.isOpened():
            self.open()

        assert self.cap is not None
        assert self._metadata is not None
        frame_idx = 0
        skip = max(0, self.config.frame_skip)
        is_live = self._metadata.is_live_stream

        # Ensure monotonic clock is set for live stream
        if is_live and self._stream_start_monotonic is None:
            self._stream_start_monotonic = time.monotonic()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                if self.config.loop_video and not is_live:
                    logger.debug("Looping video stream back to beginning.")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                else:
                    break

            if skip > 0 and (frame_idx % (skip + 1) != 0):
                frame_idx += 1
                continue

            if self.config.resize_dims:
                frame = cv2.resize(frame, self.config.resize_dims, interpolation=cv2.INTER_LINEAR)

            # Determine timestamp based on stream type
            if is_live:
                # Live stream: derive from monotonic clock
                assert self._stream_start_monotonic is not None
                elapsed_ms = (time.monotonic() - self._stream_start_monotonic) * 1000.0
                timestamp_ms = max(self._last_emitted_timestamp_ms, elapsed_ms)
            else:
                # Recorded media: derive from container PTS with fallback
                pos_msec = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
                if pos_msec > 0.0:
                    raw_ts = pos_msec
                elif self._metadata.fps > 0:
                    raw_ts = (frame_idx / self._metadata.fps) * 1000.0
                else:
                    raw_ts = frame_idx * 33.333

                # Guarantee strictly non-decreasing timestamps (e.g. video loops or corrupted PTS)
                timestamp_ms = max(self._last_emitted_timestamp_ms, raw_ts)

            self._last_emitted_timestamp_ms = timestamp_ms

            yield frame_idx, timestamp_ms, frame
            frame_idx += 1

    def release(self) -> None:
        """Releases underlying OpenCV resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.debug(f"Released video capture for source: {self.source_str}")

    def __enter__(self) -> "VideoInput":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

