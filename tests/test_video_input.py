"""Unit tests for the VideoInput module."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from safekid_ai.config import VideoInputConfig
from safekid_ai.video_input import VideoInput


@pytest.fixture
def sample_image_file():
    """Creates a temporary test image file."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img_path = f.name
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(img, "SafeKid AI Test", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(img_path, img)
    yield Path(img_path)
    if Path(img_path).exists():
        Path(img_path).unlink()


@pytest.fixture
def sample_video_file():
    """Creates a temporary short synthetic MP4 video file (5 frames)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        vid_path = f.name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(vid_path, fourcc, 10.0, (160, 120))
    for i in range(5):
        frame = np.full((120, 160, 3), i * 40, dtype=np.uint8)
        out.write(frame)
    out.release()
    yield Path(vid_path)
    if Path(vid_path).exists():
        Path(vid_path).unlink()


def test_single_image_input(sample_image_file):
    """Verifies single image loading, metadata, and single frame yield."""
    vin = VideoInput(source=str(sample_image_file))
    meta = vin.get_metadata()
    assert meta.is_single_image is True
    assert meta.width == 320
    assert meta.height == 240
    assert meta.total_frames == 1

    frames = list(vin.frames())
    assert len(frames) == 1
    fid, ts, frame = frames[0]
    assert fid == 0
    assert frame.shape == (240, 320, 3)
    vin.release()


def test_video_file_input(sample_video_file):
    """Verifies reading multiple frames from a synthetic video."""
    vin = VideoInput(source=str(sample_video_file))
    with vin:
        meta = vin.get_metadata()
        assert meta.is_single_image is False
        assert meta.width == 160
        assert meta.height == 120

        frames = list(vin.frames())
        assert len(frames) == 5
        assert frames[0][0] == 0
        assert frames[4][0] == 4


def test_frame_skipping_and_resizing(sample_video_file):
    """Verifies frame skip and resizing parameters."""
    config = VideoInputConfig(
        source=str(sample_video_file),
        frame_skip=1,  # Process every 2nd frame
        resize_dims=(80, 60),
    )
    vin = VideoInput(config=config)
    with vin:
        frames = list(vin.frames())
        # From 5 frames, skip 1 -> frames 0, 2, 4 = 3 frames
        assert len(frames) == 3
        for fid, ts, frame in frames:
            assert frame.shape == (60, 80, 3)


def test_non_existent_source():
    """Ensures meaningful exceptions on missing files."""
    vin = VideoInput(source="non_existent_camera_stream_path_12345.mp4")
    with pytest.raises(RuntimeError):
        vin.open()

