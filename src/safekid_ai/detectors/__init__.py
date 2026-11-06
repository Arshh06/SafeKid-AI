"""SafeKid AI person detection subsystem and detector adapters."""

from typing import Optional, Union
from safekid_ai.config import DetectorConfig, TrackerConfig
from safekid_ai.detectors.base import BasePersonDetector
from safekid_ai.detectors.yolov8_detector import YOLOv8DetectorAdapter
from safekid_ai.detectors.rtdetrv2_detector import RTDETRv2DetectorAdapter
from safekid_ai.detectors.dfine_detector import DFINEDetectorAdapter
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.detectors")

__all__ = [
    "BasePersonDetector",
    "YOLOv8DetectorAdapter",
    "RTDETRv2DetectorAdapter",
    "DFINEDetectorAdapter",
    "create_detector",
]


def create_detector(
    detector_config: Optional[Union[DetectorConfig, dict]] = None,
    tracker_config: Optional[TrackerConfig] = None,
) -> BasePersonDetector:
    """Factory function creating a BasePersonDetector adapter according to configuration.

    Args:
        detector_config: DetectorConfig instance or dictionary of parameters.
        tracker_config: Optional TrackerConfig instance.

    Returns:
        Instantiated BasePersonDetector adapter (YOLOv8DetectorAdapter, RTDETRv2DetectorAdapter, or DFINEDetectorAdapter).
    """
    if detector_config is None:
        cfg = DetectorConfig()
    elif isinstance(detector_config, str):
        cfg = DetectorConfig(backend=detector_config)
    elif isinstance(detector_config, dict):
        cfg = DetectorConfig(**detector_config)
    else:
        cfg = detector_config

    backend = getattr(cfg, "backend", "yolov8").lower()

    if backend in ("dfine", "dfine_n", "dfine_s", "dfine-n", "dfine-s"):
        variant = "dfine_s" if "s" in backend else getattr(cfg, "model_variant", "dfine_n")
        checkpoint = getattr(cfg, "model_name_or_path", None)
        logger.info(f"Instantiating DFINEDetectorAdapter with variant: {variant}")
        return DFINEDetectorAdapter(
            config=cfg,
            tracker_config=tracker_config,
            variant=variant,
            checkpoint_path=checkpoint if checkpoint and checkpoint.endswith(".pth") else None,
        )
    elif backend in ("rtdetr", "rtdetrv2", "transformer"):
        checkpoint = getattr(cfg, "model_name_or_path", "rtdetr-l.pt")
        if not checkpoint or checkpoint.startswith("yolo"):
            checkpoint = "rtdetr-l.pt"
        logger.info(f"Instantiating RTDETRv2DetectorAdapter with checkpoint: {checkpoint}")
        return RTDETRv2DetectorAdapter(
            config=cfg,
            tracker_config=tracker_config,
            checkpoint_path=checkpoint,
        )
    elif backend in ("yolo", "yolov8", "baseline"):
        logger.info(f"Instantiating YOLOv8DetectorAdapter with model: {cfg.model_name_or_path}")
        return YOLOv8DetectorAdapter(
            config=cfg,
            tracker_config=tracker_config,
        )
    else:
        raise ValueError(
            f"Unsupported detector backend '{backend}'. Supported backends: 'yolov8', 'rtdetrv2', 'dfine'."
        )

