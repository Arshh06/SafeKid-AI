"""MiVOLO age estimation research module for SafeKid AI.

STRICT GOVERNANCE NOTE:
Age estimation is strictly restricted to consented research images and benchmark
evaluation datasets. It is NOT permitted to make operational surveillance,
access-control, or automated safety enforcement decisions in residential communities.
Preserves an 'UNKNOWN' / unavailable output whenever estimation is not permitted
or not possible.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from safekid_ai.config import AgeEstimatorConfig
from safekid_ai.detector import DetectionBox
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.age_estimator")


class AgeEstimationStatus(str, Enum):
    """Status outcomes for age estimation."""
    PREDICTED = "PREDICTED"
    RESTRICTED_OPERATIONAL_MODE = "RESTRICTED_OPERATIONAL_MODE"
    NO_CONSENT_VERIFIED = "NO_CONSENT_VERIFIED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    CROP_UNAVAILABLE = "CROP_UNAVAILABLE"
    ESTIMATION_FAILED = "ESTIMATION_FAILED"


@dataclass
class AgeEstimationResult:
    """Structured result of age estimation."""
    status: AgeEstimationStatus
    estimated_age: Optional[float] = None
    age_group: str = "UNKNOWN"  # "UNKNOWN", "CHILD", "TEEN", "ADULT"
    gender: Optional[str] = None
    gender_confidence: Optional[float] = None
    inference_time_ms: float = 0.0
    reason: str = ""

    @property
    def is_child(self) -> Optional[bool]:
        """Returns True if estimated age is < 13, False if >= 13, None if unknown."""
        if self.estimated_age is None or self.status != AgeEstimationStatus.PREDICTED:
            return None
        return self.estimated_age < 13.0


def classify_age_group(age: Optional[float]) -> str:
    """Classifies a numeric age into standard demographic categories."""
    if age is None or np.isnan(age):
        return "UNKNOWN"
    if age < 13.0:
        return "CHILD"
    elif age < 18.0:
        return "TEEN"
    else:
        return "ADULT"


class MiVOLOAgeEstimator:
    """Official MiVOLO multi-input age & gender estimator for research evaluation.

    Complies strictly with community governance and privacy constraints:
    Requires explicit `allow_research_inference=True` and `consent_verified=True`.
    """

    def __init__(self, config: Optional[AgeEstimatorConfig] = None):
        self.config = config or AgeEstimatorConfig()
        self.model = None
        self._is_ready = False
        self._init_error: Optional[str] = None

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Loads official MiVOLO checkpoint if available."""
        try:
            from mivolo.model.mi_volo import MiVOLO
            ckpt_path = self.config.local_checkpoint_path

            if ckpt_path is None or not Path(ckpt_path).exists():
                logger.info(
                    f"Local checkpoint not specified or not found. "
                    f"Fetching official checkpoint '{self.config.checkpoint_filename}' from '{self.config.checkpoint_repo_or_path}'..."
                )
                from huggingface_hub import hf_hub_download
                ckpt_path = hf_hub_download(
                    repo_id=self.config.checkpoint_repo_or_path,
                    filename=self.config.checkpoint_filename,
                )
                self.config.local_checkpoint_path = ckpt_path

            logger.info(f"Loading official MiVOLO weights from: {ckpt_path} on device: {self.config.device}")
            self.model = MiVOLO(
                ckpt_path=ckpt_path,
                device=self.config.device,
                half=self.config.half,
                disable_faces=self.config.disable_faces,
                use_persons=self.config.use_persons,
                verbose=self.config.verbose,
            )
            self._is_ready = True
            logger.info("Official MiVOLO model initialized successfully.")
        except Exception as e:
            self._is_ready = False
            self._init_error = str(e)
            logger.warning(
                f"MiVOLO model could not be initialized ({e}). "
                f"Age estimation research module will report MODEL_UNAVAILABLE while keeping remaining pipeline functional."
            )

    @property
    def is_available(self) -> bool:
        """Whether the underlying MiVOLO model is loaded and ready for inference."""
        return self._is_ready and self.model is not None

    def estimate(
        self,
        face_crop: Optional[np.ndarray] = None,
        person_crop: Optional[np.ndarray] = None,
        consent_override: Optional[bool] = None,
        research_mode_override: Optional[bool] = None,
    ) -> AgeEstimationResult:
        """Estimates age using official MiVOLO multi-input model under research consent constraints.

        Args:
            face_crop: Optional face crop (BGR uint8 numpy array).
            person_crop: Optional person/body crop (BGR uint8 numpy array).
            consent_override: Explicit flag indicating dataset/subject consent.
            research_mode_override: Explicit flag indicating research/benchmark mode.

        Returns:
            AgeEstimationResult with predicted age or structured unknown/restricted status.
        """
        # 1. Enforce Community Safety and Privacy Governance
        is_research = (
            research_mode_override
            if research_mode_override is not None
            else self.config.allow_research_inference
        )
        is_consented = (
            consent_override
            if consent_override is not None
            else self.config.consent_verified
        )

        if not is_research:
            return AgeEstimationResult(
                status=AgeEstimationStatus.RESTRICTED_OPERATIONAL_MODE,
                estimated_age=None,
                age_group="UNKNOWN",
                reason=(
                    "Age estimation disabled in operational residential monitoring mode. "
                    "Per community ethics policy, age estimation is restricted to consented research datasets."
                ),
            )

        if not is_consented:
            return AgeEstimationResult(
                status=AgeEstimationStatus.NO_CONSENT_VERIFIED,
                estimated_age=None,
                age_group="UNKNOWN",
                reason="Research evaluation requires verified subject or benchmark dataset consent.",
            )

        # 2. Check model readiness
        if not self.is_available:
            return AgeEstimationResult(
                status=AgeEstimationStatus.MODEL_UNAVAILABLE,
                estimated_age=None,
                age_group="UNKNOWN",
                reason=f"MiVOLO model weights not available: {self._init_error}",
            )

        # 3. Check inputs
        has_face = face_crop is not None and face_crop.size > 0
        has_person = person_crop is not None and person_crop.size > 0

        if not has_face and not has_person:
            return AgeEstimationResult(
                status=AgeEstimationStatus.CROP_UNAVAILABLE,
                estimated_age=None,
                age_group="UNKNOWN",
                reason="No valid face or person image crop provided.",
            )

        # 4. Perform official MiVOLO inference
        import time
        start_t = time.perf_counter()

        try:
            from mivolo.model.mi_volo import prepare_classification_images

            f_input = prepare_classification_images(
                [face_crop if has_face else None],
                self.model.input_size,
                self.model.data_config["mean"],
                self.model.data_config["std"],
                device=self.model.device,
            )
            p_input = prepare_classification_images(
                [person_crop if has_person else None],
                self.model.input_size,
                self.model.data_config["mean"],
                self.model.data_config["std"],
                device=self.model.device,
            )

            if self.model.meta.with_persons_model:
                model_input = torch.cat((f_input, p_input), dim=1)
            else:
                model_input = f_input

            output = self.model.inference(model_input)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            if self.model.meta.only_age:
                age_raw = output[0].item()
                gender_str = None
                gender_conf = None
            else:
                age_raw = output[0, 2].item()
                g_probs = output[0, :2].softmax(-1)
                best_conf, best_idx = g_probs.topk(1)
                gender_conf = float(best_conf.item())
                # Class 0: male, Class 1: female per MiVOLO convention
                gender_classes = getattr(self.model.meta, "gender_classes", ["male", "female"])
                idx = int(best_idx.item())
                gender_str = gender_classes[idx] if idx < len(gender_classes) else "unknown"

            # Scale age using official min/max/avg meta parameters
            age_scaled = age_raw * (self.model.meta.max_age - self.model.meta.min_age) + self.model.meta.avg_age
            age_final = max(0.0, float(round(age_scaled, 2)))
            group = classify_age_group(age_final)

            return AgeEstimationResult(
                status=AgeEstimationStatus.PREDICTED,
                estimated_age=age_final,
                age_group=group,
                gender=gender_str,
                gender_confidence=gender_conf,
                inference_time_ms=elapsed_ms,
                reason="Official MiVOLO inference successful.",
            )
        except Exception as e:
            logger.error(f"Error during MiVOLO inference: {e}")
            return AgeEstimationResult(
                status=AgeEstimationStatus.ESTIMATION_FAILED,
                estimated_age=None,
                age_group="UNKNOWN",
                reason=f"Inference execution failed: {e}",
            )
