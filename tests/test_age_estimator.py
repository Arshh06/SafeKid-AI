"""Unit tests for MiVOLOAgeEstimator with privacy and research consent gating."""

import numpy as np
import pytest

from safekid_ai.age_estimator import (
    AgeEstimationResult,
    AgeEstimationStatus,
    MiVOLOAgeEstimator,
    classify_age_group,
)
from safekid_ai.config import AgeEstimatorConfig


@pytest.fixture(scope="module")
def age_estimator():
    """Initializes official MiVOLO estimator."""
    config = AgeEstimatorConfig(
        device="cpu",
        half=False,
        allow_research_inference=False,
        consent_verified=False,
    )
    return MiVOLOAgeEstimator(config)


def test_operational_mode_blocks_estimation(age_estimator):
    """Ensures age estimation is strictly prohibited in operational community monitoring."""
    crop = np.full((120, 120, 3), 128, dtype=np.uint8)
    res: AgeEstimationResult = age_estimator.estimate(
        face_crop=crop,
        consent_override=False,
        research_mode_override=False,
    )
    assert res.status == AgeEstimationStatus.RESTRICTED_OPERATIONAL_MODE
    assert res.estimated_age is None
    assert res.age_group == "UNKNOWN"
    assert res.is_child is None
    assert "operational residential monitoring mode" in res.reason


def test_research_mode_without_consent_blocks(age_estimator):
    """Ensures research evaluation fails if consent is not verified."""
    crop = np.full((120, 120, 3), 128, dtype=np.uint8)
    res = age_estimator.estimate(
        face_crop=crop,
        consent_override=False,
        research_mode_override=True,
    )
    assert res.status == AgeEstimationStatus.NO_CONSENT_VERIFIED
    assert res.estimated_age is None
    assert res.age_group == "UNKNOWN"


def test_missing_crop_fails_gracefully(age_estimator):
    """Ensures missing crop is handled gracefully without exception."""
    res = age_estimator.estimate(
        face_crop=None,
        person_crop=None,
        consent_override=True,
        research_mode_override=True,
    )
    assert res.status == AgeEstimationStatus.CROP_UNAVAILABLE
    assert res.estimated_age is None


def test_consented_research_inference_succeeds(age_estimator):
    """Verifies genuine MiVOLO inference when research mode and consent are both verified."""
    if not age_estimator.is_available:
        pytest.skip("MiVOLO weights not available in this test environment.")

    # Simulated test crop
    crop = np.full((224, 224, 3), 140, dtype=np.uint8)
    res = age_estimator.estimate(
        face_crop=crop,
        consent_override=True,
        research_mode_override=True,
    )
    assert res.status == AgeEstimationStatus.PREDICTED
    assert res.estimated_age is not None
    assert isinstance(res.estimated_age, float)
    assert res.estimated_age >= 0.0
    assert res.age_group in ("CHILD", "TEEN", "ADULT")
    assert res.inference_time_ms > 0.0


def test_classify_age_group():
    """Tests age group categorization thresholds."""
    assert classify_age_group(None) == "UNKNOWN"
    assert classify_age_group(float("nan")) == "UNKNOWN"
    assert classify_age_group(5.5) == "CHILD"
    assert classify_age_group(12.9) == "CHILD"
    assert classify_age_group(13.0) == "TEEN"
    assert classify_age_group(17.9) == "TEEN"
    assert classify_age_group(18.0) == "ADULT"
    assert classify_age_group(45.0) == "ADULT"

