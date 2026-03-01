"""Tests for the homography estimator."""

import numpy as np
import pytest

from src.data_models import FrameState, Keypoint
from src.homography.estimator import HomographyEstimator


@pytest.fixture
def config():
    return {
        "homography": {
            "method": "dlt",
            "min_points": 4,
            "quality_threshold": 0.6,
            "smoothing_window": 5,
        },
        "pitch": {"length": 105.0, "width": 68.0},
    }


class TestHomographyEstimator:
    def test_insufficient_keypoints(self, config):
        est = HomographyEstimator(config)
        fs = FrameState(frame_idx=0, timestamp_sec=0.0)
        fs.keypoints = [Keypoint(x=100, y=100, confidence=0.8)]
        est.estimate(fs)
        assert fs.homography.available is False

    def test_pixel_to_pitch(self, config):
        est = HomographyEstimator(config)
        # Identity homography
        H = np.eye(3)
        px, py = est.pixel_to_pitch(H, 52.5, 34.0)
        assert px == pytest.approx(52.5, abs=0.01)
        assert py == pytest.approx(34.0, abs=0.01)

    def test_pixel_to_pitch_with_scaling(self, config):
        est = HomographyEstimator(config)
        # Scale homography: maps (0..1920, 0..1080) to (0..105, 0..68)
        H = np.array([
            [105.0 / 1920, 0, 0],
            [0, 68.0 / 1080, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        px, py = est.pixel_to_pitch(H, 960, 540)
        assert px == pytest.approx(52.5, abs=0.1)
        assert py == pytest.approx(34.0, abs=0.1)
