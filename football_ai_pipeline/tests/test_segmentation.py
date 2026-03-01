"""Tests for the in-play filter."""

import numpy as np
import pytest

from src.data_models import FrameFlag, FrameState
from src.segmentation import InPlayFilter


@pytest.fixture
def config():
    return {
        "segmentation": {
            "scene_change_threshold": 30.0,
            "min_play_duration": 30,
        }
    }


class TestInPlayFilter:
    def test_green_frame_is_in_play(self, config):
        filt = InPlayFilter(config)
        # Create a mostly-green image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [40, 180, 60]  # BGR green
        fs = FrameState(frame_idx=0, timestamp_sec=0.0, image=img)
        result = filt.classify(fs)
        assert result.flag == FrameFlag.IN_PLAY

    def test_black_frame_is_not_in_play(self, config):
        filt = InPlayFilter(config)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        fs = FrameState(frame_idx=0, timestamp_sec=0.0, image=img)
        result = filt.classify(fs)
        assert result.flag == FrameFlag.NOT_IN_PLAY
        assert any("low_green" in r for r in result.flag_reasons)

    def test_no_image(self, config):
        filt = InPlayFilter(config)
        fs = FrameState(frame_idx=0, timestamp_sec=0.0)
        result = filt.classify(fs)
        assert result.flag == FrameFlag.NOT_IN_PLAY

    def test_scene_change_detection(self, config):
        filt = InPlayFilter(config)
        # First frame: bright
        img1 = np.full((100, 100, 3), 200, dtype=np.uint8)
        img1[:, :] = [40, 180, 60]  # green enough
        fs1 = FrameState(frame_idx=0, timestamp_sec=0.0, image=img1)
        filt.classify(fs1)

        # Second frame: dramatically different (scene change)
        img2 = np.full((100, 100, 3), 10, dtype=np.uint8)
        fs2 = FrameState(frame_idx=1, timestamp_sec=0.033, image=img2)
        result = filt.classify(fs2)
        assert result.flag == FrameFlag.NOT_IN_PLAY
