"""Tests for data_models.py."""

import json
import numpy as np
import pytest

from src.data_models import (
    BBox, BallState, Detection, FrameFlag, FrameState,
    HomographyResult, Keypoint, ObjectClass, PitchPosition, PlayerState,
)


class TestBBox:
    def test_center(self):
        b = BBox(10, 20, 30, 40)
        assert b.center == (20.0, 30.0)

    def test_bottom_center(self):
        b = BBox(10, 20, 30, 40)
        assert b.bottom_center == (20.0, 40.0)

    def test_area(self):
        b = BBox(0, 0, 10, 20)
        assert b.area == 200.0

    def test_area_zero(self):
        b = BBox(5, 5, 5, 5)
        assert b.area == 0.0


class TestFrameState:
    def test_basic_serialization(self):
        fs = FrameState(frame_idx=0, timestamp_sec=0.0)
        d = fs.to_serializable()
        assert d["frame_idx"] == 0
        assert d["flag"] == "in_play"
        assert d["players"] == []
        assert d["ball"] is None

    def test_with_player(self):
        det = Detection(
            bbox=BBox(10, 20, 50, 80),
            class_id=ObjectClass.PLAYER,
            confidence=0.9,
        )
        player = PlayerState(
            track_id=1,
            detection=det,
            team_id=0,
            pitch_pos=PitchPosition(x=52.5, y=34.0),
            speed_mps=3.5,
        )
        fs = FrameState(
            frame_idx=10,
            timestamp_sec=0.333,
            players=[player],
        )
        d = fs.to_serializable()
        assert len(d["players"]) == 1
        assert d["players"][0]["track_id"] == 1
        assert d["players"][0]["pitch_x"] == 52.5

    def test_with_ball(self):
        ball_det = Detection(
            bbox=BBox(100, 200, 110, 210),
            class_id=ObjectClass.BALL,
            confidence=0.8,
        )
        ball = BallState(
            detection=ball_det,
            pitch_pos=PitchPosition(x=60.0, y=30.0),
            speed_mps=15.0,
        )
        fs = FrameState(frame_idx=5, timestamp_sec=0.166, ball=ball)
        d = fs.to_serializable()
        assert d["ball"]["pitch_x"] == 60.0
        assert d["ball"]["speed_mps"] == 15.0

    def test_to_json(self):
        fs = FrameState(frame_idx=0, timestamp_sec=0.0)
        j = fs.to_json()
        parsed = json.loads(j)
        assert parsed["frame_idx"] == 0


class TestHomographyResult:
    def test_serialization_none(self):
        h = HomographyResult()
        d = h.to_serializable()
        assert d["matrix"] is None
        assert d["available"] is False

    def test_serialization_with_matrix(self):
        H = np.eye(3)
        h = HomographyResult(matrix=H, quality=0.9, num_inliers=8, available=True)
        d = h.to_serializable()
        assert d["matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert d["available"] is True
