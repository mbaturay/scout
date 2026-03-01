"""Tests for stats modules."""

import numpy as np
import pytest

from src.data_models import (
    BallState, BBox, Detection, FrameFlag, FrameState,
    ObjectClass, PitchPosition, PlayerState,
)
from src.stats.physical import PhysicalStats
from src.stats.spatial import SpatialStats
from src.stats.ball_progression import BallProgressionStats
from src.stats.pressure import PressureStats
from src.stats.threat import ThreatStats
from src.stats.aggregator import StatsAggregator


def _make_frame(
    idx: int, ts: float, players: list[PlayerState] | None = None,
    ball_pos: tuple[float, float] | None = None,
) -> FrameState:
    fs = FrameState(frame_idx=idx, timestamp_sec=ts)
    if players:
        fs.players = players
    if ball_pos:
        ball_det = Detection(
            bbox=BBox(0, 0, 10, 10), class_id=ObjectClass.BALL, confidence=0.9,
        )
        fs.ball = BallState(
            detection=ball_det,
            pitch_pos=PitchPosition(x=ball_pos[0], y=ball_pos[1]),
            speed_mps=5.0,
        )
    return fs


def _make_player(
    track_id: int, team_id: int, x: float, y: float, speed: float = 3.0,
) -> PlayerState:
    det = Detection(
        bbox=BBox(0, 0, 50, 80), class_id=ObjectClass.PLAYER, confidence=0.9,
    )
    return PlayerState(
        track_id=track_id,
        detection=det,
        team_id=team_id,
        pitch_pos=PitchPosition(x=x, y=y),
        speed_mps=speed,
    )


@pytest.fixture
def config():
    return {
        "stats": {
            "sprint_speed_threshold": 7.0,
            "high_speed_threshold": 5.5,
            "pressure_radius": 5.0,
            "rolling_window_sec": 300,
            "thirds_boundaries": [35.0, 70.0],
        },
        "pitch": {"length": 105.0, "width": 68.0},
    }


class TestPhysicalStats:
    def test_distance_accumulation(self, config):
        stats = PhysicalStats(config)
        p1 = _make_player(1, 0, 50.0, 34.0, speed=5.0)
        f1 = _make_frame(0, 0.0, players=[p1])
        stats.update(f1)

        p2 = _make_player(1, 0, 55.0, 34.0, speed=5.0)
        f2 = _make_frame(1, 1.0, players=[p2])
        stats.update(f2)

        summary = stats.get_player_summary()
        assert 1 in summary
        assert summary[1]["distance_m"] == pytest.approx(5.0, abs=0.1)

    def test_sprint_detection(self, config):
        stats = PhysicalStats(config)
        # Not sprinting
        p1 = _make_player(1, 0, 50.0, 34.0, speed=3.0)
        stats.update(_make_frame(0, 0.0, players=[p1]))
        # Sprinting
        p2 = _make_player(1, 0, 55.0, 34.0, speed=8.0)
        stats.update(_make_frame(1, 1.0, players=[p2]))

        summary = stats.get_player_summary()
        assert summary[1]["sprint_count"] == 1
        assert summary[1]["top_speed_mps"] == 8.0

    def test_team_summary(self, config):
        stats = PhysicalStats(config)
        for i in range(3):
            players = [
                _make_player(1, 0, 50.0 + i, 34.0, speed=4.0),
                _make_player(2, 1, 55.0 + i, 34.0, speed=5.0),
            ]
            stats.update(_make_frame(i, float(i), players=players))

        team_summary = stats.get_team_summary()
        assert 0 in team_summary
        assert 1 in team_summary


class TestSpatialStats:
    def test_centroid_and_shape(self, config):
        stats = SpatialStats(config)
        players = [
            _make_player(1, 0, 40.0, 20.0),
            _make_player(2, 0, 60.0, 48.0),
            _make_player(3, 1, 70.0, 25.0),
            _make_player(4, 1, 80.0, 45.0),
        ]
        fs = _make_frame(0, 0.0, players=players)
        stats.update(fs)

        assert "spatial" in fs.analytics
        spatial = fs.analytics["spatial"]
        assert "team_0" in spatial
        assert "centroid" in spatial["team_0"]

    def test_team_summary(self, config):
        stats = SpatialStats(config)
        for i in range(5):
            players = [
                _make_player(1, 0, 40.0 + i, 20.0),
                _make_player(2, 0, 60.0 + i, 48.0),
            ]
            stats.update(_make_frame(i, float(i), players=players))
        summary = stats.get_team_summary()
        assert 0 in summary


class TestBallProgressionStats:
    def test_third_classification(self, config):
        stats = BallProgressionStats(config)
        # Ball in defensive third
        fs = _make_frame(0, 0.0, ball_pos=(10.0, 34.0))
        stats.update(fs)
        assert fs.analytics["ball_progression"]["third"] == "defensive"

        # Ball in attacking third
        fs2 = _make_frame(1, 1.0, ball_pos=(80.0, 34.0))
        stats.update(fs2)
        assert fs2.analytics["ball_progression"]["third"] == "attacking"

    def test_possession_proxy(self, config):
        stats = BallProgressionStats(config)
        players = [
            _make_player(1, 0, 50.0, 34.0),
            _make_player(2, 1, 70.0, 34.0),
        ]
        fs = _make_frame(0, 0.0, players=players, ball_pos=(52.0, 34.0))
        stats.update(fs)
        assert fs.analytics["ball_progression"]["nearest_team"] == 0


class TestPressureStats:
    def test_pressure_index(self, config):
        stats = PressureStats(config)
        players = [
            _make_player(1, 0, 50.0, 34.0),  # ball carrier
            _make_player(2, 1, 52.0, 34.0),   # pressing opponent (2m away)
            _make_player(3, 1, 53.0, 34.0),   # another opponent (3m away)
        ]
        fs = _make_frame(0, 0.0, players=players, ball_pos=(50.0, 34.0))
        stats.update(fs)
        assert "pressure" in fs.analytics
        assert fs.analytics["pressure"]["pressure_index"] > 0

    def test_pitch_control(self, config):
        stats = PressureStats(config)
        players = [
            _make_player(1, 0, 30.0, 34.0),
            _make_player(2, 0, 40.0, 34.0),
            _make_player(3, 1, 70.0, 34.0),
            _make_player(4, 1, 80.0, 34.0),
        ]
        fs = _make_frame(0, 0.0, players=players, ball_pos=(50.0, 34.0))
        stats.update(fs)
        control = fs.analytics["pressure"]["pitch_control"]
        assert "0" in control
        assert "1" in control
        total = sum(float(v) for v in control.values())
        assert total == pytest.approx(1.0, abs=0.01)


class TestThreatStats:
    def test_xt_values(self, config):
        stats = ThreatStats(config)
        players = [
            _make_player(1, 0, 90.0, 34.0),  # high threat zone
            _make_player(2, 1, 10.0, 34.0),   # low threat zone
        ]
        fs = _make_frame(0, 0.0, players=players, ball_pos=(90.0, 34.0))
        stats.update(fs)
        assert "threat" in fs.analytics
        assert fs.analytics["threat"]["ball_xt"] > 0.1

    def test_team_summary(self, config):
        stats = ThreatStats(config)
        for i in range(5):
            players = [_make_player(1, 0, 80.0, 34.0)]
            stats.update(_make_frame(i, float(i), players=players))
        summary = stats.get_team_summary()
        assert 0 in summary
        assert summary[0]["avg_weighted_threat"] > 0


class TestStatsAggregator:
    def test_full_update(self, config):
        agg = StatsAggregator(config)
        players = [
            _make_player(1, 0, 50.0, 34.0),
            _make_player(2, 1, 60.0, 34.0),
        ]
        fs = _make_frame(0, 0.0, players=players, ball_pos=(55.0, 34.0))
        agg.update(fs)

        assert "physical" in fs.analytics
        assert "spatial" in fs.analytics
        assert "ball_progression" in fs.analytics
        assert "pressure" in fs.analytics
        assert "threat" in fs.analytics

    def test_full_report(self, config):
        agg = StatsAggregator(config)
        for i in range(10):
            players = [
                _make_player(1, 0, 50.0 + i, 34.0, speed=4.0),
                _make_player(2, 1, 60.0 + i, 34.0, speed=5.0),
            ]
            agg.update(_make_frame(i, float(i), players=players, ball_pos=(55.0 + i, 34.0)))

        report = agg.get_full_report()
        assert "player_summary" in report
        assert "team_summary" in report
        assert "rolling_summary" in report
