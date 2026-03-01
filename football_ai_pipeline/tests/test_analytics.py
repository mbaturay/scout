"""Tests for analytics: association, events, xG."""

from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.association import BallOwnerAssigner, OwnerRecord
from src.analytics.events import EventDetector, MatchEvent
from src.analytics.metrics import compute_xg, _goal_angle

# Default config for tests
_CFG: dict = {
    "analytics": {
        "pitch_threshold_m": 5.0,
        "pixel_threshold_px": 80.0,
        "hysteresis_frames": 3,
        "pass_min_distance_m": 3.0,
        "pass_max_gap_sec": 3.0,
        "inflight_min_frames": 2,
        "left_to_right": True,
        "shot_min_speed_mps": 5.0,
    },
    "pitch": {"length": 105.0, "width": 68.0},
    "video": {"target_fps": 30.0},
}


# =========================================================================
# Association tests
# =========================================================================

class TestBallOwnerAssigner:
    def _make_frame(self, frame_idx, players, ball):
        return {"frame_idx": frame_idx, "players": players, "ball": ball}

    def test_first_owner_assigned_immediately(self):
        assigner = BallOwnerAssigner(_CFG)
        rec = assigner.update(self._make_frame(
            0,
            [{"track_id": 1, "team_id": 0, "x": 10.0, "y": 10.0}],
            {"x": 11.0, "y": 10.0, "is_pitch": True},
        ))
        assert rec.owner_player_id == 1
        assert rec.owner_team_id == 0
        assert rec.ball_available is True

    def test_no_ball_returns_none_owner(self):
        assigner = BallOwnerAssigner(_CFG)
        rec = assigner.update(self._make_frame(
            0,
            [{"track_id": 1, "team_id": 0, "x": 10.0, "y": 10.0}],
            None,
        ))
        assert rec.owner_player_id is None
        assert rec.ball_available is False

    def test_hysteresis_prevents_instant_switch(self):
        """Owner should not switch until hysteresis_frames consecutive frames."""
        assigner = BallOwnerAssigner(_CFG)

        # Establish player 1 as owner
        assigner.update(self._make_frame(
            0,
            [
                {"track_id": 1, "team_id": 0, "x": 10.0, "y": 10.0},
                {"track_id": 2, "team_id": 1, "x": 50.0, "y": 10.0},
            ],
            {"x": 11.0, "y": 10.0, "is_pitch": True},
        ))

        # Ball moves near player 2 — should NOT switch on frame 1
        rec = assigner.update(self._make_frame(
            1,
            [
                {"track_id": 1, "team_id": 0, "x": 10.0, "y": 10.0},
                {"track_id": 2, "team_id": 1, "x": 50.0, "y": 10.0},
            ],
            {"x": 51.0, "y": 10.0, "is_pitch": True},
        ))
        # Still player 1 due to hysteresis
        assert rec.owner_player_id == 1

        # Continue near player 2 for hysteresis_frames (3 total)
        for i in range(2, 5):
            rec = assigner.update(self._make_frame(
                i,
                [
                    {"track_id": 1, "team_id": 0, "x": 10.0, "y": 10.0},
                    {"track_id": 2, "team_id": 1, "x": 50.0, "y": 10.0},
                ],
                {"x": 51.0, "y": 10.0, "is_pitch": True},
            ))

        # After 3+ consecutive frames near player 2, should have switched
        assert rec.owner_player_id == 2
        assert rec.owner_team_id == 1

    def test_ball_too_far_no_owner(self):
        assigner = BallOwnerAssigner(_CFG)
        rec = assigner.update(self._make_frame(
            0,
            [{"track_id": 1, "team_id": 0, "x": 10.0, "y": 10.0}],
            {"x": 100.0, "y": 100.0, "is_pitch": True},  # far away
        ))
        # No current owner, so remains None
        assert rec.owner_player_id is None
        assert rec.owner_confidence == 0.0

    def test_pixel_threshold_used_without_pitch(self):
        assigner = BallOwnerAssigner(_CFG)
        rec = assigner.update(self._make_frame(
            0,
            [{"track_id": 1, "team_id": 0, "x": 100.0, "y": 200.0}],
            {"x": 150.0, "y": 200.0, "is_pitch": False},  # 50px away, under 80px threshold
        ))
        assert rec.owner_player_id == 1


# =========================================================================
# Event detection tests
# =========================================================================

class TestEventDetector:
    def _make_ownership(self, sequence):
        """Create OwnerRecord list from [(player_id, team_id, confidence)]."""
        records = []
        for i, (pid, tid, conf) in enumerate(sequence):
            records.append(OwnerRecord(
                frame_idx=i,
                owner_player_id=pid,
                owner_team_id=tid,
                owner_confidence=conf,
                ball_available=True,
            ))
        return records

    def test_pass_same_team(self):
        """A->B same team with sufficient ball travel = pass."""
        detector = EventDetector(_CFG)
        detector.set_fps(30.0)

        # Player 1 (team 0) owns for 10 frames, then player 2 (team 0) owns
        seq = [(1, 0, 0.8)] * 10 + [(2, 0, 0.8)] * 10
        ownership = self._make_ownership(seq)

        # Ball positions: starts at (10, 10), ends at (20, 10) — 10m travel
        positions = [(10.0, 10.0)] * 10 + [(20.0, 10.0)] * 10
        speeds = [0.0] * 20
        player_pos = [{} for _ in range(20)]

        events = detector.detect(ownership, positions, speeds, player_pos)
        pass_events = [e for e in events if e.event_type == "pass"]
        assert len(pass_events) >= 1
        assert pass_events[0].team_id == 0
        assert pass_events[0].player_id == 1
        assert pass_events[0].target_player_id == 2

        # Should also have a reception event
        reception_events = [e for e in events if e.event_type == "reception"]
        assert len(reception_events) >= 1

    def test_interception_cross_team_with_inflight(self):
        """A(team0) -> gap -> B(team1) with inflight frames = interception."""
        detector = EventDetector(_CFG)
        detector.set_fps(30.0)

        # Player 1 team 0 for 5 frames, 3 no-owner frames, player 3 team 1 for 5 frames
        seq = (
            [(1, 0, 0.8)] * 5
            + [(None, None, 0.0)] * 3
            + [(3, 1, 0.8)] * 5
        )
        ownership = self._make_ownership(seq)
        positions = [(10.0, 10.0)] * 5 + [(None, None)] * 3 + [(30.0, 10.0)] * 5
        # Fix: positions with None should be actual None
        positions = [(10.0, 10.0)] * 5 + [None] * 3 + [(30.0, 10.0)] * 5
        speeds = [0.0] * 13
        player_pos = [{} for _ in range(13)]

        events = detector.detect(ownership, positions, speeds, player_pos)
        interceptions = [e for e in events if e.event_type == "interception"]
        assert len(interceptions) >= 1
        assert interceptions[0].team_id == 1
        assert interceptions[0].player_id == 3

    def test_tackle_cross_team_no_inflight(self):
        """A(team0) -> B(team1) directly (no gap) = tackle."""
        detector = EventDetector(_CFG)
        detector.set_fps(30.0)

        # Direct transition: player 1 team 0 -> player 3 team 1
        seq = [(1, 0, 0.8)] * 10 + [(3, 1, 0.8)] * 10
        ownership = self._make_ownership(seq)
        positions = [(10.0, 10.0)] * 10 + [(12.0, 10.0)] * 10
        speeds = [0.0] * 20
        player_pos = [{} for _ in range(20)]

        events = detector.detect(ownership, positions, speeds, player_pos)
        tackles = [e for e in events if e.event_type == "tackle"]
        assert len(tackles) >= 1
        assert tackles[0].team_id == 1


# =========================================================================
# xG tests
# =========================================================================

class TestXG:
    def test_xg_in_range(self):
        """xG should always be in [0, 1]."""
        for dist in [1, 5, 10, 20, 30, 50]:
            for angle in [0.1, 0.3, 0.6, 1.0]:
                xg = compute_xg(dist, angle)
                assert 0.0 <= xg <= 1.0, f"xG={xg} out of range for d={dist}, a={angle}"

    def test_xg_closer_higher(self):
        """xG should be higher when closer to goal."""
        angle = 0.5
        xg_close = compute_xg(5.0, angle)
        xg_far = compute_xg(30.0, angle)
        assert xg_close > xg_far

    def test_xg_wider_angle_higher(self):
        """xG should be higher with wider angle to goal."""
        dist = 15.0
        xg_wide = compute_xg(dist, 0.8)
        xg_narrow = compute_xg(dist, 0.1)
        assert xg_wide > xg_narrow

    def test_xg_penalty_spot(self):
        """xG from penalty spot should be reasonable (0.5-0.9)."""
        # Penalty spot: 11m from goal, ~0.4 rad angle
        angle = _goal_angle(94.0, 34.0, 105.0, 68.0)
        xg = compute_xg(11.0, angle)
        assert 0.3 <= xg <= 0.95, f"Penalty xG={xg} seems unreasonable"

    def test_goal_angle_reasonable(self):
        """Goal angle from centre should be positive and < pi."""
        angle = _goal_angle(52.5, 34.0, 105.0, 68.0)
        assert 0 < angle < math.pi
