"""Tests for v1 pass detection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.passes import (
    PassEvent,
    aggregate_pass_stats,
    compute_passes,
    write_pass_events,
)
from src.analytics.possession import FrameBallOwner


# =========================================================================
# Helpers
# =========================================================================

def _timeline(
    segments: list[tuple[int, int | None, int | None, int]],
) -> tuple[list[FrameBallOwner], list[tuple[float, float] | None]]:
    """Build timeline + ball positions from segment specs.

    Each segment is (owner_track_id | None, team_id | None, ball_x, n_frames).
    Returns (timeline, ball_positions).
    """
    timeline: list[FrameBallOwner] = []
    positions: list[tuple[float, float] | None] = []
    idx = 0
    for owner, team, bx, n in segments:
        for _ in range(n):
            timeline.append(FrameBallOwner(
                frame_idx=idx,
                owner_track_id=owner,
                owner_team_id=team,
                ball_available=True,
            ))
            positions.append((float(bx), 34.0) if bx is not None else None)
            idx += 1
    return timeline, positions


# =========================================================================
# compute_passes
# =========================================================================

class TestComputePasses:
    def test_completed_pass_same_team(self):
        """Player 1 (team 0) → Player 2 (team 0) with 10m ball travel."""
        timeline, positions = _timeline([
            (1, 0, 10, 10),  # player 1 owns, ball at x=10
            (2, 0, 20, 10),  # player 2 owns, ball at x=20 (10m travel)
        ])
        player_team = {1: 0, 2: 0}
        passes = compute_passes(timeline, positions, player_team, min_pass_dist_m=3.0)
        assert len(passes) == 1
        p = passes[0]
        assert p.from_track == 1
        assert p.to_track == 2
        assert p.is_completed is True
        assert p.reason == "completed_pass"
        assert p.dist_m >= 9.0  # ~10m

    def test_interception_cross_team(self):
        """Player 1 (team 0) → Player 3 (team 1) = interception."""
        timeline, positions = _timeline([
            (1, 0, 10, 10),
            (3, 1, 25, 10),  # 15m travel, different team
        ])
        player_team = {1: 0, 3: 1}
        passes = compute_passes(timeline, positions, player_team, min_pass_dist_m=3.0)
        assert len(passes) == 1
        assert passes[0].is_completed is False
        assert passes[0].reason == "interception"
        assert passes[0].team_id == 0
        assert passes[0].to_team_id == 1

    def test_short_distance_not_a_pass(self):
        """Transition with < min_pass_dist_m is ignored."""
        timeline, positions = _timeline([
            (1, 0, 10, 10),
            (2, 0, 11, 10),  # only 1m travel
        ])
        player_team = {1: 0, 2: 0}
        passes = compute_passes(timeline, positions, player_team, min_pass_dist_m=3.0)
        assert len(passes) == 0

    def test_gap_frames_bridged(self):
        """None-owned gap up to max_gap_frames is bridged as single transition."""
        timeline, positions = _timeline([
            (1, 0, 10, 5),
            (None, None, 15, 5),  # ball in air for 5 frames
            (2, 0, 20, 5),
        ])
        player_team = {1: 0, 2: 0}
        passes = compute_passes(
            timeline, positions, player_team,
            min_pass_dist_m=3.0, max_gap_frames=10,
        )
        assert len(passes) == 1
        assert passes[0].is_completed is True

    def test_gap_too_large_no_pass(self):
        """Gap exceeding max_gap_frames breaks the transition."""
        timeline, positions = _timeline([
            (1, 0, 10, 5),
            (None, None, 15, 15),  # 15 frames gap
            (2, 0, 20, 5),
        ])
        player_team = {1: 0, 2: 0}
        passes = compute_passes(
            timeline, positions, player_team,
            min_pass_dist_m=3.0, max_gap_frames=10,
        )
        assert len(passes) == 0

    def test_same_owner_resumes_not_a_pass(self):
        """Same player regains ball after gap — not a pass."""
        timeline, positions = _timeline([
            (1, 0, 10, 5),
            (None, None, 12, 3),
            (1, 0, 15, 5),  # same player
        ])
        player_team = {1: 0}
        passes = compute_passes(timeline, positions, player_team, min_pass_dist_m=3.0)
        assert len(passes) == 0

    def test_multiple_passes_detected(self):
        """A → B → C on same team = two completed passes."""
        timeline, positions = _timeline([
            (1, 0, 10, 5),
            (2, 0, 20, 5),  # pass 1: 1→2
            (3, 0, 35, 5),  # pass 2: 2→3
        ])
        player_team = {1: 0, 2: 0, 3: 0}
        passes = compute_passes(timeline, positions, player_team, min_pass_dist_m=3.0)
        assert len(passes) == 2
        assert passes[0].from_track == 1
        assert passes[0].to_track == 2
        assert passes[1].from_track == 2
        assert passes[1].to_track == 3

    def test_empty_timeline(self):
        passes = compute_passes([], [], {}, min_pass_dist_m=3.0)
        assert passes == []

    def test_single_frame(self):
        timeline = [FrameBallOwner(frame_idx=0, owner_track_id=1, owner_team_id=0)]
        passes = compute_passes(timeline, [(10.0, 34.0)], {1: 0}, min_pass_dist_m=3.0)
        assert passes == []

    def test_turnover_unknown_team(self):
        """If one side has unknown team → reason='turnover'."""
        timeline, positions = _timeline([
            (1, 0, 10, 5),
            (2, None, 20, 5),  # team unknown
        ])
        player_team = {1: 0}  # player 2 not in lookup
        passes = compute_passes(timeline, positions, player_team, min_pass_dist_m=3.0)
        assert len(passes) == 1
        assert passes[0].reason == "turnover"
        assert passes[0].is_completed is False


# =========================================================================
# aggregate_pass_stats
# =========================================================================

class TestAggregatePassStats:
    def test_team_stats(self):
        passes = [
            PassEvent(t_start=0, t_end=10, from_track=1, to_track=2,
                      team_id=0, to_team_id=0, dist_m=10.0,
                      is_completed=True, reason="completed_pass"),
            PassEvent(t_start=10, t_end=20, from_track=2, to_track=3,
                      team_id=0, to_team_id=1, dist_m=15.0,
                      is_completed=False, reason="interception"),
        ]
        team_stats, player_stats = aggregate_pass_stats(passes, {1: 0, 2: 0, 3: 1})

        assert team_stats[0]["pass_count"] == 2        # both attempts by team 0
        assert team_stats[0]["completed_pass_count"] == 1
        assert team_stats[0]["pass_accuracy_pct"] == 50.0

    def test_player_stats(self):
        passes = [
            PassEvent(t_start=0, t_end=10, from_track=1, to_track=2,
                      team_id=0, to_team_id=0, dist_m=10.0,
                      is_completed=True, reason="completed_pass"),
            PassEvent(t_start=10, t_end=20, from_track=1, to_track=3,
                      team_id=0, to_team_id=1, dist_m=15.0,
                      is_completed=False, reason="interception"),
        ]
        _, player_stats = aggregate_pass_stats(passes, {1: 0, 2: 0, 3: 1})

        assert player_stats[1]["passes_attempted"] == 2
        assert player_stats[1]["passes_completed"] == 1
        assert player_stats[1]["pass_accuracy_pct"] == 50.0

    def test_empty_passes(self):
        team_stats, player_stats = aggregate_pass_stats([], {})
        assert team_stats == {}
        assert player_stats == {}


# =========================================================================
# Output writers
# =========================================================================

class TestPassWriters:
    def test_write_pass_events(self, tmp_path):
        passes = [
            PassEvent(t_start=0, t_end=10, from_track=1, to_track=2,
                      team_id=0, to_team_id=0, dist_m=10.5,
                      is_completed=True, reason="completed_pass"),
        ]
        path = tmp_path / "pass_events.json"
        write_pass_events(path, passes)
        data = json.loads(path.read_text())
        assert "pass_events" in data
        assert len(data["pass_events"]) == 1
        evt = data["pass_events"][0]
        assert evt["from_track"] == 1
        assert evt["to_track"] == 2
        assert evt["is_completed"] is True
        assert evt["dist_m"] == 10.5
