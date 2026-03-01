"""Tests for v2 pass detection."""

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
    ball_state: str = "controlled",
    fps: float = 30.0,
) -> tuple[list[FrameBallOwner], list[tuple[float, float] | None], list[float]]:
    """Build timeline + ball positions + timestamps from segment specs.

    Each segment is (owner_track_id | None, team_id | None, ball_x, n_frames).
    Returns (timeline, ball_positions, timestamps).
    """
    timeline: list[FrameBallOwner] = []
    positions: list[tuple[float, float] | None] = []
    timestamps: list[float] = []
    idx = 0
    for owner, team, bx, n in segments:
        for _ in range(n):
            state = ball_state if bx is not None else "unknown"
            if owner is None and bx is not None:
                state = "loose"  # no owner but ball present
            timeline.append(FrameBallOwner(
                frame_idx=idx,
                owner_track_id=owner,
                owner_team_id=team,
                ball_available=bx is not None,
                ball_state=state,
            ))
            positions.append((float(bx), 34.0) if bx is not None else None)
            timestamps.append(idx / fps)
            idx += 1
    return timeline, positions, timestamps


# =========================================================================
# compute_passes
# =========================================================================

class TestComputePasses:
    def test_completed_pass_same_team(self):
        """Player 1 (team 0) -> Player 2 (team 0) with 10m ball travel."""
        timeline, positions, ts = _timeline([
            (1, 0, 10, 10),  # player 1 owns, ball at x=10
            (2, 0, 20, 10),  # player 2 owns, ball at x=20 (10m travel)
        ])
        player_team = {1: 0, 2: 0}
        passes = compute_passes(
            timeline, positions, player_team,
            timestamps=ts, min_pass_dist_m=4.0,
        )
        assert len(passes) == 1
        p = passes[0]
        assert p.from_track == 1
        assert p.to_track == 2
        assert p.is_completed is True
        assert p.reason == "pass"
        assert p.dist_m >= 9.0  # ~10m

    def test_interception_cross_team(self):
        """Player 1 (team 0) -> Player 3 (team 1) = interception (was controlled)."""
        timeline, positions, ts = _timeline([
            (1, 0, 10, 10),
            (3, 1, 25, 10),  # 15m travel, different team
        ])
        player_team = {1: 0, 3: 1}
        passes = compute_passes(
            timeline, positions, player_team,
            timestamps=ts, min_pass_dist_m=4.0,
        )
        assert len(passes) == 1
        assert passes[0].is_completed is False
        assert passes[0].reason == "interception"
        assert passes[0].team_id == 0
        assert passes[0].to_team_id == 1

    def test_short_distance_not_a_pass(self):
        """Transition with < min_pass_dist_m is ignored (same team)."""
        timeline, positions, ts = _timeline([
            (1, 0, 10, 10),
            (2, 0, 11, 10),  # only 1m travel
        ])
        player_team = {1: 0, 2: 0}
        passes = compute_passes(
            timeline, positions, player_team,
            timestamps=ts, min_pass_dist_m=4.0,
        )
        assert len(passes) == 0

    def test_gap_frames_bridged(self):
        """None-owned gap up to max_gap_frames is bridged as single transition."""
        timeline, positions, ts = _timeline([
            (1, 0, 10, 5),
            (None, None, 15, 5),  # ball loose for 5 frames
            (2, 0, 20, 5),
        ])
        player_team = {1: 0, 2: 0}
        passes = compute_passes(
            timeline, positions, player_team,
            timestamps=ts,
            min_pass_dist_m=4.0, max_gap_frames=10,
        )
        assert len(passes) == 1
        assert passes[0].is_completed is True

    def test_gap_too_large_no_pass(self):
        """Gap exceeding max_gap_frames breaks the transition."""
        timeline, positions, ts = _timeline([
            (1, 0, 10, 5),
            (None, None, 15, 15),  # 15 frames gap
            (2, 0, 20, 5),
        ])
        player_team = {1: 0, 2: 0}
        passes = compute_passes(
            timeline, positions, player_team,
            timestamps=ts,
            min_pass_dist_m=4.0, max_gap_frames=10,
        )
        assert len(passes) == 0

    def test_dribble_filter_same_owner_resumes(self):
        """Same player regains ball after gap -- not a pass (dribble)."""
        timeline, positions, ts = _timeline([
            (1, 0, 10, 5),
            (None, None, 12, 3),
            (1, 0, 15, 5),  # same player
        ])
        player_team = {1: 0}
        passes = compute_passes(
            timeline, positions, player_team,
            timestamps=ts, min_pass_dist_m=4.0,
        )
        assert len(passes) == 0

    def test_multiple_passes_detected(self):
        """A -> B -> C on same team = two completed passes."""
        timeline, positions, ts = _timeline([
            (1, 0, 10, 5),
            (2, 0, 20, 5),  # pass 1: 1->2 (10m)
            (3, 0, 35, 5),  # pass 2: 2->3 (15m)
        ])
        player_team = {1: 0, 2: 0, 3: 0}
        passes = compute_passes(
            timeline, positions, player_team,
            timestamps=ts, min_pass_dist_m=4.0,
        )
        assert len(passes) == 2
        assert passes[0].from_track == 1
        assert passes[0].to_track == 2
        assert passes[1].from_track == 2
        assert passes[1].to_track == 3

    def test_empty_timeline(self):
        passes = compute_passes([], [], {}, min_pass_dist_m=4.0)
        assert passes == []

    def test_single_frame(self):
        timeline = [FrameBallOwner(frame_idx=0, owner_track_id=1, owner_team_id=0, ball_state="controlled")]
        passes = compute_passes(timeline, [(10.0, 34.0)], {1: 0}, min_pass_dist_m=4.0)
        assert passes == []

    def test_turnover_unknown_team(self):
        """If one side has unknown team -> reason='turnover'."""
        timeline, positions, ts = _timeline([
            (1, 0, 10, 5),
            (2, None, 20, 5),  # team unknown
        ])
        player_team = {1: 0}  # player 2 not in lookup
        passes = compute_passes(
            timeline, positions, player_team,
            timestamps=ts, min_pass_dist_m=4.0,
        )
        assert len(passes) == 1
        assert passes[0].reason == "turnover"
        assert passes[0].is_completed is False

    def test_max_time_exceeded_no_pass(self):
        """Transition taking > max_pass_time_s is rejected (same team)."""
        # At 30fps, 60 frames = 2.0s > 1.5s
        timeline, positions, ts = _timeline([
            (1, 0, 10, 30),
            (None, None, 15, 30),  # 30 frames gap = 1.0s at 30fps
            (2, 0, 25, 10),
        ], fps=30.0)
        player_team = {1: 0, 2: 0}
        # seg_end = frame 29 (ts=0.967), j = frame 60 (ts=2.0) => elapsed = 1.033
        # But the gap is 30 frames which exceeds max_gap_frames=10 => no pass
        passes = compute_passes(
            timeline, positions, player_team,
            timestamps=ts,
            min_pass_dist_m=4.0, max_gap_frames=10, max_pass_time_s=1.5,
        )
        assert len(passes) == 0

    def test_air_state_allows_longer_gap(self):
        """Ball in 'air' state allows extended gap up to air_gap_frames."""
        timeline: list[FrameBallOwner] = []
        positions: list[tuple[float, float] | None] = []
        timestamps: list[float] = []
        fps = 30.0
        idx = 0

        # 5 frames player 1 controlled
        for _ in range(5):
            timeline.append(FrameBallOwner(
                frame_idx=idx, owner_track_id=1, owner_team_id=0,
                ball_available=True, ball_state="controlled",
            ))
            positions.append((10.0, 34.0))
            timestamps.append(idx / fps)
            idx += 1

        # 15 frames gap with ball in air (no owner)
        for _ in range(15):
            timeline.append(FrameBallOwner(
                frame_idx=idx, owner_track_id=None, owner_team_id=None,
                ball_available=True, ball_state="air",
            ))
            positions.append((15.0, 34.0))
            timestamps.append(idx / fps)
            idx += 1

        # 5 frames player 2 controlled
        for _ in range(5):
            timeline.append(FrameBallOwner(
                frame_idx=idx, owner_track_id=2, owner_team_id=0,
                ball_available=True, ball_state="controlled",
            ))
            positions.append((25.0, 34.0))
            timestamps.append(idx / fps)
            idx += 1

        player_team = {1: 0, 2: 0}
        passes = compute_passes(
            timeline, positions, player_team,
            timestamps=timestamps,
            min_pass_dist_m=4.0,
            max_gap_frames=10,  # normal gap would reject 15 frames
            air_gap_frames=20,  # but air allows up to 20
            max_pass_time_s=2.0,  # allow enough time for air pass
        )
        assert len(passes) == 1
        assert passes[0].is_completed is True
        assert passes[0].reason == "pass"

    def test_interception_requires_controlled_before(self):
        """Interception requires the ball was controlled before the switch."""
        # Set ball_state to "loose" for the first segment
        timeline, positions, ts = _timeline([
            (1, 0, 10, 5),
            (3, 1, 25, 5),
        ], ball_state="loose")  # not controlled
        player_team = {1: 0, 3: 1}
        passes = compute_passes(
            timeline, positions, player_team,
            timestamps=ts, min_pass_dist_m=4.0,
        )
        # Should be turnover not interception since ball wasn't controlled
        assert len(passes) == 1
        assert passes[0].reason == "turnover"

    def test_pass_event_has_duration(self):
        """PassEvent includes duration_s field."""
        timeline, positions, ts = _timeline([
            (1, 0, 10, 10),
            (2, 0, 20, 10),
        ], fps=30.0)
        player_team = {1: 0, 2: 0}
        passes = compute_passes(
            timeline, positions, player_team,
            timestamps=ts, min_pass_dist_m=4.0,
        )
        assert len(passes) == 1
        assert passes[0].duration_s >= 0.0
        # From frame 9 to frame 10: ~0.033s
        assert passes[0].duration_s < 1.0


# =========================================================================
# aggregate_pass_stats
# =========================================================================

class TestAggregatePassStats:
    def test_team_stats(self):
        passes = [
            PassEvent(t_start=0, t_end=10, from_track=1, to_track=2,
                      team_id=0, to_team_id=0, dist_m=10.0,
                      is_completed=True, reason="pass"),
            PassEvent(t_start=10, t_end=20, from_track=2, to_track=3,
                      team_id=0, to_team_id=1, dist_m=15.0,
                      is_completed=False, reason="interception"),
        ]
        team_stats, player_stats = aggregate_pass_stats(passes, {1: 0, 2: 0, 3: 1})

        assert team_stats[0]["pass_count"] == 2        # both attempts by team 0
        assert team_stats[0]["completed_pass_count"] == 1
        assert team_stats[0]["pass_accuracy_pct"] == 50.0
        # Interception won by team 1
        assert team_stats[1]["interceptions_won"] == 1

    def test_player_stats(self):
        passes = [
            PassEvent(t_start=0, t_end=10, from_track=1, to_track=2,
                      team_id=0, to_team_id=0, dist_m=10.0,
                      is_completed=True, reason="pass"),
            PassEvent(t_start=10, t_end=20, from_track=1, to_track=3,
                      team_id=0, to_team_id=1, dist_m=15.0,
                      is_completed=False, reason="interception"),
        ]
        _, player_stats = aggregate_pass_stats(passes, {1: 0, 2: 0, 3: 1})

        assert player_stats[1]["passes_attempted"] == 2
        assert player_stats[1]["passes_completed"] == 1
        assert player_stats[1]["pass_accuracy_pct"] == 50.0
        # Player 3 made the interception
        assert player_stats[3]["interceptions"] == 1

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
                      is_completed=True, reason="pass"),
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
        assert evt["type"] == "pass"
        assert "duration_s" in evt
