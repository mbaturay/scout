"""Tests for possession v1: ball-to-nearest-player ownership."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.possession import (
    FrameBallOwner,
    PossessionResult,
    assign_ball_owner,
    build_player_team_lookup,
    compute_possession,
    extract_tracks_from_frame,
    write_all_outputs,
    write_player_touches,
    write_team_possession,
)


# =========================================================================
# Helpers
# =========================================================================

def _make_frame(
    frame_idx: int,
    ball_xy: tuple[float, float] | None = None,
    players: list[dict] | None = None,
    flag: str = "in_play",
) -> dict:
    """Build a minimal serialised frame dict."""
    frame: dict = {"frame_idx": frame_idx, "flag": flag, "players": []}
    if ball_xy is not None:
        frame["ball"] = {"pitch_x": ball_xy[0], "pitch_y": ball_xy[1]}
    else:
        frame["ball"] = None
    for p in (players or []):
        frame["players"].append({
            "track_id": p["track_id"],
            "team_id": p.get("team_id"),
            "pitch_x": p.get("x"),
            "pitch_y": p.get("y"),
            "class": p.get("class", "player"),
        })
    return frame


# =========================================================================
# extract_tracks_from_frame
# =========================================================================

class TestExtractTracks:
    def test_basic_extraction(self):
        frame = _make_frame(
            0,
            ball_xy=(50.0, 34.0),
            players=[
                {"track_id": 1, "team_id": 0, "x": 48.0, "y": 34.0},
                {"track_id": 2, "team_id": 1, "x": 55.0, "y": 34.0},
            ],
        )
        ball, players = extract_tracks_from_frame(frame)
        assert ball == (50.0, 34.0)
        assert len(players) == 2
        assert players[0]["track_id"] == 1

    def test_no_ball(self):
        frame = _make_frame(0, ball_xy=None, players=[
            {"track_id": 1, "team_id": 0, "x": 10.0, "y": 10.0},
        ])
        ball, players = extract_tracks_from_frame(frame)
        assert ball is None
        assert len(players) == 1

    def test_ball_class_excluded_from_players(self):
        frame = _make_frame(0, ball_xy=(50.0, 34.0), players=[
            {"track_id": 1, "team_id": 0, "x": 48.0, "y": 34.0},
            {"track_id": 99, "team_id": None, "x": 50.0, "y": 34.0, "class": "ball"},
        ])
        _, players = extract_tracks_from_frame(frame)
        assert len(players) == 1
        assert players[0]["track_id"] == 1

    def test_player_without_pitch_coords_excluded(self):
        frame = {"frame_idx": 0, "ball": {"pitch_x": 50, "pitch_y": 34}, "players": [
            {"track_id": 1, "team_id": 0, "pitch_x": None, "pitch_y": None, "class": "player"},
            {"track_id": 2, "team_id": 1, "pitch_x": 55, "pitch_y": 34, "class": "player"},
        ]}
        _, players = extract_tracks_from_frame(frame)
        assert len(players) == 1
        assert players[0]["track_id"] == 2


# =========================================================================
# assign_ball_owner
# =========================================================================

class TestAssignBallOwner:
    def test_nearest_player_within_threshold(self):
        players = [
            {"track_id": 1, "team_id": 0, "x": 50.5, "y": 34.0},
            {"track_id": 2, "team_id": 1, "x": 55.0, "y": 34.0},
        ]
        tid, team, dist = assign_ball_owner((50.0, 34.0), players, max_dist_m=1.25)
        assert tid == 1
        assert team == 0
        assert dist is not None
        assert dist < 1.25

    def test_no_player_within_threshold(self):
        players = [
            {"track_id": 1, "team_id": 0, "x": 55.0, "y": 34.0},
        ]
        tid, team, dist = assign_ball_owner((50.0, 34.0), players, max_dist_m=1.25)
        assert tid is None
        assert team is None
        assert dist is None

    def test_empty_players(self):
        tid, team, dist = assign_ball_owner((50.0, 34.0), [], max_dist_m=1.25)
        assert tid is None

    def test_exact_threshold_boundary(self):
        players = [{"track_id": 1, "team_id": 0, "x": 51.25, "y": 34.0}]
        tid, _, _ = assign_ball_owner((50.0, 34.0), players, max_dist_m=1.25)
        assert tid == 1  # exactly at threshold

    def test_just_outside_threshold(self):
        players = [{"track_id": 1, "team_id": 0, "x": 51.26, "y": 34.0}]
        tid, _, _ = assign_ball_owner((50.0, 34.0), players, max_dist_m=1.25)
        assert tid is None


# =========================================================================
# build_player_team_lookup
# =========================================================================

class TestBuildPlayerTeamLookup:
    def test_majority_vote(self):
        frames = [
            _make_frame(0, players=[{"track_id": 1, "team_id": 0, "x": 10, "y": 10}]),
            _make_frame(1, players=[{"track_id": 1, "team_id": 0, "x": 10, "y": 10}]),
            _make_frame(2, players=[{"track_id": 1, "team_id": 1, "x": 10, "y": 10}]),  # minority
        ]
        lookup = build_player_team_lookup(frames)
        assert lookup[1] == 0  # team 0 appeared 2/3 times

    def test_null_team_id_ignored(self):
        frames = [
            _make_frame(0, players=[{"track_id": 1, "team_id": None, "x": 10, "y": 10}]),
            _make_frame(1, players=[{"track_id": 1, "team_id": 0, "x": 10, "y": 10}]),
        ]
        lookup = build_player_team_lookup(frames)
        assert lookup[1] == 0


# =========================================================================
# compute_possession
# =========================================================================

class TestComputePossession:
    def test_single_owner_100_pct(self):
        """One player near ball for all frames → 100% possession for their team."""
        frames = [
            _make_frame(i, ball_xy=(50.0, 34.0), players=[
                {"track_id": 1, "team_id": 0, "x": 50.5, "y": 34.0},
                {"track_id": 2, "team_id": 1, "x": 80.0, "y": 34.0},
            ])
            for i in range(10)
        ]
        result = compute_possession(frames, max_dist_m=1.25)
        assert result.team_possession[0] == 100.0
        assert 1 not in result.team_possession or result.team_possession.get(1, 0) == 0.0
        assert result.owned_frames == 10

    def test_two_teams_split(self):
        """First 5 frames team 0 owns, next 5 frames team 1 owns → ~50/50."""
        frames = []
        for i in range(5):
            frames.append(_make_frame(i, ball_xy=(50.0, 34.0), players=[
                {"track_id": 1, "team_id": 0, "x": 50.5, "y": 34.0},
                {"track_id": 2, "team_id": 1, "x": 80.0, "y": 34.0},
            ]))
        for i in range(5, 10):
            frames.append(_make_frame(i, ball_xy=(80.0, 34.0), players=[
                {"track_id": 1, "team_id": 0, "x": 50.0, "y": 34.0},
                {"track_id": 2, "team_id": 1, "x": 80.5, "y": 34.0},
            ]))
        result = compute_possession(frames, max_dist_m=1.25)
        assert result.team_possession[0] == 50.0
        assert result.team_possession[1] == 50.0

    def test_missing_ball_not_counted(self):
        frames = [
            _make_frame(0, ball_xy=(50.0, 34.0), players=[
                {"track_id": 1, "team_id": 0, "x": 50.5, "y": 34.0},
            ]),
            _make_frame(1, ball_xy=None, players=[
                {"track_id": 1, "team_id": 0, "x": 50.5, "y": 34.0},
            ]),
        ]
        result = compute_possession(frames, max_dist_m=1.25)
        assert result.ball_missing_frames == 1
        assert result.owned_frames == 1

    def test_not_in_play_skipped(self):
        frames = [
            _make_frame(0, ball_xy=(50.0, 34.0), players=[
                {"track_id": 1, "team_id": 0, "x": 50.5, "y": 34.0},
            ]),
            _make_frame(1, ball_xy=(50.0, 34.0), flag="not_in_play", players=[
                {"track_id": 1, "team_id": 0, "x": 50.5, "y": 34.0},
            ]),
        ]
        result = compute_possession(frames, max_dist_m=1.25)
        assert result.owned_frames == 1

    def test_touch_segmentation(self):
        """Touch increments when owner changes."""
        frames = [
            _make_frame(0, ball_xy=(50.0, 34.0), players=[
                {"track_id": 1, "team_id": 0, "x": 50.5, "y": 34.0},
                {"track_id": 2, "team_id": 0, "x": 80.0, "y": 34.0},
            ]),
            _make_frame(1, ball_xy=(50.0, 34.0), players=[
                {"track_id": 1, "team_id": 0, "x": 50.5, "y": 34.0},
                {"track_id": 2, "team_id": 0, "x": 80.0, "y": 34.0},
            ]),
            # Ball moves to player 2
            _make_frame(2, ball_xy=(80.0, 34.0), players=[
                {"track_id": 1, "team_id": 0, "x": 50.0, "y": 34.0},
                {"track_id": 2, "team_id": 0, "x": 80.5, "y": 34.0},
            ]),
        ]
        result = compute_possession(frames, max_dist_m=1.25)
        # Player 1: first touch at frame 0
        # Player 2: touch at frame 2 (owner changed from 1→2)
        assert result.player_touches[1] == 1
        assert result.player_touches[2] == 1

    def test_unowned_frames_counted(self):
        """Ball present but no player nearby → unowned."""
        frames = [
            _make_frame(0, ball_xy=(50.0, 34.0), players=[
                {"track_id": 1, "team_id": 0, "x": 80.0, "y": 80.0},  # far away
            ]),
        ]
        result = compute_possession(frames, max_dist_m=1.25)
        assert result.owned_frames == 0
        assert result.unowned_frames == 1

    def test_player_team_override(self):
        """Explicit team override should take precedence."""
        frames = [
            _make_frame(0, ball_xy=(50.0, 34.0), players=[
                {"track_id": 1, "team_id": None, "x": 50.5, "y": 34.0},
            ]),
        ]
        result = compute_possession(
            frames, max_dist_m=1.25,
            player_team_override={1: 0},
        )
        assert result.team_possession[0] == 100.0

    def test_empty_frames(self):
        result = compute_possession([], max_dist_m=1.25)
        assert result.total_frames == 0
        assert result.owned_frames == 0
        assert result.team_possession == {}


# =========================================================================
# Output writers
# =========================================================================

class TestPossessionWriters:
    def test_write_team_possession(self, tmp_path):
        result = PossessionResult(
            team_possession={0: 60.0, 1: 40.0},
            player_touches={1: 5, 2: 3},
            player_team={1: 0, 2: 1},
            timeline=[],
            total_frames=100,
            owned_frames=80,
            unowned_frames=10,
            ball_missing_frames=10,
        )
        path = tmp_path / "team_possession.json"
        write_team_possession(path, result)
        data = json.loads(path.read_text())
        assert "teams" in data
        assert len(data["teams"]) == 2
        assert data["teams"][0]["team_id"] == 0
        assert data["teams"][0]["possession_pct"] == 60.0
        assert data["total_frames"] == 100

    def test_write_player_touches(self, tmp_path):
        result = PossessionResult(
            team_possession={},
            player_touches={1: 5, 2: 3},
            player_team={1: 0, 2: 1},
            timeline=[],
        )
        path = tmp_path / "player_touches.csv"
        write_player_touches(path, result)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        assert "track_id" in lines[0]

    def test_write_all_outputs(self, tmp_path):
        timeline = [FrameBallOwner(frame_idx=0, owner_track_id=1, ball_available=True)]
        result = PossessionResult(
            team_possession={0: 100.0},
            player_touches={1: 1},
            player_team={1: 0},
            timeline=timeline,
            total_frames=1,
            owned_frames=1,
        )
        write_all_outputs(tmp_path, result, write_timeline=True)
        assert (tmp_path / "team_possession.json").exists()
        assert (tmp_path / "player_touches.csv").exists()
        assert (tmp_path / "ball_owner_timeline.csv").exists()

    def test_write_all_no_timeline(self, tmp_path):
        result = PossessionResult(
            team_possession={0: 100.0},
            player_touches={1: 1},
            player_team={1: 0},
            timeline=[],
        )
        write_all_outputs(tmp_path, result, write_timeline=False)
        assert (tmp_path / "team_possession.json").exists()
        assert not (tmp_path / "ball_owner_timeline.csv").exists()
