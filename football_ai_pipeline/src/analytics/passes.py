"""V1 pass detection from the ball-ownership timeline.

Scans the per-frame FrameBallOwner sequence for ownership transitions and
classifies each as a completed pass, interception, or turnover based on
team membership and ball travel distance.

This is a heuristic detector — not a classifier.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .possession import FrameBallOwner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PassEvent:
    """A single pass / turnover detected from the ownership timeline."""

    t_start: int          # frame_idx where from_track last owned
    t_end: int            # frame_idx where to_track first owned
    from_track: int       # track_id of passer
    to_track: int         # track_id of receiver
    team_id: Optional[int] = None   # team of passer
    to_team_id: Optional[int] = None  # team of receiver
    dist_m: float = 0.0  # ball travel distance (metres)
    is_completed: bool = False  # same team = completed
    reason: str = ""      # "completed_pass" | "interception" | "turnover"

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_start": self.t_start,
            "t_end": self.t_end,
            "from_track": self.from_track,
            "to_track": self.to_track,
            "team_id": self.team_id,
            "to_team_id": self.to_team_id,
            "dist_m": round(self.dist_m, 2),
            "is_completed": self.is_completed,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Core pass detection
# ---------------------------------------------------------------------------

def compute_passes(
    timeline: list[FrameBallOwner],
    ball_positions: list[Optional[tuple[float, float]]],
    player_team: dict[int, int],
    min_pass_dist_m: float = 3.0,
    max_gap_frames: int = 10,
) -> list[PassEvent]:
    """Detect passes from the ownership timeline.

    Args:
        timeline:         Per-frame FrameBallOwner list (from compute_possession).
        ball_positions:   Per-frame (pitch_x, pitch_y) or None, aligned with timeline.
        player_team:      track_id → team_id lookup.
        min_pass_dist_m:  Minimum ball travel to qualify as a pass (metres).
        max_gap_frames:   Maximum consecutive unowned frames to bridge as a
                          single transition (ball in air / loose).

    Returns:
        List of PassEvent sorted by t_start.
    """
    n = len(timeline)
    if n < 2:
        return []

    passes: list[PassEvent] = []

    # Walk the timeline looking for ownership segments
    i = 0
    while i < n:
        # Skip to next owned frame
        if timeline[i].owner_track_id is None:
            i += 1
            continue

        # Start of an ownership segment
        seg_start = i
        seg_owner = timeline[i].owner_track_id

        # Advance while same owner
        j = i + 1
        while j < n and timeline[j].owner_track_id == seg_owner:
            j += 1

        # j is now either past end, or a different/None owner
        seg_end = j - 1  # last frame owned by seg_owner

        # Look ahead through a gap of None-owned frames
        gap_start = j
        gap_count = 0
        while j < n and timeline[j].owner_track_id is None:
            gap_count += 1
            j += 1
            if gap_count > max_gap_frames:
                break

        if gap_count > max_gap_frames or j >= n:
            # Gap too large or end of timeline — no transition
            i = j
            continue

        # j is now the first frame of the next owner
        next_owner = timeline[j].owner_track_id
        if next_owner is None or next_owner == seg_owner:
            # Same owner resumed after gap — not a pass
            i = j
            continue

        # We have a transition: seg_owner → next_owner
        # Compute ball travel distance
        ball_start = _find_ball_pos(ball_positions, seg_start, seg_end)
        ball_end = _find_ball_pos(ball_positions, j, min(j + 5, n - 1))
        dist = _euclidean(ball_start, ball_end)

        if dist < min_pass_dist_m:
            # Too short — not a pass, just a contested ball / dribble turnover
            i = j
            continue

        # Determine teams
        from_team = player_team.get(seg_owner)
        to_team = player_team.get(next_owner)

        if from_team is not None and to_team is not None and from_team == to_team:
            reason = "completed_pass"
            is_completed = True
        elif from_team is not None and to_team is not None and from_team != to_team:
            reason = "interception"
            is_completed = False
        else:
            # Unknown team for one side — mark as attempted, not completed
            reason = "turnover"
            is_completed = False

        passes.append(PassEvent(
            t_start=timeline[seg_end].frame_idx,
            t_end=timeline[j].frame_idx,
            from_track=seg_owner,
            to_track=next_owner,
            team_id=from_team,
            to_team_id=to_team,
            dist_m=dist,
            is_completed=is_completed,
            reason=reason,
        ))

        i = j

    passes.sort(key=lambda p: p.t_start)
    return passes


def _find_ball_pos(
    positions: list[Optional[tuple[float, float]]],
    start: int,
    end: int,
) -> Optional[tuple[float, float]]:
    """Find the last non-None ball position in [start, end]."""
    for k in range(end, start - 1, -1):
        if 0 <= k < len(positions) and positions[k] is not None:
            return positions[k]
    return None


def _euclidean(
    a: Optional[tuple[float, float]],
    b: Optional[tuple[float, float]],
) -> float:
    """Euclidean distance, returns 0 if either point is None."""
    if a is None or b is None:
        return 0.0
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_pass_stats(
    passes: list[PassEvent],
    player_team: dict[int, int],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Compute team-level and player-level pass stats.

    Returns:
        (team_pass_stats, player_pass_stats)

        team_pass_stats[team_id] = {
            pass_count, completed_pass_count, pass_accuracy_pct
        }
        player_pass_stats[track_id] = {
            passes_attempted, passes_completed, pass_accuracy_pct
        }
    """
    from collections import defaultdict

    team_attempted: dict[int, int] = defaultdict(int)
    team_completed: dict[int, int] = defaultdict(int)
    player_attempted: dict[int, int] = defaultdict(int)
    player_completed: dict[int, int] = defaultdict(int)

    for p in passes:
        # Every pass is an attempt by from_track
        player_attempted[p.from_track] += 1
        if p.team_id is not None:
            team_attempted[p.team_id] += 1

        if p.is_completed:
            player_completed[p.from_track] += 1
            if p.team_id is not None:
                team_completed[p.team_id] += 1

    # Build team stats
    team_stats: dict[int, dict[str, Any]] = {}
    all_team_ids = set(team_attempted) | set(team_completed)
    for tid in sorted(all_team_ids):
        att = team_attempted[tid]
        comp = team_completed[tid]
        team_stats[tid] = {
            "pass_count": att,
            "completed_pass_count": comp,
            "pass_accuracy_pct": round((comp / max(1, att)) * 100.0, 1),
        }

    # Build player stats
    player_stats: dict[int, dict[str, Any]] = {}
    all_players = set(player_attempted) | set(player_completed)
    for pid in sorted(all_players):
        att = player_attempted[pid]
        comp = player_completed[pid]
        player_stats[pid] = {
            "passes_attempted": att,
            "passes_completed": comp,
            "pass_accuracy_pct": round((comp / max(1, att)) * 100.0, 1),
        }

    return team_stats, player_stats


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_pass_events(path: Path, passes: list[PassEvent]) -> None:
    """Write out/pass_events.json."""
    data = {"pass_events": [p.to_dict() for p in passes]}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Wrote pass_events.json (%d events)", len(passes))
