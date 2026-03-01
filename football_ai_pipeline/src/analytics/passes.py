"""V2 pass detection from the ball-ownership timeline.

Scans the per-frame FrameBallOwner sequence for ownership transitions and
classifies each as a completed pass, interception, or turnover based on
team membership, ball travel distance, elapsed time, and ball state.

Rules:
  - A completed pass requires: same team, ball >= min_pass_dist_m,
    elapsed <= max_pass_time_s, and ball was controlled before switch.
  - An interception requires: different team, ball was controlled before switch.
  - A turnover: different team but ball was not controlled (contested).
  - Dribble filter: A→None→A within a short window is not a pass.
  - Air state: allows a longer gap window but requires travel distance.

This is a heuristic detector — not a classifier.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
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
    """A single pass / interception / turnover detected from the ownership timeline."""

    t_start: int          # frame_idx where from_track last owned
    t_end: int            # frame_idx where to_track first owned
    from_track: int       # track_id of passer
    to_track: int         # track_id of receiver
    team_id: Optional[int] = None   # team of passer
    to_team_id: Optional[int] = None  # team of receiver
    dist_m: float = 0.0  # ball travel distance (metres)
    duration_s: float = 0.0  # time between A-last and B-first
    is_completed: bool = False  # same team = completed
    reason: str = ""      # "pass" | "interception" | "turnover"
    ball_state_before: str = ""  # ball state at transition start

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_start": self.t_start,
            "t_end": self.t_end,
            "from_track": self.from_track,
            "to_track": self.to_track,
            "team_id": self.team_id,
            "to_team_id": self.to_team_id,
            "dist_m": round(self.dist_m, 2),
            "duration_s": round(self.duration_s, 3),
            "is_completed": self.is_completed,
            "type": self.reason,
        }


# ---------------------------------------------------------------------------
# Core pass detection
# ---------------------------------------------------------------------------

def compute_passes(
    timeline: list[FrameBallOwner],
    ball_positions: list[Optional[tuple[float, float]]],
    player_team: dict[int, int],
    fps: float = 30.0,
    timestamps: Optional[list[float]] = None,
    min_pass_dist_m: float = 4.0,
    max_gap_frames: int = 10,
    max_pass_time_s: float = 1.5,
    air_gap_frames: int = 20,
) -> list[PassEvent]:
    """Detect passes from the ownership timeline.

    Args:
        timeline:         Per-frame FrameBallOwner list (from compute_possession).
        ball_positions:   Per-frame (pitch_x, pitch_y) or None, aligned with timeline.
        player_team:      track_id → team_id lookup.
        fps:              Frames per second (used when timestamps not provided).
        timestamps:       Per-frame timestamps in seconds (aligned with timeline).
                          If None, frame_idx / fps is used.
        min_pass_dist_m:  Minimum ball travel to qualify as a pass (metres).
        max_gap_frames:   Maximum consecutive unowned frames to bridge.
        max_pass_time_s:  Maximum elapsed time (seconds) for a completed pass.
        air_gap_frames:   Extended gap allowance when ball state is "air".

    Returns:
        List of PassEvent sorted by t_start.
    """
    n = len(timeline)
    if n < 2:
        return []

    passes: list[PassEvent] = []

    def _get_ts(idx: int) -> float:
        if timestamps is not None and idx < len(timestamps):
            return timestamps[idx]
        return timeline[idx].frame_idx / fps if fps > 0 else 0.0

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

        # Check ball state in the segment (use last controlled frame)
        seg_ball_state = _last_ball_state(timeline, seg_start, seg_end)
        was_controlled = seg_ball_state == "controlled"

        # Determine gap allowance based on ball state during gap
        gap_has_air = False
        effective_max_gap = max_gap_frames

        # Look ahead through a gap of None-owned frames
        gap_start = j
        gap_count = 0
        while j < n and timeline[j].owner_track_id is None:
            if timeline[j].ball_state == "air":
                gap_has_air = True
            gap_count += 1
            j += 1
            # Use extended gap for air state
            cur_max = air_gap_frames if gap_has_air else max_gap_frames
            if gap_count > cur_max:
                break

        effective_max_gap = air_gap_frames if gap_has_air else max_gap_frames

        if gap_count > effective_max_gap or j >= n:
            # Gap too large or end of timeline — no transition
            i = j
            continue

        # j is now the first frame of the next owner
        next_owner = timeline[j].owner_track_id
        if next_owner is None:
            i = j
            continue

        # Dribble filter: A→None→A = same player resumed, not a pass
        if next_owner == seg_owner:
            i = j
            continue

        # Compute ball travel distance
        ball_start = _find_ball_pos(ball_positions, seg_start, seg_end)
        ball_end = _find_ball_pos(ball_positions, j, min(j + 5, n - 1))
        dist = _euclidean(ball_start, ball_end)

        # Compute elapsed time
        ts_start = _get_ts(seg_end)
        ts_end = _get_ts(j)
        elapsed = ts_end - ts_start

        # Determine teams
        from_team = player_team.get(seg_owner)
        to_team = player_team.get(next_owner)

        same_team = (from_team is not None and to_team is not None
                     and from_team == to_team)
        diff_team = (from_team is not None and to_team is not None
                     and from_team != to_team)

        # Classify the event
        if same_team:
            # Completed pass check
            if dist < min_pass_dist_m:
                # Too short — not a pass, just a close-range exchange/dribble
                i = j
                continue
            if elapsed > max_pass_time_s and not gap_has_air:
                # Took too long (unless ball was in air)
                i = j
                continue
            # Air passes: require distance but allow longer time
            if gap_has_air and dist < min_pass_dist_m:
                i = j
                continue
            reason = "pass"
            is_completed = True
        elif diff_team:
            if was_controlled:
                reason = "interception"
            else:
                reason = "turnover"
            is_completed = False
            # Interceptions/turnovers don't require min distance
            # but skip trivially short contested balls
            if dist < 1.0 and gap_count == 0:
                i = j
                continue
        else:
            # Unknown team for one side
            reason = "turnover"
            is_completed = False
            if dist < 1.0:
                i = j
                continue

        passes.append(PassEvent(
            t_start=timeline[seg_end].frame_idx,
            t_end=timeline[j].frame_idx,
            from_track=seg_owner,
            to_track=next_owner,
            team_id=from_team,
            to_team_id=to_team,
            dist_m=dist,
            duration_s=max(elapsed, 0.0),
            is_completed=is_completed,
            reason=reason,
            ball_state_before=seg_ball_state,
        ))

        i = j

    passes.sort(key=lambda p: p.t_start)
    return passes


def _last_ball_state(
    timeline: list[FrameBallOwner], start: int, end: int,
) -> str:
    """Return the last non-unknown ball_state in [start, end]."""
    for k in range(end, start - 1, -1):
        state = timeline[k].ball_state
        if state != "unknown":
            return state
    return "unknown"


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
            pass_count, completed_pass_count, pass_accuracy_pct,
            interceptions_won
        }
        player_pass_stats[track_id] = {
            passes_attempted, passes_completed, pass_accuracy_pct,
            interceptions
        }
    """
    team_attempted: dict[int, int] = defaultdict(int)
    team_completed: dict[int, int] = defaultdict(int)
    team_interceptions: dict[int, int] = defaultdict(int)
    player_attempted: dict[int, int] = defaultdict(int)
    player_completed: dict[int, int] = defaultdict(int)
    player_interceptions: dict[int, int] = defaultdict(int)

    for p in passes:
        if p.reason == "pass":
            # Attempted by passer
            player_attempted[p.from_track] += 1
            if p.team_id is not None:
                team_attempted[p.team_id] += 1
            # Completed
            player_completed[p.from_track] += 1
            if p.team_id is not None:
                team_completed[p.team_id] += 1
        elif p.reason == "interception":
            # Count as attempted pass by from_track (failed)
            player_attempted[p.from_track] += 1
            if p.team_id is not None:
                team_attempted[p.team_id] += 1
            # Count interception won by receiver's team
            if p.to_team_id is not None:
                team_interceptions[p.to_team_id] += 1
            player_interceptions[p.to_track] += 1
        elif p.reason == "turnover":
            # Count as attempted pass by from_track (failed)
            player_attempted[p.from_track] += 1
            if p.team_id is not None:
                team_attempted[p.team_id] += 1

    # Build team stats
    team_stats: dict[int, dict[str, Any]] = {}
    all_team_ids = set(team_attempted) | set(team_completed) | set(team_interceptions)
    for tid in sorted(all_team_ids):
        att = team_attempted.get(tid, 0)
        comp = team_completed.get(tid, 0)
        team_stats[tid] = {
            "pass_count": att,
            "completed_pass_count": comp,
            "pass_accuracy_pct": round((comp / max(1, att)) * 100.0, 1),
            "interceptions_won": team_interceptions.get(tid, 0),
        }

    # Build player stats
    player_stats: dict[int, dict[str, Any]] = {}
    all_players = set(player_attempted) | set(player_completed) | set(player_interceptions)
    for pid in sorted(all_players):
        att = player_attempted.get(pid, 0)
        comp = player_completed.get(pid, 0)
        player_stats[pid] = {
            "passes_attempted": att,
            "passes_completed": comp,
            "pass_accuracy_pct": round((comp / max(1, att)) * 100.0, 1),
            "interceptions": player_interceptions.get(pid, 0),
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
