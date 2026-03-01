"""V2 possession: ball-state classification, hysteresis, and team possession stats.

Provides standalone possession computation that can run:
1. Inline during the analytics pipeline (via AnalyticsEngine)
2. Post-hoc from frames.jsonl for inspection/debugging

Key design decisions:
  - max_dist_m = 1.25 (tight threshold for controlled state)
  - Ball states: controlled (player within threshold), loose (no player near),
    air (ball speed above threshold or height proxy)
  - Hysteresis: min_control_frames=5 to confirm new owner,
    max_gap_frames=10 to retain owner through short gaps
  - Touch increments when confirmed owner changes
  - Team possession % computed from controlled frames only
  - Player→team lookup built from most-frequent assignment across all frames
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrameBallOwner:
    """Ownership state for a single frame."""

    frame_idx: int
    owner_track_id: Optional[int] = None
    owner_team_id: Optional[int] = None
    distance_m: Optional[float] = None
    ball_available: bool = False
    ball_state: str = "unknown"  # "controlled" | "loose" | "air" | "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "owner_track_id": self.owner_track_id,
            "owner_team_id": self.owner_team_id,
            "distance_m": round(self.distance_m, 3) if self.distance_m is not None else None,
            "ball_available": self.ball_available,
            "ball_state": self.ball_state,
        }


@dataclass
class PossessionResult:
    """Aggregate possession stats."""

    team_possession: dict[int, float]        # team_id → pct (0-100)
    player_touches: dict[int, int]           # track_id → touch count
    player_team: dict[int, int]              # track_id → team_id
    timeline: list[FrameBallOwner]           # per-frame timeline
    total_frames: int = 0
    owned_frames: int = 0
    unowned_frames: int = 0
    ball_missing_frames: int = 0
    controlled_frames: int = 0
    loose_frames: int = 0
    air_frames: int = 0
    ball_detected_frames: int = 0


# ---------------------------------------------------------------------------
# Frame-level helpers
# ---------------------------------------------------------------------------

def extract_tracks_from_frame(frame: dict[str, Any]) -> tuple[
    Optional[tuple[float, float]],
    list[dict[str, Any]],
]:
    """Extract ball position and player tracks from a serialised frame dict.

    Returns:
        (ball_xy, players) where ball_xy is (pitch_x, pitch_y) or None,
        and players is a list of dicts with track_id, team_id, x, y.
    """
    # Ball position
    ball_xy: Optional[tuple[float, float]] = None
    ball = frame.get("ball")
    if ball is not None:
        px, py = ball.get("pitch_x"), ball.get("pitch_y")
        if px is not None and py is not None:
            ball_xy = (float(px), float(py))

    # Players with pitch coordinates
    players: list[dict[str, Any]] = []
    for p in frame.get("players", []):
        cls = p.get("class", "player")
        if cls == "ball":
            continue
        px, py = p.get("pitch_x"), p.get("pitch_y")
        if px is None or py is None:
            continue
        players.append({
            "track_id": p["track_id"],
            "team_id": p.get("team_id"),
            "x": float(px),
            "y": float(py),
        })

    return ball_xy, players


def extract_ball_speed_from_frame(frame: dict[str, Any]) -> Optional[float]:
    """Extract ball speed (m/s) from a serialised frame dict."""
    ball = frame.get("ball")
    if ball is not None:
        spd = ball.get("speed_mps")
        if spd is not None:
            return float(spd)
    return None


def assign_ball_owner(
    ball_xy: tuple[float, float],
    players: list[dict[str, Any]],
    max_dist_m: float = 1.25,
) -> tuple[Optional[int], Optional[int], Optional[float]]:
    """Find the nearest player within *max_dist_m* of the ball.

    Args:
        ball_xy:    (x, y) ball position in pitch metres.
        players:    List of dicts with track_id, team_id, x, y.
        max_dist_m: Maximum distance to claim ownership.

    Returns:
        (track_id, team_id, distance) of nearest player, or (None, None, None)
        if no player is within range.
    """
    bx, by = ball_xy
    best_id: Optional[int] = None
    best_team: Optional[int] = None
    best_dist = float("inf")

    for p in players:
        dx = p["x"] - bx
        dy = p["y"] - by
        d = math.sqrt(dx * dx + dy * dy)
        if d < best_dist:
            best_dist = d
            best_id = p["track_id"]
            best_team = p.get("team_id")

    if best_dist <= max_dist_m:
        return best_id, best_team, best_dist
    return None, None, None


# ---------------------------------------------------------------------------
# Player → team lookup builder
# ---------------------------------------------------------------------------

def build_player_team_lookup(
    frames: list[dict[str, Any]],
) -> dict[int, int]:
    """Build a track_id → team_id map using the most-frequent team assignment.

    This resolves cases where team_id is null in some frames but populated
    in others (e.g. team classification runs after tracking).
    """
    votes: dict[int, Counter[int]] = defaultdict(Counter)
    for frame in frames:
        for p in frame.get("players", []):
            tid = p.get("team_id")
            track = p.get("track_id")
            if track is not None and tid is not None:
                votes[track][tid] += 1

    lookup: dict[int, int] = {}
    for track_id, counter in votes.items():
        if counter:
            lookup[track_id] = counter.most_common(1)[0][0]
    return lookup


# ---------------------------------------------------------------------------
# Core possession computation
# ---------------------------------------------------------------------------

def classify_ball_state(
    ball_xy: Optional[tuple[float, float]],
    players: list[dict[str, Any]],
    ball_speed_mps: Optional[float],
    max_dist_m: float,
    air_speed_threshold: float = 15.0,
) -> str:
    """Classify the ball state for this frame.

    Returns one of: "controlled", "loose", "air", "unknown".
    """
    if ball_xy is None:
        return "unknown"

    # Air: ball moving fast (likely a long pass/shot in flight)
    if ball_speed_mps is not None and ball_speed_mps > air_speed_threshold:
        return "air"

    # Check if any player is within control range
    _, _, dist = assign_ball_owner(ball_xy, players, max_dist_m)
    if dist is not None:
        return "controlled"

    return "loose"


def compute_possession(
    frames: list[dict[str, Any]],
    max_dist_m: float = 1.25,
    player_team_override: Optional[dict[int, int]] = None,
    min_control_frames: int = 5,
    max_gap_frames: int = 10,
    air_speed_threshold: float = 15.0,
) -> PossessionResult:
    """Compute team/player possession from a list of serialised frame dicts.

    Uses ball state classification (controlled/loose/air) and hysteresis
    to prevent flickering in ownership assignment.

    Args:
        frames:                List of frame dicts (from frames.jsonl or FrameState.to_serializable()).
        max_dist_m:            Maximum ball-to-player distance for ownership.
        player_team_override:  Optional pre-built track_id→team_id map.
                               If None, built automatically from frames.
        min_control_frames:    Consecutive frames a candidate must be nearest
                               before ownership switches (hysteresis).
        max_gap_frames:        Maximum consecutive non-controlled frames to
                               retain the current owner through.
        air_speed_threshold:   Ball speed (m/s) above which the ball is
                               classified as "air".

    Returns:
        PossessionResult with team possession %, player touches, and timeline.
    """
    if player_team_override is not None:
        pt_lookup = dict(player_team_override)
    else:
        pt_lookup = build_player_team_lookup(frames)

    timeline: list[FrameBallOwner] = []
    team_frames: dict[int, int] = defaultdict(int)
    player_touches: dict[int, int] = defaultdict(int)

    # Hysteresis state
    confirmed_owner: Optional[int] = None   # currently confirmed owner
    confirmed_team: Optional[int] = None
    candidate_owner: Optional[int] = None   # candidate waiting to be confirmed
    candidate_count: int = 0                # consecutive frames candidate has been nearest
    gap_count: int = 0                      # frames since confirmed owner lost control

    owned_frames = 0
    ball_missing = 0
    controlled_frames = 0
    loose_frames = 0
    air_frames = 0
    ball_detected = 0

    for frame in frames:
        fidx = frame.get("frame_idx", len(timeline))

        # Skip not-in-play frames
        if frame.get("flag") == "not_in_play":
            timeline.append(FrameBallOwner(frame_idx=fidx))
            confirmed_owner = None
            confirmed_team = None
            candidate_owner = None
            candidate_count = 0
            gap_count = 0
            continue

        ball_xy, players = extract_tracks_from_frame(frame)
        ball_speed = extract_ball_speed_from_frame(frame)

        rec = FrameBallOwner(frame_idx=fidx)

        if ball_xy is None:
            ball_missing += 1
            rec.ball_available = False
            rec.ball_state = "unknown"
            # Count gap for hysteresis
            gap_count += 1
            if gap_count > max_gap_frames:
                confirmed_owner = None
                confirmed_team = None
                candidate_owner = None
                candidate_count = 0
            timeline.append(rec)
            continue

        ball_detected += 1
        rec.ball_available = True

        # Classify ball state
        state = classify_ball_state(
            ball_xy, players, ball_speed, max_dist_m, air_speed_threshold,
        )
        rec.ball_state = state

        if state == "controlled":
            controlled_frames += 1
        elif state == "loose":
            loose_frames += 1
        elif state == "air":
            air_frames += 1

        # Find nearest player candidate
        track_id, frame_team, dist = assign_ball_owner(ball_xy, players, max_dist_m)

        if state == "controlled" and track_id is not None:
            team_id = pt_lookup.get(track_id, frame_team)
            rec.distance_m = dist

            if track_id == confirmed_owner:
                # Same owner — reset gap, keep going
                gap_count = 0
                candidate_owner = None
                candidate_count = 0
            elif track_id == candidate_owner:
                # Same candidate — increment count
                candidate_count += 1
            else:
                # New candidate — start counting
                candidate_owner = track_id
                candidate_count = 1

            # Check if candidate has earned ownership
            if candidate_owner is not None and candidate_count >= min_control_frames:
                # Ownership switch
                old_owner = confirmed_owner
                confirmed_owner = candidate_owner
                confirmed_team = pt_lookup.get(confirmed_owner, frame_team)
                candidate_owner = None
                candidate_count = 0
                gap_count = 0

                # Record touch on ownership change
                if confirmed_owner is not None:
                    if old_owner is not None and old_owner != confirmed_owner:
                        player_touches[confirmed_owner] += 1
                    elif old_owner is None:
                        player_touches[confirmed_owner] += 1

            # If no confirmed owner yet and we have a candidate under threshold,
            # accept immediately on first ever assignment
            if confirmed_owner is None and track_id is not None:
                confirmed_owner = track_id
                confirmed_team = team_id
                candidate_owner = None
                candidate_count = 0
                gap_count = 0
                player_touches[confirmed_owner] += 1

            # Assign confirmed owner to this frame
            if confirmed_owner is not None:
                rec.owner_track_id = confirmed_owner
                rec.owner_team_id = confirmed_team
                owned_frames += 1
                if confirmed_team is not None:
                    team_frames[confirmed_team] += 1
                gap_count = 0

        elif state in ("loose", "air"):
            # No player in control range — count as gap
            gap_count += 1
            if gap_count <= max_gap_frames and confirmed_owner is not None:
                # Retain owner through short gap
                rec.owner_track_id = confirmed_owner
                rec.owner_team_id = confirmed_team
                owned_frames += 1
                if confirmed_team is not None:
                    team_frames[confirmed_team] += 1
            else:
                # Gap too long — drop ownership
                confirmed_owner = None
                confirmed_team = None
                candidate_owner = None
                candidate_count = 0

        timeline.append(rec)

    # Compute team possession percentages (from controlled+retained frames)
    total = len(timeline)
    team_possession: dict[int, float] = {}
    if owned_frames > 0:
        for tid, count in team_frames.items():
            team_possession[tid] = round((count / owned_frames) * 100.0, 1)

    return PossessionResult(
        team_possession=team_possession,
        player_touches=dict(player_touches),
        player_team=pt_lookup,
        timeline=timeline,
        total_frames=total,
        owned_frames=owned_frames,
        unowned_frames=total - owned_frames - ball_missing,
        ball_missing_frames=ball_missing,
        controlled_frames=controlled_frames,
        loose_frames=loose_frames,
        air_frames=air_frames,
        ball_detected_frames=ball_detected,
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_team_possession(path: Path, result: PossessionResult) -> None:
    """Write out/team_possession.json with controlled/loose/air breakdown."""
    teams = []
    for tid in sorted(result.team_possession):
        teams.append({
            "team_id": tid,
            "possession_pct": result.team_possession[tid],
        })

    # Ball state breakdown percentages (relative to ball_detected_frames)
    det = max(result.ball_detected_frames, 1)
    data = {
        "teams": teams,
        "total_frames": result.total_frames,
        "ball_detected_frames": result.ball_detected_frames,
        "owned_frames": result.owned_frames,
        "unowned_frames": result.unowned_frames,
        "ball_missing_frames": result.ball_missing_frames,
        "controlled_frames": result.controlled_frames,
        "loose_frames": result.loose_frames,
        "air_frames": result.air_frames,
        "controlled_pct": round(result.controlled_frames / det * 100.0, 1),
        "loose_pct": round(result.loose_frames / det * 100.0, 1),
        "air_pct": round(result.air_frames / det * 100.0, 1),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Wrote team_possession.json (%d teams)", len(teams))


def write_player_touches(path: Path, result: PossessionResult) -> None:
    """Write out/player_touches.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for track_id in sorted(result.player_touches):
        rows.append({
            "track_id": track_id,
            "team_id": result.player_team.get(track_id),
            "touches": result.player_touches[track_id],
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "team_id", "touches"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote player_touches.csv (%d players)", len(rows))


def write_ball_owner_timeline(path: Path, result: PossessionResult) -> None:
    """Write out/ball_owner_timeline.csv (optional, for debugging)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame_idx", "owner_track_id", "owner_team_id", "distance_m", "ball_available", "ball_state"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in result.timeline:
            writer.writerow(rec.to_dict())
    logger.info("Wrote ball_owner_timeline.csv (%d frames)", len(result.timeline))


def write_all_outputs(
    output_dir: Path,
    result: PossessionResult,
    write_timeline: bool = False,
) -> None:
    """Write all possession output files."""
    write_team_possession(output_dir / "team_possession.json", result)
    write_player_touches(output_dir / "player_touches.csv", result)
    if write_timeline:
        write_ball_owner_timeline(output_dir / "ball_owner_timeline.csv", result)
