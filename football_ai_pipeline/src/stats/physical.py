"""Stats A — Physical & Movement analytics.

Computes per-player:
  - Estimated distance covered (pitch-mapped)
  - Sprint counts (threshold-based) + top speed
  - Team tempo: median ball speed, median player speed by team
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from ..data_models import FrameState, FrameFlag

logger = logging.getLogger(__name__)


class PhysicalStats:
    """Accumulates physical & movement statistics across frames."""

    def __init__(self, config: dict[str, Any]) -> None:
        stats_cfg = config.get("stats", {})
        self.sprint_threshold: float = stats_cfg.get("sprint_speed_threshold", 7.0)
        self.high_speed_threshold: float = stats_cfg.get("high_speed_threshold", 5.5)

        # Per-player accumulators
        self._distance: dict[int, float] = defaultdict(float)
        self._prev_pos: dict[int, tuple[float, float]] = {}
        self._sprint_count: dict[int, int] = defaultdict(int)
        self._was_sprinting: dict[int, bool] = defaultdict(lambda: False)
        self._top_speed: dict[int, float] = defaultdict(float)
        self._speeds: dict[int, list[float]] = defaultdict(list)
        self._team_map: dict[int, int] = {}

        # Ball speed accumulator
        self._ball_speeds: list[float] = []

    def update(self, frame_state: FrameState) -> FrameState:
        """Process one frame and accumulate physical stats."""
        if frame_state.flag != FrameFlag.IN_PLAY:
            return frame_state

        per_frame: dict[str, Any] = {}

        for player in frame_state.players:
            tid = player.track_id
            if player.team_id is not None:
                self._team_map[tid] = player.team_id

            if player.pitch_pos is None:
                continue

            pos = (player.pitch_pos.x, player.pitch_pos.y)

            # Distance
            if tid in self._prev_pos:
                prev = self._prev_pos[tid]
                d = ((pos[0] - prev[0]) ** 2 + (pos[1] - prev[1]) ** 2) ** 0.5
                # Sanity: ignore teleports > 5m between frames (tracking glitch)
                if d <= 5.0:
                    self._distance[tid] += d
            self._prev_pos[tid] = pos

            # Speed + sprints
            speed = player.speed_mps
            if speed is not None:
                self._speeds[tid].append(speed)
                if speed > self._top_speed[tid]:
                    self._top_speed[tid] = speed

                is_sprinting = speed >= self.sprint_threshold
                player.is_sprinting = is_sprinting
                if is_sprinting and not self._was_sprinting[tid]:
                    self._sprint_count[tid] += 1
                self._was_sprinting[tid] = is_sprinting

        # Ball speed
        if frame_state.ball and frame_state.ball.speed_mps is not None:
            self._ball_speeds.append(frame_state.ball.speed_mps)

        # Per-frame analytics
        team_speeds: dict[int, list[float]] = defaultdict(list)
        for player in frame_state.players:
            if player.speed_mps is not None and player.team_id is not None:
                team_speeds[player.team_id].append(player.speed_mps)

        per_frame["team_median_speed"] = {
            str(t): float(np.median(v)) for t, v in team_speeds.items() if v
        }
        if self._ball_speeds:
            per_frame["ball_speed_mps"] = (
                frame_state.ball.speed_mps if frame_state.ball else None
            )

        frame_state.analytics["physical"] = per_frame
        return frame_state

    def get_player_summary(self) -> dict[int, dict[str, Any]]:
        """Return per-player aggregate physical stats."""
        summary: dict[int, dict[str, Any]] = {}
        for tid in set(self._distance.keys()) | set(self._sprint_count.keys()):
            speeds = self._speeds.get(tid, [])
            summary[tid] = {
                "distance_m": round(self._distance.get(tid, 0.0), 1),
                "sprint_count": self._sprint_count.get(tid, 0),
                "top_speed_mps": round(self._top_speed.get(tid, 0.0), 2),
                "avg_speed_mps": round(float(np.mean(speeds)), 2) if speeds else 0.0,
                "team_id": self._team_map.get(tid),
            }
        return summary

    def get_team_summary(self) -> dict[int, dict[str, Any]]:
        """Return per-team aggregate physical stats."""
        player_summary = self.get_player_summary()
        teams: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for tid, ps in player_summary.items():
            t = ps.get("team_id")
            if t is not None and t >= 0:
                teams[t].append(ps)

        result: dict[int, dict[str, Any]] = {}
        for t, players in teams.items():
            result[t] = {
                "total_distance_m": round(sum(p["distance_m"] for p in players), 1),
                "avg_distance_m": round(
                    float(np.mean([p["distance_m"] for p in players])), 1
                ) if players else 0.0,
                "total_sprints": sum(p["sprint_count"] for p in players),
                "team_top_speed_mps": round(
                    max((p["top_speed_mps"] for p in players), default=0.0), 2
                ),
                "median_ball_speed_mps": round(
                    float(np.median(self._ball_speeds)), 2
                ) if self._ball_speeds else 0.0,
                "num_players_tracked": len(players),
            }
        return result
