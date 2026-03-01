"""Stats C — Ball Progression & Territory analytics.

Computes:
  - Ball progression rate (m/s toward goal)
  - Time in thirds (defensive / middle / attacking)
  - Possession proxy (time ball nearest to team)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from ..data_models import FrameFlag, FrameState

logger = logging.getLogger(__name__)


class BallProgressionStats:
    """Accumulates ball progression and territory statistics."""

    def __init__(self, config: dict[str, Any]) -> None:
        stats_cfg = config.get("stats", {})
        pitch_cfg = config.get("pitch", {})

        self.pitch_length: float = pitch_cfg.get("length", 105.0)
        self.pitch_width: float = pitch_cfg.get("width", 68.0)
        self.thirds_boundaries: list[float] = stats_cfg.get(
            "thirds_boundaries", [35.0, 70.0]
        )

        # Accumulators
        self._ball_positions: list[tuple[float, float, float]] = []  # (ts, x, y)
        self._time_in_thirds: dict[str, float] = {
            "defensive": 0.0,
            "middle": 0.0,
            "attacking": 0.0,
        }
        self._possession_frames: dict[int, int] = defaultdict(int)  # team_id -> count
        self._total_frames: int = 0
        self._prev_ts: Optional[float] = None
        self._progression_rates: list[float] = []

    def update(self, frame_state: FrameState) -> FrameState:
        """Process one frame for ball progression stats."""
        if frame_state.flag != FrameFlag.IN_PLAY:
            return frame_state

        ts = frame_state.timestamp_sec
        per_frame: dict[str, Any] = {}
        self._total_frames += 1

        ball_pos: Optional[tuple[float, float]] = None
        if frame_state.ball and frame_state.ball.pitch_pos:
            bx = frame_state.ball.pitch_pos.x
            by = frame_state.ball.pitch_pos.y
            ball_pos = (bx, by)
            self._ball_positions.append((ts, bx, by))

            # Third classification
            third = self._classify_third(bx)
            dt = ts - self._prev_ts if self._prev_ts is not None else 0.0
            dt = min(dt, 1.0)  # cap to avoid huge jumps
            self._time_in_thirds[third] += dt

            # Progression rate toward right goal (x=105)
            if len(self._ball_positions) >= 2:
                prev_ts, prev_x, _ = self._ball_positions[-2]
                delta_t = ts - prev_ts
                if delta_t > 0:
                    # Positive = moving toward right goal
                    prog_rate = (bx - prev_x) / delta_t
                    self._progression_rates.append(prog_rate)
                    per_frame["progression_rate_mps"] = round(prog_rate, 2)

            per_frame["third"] = third
            per_frame["ball_x"] = round(bx, 2)

        # Possession proxy: which team is nearest to ball?
        if ball_pos:
            min_dist = float("inf")
            nearest_team: Optional[int] = None
            for player in frame_state.players:
                if player.pitch_pos is None or player.team_id is None or player.team_id < 0:
                    continue
                d = ((player.pitch_pos.x - ball_pos[0]) ** 2
                     + (player.pitch_pos.y - ball_pos[1]) ** 2) ** 0.5
                if d < min_dist:
                    min_dist = d
                    nearest_team = player.team_id
            if nearest_team is not None:
                self._possession_frames[nearest_team] += 1
                per_frame["nearest_team"] = nearest_team

        self._prev_ts = ts
        frame_state.analytics["ball_progression"] = per_frame
        return frame_state

    def _classify_third(self, x: float) -> str:
        if x < self.thirds_boundaries[0]:
            return "defensive"
        elif x < self.thirds_boundaries[1]:
            return "middle"
        else:
            return "attacking"

    def get_team_summary(self) -> dict[str, Any]:
        """Aggregate ball progression & territory stats."""
        total = self._total_frames or 1
        possession: dict[int, float] = {}
        for t, count in self._possession_frames.items():
            possession[t] = round(count / total * 100, 1)

        total_time = sum(self._time_in_thirds.values()) or 1.0
        return {
            "time_in_thirds_pct": {
                k: round(v / total_time * 100, 1)
                for k, v in self._time_in_thirds.items()
            },
            "possession_pct": possession,
            "avg_progression_rate_mps": round(
                float(np.mean(self._progression_rates)), 3
            ) if self._progression_rates else 0.0,
            "total_ball_positions": len(self._ball_positions),
        }
