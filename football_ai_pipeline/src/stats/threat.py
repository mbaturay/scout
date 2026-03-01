"""Stats E — Threat / Danger analytics.

Computes:
  - xT (expected threat) from ball movement using an open xT grid
  - Threat by zone occupancy (zone-based heuristics)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from ..data_models import FrameFlag, FrameState

logger = logging.getLogger(__name__)

# Karun Singh's 12x8 xT grid (simplified — values increase toward goal)
# Rows = pitch length zones (0=own goal, 11=opp goal)
# Cols = pitch width zones (0=left, 7=right)
_XT_GRID_12x8 = np.array([
    [0.006, 0.007, 0.007, 0.008, 0.008, 0.007, 0.007, 0.006],
    [0.008, 0.009, 0.010, 0.011, 0.011, 0.010, 0.009, 0.008],
    [0.010, 0.012, 0.014, 0.015, 0.015, 0.014, 0.012, 0.010],
    [0.013, 0.016, 0.019, 0.021, 0.021, 0.019, 0.016, 0.013],
    [0.017, 0.021, 0.026, 0.030, 0.030, 0.026, 0.021, 0.017],
    [0.022, 0.028, 0.036, 0.043, 0.043, 0.036, 0.028, 0.022],
    [0.029, 0.038, 0.051, 0.063, 0.063, 0.051, 0.038, 0.029],
    [0.039, 0.053, 0.074, 0.095, 0.095, 0.074, 0.053, 0.039],
    [0.052, 0.074, 0.108, 0.145, 0.145, 0.108, 0.074, 0.052],
    [0.071, 0.107, 0.163, 0.215, 0.215, 0.163, 0.107, 0.071],
    [0.105, 0.163, 0.251, 0.367, 0.367, 0.251, 0.163, 0.105],
    [0.150, 0.234, 0.365, 0.580, 0.580, 0.365, 0.234, 0.150],
])


class ThreatStats:
    """Accumulates expected threat and zone occupancy statistics."""

    def __init__(self, config: dict[str, Any]) -> None:
        pitch_cfg = config.get("pitch", {})
        self.pitch_length: float = pitch_cfg.get("length", 105.0)
        self.pitch_width: float = pitch_cfg.get("width", 68.0)

        self.xt_grid = _XT_GRID_12x8
        self.grid_rows = self.xt_grid.shape[0]  # 12
        self.grid_cols = self.xt_grid.shape[1]  # 8

        # Accumulators
        self._xt_deltas: dict[int, list[float]] = defaultdict(list)  # team -> [delta_xt]
        self._zone_occupancy: dict[int, np.ndarray] = {}  # team -> grid counts
        self._total_frames: int = 0

    def _pitch_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Map pitch coordinates to xT grid cell."""
        row = int(np.clip(x / self.pitch_length * self.grid_rows, 0, self.grid_rows - 1))
        col = int(np.clip(y / self.pitch_width * self.grid_cols, 0, self.grid_cols - 1))
        return row, col

    def _get_xt(self, x: float, y: float) -> float:
        """Get xT value at pitch position."""
        r, c = self._pitch_to_grid(x, y)
        return float(self.xt_grid[r, c])

    def update(self, frame_state: FrameState) -> FrameState:
        """Compute threat stats for one frame."""
        if frame_state.flag != FrameFlag.IN_PLAY:
            return frame_state

        self._total_frames += 1
        per_frame: dict[str, Any] = {}

        # xT from ball movement
        if frame_state.ball and frame_state.ball.pitch_pos:
            bx = frame_state.ball.pitch_pos.x
            by = frame_state.ball.pitch_pos.y
            current_xt = self._get_xt(bx, by)
            per_frame["ball_xt"] = round(current_xt, 4)

        # Zone occupancy per team
        team_zones: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for player in frame_state.players:
            if player.pitch_pos is None or player.team_id is None or player.team_id < 0:
                continue
            r, c = self._pitch_to_grid(player.pitch_pos.x, player.pitch_pos.y)
            team_zones[player.team_id].append((r, c))

            if player.team_id not in self._zone_occupancy:
                self._zone_occupancy[player.team_id] = np.zeros(
                    (self.grid_rows, self.grid_cols)
                )
            self._zone_occupancy[player.team_id][r, c] += 1

        # Team threat = sum of xT at player positions
        for team_id, zones in team_zones.items():
            threat = sum(self.xt_grid[r, c] for r, c in zones)
            per_frame[f"team_{team_id}_threat"] = round(float(threat), 4)

        frame_state.analytics["threat"] = per_frame
        return frame_state

    def get_team_summary(self) -> dict[int, dict[str, Any]]:
        """Aggregate threat stats per team."""
        result: dict[int, dict[str, Any]] = {}
        for team_id, grid in self._zone_occupancy.items():
            total = grid.sum()
            if total == 0:
                continue
            # Normalised zone occupancy
            norm = grid / total
            # Weighted threat = occupancy * xt_grid
            weighted_threat = float((norm * self.xt_grid).sum())

            # Find highest occupancy zones
            flat = grid.flatten()
            top_indices = flat.argsort()[-3:][::-1]
            top_zones = []
            for idx in top_indices:
                r, c = divmod(idx, self.grid_cols)
                top_zones.append({
                    "row": int(r), "col": int(c),
                    "count": int(flat[idx]),
                    "xt": round(float(self.xt_grid[r, c]), 4),
                })

            result[team_id] = {
                "avg_weighted_threat": round(weighted_threat, 4),
                "top_zones": top_zones,
                "total_frames": self._total_frames,
            }
        return result
