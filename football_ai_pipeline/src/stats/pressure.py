"""Stats D — Pressure & Control analytics (proxy models).

Computes:
  - Pitch control (Voronoi-based) per frame and team control ratio
  - Pressure index on ball carrier: nearby opponents within radius + closing speed
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from ..data_models import FrameFlag, FrameState

logger = logging.getLogger(__name__)


def _voronoi_control(
    positions: dict[int, list[tuple[float, float]]],
    pitch_length: float,
    pitch_width: float,
    grid_resolution: int = 20,
) -> dict[int, float]:
    """Compute Voronoi-based pitch control ratio per team.

    Discretises the pitch into a grid and assigns each cell to the
    nearest player's team.
    """
    xs = np.linspace(0, pitch_length, grid_resolution)
    ys = np.linspace(0, pitch_width, grid_resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    all_players: list[tuple[float, float, int]] = []
    for team_id, pos_list in positions.items():
        for px, py in pos_list:
            all_players.append((px, py, team_id))

    if not all_players:
        return {}

    player_arr = np.array([(p[0], p[1]) for p in all_players])
    team_arr = np.array([p[2] for p in all_players])

    # Distance from each grid point to each player
    diffs = grid_pts[:, None, :] - player_arr[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    nearest = np.argmin(dists, axis=1)
    assigned_teams = team_arr[nearest]

    total = len(assigned_teams)
    control: dict[int, float] = {}
    for tid in positions:
        control[tid] = float(np.sum(assigned_teams == tid)) / total
    return control


class PressureStats:
    """Accumulates pressure & pitch control statistics."""

    def __init__(self, config: dict[str, Any]) -> None:
        stats_cfg = config.get("stats", {})
        pitch_cfg = config.get("pitch", {})

        self.pressure_radius: float = stats_cfg.get("pressure_radius", 5.0)
        self.pitch_length: float = pitch_cfg.get("length", 105.0)
        self.pitch_width: float = pitch_cfg.get("width", 68.0)

        self._control_history: dict[int, list[float]] = defaultdict(list)
        self._pressure_history: list[float] = []

    def update(self, frame_state: FrameState) -> FrameState:
        """Compute pressure & control for one frame."""
        if frame_state.flag != FrameFlag.IN_PLAY:
            return frame_state

        per_frame: dict[str, Any] = {}

        # Collect positions by team
        team_positions: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for player in frame_state.players:
            if player.pitch_pos is None or player.team_id is None or player.team_id < 0:
                continue
            team_positions[player.team_id].append(
                (player.pitch_pos.x, player.pitch_pos.y)
            )

        # Pitch control
        if team_positions:
            control = _voronoi_control(
                team_positions, self.pitch_length, self.pitch_width,
            )
            per_frame["pitch_control"] = {
                str(k): round(v, 3) for k, v in control.items()
            }
            for k, v in control.items():
                self._control_history[k].append(v)

        # Pressure index on ball carrier
        pressure_index = self._compute_pressure(frame_state, team_positions)
        if pressure_index is not None:
            per_frame["pressure_index"] = round(pressure_index, 2)
            self._pressure_history.append(pressure_index)

        frame_state.analytics["pressure"] = per_frame
        return frame_state

    def _compute_pressure(
        self,
        frame_state: FrameState,
        team_positions: dict[int, list[tuple[float, float]]],
    ) -> Optional[float]:
        """Pressure = number of opponents within radius of ball, weighted by closing speed."""
        if not frame_state.ball or not frame_state.ball.pitch_pos:
            return None

        bx = frame_state.ball.pitch_pos.x
        by = frame_state.ball.pitch_pos.y

        # Find nearest team to ball (ball carrier's team)
        carrier_team: Optional[int] = None
        min_dist = float("inf")
        for player in frame_state.players:
            if player.pitch_pos is None or player.team_id is None or player.team_id < 0:
                continue
            d = ((player.pitch_pos.x - bx) ** 2 + (player.pitch_pos.y - by) ** 2) ** 0.5
            if d < min_dist:
                min_dist = d
                carrier_team = player.team_id

        if carrier_team is None:
            return None

        # Count opponents within pressure radius
        pressure = 0.0
        for player in frame_state.players:
            if player.pitch_pos is None or player.team_id is None:
                continue
            if player.team_id == carrier_team or player.team_id < 0:
                continue
            d = ((player.pitch_pos.x - bx) ** 2 + (player.pitch_pos.y - by) ** 2) ** 0.5
            if d <= self.pressure_radius:
                # Weight by inverse distance (closer = more pressure)
                weight = 1.0 - (d / self.pressure_radius)
                # Bonus for closing speed
                if player.speed_mps and player.speed_mps > 0:
                    weight += min(player.speed_mps / 10.0, 0.5)
                pressure += weight

        return pressure

    def get_team_summary(self) -> dict[str, Any]:
        """Aggregate pressure & control stats."""
        result: dict[str, Any] = {
            "avg_pressure_index": round(
                float(np.mean(self._pressure_history)), 2
            ) if self._pressure_history else 0.0,
        }
        for tid, vals in self._control_history.items():
            result[f"team_{tid}_avg_control"] = round(float(np.mean(vals)), 3)
        return result
