"""Stats B — Spatial & Tactical Shape analytics.

Computes:
  - Team centroid over time (attack/defense phases)
  - Team width/length and compactness (per minute + rolling window)
  - Defensive line height estimate (per team, per window)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from ..data_models import FrameFlag, FrameState

logger = logging.getLogger(__name__)


class SpatialStats:
    """Accumulates spatial / tactical shape statistics."""

    def __init__(self, config: dict[str, Any]) -> None:
        stats_cfg = config.get("stats", {})
        self.rolling_window_sec: float = stats_cfg.get("rolling_window_sec", 300)

        pitch_cfg = config.get("pitch", {})
        self.pitch_length: float = pitch_cfg.get("length", 105.0)
        self.pitch_width: float = pitch_cfg.get("width", 68.0)

        # Time series
        self._centroids: dict[int, list[tuple[float, float, float]]] = defaultdict(list)  # team -> [(ts, x, y)]
        self._widths: dict[int, list[tuple[float, float]]] = defaultdict(list)  # team -> [(ts, width)]
        self._lengths: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self._compactness: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self._def_line: dict[int, list[tuple[float, float]]] = defaultdict(list)

    def update(self, frame_state: FrameState) -> FrameState:
        """Compute spatial stats for one frame."""
        if frame_state.flag != FrameFlag.IN_PLAY:
            return frame_state

        ts = frame_state.timestamp_sec
        team_positions: dict[int, list[tuple[float, float]]] = defaultdict(list)

        for player in frame_state.players:
            if player.pitch_pos is None or player.team_id is None or player.team_id < 0:
                continue
            team_positions[player.team_id].append(
                (player.pitch_pos.x, player.pitch_pos.y)
            )

        per_frame: dict[str, Any] = {}

        for team_id, positions in team_positions.items():
            if len(positions) < 2:
                continue
            arr = np.array(positions)
            cx, cy = float(arr[:, 0].mean()), float(arr[:, 1].mean())
            self._centroids[team_id].append((ts, cx, cy))

            # Width = max lateral spread, Length = max longitudinal spread
            width = float(arr[:, 1].max() - arr[:, 1].min())
            length = float(arr[:, 0].max() - arr[:, 0].min())
            self._widths[team_id].append((ts, width))
            self._lengths[team_id].append((ts, length))

            # Compactness = convex hull area (simplified as width * length)
            compactness = width * length
            self._compactness[team_id].append((ts, compactness))

            # Defensive line height: average x of the 4 deepest outfield players
            sorted_x = sorted(arr[:, 0])
            def_line_x = float(np.mean(sorted_x[:min(4, len(sorted_x))]))
            self._def_line[team_id].append((ts, def_line_x))

            per_frame[f"team_{team_id}"] = {
                "centroid": [round(cx, 2), round(cy, 2)],
                "width": round(width, 2),
                "length": round(length, 2),
                "compactness": round(compactness, 2),
                "defensive_line_x": round(def_line_x, 2),
            }

        frame_state.analytics["spatial"] = per_frame
        return frame_state

    def get_team_summary(self) -> dict[int, dict[str, Any]]:
        """Per-team aggregate spatial stats."""
        result: dict[int, dict[str, Any]] = {}
        for team_id in self._centroids:
            centroids = self._centroids[team_id]
            widths = [w for _, w in self._widths[team_id]]
            lengths = [l for _, l in self._lengths[team_id]]
            compactness = [c for _, c in self._compactness[team_id]]
            def_lines = [d for _, d in self._def_line[team_id]]

            result[team_id] = {
                "avg_centroid_x": round(float(np.mean([c[1] for c in centroids])), 2),
                "avg_centroid_y": round(float(np.mean([c[2] for c in centroids])), 2),
                "avg_width": round(float(np.mean(widths)), 2) if widths else 0.0,
                "avg_length": round(float(np.mean(lengths)), 2) if lengths else 0.0,
                "avg_compactness": round(float(np.mean(compactness)), 2) if compactness else 0.0,
                "avg_defensive_line_x": round(float(np.mean(def_lines)), 2) if def_lines else 0.0,
            }
        return result

    def get_rolling_summary(self) -> dict[int, list[dict[str, Any]]]:
        """Per-team rolling-window summaries."""
        result: dict[int, list[dict[str, Any]]] = {}
        for team_id in self._centroids:
            windows: list[dict[str, Any]] = []
            centroids = self._centroids[team_id]
            if not centroids:
                continue
            start_ts = centroids[0][0]
            end_ts = centroids[-1][0]
            t = start_ts
            while t < end_ts:
                t_end = t + self.rolling_window_sec
                w_centroids = [(ts, cx, cy) for ts, cx, cy in centroids if t <= ts < t_end]
                w_widths = [w for ts, w in self._widths[team_id] if t <= ts < t_end]
                w_lengths = [l for ts, l in self._lengths[team_id] if t <= ts < t_end]
                w_def = [d for ts, d in self._def_line[team_id] if t <= ts < t_end]

                if w_centroids:
                    windows.append({
                        "window_start_sec": round(t, 1),
                        "window_end_sec": round(t_end, 1),
                        "avg_centroid_x": round(float(np.mean([c[1] for c in w_centroids])), 2),
                        "avg_centroid_y": round(float(np.mean([c[2] for c in w_centroids])), 2),
                        "avg_width": round(float(np.mean(w_widths)), 2) if w_widths else 0.0,
                        "avg_length": round(float(np.mean(w_lengths)), 2) if w_lengths else 0.0,
                        "avg_defensive_line_x": round(float(np.mean(w_def)), 2) if w_def else 0.0,
                    })
                t = t_end
            result[team_id] = windows
        return result
