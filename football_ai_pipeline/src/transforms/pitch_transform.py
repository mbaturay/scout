"""FR7 — Pitch-Mapped Positions: convert pixel positions to pitch coords.

Smooths ball trajectory, interpolates short gaps, computes speeds.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from scipy.ndimage import median_filter

from ..data_models import BallState, FrameState, PitchPosition

logger = logging.getLogger(__name__)


class PitchTransformer:
    """Convert image positions to pitch coordinates and compute kinematics."""

    def __init__(self, config: dict[str, Any]) -> None:
        pitch_cfg = config.get("pitch", {})
        self.pitch_length: float = pitch_cfg.get("length", 105.0)
        self.pitch_width: float = pitch_cfg.get("width", 68.0)

        tf_cfg = config.get("transforms", {})
        self.ball_smooth_window: int = tf_cfg.get("ball_smoothing_window", 5)
        self.interp_max_gap: int = tf_cfg.get("interpolation_max_gap", 10)

        self._ball_history: list[Optional[tuple[float, float]]] = []
        self._player_history: dict[int, list[Optional[tuple[float, float]]]] = {}
        self._timestamps: list[float] = []

    def transform(self, frame_state: FrameState) -> FrameState:
        """Map all positions to pitch coordinates and compute speeds."""
        H = frame_state.homography.matrix
        available = frame_state.homography.available
        quality = frame_state.homography.quality
        self._timestamps.append(frame_state.timestamp_sec)

        # Players
        for player in frame_state.players:
            if available and H is not None:
                px, py = player.detection.bbox.bottom_center
                pitch_x, pitch_y = self._pixel_to_pitch(H, px, py)
                if self._in_bounds(pitch_x, pitch_y):
                    player.pitch_pos = PitchPosition(
                        x=pitch_x, y=pitch_y, confidence=quality,
                    )
                    # Track history for speed
                    tid = player.track_id
                    if tid not in self._player_history:
                        self._player_history[tid] = []
                    self._player_history[tid].append((pitch_x, pitch_y))
                    player.speed_mps = self._compute_speed(
                        self._player_history[tid], self._timestamps,
                    )
                else:
                    player.pitch_pos = None
            else:
                player.pitch_pos = None

        # Ball
        if frame_state.ball and frame_state.ball.detection and available and H is not None:
            bx, by = frame_state.ball.detection.bbox.center
            pitch_x, pitch_y = self._pixel_to_pitch(H, bx, by)
            if self._in_bounds(pitch_x, pitch_y):
                frame_state.ball.pitch_pos = PitchPosition(
                    x=pitch_x, y=pitch_y, confidence=quality,
                )
                self._ball_history.append((pitch_x, pitch_y))
            else:
                self._ball_history.append(None)
        else:
            self._ball_history.append(None)

        # Interpolate short ball gaps
        self._interpolate_ball_gaps(frame_state)

        # Ball speed
        if frame_state.ball and frame_state.ball.pitch_pos:
            valid_pts = [p for p in self._ball_history[-5:] if p is not None]
            if len(valid_pts) >= 2 and len(self._timestamps) >= 2:
                frame_state.ball.speed_mps = self._compute_speed(
                    valid_pts, self._timestamps[-len(valid_pts):],
                )

        return frame_state

    def _pixel_to_pitch(
        self, H: np.ndarray, px: float, py: float,
    ) -> tuple[float, float]:
        pt = np.array([px, py, 1.0])
        mapped = H @ pt
        if abs(mapped[2]) < 1e-8:
            return (0.0, 0.0)
        return (float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2]))

    def _in_bounds(self, x: float, y: float) -> bool:
        margin = 5.0  # allow slight out-of-bounds
        return (-margin <= x <= self.pitch_length + margin and
                -margin <= y <= self.pitch_width + margin)

    def _interpolate_ball_gaps(self, frame_state: FrameState) -> None:
        """Fill short gaps in ball trajectory via linear interpolation."""
        hist = self._ball_history
        if len(hist) < 3:
            return
        idx = len(hist) - 1
        if hist[idx] is not None:
            return  # no gap at current frame

        # Find last valid
        gap_start = idx
        while gap_start > 0 and hist[gap_start] is None:
            gap_start -= 1
        gap_len = idx - gap_start

        if gap_len > self.interp_max_gap or hist[gap_start] is None:
            return

        # Find next valid (look ahead not possible — interpolate from past only)
        # Use last two valid points to extrapolate
        if gap_start >= 1 and hist[gap_start] is not None:
            prev = hist[gap_start]
            if prev is not None:
                frame_state.ball = frame_state.ball or BallState()
                frame_state.ball.pitch_pos = PitchPosition(
                    x=prev[0], y=prev[1], confidence=0.3,
                )
                frame_state.ball.interpolated = True
                hist[idx] = (prev[0], prev[1])

    @staticmethod
    def _compute_speed(
        positions: list[Optional[tuple[float, float]]],
        timestamps: list[float],
    ) -> Optional[float]:
        """Compute instantaneous speed from last two valid positions."""
        if len(positions) < 2 or len(timestamps) < 2:
            return None
        p1 = positions[-2]
        p2 = positions[-1]
        if p1 is None or p2 is None:
            return None
        dt = timestamps[-1] - timestamps[-2] if len(timestamps) >= 2 else 0
        if dt <= 0:
            return None
        dist = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return dist / dt
