"""FR7 — Pitch-Mapped Positions: convert pixel positions to pitch coords.

Smooths ball trajectory, interpolates short gaps, computes speeds.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from scipy.ndimage import median_filter

from ..analytics.motion_smoothing import PositionSmoother
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
        self._player_history: dict[int, list[tuple[float, float]]] = {}
        self._player_timestamps: dict[int, list[float]] = {}
        self._timestamps: list[float] = []

        # EMA position smoother
        acfg = config.get("analytics", {})
        smoothing_alpha: float = acfg.get("smoothing_alpha", 0.35)
        self._smoother = PositionSmoother(alpha=smoothing_alpha)

        # Glitch filter: max plausible displacement per frame
        self._max_displacement_m: float = tf_cfg.get("max_displacement_m", 4.0)
        self._speed_cap_mps: float = tf_cfg.get("speed_cap_mps", 12.0)
        self._max_dt: float = 0.5  # skip segments with dt > this

        # Camera-motion stabilization (cumulative pixel offset)
        self._cumulative_dx: float = 0.0
        self._cumulative_dy: float = 0.0
        self._stabilize: bool = tf_cfg.get("stabilize_with_camera_motion", True)

        # Diagnostics (logged once at end)
        self._raw_speeds: list[float] = []
        self._displacements: list[float] = []
        self._dts: list[float] = []
        self._glitch_count: int = 0
        self._max_speed_observed: float = 0.0  # pre-filter max
        self._max_speed_kept: float = 0.0  # post-filter max

    def transform(self, frame_state: FrameState) -> FrameState:
        """Map all positions to pitch coordinates and compute speeds."""
        H = frame_state.homography.matrix
        available = frame_state.homography.available
        quality = frame_state.homography.quality
        self._timestamps.append(frame_state.timestamp_sec)

        # Accumulate camera motion for pixel stabilization
        self._update_cumulative_camera_motion(frame_state)

        # Players
        for player in frame_state.players:
            if available and H is not None:
                px, py = player.detection.bbox.bottom_center
                px, py = self._stabilize_pixel(px, py)
                raw_x, raw_y = self._pixel_to_pitch(H, px, py)
                if self._in_bounds(raw_x, raw_y):
                    # EMA-smooth the pitch position
                    tid = player.track_id
                    sm_x, sm_y = self._smoother.smooth(tid, raw_x, raw_y)
                    player.pitch_pos = PitchPosition(
                        x=sm_x, y=sm_y, confidence=quality,
                    )
                    # Track history for speed (per-player timestamps)
                    if tid not in self._player_history:
                        self._player_history[tid] = []
                        self._player_timestamps[tid] = []
                    player.speed_mps = self._compute_speed_safe(
                        self._player_history[tid],
                        self._player_timestamps[tid],
                        (sm_x, sm_y),
                        frame_state.timestamp_sec,
                    )
                else:
                    player.pitch_pos = None
            else:
                player.pitch_pos = None

        # Ball
        if frame_state.ball and frame_state.ball.detection and available and H is not None:
            bx, by = frame_state.ball.detection.bbox.center
            bx, by = self._stabilize_pixel(bx, by)
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

        # Ball speed (use last two valid ball entries with their timestamps)
        if frame_state.ball and frame_state.ball.pitch_pos:
            # Find the two most recent valid ball positions and their timestamps
            ball_ts = self._timestamps
            valid_pairs: list[tuple[tuple[float, float], float]] = []
            for k in range(len(self._ball_history) - 1, -1, -1):
                if self._ball_history[k] is not None and k < len(ball_ts):
                    valid_pairs.append((self._ball_history[k], ball_ts[k]))
                if len(valid_pairs) == 2:
                    break
            if len(valid_pairs) == 2:
                (p2, t2), (p1, t1) = valid_pairs[0], valid_pairs[1]
                dt = t2 - t1
                if dt > 0:
                    dist = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
                    raw = dist / dt
                    frame_state.ball.speed_mps = min(raw, 50.0)  # ball can go fast

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

    def _update_cumulative_camera_motion(self, frame_state: FrameState) -> None:
        """Accumulate smoothed camera-motion translation from the current frame."""
        if not self._stabilize:
            return
        cam = frame_state.analytics.get("camera_motion")
        if cam is None:
            return
        # Use smoothed per-frame delta (already EMA-filtered by CameraMotionEstimator)
        self._cumulative_dx += cam.get("smoothed_dx_px", 0.0)
        self._cumulative_dy += cam.get("smoothed_dy_px", 0.0)

    def _stabilize_pixel(self, px: float, py: float) -> tuple[float, float]:
        """Subtract cumulative camera translation from pixel coordinates."""
        if not self._stabilize:
            return (px, py)
        return (px - self._cumulative_dx, py - self._cumulative_dy)

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

    def _compute_speed_safe(
        self,
        history: list[tuple[float, float]],
        ts_history: list[float],
        current_pos: tuple[float, float],
        current_ts: float,
    ) -> Optional[float]:
        """Compute speed from per-entity position/timestamp history.

        Filters tracking glitches and caps at a plausible maximum.
        Appends current_pos/current_ts to the history lists.
        """
        speed: Optional[float] = None

        if history and ts_history:
            prev = history[-1]
            dt = current_ts - ts_history[-1]
            self._dts.append(dt)

            if dt > 0 and dt <= self._max_dt:
                dx = current_pos[0] - prev[0]
                dy = current_pos[1] - prev[1]
                dist = (dx * dx + dy * dy) ** 0.5
                self._displacements.append(dist)

                if dist > self._max_displacement_m:
                    # Tracking glitch — skip this segment
                    self._glitch_count += 1
                else:
                    raw = dist / dt
                    self._raw_speeds.append(raw)
                    # Track pre-filter max
                    if raw > self._max_speed_observed:
                        self._max_speed_observed = raw
                    if raw > self._speed_cap_mps:
                        speed = self._speed_cap_mps
                        self._glitch_count += 1
                    else:
                        speed = raw
                    # Track post-filter max
                    if speed is not None and speed > self._max_speed_kept:
                        self._max_speed_kept = speed

        history.append(current_pos)
        ts_history.append(current_ts)
        return speed

    def log_diagnostics(self) -> None:
        """Log motion physics diagnostics. Call once after all frames."""
        if not self._dts:
            return
        avg_dt = sum(self._dts) / len(self._dts)
        avg_disp = sum(self._displacements) / len(self._displacements) if self._displacements else 0
        top_raw = max(self._raw_speeds) if self._raw_speeds else 0
        avg_raw = sum(self._raw_speeds) / len(self._raw_speeds) if self._raw_speeds else 0
        logger.info(
            "Motion diagnostics: avg_dt=%.4fs, avg_displacement=%.3fm, "
            "top_raw_speed=%.2f m/s, avg_raw_speed=%.2f m/s, "
            "glitches_filtered=%d, speed_cap=%.1f m/s",
            avg_dt, avg_disp, top_raw, avg_raw,
            self._glitch_count, self._speed_cap_mps,
        )

    def get_motion_diagnostics(self) -> dict[str, Any]:
        """Return motion diagnostics dict for inclusion in run_report."""
        avg_fps = 0.0
        if self._dts:
            valid_dts = [dt for dt in self._dts if 0 < dt <= self._max_dt]
            if valid_dts:
                avg_dt = sum(valid_dts) / len(valid_dts)
                avg_fps = round(1.0 / avg_dt, 1) if avg_dt > 0 else 0.0
        return {
            "avg_fps_estimate": avg_fps,
            "motion_glitch_count": self._glitch_count,
            "max_speed_mps_observed": round(self._max_speed_observed, 2),
            "max_speed_mps_kept": round(self._max_speed_kept, 2),
        }
