"""Motion smoothing — EMA filter for per-track pitch positions.

Applies an Exponential Moving Average to (x, y) pitch coordinates
before speed / distance computation, reducing jitter from noisy
homography or detection wobble.

    x_s[t] = α·x[t] + (1 − α)·x_s[t−1]

A higher α (closer to 1) trusts the raw measurement more; a lower α
produces heavier smoothing.  Default α = 0.35 balances responsiveness
with noise suppression for typical 15-30 fps football footage.
"""

from __future__ import annotations

from typing import Optional


class PositionSmoother:
    """Per-track EMA smoother for 2-D pitch coordinates."""

    def __init__(self, alpha: float = 0.35) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._state: dict[int, tuple[float, float]] = {}

    @property
    def alpha(self) -> float:
        return self._alpha

    def smooth(
        self,
        track_id: int,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        """Return EMA-smoothed (x, y) for *track_id*.

        On the first observation for a track the raw value is returned
        (the filter needs one sample to initialise).
        """
        prev = self._state.get(track_id)
        if prev is None:
            self._state[track_id] = (x, y)
            return (x, y)

        a = self._alpha
        sx = a * x + (1.0 - a) * prev[0]
        sy = a * y + (1.0 - a) * prev[1]
        self._state[track_id] = (sx, sy)
        return (sx, sy)

    def reset(self, track_id: int) -> None:
        """Drop state for a single track (e.g. after a long gap)."""
        self._state.pop(track_id, None)

    def reset_all(self) -> None:
        """Drop state for every track."""
        self._state.clear()

    def peek(self, track_id: int) -> Optional[tuple[float, float]]:
        """Return the last smoothed position without updating state."""
        return self._state.get(track_id)
