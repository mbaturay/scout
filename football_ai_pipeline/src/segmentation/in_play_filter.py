"""FR2 — In-Play Filter: classify frames as IN_PLAY vs NOT_IN_PLAY.

Uses heuristic scene-change detection (large pixel difference = camera cut)
and green-pitch ratio to determine if the frame shows active play.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

from ..data_models import FrameFlag, FrameState

logger = logging.getLogger(__name__)


class InPlayFilter:
    """Heuristic in-play / not-in-play frame classifier."""

    def __init__(self, config: dict[str, Any]) -> None:
        seg_cfg = config.get("segmentation", {})
        self.scene_change_threshold: float = seg_cfg.get("scene_change_threshold", 30.0)
        self.min_play_duration: int = seg_cfg.get("min_play_duration", 30)
        self._prev_gray: Optional[np.ndarray] = None
        self._green_low = np.array([35, 40, 40])
        self._green_high = np.array([80, 255, 255])

    def classify(self, frame_state: FrameState) -> FrameState:
        """Classify a single frame and set flag + reasons."""
        if frame_state.image is None:
            frame_state.flag = FrameFlag.NOT_IN_PLAY
            frame_state.flag_reasons.append("no_image")
            return frame_state

        reasons: list[str] = []
        gray = cv2.cvtColor(frame_state.image, cv2.COLOR_BGR2GRAY)

        # Scene-change detection
        if self._prev_gray is not None:
            diff = cv2.absdiff(gray, self._prev_gray)
            mean_diff = float(np.mean(diff))
            if mean_diff > self.scene_change_threshold:
                reasons.append(f"scene_change(diff={mean_diff:.1f})")
        self._prev_gray = gray

        # Green-pitch ratio
        hsv = cv2.cvtColor(frame_state.image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._green_low, self._green_high)
        green_ratio = float(np.count_nonzero(mask)) / mask.size
        if green_ratio < 0.15:
            reasons.append(f"low_green({green_ratio:.2f})")

        if reasons:
            frame_state.flag = FrameFlag.NOT_IN_PLAY
            frame_state.flag_reasons = reasons
        else:
            frame_state.flag = FrameFlag.IN_PLAY

        return frame_state

    def reset(self) -> None:
        self._prev_gray = None
