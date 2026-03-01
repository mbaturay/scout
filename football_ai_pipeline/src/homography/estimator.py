"""FR6 — Homography Estimation: compute pixel↔pitch transform.

Uses detected keypoints matched to a canonical pitch model to
compute a homography via DLT or RANSAC.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

from ..data_models import FrameState, HomographyResult, Keypoint

logger = logging.getLogger(__name__)

# Canonical pitch keypoints (subset) in metres — FIFA 105x68 pitch
# These represent common landmarks: corners, penalty spots, center, etc.
CANONICAL_PITCH_POINTS: dict[str, tuple[float, float]] = {
    "top_left": (0.0, 0.0),
    "top_right": (105.0, 0.0),
    "bottom_left": (0.0, 68.0),
    "bottom_right": (105.0, 68.0),
    "center": (52.5, 34.0),
    "left_penalty": (11.0, 34.0),
    "right_penalty": (94.0, 34.0),
    "top_center": (52.5, 0.0),
    "bottom_center": (52.5, 68.0),
    "left_goal_top": (0.0, 24.84),
    "left_goal_bottom": (0.0, 43.16),
    "right_goal_top": (105.0, 24.84),
    "right_goal_bottom": (105.0, 43.16),
}


class HomographyEstimator:
    """Estimate frame-to-pitch homography from detected keypoints."""

    def __init__(self, config: dict[str, Any]) -> None:
        hom_cfg = config.get("homography", {})
        self.method: str = hom_cfg.get("method", "dlt")
        self.min_points: int = hom_cfg.get("min_points", 4)
        self.quality_threshold: float = hom_cfg.get("quality_threshold", 0.6)
        self.smoothing_window: int = hom_cfg.get("smoothing_window", 5)

        pitch_cfg = config.get("pitch", {})
        self.pitch_length: float = pitch_cfg.get("length", 105.0)
        self.pitch_width: float = pitch_cfg.get("width", 68.0)

        self._history: list[Optional[np.ndarray]] = []

    def estimate(self, frame_state: FrameState) -> FrameState:
        """Compute homography from keypoints and set frame_state.homography."""
        kps = frame_state.keypoints
        if len(kps) < self.min_points:
            frame_state.homography = HomographyResult(
                available=False,
                quality=0.0,
            )
            self._history.append(None)
            # Try to use smoothed fallback
            fallback = self._get_smoothed()
            if fallback is not None:
                frame_state.homography = HomographyResult(
                    matrix=fallback,
                    quality=0.3,
                    available=True,
                )
            return frame_state

        # Build correspondences: image keypoints → nearest canonical points
        src_pts, dst_pts = self._match_keypoints(kps)
        if len(src_pts) < self.min_points:
            frame_state.homography = HomographyResult(available=False, quality=0.0)
            self._history.append(None)
            return frame_state

        src = np.array(src_pts, dtype=np.float64)
        dst = np.array(dst_pts, dtype=np.float64)

        if self.method == "ransac":
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            inliers = int(mask.sum()) if mask is not None else 0
        else:
            H, mask = cv2.findHomography(src, dst, 0)
            inliers = len(src_pts)

        if H is None:
            frame_state.homography = HomographyResult(available=False, quality=0.0)
            self._history.append(None)
            return frame_state

        quality = min(1.0, inliers / max(self.min_points, 1))
        self._history.append(H)

        # Temporal smoothing
        smoothed = self._get_smoothed()
        if smoothed is not None and quality < self.quality_threshold:
            H = smoothed
            quality = max(quality, 0.4)

        frame_state.homography = HomographyResult(
            matrix=H,
            quality=quality,
            num_inliers=inliers,
            available=True,
        )
        return frame_state

    def _match_keypoints(
        self, kps: list[Keypoint],
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Naive nearest-canonical matching for heuristic keypoints.

        In a production system this would use learned descriptors.
        Here we simply assign each detected keypoint to the nearest
        canonical point based on normalised image position.
        """
        canonical = list(CANONICAL_PITCH_POINTS.values())
        src: list[list[float]] = []
        dst: list[list[float]] = []

        # Simple greedy assignment (for heuristic detector)
        used: set[int] = set()
        for kp in sorted(kps, key=lambda k: -k.confidence):
            if len(src) >= 12:  # cap matches
                break
            best_idx = -1
            best_dist = float("inf")
            for ci, (cx, cy) in enumerate(canonical):
                if ci in used:
                    continue
                # normalise: assume image maps roughly to pitch
                dist = ((kp.x / 1920 * self.pitch_length - cx) ** 2
                        + (kp.y / 1080 * self.pitch_width - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = ci
            if best_idx >= 0 and best_dist < 40:
                src.append([kp.x, kp.y])
                dst.append(list(canonical[best_idx]))
                used.add(best_idx)
        return src, dst

    def _get_smoothed(self) -> Optional[np.ndarray]:
        """Return temporally smoothed homography from recent history."""
        valid = [h for h in self._history[-self.smoothing_window:] if h is not None]
        if not valid:
            return None
        return np.mean(valid, axis=0)

    def pixel_to_pitch(
        self, H: np.ndarray, px: float, py: float,
    ) -> tuple[float, float]:
        """Transform a pixel coordinate to pitch coordinates."""
        pt = np.array([px, py, 1.0])
        mapped = H @ pt
        if abs(mapped[2]) < 1e-8:
            return (0.0, 0.0)
        return (float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2]))
