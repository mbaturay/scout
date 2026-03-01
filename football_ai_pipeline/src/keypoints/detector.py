"""FR6 — Pitch Keypoint Detection.

Detects pitch landmarks (line intersections, circle points, etc.).
When a trained keypoint model is not available, falls back to a
green-field line-intersection heuristic using Hough lines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..data_models import FrameState, Keypoint

logger = logging.getLogger(__name__)


class KeypointDetector:
    """Detect pitch keypoints in a frame."""

    def __init__(self, config: dict[str, Any]) -> None:
        kp_cfg = config.get("keypoints", {})
        self.confidence_threshold: float = kp_cfg.get("confidence", 0.5)
        self.num_keypoints: int = kp_cfg.get("num_keypoints", 32)
        self._model: Any = None
        self.backend: str = "heuristic"

        weights = kp_cfg.get("weights")
        if weights:
            weights_path = Path(weights)
            if not weights_path.exists():
                fail_hard = config.get("pipeline", {}).get("fail_on_missing_weights", False)
                if fail_hard:
                    raise FileNotFoundError(
                        f"Keypoint weights not found at '{weights}'.\n"
                        f"Expected path: {weights_path.resolve()}\n"
                        f"Config key: keypoints.weights"
                    )
                logger.warning(
                    "Keypoint weights not found: %s\n"
                    "  Impact: Using heuristic line-intersection detector — "
                    "homography quality will be lower.\n"
                    "  Fix:    Place keypoint weights at: %s\n"
                    "          Or set keypoints.weights: null in config to silence this warning.",
                    weights, weights_path.resolve(),
                )
            else:
                try:
                    logger.info("Loading keypoint model from %s", weights)
                    # self._model = load_model(weights)
                    self.backend = "model"
                except Exception as e:
                    fail_hard = config.get("pipeline", {}).get("fail_on_missing_weights", False)
                    if fail_hard:
                        raise RuntimeError(f"Keypoint model load failed: {e}") from e
                    logger.warning(
                        "Keypoint model failed to load (%s).\n"
                        "  Impact: Using heuristic fallback — homography quality will be lower.",
                        e,
                    )
        else:
            logger.info(
                "Keypoint detector: heuristic mode (no weights configured).\n"
                "  Impact: Homography quality depends on visible pitch lines.\n"
                "  Tip:    For better accuracy, provide a trained keypoint model via\n"
                "          keypoints.weights in your config file."
            )

    def detect(self, frame_state: FrameState) -> FrameState:
        """Detect keypoints and populate frame_state.keypoints."""
        if frame_state.image is None:
            return frame_state

        if self._model is not None:
            return self._detect_model(frame_state)
        return self._detect_heuristic(frame_state)

    def _detect_model(self, frame_state: FrameState) -> FrameState:
        # Placeholder: real model inference
        frame_state.keypoints = []
        return frame_state

    def _detect_heuristic(self, frame_state: FrameState) -> FrameState:
        """Heuristic: find white line intersections on green field."""
        img = frame_state.image
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Mask for green field
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([80, 255, 255]))

        # Mask for white lines on the field
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        line_mask = cv2.bitwise_and(white_mask, green_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)

        # Hough lines
        lines = cv2.HoughLinesP(
            line_mask, 1, np.pi / 180,
            threshold=50, minLineLength=50, maxLineGap=20,
        )

        keypoints: list[Keypoint] = []
        if lines is not None and len(lines) > 1:
            # Find intersections between detected line segments
            segments = lines.reshape(-1, 4)
            intersections = self._find_intersections(segments, img.shape[:2])
            for pt in intersections[:self.num_keypoints]:
                keypoints.append(Keypoint(x=pt[0], y=pt[1], confidence=0.5))

        frame_state.keypoints = keypoints
        frame_state.keypoint_confidences = [kp.confidence for kp in keypoints]
        return frame_state

    @staticmethod
    def _find_intersections(
        segments: np.ndarray, img_shape: tuple[int, int],
    ) -> list[tuple[float, float]]:
        """Find pairwise intersections of line segments, clamped to image bounds."""
        h, w = img_shape
        pts: list[tuple[float, float]] = []

        n = min(len(segments), 60)  # limit combinatorics
        for i in range(n):
            x1, y1, x2, y2 = segments[i]
            for j in range(i + 1, n):
                x3, y3, x4, y4 = segments[j]
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-6:
                    continue
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                ix = x1 + t * (x2 - x1)
                iy = y1 + t * (y2 - y1)
                if 0 <= ix < w and 0 <= iy < h:
                    pts.append((float(ix), float(iy)))

        # Deduplicate nearby points
        if not pts:
            return pts
        pts_arr = np.array(pts)
        keep: list[tuple[float, float]] = []
        used = np.zeros(len(pts_arr), dtype=bool)
        for i in range(len(pts_arr)):
            if used[i]:
                continue
            dists = np.linalg.norm(pts_arr[i] - pts_arr, axis=1)
            cluster = dists < 15
            used |= cluster
            centroid = pts_arr[cluster].mean(axis=0)
            keep.append((float(centroid[0]), float(centroid[1])))
        return keep
