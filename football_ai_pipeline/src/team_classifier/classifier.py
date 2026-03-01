"""FR5 — Team Classification: assign team_id to each player.

Uses K-Means clustering on jersey colour features extracted from the
bounding-box interior. Goalkeeper assignment uses a positional heuristic.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans

from ..data_models import FrameState, ObjectClass, PlayerState

logger = logging.getLogger(__name__)


def _extract_color_features(
    image: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    color_space: str = "hsv",
) -> np.ndarray:
    """Extract a colour histogram feature from the central crop of a bbox."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (
        max(0, int(bbox_xyxy[0])),
        max(0, int(bbox_xyxy[1])),
        min(w, int(bbox_xyxy[2])),
        min(h, int(bbox_xyxy[3])),
    )
    # Use central 60% to reduce background contamination
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cw, ch = int((x2 - x1) * 0.3), int((y2 - y1) * 0.3)
    crop = image[max(0, cy - ch):cy + ch, max(0, cx - cw):cx + cw]
    if crop.size == 0:
        return np.zeros(48)

    if color_space == "hsv":
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    elif color_space == "lab":
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

    # 16-bin histogram per channel
    features = []
    for ch_i in range(3):
        hist = cv2.calcHist([crop], [ch_i], None, [16], [0, 256])
        hist = hist.flatten()
        total = hist.sum()
        if total > 0:
            hist = hist / total
        features.append(hist)
    return np.concatenate(features)


class TeamClassifier:
    """Classify players into two teams based on jersey colour clustering."""

    def __init__(self, config: dict[str, Any]) -> None:
        tc_cfg = config.get("team_classifier", {})
        self.n_clusters: int = tc_cfg.get("n_clusters", 2)
        self.color_space: str = tc_cfg.get("color_space", "hsv")
        self.sample_frames: int = tc_cfg.get("sample_frames", 50)
        self.gk_heuristic: bool = tc_cfg.get("goalkeeper_heuristic", True)

        self._kmeans: Optional[KMeans] = None
        self._feature_buffer: list[np.ndarray] = []
        self._frames_seen: int = 0
        self._fitted: bool = False

    def update(self, frame_state: FrameState) -> FrameState:
        """Classify players in a single frame. Collects samples until fit."""
        if frame_state.image is None:
            return frame_state

        # Extract features for all player detections
        for player in frame_state.players:
            if player.detection.class_id in (ObjectClass.REFEREE,):
                continue
            feat = _extract_color_features(
                frame_state.image,
                (player.detection.bbox.x1, player.detection.bbox.y1,
                 player.detection.bbox.x2, player.detection.bbox.y2),
                self.color_space,
            )
            player.detection.color_features = feat.tolist()

            if not self._fitted:
                self._feature_buffer.append(feat)

        self._frames_seen += 1

        # Fit once we have enough samples
        if not self._fitted and self._frames_seen >= self.sample_frames:
            self._fit()

        # Predict if fitted
        if self._fitted:
            self._predict(frame_state)

        return frame_state

    def _fit(self) -> None:
        if len(self._feature_buffer) < self.n_clusters:
            logger.warning("Not enough samples to fit team classifier (%d)", len(self._feature_buffer))
            return
        X = np.array(self._feature_buffer, dtype=np.float64)
        self._kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        self._kmeans.fit(X)
        self._fitted = True
        logger.info("Team classifier fitted on %d samples", len(X))

    def _predict(self, frame_state: FrameState) -> None:
        if self._kmeans is None:
            return
        for player in frame_state.players:
            if player.detection.class_id == ObjectClass.REFEREE:
                player.team_id = -1
                continue
            feat = player.detection.color_features
            if feat is None:
                continue
            label = int(self._kmeans.predict(np.array([feat], dtype=np.float64))[0])
            player.team_id = label
            player.detection.team_id = label

    def force_fit(self) -> None:
        """Force fitting even if sample count hasn't been reached."""
        if not self._fitted and self._feature_buffer:
            self._fit()
