"""FR9 — Visualization: annotated video with IDs, team colors, ball marker."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from ..data_models import FrameFlag, FrameState, ObjectClass

logger = logging.getLogger(__name__)

TEAM_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 100, 100),   # Blue-ish (BGR)
    1: (100, 100, 255),   # Red-ish (BGR)
    -1: (0, 255, 255),    # Yellow for referee
}

BALL_COLOR = (0, 255, 0)  # Green


class FrameAnnotator:
    """Draw annotations on frames."""

    def __init__(self, config: dict[str, Any]) -> None:
        viz_cfg = config.get("visualization", {})
        self.draw_ids: bool = viz_cfg.get("draw_ids", True)
        self.draw_ball: bool = viz_cfg.get("draw_ball", True)
        self.draw_team_colors: bool = viz_cfg.get("draw_team_colors", True)
        self.radar_overlay: bool = viz_cfg.get("radar_overlay", False)
        self.enabled: bool = viz_cfg.get("enabled", True)

        pitch_cfg = config.get("pitch", {})
        self.pitch_length: float = pitch_cfg.get("length", 105.0)
        self.pitch_width: float = pitch_cfg.get("width", 68.0)

    def annotate(self, frame_state: FrameState) -> np.ndarray | None:
        """Annotate a frame and return the annotated image."""
        if not self.enabled or frame_state.image is None:
            return frame_state.image

        img = frame_state.image.copy()

        # Draw not-in-play indicator
        if frame_state.flag == FrameFlag.NOT_IN_PLAY:
            cv2.putText(
                img, "NOT IN PLAY", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
            )

        # Draw players
        for player in frame_state.players:
            bbox = player.detection.bbox
            team_id = player.team_id if player.team_id is not None else -1
            color = TEAM_COLORS.get(team_id, (200, 200, 200))

            if self.draw_team_colors:
                cv2.rectangle(
                    img,
                    (int(bbox.x1), int(bbox.y1)),
                    (int(bbox.x2), int(bbox.y2)),
                    color, 2,
                )

            if self.draw_ids:
                label = f"#{player.track_id}"
                if player.speed_mps is not None:
                    label += f" {player.speed_mps:.1f}m/s"
                cv2.putText(
                    img, label,
                    (int(bbox.x1), int(bbox.y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                )

        # Draw ball
        if self.draw_ball and frame_state.ball and frame_state.ball.detection:
            bbox = frame_state.ball.detection.bbox
            cx, cy = int(bbox.center[0]), int(bbox.center[1])
            radius = max(5, int(bbox.width / 2))
            cv2.circle(img, (cx, cy), radius, BALL_COLOR, 2)
            cv2.circle(img, (cx, cy), 3, BALL_COLOR, -1)

        # Optional radar overlay
        if self.radar_overlay:
            radar = self._draw_radar(frame_state)
            h, w = radar.shape[:2]
            img[10:10 + h, img.shape[1] - w - 10:img.shape[1] - 10] = radar

        # Frame info
        info = f"F:{frame_state.frame_idx} T:{frame_state.timestamp_sec:.1f}s"
        cv2.putText(
            img, info, (10, img.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        return img

    def _draw_radar(self, frame_state: FrameState) -> np.ndarray:
        """Draw a minimap radar overlay."""
        rw, rh = 200, 130
        radar = np.zeros((rh, rw, 3), dtype=np.uint8)
        radar[:] = (30, 80, 30)  # Dark green

        # Pitch outline
        cv2.rectangle(radar, (2, 2), (rw - 3, rh - 3), (255, 255, 255), 1)
        # Center line
        cv2.line(radar, (rw // 2, 2), (rw // 2, rh - 3), (255, 255, 255), 1)

        scale_x = (rw - 6) / self.pitch_length
        scale_y = (rh - 6) / self.pitch_width

        for player in frame_state.players:
            if player.pitch_pos is None:
                continue
            px = int(3 + player.pitch_pos.x * scale_x)
            py = int(3 + player.pitch_pos.y * scale_y)
            team_id = player.team_id if player.team_id is not None else -1
            color = TEAM_COLORS.get(team_id, (200, 200, 200))
            cv2.circle(radar, (px, py), 3, color, -1)

        if frame_state.ball and frame_state.ball.pitch_pos:
            bx = int(3 + frame_state.ball.pitch_pos.x * scale_x)
            by = int(3 + frame_state.ball.pitch_pos.y * scale_y)
            cv2.circle(radar, (bx, by), 4, BALL_COLOR, -1)

        return radar
