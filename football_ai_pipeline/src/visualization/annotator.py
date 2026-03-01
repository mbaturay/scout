"""FR9 — Visualization: broadcast-style annotated video overlay.

Overlays per-player speed (km/h), cumulative distance, track IDs,
ball halo, owner ring, team ball control %, and optional camera debug.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

from ..data_models import FrameFlag, FrameState, ObjectClass
from ..utils.tensors import to_cpu_numpy

logger = logging.getLogger(__name__)

_ANNOTATE_LOG_EVERY = 50  # debug-log annotation stats every N frames

TEAM_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 100, 100),   # Blue-ish (BGR)
    1: (100, 100, 255),   # Red-ish (BGR)
    -1: (0, 255, 255),    # Yellow for referee
}

BALL_COLOR = (0, 255, 0)  # Green
OWNER_RING_COLOR = (0, 215, 255)  # Gold (BGR)
HALO_COLOR = (0, 255, 180)  # Bright cyan-green


class FrameAnnotator:
    """Draw broadcast-style annotations on frames."""

    def __init__(self, config: dict[str, Any]) -> None:
        viz_cfg = config.get("visualization", {})
        self.draw_ids: bool = viz_cfg.get("draw_ids", True)
        self.draw_ball: bool = viz_cfg.get("draw_ball", True)
        self.draw_team_colors: bool = viz_cfg.get("draw_team_colors", True)
        self.radar_overlay: bool = viz_cfg.get("radar_overlay", False)
        self.enabled: bool = viz_cfg.get("enabled", True)

        overlay_cfg = config.get("overlay", {})
        self.show_speed: bool = overlay_cfg.get("show_speed", True)
        self.show_distance: bool = overlay_cfg.get("show_distance", True)
        self.show_ball_control: bool = overlay_cfg.get("show_ball_control", True)
        self.show_camera_debug: bool = overlay_cfg.get("show_camera_motion_debug", False)

        pitch_cfg = config.get("pitch", {})
        self.pitch_length: float = pitch_cfg.get("length", 105.0)
        self.pitch_width: float = pitch_cfg.get("width", 68.0)

        # Cumulative distance tracker (track_id -> metres)
        self._cum_dist: dict[int, float] = {}
        self._prev_pos: dict[int, tuple[float, float]] = {}

        # Running team control counters
        self._team_control_frames: dict[int, int] = {}
        self._total_owned_frames: int = 0

        # Debug counter
        self._annotate_count: int = 0

    @staticmethod
    def _safe_int(v: Any) -> int:
        """Convert any numeric (numpy scalar, tensor, float) to Python int."""
        return int(float(v))

    def annotate(self, frame_state: FrameState) -> np.ndarray | None:
        """Annotate a frame and return the annotated image."""
        if not self.enabled or frame_state.image is None:
            return frame_state.image

        # Force a writeable uint8 numpy copy — guarantees cv2 can draw on it
        raw = to_cpu_numpy(frame_state.image)
        img = np.array(raw, dtype=np.uint8, copy=True)

        # Update cumulative distance for each player
        self._update_distances(frame_state)

        # Update ball control tracking
        self._update_ball_control(frame_state)

        # Draw not-in-play indicator
        if frame_state.flag == FrameFlag.NOT_IN_PLAY:
            cv2.putText(
                img, "NOT IN PLAY", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
            )

        # Get current ball owner for highlight
        ball_owner_id = self._get_ball_owner(frame_state)

        n_drawn = 0
        # Draw players
        for player in frame_state.players:
            try:
                bbox = player.detection.bbox
                x1 = self._safe_int(bbox.x1)
                y1 = self._safe_int(bbox.y1)
                x2 = self._safe_int(bbox.x2)
                y2 = self._safe_int(bbox.y2)
                team_id = player.team_id if player.team_id is not None else -1
                color = TEAM_COLORS.get(team_id, (200, 200, 200))
                is_owner = (ball_owner_id is not None
                            and player.track_id == ball_owner_id)

                # Owner ring — thick gold border
                if is_owner:
                    cv2.rectangle(
                        img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2),
                        OWNER_RING_COLOR, 3,
                    )

                if self.draw_team_colors:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                if self.draw_ids:
                    self._draw_player_label(img, player, color)

                n_drawn += 1
            except Exception as exc:
                logger.warning(
                    "Annotator: skipped player track=%s: %s",
                    getattr(player, "track_id", "?"), exc,
                )

        # Draw ball with halo
        if self.draw_ball and frame_state.ball and frame_state.ball.detection:
            try:
                self._draw_ball_halo(img, frame_state)
            except Exception as exc:
                logger.warning("Annotator: ball halo failed: %s", exc)

        # Optional radar overlay
        if self.radar_overlay:
            radar = self._draw_radar(frame_state)
            h, w = radar.shape[:2]
            img[10:10 + h, img.shape[1] - w - 10:img.shape[1] - 10] = radar

        # Ball control bar (bottom right)
        if self.show_ball_control:
            self._draw_ball_control(img)

        # Camera motion debug (top right)
        if self.show_camera_debug:
            self._draw_camera_debug(img, frame_state)

        # Frame info (bottom left)
        info = f"F:{frame_state.frame_idx} T:{frame_state.timestamp_sec:.1f}s"
        cv2.putText(
            img, info, (10, img.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        self._annotate_count += 1
        if self._annotate_count % _ANNOTATE_LOG_EVERY == 0:
            n_players = len(frame_state.players)
            has_ball = frame_state.ball is not None and frame_state.ball.detection is not None
            print(
                f"[Annotator] frame {frame_state.frame_idx} "
                f"drawing {n_drawn}/{n_players} boxes, ball={has_ball}",
                flush=True,
            )

        return img

    # ------------------------------------------------------------------
    # Player label with speed + distance
    # ------------------------------------------------------------------

    def _draw_player_label(
        self, img: np.ndarray, player: Any, color: tuple[int, int, int],
    ) -> None:
        bbox = player.detection.bbox
        tid = int(player.track_id)

        # Line 1: track ID + speed
        label = f"#{tid}"
        if self.show_speed and player.speed_mps is not None:
            kmh = float(player.speed_mps) * 3.6
            label += f" {kmh:.1f}km/h"

        # Draw label background for readability
        x_pos = self._safe_int(bbox.x1)
        y_pos = self._safe_int(bbox.y1) - 5

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(
            img,
            (x_pos - 1, y_pos - th - 2),
            (x_pos + tw + 1, y_pos + 2),
            (0, 0, 0), -1,
        )
        cv2.putText(
            img, label, (x_pos, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
        )

        # Line 2: cumulative distance
        if self.show_distance and tid in self._cum_dist:
            dist_m = self._cum_dist[tid]
            dist_label = f"{dist_m:.0f}m"
            y2 = y_pos - th - 5
            (tw2, th2), _ = cv2.getTextSize(dist_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(
                img,
                (x_pos - 1, y2 - th2 - 2),
                (x_pos + tw2 + 1, y2 + 2),
                (0, 0, 0), -1,
            )
            cv2.putText(
                img, dist_label, (x_pos, y2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1,
            )

    # ------------------------------------------------------------------
    # Ball halo
    # ------------------------------------------------------------------

    def _draw_ball_halo(self, img: np.ndarray, frame_state: FrameState) -> None:
        bbox = frame_state.ball.detection.bbox
        cx, cy = int(float(bbox.center[0])), int(float(bbox.center[1]))
        radius = max(5, int(float(bbox.width) / 2))

        # Outer halo (translucent effect via thick ring)
        cv2.circle(img, (cx, cy), radius + 8, HALO_COLOR, 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), radius + 4, HALO_COLOR, 1, cv2.LINE_AA)
        # Ball circle
        cv2.circle(img, (cx, cy), radius, BALL_COLOR, 2)
        cv2.circle(img, (cx, cy), 3, BALL_COLOR, -1)

    # ------------------------------------------------------------------
    # Ball control bar
    # ------------------------------------------------------------------

    def _draw_ball_control(self, img: np.ndarray) -> None:
        if self._total_owned_frames == 0:
            return

        h, w = img.shape[:2]
        bar_w = 200
        bar_h = 20
        margin = 10
        x0 = w - bar_w - margin
        y0 = h - bar_h - margin - 25  # above frame info line

        # Compute percentages
        teams = sorted(self._team_control_frames.keys())
        if len(teams) < 2:
            return

        pct_0 = self._team_control_frames.get(teams[0], 0) / self._total_owned_frames
        pct_1 = self._team_control_frames.get(teams[1], 0) / self._total_owned_frames

        # Background
        cv2.rectangle(img, (x0 - 2, y0 - 18), (x0 + bar_w + 2, y0 + bar_h + 4), (0, 0, 0), -1)

        # Label
        label = f"Ball Control  {pct_0 * 100:.0f}% - {pct_1 * 100:.0f}%"
        cv2.putText(
            img, label, (x0, y0 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1,
        )

        # Two-sided bar
        color_0 = TEAM_COLORS.get(teams[0], (200, 200, 200))
        color_1 = TEAM_COLORS.get(teams[1], (200, 200, 200))
        split_x = x0 + int(bar_w * pct_0)

        cv2.rectangle(img, (x0, y0), (split_x, y0 + bar_h), color_0, -1)
        cv2.rectangle(img, (split_x, y0), (x0 + bar_w, y0 + bar_h), color_1, -1)
        cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1)

    # ------------------------------------------------------------------
    # Camera motion debug overlay
    # ------------------------------------------------------------------

    def _draw_camera_debug(self, img: np.ndarray, frame_state: FrameState) -> None:
        cam = frame_state.analytics.get("camera_motion")
        if cam is None:
            return

        dx = cam.get("smoothed_dx_px", 0.0)
        dy = cam.get("smoothed_dy_px", 0.0)
        conf = cam.get("confidence", "?")
        matches = cam.get("good_matches", 0)

        lines = [
            f"Cam dx:{dx:+.1f} dy:{dy:+.1f}",
            f"conf:{conf} matches:{matches}",
        ]

        x0 = img.shape[1] - 220
        y0 = 20
        for i, line in enumerate(lines):
            y = y0 + i * 16
            cv2.putText(
                img, line, (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1,
            )

    # ------------------------------------------------------------------
    # Radar minimap
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Internal state trackers
    # ------------------------------------------------------------------

    def _update_distances(self, frame_state: FrameState) -> None:
        """Update cumulative distance for each player with pitch coordinates."""
        for player in frame_state.players:
            if player.pitch_pos is None:
                continue
            tid = player.track_id
            pos = (player.pitch_pos.x, player.pitch_pos.y)
            if tid in self._prev_pos:
                dx = pos[0] - self._prev_pos[tid][0]
                dy = pos[1] - self._prev_pos[tid][1]
                step = (dx * dx + dy * dy) ** 0.5
                # Same glitch filter as pipeline: skip > 4m steps
                if step <= 4.0:
                    self._cum_dist[tid] = self._cum_dist.get(tid, 0.0) + step
            elif tid not in self._cum_dist:
                self._cum_dist[tid] = 0.0
            self._prev_pos[tid] = pos

    def _update_ball_control(self, frame_state: FrameState) -> None:
        """Track running team ball control from analytics."""
        owner_info = frame_state.analytics.get("ball_owner")
        if owner_info is None:
            return
        team_id = owner_info.get("owner_team_id")
        if team_id is not None:
            self._team_control_frames[team_id] = (
                self._team_control_frames.get(team_id, 0) + 1
            )
            self._total_owned_frames += 1

    def _get_ball_owner(self, frame_state: FrameState) -> Optional[int]:
        """Get current ball owner track_id from analytics."""
        owner_info = frame_state.analytics.get("ball_owner")
        if owner_info is not None:
            return owner_info.get("owner_player_id")
        return None
