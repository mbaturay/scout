"""Ball-to-player association: determine ball owner per frame.

Uses nearest-player assignment with hysteresis smoothing to prevent
rapid ownership flickering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class OwnerRecord:
    """Ball ownership state for a single frame."""

    frame_idx: int
    owner_player_id: Optional[int] = None
    owner_team_id: Optional[int] = None
    owner_confidence: float = 0.0
    ball_available: bool = False
    distance: float = float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "owner_player_id": self.owner_player_id,
            "owner_team_id": self.owner_team_id,
            "owner_confidence": self.owner_confidence,
            "ball_available": self.ball_available,
            "distance": round(self.distance, 3) if math.isfinite(self.distance) else None,
        }


@dataclass
class _PlayerSnapshot:
    track_id: int
    team_id: Optional[int]
    x: float
    y: float
    is_pitch: bool  # True = pitch metres, False = pixel coords


class BallOwnerAssigner:
    """Assign ball ownership per frame with hysteresis smoothing.

    Parameters (from config["analytics"]):
        pitch_threshold_m:  max distance (metres) to claim ownership on pitch.
        pixel_threshold_px: fallback max distance (pixels) when no homography.
        hysteresis_frames:  consecutive frames required to switch owner.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        acfg = config.get("analytics", {})
        self.pitch_threshold: float = acfg.get("pitch_threshold_m", 5.0)
        self.pixel_threshold: float = acfg.get("pixel_threshold_px", 80.0)
        self.hysteresis: int = acfg.get("hysteresis_frames", 3)

        # Internal state
        self._current_owner: Optional[int] = None
        self._candidate: Optional[int] = None
        self._candidate_team: Optional[int] = None
        self._candidate_count: int = 0
        self._history: list[OwnerRecord] = []

    @property
    def history(self) -> list[OwnerRecord]:
        return self._history

    def update(self, frame_info: dict[str, Any]) -> OwnerRecord:
        """Process one frame and return ownership record.

        frame_info keys:
            frame_idx: int
            players: list of dict with track_id, team_id, x, y, is_pitch
            ball: dict with x, y, is_pitch  (or None if no ball)
        """
        frame_idx = frame_info["frame_idx"]
        ball = frame_info.get("ball")
        players = frame_info.get("players", [])

        rec = OwnerRecord(frame_idx=frame_idx)

        if ball is None or not players:
            rec.ball_available = ball is not None
            # Keep current owner during brief ball gaps (up to hysteresis)
            if self._current_owner is not None:
                rec.owner_player_id = self._current_owner
                rec.owner_team_id = self._current_owner_team
                rec.owner_confidence = 0.3
            self._history.append(rec)
            return rec

        rec.ball_available = True
        bx, by = ball["x"], ball["y"]
        is_pitch = ball.get("is_pitch", False)
        threshold = self.pitch_threshold if is_pitch else self.pixel_threshold

        # Find nearest player
        best_id: Optional[int] = None
        best_team: Optional[int] = None
        best_dist = float("inf")

        for p in players:
            dx = p["x"] - bx
            dy = p["y"] - by
            d = math.sqrt(dx * dx + dy * dy)
            if d < best_dist:
                best_dist = d
                best_id = p["track_id"]
                best_team = p.get("team_id")

        if best_dist > threshold:
            # Ball is loose — no owner
            rec.owner_confidence = 0.0
            rec.distance = best_dist
            self._candidate = None
            self._candidate_count = 0
            if self._current_owner is not None:
                rec.owner_player_id = self._current_owner
                rec.owner_team_id = self._current_owner_team
                rec.owner_confidence = 0.2
            self._history.append(rec)
            return rec

        rec.distance = best_dist

        # Hysteresis: require N consecutive frames to switch
        if best_id == self._current_owner:
            # Same owner — confirm
            rec.owner_player_id = best_id
            rec.owner_team_id = best_team
            confidence = max(0.0, 1.0 - best_dist / threshold)
            rec.owner_confidence = round(confidence, 3)
            self._candidate = None
            self._candidate_count = 0
        elif best_id == self._candidate:
            self._candidate_count += 1
            if self._candidate_count >= self.hysteresis:
                # Switch owner
                self._current_owner = best_id
                self._current_owner_team = best_team
                rec.owner_player_id = best_id
                rec.owner_team_id = best_team
                confidence = max(0.0, 1.0 - best_dist / threshold)
                rec.owner_confidence = round(confidence, 3)
                self._candidate = None
                self._candidate_count = 0
            else:
                # Still in transition — keep old owner
                rec.owner_player_id = self._current_owner
                rec.owner_team_id = self._current_owner_team if self._current_owner is not None else None
                rec.owner_confidence = 0.4
        else:
            # New candidate
            self._candidate = best_id
            self._candidate_team = best_team
            self._candidate_count = 1
            if self._current_owner is None:
                # First owner ever — assign immediately
                self._current_owner = best_id
                self._current_owner_team = best_team
                rec.owner_player_id = best_id
                rec.owner_team_id = best_team
                confidence = max(0.0, 1.0 - best_dist / threshold)
                rec.owner_confidence = round(confidence, 3)
            else:
                rec.owner_player_id = self._current_owner
                rec.owner_team_id = self._current_owner_team
                rec.owner_confidence = 0.4

        self._history.append(rec)
        return rec

    # Initialised lazily on first use
    _current_owner_team: Optional[int] = None
