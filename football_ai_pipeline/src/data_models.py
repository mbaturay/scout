"""Core data models for the football AI pipeline.

All pipeline stages communicate through these dataclasses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ObjectClass(Enum):
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    BALL = "ball"


class FrameFlag(Enum):
    IN_PLAY = "in_play"
    NOT_IN_PLAY = "not_in_play"


# ---------------------------------------------------------------------------
# Low-level data carriers
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coords (x1, y1, x2, y2)."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Foot position estimate."""
        return ((self.x1 + self.x2) / 2, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0, self.width) * max(0, self.height)


@dataclass
class Detection:
    """A single object detection in one frame."""
    bbox: BBox
    class_id: ObjectClass
    confidence: float
    track_id: Optional[int] = None
    team_id: Optional[int] = None
    color_features: Optional[list[float]] = None


@dataclass
class Keypoint:
    """A single pitch landmark keypoint."""
    x: float
    y: float
    confidence: float
    label: Optional[str] = None


@dataclass
class HomographyResult:
    """Homography matrix + quality metadata."""
    matrix: Optional[np.ndarray] = None  # 3x3
    quality: float = 0.0
    num_inliers: int = 0
    available: bool = False

    def to_serializable(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix.tolist() if self.matrix is not None else None,
            "quality": self.quality,
            "num_inliers": self.num_inliers,
            "available": self.available,
        }


@dataclass
class PitchPosition:
    """A position in pitch coordinates (metres)."""
    x: float  # 0..pitch_length
    y: float  # 0..pitch_width
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Player / Ball state
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    """Per-player state for a single frame."""
    track_id: int
    detection: Detection
    team_id: Optional[int] = None
    pitch_pos: Optional[PitchPosition] = None
    speed_mps: Optional[float] = None  # metres per second
    is_sprinting: bool = False


@dataclass
class BallState:
    """Ball state for a single frame."""
    detection: Optional[Detection] = None
    pitch_pos: Optional[PitchPosition] = None
    speed_mps: Optional[float] = None
    interpolated: bool = False


# ---------------------------------------------------------------------------
# FrameState  — the central per-frame data object
# ---------------------------------------------------------------------------

@dataclass
class FrameState:
    """Complete state for one processed frame.

    Every pipeline stage enriches relevant fields.
    """
    frame_idx: int
    timestamp_sec: float

    # FR2 — in-play filter
    flag: FrameFlag = FrameFlag.IN_PLAY
    flag_reasons: list[str] = field(default_factory=list)

    # FR3 — detections (raw)
    detections: list[Detection] = field(default_factory=list)

    # FR4 — tracked players & ball
    players: list[PlayerState] = field(default_factory=list)
    ball: Optional[BallState] = None

    # FR5 — team assignments stored inside PlayerState.team_id

    # FR6 — keypoints + homography
    keypoints: list[Keypoint] = field(default_factory=list)
    keypoint_confidences: list[float] = field(default_factory=list)
    homography: HomographyResult = field(default_factory=HomographyResult)

    # FR7 — pitch positions stored inside PlayerState / BallState

    # FR8 — per-frame analytics (populated by stats modules)
    analytics: dict[str, Any] = field(default_factory=dict)

    # Image data (not serialised)
    image: Optional[np.ndarray] = field(default=None, repr=False)

    def to_serializable(self) -> dict[str, Any]:
        """Convert to JSON-safe dict (excludes image)."""
        d: dict[str, Any] = {
            "frame_idx": self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 4),
            "flag": self.flag.value,
            "flag_reasons": self.flag_reasons,
            "homography": self.homography.to_serializable(),
            "players": [],
            "ball": None,
            "analytics": self.analytics,
        }
        for p in self.players:
            pd: dict[str, Any] = {
                "track_id": p.track_id,
                "team_id": p.team_id,
                "bbox": [p.detection.bbox.x1, p.detection.bbox.y1,
                         p.detection.bbox.x2, p.detection.bbox.y2],
                "class": p.detection.class_id.value,
                "confidence": round(p.detection.confidence, 3),
                "pitch_x": round(p.pitch_pos.x, 2) if p.pitch_pos else None,
                "pitch_y": round(p.pitch_pos.y, 2) if p.pitch_pos else None,
                "speed_mps": round(p.speed_mps, 2) if p.speed_mps is not None else None,
                "is_sprinting": p.is_sprinting,
            }
            d["players"].append(pd)
        if self.ball:
            bd: dict[str, Any] = {
                "pitch_x": round(self.ball.pitch_pos.x, 2) if self.ball.pitch_pos else None,
                "pitch_y": round(self.ball.pitch_pos.y, 2) if self.ball.pitch_pos else None,
                "speed_mps": round(self.ball.speed_mps, 2) if self.ball.speed_mps is not None else None,
                "interpolated": self.ball.interpolated,
            }
            if self.ball.detection:
                bd["bbox"] = [self.ball.detection.bbox.x1, self.ball.detection.bbox.y1,
                              self.ball.detection.bbox.x2, self.ball.detection.bbox.y2]
                bd["confidence"] = round(self.ball.detection.confidence, 3)
            d["ball"] = bd
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_serializable())
