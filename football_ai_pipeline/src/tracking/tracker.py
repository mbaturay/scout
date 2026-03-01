"""FR4 — Object Tracking: assign persistent track IDs.

Uses supervision's ByteTrack when available; otherwise falls back to a
simple IoU-based tracker.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from ..data_models import (
    BBox, BallState, Detection, FrameState, ObjectClass, PlayerState,
)

logger = logging.getLogger(__name__)


def _iou(a: BBox, b: BBox) -> float:
    xi1 = max(a.x1, b.x1)
    yi1 = max(a.y1, b.y1)
    xi2 = min(a.x2, b.x2)
    yi2 = min(a.y2, b.y2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


class _SimpleIoUTracker:
    """Minimal IoU-based tracker for when supervision is not available."""

    def __init__(self, iou_thresh: float = 0.3, max_lost: int = 30) -> None:
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self._next_id = 1
        self._tracks: dict[int, tuple[BBox, int]] = {}  # track_id -> (bbox, lost_frames)

    def update(self, detections: list[Detection]) -> list[Detection]:
        if not detections:
            # Age out all tracks
            self._tracks = {
                tid: (bbox, lost + 1)
                for tid, (bbox, lost) in self._tracks.items()
                if lost + 1 <= self.max_lost
            }
            return detections

        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = set(self._tracks.keys())
        matches: list[tuple[int, int]] = []  # (track_id, det_idx)

        # Greedy matching by IoU
        pairs: list[tuple[float, int, int]] = []
        for tid, (tbbox, _) in self._tracks.items():
            for di in unmatched_dets:
                score = _iou(tbbox, detections[di].bbox)
                if score >= self.iou_thresh:
                    pairs.append((score, tid, di))
        pairs.sort(key=lambda x: -x[0])

        matched_tids: set[int] = set()
        matched_dets: set[int] = set()
        for score, tid, di in pairs:
            if tid in matched_tids or di in matched_dets:
                continue
            matches.append((tid, di))
            matched_tids.add(tid)
            matched_dets.add(di)

        # Update matched
        for tid, di in matches:
            detections[di].track_id = tid
            self._tracks[tid] = (detections[di].bbox, 0)

        # Age unmatched tracks
        for tid in set(self._tracks.keys()) - matched_tids:
            bbox, lost = self._tracks[tid]
            if lost + 1 > self.max_lost:
                del self._tracks[tid]
            else:
                self._tracks[tid] = (bbox, lost + 1)

        # Create new tracks for unmatched detections
        for di in range(len(detections)):
            if di not in matched_dets:
                tid = self._next_id
                self._next_id += 1
                detections[di].track_id = tid
                self._tracks[tid] = (detections[di].bbox, 0)

        return detections


class ObjectTracker:
    """Wrap supervision ByteTrack or fallback to IoU tracker."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        trk_cfg = config.get("tracking", {})
        self._sv_tracker: Any = None
        self._simple_tracker: Optional[_SimpleIoUTracker] = None
        self.backend: str = "none"

        try:
            import supervision as sv
            self._sv_tracker = sv.ByteTrack(
                track_activation_threshold=trk_cfg.get("track_activation_threshold", 0.25),
                lost_track_buffer=trk_cfg.get("lost_track_buffer", 30),
                minimum_matching_threshold=trk_cfg.get("minimum_matching_threshold", 0.8),
                frame_rate=trk_cfg.get("frame_rate", 30),
            )
            self.backend = "bytetrack"
            logger.info("Tracker: supervision ByteTrack (recommended)")
        except ImportError:
            logger.warning(
                "supervision is not installed — using simple IoU tracker.\n"
                "  Impact: Track IDs may be less stable across occlusions.\n"
                "  Fix:    pip install supervision"
            )
            self._simple_tracker = _SimpleIoUTracker(
                iou_thresh=0.3,
                max_lost=trk_cfg.get("lost_track_buffer", 30),
            )
            self.backend = "iou_fallback"
        except Exception as e:
            logger.warning(
                "supervision failed to initialize (%s) — using simple IoU tracker.\n"
                "  Impact: Track IDs may be less stable across occlusions.",
                e,
            )
            self._simple_tracker = _SimpleIoUTracker(
                iou_thresh=0.3,
                max_lost=trk_cfg.get("lost_track_buffer", 30),
            )
            self.backend = "iou_fallback"

    def track(self, frame_state: FrameState) -> FrameState:
        """Assign track IDs and build PlayerState / BallState lists."""
        dets = frame_state.detections
        if not dets:
            frame_state.players = []
            frame_state.ball = None
            return frame_state

        # Separate ball detections
        ball_dets = [d for d in dets if d.class_id == ObjectClass.BALL]
        person_dets = [d for d in dets if d.class_id != ObjectClass.BALL]

        # Track persons
        person_dets = self._assign_ids(person_dets)

        # Build PlayerState
        frame_state.players = [
            PlayerState(
                track_id=d.track_id or 0,
                detection=d,
                team_id=d.team_id,
            )
            for d in person_dets
        ]

        # Ball: pick highest confidence
        if ball_dets:
            best_ball = max(ball_dets, key=lambda d: d.confidence)
            frame_state.ball = BallState(detection=best_ball)
        else:
            frame_state.ball = BallState()  # missing ball

        return frame_state

    def _assign_ids(self, detections: list[Detection]) -> list[Detection]:
        if not detections:
            return detections

        if self._sv_tracker is not None:
            return self._assign_ids_sv(detections)

        if self._simple_tracker is not None:
            return self._simple_tracker.update(detections)

        return detections

    def _assign_ids_sv(self, detections: list[Detection]) -> list[Detection]:
        import supervision as sv

        xyxy = np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections])
        confs = np.array([d.confidence for d in detections])
        class_ids = np.array([0] * len(detections))  # all "person" for tracker

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=confs,
            class_id=class_ids,
        )
        tracked = self._sv_tracker.update_with_detections(sv_dets)

        for i, det in enumerate(detections):
            if i < len(tracked.tracker_id):
                det.track_id = int(tracked.tracker_id[i])
            else:
                det.track_id = 0

        return detections
