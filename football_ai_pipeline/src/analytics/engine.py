"""Analytics engine — orchestrates association, events, and metrics.

Called by the pipeline runner after all frames are processed.  Collects
per-frame data during the run, then computes aggregate analytics at the end.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from ..data_models import FrameState, FrameFlag
from .association import BallOwnerAssigner, OwnerRecord
from .events import EventDetector, MatchEvent
from .metrics import MetricsComputer

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """Collects per-frame data and computes post-run analytics."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.assigner = BallOwnerAssigner(config)
        self.event_detector = EventDetector(config)
        self.metrics = MetricsComputer(config)

        # Per-frame accumulators (populated during run)
        self._per_frame_players: list[dict[int, dict[str, Any]]] = []
        self._per_frame_ball: list[Optional[tuple[float, float]]] = []
        self._ball_speeds: list[Optional[float]] = []
        self._player_positions: list[dict[int, tuple[float, float]]] = []
        self._fps: float = 30.0

    def set_fps(self, fps: float) -> None:
        if fps > 0:
            self._fps = fps
            self.event_detector.set_fps(fps)

    def update(self, frame_state: FrameState) -> FrameState:
        """Collect data from one processed frame and run ball association.

        Called per-frame during the pipeline run, after all other stages.
        """
        # Skip out-of-play frames
        if frame_state.flag == FrameFlag.NOT_IN_PLAY:
            self._per_frame_players.append({})
            self._per_frame_ball.append(None)
            self._ball_speeds.append(None)
            self._player_positions.append({})
            return frame_state

        # Collect player data
        players_info: list[dict[str, Any]] = []
        frame_player_data: dict[int, dict[str, Any]] = {}
        frame_player_pos: dict[int, tuple[float, float]] = {}

        for p in frame_state.players:
            if p.pitch_pos is not None:
                x, y = p.pitch_pos.x, p.pitch_pos.y
                is_pitch = True
            else:
                x, y = p.detection.bbox.center
                is_pitch = False

            pdata = {
                "track_id": p.track_id,
                "team_id": p.team_id,
                "x": x,
                "y": y,
                "is_pitch": is_pitch,
                "speed_mps": p.speed_mps,
            }
            players_info.append(pdata)
            frame_player_data[p.track_id] = pdata
            frame_player_pos[p.track_id] = (x, y)

        # Ball data
        ball_info = None
        ball_pos: Optional[tuple[float, float]] = None
        ball_speed: Optional[float] = None

        if frame_state.ball is not None:
            bs = frame_state.ball
            if bs.pitch_pos is not None:
                ball_info = {"x": bs.pitch_pos.x, "y": bs.pitch_pos.y, "is_pitch": True}
                ball_pos = (bs.pitch_pos.x, bs.pitch_pos.y)
            elif bs.detection is not None:
                bx, by = bs.detection.bbox.center
                ball_info = {"x": bx, "y": by, "is_pitch": False}
                ball_pos = (bx, by)
            ball_speed = bs.speed_mps

        self._per_frame_players.append(frame_player_data)
        self._per_frame_ball.append(ball_pos)
        self._ball_speeds.append(ball_speed)
        self._player_positions.append(frame_player_pos)

        # Run ball ownership assignment
        owner_rec = self.assigner.update({
            "frame_idx": frame_state.frame_idx,
            "players": players_info,
            "ball": ball_info,
        })

        # Store ownership in frame analytics
        frame_state.analytics["ball_owner"] = owner_rec.to_dict()

        return frame_state

    def finalize(self, output_dir: Path) -> dict[str, Any]:
        """Compute all analytics after the run and write outputs.

        Returns analytics summary for inclusion in run_report.
        """
        ownership = self.assigner.history
        if not ownership:
            logger.warning("No frames processed — skipping analytics.")
            return {"analytics": {"status": "no_data"}}

        # Detect events
        events = self.event_detector.detect(
            ownership,
            self._per_frame_ball,
            self._ball_speeds,
            self._player_positions,
        )
        logger.info("Detected %d events", len(events))

        # Compute metrics
        result = self.metrics.compute(
            ownership,
            events,
            self._per_frame_players,
            self._per_frame_ball,
            self._fps,
        )

        # Write outputs
        self.metrics.write_outputs(
            output_dir,
            result["team_stats"],
            result["player_stats"],
            events,
            self._per_frame_ball,
        )

        # Build summary for run_report
        event_counts: dict[str, int] = {}
        for ev in events:
            event_counts[ev.event_type] = event_counts.get(ev.event_type, 0) + 1

        summary = {
            "ball_owner_pct": result["coverage"]["owner_assigned_pct"],
            "ball_detected_pct": result["coverage"]["ball_detected_pct"],
            "high_confidence_pct": result["coverage"]["high_confidence_pct"],
            "event_counts": event_counts,
            "total_events": len(events),
            "warnings": result["warnings"],
        }

        # Add per-team possession headline
        for tid, ts in result["team_stats"].items():
            summary[f"team_{tid}_possession_pct"] = ts["possession_pct"]
            summary[f"team_{tid}_xG"] = ts["xG_total"]

        return summary
