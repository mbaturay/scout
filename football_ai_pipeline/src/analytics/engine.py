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
from .passes import compute_passes, aggregate_pass_stats, write_pass_events
from .possession import compute_possession, extract_tracks_from_frame, write_all_outputs as write_possession_outputs

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
        self._serialized_frames: list[dict[str, Any]] = []
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
            self._serialized_frames.append(frame_state.to_serializable())
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

        # Accumulate serialized frame for post-run possession computation
        self._serialized_frames.append(frame_state.to_serializable())

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

        # Compute v1 possession (tight 1.25 m threshold)
        acfg = self.config.get("analytics", {})
        max_dist = acfg.get("possession_max_dist_m", 1.25)
        write_timeline = acfg.get("possession_write_timeline", False)
        poss_result = compute_possession(
            self._serialized_frames, max_dist_m=max_dist,
        )
        logger.info(
            "Possession computed: %d owned / %d total frames",
            poss_result.owned_frames, poss_result.total_frames,
        )

        # Merge possession_pct into team_stats
        for tid, pct in poss_result.team_possession.items():
            if tid in result["team_stats"]:
                result["team_stats"][tid]["possession_pct"] = pct
            else:
                result["team_stats"][tid] = {"possession_pct": pct}

        # Merge touches into player_stats
        for track_id, touches in poss_result.player_touches.items():
            if track_id in result["player_stats"]:
                result["player_stats"][track_id]["touches"] = touches
            else:
                team_id = poss_result.player_team.get(track_id)
                result["player_stats"][track_id] = {
                    "team_id": team_id,
                    "touches": touches,
                }

        # --- V1 pass detection ---
        # Extract ball positions aligned with the possession timeline
        ball_positions_for_passes: list[Optional[tuple[float, float]]] = []
        for frame in self._serialized_frames:
            ball_xy, _ = extract_tracks_from_frame(frame)
            ball_positions_for_passes.append(ball_xy)

        pass_min_dist = acfg.get("pass_min_dist_m", 3.0)
        pass_max_gap = acfg.get("pass_max_gap_frames", 10)
        pass_events = compute_passes(
            poss_result.timeline,
            ball_positions_for_passes,
            poss_result.player_team,
            min_pass_dist_m=pass_min_dist,
            max_gap_frames=pass_max_gap,
        )
        logger.info("Detected %d pass events (v1)", len(pass_events))

        # Aggregate pass stats and merge into team/player stats
        team_pass_stats, player_pass_stats = aggregate_pass_stats(
            pass_events, poss_result.player_team,
        )
        for tid, pstats in team_pass_stats.items():
            if tid in result["team_stats"]:
                result["team_stats"][tid]["pass_count"] = pstats["pass_count"]
                result["team_stats"][tid]["pass_completed"] = pstats["completed_pass_count"]
                result["team_stats"][tid]["pass_completion_pct"] = pstats["pass_accuracy_pct"]
            else:
                result["team_stats"][tid] = pstats

        for pid, pstats in player_pass_stats.items():
            if pid in result["player_stats"]:
                result["player_stats"][pid]["passes_attempted"] = pstats["passes_attempted"]
                result["player_stats"][pid]["passes_completed"] = pstats["passes_completed"]
                result["player_stats"][pid]["pass_accuracy_pct"] = pstats["pass_accuracy_pct"]
                result["player_stats"][pid]["passes"] = pstats["passes_attempted"]
            else:
                team_id = poss_result.player_team.get(pid)
                result["player_stats"][pid] = {
                    "team_id": team_id,
                    **pstats,
                    "passes": pstats["passes_attempted"],
                }

        # Append pass/interception events to the main event list
        for pe in pass_events:
            ev = MatchEvent(
                event_type="pass" if pe.is_completed else "interception",
                frame_idx=pe.t_start,
                timestamp_sec=pe.t_start / self._fps,
                team_id=pe.team_id,
                player_id=pe.from_track,
                target_player_id=pe.to_track,
                target_team_id=pe.to_team_id,
                confidence=0.6,
                features={
                    "dist_m": round(pe.dist_m, 2),
                    "reason": pe.reason,
                    "source": "v1_pass_detection",
                },
            )
            events.append(ev)
        events.sort(key=lambda e: e.frame_idx)

        # Write outputs
        self.metrics.write_outputs(
            output_dir,
            result["team_stats"],
            result["player_stats"],
            events,
            self._per_frame_ball,
        )

        # Write possession-specific outputs
        write_possession_outputs(output_dir, poss_result, write_timeline=write_timeline)

        # Write pass events
        write_pass_events(output_dir / "pass_events.json", pass_events)

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
            "possession_computed": True,
            "possession_owned_frames": poss_result.owned_frames,
            "possession_total_frames": poss_result.total_frames,
            "pass_events_detected": len(pass_events),
            "pass_events_completed": sum(1 for p in pass_events if p.is_completed),
        }

        # Add per-team possession headline
        for tid, ts in result["team_stats"].items():
            summary[f"team_{tid}_possession_pct"] = ts.get("possession_pct", 0.0)
            summary[f"team_{tid}_xG"] = ts.get("xG_total", 0.0)

        return summary
