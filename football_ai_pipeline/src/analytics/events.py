"""Event detection from ball-ownership timeline.

Derives passes, receptions, interceptions, tackles, shots from the
per-frame owner sequence produced by BallOwnerAssigner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from .association import OwnerRecord


@dataclass
class MatchEvent:
    """A single match event."""

    event_type: str  # touch, pass, reception, interception, tackle, shot
    frame_idx: int
    timestamp_sec: float
    team_id: Optional[int] = None
    player_id: Optional[int] = None
    target_player_id: Optional[int] = None
    target_team_id: Optional[int] = None
    confidence: float = 0.0
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "event_type": self.event_type,
            "frame_idx": self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "team_id": self.team_id,
            "player_id": self.player_id,
            "confidence": round(self.confidence, 3),
        }
        if self.target_player_id is not None:
            d["target_player_id"] = self.target_player_id
        if self.target_team_id is not None:
            d["target_team_id"] = self.target_team_id
        if self.features:
            d["features"] = self.features
        return d


class EventDetector:
    """Detect match events from the ownership timeline.

    Parameters (from config["analytics"]):
        pass_min_distance_m:  minimum ball travel for a pass (metres).
        pass_max_gap_sec:     max time gap between owners for a pass.
        inflight_min_frames:  frames without owner to qualify as in-flight.
        shot_attack_dir:      "left_to_right" or "right_to_left".
        shot_min_speed_mps:   min ball speed to tag as shot.
        fps:                  video fps for timestamp computation.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        acfg = config.get("analytics", {})
        self.pass_min_dist: float = acfg.get("pass_min_distance_m", 3.0)
        self.pass_max_gap: float = acfg.get("pass_max_gap_sec", 3.0)
        self.inflight_min: int = acfg.get("inflight_min_frames", 2)
        self.left_to_right: bool = acfg.get("left_to_right", True)
        self.shot_min_speed: float = acfg.get("shot_min_speed_mps", 5.0)
        self.pitch_length: float = config.get("pitch", {}).get("length", 105.0)

        # fps resolved later from metadata
        self._fps: float = config.get("video", {}).get("target_fps", None) or 30.0

    def set_fps(self, fps: float) -> None:
        if fps > 0:
            self._fps = fps

    def detect(
        self,
        ownership: list[OwnerRecord],
        ball_positions: list[Optional[tuple[float, float]]],
        ball_speeds: list[Optional[float]],
        player_positions: list[dict[int, tuple[float, float]]],
    ) -> list[MatchEvent]:
        """Detect events from the full ownership timeline.

        Args:
            ownership:        per-frame OwnerRecord list
            ball_positions:   per-frame (x, y) or None
            ball_speeds:      per-frame speed (m/s) or None
            player_positions: per-frame {track_id: (x, y)}

        Returns:
            List of MatchEvent sorted by frame_idx.
        """
        events: list[MatchEvent] = []
        n = len(ownership)
        if n < 2:
            return events

        # Find ownership transitions
        i = 0
        while i < n:
            # Skip frames with no owner
            if ownership[i].owner_player_id is None:
                i += 1
                continue

            # Find next transition (owner changes)
            j = i + 1
            while j < n and (
                ownership[j].owner_player_id == ownership[i].owner_player_id
                or ownership[j].owner_player_id is None
            ):
                j += 1

            if j >= n:
                break

            prev = ownership[i]
            curr = ownership[j]

            # Count in-flight frames (no owner between i and j)
            inflight = sum(
                1 for k in range(i + 1, j)
                if ownership[k].owner_player_id is None
                or ownership[k].owner_confidence < 0.3
            )

            time_gap = (j - i) / self._fps
            ts_prev = i / self._fps
            ts_curr = j / self._fps

            # Ball travel distance
            ball_dist = self._ball_distance(ball_positions, i, j)

            same_team = (
                prev.owner_team_id is not None
                and curr.owner_team_id is not None
                and prev.owner_team_id == curr.owner_team_id
            )

            confidence = min(prev.owner_confidence, curr.owner_confidence)

            if same_team:
                # Pass or touch within same team
                if ball_dist >= self.pass_min_dist and time_gap <= self.pass_max_gap:
                    events.append(MatchEvent(
                        event_type="pass",
                        frame_idx=i,
                        timestamp_sec=ts_prev,
                        team_id=prev.owner_team_id,
                        player_id=prev.owner_player_id,
                        target_player_id=curr.owner_player_id,
                        target_team_id=curr.owner_team_id,
                        confidence=confidence,
                        features={"distance": round(ball_dist, 2), "dt_sec": round(time_gap, 3)},
                    ))
                    events.append(MatchEvent(
                        event_type="reception",
                        frame_idx=j,
                        timestamp_sec=ts_curr,
                        team_id=curr.owner_team_id,
                        player_id=curr.owner_player_id,
                        confidence=confidence,
                        features={"from_player": prev.owner_player_id},
                    ))
                else:
                    events.append(MatchEvent(
                        event_type="touch",
                        frame_idx=j,
                        timestamp_sec=ts_curr,
                        team_id=curr.owner_team_id,
                        player_id=curr.owner_player_id,
                        confidence=confidence,
                    ))
            else:
                # Cross-team transition
                if inflight >= self.inflight_min:
                    events.append(MatchEvent(
                        event_type="interception",
                        frame_idx=j,
                        timestamp_sec=ts_curr,
                        team_id=curr.owner_team_id,
                        player_id=curr.owner_player_id,
                        target_team_id=prev.owner_team_id,
                        target_player_id=prev.owner_player_id,
                        confidence=confidence * 0.8,
                        features={"inflight_frames": inflight, "distance": round(ball_dist, 2)},
                    ))
                else:
                    events.append(MatchEvent(
                        event_type="tackle",
                        frame_idx=j,
                        timestamp_sec=ts_curr,
                        team_id=curr.owner_team_id,
                        player_id=curr.owner_player_id,
                        target_team_id=prev.owner_team_id,
                        target_player_id=prev.owner_player_id,
                        confidence=confidence * 0.7,
                        features={"distance": round(ball_dist, 2)},
                    ))

            i = j

        # Detect shots: ball leaves owner at high speed toward goal
        events.extend(self._detect_shots(ownership, ball_positions, ball_speeds))

        events.sort(key=lambda e: e.frame_idx)
        return events

    def _ball_distance(
        self,
        positions: list[Optional[tuple[float, float]]],
        i: int,
        j: int,
    ) -> float:
        """Euclidean distance ball traveled between frames i and j."""
        pi = positions[i] if i < len(positions) else None
        pj = positions[j] if j < len(positions) else None
        if pi is None or pj is None:
            return 0.0
        dx = pj[0] - pi[0]
        dy = pj[1] - pi[1]
        return math.sqrt(dx * dx + dy * dy)

    def _detect_shots(
        self,
        ownership: list[OwnerRecord],
        ball_positions: list[Optional[tuple[float, float]]],
        ball_speeds: list[Optional[float]],
    ) -> list[MatchEvent]:
        """Heuristic shot detection: owner loses ball + speed spike toward goal."""
        shots: list[MatchEvent] = []
        n = len(ownership)

        # Goal x-positions
        attack_goal_x = self.pitch_length if self.left_to_right else 0.0

        for i in range(1, n):
            prev = ownership[i - 1]
            curr = ownership[i]

            # Owner just lost the ball
            if prev.owner_player_id is None or curr.owner_player_id is not None:
                continue

            # Check speed
            speed = ball_speeds[i] if i < len(ball_speeds) else None
            if speed is None or speed < self.shot_min_speed:
                continue

            # Check direction toward goal
            bp = ball_positions[i] if i < len(ball_positions) else None
            if bp is None:
                continue

            dist_to_goal = abs(bp[0] - attack_goal_x)
            # Must be in attacking third
            if dist_to_goal > self.pitch_length * 0.4:
                continue

            angle_to_goal = math.atan2(34.0 - bp[1], max(0.1, dist_to_goal))
            on_target = abs(bp[1] - 34.0) < 12.0 and dist_to_goal < 30.0

            shots.append(MatchEvent(
                event_type="shot",
                frame_idx=i,
                timestamp_sec=i / self._fps,
                team_id=prev.owner_team_id,
                player_id=prev.owner_player_id,
                confidence=min(0.9, prev.owner_confidence),
                features={
                    "speed_mps": round(speed, 2),
                    "distance_to_goal": round(dist_to_goal, 2),
                    "angle_rad": round(angle_to_goal, 4),
                    "on_target": on_target,
                },
            ))

        return shots
