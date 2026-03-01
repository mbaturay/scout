"""Stats Aggregator — orchestrates all stat modules and produces final summaries."""

from __future__ import annotations

import logging
from typing import Any

from ..data_models import FrameState
from .physical import PhysicalStats
from .spatial import SpatialStats
from .ball_progression import BallProgressionStats
from .pressure import PressureStats
from .threat import ThreatStats

logger = logging.getLogger(__name__)


class StatsAggregator:
    """Orchestrates all stat modules: per-frame update + final aggregation."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.physical = PhysicalStats(config)
        self.spatial = SpatialStats(config)
        self.ball_progression = BallProgressionStats(config)
        self.pressure = PressureStats(config)
        self.threat = ThreatStats(config)

    def update(self, frame_state: FrameState) -> FrameState:
        """Run all stat modules on a single frame."""
        frame_state = self.physical.update(frame_state)
        frame_state = self.spatial.update(frame_state)
        frame_state = self.ball_progression.update(frame_state)
        frame_state = self.pressure.update(frame_state)
        frame_state = self.threat.update(frame_state)
        return frame_state

    def get_player_summary(self) -> dict[int, dict[str, Any]]:
        """Aggregate per-player stats across all modules."""
        return self.physical.get_player_summary()

    def get_team_summary(self) -> dict[str, Any]:
        """Aggregate per-team stats across all modules."""
        return {
            "physical": self.physical.get_team_summary(),
            "spatial": self.spatial.get_team_summary(),
            "ball_progression": self.ball_progression.get_team_summary(),
            "pressure": self.pressure.get_team_summary(),
            "threat": self.threat.get_team_summary(),
        }

    def get_rolling_summary(self) -> dict[str, Any]:
        """Rolling-window summaries."""
        return {
            "spatial": self.spatial.get_rolling_summary(),
        }

    def get_full_report(self) -> dict[str, Any]:
        """Complete stats report."""
        return {
            "player_summary": {
                str(k): v for k, v in self.get_player_summary().items()
            },
            "team_summary": self.get_team_summary(),
            "rolling_summary": self.get_rolling_summary(),
        }
