"""FR10 — Exports: write structured outputs to disk.

Outputs:
  - frames.jsonl (or Parquet)
  - players_summary.csv
  - teams_summary.csv
  - stats/ folder with detailed analytics
  - run_report.json
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from ..data_models import FrameState

logger = logging.getLogger(__name__)


class Exporter:
    """Write pipeline outputs to the output directory."""

    def __init__(self, output_dir: str | Path, config: dict[str, Any]) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        exp_cfg = config.get("exports", {})
        self.format: str = exp_cfg.get("format", "jsonl")
        self.write_players: bool = exp_cfg.get("players_summary", True)
        self.write_teams: bool = exp_cfg.get("teams_summary", True)
        self.write_stats: bool = exp_cfg.get("stats_folder", True)
        self.write_report: bool = exp_cfg.get("run_report", True)

        self._frames_file = None
        if self.format == "jsonl":
            frames_path = self.output_dir / "frames.jsonl"
            self._frames_file = open(frames_path, "w")
            logger.info("Writing frames to %s", frames_path)

    def write_frame(self, frame_state: FrameState) -> None:
        """Write one frame's data."""
        if self._frames_file:
            self._frames_file.write(frame_state.to_json() + "\n")

    def write_players_summary(self, player_data: dict[int, dict[str, Any]]) -> None:
        """Write players_summary.csv."""
        if not self.write_players or not player_data:
            return
        path = self.output_dir / "players_summary.csv"
        # Collect all keys
        all_keys: set[str] = set()
        for v in player_data.values():
            all_keys.update(v.keys())
        fieldnames = ["track_id"] + sorted(all_keys)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for track_id, stats in sorted(player_data.items()):
                row = {"track_id": track_id, **stats}
                writer.writerow(row)
        logger.info("Wrote players summary: %s (%d players)", path, len(player_data))

    def write_teams_summary(self, team_data: dict[str, Any]) -> None:
        """Write teams_summary.csv."""
        if not self.write_teams:
            return
        path = self.output_dir / "teams_summary.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["category", "metric", "value"])
            self._flatten_dict_to_csv(writer, team_data)
        logger.info("Wrote teams summary: %s", path)

    def write_stats_folder(self, full_report: dict[str, Any]) -> None:
        """Write detailed stats to stats/ subfolder."""
        if not self.write_stats:
            return
        stats_dir = self.output_dir / "stats"
        stats_dir.mkdir(exist_ok=True)

        for key, value in full_report.items():
            path = stats_dir / f"{key}.json"
            with open(path, "w") as f:
                json.dump(value, f, indent=2, default=str)
        logger.info("Wrote stats to %s", stats_dir)

    def write_run_report(
        self,
        metadata: dict[str, Any],
        coverage: dict[str, Any],
        degradation: dict[str, Any] | None = None,
    ) -> None:
        """Write run_report.json with coverage, quality metrics, and degradation info."""
        if not self.write_report:
            return
        report: dict[str, Any] = {
            "metadata": metadata,
            "coverage": coverage,
        }
        if degradation is not None:
            report["degradation"] = degradation
        path = self.output_dir / "run_report.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Wrote run report: %s", path)

    def close(self) -> None:
        if self._frames_file:
            self._frames_file.close()
            self._frames_file = None

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def _flatten_dict_to_csv(
        writer: Any, data: dict[str, Any], prefix: str = "",
    ) -> None:
        for k, v in data.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                Exporter._flatten_dict_to_csv(writer, v, key)
            else:
                writer.writerow([prefix or k, k, v])
