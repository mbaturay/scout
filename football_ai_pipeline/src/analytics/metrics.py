"""Team and player metrics, xG model, heatmap generation.

Computes aggregate statistics from the ownership timeline, detected events,
and per-frame tracking data.  Generates heatmap images (PNG) using matplotlib.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .association import OwnerRecord
from .events import MatchEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# xG model v1 — logistic regression on distance and angle
# ---------------------------------------------------------------------------
# Coefficients inspired by publicly available research (Karun Singh, StatsBomb).
# xG = sigmoid(b0 + b1*distance + b2*angle)
_XG_B0 = 1.10
_XG_B1 = -0.10   # further from goal → lower xG
_XG_B2 = 0.80    # wider angle → higher xG


def compute_xg(distance_m: float, angle_rad: float) -> float:
    """Simple logistic xG from distance to goal centre and angle.

    Args:
        distance_m: Euclidean distance from shot location to goal centre (m).
        angle_rad:  Angle subtended by the goal posts from the shot location.

    Returns:
        xG value in [0, 1].
    """
    z = _XG_B0 + _XG_B1 * distance_m + _XG_B2 * angle_rad
    return 1.0 / (1.0 + math.exp(-z))


def _goal_angle(x: float, y: float, goal_x: float, pitch_width: float = 68.0) -> float:
    """Angle subtended by goal posts (7.32 m wide, centred) from (x, y)."""
    goal_half = 3.66
    goal_cy = pitch_width / 2.0
    post_top = (goal_x, goal_cy - goal_half)
    post_bot = (goal_x, goal_cy + goal_half)

    dx1, dy1 = post_top[0] - x, post_top[1] - y
    dx2, dy2 = post_bot[0] - x, post_bot[1] - y

    dot = dx1 * dx2 + dy1 * dy2
    mag1 = math.sqrt(dx1 * dx1 + dy1 * dy1) + 1e-9
    mag2 = math.sqrt(dx2 * dx2 + dy2 * dy2) + 1e-9
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.acos(cos_angle)


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------

def _build_heatmap(
    positions: list[tuple[float, float]],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    grid_x: int = 21,
    grid_y: int = 14,
) -> np.ndarray:
    """2D histogram of positions on the pitch."""
    grid = np.zeros((grid_y, grid_x), dtype=np.float64)
    for x, y in positions:
        gx = int(x / pitch_length * (grid_x - 1))
        gy = int(y / pitch_width * (grid_y - 1))
        gx = max(0, min(grid_x - 1, gx))
        gy = max(0, min(grid_y - 1, gy))
        grid[gy, gx] += 1.0
    return grid


def _save_heatmap_png(
    grid: np.ndarray,
    path: Path,
    title: str = "",
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> None:
    """Render a heatmap grid to a PNG file using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping heatmap PNG: %s", path)
        return

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    extent = [0, pitch_length, pitch_width, 0]
    ax.imshow(grid, cmap="YlOrRd", interpolation="bilinear",
              aspect="auto", extent=extent)
    # Pitch outline
    ax.plot([0, pitch_length, pitch_length, 0, 0],
            [0, 0, pitch_width, pitch_width, 0], "w-", linewidth=1)
    ax.axvline(pitch_length / 2, color="w", linewidth=0.5, linestyle="--")
    ax.set_xlim(0, pitch_length)
    ax.set_ylim(pitch_width, 0)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Length (m)")
    ax.set_ylabel("Width (m)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metrics computer
# ---------------------------------------------------------------------------

class MetricsComputer:
    """Compute team + player aggregate metrics from analytics data.

    Parameters (from config):
        pitch.length, pitch.width
        analytics.left_to_right
        analytics.press_radius_m
        analytics.heatmap_grid_x / heatmap_grid_y
        analytics.top_n_player_heatmaps
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.pitch_length: float = config.get("pitch", {}).get("length", 105.0)
        self.pitch_width: float = config.get("pitch", {}).get("width", 68.0)
        acfg = config.get("analytics", {})
        self.left_to_right: bool = acfg.get("left_to_right", True)
        self.press_radius: float = acfg.get("press_radius_m", 10.0)
        self.grid_x: int = acfg.get("heatmap_grid_x", 21)
        self.grid_y: int = acfg.get("heatmap_grid_y", 14)
        self.top_n_heatmaps: int = acfg.get("top_n_player_heatmaps", 5)

    def compute(
        self,
        ownership: list[OwnerRecord],
        events: list[MatchEvent],
        per_frame_players: list[dict[int, dict[str, Any]]],
        per_frame_ball: list[Optional[tuple[float, float]]],
        fps: float = 30.0,
    ) -> dict[str, Any]:
        """Compute all metrics.

        Args:
            ownership:         per-frame OwnerRecord
            events:            list of MatchEvent
            per_frame_players: per-frame {track_id: {team_id, x, y, speed_mps, is_pitch}}
            per_frame_ball:    per-frame (x, y) or None
            fps:               frames per second

        Returns:
            dict with keys: team_stats, player_stats, heatmaps (grid data),
                            warnings, coverage.
        """
        team_stats = self._compute_team_stats(ownership, events, per_frame_players, per_frame_ball)
        player_stats = self._compute_player_stats(events, per_frame_players, fps)
        coverage = self._compute_coverage(ownership, per_frame_ball)

        return {
            "team_stats": team_stats,
            "player_stats": player_stats,
            "coverage": coverage,
            "warnings": coverage.get("warnings", []),
        }

    # ---- Team stats ----

    def _compute_team_stats(
        self,
        ownership: list[OwnerRecord],
        events: list[MatchEvent],
        per_frame_players: list[dict[int, dict[str, Any]]],
        per_frame_ball: list[Optional[tuple[float, float]]],
    ) -> dict[int, dict[str, Any]]:
        # Discover teams
        team_ids: set[int] = set()
        for rec in ownership:
            if rec.owner_team_id is not None:
                team_ids.add(rec.owner_team_id)
        for frame_p in per_frame_players:
            for pid, pinfo in frame_p.items():
                tid = pinfo.get("team_id")
                if tid is not None and tid >= 0:
                    team_ids.add(tid)

        if not team_ids:
            team_ids = {0, 1}

        # Possession
        frames_with_owner = sum(1 for r in ownership if r.owner_player_id is not None)
        team_own_frames: dict[int, int] = defaultdict(int)
        for r in ownership:
            if r.owner_team_id is not None:
                team_own_frames[r.owner_team_id] += 1

        # Event counts per team
        team_events: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        pass_completed: dict[int, int] = defaultdict(int)
        shot_events: dict[int, list[MatchEvent]] = defaultdict(list)

        for ev in events:
            if ev.team_id is not None:
                team_events[ev.team_id][ev.event_type] += 1
                if ev.event_type == "pass":
                    # Completed if a reception follows (already paired in detection)
                    pass_completed[ev.team_id] += 1
                if ev.event_type == "shot":
                    shot_events[ev.team_id].append(ev)

        # xG per shot
        attack_goal_x = self.pitch_length if self.left_to_right else 0.0

        result: dict[int, dict[str, Any]] = {}
        for tid in sorted(team_ids):
            poss_pct = (team_own_frames[tid] / max(1, frames_with_owner)) * 100.0
            passes = team_events[tid].get("pass", 0)
            completions = pass_completed[tid]
            comp_pct = (completions / max(1, passes)) * 100.0 if passes else 0.0

            # xG
            xg_total = 0.0
            for sev in shot_events[tid]:
                dist = sev.features.get("distance_to_goal", 20.0)
                angle = sev.features.get("angle_rad", 0.3)
                if angle == 0.0:
                    angle = 0.1
                xg_total += compute_xg(dist, angle)

            # Territory: avg x-position of team centroid
            cx_sum, cx_count = 0.0, 0
            for frame_p in per_frame_players:
                xs = [
                    v["x"] for v in frame_p.values()
                    if v.get("team_id") == tid and v.get("is_pitch", False)
                ]
                if xs:
                    cx_sum += sum(xs) / len(xs)
                    cx_count += 1
            territory = cx_sum / max(1, cx_count)

            # Press intensity: avg defenders within R of ball when opponent owns
            press_frames, press_sum = 0, 0.0
            for idx, rec in enumerate(ownership):
                if rec.owner_team_id is None or rec.owner_team_id == tid:
                    continue
                bp = per_frame_ball[idx] if idx < len(per_frame_ball) else None
                if bp is None:
                    continue
                fp = per_frame_players[idx] if idx < len(per_frame_players) else {}
                defenders_near = 0
                for pid, pinfo in fp.items():
                    if pinfo.get("team_id") != tid:
                        continue
                    dx = pinfo["x"] - bp[0]
                    dy = pinfo["y"] - bp[1]
                    if math.sqrt(dx * dx + dy * dy) <= self.press_radius:
                        defenders_near += 1
                press_sum += defenders_near
                press_frames += 1
            press_intensity = press_sum / max(1, press_frames)

            # Heatmap positions
            positions = []
            for frame_p in per_frame_players:
                for pid, pinfo in frame_p.items():
                    if pinfo.get("team_id") == tid and pinfo.get("is_pitch", False):
                        positions.append((pinfo["x"], pinfo["y"]))

            heatmap = _build_heatmap(
                positions, self.pitch_length, self.pitch_width,
                self.grid_x, self.grid_y,
            )

            result[tid] = {
                "possession_pct": round(poss_pct, 1),
                "pass_count": passes,
                "pass_completed": completions,
                "pass_completion_pct": round(comp_pct, 1),
                "shots_count": len(shot_events[tid]),
                "xG_total": round(xg_total, 3),
                "territory_avg_x": round(territory, 2),
                "press_intensity": round(press_intensity, 3),
                "interceptions": team_events[tid].get("interception", 0),
                "tackles": team_events[tid].get("tackle", 0),
                "touches": team_events[tid].get("touch", 0),
                "heatmap_grid": heatmap.tolist(),
            }

        return result

    # ---- Player stats ----

    def _compute_player_stats(
        self,
        events: list[MatchEvent],
        per_frame_players: list[dict[int, dict[str, Any]]],
        fps: float,
    ) -> dict[int, dict[str, Any]]:
        # Accumulate per-player data
        player_team: dict[int, int] = {}
        player_distances: dict[int, float] = defaultdict(float)
        player_speeds: dict[int, list[float]] = defaultdict(list)
        player_positions: dict[int, list[tuple[float, float]]] = defaultdict(list)
        prev_pos: dict[int, tuple[float, float]] = {}

        for frame_p in per_frame_players:
            for pid, pinfo in frame_p.items():
                tid = pinfo.get("team_id")
                if tid is not None and tid >= 0:
                    player_team[pid] = tid

                if pinfo.get("is_pitch", False):
                    pos = (pinfo["x"], pinfo["y"])
                    player_positions[pid].append(pos)
                    speed = pinfo.get("speed_mps")
                    if speed is not None:
                        player_speeds[pid].append(speed)
                    if pid in prev_pos:
                        dx = pos[0] - prev_pos[pid][0]
                        dy = pos[1] - prev_pos[pid][1]
                        player_distances[pid] += math.sqrt(dx * dx + dy * dy)
                    prev_pos[pid] = pos

        # Event counts per player
        player_events: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for ev in events:
            if ev.player_id is not None:
                player_events[ev.player_id][ev.event_type] += 1

        result: dict[int, dict[str, Any]] = {}
        for pid in sorted(set(player_team.keys()) | set(player_distances.keys())):
            speeds = player_speeds.get(pid, [])
            positions = player_positions.get(pid, [])
            is_pitch = len(positions) > 0

            heatmap = None
            if positions:
                heatmap = _build_heatmap(
                    positions, self.pitch_length, self.pitch_width,
                    self.grid_x, self.grid_y,
                )

            evts = player_events.get(pid, {})
            result[pid] = {
                "team_id": player_team.get(pid),
                "distance_covered_m": round(player_distances.get(pid, 0.0), 1),
                "avg_speed_mps": round(float(np.mean(speeds)), 2) if speeds else 0.0,
                "top_speed_mps": round(max(speeds), 2) if speeds else 0.0,
                "confidence": "pitch" if is_pitch else "pixel",
                "touches": evts.get("touch", 0),
                "receptions": evts.get("reception", 0),
                "passes": evts.get("pass", 0),
                "interceptions": evts.get("interception", 0),
                "tackles": evts.get("tackle", 0),
                "shots": evts.get("shot", 0),
                "heatmap_grid": heatmap.tolist() if heatmap is not None else None,
            }

        return result

    # ---- Coverage ----

    def _compute_coverage(
        self,
        ownership: list[OwnerRecord],
        per_frame_ball: list[Optional[tuple[float, float]]],
    ) -> dict[str, Any]:
        total = len(ownership)
        ball_detected = sum(1 for r in ownership if r.ball_available)
        owner_assigned = sum(1 for r in ownership if r.owner_player_id is not None)
        high_conf = sum(1 for r in ownership if r.owner_confidence >= 0.5)

        warnings: list[str] = []
        ball_pct = ball_detected / max(1, total) * 100
        owner_pct = owner_assigned / max(1, total) * 100

        if ball_pct < 30:
            warnings.append(
                f"Ball detected in only {ball_pct:.0f}% of frames — "
                "possession and event stats will be unreliable. "
                "Consider using a football-specific detection model."
            )
        if owner_pct < 20:
            warnings.append(
                f"Ball owner assigned in only {owner_pct:.0f}% of frames — "
                "pass/shot/tackle counts may be incomplete."
            )

        return {
            "total_frames": total,
            "ball_detected_pct": round(ball_pct, 1),
            "owner_assigned_pct": round(owner_pct, 1),
            "high_confidence_pct": round(high_conf / max(1, total) * 100, 1),
            "warnings": warnings,
        }

    # ---- Export helpers ----

    def write_outputs(
        self,
        output_dir: Path,
        team_stats: dict[int, dict[str, Any]],
        player_stats: dict[int, dict[str, Any]],
        events: list[MatchEvent],
        ball_positions: list[Optional[tuple[float, float]]],
    ) -> None:
        """Write all analytics outputs to disk."""
        stats_dir = output_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir = output_dir / "heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        # Team stats CSV (legacy) + JSON contract
        self._write_team_csv(stats_dir / "team_stats.csv", team_stats)
        self._write_team_json(output_dir / "team_stats.json", team_stats)

        # Player stats CSV (legacy) + JSON contract
        self._write_player_csv(stats_dir / "player_stats.csv", player_stats)
        self._write_player_json(output_dir / "player_stats.json", player_stats)

        # Events JSON — always written, even if empty
        events_data = [e.to_dict() for e in events]
        with open(output_dir / "events.json", "w", encoding="utf-8") as f:
            json.dump({"events": events_data}, f, indent=2)
        logger.info("Wrote %d events to events.json", len(events_data))

        # Team heatmaps
        for tid, ts in team_stats.items():
            grid = ts.get("heatmap_grid")
            if grid is not None:
                _save_heatmap_png(
                    np.array(grid),
                    heatmap_dir / f"team_{tid}.png",
                    title=f"Team {tid} Heatmap",
                    pitch_length=self.pitch_length,
                    pitch_width=self.pitch_width,
                )

        # Ball heatmap
        ball_pos = [p for p in ball_positions if p is not None]
        if ball_pos:
            ball_grid = _build_heatmap(
                ball_pos, self.pitch_length, self.pitch_width,
                self.grid_x, self.grid_y,
            )
            _save_heatmap_png(
                ball_grid,
                heatmap_dir / "ball.png",
                title="Ball Heatmap",
                pitch_length=self.pitch_length,
                pitch_width=self.pitch_width,
            )

        # Top-N player heatmaps
        sorted_players = sorted(
            player_stats.items(),
            key=lambda kv: kv[1].get("touches", 0) + kv[1].get("passes", 0),
            reverse=True,
        )
        for pid, ps in sorted_players[: self.top_n_heatmaps]:
            grid = ps.get("heatmap_grid")
            if grid is not None:
                _save_heatmap_png(
                    np.array(grid),
                    heatmap_dir / f"player_{pid}.png",
                    title=f"Player {pid} (Team {ps.get('team_id', '?')})",
                    pitch_length=self.pitch_length,
                    pitch_width=self.pitch_width,
                )

        logger.info("Analytics outputs written to %s", output_dir)

    @staticmethod
    def _write_team_csv(path: Path, team_stats: dict[int, dict[str, Any]]) -> None:
        import csv

        # Exclude heatmap_grid from CSV
        exclude = {"heatmap_grid"}
        if not team_stats:
            return

        fieldnames = ["team_id"]
        sample = next(iter(team_stats.values()))
        fieldnames.extend(k for k in sample if k not in exclude)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for tid in sorted(team_stats):
                row = {"team_id": tid}
                row.update({k: v for k, v in team_stats[tid].items() if k not in exclude})
                writer.writerow(row)

    @staticmethod
    def _write_player_csv(path: Path, player_stats: dict[int, dict[str, Any]]) -> None:
        import csv

        exclude = {"heatmap_grid"}
        if not player_stats:
            return

        fieldnames = ["track_id"]
        sample = next(iter(player_stats.values()))
        fieldnames.extend(k for k in sample if k not in exclude)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for pid in sorted(player_stats):
                row = {"track_id": pid}
                row.update({k: v for k, v in player_stats[pid].items() if k not in exclude})
                writer.writerow(row)

    # ---- JSON contract writers (always produce a file, even if empty) ----

    @staticmethod
    def _write_team_json(path: Path, team_stats: dict[int, dict[str, Any]]) -> None:
        exclude = {"heatmap_grid"}
        teams: list[dict[str, Any]] = []
        for tid in sorted(team_stats):
            row = {"team_id": tid}
            row.update({k: v for k, v in team_stats[tid].items() if k not in exclude})
            teams.append(row)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"teams": teams}, f, indent=2)

    @staticmethod
    def _write_player_json(path: Path, player_stats: dict[int, dict[str, Any]]) -> None:
        exclude = {"heatmap_grid"}
        players: list[dict[str, Any]] = []
        for pid in sorted(player_stats):
            row = {"track_id": pid}
            row.update({k: v for k, v in player_stats[pid].items() if k not in exclude})
            players.append(row)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"players": players}, f, indent=2)
