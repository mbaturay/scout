#!/usr/bin/env python3
"""Inspect out/frames.jsonl to document the data contract.

Usage:
    cd football_ai_pipeline
    python scripts/inspect_frames.py [path/to/frames.jsonl] [--max-lines N]

Prints schema info, ball detection stats, and analytics coverage
without modifying any pipeline code.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def _load_frames(path: Path, max_lines: int = 200) -> list[dict]:
    frames: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main() -> None:
    # --- Parse args ---
    args = sys.argv[1:]
    max_lines = 200
    path = Path("out/frames.jsonl")

    i = 0
    while i < len(args):
        if args[i] == "--max-lines" and i + 1 < len(args):
            max_lines = int(args[i + 1])
            i += 2
        else:
            path = Path(args[i])
            i += 1

    if not path.exists():
        print(f"ERROR: {path} not found.")
        print("Run the pipeline first, or pass the path as an argument.")
        sys.exit(1)

    # Count total lines
    with open(path, encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    n_read = min(max_lines, total_lines)

    print(f"Reading {n_read} / {total_lines} frames from: {path.resolve()}")
    frames = _load_frames(path, max_lines)

    if not frames:
        print("No frames found.")
        sys.exit(1)

    # ---------------------------------------------------------------
    # 1) Top-level schema
    # ---------------------------------------------------------------
    _print_section("TOP-LEVEL KEYS")
    sample = frames[0]
    for key in sample:
        val = sample[key]
        vtype = type(val).__name__
        if isinstance(val, list):
            inner = type(val[0]).__name__ if val else "empty"
            print(f"  {key:20s}  list[{inner}]  (len={len(val)})")
        elif isinstance(val, dict):
            subkeys = list(val.keys())
            print(f"  {key:20s}  dict  keys={subkeys}")
        else:
            print(f"  {key:20s}  {vtype}  = {val!r}")

    # ---------------------------------------------------------------
    # 2) Ball detection representation
    # ---------------------------------------------------------------
    _print_section("BALL FIELD SCHEMA")
    ball_sample = sample.get("ball")
    if ball_sample is None:
        print("  ball field is NULL in first frame")
    else:
        for k, v in ball_sample.items():
            print(f"  ball.{k:20s}  {type(v).__name__:8s}  = {v!r}")

    # Check if ball appears in players list (class="ball")
    ball_in_players = [p for p in sample.get("players", []) if p.get("class") == "ball"]
    if ball_in_players:
        print(f"\n  Ball also appears in players[] with class='ball':")
        bp = ball_in_players[0]
        for k, v in bp.items():
            print(f"    players[].{k:16s}  = {v!r}")
    else:
        print(f"\n  Ball does NOT appear in players[] (separate 'ball' field only)")

    # ---------------------------------------------------------------
    # 3) Ball statistics across frames
    # ---------------------------------------------------------------
    _print_section("BALL STATISTICS")

    n_ball_field_present = 0
    n_ball_has_pitch = 0
    n_ball_has_bbox = 0
    n_ball_has_speed = 0
    n_ball_interpolated = 0
    ball_confidences: list[float] = []
    ball_track_ids: set[int] = set()

    for f in frames:
        ball = f.get("ball")
        if ball is None:
            continue
        n_ball_field_present += 1

        if ball.get("pitch_x") is not None and ball.get("pitch_y") is not None:
            n_ball_has_pitch += 1
        if ball.get("bbox") is not None:
            n_ball_has_bbox += 1
        if ball.get("speed_mps") is not None:
            n_ball_has_speed += 1
        if ball.get("interpolated"):
            n_ball_interpolated += 1
        if ball.get("confidence") is not None:
            ball_confidences.append(ball["confidence"])
        if ball.get("track_id") is not None:
            ball_track_ids.add(ball["track_id"])

        # Also check players[] for class=ball
        for p in f.get("players", []):
            if p.get("class") == "ball" and p.get("track_id") is not None:
                ball_track_ids.add(p["track_id"])

    pct = lambda n: f"{n / len(frames) * 100:.1f}%" if frames else "N/A"

    print(f"  Frames analyzed:          {len(frames)}")
    print(f"  ball field present:       {n_ball_field_present}  ({pct(n_ball_field_present)})")
    print(f"  ball has pitch coords:    {n_ball_has_pitch}  ({pct(n_ball_has_pitch)})")
    print(f"  ball has bbox:            {n_ball_has_bbox}  ({pct(n_ball_has_bbox)})")
    print(f"  ball has speed_mps:       {n_ball_has_speed}  ({pct(n_ball_has_speed)})")
    print(f"  ball interpolated:        {n_ball_interpolated}  ({pct(n_ball_interpolated)})")
    print(f"  unique ball track_ids:    {len(ball_track_ids)}  {sorted(ball_track_ids) if ball_track_ids else '(none)'}")

    if ball_confidences:
        avg_conf = sum(ball_confidences) / len(ball_confidences)
        min_conf = min(ball_confidences)
        max_conf = max(ball_confidences)
        print(f"  ball confidence:          avg={avg_conf:.3f}  min={min_conf:.3f}  max={max_conf:.3f}  (n={len(ball_confidences)})")
    else:
        print(f"  ball confidence:          no confidence values found")

    # ---------------------------------------------------------------
    # 4) Analytics / ball_owner coverage
    # ---------------------------------------------------------------
    _print_section("ANALYTICS — BALL OWNER")

    n_has_analytics = 0
    n_has_ball_owner = 0
    n_owner_assigned = 0
    n_ball_available = 0
    owner_confs: list[float] = []
    owner_players: Counter[int | None] = Counter()
    owner_teams: Counter[int | None] = Counter()

    for f in frames:
        analytics = f.get("analytics")
        if analytics is None:
            continue
        n_has_analytics += 1

        bo = analytics.get("ball_owner")
        if bo is None:
            continue
        n_has_ball_owner += 1

        if bo.get("ball_available"):
            n_ball_available += 1
        if bo.get("owner_player_id") is not None:
            n_owner_assigned += 1
            owner_players[bo["owner_player_id"]] += 1
            owner_teams[bo.get("owner_team_id")] += 1
        owner_confs.append(bo.get("owner_confidence", 0.0))

    print(f"  analytics field present:  {n_has_analytics}  ({pct(n_has_analytics)})")
    print(f"  ball_owner present:       {n_has_ball_owner}  ({pct(n_has_ball_owner)})")
    print(f"  ball_available=true:      {n_ball_available}  ({pct(n_ball_available)})")
    print(f"  owner assigned:           {n_owner_assigned}  ({pct(n_owner_assigned)})")

    if owner_confs:
        avg_oc = sum(owner_confs) / len(owner_confs)
        print(f"  owner_confidence:         avg={avg_oc:.3f}  max={max(owner_confs):.3f}")
    if owner_players:
        top5 = owner_players.most_common(5)
        print(f"  top owner players:        {top5}")
    if owner_teams:
        print(f"  owner teams dist:         {dict(owner_teams)}")

    # ---------------------------------------------------------------
    # 5) Player field schema
    # ---------------------------------------------------------------
    _print_section("PLAYER SCHEMA (first player in first frame)")
    players = sample.get("players", [])
    if players:
        p = players[0]
        for k, v in p.items():
            print(f"  players[].{k:16s}  {type(v).__name__:8s}  = {v!r}")
        print(f"\n  Total players in frame 0: {len(players)}")

        # Player class distribution
        class_counts: Counter[str] = Counter()
        team_counts: Counter[int | None] = Counter()
        n_with_pitch = 0
        for f in frames:
            for p in f.get("players", []):
                class_counts[p.get("class", "?")] += 1
                team_counts[p.get("team_id")] += 1
                if p.get("pitch_x") is not None:
                    n_with_pitch += 1
        total_player_entries = sum(class_counts.values())
        print(f"  player class distribution: {dict(class_counts)}")
        print(f"  team_id distribution:      {dict(team_counts)}")
        print(f"  players with pitch coords: {n_with_pitch} / {total_player_entries} ({n_with_pitch / total_player_entries * 100:.1f}%)" if total_player_entries else "")

    # ---------------------------------------------------------------
    # 6) Analytics sub-modules present
    # ---------------------------------------------------------------
    _print_section("ANALYTICS SUB-MODULES")
    analytics = sample.get("analytics", {})
    for mod_name, mod_val in analytics.items():
        if isinstance(mod_val, dict):
            subkeys = list(mod_val.keys())
            non_empty = {k: v for k, v in mod_val.items() if v}
            print(f"  {mod_name:20s}  keys={subkeys}")
            if non_empty:
                for k, v in non_empty.items():
                    print(f"    .{k} = {v!r}")
        else:
            print(f"  {mod_name:20s}  = {mod_val!r}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    _print_section("DATA CONTRACT SUMMARY")
    print(f"""
  Frame schema:
    frame_idx          int
    timestamp_sec      float
    flag               "in_play" | "not_in_play"
    flag_reasons       list[str]
    homography         dict (matrix, quality, num_inliers, available)
    players            list[dict]  — tracked players/goalkeepers/referees
    ball               dict (pitch_x, pitch_y, speed_mps, interpolated, bbox?, confidence?)
    analytics          dict (ball_owner, physical, spatial, ball_progression, pressure, threat)

  Ball possession contract:
    analytics.ball_owner.owner_player_id   int | null
    analytics.ball_owner.owner_team_id     int | null
    analytics.ball_owner.owner_confidence  float (0.0 – 1.0)
    analytics.ball_owner.ball_available    bool
    analytics.ball_owner.distance          float | null

  Key findings for possession/pass detection:
    - Ball detected in {pct(n_ball_has_pitch)} of frames (pitch coords)
    - Ball owner assigned in {pct(n_owner_assigned)} of frames
    - Ball available in {pct(n_ball_available)} of frames
    - {len(owner_players)} unique players held possession
    """)


if __name__ == "__main__":
    main()
