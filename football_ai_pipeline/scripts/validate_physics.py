"""Validate motion-physics outputs from a pipeline run.

Reads out/frames.jsonl and out/stats/player_stats.csv, prints diagnostics,
and exits with code 1 if any sanity check fails.

Usage:
    python scripts/validate_physics.py [--output-dir out]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

FPS_MIN = 10
FPS_MAX = 60
MAX_SPEED_LIMIT = 25.0        # absolute hard cap (m/s)
P95_SPEED_LIMIT = 15.0        # 95th-percentile cap (m/s)
EMA_ALPHA = 0.35              # EMA smoothing factor (matches pipeline default)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile(vals: list[float], pct: float) -> float:
    """Simple percentile without numpy."""
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * pct / 100.0
    lo = int(math.floor(k))
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _median(vals: list[float]) -> float:
    return _percentile(vals, 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate motion physics outputs")
    parser.add_argument(
        "--output-dir", "-o", default="out",
        help="Pipeline output directory (default: out)",
    )
    args = parser.parse_args()

    pkg_dir = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = pkg_dir / out_dir

    frames_path = out_dir / "frames.jsonl"
    if not frames_path.exists():
        print(f"ERROR: {frames_path} not found")
        return 1

    # ------------------------------------------------------------------
    # Parse frames.jsonl
    # ------------------------------------------------------------------
    timestamps: list[float] = []
    # Per-player: list of (frame_idx, timestamp, pitch_x, pitch_y, speed_mps)
    player_samples: dict[int, list[tuple[int, float, float, float, float | None]]] = defaultdict(list)
    ball_detected_count = 0
    player_pitch_count = 0
    player_total_count = 0
    total_frames = 0

    # For outlier tracking: (frame_idx, track_id, speed, dx, dy, dt)
    all_speed_records: list[tuple[int, int, float, float, float, float]] = []

    with open(frames_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            total_frames += 1
            ts = frame.get("timestamp_sec", 0.0)
            timestamps.append(ts)
            fidx = frame.get("frame_idx", total_frames - 1)

            # Ball
            ball = frame.get("ball")
            if ball and (ball.get("pitch_x") is not None or ball.get("pitch_y") is not None):
                ball_detected_count += 1

            # Players
            for p in frame.get("players", []):
                player_total_count += 1
                tid = p.get("track_id")
                px = p.get("pitch_x")
                py = p.get("pitch_y")
                spd = p.get("speed_mps")
                if px is not None and py is not None:
                    player_pitch_count += 1
                    player_samples[tid].append((fidx, ts, px, py, spd))

    if total_frames < 2:
        print(f"ERROR: Only {total_frames} frames in {frames_path}")
        return 1

    # ------------------------------------------------------------------
    # FPS estimate from timestamp deltas
    # ------------------------------------------------------------------
    dts_global: list[float] = []
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt > 0:
            dts_global.append(dt)

    if not dts_global:
        print("ERROR: No valid timestamp deltas found")
        return 1

    median_dt = _median(dts_global)
    fps_est = 1.0 / median_dt if median_dt > 0 else 0.0

    print("=" * 65)
    print("MOTION PHYSICS VALIDATION")
    print("=" * 65)
    print(f"Frames parsed:    {total_frames}")
    print(f"Timestamp deltas:  min={min(dts_global):.6f}s  "
          f"median={median_dt:.6f}s  max={max(dts_global):.6f}s")
    print(f"FPS estimate:      {fps_est:.1f} Hz  "
          f"(range: {1/max(dts_global):.1f} – {1/min(dts_global):.1f})")
    print()

    # ------------------------------------------------------------------
    # Per-player analysis
    # ------------------------------------------------------------------
    print(f"{'Track':>6}  {'Samples':>8}  {'Med dt':>8}  {'Med step':>10}  "
          f"{'p95 spd':>9}  {'Max spd':>9}")
    print("-" * 65)

    global_max_speed = 0.0
    global_p95_speed = 0.0
    all_player_speeds: list[float] = []

    for tid in sorted(player_samples.keys()):
        samples = player_samples[tid]
        if len(samples) < 2:
            continue

        player_dts: list[float] = []
        player_dists: list[float] = []
        player_speeds: list[float] = []

        for i in range(1, len(samples)):
            fidx_prev, ts_prev, x_prev, y_prev, _ = samples[i - 1]
            fidx_cur, ts_cur, x_cur, y_cur, spd_cur = samples[i]
            dt = ts_cur - ts_prev
            if dt <= 0:
                continue
            dx = x_cur - x_prev
            dy = y_cur - y_prev
            dist = math.sqrt(dx * dx + dy * dy)
            computed_speed = dist / dt

            player_dts.append(dt)
            player_dists.append(dist)
            player_speeds.append(computed_speed)
            all_player_speeds.append(computed_speed)

            all_speed_records.append((fidx_cur, tid, computed_speed, dx, dy, dt))

        if not player_speeds:
            continue

        med_dt = _median(player_dts)
        med_step = _median(player_dists)
        p95 = _percentile(player_speeds, 95)
        mx = max(player_speeds)

        if mx > global_max_speed:
            global_max_speed = mx
        if p95 > global_p95_speed:
            global_p95_speed = p95

        print(f"{tid:>6}  {len(samples):>8}  {med_dt:>8.4f}  {med_step:>10.3f}  "
              f"{p95:>9.2f}  {mx:>9.2f}")

    print()

    # ------------------------------------------------------------------
    # EMA-smoothed speed analysis
    # ------------------------------------------------------------------
    ema_all_speeds: list[float] = []
    ema_state: dict[int, tuple[float, float]] = {}
    a = EMA_ALPHA

    for tid in sorted(player_samples.keys()):
        samples = player_samples[tid]
        if len(samples) < 2:
            continue

        # Reset EMA state per track
        prev_sm: tuple[float, float] | None = None

        for i, (fidx, ts, x, y, _spd) in enumerate(samples):
            if prev_sm is None:
                prev_sm = (x, y)
                prev_ts = ts
                continue
            sx = a * x + (1.0 - a) * prev_sm[0]
            sy = a * y + (1.0 - a) * prev_sm[1]
            dt = ts - prev_ts
            if dt > 0:
                dx = sx - prev_sm[0]
                dy = sy - prev_sm[1]
                dist = math.sqrt(dx * dx + dy * dy)
                ema_all_speeds.append(dist / dt)
            prev_sm = (sx, sy)
            prev_ts = ts

    ema_max = max(ema_all_speeds) if ema_all_speeds else 0.0
    ema_p95 = _percentile(ema_all_speeds, 95) if ema_all_speeds else 0.0

    # ------------------------------------------------------------------
    # Global stats
    # ------------------------------------------------------------------
    pitch_pct = (player_pitch_count / max(player_total_count, 1)) * 100
    ball_pct = (ball_detected_count / max(total_frames, 1)) * 100

    overall_p95 = _percentile(all_player_speeds, 95) if all_player_speeds else 0.0
    overall_max = max(all_player_speeds) if all_player_speeds else 0.0

    print("GLOBAL SUMMARY")
    print("-" * 65)
    print(f"Players with pitch coords:  {player_pitch_count}/{player_total_count} "
          f"({pitch_pct:.1f}%)")
    print(f"Frames with ball detected:  {ball_detected_count}/{total_frames} "
          f"({ball_pct:.1f}%)")
    print(f"Overall 95th-pct speed:     {overall_p95:.2f} m/s")
    print(f"Overall max speed:          {overall_max:.2f} m/s")
    print(f"EMA-smoothed max speed:     {ema_max:.2f} m/s  (alpha={a})")
    print(f"EMA-smoothed 95th-pct:      {ema_p95:.2f} m/s")
    print()

    # ------------------------------------------------------------------
    # Top 20 speed outliers
    # ------------------------------------------------------------------
    all_speed_records.sort(key=lambda r: r[2], reverse=True)
    top_outliers = all_speed_records[:20]
    if top_outliers:
        print("TOP 20 SPEED OUTLIERS")
        print(f"{'Frame':>7}  {'Track':>6}  {'Speed':>9}  {'dx':>8}  {'dy':>8}  {'dt':>8}")
        print("-" * 55)
        for fidx, tid, spd, dx, dy, dt in top_outliers:
            print(f"{fidx:>7}  {tid:>6}  {spd:>9.2f}  {dx:>8.2f}  {dy:>8.2f}  {dt:>8.4f}")
        print()

    # ------------------------------------------------------------------
    # PASS / FAIL checks
    # ------------------------------------------------------------------
    failures: list[str] = []

    if fps_est < FPS_MIN:
        failures.append(f"FPS too low: {fps_est:.1f} < {FPS_MIN}")
    if fps_est > FPS_MAX:
        failures.append(f"FPS too high: {fps_est:.1f} > {FPS_MAX}")
    if overall_max > MAX_SPEED_LIMIT:
        failures.append(
            f"Max speed {overall_max:.2f} m/s > {MAX_SPEED_LIMIT} m/s limit"
        )
    if overall_p95 > P95_SPEED_LIMIT:
        failures.append(
            f"95th-pct speed {overall_p95:.2f} m/s > {P95_SPEED_LIMIT} m/s limit"
        )

    print("=" * 65)
    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print(f"  - {f}")
        print("=" * 65)
        return 1
    else:
        print("RESULT: PASS")
        print("  All physics checks within expected bounds.")
        print("=" * 65)
        return 0


if __name__ == "__main__":
    sys.exit(main())
