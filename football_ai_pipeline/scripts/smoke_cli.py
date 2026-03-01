#!/usr/bin/env python
"""CLI smoke test — runs the pipeline in minimal mode with stage markers.

Wraps the existing PipelineRunner with observability hooks so you can see
which stage the pipeline is in if it stalls.

Usage:
    cd football_ai_pipeline
    python scripts/smoke_cli.py --config configs/default.yaml --input test-video-1.mp4 --output out_smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure package imports work
_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_DIR = _SCRIPT_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import yaml

from src.data_models import FrameState
from src.pipeline.runner import PipelineRunner

logger = logging.getLogger("smoke_cli")

# Stage names for diagnostics
_STAGES = [
    "in_play_filter",
    "detection",
    "tracking",
    "team_classifier",
    "keypoints",
    "homography",
    "pitch_transform",
    "stats",
]


class InstrumentedRunner(PipelineRunner):
    """PipelineRunner subclass that prints STAGE markers per frame."""

    def __init__(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        config: dict[str, Any],
    ) -> None:
        super().__init__(video_path, output_dir, config)
        self._last_stage: str = "init"
        self._last_frame_idx: int = -1

    def _process_frame(
        self, frame_idx: int, timestamp: float, image: Any,
    ) -> FrameState:
        """Run all pipeline stages with per-stage timing."""
        self._last_frame_idx = frame_idx
        fs = FrameState(frame_idx=frame_idx, timestamp_sec=timestamp, image=image)

        stages: list[tuple[str, float]] = []

        def _run(name: str, fn: Any, *args: Any) -> Any:
            self._last_stage = name
            t0 = time.perf_counter()
            result = fn(*args)
            dt = time.perf_counter() - t0
            stages.append((name, dt))
            return result

        fs = _run("in_play_filter", self.in_play_filter.classify, fs)
        fs = _run("detection", self.detector.detect, fs)
        fs = _run("tracking", self.tracker.track, fs)
        fs = _run("team_classifier", self.team_classifier.update, fs)
        fs = _run("keypoints", self.keypoint_detector.detect, fs)
        fs = _run("homography", self.homography_estimator.estimate, fs)
        fs = _run("pitch_transform", self.pitch_transformer.transform, fs)
        fs = _run("stats", self.stats_aggregator.update, fs)
        fs = _run("analytics", self.analytics_engine.update, fs)

        # Print per-frame summary every 10 frames
        if frame_idx % 10 == 0:
            parts = " | ".join(f"{n}={dt*1000:.0f}ms" for n, dt in stages)
            n_det = len(fs.detections) if fs.detections else 0
            logger.info(
                "FRAME %d: %d detections | %s",
                frame_idx, n_det, parts,
            )

        self._last_stage = "export"
        return fs


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline smoke test with stage markers")
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default="out_smoke")
    args = parser.parse_args()

    # Resolve config
    config_path = Path(args.config)
    if not config_path.exists():
        alt = _PKG_DIR / args.config
        if alt.exists():
            config_path = alt
        else:
            print(f"ERROR: Config not found: {config_path}")
            sys.exit(1)

    try:
        with open(config_path, "r", encoding="utf-8-sig") as f:
            config = yaml.safe_load(f) or {}
    except UnicodeDecodeError:
        print(
            f"ERROR: Could not decode config file. Please save as UTF-8.\n"
            f"  Path: {config_path.resolve()}")
        sys.exit(1)

    # Force safe defaults for smoke test
    config.setdefault("video", {})["max_frames"] = 200
    config.setdefault("video", {})["stride"] = 2
    config.setdefault("visualization", {})["enabled"] = False

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        alt = _PKG_DIR / args.input
        if alt.exists():
            input_path = alt
        else:
            print(f"ERROR: Video not found: {input_path}")
            sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("  SMOKE TEST — max_frames=200, stride=2, save_video=false")
    print(f"  Input:  {input_path}")
    print(f"  Output: {args.output}")
    print(f"  Start:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    runner = InstrumentedRunner(
        video_path=input_path,
        output_dir=args.output,
        config=config,
    )

    t0 = time.perf_counter()
    try:
        runner.run()
    except KeyboardInterrupt:
        print(f"\n  INTERRUPTED at frame {runner._last_frame_idx}, "
              f"stage: {runner._last_stage}")
        sys.exit(130)
    except Exception as e:
        print(f"\n  FAILED at frame {runner._last_frame_idx}, "
              f"stage: {runner._last_stage}")
        print(f"  Error: {e}")
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    print("=" * 60)
    print(f"  SMOKE TEST PASSED in {elapsed:.1f}s")
    print(f"  End:    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
