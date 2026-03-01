"""Football AI Pipeline — CLI entry point.

Usage:
    python -m football_ai_pipeline --config configs/default.yaml --input <video.mp4> --output <out_dir>

Quickstart (no weights needed):
    python -m football_ai_pipeline --input match.mp4 --max-frames 100 --stride 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure `src` is importable when run as `python -m football_ai_pipeline`
_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import yaml

from src.pipeline.runner import PipelineRunner


def _str_to_bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Football Match Analytics Engine — single broadcast video to advanced stats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Full run with default config
  python -m football_ai_pipeline --config configs/default.yaml --input match.mp4 --output out/

  # Quick debug run (no weights needed, first 100 frames)
  python -m football_ai_pipeline -i match.mp4 --max-frames 100 --stride 5

  # Fast run without annotated video
  python -m football_ai_pipeline -i match.mp4 --save-video false --stride 4
""",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input video file (e.g. match.mp4)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="out",
        help="Output directory (default: out/)",
    )
    # Quickstart / debug overrides
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most N frames (overrides config video.max_frames)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Process every Nth frame (overrides config video.stride)",
    )
    parser.add_argument(
        "--save-video",
        type=_str_to_bool,
        default=None,
        metavar="BOOL",
        help="Write annotated video (true/false, overrides config visualization.enabled)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEVICE",
        help="Compute device: auto, cuda, cuda:0, cpu (overrides config detection.device)",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to package dir (for `python -m` from any cwd)
        alt_path = _PKG_DIR / args.config
        if alt_path.exists():
            config_path = alt_path
        else:
            print(
                f"Error: Config file not found: {config_path}\n"
                f"  Looked also at: {alt_path}\n"
                f"  Hint: run from inside the football_ai_pipeline/ directory,\n"
                f"        or pass an absolute path with --config.",
                file=sys.stderr,
            )
            sys.exit(1)

    try:
        with open(config_path, "r", encoding="utf-8-sig") as f:
            config = yaml.safe_load(f) or {}
    except UnicodeDecodeError:
        print(
            f"ERROR: Could not decode config file. Please save as UTF-8.\n"
            f"  Path: {config_path.resolve()}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Apply CLI overrides to config
    if args.max_frames is not None:
        config.setdefault("video", {})["max_frames"] = args.max_frames
    if args.stride is not None:
        config.setdefault("video", {})["stride"] = args.stride
    if args.save_video is not None:
        config.setdefault("visualization", {})["enabled"] = args.save_video
    if args.device is not None:
        config.setdefault("detection", {})["device"] = args.device

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input video not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Run pipeline
    try:
        runner = PipelineRunner(
            video_path=input_path,
            output_dir=args.output,
            config=config,
        )
        runner.run()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nPipeline failed: {e}", file=sys.stderr)
        print("Run with --max-frames 10 --stride 1 to debug on a small sample.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
