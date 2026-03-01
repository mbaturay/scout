#!/usr/bin/env python
"""Decode-only video reader diagnostic.

Tests whether OpenCV can read every frame of a video without hanging.
Uses a watchdog timer to detect stalls.

Usage:
    cd football_ai_pipeline
    python scripts/debug_video_read.py test-video-1.mp4
    python scripts/debug_video_read.py test-video-1.mp4 --report-every 20 --timeout 30
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode-only video read diagnostic")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument(
        "--report-every", type=int, default=10,
        help="Print progress every N frames (default: 10)",
    )
    parser.add_argument(
        "--timeout", type=float, default=15.0,
        help="Seconds to wait before declaring a frame read as hung (default: 15)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: cv2.VideoCapture could not open: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 60)
    print("  Video Decode Diagnostic")
    print("=" * 60)
    print(f"  File:       {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS:        {fps}")
    print(f"  Frames:     {total_frames}")
    print(f"  Timeout:    {args.timeout}s per frame")
    print("=" * 60)

    frame_idx: int = 0
    read_ok: bool = False
    hung: bool = False
    t0 = time.perf_counter()

    def _watchdog() -> None:
        nonlocal hung
        time.sleep(args.timeout)
        if not read_ok:
            hung = True
            print(f"\n  WATCHDOG: Frame read HUNG at frame {frame_idx} "
                  f"(no response for {args.timeout}s)")
            print(f"  Probable cause: video codec issue or corrupted frame.")
            print(f"  Fix: re-encode the video:")
            print(f"    ffmpeg -i \"{video_path}\" -c:v libx264 -preset fast -crf 23 output.mp4")
            # Force exit — cap.read() may be stuck in a C extension
            import os
            os._exit(2)

    while True:
        read_ok = False
        watchdog = threading.Thread(target=_watchdog, daemon=True)
        watchdog.start()

        ret, frame = cap.read()
        read_ok = True

        if not ret:
            break

        frame_idx += 1
        if frame_idx % args.report_every == 0:
            elapsed = time.perf_counter() - t0
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            print(f"  Frame {frame_idx:>6d} / {total_frames}  "
                  f"({frame_idx * 100 // max(total_frames, 1):>3d}%)  "
                  f"[{fps_actual:.1f} fps]")

    cap.release()
    elapsed = time.perf_counter() - t0

    print("=" * 60)
    print(f"  DONE: Decoded {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / elapsed:.1f} fps)")
    if frame_idx < total_frames:
        print(f"  WARNING: Expected {total_frames} frames but only read {frame_idx}.")
        print(f"  The video may be truncated or have metadata issues.")
    else:
        print(f"  All {frame_idx} frames decoded successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
