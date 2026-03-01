#!/usr/bin/env python
"""Verify YOLO weights loading without running the full pipeline.

Usage:
    cd football_ai_pipeline
    python scripts/verify_yolo.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure package imports work
_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_DIR = _SCRIPT_DIR.parent  # football_ai_pipeline/
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify YOLO weights loading")
    parser.add_argument(
        "--config", "-c",
        default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.exists():
        alt = _PKG_DIR / args.config
        if alt.exists():
            config_path = alt
        else:
            print(f"ERROR: Config not found: {config_path}")
            sys.exit(1)

    import yaml
    try:
        with open(config_path, "r", encoding="utf-8-sig") as f:
            config = yaml.safe_load(f) or {}
    except UnicodeDecodeError:
        print(
            f"ERROR: Could not decode config file. Please save as UTF-8.\n"
            f"  Path: {config_path.resolve()}")
        sys.exit(1)

    print("=" * 60)
    print("  YOLO Weights Verification")
    print("=" * 60)
    print(f"  Python:          {sys.executable}")
    print(f"  Config:          {config_path.resolve()}")

    det_cfg = config.get("detection", {})
    raw_weights = det_cfg.get("weights")
    print(f"  detection.weights: {raw_weights!r}")

    if not raw_weights:
        print("\n  PROBLEM: detection.weights is not set (null).")
        print("  Fix:    Set detection.weights in your config to a local .pt path.")
        sys.exit(1)

    # Resolve the same way detector.py does
    from src.detection.detector import _resolve_weights_path
    resolved = _resolve_weights_path(raw_weights)
    exists = resolved.exists()

    print(f"  Resolved path:   {resolved}")
    print(f"  File exists:     {exists}")
    if exists:
        size_mb = resolved.stat().st_size / (1024 * 1024)
        print(f"  File size:       {size_mb:.1f} MB")

    if not exists:
        print(f"\n  PROBLEM: Weights file not found at resolved path.")
        print(f"  Fix:    Download .pt and place at: {resolved}")
        sys.exit(1)

    # Check ultralytics
    try:
        import ultralytics
        uv = ultralytics.__version__
        print(f"  ultralytics:     {uv}")
    except ImportError:
        print(f"  ultralytics:     NOT INSTALLED")
        print(f"\n  PROBLEM: ultralytics package is missing.")
        print(f"  Fix:    pip install ultralytics")
        sys.exit(1)

    # Check torch
    try:
        import torch
        print(f"  torch:           {torch.__version__}")
        print(f"  CUDA available:  {torch.cuda.is_available()}")
    except ImportError:
        print(f"  torch:           NOT INSTALLED")
        print(f"\n  WARNING: torch not found. ultralytics may still work with CPU.")

    # Try loading
    print()
    print("  Loading YOLO model...")
    try:
        from ultralytics import YOLO
        model = YOLO(str(resolved))
        print(f"  SUCCESS: Model loaded from {resolved}")
        print(f"  Model type:      {type(model).__name__}")
        if hasattr(model, "names"):
            print(f"  Classes:         {model.names}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print()
        print("  Probable causes:")
        print("    - Corrupted or incomplete .pt file (re-download)")
        print("    - torch/CPU architecture mismatch (pip install torch --force-reinstall)")
        print("    - Incompatible ultralytics/torch versions (pip install -U ultralytics torch)")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
