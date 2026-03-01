#!/usr/bin/env python3
"""Verify CUDA / GPU availability for the football AI pipeline."""

from __future__ import annotations

import sys


def main() -> None:
    try:
        import torch
    except ImportError:
        print("ERROR: torch is not installed.", file=sys.stderr)
        sys.exit(1)

    print(f"torch.__version__  : {torch.__version__}")
    print(f"cuda.is_available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        dev_id = torch.cuda.current_device()
        dev_name = torch.cuda.get_device_name(dev_id)
        print(f"cuda.device_count  : {torch.cuda.device_count()}")
        print(f"cuda.current_device: {dev_id}")
        print(f"cuda.device_name   : {dev_name}")
        print(f"cuda.capability    : {torch.cuda.get_device_capability(dev_id)}")

        mem = torch.cuda.get_device_properties(dev_id).total_memory
        print(f"cuda.total_memory  : {mem / 1024**3:.1f} GB")
    else:
        print()
        print("CUDA is NOT available. Possible causes:")
        print("  - torch was installed without CUDA support (CPU-only wheel)")
        print("  - NVIDIA drivers are not installed or outdated")
        print("  - No compatible NVIDIA GPU detected")
        print()
        print("Fix: reinstall torch with CUDA index:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)


if __name__ == "__main__":
    main()
