"""Pytest configuration — ensure src is importable."""

import sys
from pathlib import Path

# Add the football_ai_pipeline root to sys.path so `from src.xxx` works
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
