"""Tensor / numpy conversion helpers for CUDA-safe annotation."""

from __future__ import annotations

from typing import Any

import numpy as np

# Lazy-cached torch reference (None = not yet checked, False = not installed)
_torch_module: Any = None


def _get_torch() -> Any:
    """Return the ``torch`` module, or ``False`` if not installed."""
    global _torch_module
    if _torch_module is None:
        try:
            import torch
            _torch_module = torch
        except ImportError:
            _torch_module = False
    return _torch_module


def to_cpu_numpy(x: Any) -> Any:
    """Ensure *x* is a CPU numpy array (or plain Python scalar).

    Handles:
    - torch.Tensor  → .detach().cpu().numpy()
    - numpy ndarray → returned as-is
    - list / tuple  → returned unchanged
    - scalar        → returned unchanged
    """
    # Fast path: already numpy
    if isinstance(x, np.ndarray):
        return x

    # Check for torch tensor using the actual torch.Tensor class
    torch = _get_torch()
    if torch and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return x
