"""FR3 — Object Detection: detect player, goalkeeper, referee, ball.

Wraps ultralytics YOLO when available; falls back to a dummy detector
that produces no detections (pipeline still runs, just with empty data).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..data_models import BBox, Detection, ObjectClass, FrameState
from ..utils.tensors import to_cpu_numpy

logger = logging.getLogger(__name__)

# Class-ID mapping for football-specific YOLO models
_FOOTBALL_CLASS_MAP: dict[int, ObjectClass] = {
    0: ObjectClass.PLAYER,
    1: ObjectClass.GOALKEEPER,
    2: ObjectClass.REFEREE,
    3: ObjectClass.BALL,
}

# Class-ID mapping for generic COCO-trained YOLO models (yolov8n/s/m/l/x.pt)
_COCO_CLASS_MAP: dict[int, ObjectClass] = {
    0: ObjectClass.PLAYER,      # COCO "person"
    32: ObjectClass.BALL,        # COCO "sports ball"
}

# Package root: football_ai_pipeline/
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent

# Human-readable degradation explanation
_DEGRADATION_NOTICE = """\
========================================================================
  DETECTION MODEL NOT AVAILABLE — running in fallback mode
========================================================================
  Reason : {reason}

  What this means:
    - No player/ball detections will be produced
    - Tracking, team classification, and ball stats will be empty
    - Spatial/tactical stats require detections and will show zeros
    - In-play filtering + video annotation still work

  To enable full detection:

    Step 1: pip install ultralytics
    Step 2: Download YOLO weights manually (no auto-download):
              https://github.com/ultralytics/assets/releases
            Place the .pt file locally, e.g.:
              models/yolo/yolov8n.pt
    Step 3: Set the local path in configs/default.yaml:
              detection.weights: "models/yolo/yolov8n.pt"

  Diagnostics:
    detection.model    = {model}
    detection.weights  = {weights}
    resolved path      = {resolved}
    file exists        = {exists}
    detection.device   = {device}
    ultralytics        = {ultralytics_status}
========================================================================"""


def _resolve_weights_path(raw: str) -> Path:
    """Resolve a weights path: try as-is first, then relative to package root."""
    p = Path(raw)
    if p.exists():
        return p.resolve()
    # Try relative to package root (football_ai_pipeline/)
    pkg_relative = _PKG_ROOT / raw
    if pkg_relative.exists():
        return pkg_relative.resolve()
    # Return the pkg-relative path even if missing so the error message is useful
    return pkg_relative


def _ultralytics_version() -> str:
    try:
        import ultralytics
        return getattr(ultralytics, "__version__", "unknown")
    except ImportError:
        return "NOT INSTALLED"


def _try_load_yolo(config: dict[str, Any]) -> tuple[Any | None, str]:
    """Try to load YOLO model from a local weights file. Returns (model, status_message).

    IMPORTANT: This function never triggers network downloads.
    If detection.weights is not set, it skips loading entirely.
    """
    det_cfg = config.get("detection", {})
    model_name = det_cfg.get("model", "yolov8n")
    weights = det_cfg.get("weights")
    device = det_cfg.get("device", "auto")

    # Require an explicit local weights path — never auto-download
    if not weights:
        reason = (
            "detection.weights is not set. "
            "This pipeline does not download models at runtime. "
            "Download weights manually and set detection.weights to the local .pt file path."
        )
        _log_degradation(reason, model_name, weights, device)
        return None, reason

    # Check if ultralytics is installed
    try:
        from ultralytics import YOLO
    except ImportError:
        reason = "ultralytics package is not installed (pip install ultralytics)"
        _log_degradation(reason, model_name, weights, device)
        return None, reason

    # Resolve the weights path robustly
    weights_path = _resolve_weights_path(weights)

    if not weights_path.exists():
        reason = (
            f"Weights file not found at '{weights}'.\n"
            f"  Tried: {Path(weights).resolve()}\n"
            f"  Tried: {(_PKG_ROOT / weights).resolve()}\n"
            f"  Fix:   Place your .pt file at one of these locations,\n"
            f"         or set detection.weights to an absolute path."
        )
        fail_hard = config.get("pipeline", {}).get("fail_on_missing_weights", False)
        if fail_hard:
            raise FileNotFoundError(reason)
        _log_degradation(reason, model_name, weights, device)
        return None, reason

    # Load from the verified local path only
    try:
        load_device = None if device == "auto" else device
        model = YOLO(str(weights_path))
        status = f"YOLO loaded: {weights_path} (device={load_device or 'auto'})"
        logger.info(status)
        return model, status
    except Exception as e:
        reason = (
            f"YOLO model load FAILED — weights file exists but could not be loaded.\n"
            f"  Path:  {weights_path}\n"
            f"  Error: {e}\n"
            f"\n"
            f"  Probable causes:\n"
            f"    - Corrupted or incomplete .pt file (re-download it)\n"
            f"    - torch/CPU architecture mismatch (try: pip install torch --force-reinstall)\n"
            f"    - Incompatible ultralytics/torch versions (try: pip install -U ultralytics torch)\n"
            f"\n"
            f"  Quick check:\n"
            f"    python scripts/verify_yolo.py --config configs/default.yaml"
        )
        fail_hard = config.get("pipeline", {}).get("fail_on_missing_weights", False)
        if fail_hard:
            raise RuntimeError(reason) from e
        _log_degradation(reason, model_name, weights, device)
        return None, reason


def _log_degradation(
    reason: str, model: str, weights: Any, device: str,
) -> None:
    resolved = "N/A"
    exists = False
    if weights:
        rp = _resolve_weights_path(str(weights))
        resolved = str(rp)
        exists = rp.exists()

    msg = _DEGRADATION_NOTICE.format(
        reason=reason,
        model=model,
        weights=weights or "(not set — provide a local .pt path)",
        resolved=resolved,
        exists=exists,
        device=device,
        ultralytics_status=_ultralytics_version(),
    )
    logger.warning(msg)


def resolve_device(choice: str = "auto") -> str:
    """Resolve a device selection string to a torch device name.

    Returns ``"cuda:0"`` or ``"cpu"``.  Never imports torch if *choice*
    is ``"cpu"``; on failure silently falls back to CPU.
    """
    if choice in ("cpu",):
        return "cpu"
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if choice == "auto":
        return "cuda:0" if has_cuda else "cpu"
    # Explicit cuda request
    if has_cuda:
        return "cuda:0"
    logger.warning("CUDA requested but not available — falling back to CPU")
    return "cpu"


def get_device_label() -> str:
    """Return a human-readable compute badge, e.g. ``'RTX 3070 (CUDA)'``."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"{name} (CUDA)"
    except Exception:
        pass
    return "CPU"


class ObjectDetector:
    """Detect objects in frames using YOLO or fallback."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        det_cfg = config.get("detection", {})
        self.confidence_threshold: float = det_cfg.get("confidence", 0.25)
        self.iou_threshold: float = det_cfg.get("iou_threshold", 0.45)
        self._device: str = resolve_device(det_cfg.get("device", "auto"))
        self._model, self.status = _try_load_yolo(config)
        self._class_map = self._pick_class_map()
        if self._model is not None:
            self._move_model_to_device()

    def _pick_class_map(self) -> dict[int, ObjectClass]:
        """Select class map: football-specific if model has <=10 classes, COCO otherwise."""
        if self._model is None:
            return _FOOTBALL_CLASS_MAP
        names = getattr(self._model, "names", {})
        if len(names) <= 10:
            # Likely a football-specific fine-tuned model
            logger.info("Using football-specific class map (%d classes)", len(names))
            return _FOOTBALL_CLASS_MAP
        # Generic COCO model — map person->PLAYER, sports ball->BALL
        logger.info("Using COCO class map (model has %d classes: person->PLAYER, sports ball->BALL)", len(names))
        return _COCO_CLASS_MAP

    def _move_model_to_device(self) -> None:
        """Move YOLO model to the resolved device (GPU or CPU)."""
        try:
            self._model.to(self._device)
            logger.info("Detection model moved to %s", self._device)
        except Exception as e:
            logger.warning("Could not move model to %s: %s — using default", self._device, e)

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def detect(self, frame_state: FrameState) -> FrameState:
        """Run detection on frame_state.image and populate frame_state.detections."""
        if frame_state.image is None:
            return frame_state

        if self._model is not None:
            return self._detect_yolo(frame_state)
        return self._detect_fallback(frame_state)

    def _detect_yolo(self, frame_state: FrameState) -> FrameState:
        results = self._model(
            frame_state.image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self._device,
            verbose=False,
        )
        detections: list[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            # Bulk-convert entire tensors to CPU numpy once (not per-row)
            all_xyxy = to_cpu_numpy(boxes.xyxy)   # (N, 4)
            all_conf = to_cpu_numpy(boxes.conf)    # (N,)
            all_cls = to_cpu_numpy(boxes.cls)      # (N,)
            for i in range(len(all_xyxy)):
                cls_id = int(all_cls[i])
                obj_class = self._class_map.get(cls_id)
                if obj_class is None:
                    continue  # skip classes not relevant to football
                det = Detection(
                    bbox=BBox(
                        x1=float(all_xyxy[i][0]),
                        y1=float(all_xyxy[i][1]),
                        x2=float(all_xyxy[i][2]),
                        y2=float(all_xyxy[i][3]),
                    ),
                    class_id=obj_class,
                    confidence=float(all_conf[i]),
                )
                detections.append(det)
        frame_state.detections = detections
        return frame_state

    def _detect_fallback(self, frame_state: FrameState) -> FrameState:
        """Fallback: no detections produced."""
        frame_state.detections = []
        return frame_state
