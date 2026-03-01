"""FR9 support — Write annotated frames to output video."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoWriter:
    """Writes annotated frames to an output video file."""

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
        config: dict[str, Any],
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        viz_cfg = config.get("visualization", {})
        codec = viz_cfg.get("output_codec", "mp4v")
        out_fps = viz_cfg.get("output_fps") or fps

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path), fourcc, out_fps, (width, height)
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {self.output_path}")
        logger.info("VideoWriter: %s (%dx%d @ %.1f fps)", self.output_path, width, height, out_fps)

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        if self.writer:
            self.writer.release()
            logger.info("Video written to %s", self.output_path)

    def __del__(self) -> None:
        self.release()
