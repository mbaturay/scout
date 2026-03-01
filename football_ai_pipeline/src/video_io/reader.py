"""FR1 — Video Ingestion: load video, extract metadata, yield frames."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Generator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """Reads a video file and yields (frame_idx, timestamp_sec, image) tuples."""

    def __init__(self, video_path: str | Path, config: dict[str, Any]) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        self.fps: float = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        vcfg = config.get("video", {})
        self.stride: int = vcfg.get("stride", 1)
        self.max_frames: Optional[int] = vcfg.get("max_frames")

        logger.info(
            "Video: %s | %dx%d | %.1f fps | %d frames | stride=%d",
            self.video_path.name, self.width, self.height,
            self.fps, self.total_frames, self.stride,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "source": str(self.video_path),
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "stride": self.stride,
        }

    def save_metadata(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        logger.info("Saved metadata to %s", meta_path)

    def frames(self) -> Generator[tuple[int, float, np.ndarray], None, None]:
        """Yield (frame_idx, timestamp_sec, bgr_image) for selected frames."""
        idx = 0
        yielded = 0
        while True:
            if self.max_frames and yielded >= self.max_frames:
                break
            ret, frame = self.cap.read()
            if not ret:
                break
            if idx % self.stride == 0:
                ts = idx / self.fps
                yield idx, ts, frame
                yielded += 1
            idx += 1
        logger.info("Yielded %d frames out of %d total", yielded, idx)

    def release(self) -> None:
        if self.cap:
            self.cap.release()

    def __del__(self) -> None:
        self.release()
