"""Pipeline Runner — orchestrates the full end-to-end pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..data_models import FrameFlag, FrameState
from ..video_io import VideoReader, VideoWriter
from ..segmentation import InPlayFilter
from ..detection import ObjectDetector
from ..tracking import ObjectTracker
from ..team_classifier import TeamClassifier
from ..keypoints import KeypointDetector
from ..homography import HomographyEstimator
from ..transforms import PitchTransformer
from ..stats import StatsAggregator
from ..visualization import FrameAnnotator
from ..exports import Exporter
from ..analytics.engine import AnalyticsEngine

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Runs the complete football analytics pipeline."""

    def __init__(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        config: dict[str, Any],
    ) -> None:
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        log_level = config.get("pipeline", {}).get("log_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        # Initialize all modules
        logger.info("Initializing pipeline modules...")
        self.reader = VideoReader(video_path, config)
        self.in_play_filter = InPlayFilter(config)
        self.detector = ObjectDetector(config)
        self.tracker = ObjectTracker(config)
        self.team_classifier = TeamClassifier(config)
        self.keypoint_detector = KeypointDetector(config)
        self.homography_estimator = HomographyEstimator(config)
        self.pitch_transformer = PitchTransformer(config)
        self.stats_aggregator = StatsAggregator(config)
        self.annotator = FrameAnnotator(config)
        self.exporter = Exporter(self.output_dir, config)
        self.analytics_engine = AnalyticsEngine(config)

        viz_cfg = config.get("visualization", {})
        self._write_video = viz_cfg.get("enabled", True)
        self._video_writer: VideoWriter | None = None

        # Coverage counters
        self._total_frames = 0
        self._in_play_frames = 0
        self._homography_frames = 0
        self._ball_frames = 0
        self._detection_frames = 0

        # Print capability summary
        self._print_capability_summary()

    def _print_capability_summary(self) -> None:
        """Log which modules are fully operational vs degraded."""
        lines = [
            "",
            "=" * 60,
            "  Module Capability Summary",
            "=" * 60,
        ]

        # Detection
        if self.detector.is_available:
            lines.append("  [OK]  Detection     : YOLO model loaded")
        else:
            lines.append("  [--]  Detection     : FALLBACK (no detections — install ultralytics)")

        # Tracking
        if self.tracker.backend == "bytetrack":
            lines.append("  [OK]  Tracking      : supervision ByteTrack")
        else:
            lines.append("  [--]  Tracking      : Simple IoU fallback (install supervision for better tracking)")

        # Keypoints
        if self.keypoint_detector.backend == "model":
            lines.append("  [OK]  Keypoints     : Trained model")
        else:
            lines.append("  [--]  Keypoints     : Heuristic line detector (provide keypoints.weights for better accuracy)")

        # Always available
        lines.append("  [OK]  In-play filter: Heuristic (scene change + green ratio)")
        lines.append("  [OK]  Team classify  : K-Means jersey colour clustering")
        lines.append("  [OK]  Homography    : DLT/RANSAC with temporal smoothing")
        lines.append("  [OK]  Stats (A-E)   : All analytics modules active")
        lines.append("  [OK]  Analytics     : Possession, events, xG, heatmaps")
        lines.append("  [OK]  Exports       : JSONL + CSV + JSON")

        if self._write_video:
            lines.append("  [OK]  Video output  : Annotated video enabled")
        else:
            lines.append("  [--]  Video output  : Disabled (--save-video false)")

        # Impact summary for degraded modules
        if not self.detector.is_available:
            lines.append("")
            lines.append("  NOTE: Without a detection model, the following outputs will be reduced:")
            lines.append("    - Player tracking:      No player/ball tracks")
            lines.append("    - Team classification:  No team assignments")
            lines.append("    - Physical stats (A):   Distance/speed/sprints = 0")
            lines.append("    - Spatial stats (B):    No team shape data")
            lines.append("    - Ball stats (C):       No possession/progression")
            lines.append("    - Pressure stats (D):   No pitch control data")
            lines.append("    - Threat stats (E):     No xT or zone data")
            lines.append("    - Ball coverage:        0%")
            lines.append("")
            lines.append("  The pipeline WILL still produce valid exports (metadata, run_report,")
            lines.append("  frames.jsonl, summaries) — they will just contain empty/zero values.")

        lines.append("=" * 60)
        logger.info("\n".join(lines))

    def run(self) -> None:
        """Execute the full pipeline."""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("Football AI Pipeline — Starting")
        logger.info("Input: %s", self.video_path)
        logger.info("Output: %s", self.output_dir)
        logger.info("=" * 60)

        # Save video metadata
        self.reader.save_metadata(self.output_dir)
        self.analytics_engine.set_fps(self.reader.fps / max(self.reader.stride, 1))

        # Initialize video writer
        if self._write_video:
            try:
                self._video_writer = VideoWriter(
                    self.output_dir / "annotated.mp4",
                    self.reader.width,
                    self.reader.height,
                    self.reader.fps / self.reader.stride,
                    self.config,
                )
            except Exception as e:
                logger.warning(
                    "Could not initialize video writer (%s). "
                    "Annotated video will not be produced. "
                    "Other exports will still be written.",
                    e,
                )
                self._write_video = False
                self._video_writer = None

        # Process frames
        estimated_total = self.reader.total_frames // max(self.reader.stride, 1)
        max_frames = self.config.get("video", {}).get("max_frames")
        if max_frames:
            estimated_total = min(estimated_total, max_frames)

        for frame_idx, timestamp, image in tqdm(
            self.reader.frames(),
            desc="Processing frames",
            total=estimated_total,
        ):
            frame_state = self._process_frame(frame_idx, timestamp, image)
            self._update_counters(frame_state)

            # Export frame data
            self.exporter.write_frame(frame_state)

            # Write annotated video
            if self._write_video and self._video_writer:
                annotated = self.annotator.annotate(frame_state)
                if annotated is not None:
                    self._video_writer.write(annotated)

        # Force-fit team classifier if not enough frames were seen
        self.team_classifier.force_fit()

        # Finalize
        self._finalize(start_time)

    def _process_frame(
        self, frame_idx: int, timestamp: float, image: Any,
    ) -> FrameState:
        """Run all pipeline stages on one frame."""
        fs = FrameState(
            frame_idx=frame_idx,
            timestamp_sec=timestamp,
            image=image,
        )

        # FR2 — In-play filter
        fs = self.in_play_filter.classify(fs)

        # FR3 — Detection (even for non-play frames, for tracking continuity)
        fs = self.detector.detect(fs)

        # FR4 — Tracking
        fs = self.tracker.track(fs)

        # FR5 — Team classification
        fs = self.team_classifier.update(fs)

        # FR6 — Keypoints & Homography
        fs = self.keypoint_detector.detect(fs)
        fs = self.homography_estimator.estimate(fs)

        # FR7 — Pitch transform
        fs = self.pitch_transformer.transform(fs)

        # FR8 — Stats
        fs = self.stats_aggregator.update(fs)

        # FR9 — Analytics (ball ownership per frame)
        fs = self.analytics_engine.update(fs)

        return fs

    def _update_counters(self, fs: FrameState) -> None:
        self._total_frames += 1
        if fs.flag == FrameFlag.IN_PLAY:
            self._in_play_frames += 1
        if fs.homography.available:
            self._homography_frames += 1
        if fs.ball and fs.ball.pitch_pos:
            self._ball_frames += 1
        if fs.detections:
            self._detection_frames += 1

    def _build_degradation_report(self) -> dict[str, Any]:
        """Build a report of which modules ran in degraded mode."""
        warnings: list[str] = []
        modules: dict[str, dict[str, Any]] = {}

        # Detection
        modules["detection"] = {
            "available": self.detector.is_available,
            "status": self.detector.status,
        }
        if not self.detector.is_available:
            warnings.append(
                "Detection model not loaded — all player/ball stats are empty. "
                "Install ultralytics and provide weights for full output."
            )

        # Tracking
        modules["tracking"] = {
            "backend": self.tracker.backend,
        }
        if self.tracker.backend != "bytetrack":
            warnings.append(
                "Using simple IoU tracker instead of ByteTrack — "
                "track stability may be reduced. Install supervision for better results."
            )

        # Keypoints
        modules["keypoints"] = {
            "backend": self.keypoint_detector.backend,
        }
        if self.keypoint_detector.backend != "model":
            warnings.append(
                "Using heuristic keypoint detector — homography quality may be low. "
                "Provide keypoints.weights for better pitch mapping."
            )

        # Homography quality assessment
        total = max(self._total_frames, 1)
        in_play = max(self._in_play_frames, 1)
        hom_pct = self._homography_frames / in_play * 100
        if hom_pct < 60:
            warnings.append(
                f"Homography available for only {hom_pct:.1f}% of in-play frames "
                f"(target: >=60%). Pitch-mapped stats have low confidence."
            )

        # Ball coverage
        ball_pct = self._ball_frames / in_play * 100
        if ball_pct < 40:
            warnings.append(
                f"Ball position available for only {ball_pct:.1f}% of in-play frames "
                f"(target: >=40%). Ball-related stats have low confidence."
            )

        return {
            "modules": modules,
            "warnings": warnings,
            "overall_confidence": self._compute_overall_confidence(hom_pct, ball_pct),
        }

    @staticmethod
    def _compute_overall_confidence(hom_pct: float, ball_pct: float) -> str:
        """Summarize overall output quality as a human-readable label."""
        if hom_pct >= 60 and ball_pct >= 40:
            return "good"
        if hom_pct >= 30 or ball_pct >= 20:
            return "low — some stats may be unreliable"
        return "very_low — most stats will be empty or unreliable"

    def _finalize(self, start_time: float) -> None:
        """Write summaries, close resources."""
        elapsed = time.time() - start_time
        logger.info("Processing complete in %.1f seconds", elapsed)

        # Get aggregated stats
        full_report = self.stats_aggregator.get_full_report()

        # Write player & team summaries
        self.exporter.write_players_summary(
            self.stats_aggregator.get_player_summary()
        )
        self.exporter.write_teams_summary(
            self.stats_aggregator.get_team_summary()
        )

        # Write detailed stats
        self.exporter.write_stats_folder(full_report)

        # Coverage metrics
        total = max(self._total_frames, 1)
        in_play = max(self._in_play_frames, 1)
        coverage = {
            "total_frames_processed": self._total_frames,
            "in_play_frames": self._in_play_frames,
            "in_play_pct": round(self._in_play_frames / total * 100, 1),
            "homography_available_pct": round(self._homography_frames / in_play * 100, 1),
            "ball_position_available_pct": round(self._ball_frames / in_play * 100, 1),
            "detection_frames_pct": round(self._detection_frames / total * 100, 1),
            "processing_time_sec": round(elapsed, 1),
            "fps_throughput": round(self._total_frames / elapsed, 1) if elapsed > 0 else 0,
        }

        # Run analytics finalization (events, metrics, heatmaps)
        analytics_summary = self.analytics_engine.finalize(self.output_dir)
        coverage["analytics"] = analytics_summary

        # Build degradation report
        degradation = self._build_degradation_report()

        self.exporter.write_run_report(
            metadata=self.reader.metadata,
            coverage=coverage,
            degradation=degradation,
        )

        # Close resources
        self.exporter.close()
        if self._video_writer:
            self._video_writer.release()
        self.reader.release()

        # Final summary
        logger.info("=" * 60)
        logger.info("Pipeline finished successfully!")
        logger.info("Frames: %d total, %d in-play", self._total_frames, self._in_play_frames)
        logger.info(
            "Coverage: homography %.1f%%, ball %.1f%%, detections %.1f%%",
            coverage["homography_available_pct"],
            coverage["ball_position_available_pct"],
            coverage["detection_frames_pct"],
        )
        if degradation["warnings"]:
            logger.warning("Quality warnings:")
            for w in degradation["warnings"]:
                logger.warning("  - %s", w)
        logger.info("Overall confidence: %s", degradation["overall_confidence"])
        logger.info("Output: %s", self.output_dir)
        logger.info("=" * 60)
