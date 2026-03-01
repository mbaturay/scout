"""Camera motion estimation for broadcast / tactical football video.

Uses ORB feature matching + RANSAC affine estimation between consecutive
grayscale frames to recover per-frame (dx, dy, rotation, scale).

Results are EMA-smoothed over time and written to camera_motion.jsonl.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_ORB_FEATURES = 500
_DEFAULT_MATCH_RATIO = 0.75        # Lowe ratio test threshold
_DEFAULT_MIN_MATCHES = 8           # below this → low-confidence
_DEFAULT_RANSAC_REPROJ = 3.0       # reprojection threshold (px)
_DEFAULT_EMA_ALPHA = 0.4


class CameraMotionEstimator:
    """Estimate inter-frame camera motion from ORB features + affine RANSAC.

    Parameters (from ``config["vision"]``):
        orb_features:      max ORB keypoints per frame (default 500)
        match_ratio:       Lowe ratio-test threshold   (default 0.75)
        min_matches:       minimum good matches needed  (default 8)
        ransac_reproj:     RANSAC reprojection threshold (default 3.0 px)
        ema_alpha:         EMA smoothing factor          (default 0.4)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        vis_cfg = config.get("vision", {})
        self._orb = cv2.ORB_create(nfeatures=vis_cfg.get("orb_features", _DEFAULT_ORB_FEATURES))
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._match_ratio: float = vis_cfg.get("match_ratio", _DEFAULT_MATCH_RATIO)
        self._min_matches: int = vis_cfg.get("min_matches", _DEFAULT_MIN_MATCHES)
        self._ransac_reproj: float = vis_cfg.get("ransac_reproj", _DEFAULT_RANSAC_REPROJ)
        self._ema_alpha: float = vis_cfg.get("ema_alpha", _DEFAULT_EMA_ALPHA)

        # State
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_kp: Optional[list] = None
        self._prev_desc: Optional[np.ndarray] = None

        # EMA state (smoothed transform)
        self._sm_dx: float = 0.0
        self._sm_dy: float = 0.0
        self._sm_rot: float = 0.0
        self._sm_scale: float = 1.0

        # Per-frame results (kept in memory for summary)
        self._records: list[dict[str, Any]] = []

        # Diagnostics
        self._good_frames: int = 0
        self._low_conf_frames: int = 0
        self._total_frames: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame_idx: int, timestamp_sec: float, image: np.ndarray) -> dict[str, Any]:
        """Process one BGR frame and return the camera-motion record.

        Returns a dict with keys:
            frame_idx, timestamp_sec, dx_px, dy_px, rot_deg, scale,
            smoothed_dx_px, smoothed_dy_px, smoothed_rot_deg, smoothed_scale,
            good_matches, confidence
        """
        self._total_frames += 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kp, desc = self._orb.detectAndCompute(gray, None)

        record = self._estimate(frame_idx, timestamp_sec, gray, kp, desc)

        # Advance state
        self._prev_gray = gray
        self._prev_kp = kp
        self._prev_desc = desc

        self._records.append(record)
        return record

    def write_jsonl(self, output_dir: Path) -> Path:
        """Flush all records to ``camera_motion.jsonl`` and return the path."""
        out_path = output_dir / "camera_motion.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in self._records:
                f.write(json.dumps(rec) + "\n")
        logger.info("Wrote %d camera-motion records to %s", len(self._records), out_path)
        return out_path

    def get_summary(self) -> dict[str, Any]:
        """Return aggregate stats for inclusion in run_report.json."""
        total = max(self._total_frames, 1)
        good_pct = round(self._good_frames / total * 100, 1)

        dx_vals = [r["dx_px"] for r in self._records if r["confidence"] == "good"]
        dy_vals = [r["dy_px"] for r in self._records if r["confidence"] == "good"]

        return {
            "total_frames": self._total_frames,
            "good_estimate_pct": good_pct,
            "low_confidence_frames": self._low_conf_frames,
            "mean_dx_px": round(sum(dx_vals) / len(dx_vals), 2) if dx_vals else 0.0,
            "mean_dy_px": round(sum(dy_vals) / len(dy_vals), 2) if dy_vals else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _estimate(
        self,
        frame_idx: int,
        timestamp_sec: float,
        gray: np.ndarray,
        kp: list,
        desc: Optional[np.ndarray],
    ) -> dict[str, Any]:
        """Match against previous frame and compute affine transform."""

        # First frame or missing descriptors → identity
        if self._prev_desc is None or desc is None or len(kp) < 2 or len(self._prev_kp) < 2:
            self._good_frames += 1  # trivially "good" for the first frame
            return self._make_record(
                frame_idx, timestamp_sec,
                dx=0.0, dy=0.0, rot=0.0, scale=1.0,
                n_matches=0, confidence="good",
            )

        # KNN match (k=2 for ratio test)
        try:
            raw_matches = self._matcher.knnMatch(self._prev_desc, desc, k=2)
        except cv2.error:
            return self._carry_forward(frame_idx, timestamp_sec)

        # Lowe ratio test
        good: list[cv2.DMatch] = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self._match_ratio * n.distance:
                    good.append(m)

        if len(good) < self._min_matches:
            self._low_conf_frames += 1
            return self._carry_forward(frame_idx, timestamp_sec, n_matches=len(good))

        # Build point arrays
        src_pts = np.float32([self._prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Estimate partial affine (rotation + uniform scale + translation)
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self._ransac_reproj,
        )

        if M is None:
            self._low_conf_frames += 1
            return self._carry_forward(frame_idx, timestamp_sec, n_matches=len(good))

        # Decompose: M = [[s*cos(θ), -s*sin(θ), tx],
        #                  [s*sin(θ),  s*cos(θ), ty]]
        dx = float(M[0, 2])
        dy = float(M[1, 2])
        scale = math.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
        rot_rad = math.atan2(M[1, 0], M[0, 0])
        rot_deg = math.degrees(rot_rad)

        self._good_frames += 1
        return self._make_record(
            frame_idx, timestamp_sec,
            dx=dx, dy=dy, rot=rot_deg, scale=scale,
            n_matches=len(good), confidence="good",
        )

    def _carry_forward(
        self, frame_idx: int, timestamp_sec: float, n_matches: int = 0,
    ) -> dict[str, Any]:
        """Use the last smoothed transform when estimation fails."""
        return self._make_record(
            frame_idx, timestamp_sec,
            dx=self._sm_dx, dy=self._sm_dy,
            rot=self._sm_rot, scale=self._sm_scale,
            n_matches=n_matches,
            confidence="camera_motion_low_conf",
        )

    def _make_record(
        self,
        frame_idx: int,
        timestamp_sec: float,
        *,
        dx: float,
        dy: float,
        rot: float,
        scale: float,
        n_matches: int,
        confidence: str,
    ) -> dict[str, Any]:
        """Build the output dict and update EMA state."""
        a = self._ema_alpha
        self._sm_dx = a * dx + (1.0 - a) * self._sm_dx
        self._sm_dy = a * dy + (1.0 - a) * self._sm_dy
        self._sm_rot = a * rot + (1.0 - a) * self._sm_rot
        self._sm_scale = a * scale + (1.0 - a) * self._sm_scale

        return {
            "frame_idx": frame_idx,
            "timestamp_sec": round(timestamp_sec, 4),
            "dx_px": round(dx, 2),
            "dy_px": round(dy, 2),
            "rot_deg": round(rot, 4),
            "scale": round(scale, 4),
            "smoothed_dx_px": round(self._sm_dx, 2),
            "smoothed_dy_px": round(self._sm_dy, 2),
            "smoothed_rot_deg": round(self._sm_rot, 4),
            "smoothed_scale": round(self._sm_scale, 4),
            "good_matches": n_matches,
            "confidence": confidence,
        }
