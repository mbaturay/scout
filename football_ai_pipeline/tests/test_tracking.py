"""Tests for the IoU tracker fallback."""

import pytest

from src.data_models import BBox, Detection, ObjectClass
from src.tracking.tracker import _SimpleIoUTracker


class TestSimpleIoUTracker:
    def test_first_frame_assigns_ids(self):
        tracker = _SimpleIoUTracker(iou_thresh=0.3)
        dets = [
            Detection(bbox=BBox(10, 10, 50, 50), class_id=ObjectClass.PLAYER, confidence=0.9),
            Detection(bbox=BBox(100, 100, 150, 150), class_id=ObjectClass.PLAYER, confidence=0.9),
        ]
        result = tracker.update(dets)
        assert result[0].track_id is not None
        assert result[1].track_id is not None
        assert result[0].track_id != result[1].track_id

    def test_consistent_ids_across_frames(self):
        tracker = _SimpleIoUTracker(iou_thresh=0.3)
        dets1 = [
            Detection(bbox=BBox(10, 10, 50, 50), class_id=ObjectClass.PLAYER, confidence=0.9),
        ]
        tracker.update(dets1)
        id1 = dets1[0].track_id

        # Slightly moved detection
        dets2 = [
            Detection(bbox=BBox(12, 12, 52, 52), class_id=ObjectClass.PLAYER, confidence=0.9),
        ]
        tracker.update(dets2)
        assert dets2[0].track_id == id1

    def test_new_id_for_far_detection(self):
        tracker = _SimpleIoUTracker(iou_thresh=0.3)
        dets1 = [
            Detection(bbox=BBox(10, 10, 50, 50), class_id=ObjectClass.PLAYER, confidence=0.9),
        ]
        tracker.update(dets1)
        id1 = dets1[0].track_id

        # Far away detection = new track
        dets2 = [
            Detection(bbox=BBox(500, 500, 550, 550), class_id=ObjectClass.PLAYER, confidence=0.9),
        ]
        tracker.update(dets2)
        assert dets2[0].track_id != id1

    def test_empty_detections(self):
        tracker = _SimpleIoUTracker()
        result = tracker.update([])
        assert result == []
