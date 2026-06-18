"""
yolo_tracker.py
===============
CPE Perception Stack — YOLO + ByteTrack wrapper.

Provides a thin, stateless class around the Ultralytics YOLO model configured
to run ByteTrack for persistent per-object track IDs across frames.
"""

from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

DEFAULT_MODEL      = "yolo26n.pt"   # YOLO26n: edge-optimized, auto-downloads on first run
DEFAULT_CONF       = 0.30
DEFAULT_TRACKER    = "bytetrack.yaml"   # built into ultralytics ≥ 8.1


class YoloTracker:
    """
    Wraps a YOLO model with ByteTrack for stateful multi-object tracking.

    Usage:
        tracker = YoloTracker()
        for frame in frames:
            detections = tracker.track(frame)
            # detections = list of dicts with keys:
            #   track_id, class_name, confidence, x1, y1, x2, y2, cx
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        conf: float = DEFAULT_CONF,
        tracker: str = DEFAULT_TRACKER,
        device: str | None = None,
    ):
        self.model = YOLO(model_path)
        self.model.overrides["conf"]    = conf
        self.model.overrides["tracker"] = tracker
        self.device = device or self._resolve_device()
        self.model.to(self.device)

    @staticmethod
    def _resolve_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"

    def track(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLO + ByteTrack on a single BGR frame.

        Returns:
            List of detection dicts. Empty list if no objects detected/tracked.
            Each dict:
                track_id   (int)   : persistent ByteTrack ID
                class_name (str)   : COCO class label
                confidence (float) : detection confidence [0, 1]
                x1, y1, x2, y2 (int) : bounding box corners
                cx (float)         : horizontal centre pixel
        """
        results = self.model.track(frame, persist=True, verbose=False)

        if results[0].boxes is None or results[0].boxes.id is None:
            return []

        boxes   = results[0].boxes.xyxy.cpu().numpy()
        ids     = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs   = results[0].boxes.conf.cpu().numpy()
        names   = results[0].names

        detections = []
        for box, tid, cls_idx, conf in zip(boxes, ids, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            detections.append({
                "track_id":   int(tid),
                "class_name": names[cls_idx],
                "confidence": round(float(conf), 3),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1 + x2) / 2,
            })
        return detections
