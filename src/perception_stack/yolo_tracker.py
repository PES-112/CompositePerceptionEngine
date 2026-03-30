"""
yolo_tracker.py
===============
CPE Perception Stack — YOLO + ByteTrack wrapper with depth-guided post-processing.

Provides a thin class around the Ultralytics YOLO model configured
to run ByteTrack for persistent per-object track IDs across frames.

Post-processing pipeline (ported from notebook):
  1. Class whitelist filter — only keep navigation-relevant classes
  2. Geometric validation — reject bad aspect ratios, tiny boxes, edge FPs
  3. depth_rescore() — 3-pass depth-guided filtering:
     Pass 1: Size sanity check (pixel height vs expected height at depth)
     Pass 2: Confidence rescoring (penalise implausible depth/class combos)
     Pass 3: Depth-guided NMS (overlapping boxes → keep more depth-consistent one)
"""

from pathlib import Path

import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL      = "yolo26n.pt"
DEFAULT_CONF       = 0.30
DEFAULT_TRACKER    = "bytetrack.yaml"

# ── Navigation-relevant classes (COCO subset) ─────────────────────────────────
ALLOWED_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "dog", "cat", "traffic light", "stop sign", "umbrella",
    "backpack", "suitcase", "unlabeled_obstacle",
}

# ── Expected pixel height at 1m for size sanity check ─────────────────────────
# pixel_height ≈ (real_height_m / distance_m) * focal_length_px
# For SANPO chest cam at 1280px wide, ~70° HFOV → focal ≈ 960px
FOCAL_PX = 960.0
CLASS_REAL_HEIGHT_M = {
    "person":        1.7,
    "bicycle":       1.1,
    "car":           1.5,
    "motorcycle":    1.1,
    "bus":           3.0,
    "truck":         2.5,
    "dog":           0.5,
    "cat":           0.3,
    "traffic light": 0.9,
    "stop sign":     0.75,
    "umbrella":      1.0,
    "backpack":      0.5,
    "suitcase":      0.6,
}
SIZE_TOLERANCE = 4.0   # allow 4× slack — depth is noisy


# ── Geometric validation ──────────────────────────────────────────────────────

def is_valid_detection(det: dict, frame_w: int, frame_h: int) -> bool:
    """Reject detections with bad aspect ratios, tiny area, or edge FPs."""
    bh = det["y2"] - det["y1"]
    bw = det["x2"] - det["x1"]
    # Reject ultra-wide boxes (aspect ratio > 4:1)
    if bw / max(bh, 1) > 4.0:
        return False
    # Reject tiny boxes (area < 500 px²)
    if bw * bh < 500:
        return False
    # Reject full-width boxes touching bottom edge (ground plane FPs)
    if det["y2"] >= frame_h - 5 and bw > frame_w * 0.5:
        return False
    return True


# ── Depth-guided 3-pass post-processing ───────────────────────────────────────

def depth_rescore(detections: list[dict], depth_map: np.ndarray | None) -> list[dict]:
    """
    Post-process YOLO detections using the depth map.  Three passes:

    Pass 1 — Size sanity check:
        At depth d, an object of real height H should have pixel height
        ≈ H * FOCAL_PX / d.  If actual is wildly different → drop.

    Pass 2 — Confidence rescoring:
        Penalise detections whose depth is implausible for their class
        (e.g. "car" at 0.4m, "person" at 28m).

    Pass 3 — Depth-guided NMS:
        For heavily overlapping boxes, keep the one whose depth is more
        internally consistent instead of just the highest confidence.
    """
    if depth_map is None:
        return detections

    h_dm, w_dm = depth_map.shape[:2]

    # ── Attach raw center depth to every detection ────────────────
    for det in detections:
        cx = min(int(det["cx"]), w_dm - 1)
        cy = min(int((det["y1"] + det["y2"]) / 2), h_dm - 1)
        v  = float(depth_map[cy, cx])
        det["_depth"] = v if (0.3 < v < 30.0) else None

    # ── Pass 1: Size sanity check ─────────────────────────────────
    passed = []
    for det in detections:
        d   = det["_depth"]
        cls = det["class_name"]
        if d is None or cls not in CLASS_REAL_HEIGHT_M:
            passed.append(det)
            continue
        expected_px = CLASS_REAL_HEIGHT_M[cls] * FOCAL_PX / d
        actual_px   = det["y2"] - det["y1"]
        ratio = actual_px / max(expected_px, 1.0)
        if (1.0 / SIZE_TOLERANCE) < ratio < SIZE_TOLERANCE:
            passed.append(det)
    detections = passed

    # ── Pass 2: Confidence rescoring ──────────────────────────────
    for det in detections:
        d   = det["_depth"]
        cls = det["class_name"]
        if d is None:
            continue
        penalty = 1.0
        # Cars/trucks/buses physically can't be closer than ~1.5m to chest cam
        if cls in ("car", "truck", "bus") and d < 1.5:
            penalty *= 0.3
        # People shouldn't register past 15m reliably
        if cls == "person" and d > 15.0:
            penalty *= 0.5
        # Anything < 0.5m is almost certainly noise
        if d < 0.5:
            penalty *= 0.2
        det["confidence"] = round(det["confidence"] * penalty, 3)

    # Drop detections whose rescored confidence fell below threshold
    detections = [d for d in detections if d["confidence"] >= 0.25]

    # ── Pass 3: Depth-guided NMS ──────────────────────────────────
    IOU_THRESH = 0.5
    keep = [True] * len(detections)

    for i in range(len(detections)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(detections)):
            if not keep[j]:
                continue
            a, b = detections[i], detections[j]
            # Compute IoU
            ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
            ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
            area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
            iou    = inter / (area_a + area_b - inter)
            if iou < IOU_THRESH:
                continue
            # Heavy overlap — keep more depth-consistent + closer one
            da, db = a["_depth"], b["_depth"]
            if da is None and db is None:
                if a["confidence"] >= b["confidence"]:
                    keep[j] = False
                else:
                    keep[i] = False
            elif da is None:
                keep[i] = False
            elif db is None:
                keep[j] = False
            else:
                score_a = a["confidence"] / max(da, 0.1)
                score_b = b["confidence"] / max(db, 0.1)
                if score_a >= score_b:
                    keep[j] = False
                else:
                    keep[i] = False

    detections = [d for i, d in enumerate(detections) if keep[i]]

    # Clean up internal key
    for det in detections:
        det.pop("_depth", None)

    return detections


class YoloTracker:
    """
    Wraps a YOLO model with ByteTrack for stateful multi-object tracking.

    Includes depth-guided post-processing:
      - Class whitelist filtering
      - Geometric validation (aspect ratio, area, edge rejection)
      - 3-pass depth rescoring (size sanity, confidence, depth-guided NMS)

    Usage:
        tracker = YoloTracker()
        for frame in frames:
            detections = tracker.track(frame, depth_map=depth_map)
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        conf: float = DEFAULT_CONF,
        tracker: str = DEFAULT_TRACKER,
    ):
        self.model = YOLO(model_path)
        self.model.overrides["conf"]    = conf
        self.model.overrides["tracker"] = tracker

    def track(self, frame: np.ndarray, depth_map: np.ndarray | None = None) -> list[dict]:
        """
        Run YOLO + ByteTrack on a single BGR frame with optional depth post-processing.

        Args:
            frame:     BGR image (H, W, 3).
            depth_map: Optional float32 depth array (metres) for depth rescoring.

        Returns:
            List of detection dicts. Each dict:
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
        h, w    = frame.shape[:2]

        detections = []
        for box, tid, cls_idx, conf in zip(boxes, ids, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            class_name = names[cls_idx]
            det = {
                "track_id":   int(tid),
                "class_name": class_name,
                "confidence": round(float(conf), 3),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1 + x2) / 2,
            }
            # Class whitelist filter
            if class_name not in ALLOWED_CLASSES:
                continue
            # Geometric validation
            if not is_valid_detection(det, w, h):
                continue
            detections.append(det)

        # ── Depth-guided post-processing ──────────────────────────
        detections = depth_rescore(detections, depth_map)

        return detections
