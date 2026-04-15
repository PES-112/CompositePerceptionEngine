# CPE Perception Pipeline — Performance Overhaul Walkthrough

## Summary

Ported 6 major features from the Colab notebook into the perception layer's modular codebase, added streaming/nth-frame strategy, and tensor-based batch physics. **8 files modified**, all backward-compatible.

---

## Changes Made

### 1. Center-Patch Depth Extraction — [depth_loader.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/depth_loader.py)

**Before:** [median_depth_in_box()](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/depth_loader.py#78-152) sampled the *entire* bounding box — averaging noisy edge/background pixels.

**After:** Samples only the central 20% patch of the bbox. Supports `exclude_boxes` for occlusion masking (closer objects mask behind ones). Auto-expands 3× for sparse ZED depth fallback.

```diff:depth_loader.py
"""
depth_loader.py
===============
CPE Perception Stack — Dataset-aware depth map loader.

Handles both SANPO and UASOL depth formats:

  SANPO-Synthetic / SANPO-Real:
      File   : <frame>.float16.gz
      Format : gzip-compressed float16 numpy array, shape (H, W)
      Values : metric depth in metres (no scale needed)
      Res    : 2208×1242 (downsized to match RGB frame by load_depth_map)

  UASOL:
      File   : img_depth.png (16-bit PNG)
      Format : uint16, scale × 0.256 = metres
"""

from __future__ import annotations
import gzip
from pathlib import Path

import cv2
import numpy as np

# Physical depth range filter — anything outside this is sensor noise or irrelevant
MIN_DEPTH_M = 0.3    # below this = too close / noise
MAX_DEPTH_M = 30.0   # beyond this = not actionable

# UASOL-only scale factor (SANPO values are already in metres as float16)
DEPTH_SCALES = {
    "sanpo": 1.0,     # float16 gz — already metres
    "uasol": 0.256,   # uint16 PNG — raw × 0.256 = metres
}

# Known SANPO depth resolution (synthetic + real)
SANPO_DEPTH_H = 1242
SANPO_DEPTH_W = 2208


def load_depth_map(depth_path: Path, source: str = "sanpo") -> np.ndarray | None:
    """
    Load a depth map and return a float32 array in metres, sized to match
    the RGB frame (resized if needed).

    Auto-detects format by file extension:
      *.float16.gz  →  SANPO (decompress → reshape float16 → float32)
      *.png         →  UASOL (cv2 ANYDEPTH → scale)

    Returns None if the file does not exist.
    """
    if not depth_path.exists():
        return None

    suffix = "".join(depth_path.suffixes).lower()   # e.g. ".float16.gz" or ".png"

    if suffix == ".float16.gz":
        # SANPO: gzip-compressed raw float16 binary, row-major (H, W).
        # Files contain 2 extra padding float16 values at the end — truncate to exact size.
        with gzip.open(depth_path, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.float16)
        n = SANPO_DEPTH_H * SANPO_DEPTH_W
        depth = raw[:n].reshape(SANPO_DEPTH_H, SANPO_DEPTH_W).astype(np.float32)
        return depth   # already in metres

    elif suffix == ".png":
        # UASOL: 16-bit PNG, scale to metres
        raw = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if raw is None:
            return None
        scale = DEPTH_SCALES.get(source, 1.0)
        return raw.astype(np.float32) * scale

    else:
        raise ValueError(f"Unknown depth format: {depth_path.name!r}")


def median_depth_in_box(
    depth_map: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> float | None:
    """
    Return the median metric depth (metres) of valid pixels inside a bounding box.
    Crops ROI, filters noise, returns None if no valid pixels found.
    """
    # Clamp bbox to depth map bounds (SANPO frames may be downscaled by YOLO)
    h, w = depth_map.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x1 >= x2 or y1 >= y2:
        return None

    roi   = depth_map[y1:y2, x1:x2]
    valid = roi[(roi > MIN_DEPTH_M) & (roi < MAX_DEPTH_M)]
    if valid.size == 0:
        return None
    return float(np.median(valid))
===
"""
depth_loader.py
===============
CPE Perception Stack — Dataset-aware depth map loader.

Handles both SANPO and UASOL depth formats:

  SANPO-Synthetic / SANPO-Real:
      File   : <frame>.float16.gz
      Format : gzip-compressed float16 numpy array, shape (H, W)
      Values : metric depth in metres (no scale needed)
      Res    : 2208×1242 (downsized to match RGB frame by load_depth_map)

  UASOL:
      File   : img_depth.png (16-bit PNG)
      Format : uint16, scale × 0.256 = metres
"""

from __future__ import annotations
import gzip
from pathlib import Path

import cv2
import numpy as np

# Physical depth range filter — anything outside this is sensor noise or irrelevant
MIN_DEPTH_M = 0.3    # below this = too close / noise
MAX_DEPTH_M = 30.0   # beyond this = not actionable

# UASOL-only scale factor (SANPO values are already in metres as float16)
DEPTH_SCALES = {
    "sanpo": 1.0,     # float16 gz — already metres
    "uasol": 0.256,   # uint16 PNG — raw × 0.256 = metres
}

# Known SANPO depth resolution (synthetic + real)
SANPO_DEPTH_H = 1242
SANPO_DEPTH_W = 2208


def load_depth_map(depth_path: Path, source: str = "sanpo") -> np.ndarray | None:
    """
    Load a depth map and return a float32 array in metres, sized to match
    the RGB frame (resized if needed).

    Auto-detects format by file extension:
      *.float16.gz  →  SANPO (decompress → reshape float16 → float32)
      *.png         →  UASOL (cv2 ANYDEPTH → scale)

    Returns None if the file does not exist.
    """
    if not depth_path.exists():
        return None

    suffix = "".join(depth_path.suffixes).lower()   # e.g. ".float16.gz" or ".png"

    if suffix == ".float16.gz":
        # SANPO: gzip-compressed raw float16 binary, row-major (H, W).
        # Files contain 2 extra padding float16 values at the end — truncate to exact size.
        with gzip.open(depth_path, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.float16)
        n = SANPO_DEPTH_H * SANPO_DEPTH_W
        depth = raw[:n].reshape(SANPO_DEPTH_H, SANPO_DEPTH_W).astype(np.float32)
        return depth   # already in metres

    elif suffix == ".png":
        # UASOL: 16-bit PNG, scale to metres
        raw = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if raw is None:
            return None
        scale = DEPTH_SCALES.get(source, 1.0)
        return raw.astype(np.float32) * scale

    else:
        raise ValueError(f"Unknown depth format: {depth_path.name!r}")


def median_depth_in_box(
    depth_map: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    center_frac: float = 0.2,
    exclude_boxes: list | None = None,
) -> float | None:
    """
    Sample depth from the **central patch** of a bounding box.

    Uses only the inner `center_frac` fraction of the box — avoids averaging
    depth from edges / background pixels that bleed into the bbox.

    Args:
        depth_map:     2-D float32 depth array (metres).
        x1, y1, x2, y2: Bounding box corners.
        center_frac:   Fraction of bbox width/height to sample (default 0.2).
        exclude_boxes: List of (x1, y1, x2, y2) boxes to mask OUT before
                       sampling.  Used to prevent a background object from
                       reading depth *through* a closer foreground object
                       that overlaps it in image space.

    Returns:
        Median depth in metres, or None if no valid pixels found.
    """
    h, w = depth_map.shape[:2]

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    bw = max(1, int((x2 - x1) * center_frac / 2))
    bh = max(1, int((y2 - y1) * center_frac / 2))

    rx1, rx2 = max(0, cx - bw), min(w, cx + bw)
    ry1, ry2 = max(0, cy - bh), min(h, cy + bh)
    if rx1 >= rx2 or ry1 >= ry2:
        return None

    # Build usability mask — True = pixel is valid for sampling
    patch_h = ry2 - ry1
    patch_w = rx2 - rx1
    mask = np.ones((patch_h, patch_w), dtype=bool)

    # Mask out pixels that belong to any foreground (occluding) box
    if exclude_boxes:
        for (ex1, ey1, ex2, ey2) in exclude_boxes:
            mx1 = max(0, ex1 - rx1)
            my1 = max(0, ey1 - ry1)
            mx2 = min(patch_w, ex2 - rx1)
            my2 = min(patch_h, ey2 - ry1)
            if mx1 < mx2 and my1 < my2:
                mask[my1:my2, mx1:mx2] = False

    roi   = depth_map[ry1:ry2, rx1:rx2]
    valid = roi[(roi > MIN_DEPTH_M) & (roi < MAX_DEPTH_M) & mask]

    # ── Sparse depth fallback — expand patch 3× if no readings ────
    if valid.size == 0:
        bw2, bh2 = bw * 3, bh * 3
        rx1, rx2 = max(0, cx - bw2), min(w, cx + bw2)
        ry1, ry2 = max(0, cy - bh2), min(h, cy + bh2)
        patch_h = ry2 - ry1
        patch_w = rx2 - rx1
        mask2 = np.ones((patch_h, patch_w), dtype=bool)
        if exclude_boxes:
            for (ex1, ey1, ex2, ey2) in exclude_boxes:
                mx1 = max(0, ex1 - rx1); my1 = max(0, ey1 - ry1)
                mx2 = min(patch_w, ex2 - rx1); my2 = min(patch_h, ey2 - ry1)
                if mx1 < mx2 and my1 < my2:
                    mask2[my1:my2, mx1:mx2] = False
        roi   = depth_map[ry1:ry2, rx1:rx2]
        valid = roi[(roi > MIN_DEPTH_M) & (roi < MAX_DEPTH_M) & mask2]

    if valid.size == 0:
        return None
    return float(np.median(valid))
```

---

### 2. Depth-Guided Post-Processing — [yolo_tracker.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/yolo_tracker.py)

**Before:** Raw YOLO + ByteTrack output with no filtering.

**After:**
- **Class whitelist** — only navigation-relevant classes (person, car, bicycle, etc.)
- **Geometric validation** — rejects bad aspect ratios, tiny boxes, edge false positives
- **3-pass depth_rescore():**
  - Pass 1: Size sanity — drops boxes whose pixel height is wildly wrong for their depth
  - Pass 2: Confidence rescore — penalises implausible depth/class combos
  - Pass 3: Depth-guided NMS — overlapping boxes → keep more depth-consistent one
- [track()](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/yolo_tracker.py#215-266) now accepts [depth_map](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/depth_loader.py#41-76) for post-processing

```diff:yolo_tracker.py
"""
yolo_tracker.py
===============
CPE Perception Stack — YOLO + ByteTrack wrapper.

Provides a thin, stateless class around the Ultralytics YOLO model configured
to run ByteTrack for persistent per-object track IDs across frames.
"""

from pathlib import Path

import numpy as np
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
    ):
        self.model = YOLO(model_path)
        self.model.overrides["conf"]    = conf
        self.model.overrides["tracker"] = tracker

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
===
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

```

---

### 3. Pipeline Rewrite — [pipeline.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/pipeline.py)

**Before:** Single ROI for unlabeled obstacles, no occlusion handling, no frame skip, print every 50 frames.

**After:**
- **Grid-based unlabeled obstacles** — sweeps 5 columns across lower 60% with YOLO mask
- **Front-to-back occlusion ordering** — closer objects mask behind ones during depth sampling
- **Nth-frame skip** via `frame_step` param (default=1, set to 3 for ~67% compute reduction)
- **Streaming generator** — [run_perception_stream()](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/pipeline.py#114-229) yields per-frame for incremental output
- **tqdm progress bar** replaces sparse print
- Backward-compatible [run_perception()](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/pipeline.py#233-250) wraps the generator

```diff:pipeline.py
"""
pipeline.py
===========
CPE Perception Stack — Stage 1 Orchestrator: RGB + Depth → CSV rows.

Ties together yolo_tracker, depth_loader, and physics into the main
perception loop. Returns a list of flat row dicts ready for csv_writer.

Public API:
    run_perception(rgb_dir, depth_dir, fps, source) → list[dict]
"""

from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from src.perception_stack.yolo_tracker import YoloTracker
from src.perception_stack.depth_loader import load_depth_map, median_depth_in_box
from src.perception_stack.physics      import compute_bearing, compute_velocity
from src.perception_stack.csv_writer   import CSV_FIELDS

VELOCITY_WINDOW = 5   # number of frames in rolling velocity buffer
PROXIMITY_M = 2.5     # alert threshold for unlabeled obstacles

def detect_unlabeled_obstacle(depth_map: np.ndarray, frame_w: int, frame_h: int) -> dict | None:
    """
    Scans the central path for unclassified physical obstacles (poles, bollards, etc).
    Returns a pseudo-detection det dict if an obstacle is < PROXIMITY_M.
    """
    if depth_map is None:
        return None

    # Define walk-path ROI: bottom half, central 40% of the frame
    x1 = int(frame_w * 0.3)
    x2 = int(frame_w * 0.7)
    y1 = int(frame_h * 0.5)
    y2 = frame_h
    
    # Clip to depth map bounds (in case it is smaller than RGB)
    dh, dw = depth_map.shape[:2]
    dx1, dx2 = min(x1, dw), min(x2, dw)
    dy1, dy2 = min(y1, dh), min(y2, dh)
    
    roi = depth_map[dy1:dy2, dx1:dx2]
    
    # Find valid pixels closer than PROXIMITY_M
    close = roi[(roi > 0.3) & (roi < PROXIMITY_M)]
    
    if close.size > 0:
        dist = float(close.min())
        # Return a pseudo YOLO detection covering the ROI
        return {
            "track_id":    9999,   # reserved ID for unlabeled
            "class_name":  "unlabeled_obstacle",
            "confidence":  1.0,
            "x1":          x1,
            "y1":          y1,
            "x2":          x2,
            "y2":          y2,
            "cx":          (x1 + x2) / 2.0,
            "min_depth":   dist
        }
    return None


def run_perception(
    rgb_dir:   Path,
    depth_dir: Path | None,
    fps:       float,
    source:    str = "sanpo",
) -> list[dict]:
    """
    Stage 1 perception loop: YOLO + ByteTrack → depth → physics → CSV rows.

    Args:
        rgb_dir:   Folder of sorted RGB frames (JPEG/PNG).
        depth_dir: Folder of co-registered 16-bit depth PNGs, or None to skip depth.
        fps:       Video framerate — used to convert frame-count deltas to seconds.
        source:    'sanpo' or 'uasol' — controls depth scale applied in depth_loader.

    Returns:
        List of flat dicts with keys matching CSV_FIELDS (csv_writer.py).
        One dict per tracked object per frame.
    """
    frame_paths = sorted([
        p for p in rgb_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in {rgb_dir}")

    tracker = YoloTracker()

    # Per-track rolling depth history for velocity estimation
    # { track_id: deque[(frame_idx, distance_m)] }
    depth_history: dict = defaultdict(lambda: deque(maxlen=VELOCITY_WINDOW + 1))

    rows: list[dict] = []

    for frame_idx, rgb_path in enumerate(frame_paths):
        frame = cv2.imread(str(rgb_path))
        if frame is None:
            continue
        _, w = frame.shape[:2]

        # ── YOLO + ByteTrack ──
        detections = tracker.track(frame)

        # ── Load matched depth map ──
        # SANPO depth: <stem>.float16.gz  (gzip float16, already in metres)
        # UASOL depth: <stem>.png         (16-bit PNG, scale × 0.256)
        depth_map = None
        if depth_dir is not None:
            stem = rgb_path.stem   # e.g. "000000"
            for ext in (".float16.gz", ".png"):
                depth_path = depth_dir / (stem + ext)
                if depth_path.exists():
                    depth_map = load_depth_map(depth_path, source=source)
                    break
        # ── Find Unlabeled Obstacles (poles, walls) ──
        unlabeled = detect_unlabeled_obstacle(depth_map, w, frame.shape[0])
        if unlabeled:
            # Prevent double-counting if a YOLO object is already tracked at similar depth in that area
            closest_yolo = min(
                [median_depth_in_box(depth_map, d["x1"], d["y1"], d["x2"], d["y2"]) or 99.0 
                 for d in detections], 
                default=99.0
            )
            if unlabeled["min_depth"] < closest_yolo - 0.5:
                detections.append(unlabeled)

        for det in detections:
            tid = det["track_id"]

            # ── Depth ──
            if det.get("class_name") == "unlabeled_obstacle":
                distance_m = det["min_depth"]
                tid = f"9999_{frame_idx}"  # unique stateless ID per frame so it doesn't build fake velocity
            else:
                distance_m = None
                if depth_map is not None:
                    distance_m = median_depth_in_box(
                        depth_map, det["x1"], det["y1"], det["x2"], det["y2"]
                    )

            # ── Velocity ──
            if distance_m is not None and det.get("class_name") != "unlabeled_obstacle":
                depth_history[tid].append((frame_idx, distance_m))
            
            # Static obstacles have 0 velocity
            velocity_ms = 0.0
            if det.get("class_name") != "unlabeled_obstacle":
                velocity_ms = compute_velocity(list(depth_history[tid]), fps)

            # ── Bearing ──
            bearing = compute_bearing(det["cx"], w)

            rows.append({
                "frame_idx":   frame_idx,
                "source":      source,
                "track_id":    tid,
                "class":       det["class_name"],
                "confidence":  det["confidence"],
                "bbox_x1":     det["x1"],
                "bbox_y1":     det["y1"],
                "bbox_x2":     det["x2"],
                "bbox_y2":     det["y2"],
                "cx_px":       round(det["cx"], 1),
                "bearing_deg": round(bearing, 2),
                "distance_m":  round(distance_m, 2) if distance_m is not None else "",
                "velocity_ms": round(velocity_ms, 3),
            })

        if frame_idx % 50 == 0:
            n = sum(1 for r in rows if r["frame_idx"] == frame_idx)
            print(f"  [{frame_idx:05d}/{len(frame_paths)}] {n} objects tracked")

    return rows
===
"""
pipeline.py
===========
CPE Perception Stack — Stage 1 Orchestrator: RGB + Depth → structured rows.

Ties together yolo_tracker, depth_loader, and physics into the main
perception loop.  Supports nth-frame skip and a streaming generator API.

Public API:
    run_perception_stream(...)  → Generator[list[dict]]  (yields rows per frame)
    run_perception(...)         → list[dict]              (backward compat wrapper)
"""

from collections import defaultdict, deque
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from tqdm import tqdm

from src.perception_stack.yolo_tracker  import YoloTracker
from src.perception_stack.depth_loader  import load_depth_map, median_depth_in_box
from src.perception_stack.physics       import compute_bearing, compute_velocity

VELOCITY_WINDOW    = 5     # number of frames in rolling velocity buffer
PROXIMITY_M        = 8.0   # alert threshold for unlabeled obstacles
OBSTACLE_GRID_COLS = 5     # number of columns for grid sweep


# ── Grid-based unlabeled obstacle detection ───────────────────────────────────

def detect_unlabeled_obstacles(
    depth_map: np.ndarray,
    frame_w: int,
    frame_h: int,
    yolo_detections: list[dict],
) -> list[dict]:
    """
    Scan the lower 60% of the frame in a grid of vertical columns looking
    for depth-confirmed obstacles NOT already covered by YOLO detections.

    Returns a list of pseudo-detection dicts (may be empty).  Each covers
    one column where an unaccounted obstacle was found.
    """
    if depth_map is None:
        return []

    dh, dw = depth_map.shape[:2]
    y1_scan = int(frame_h * 0.4)
    y2_scan = frame_h
    dy1 = min(y1_scan, dh)
    dy2 = min(y2_scan, dh)

    # Build mask of pixels already claimed by YOLO boxes
    yolo_mask = np.zeros((dh, dw), dtype=bool)
    for d in yolo_detections:
        ex1 = max(0, min(d["x1"], dw))
        ey1 = max(0, min(d["y1"], dh))
        ex2 = max(0, min(d["x2"], dw))
        ey2 = max(0, min(d["y2"], dh))
        yolo_mask[ey1:ey2, ex1:ex2] = True

    obstacles = []
    col_w = frame_w // OBSTACLE_GRID_COLS

    for col in range(OBSTACLE_GRID_COLS):
        x1 = col * col_w
        x2 = x1 + col_w
        dx1 = min(x1, dw)
        dx2 = min(x2, dw)
        if dx1 >= dx2:
            continue

        roi_depth = depth_map[dy1:dy2, dx1:dx2]
        roi_mask  = yolo_mask[dy1:dy2, dx1:dx2]

        # Only look at pixels NOT already covered by YOLO
        unmasked = roi_depth[~roi_mask]
        close = unmasked[(unmasked > 0.3) & (unmasked < PROXIMITY_M)]

        if close.size > 0:
            dist = float(close.min())
            cx   = (x1 + x2) / 2.0
            obstacles.append({
                "track_id":   f"obs_{col}",
                "class_name": "unlabeled_obstacle",
                "confidence": 1.0,
                "x1": x1, "y1": y1_scan, "x2": x2, "y2": frame_h,
                "cx": cx,
                "min_depth": dist,
            })

    return obstacles


# ── Front-to-back depth sorting ───────────────────────────────────────────────

def _rough_depth(det: dict, depth_map: np.ndarray | None) -> float:
    """Quick center-pixel depth for sorting.  Unlabeled obstacles use min_depth."""
    if det.get("class_name") == "unlabeled_obstacle":
        return det.get("min_depth", 99.0)
    if depth_map is not None:
        h_dm, w_dm = depth_map.shape[:2]
        cx = min(int(det["cx"]), w_dm - 1)
        cy = min(int((det["y1"] + det["y2"]) / 2), h_dm - 1)
        v  = depth_map[cy, cx]
        return float(v) if v > 0 else 99.0
    return 99.0


# ── Streaming perception generator ───────────────────────────────────────────

def run_perception_stream(
    rgb_dir:    Path,
    depth_dir:  Path | None,
    fps:        float,
    source:     str = "sanpo",
    frame_step: int = 1,
) -> Generator[list[dict], None, None]:
    """
    Stage 1 streaming perception — yields a list of row dicts per processed frame.

    Args:
        rgb_dir:     Folder of sorted RGB frames (JPEG/PNG).
        depth_dir:   Folder of co-registered depth maps, or None.
        fps:         Video framerate for velocity calculation.
        source:      'sanpo' or 'uasol' — controls depth scale.
        frame_step:  Process every Nth frame (default 1 = all frames).
                     Set to 3 for SANPO preprocessing to cut compute by ~67%.

    Yields:
        List of flat row dicts for each processed frame.
    """
    frame_paths = sorted([
        p for p in rgb_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in {rgb_dir}")

    tracker = YoloTracker()

    # Per-track rolling depth history for velocity estimation
    depth_history: dict = defaultdict(lambda: deque(maxlen=VELOCITY_WINDOW + 1))

    # Apply nth-frame skip
    selected_indices = range(0, len(frame_paths), frame_step)

    for frame_idx in tqdm(selected_indices, desc="Perception", unit="frame"):
        rgb_path = frame_paths[frame_idx]
        frame = cv2.imread(str(rgb_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        # ── Load matched depth map ──
        depth_map = None
        if depth_dir is not None:
            stem = rgb_path.stem
            for ext in (".float16.gz", ".png", ".npz"):
                depth_path = depth_dir / (stem + ext)
                if depth_path.exists():
                    depth_map = load_depth_map(depth_path, source=source)
                    break

        # ── YOLO + ByteTrack + depth-guided post-processing ──
        detections = tracker.track(frame, depth_map=depth_map)

        # ── Grid-based unlabeled obstacle sweep ──
        unlabeled_list = detect_unlabeled_obstacles(depth_map, w, h, detections)
        detections.extend(unlabeled_list)

        # ── Sort front-to-back for occlusion masking ──
        detections.sort(key=lambda d: _rough_depth(d, depth_map))

        # ── Per-detection depth with occlusion masking ──
        frame_rows: list[dict] = []

        for det in detections:
            tid = det["track_id"]

            if det.get("class_name") == "unlabeled_obstacle":
                distance_m = det["min_depth"]
                tid = f"obs_{frame_idx}_{det['cx']:.0f}"
            else:
                distance_m = None
                if depth_map is not None:
                    # Build occlusion mask from already-processed closer objects
                    occluders = [
                        (r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"])
                        for r in frame_rows
                        if r["distance_m"] is not None
                    ]
                    distance_m = median_depth_in_box(
                        depth_map,
                        det["x1"], det["y1"], det["x2"], det["y2"],
                        exclude_boxes=occluders,
                    )

            # ── Velocity ──
            if distance_m is not None and det.get("class_name") != "unlabeled_obstacle":
                depth_history[tid].append((frame_idx, distance_m))

            velocity_ms = 0.0
            if det.get("class_name") != "unlabeled_obstacle":
                velocity_ms = compute_velocity(list(depth_history[tid]), fps)

            # ── Bearing ──
            bearing = compute_bearing(det["cx"], w)

            frame_rows.append({
                "frame_idx":   frame_idx,
                "source":      source,
                "track_id":    tid,
                "class":       det["class_name"],
                "confidence":  det["confidence"],
                "bbox_x1":     det["x1"],
                "bbox_y1":     det["y1"],
                "bbox_x2":     det["x2"],
                "bbox_y2":     det["y2"],
                "cx_px":       round(det["cx"], 1),
                "bearing_deg": round(bearing, 2),
                "distance_m":  round(distance_m, 2) if distance_m is not None else None,
                "velocity_ms": round(velocity_ms, 3),
            })

        yield frame_rows


# ── Backward-compatible wrapper ───────────────────────────────────────────────

def run_perception(
    rgb_dir:    Path,
    depth_dir:  Path | None,
    fps:        float,
    source:     str = "sanpo",
    frame_step: int = 1,
) -> list[dict]:
    """
    Stage 1 perception — returns all rows as a flat list (backward compatible).

    Wraps run_perception_stream() and materialises the generator.
    See run_perception_stream() for argument docs.
    """
    rows: list[dict] = []
    for frame_rows in run_perception_stream(rgb_dir, depth_dir, fps, source, frame_step):
        rows.extend(frame_rows)
    return rows

```

---

### 4. Batch Tensor Physics — [physics.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/physics.py)

Added two vectorised functions: [batch_compute_bearing()](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/physics.py#96-117) and [batch_kinetic_score()](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/physics.py#119-143) operating on torch tensors for frame-level batch processing. Original scalar functions unchanged.

```diff:physics.py
"""
physics.py
==========
CPE Perception Stack — Physics calculations for perceived objects.

Functions:
    compute_bearing     : pixel x-coordinate → bearing in degrees
    compute_velocity    : rolling window depth history → closing velocity m/s
    kinetic_score       : K = severity × v² / max(d, ε)
    bearing_label       : bearing degrees → human-readable direction string
"""

# ── Class severity weights ─────────────────────────────────────────────────────
# Higher = more dangerous when combined with velocity/distance in kinetic score.
CLASS_SEVERITY: dict[str, float] = {
    "person":     1.0,
    "bicycle":    1.2,
    "car":        2.0,
    "motorcycle": 1.8,
    "bus":        2.5,
    "truck":      2.5,
    "dog":        0.8,
}
DEFAULT_SEVERITY = 1.0
EPSILON = 0.5   # metres — prevents division by zero for very close objects


def compute_bearing(cx_px: float, frame_width: int, hfov_deg: float = 70.0) -> float:
    """
    Convert the pixel x-coordinate of an object's centre to a bearing in degrees.

    Returns:
        Negative = object is to the LEFT of ego path.
        Positive = object is to the RIGHT.
        0        = directly ahead.

    Assumes a horizontal field of view of 70° (typical phone / dashcam lens).
    """
    normalised = (cx_px - frame_width / 2) / (frame_width / 2)   # normalise to [-1, 1]
    return normalised * (hfov_deg / 2)


def compute_velocity(depth_history: list[tuple[int, float]], fps: float) -> float:
    """
    Estimate closing velocity (m/s) from a rolling window of (frame_idx, distance_m) pairs.

    Positive return value means the object is APPROACHING (depth decreasing).
    Returns 0.0 if fewer than 2 history samples exist or the object is moving away.

    Args:
        depth_history:  List of (frame_idx, distance_m) in chronological order.
        fps:            Video framerate — used to convert frame delta to seconds.
    """
    if len(depth_history) < 2:
        return 0.0
    (f0, d0) = depth_history[0]
    (f1, d1) = depth_history[-1]
    dt = (f1 - f0) / fps
    if dt <= 0:
        return 0.0
    raw_v = (d0 - d1) / dt   # positive = object closing in
    return max(0.0, raw_v)   # clamp: don't report negative (retreating) velocities


def kinetic_score(distance_m: float, velocity_ms: float, class_name: str) -> float:
    """
    Compute the kinetic threat score for one tracked object.

    Formula:  K = class_severity × (velocity_ms²) / max(distance_m, ε)
    Higher K → higher threat level.

    Args:
        distance_m:  Metric depth of the object in metres.
        velocity_ms: Closing velocity in m/s (positive = approaching ego).
        class_name:  COCO class name string (e.g. 'car', 'person').
    """
    severity = CLASS_SEVERITY.get(class_name, DEFAULT_SEVERITY)
    return severity * (velocity_ms ** 2) / max(distance_m, EPSILON)


def bearing_label(deg: float) -> str:
    """Convert a bearing (degrees) to a human-readable direction for the Fact Sheet."""
    if deg < -30:
        return "far-left"
    if deg < -10:
        return "left"
    if deg < 10:
        return "ahead"
    if deg < 30:
        return "right"
    return "far-right"
===
"""
physics.py
==========
CPE Perception Stack — Physics calculations for perceived objects.

Functions:
    compute_bearing     : pixel x-coordinate → bearing in degrees
    compute_velocity    : rolling window depth history → closing velocity m/s
    kinetic_score       : K = severity × v² / max(d, ε)
    bearing_label       : bearing degrees → human-readable direction string
"""

# ── Class severity weights ─────────────────────────────────────────────────────
# Higher = more dangerous when combined with velocity/distance in kinetic score.
CLASS_SEVERITY: dict[str, float] = {
    "person":     1.0,
    "bicycle":    1.2,
    "car":        2.0,
    "motorcycle": 1.8,
    "bus":        2.5,
    "truck":      2.5,
    "dog":        0.8,
}
DEFAULT_SEVERITY = 1.0
EPSILON = 0.5   # metres — prevents division by zero for very close objects


def compute_bearing(cx_px: float, frame_width: int, hfov_deg: float = 70.0) -> float:
    """
    Convert the pixel x-coordinate of an object's centre to a bearing in degrees.

    Returns:
        Negative = object is to the LEFT of ego path.
        Positive = object is to the RIGHT.
        0        = directly ahead.

    Assumes a horizontal field of view of 70° (typical phone / dashcam lens).
    """
    normalised = (cx_px - frame_width / 2) / (frame_width / 2)   # normalise to [-1, 1]
    return normalised * (hfov_deg / 2)


def compute_velocity(depth_history: list[tuple[int, float]], fps: float) -> float:
    """
    Estimate closing velocity (m/s) from a rolling window of (frame_idx, distance_m) pairs.

    Positive return value means the object is APPROACHING (depth decreasing).
    Returns 0.0 if fewer than 2 history samples exist or the object is moving away.

    Args:
        depth_history:  List of (frame_idx, distance_m) in chronological order.
        fps:            Video framerate — used to convert frame delta to seconds.
    """
    if len(depth_history) < 2:
        return 0.0
    (f0, d0) = depth_history[0]
    (f1, d1) = depth_history[-1]
    dt = (f1 - f0) / fps
    if dt <= 0:
        return 0.0
    raw_v = (d0 - d1) / dt   # positive = object closing in
    return max(0.0, raw_v)   # clamp: don't report negative (retreating) velocities


def kinetic_score(distance_m: float, velocity_ms: float, class_name: str) -> float:
    """
    Compute the kinetic threat score for one tracked object.

    Formula:  K = class_severity × (velocity_ms²) / max(distance_m, ε)
    Higher K → higher threat level.

    Args:
        distance_m:  Metric depth of the object in metres.
        velocity_ms: Closing velocity in m/s (positive = approaching ego).
        class_name:  COCO class name string (e.g. 'car', 'person').
    """
    severity = CLASS_SEVERITY.get(class_name, DEFAULT_SEVERITY)
    return severity * (velocity_ms ** 2) / max(distance_m, EPSILON)


def bearing_label(deg: float) -> str:
    """Convert a bearing (degrees) to a human-readable direction for the Fact Sheet."""
    if deg < -30:
        return "far-left"
    if deg < -10:
        return "left"
    if deg < 10:
        return "ahead"
    if deg < 30:
        return "right"
    return "far-right"


# ── Vectorised batch operations (torch tensors) ──────────────────────────────

def batch_compute_bearing(
    cx_tensor,
    frame_width: int,
    hfov_deg: float = 70.0,
):
    """
    Vectorised bearing for N detections at once.

    Args:
        cx_tensor: 1-D tensor/array of centre-x pixel coordinates (N,).
        frame_width: Frame width in pixels.
        hfov_deg:    Horizontal field-of-view in degrees.

    Returns:
        Tensor/array of bearing values in degrees (N,).
    """
    import torch
    if not isinstance(cx_tensor, torch.Tensor):
        cx_tensor = torch.tensor(cx_tensor, dtype=torch.float32)
    normalised = (cx_tensor - frame_width / 2) / (frame_width / 2)
    return normalised * (hfov_deg / 2)


def batch_kinetic_score(
    distances,
    velocities,
    severity_weights,
):
    """
    Vectorised kinetic score for N detections: K = severity × v² / max(d, ε).

    Args:
        distances:        1-D tensor/array of distances in metres (N,).
        velocities:       1-D tensor/array of closing velocities m/s (N,).
        severity_weights: 1-D tensor/array of class severity weights (N,).

    Returns:
        Tensor/array of kinetic scores (N,).
    """
    import torch
    if not isinstance(distances, torch.Tensor):
        distances = torch.tensor(distances, dtype=torch.float32)
    if not isinstance(velocities, torch.Tensor):
        velocities = torch.tensor(velocities, dtype=torch.float32)
    if not isinstance(severity_weights, torch.Tensor):
        severity_weights = torch.tensor(severity_weights, dtype=torch.float32)
    return severity_weights * (velocities ** 2) / torch.clamp(distances, min=EPSILON)
```

---

### 5. Streaming CSV Writer — [csv_writer.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/csv_writer.py)

Added [StreamingCSVWriter](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/csv_writer.py#47-82) context manager — opens file once, accepts incremental row batches from the generator without holding everything in RAM.

```diff:csv_writer.py
"""
csv_writer.py
=============
CPE Perception Stack — CSV schema definition and writer for Stage 1 output.

Defines the canonical column layout so Stage 1 (pipeline.py) and Stage 2
(fact_sheet_builder.py) share the exact same field names without duplication.
"""

import csv
from pathlib import Path

# ── Canonical CSV schema ─────────────────────────────────────────────────────
# This is the contract between Stage 1 (perception) and Stage 2 (fact sheets).
CSV_FIELDS = [
    "frame_idx",    # int    — frame number in source video/sequence
    "source",       # str    — 'sanpo' or 'uasol'
    "track_id",     # int    — persistent ByteTrack object ID
    "class",        # str    — COCO class label (e.g. 'car', 'person')
    "confidence",   # float  — YOLO detection confidence [0, 1]
    "bbox_x1",      # int    — bounding box left
    "bbox_y1",      # int    — bounding box top
    "bbox_x2",      # int    — bounding box right
    "bbox_y2",      # int    — bounding box bottom
    "cx_px",        # float  — horizontal centre pixel
    "bearing_deg",  # float  — left/right bearing from ego path (negative=left)
    "distance_m",   # float  — median metric depth inside ROI (empty if no depth map)
    "velocity_ms",  # float  — estimated closing velocity m/s (positive = approaching)
]


def write_csv(rows: list[dict], out_path: Path) -> None:
    """
    Write a list of perception row dicts to a CSV file using the canonical schema.

    Args:
        rows:     List of dicts — each must contain keys matching CSV_FIELDS.
        out_path: Destination file path. Parent directories are created if missing.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
===
"""
csv_writer.py
=============
CPE Perception Stack — CSV schema definition and writer for Stage 1 output.

Defines the canonical column layout so Stage 1 (pipeline.py) and Stage 2
(fact_sheet_builder.py) share the exact same field names without duplication.
"""

import csv
from pathlib import Path

# ── Canonical CSV schema ─────────────────────────────────────────────────────
# This is the contract between Stage 1 (perception) and Stage 2 (fact sheets).
CSV_FIELDS = [
    "frame_idx",    # int    — frame number in source video/sequence
    "source",       # str    — 'sanpo' or 'uasol'
    "track_id",     # int    — persistent ByteTrack object ID
    "class",        # str    — COCO class label (e.g. 'car', 'person')
    "confidence",   # float  — YOLO detection confidence [0, 1]
    "bbox_x1",      # int    — bounding box left
    "bbox_y1",      # int    — bounding box top
    "bbox_x2",      # int    — bounding box right
    "bbox_y2",      # int    — bounding box bottom
    "cx_px",        # float  — horizontal centre pixel
    "bearing_deg",  # float  — left/right bearing from ego path (negative=left)
    "distance_m",   # float  — median metric depth inside ROI (empty if no depth map)
    "velocity_ms",  # float  — estimated closing velocity m/s (positive = approaching)
]


def write_csv(rows: list[dict], out_path: Path) -> None:
    """
    Write a list of perception row dicts to a CSV file using the canonical schema.

    Args:
        rows:     List of dicts — each must contain keys matching CSV_FIELDS.
        out_path: Destination file path. Parent directories are created if missing.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class StreamingCSVWriter:
    """
    Context manager for streaming CSV output — writes rows incrementally
    as the perception generator yields them, without holding all data in RAM.

    Usage:
        with StreamingCSVWriter(out_path) as writer:
            for frame_rows in run_perception_stream(...):
                writer.write_rows(frame_rows)
    """

    def __init__(self, out_path: Path):
        self.out_path = out_path
        self._file = None
        self._writer = None
        self.rows_written = 0

    def __enter__(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.out_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        return self

    def write_rows(self, rows: list[dict]) -> None:
        """Write a batch of row dicts (typically one frame's worth)."""
        if self._writer is None:
            raise RuntimeError("StreamingCSVWriter not open — use as context manager")
        self._writer.writerows(rows)
        self.rows_written += len(rows)

    def __exit__(self, *exc):
        if self._file:
            self._file.close()
        return False
```

---

### 6. CLI + Exports + Dedup

| File | Change |
|---|---|
| [run_perception.py](file:///e:/capstone/CompositePerceptionEngine/tools/run_perception.py) | `--frame_step` arg, uses streaming API |
| [__init__.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/__init__.py) | Exports [run_perception_stream](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/pipeline.py#114-229), [StreamingCSVWriter](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/csv_writer.py#47-82), batch ops |
| [fact_sheet.py](file:///e:/capstone/CompositePerceptionEngine/src/shared/fact_sheet.py) | `CLASS_SEVERITY` now imports from [physics.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/physics.py) (single source of truth) |

---

## Validation

- ✅ All 8 files pass `py_compile` syntax verification
- ✅ Backward-compatible — [run_perception()](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/pipeline.py#233-250) and [write_csv()](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/csv_writer.py#32-45) APIs unchanged
- ✅ All new APIs exported through [__init__.py](file:///e:/capstone/CompositePerceptionEngine/src/__init__.py)

## Usage

```bash
# SANPO preprocessing — every 3rd frame (recommended)
python tools/run_perception.py \
    --rgb_dir data/sanpo/sample/rgb \
    --depth_dir data/sanpo/sample/depth \
    --out data/processed/sanpo_perception.csv \
    --source sanpo --fps 30 --frame_step 3
```
