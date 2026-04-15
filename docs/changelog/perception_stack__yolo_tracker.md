# Changelog — `src/perception_stack/yolo_tracker.py`

> Wraps YOLO26n + ByteTrack for stateful multi-object tracking, with depth-guided post-processing.

---

## [v1.2] — 2026-03-30 | Depth-Guided Post-Processing Pipeline

**Session:** *Optimizing CPE Perception Pipeline Performance*

### What Changed

This was a major rewrite of the post-`track()` processing layer. Three entirely new subsystems were added, all ported from prototype notebook experiments into production.

#### 1. Class Whitelist Filter — `ALLOWED_CLASSES`

**Problem:** YOLO detects all 80 COCO classes, including irrelevant ones (toothbrush, baseball bat, etc.) that waste downstream processing cycles and pollute the CSV fact sheets.

**Fix:** Added `ALLOWED_CLASSES` set. Only detections matching navigation-relevant classes pass through to the pipeline.

```python
ALLOWED_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "dog", "cat", "traffic light", "stop sign", "umbrella",
    "backpack", "suitcase", "unlabeled_obstacle",
}
```

#### 2. Geometric Validation — `is_valid_detection()`

**Problem:** YOLO produces several categories of structural false positives:
- Ultra-wide boxes from horizontal surface reflections
- Tiny sub-500px² specks from distance or lens artifacts  
- Full-width boxes touching the bottom edge (ground plane catching as object)

**Fix:** New module-level function that rejects all three patterns before depth processing:

```python
def is_valid_detection(det: dict, frame_w: int, frame_h: int) -> bool:
    # Reject aspect ratio > 4:1 (ultra-wide boxes)
    # Reject area < 500 px²
    # Reject full-width boxes at bottom edge (ground FPs)
```

#### 3. Three-Pass Depth Rescoring — `depth_rescore()`

The core new feature. Accepts a `depth_map` and runs 3 sequential filtering passes:

**Pass 1 — Size Sanity Check:**  
At depth `d`, an object of known real height `H` should appear `H * FOCAL_PX / d` pixels tall. Boxes that deviate by more than `SIZE_TOLERANCE = 4.0×` are dropped.

```python
FOCAL_PX = 960.0  # SANPO chest cam ~70° HFOV at 1280px width
CLASS_REAL_HEIGHT_M = {"person": 1.7, "car": 1.5, ...}  # physical heights
```

**Pass 2 — Confidence Rescoring:**  
Penalises physically impossible depth/class combinations:
- `car/truck/bus` at `< 1.5m` → ×0.3 penalty (chest cam can't be that close to a vehicle)
- `person` at `> 15m` → ×0.5 penalty (unreliable at range)
- Anything at `< 0.5m` → ×0.2 penalty (almost certainly sensor noise)

Detections that fall below `conf < 0.25` after rescoring are dropped.

**Pass 3 — Depth-Guided NMS:**  
Standard NMS keeps the highest-confidence box when IoU > 0.5. This replaces that with a depth-aware score: `confidence / max(depth, 0.1)`. The box that is both closer and more confident survives.

#### 4. `track()` now accepts `depth_map`

```python
# Before
def track(self, frame: np.ndarray) -> list[dict]:

# After
def track(self, frame: np.ndarray, depth_map: np.ndarray | None = None) -> list[dict]:
```

The internal loop now applies whitelist + geometric validation per detection, then calls `depth_rescore()` at the end.

### Backward Compatibility
- ✅ `depth_map=None` skips all depth post-processing — existing call sites unchanged.

---

## [v1.1] — 2026-03-27 | Unlabeled Obstacle Problem Identified

**Session:** *YOLO isn't detecting the pole in front of the person*

### What Was Identified (Not Yet Fixed Here)

YOLO's COCO training set does not include standalone poles, bollards, low fences, or similar unlabeled infrastructure objects. These near-field obstacles are significant collision risks for visually-impaired pedestrians but will never appear in YOLO's output.

**Decision:** Handle at the pipeline level (not tracker level) via depth-map scanning. This is implemented in `pipeline.py`'s `detect_unlabeled_obstacles()`. The tracker's whitelist includes `"unlabeled_obstacle"` so that pseudo-detections from the pipeline can flow through without being filtered.

---

## [v1.0] — 2026-03-25 | Initial Implementation

**Session:** *Implementing Argus-REI Perception Pipeline*

### What Was Built

- `YoloTracker` class wrapping `ultralytics.YOLO` with ByteTrack (`bytetrack.yaml`)
- Default model: `yolo26n.pt` (edge-optimized nano variant)
- Default confidence: `0.30`
- `track(frame)` — runs ByteTrack, extracts `track_id`, `class_name`, `confidence`, `x1/y1/x2/y2`, `cx`
- No filtering or post-processing — raw YOLO output passed straight through

---

## File Reference
- **Source:** [`src/perception_stack/yolo_tracker.py`](../../src/perception_stack/yolo_tracker.py)
- **Uses:** `ultralytics`, `numpy`
- **Used by:** `pipeline.py`
- **Model file:** `yolo26n.pt` (root directory)
