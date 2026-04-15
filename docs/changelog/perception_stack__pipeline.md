# Changelog — `src/perception_stack/pipeline.py`

> Stage 1 orchestrator — ties together YOLO tracking, depth loading, and physics computation into a per-frame perception loop that outputs structured CSV rows.

---

## [v2.0] — 2026-03-30 | Full Pipeline Rewrite

**Session:** *Optimizing CPE Perception Pipeline Performance*

This was the largest single-file change in the project. The pipeline was rewritten around a streaming generator architecture with 5 major new capabilities.

### What Changed

#### 1. Grid-Based Unlabeled Obstacle Detection

**Before:** A single central-column ROI (`x: 30–70%, y: 50–100%`) scanned for unclassified obstacles. One detection per frame, returned as a single pseudo-detection. No separation between columns.

**After:** `detect_unlabeled_obstacles()` splits the lower 60% of the frame into `OBSTACLE_GRID_COLS = 5` vertical columns. Each column is scanned independently. The YOLO bounding box mask is applied — only pixels **not already covered** by a YOLO detection are considered. Multiple obstacle detections can be returned per frame (one per column).

```python
# Before — single detection
def detect_unlabeled_obstacle(depth_map, frame_w, frame_h) -> dict | None

# After — list of column-wise detections
def detect_unlabeled_obstacles(depth_map, frame_w, frame_h, yolo_detections) -> list[dict]
```

Track IDs for column obstacles: `"obs_0"` through `"obs_4"`.

The `PROXIMITY_M` threshold was also bumped from `2.5m` → `8.0m` to capture medium-range unlabeled obstacles that warrant narration even if not immediately dangerous.

#### 2. Front-to-Back Occlusion Ordering

**Before:** Detections were processed in arbitrary YOLO output order. Depth was sampled independently for every box regardless of spatial overlap.

**After:** All detections (YOLO + unlabeled obstacles) are **sorted front-to-back** by center-pixel depth before the per-detection loop. As each detection is processed, its bounding box corners are added to an `occluders` list. The next (farther) detection calls `median_depth_in_box(..., exclude_boxes=occluders)` so that closer objects mask the depth sampling area for the object they occlude.

```python
# Sort front-to-back
detections.sort(key=lambda d: _rough_depth(d, depth_map))

# Build occluders list incrementally
occluders = [(r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"])
             for r in frame_rows if r["distance_m"] is not None]
distance_m = median_depth_in_box(depth_map, ..., exclude_boxes=occluders)
```

#### 3. Nth-Frame Skip — `frame_step` Parameter

**Before:** Processed every single frame, resulting in redundant computation for 30fps video where scene change between adjacent frames is minimal.

**After:** `run_perception_stream()` accepts `frame_step: int = 1`. Setting `frame_step=3` processes every 3rd frame (~67% compute reduction). The existing velocity calculation uses `frame_idx` (the true original index) for accurate time deltas even when frames are skipped.

```python
selected_indices = range(0, len(frame_paths), frame_step)
```

#### 4. Streaming Generator Architecture

**Before:** `run_perception()` accumulated all rows into a single in-memory list and returned it. For large SANPO sequences (thousands of frames), this consumed gigabytes of RAM before writing a single CSV row.

**After:** `run_perception_stream()` is a generator that **yields** `list[dict]` per processed frame. The caller (e.g., `run_perception.py`) plugs this into `StreamingCSVWriter` which writes each frame's rows immediately to disk.

```python
def run_perception_stream(...) -> Generator[list[dict], None, None]:
    ...
    yield frame_rows   # yields per frame, not at the end
```

The original `run_perception()` still exists as a backward-compatible wrapper that materialises the generator into a flat list.

#### 5. Progress Bar — `tqdm` Replaces Print Statements

**Before:** `if frame_idx % 50 == 0: print(...)` — sparse and non-interactive feedback.

**After:** `tqdm` progress bar with `desc="Perception"` and `unit="frame"` shows real-time ETA, elapsed time, and frame count. Controlled by `selected_indices` so it accounts for nth-frame skip correctly.

### Other Updates

| Item | Before | After |
|---|---|---|
| `PROXIMITY_M` | `2.5` | `8.0` (wider obstacle net) |
| depth extension check | `.float16.gz`, `.png` | `.float16.gz`, `.png`, `.npz` |
| track ID for unlabeled | `f"9999_{frame_idx}"` | `f"obs_{frame_idx}_{det['cx']:.0f}"` (col-specific) |
| `distance_m` in output | `""` if None | `None` if None |

### New Constants

```python
OBSTACLE_GRID_COLS = 5   # number of columns for grid sweep
PROXIMITY_M        = 8.0  # alert threshold (was 2.5)
```

---

## [v1.1] — 2026-03-27 | Unlabeled Obstacle Detection Added

**Session:** *YOLO isn't detecting the pole in front of the person*

### What Was Added

- `detect_unlabeled_obstacle(depth_map, frame_w, frame_h)` — scanned the central 40% of the frame's bottom half for close unclassified obstacles using raw depth values
- Track ID `9999` with per-frame unique suffix to prevent false velocity buildup
- Double-counting prevention: unlabeled obstacle only appended if `min_depth < closest_yolo_depth - 0.5m`

---

## [v1.0] — 2026-03-25 | Initial Implementation

**Session:** *Implementing Argus-REI Perception Pipeline*

### What Was Built

- `run_perception(rgb_dir, depth_dir, fps, source)` — main loop over sorted frame paths
- Per-frame: reads RGB, runs YOLO+ByteTrack, loads matched depth map (stem-based filename match), computes bearing + velocity for each detection
- Depth history: `defaultdict(deque(maxlen=6))` per track ID
- Output: flat list of row dicts with keys matching `CSV_FIELDS`
- Progress: `print()` every 50 frames

---

## File Reference
- **Source:** [`src/perception_stack/pipeline.py`](../../src/perception_stack/pipeline.py)
- **Uses:** `cv2`, `numpy`, `tqdm`, `yolo_tracker`, `depth_loader`, `physics`
- **Used by:** `tools/run_perception.py`, internal imports
- **Exports:** `run_perception_stream`, `run_perception`, `detect_unlabeled_obstacles`
