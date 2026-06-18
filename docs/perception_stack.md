# Perception Stack — Component Doc

**Package:** `src/perception_stack/`
**Role:** Stage 1 — converts raw RGB video frames + depth maps into structured CSV rows (one row per tracked object per frame). Stage 2 — converts that CSV into JSONL training data for SLM-1.

---

## Architecture at a Glance

```
RGB frames + Depth maps
        │
 depth_loader.py      ← loads SANPO .float16.gz / UASOL .png depth
        │
 yolo_tracker.py      ← YOLO26n + ByteTrack + depth-guided post-processing
        │
 pipeline.py          ← Stage 1 orchestrator (streaming generator)
   ├── unlabeled obstacle grid sweep (depth-only, 5 columns)
   ├── front-to-back occlusion ordering
   └── per-detection: depth → velocity → bearing → row dict
        │
 physics.py           ← compute_bearing, compute_velocity, kinetic_score
 csv_writer.py        ← StreamingCSVWriter → CSV (canonical schema)
        │
 fact_sheet_builder.py← Stage 2: CSV → K₀ + K₊₂ kinetic scoring → JSONL
```

---

## Modules

### `depth_loader.py`
Loads depth maps from disk and samples depth values from bounding box regions.

**Supports:**
- `SANPO` — `.float16.gz` gzip-compressed binary, already metres. Shape `(1242, 2208)` float16; 2 padding values truncated on load.
- `UASOL` — 16-bit PNG, scale `× 0.256` → metres.

**Key functions:**
- `load_depth_map(depth_path, source)` → `np.ndarray | None` — auto-detects format from file extension.
- `median_depth_in_box(depth_map, x1, y1, x2, y2, center_frac=0.2, exclude_boxes=None)` — samples only the central 20% patch of the bbox (avoids edge/background bleed). Maskes out `exclude_boxes` pixels for occlusion handling. Auto-expands 3× if centre patch returns no valid pixels (sparse ZED depth fallback).

**Constants:** `MIN_DEPTH_M = 0.3`, `MAX_DEPTH_M = 30.0`

---

### `yolo_tracker.py`
Wraps YOLO26n + ByteTrack with depth-guided post-processing.

**Model:** `yolo26n.pt` (edge-optimised nano, default conf=0.30, tracker=bytetrack.yaml)

**Processing pipeline on each frame:**
1. **Class whitelist** — drops non-navigation COCO classes. Allowed: `person, bicycle, car, motorcycle, bus, truck, dog, cat, traffic light, stop sign, umbrella, backpack, suitcase, unlabeled_obstacle`.
2. **Geometric validation** (`is_valid_detection`) — rejects: aspect ratio >4:1, area <500px², full-width boxes at bottom edge (ground-plane FPs).
3. **`depth_rescore(detections, depth_map)`** — 3-pass depth-guided filtering:
   - **Pass 1 — Size sanity:** pixel height vs `CLASS_REAL_HEIGHT_M[cls] × FOCAL_PX / d`. Drops if outside 4× tolerance.
   - **Pass 2 — Confidence rescoring:** penalises physically impossible combos (car at <1.5m →×0.3, person >15m →×0.5, anything <0.5m →×0.2). Drops conf <0.25.
   - **Pass 3 — Depth-guided NMS:** for IoU >0.5 overlap, keeps box with higher `conf / depth` score instead of just highest confidence.

**`track(frame, depth_map=None)`** → `list[dict]` with keys: `track_id, class_name, confidence, x1, y1, x2, y2, cx`

---

### `pipeline.py`
Stage 1 orchestrator. The main function is a streaming generator.

**`run_perception_stream(rgb_dir, depth_dir, fps, source, frame_step=1)`** → `Generator[list[dict]]`
- Loads sorted RGB frames, applies `frame_step` (every Nth frame).
- Per frame: loads depth → `tracker.track(frame, depth_map)` → `detect_unlabeled_obstacles()` → sort front-to-back → per-detection depth (with occlusion mask) + velocity + bearing → `yield frame_rows`.
- Progress via `tqdm`.

**`detect_unlabeled_obstacles(depth_map, frame_w, frame_h, yolo_detections)`** → `list[dict]`
- Divides lower 60% of frame into `OBSTACLE_GRID_COLS=5` vertical columns.
- Masks out pixels already covered by YOLO boxes.
- For each column, finds pixels with depth `0.3–8.0m` not covered by YOLO → emits pseudo-detection with `class_name="unlabeled_obstacle"`, `track_id="obs_N"`.

**Occlusion ordering:** Detections sorted front-to-back before the depth sampling loop. Each processed detection's bbox is added to `occluders` list passed to subsequent `median_depth_in_box` calls.

**`run_perception()`** — backward-compatible wrapper that materialises the generator into a flat list.

**`PROXIMITY_M = 8.0`** — obstacle alert range.

---

### `physics.py`
Physics calculations. Scalar and batch (tensor) APIs.

| Function | Description |
|---|---|
| `compute_bearing(cx_px, frame_width, hfov_deg=70.0)` | pixel x → degrees. Negative=left, positive=right, 0=ahead. |
| `compute_velocity(depth_history, fps)` | rolling `(frame_idx, depth_m)` window → closing velocity m/s. Clamped ≥0. |
| `kinetic_score(distance_m, velocity_ms, class_name)` | `K = severity × v² / max(d, 0.5)` |
| `bearing_label(deg)` | degrees → `far-left/left/ahead/right/far-right` |
| `batch_compute_bearing(cx_tensor, frame_width)` | torch-vectorised, N detections at once |
| `batch_kinetic_score(distances, velocities, severity_weights)` | torch-vectorised kinetic score |

**`CLASS_SEVERITY`:** `person=1.0, bicycle=1.2, car=2.0, motorcycle=1.8, bus=2.5, truck=2.5, dog=0.8`

> `CLASS_SEVERITY` is the single source of truth — `fact_sheet_builder.py` imports from here. Do not duplicate it.

---

### `csv_writer.py`
Canonical CSV schema + writers.

**`CSV_FIELDS`** (the schema contract between Stage 1 and Stage 2):
```
frame_idx, source, track_id, class, confidence,
bbox_x1, bbox_y1, bbox_x2, bbox_y2, cx_px,
bearing_deg, distance_m, velocity_ms
```

**`write_csv(rows, out_path)`** — batch writer (all rows in memory).

**`StreamingCSVWriter`** — context manager. Opens file once, writes header, accepts `write_rows(batch)` calls incrementally from the generator. `.rows_written` for final count.

---

### `fact_sheet_builder.py`
Stage 2: CSV → JSONL for SLM-1 supervised fine-tuning.

**`load_perception_csv(csv_path)`** → `dict[frame_idx → list[row_dict]]` — casts numeric fields, empty strings → None.

**`build_fact_sheets(frames, fps, lookahead_s, out_path)`** → `(written, skipped)`
1. Per frame: compute `K₀` (present kinetic score) for every object. Sort by K desc. Find `present_threat` (highest-K with K>0).
2. Look `lookahead_s × fps` frames ahead. Compute `K₊₂` for future frame. If `present_threat.track_id == future_best.track_id` → `future_confirmed=True`.
3. Write JSONL record: `{system, user: "[SCENARIO FACT SHEET] ...", assistant: {primary_threat, track_id, class, distance_m, velocity_ms, kinetic_score, future_confirmed, reason}}`.

Fact sheet format: `Object_01: person, 3.2m, v=1.2m/s, ahead [K=0.452] | Object_02: ...`

Frames with no valid threat (all K=0) are skipped.

---

## Data Flow

```
tools/run_perception.py
  └─→ run_perception_stream() + StreamingCSVWriter
        └─→ CSV (e.g. data/processed/sanpo_perception.csv)

tools/run_fact_sheets.py
  └─→ load_perception_csv() + build_fact_sheets()
        └─→ JSONL (e.g. data/training/train.jsonl)
                └─→ [Colab] SFTTrainer + LoRA on Qwen2.5-1.5B-Instruct
```

## CLI Quick Reference

```bash
# Stage 1 — every 3rd frame (recommended for SANPO at 30fps)
python tools/run_perception.py \
    --rgb_dir data/sanpo/sample/rgb \
    --depth_dir data/sanpo/sample/depth \
    --out data/processed/sanpo_perception.csv \
    --source sanpo --fps 30 --frame_step 3

# Stage 2 — CSV → JSONL
python tools/run_fact_sheets.py \
    --csv data/processed/sanpo_perception.csv \
    --out data/training/train.jsonl \
    --fps 30 --lookahead_s 2.0
```
