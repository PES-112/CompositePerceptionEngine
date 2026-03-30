# CPE Perception Pipeline — Performance Overhaul

## Background & Analysis

### Notebook vs Perception Layer Code

The notebook ([CPE_Pipeline_final.ipynb](file:///e:/capstone/CompositePerceptionEngine/notebooks/CPE_Pipeline_final.ipynb), ~6.5 MB) is a monolithic exploratory prototype. The perception layer is the modularised production code. After thorough review:

**The perception layer code is clearly better** — it has proper separation of concerns (6 focused modules), clean public APIs, a canonical CSV schema contract between Stage 1 and Stage 2, and is import-friendly for downstream tools. **All optimisations should be applied to the perception layer.**

### Current Performance Bottlenecks Identified

| Bottleneck | Location | Impact |
|---|---|---|
| **Sequential frame I/O** | [pipeline.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/pipeline.py) L102-105 | `cv2.imread()` blocks the main thread for every frame — no overlap with GPU inference |
| **Per-frame gzip decompression** | [depth_loader.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/depth_loader.py) L60-63 | Each SANPO depth file is individually `gzip.open → np.frombuffer` — Python GIL-bound |
| **GPU↔CPU round-trips** | [yolo_tracker.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/yolo_tracker.py) L60-63 | Every frame: 4× `.cpu().numpy()` calls, discarding GPU tensors immediately |
| **Python-level physics loops** | [pipeline.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/pipeline.py) L134-173 | `for det in detections:` loop doing per-object [compute_bearing](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/physics.py#28-41), [compute_velocity](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/physics.py#43-63), `round()` |
| **No prefetch/streaming** | [pipeline.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/pipeline.py) L87-90 | All frames listed and iterated sequentially — no look-ahead or batch processing |
| **Duplicate severity dicts** | [physics.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/physics.py) vs [shared/fact_sheet.py](file:///e:/capstone/CompositePerceptionEngine/src/shared/fact_sheet.py) | Two divergent `CLASS_SEVERITY` dicts with different values and class names |
| **No progress bar** | [pipeline.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/pipeline.py) L176-178 | `tqdm` is in requirements.txt but unused; only prints every 50 frames |

---

## Proposed Changes

### Perception Stack Core

#### [MODIFY] [pipeline.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/pipeline.py)

**Major rewrite** — the orchestrator becomes a streaming generator with threaded I/O prefetch:

1. **Streaming generator**: [run_perception()](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/pipeline.py#68-181) becomes `run_perception_stream()` — a Python generator that `yield`s row dicts per frame instead of accumulating a giant `rows: list[dict]`. The old [run_perception()](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/pipeline.py#68-181) stays as a thin wrapper for backward compat.
2. **Threaded I/O prefetch**: A `_FramePrefetcher` class using `threading.Thread` + `queue.Queue` pre-reads the next N RGB + depth frames while the current frame is being processed by YOLO. This overlaps disk I/O with GPU compute.
3. **Tensor-based physics**: After YOLO returns GPU tensors (from the refactored tracker), bearing/depth/velocity are computed in batch using torch ops on the GPU — no Python for-loop per detection.
4. **tqdm progress bar**: Replace the `if frame_idx % 50` print with `tqdm`.

---

#### [MODIFY] [yolo_tracker.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/yolo_tracker.py)

**Keep tensors on GPU** until physics is done:

1. [track()](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/yolo_tracker.py#42-77) returns a `TrackedDetections` dataclass containing raw torch tensors (`boxes`, `ids`, `classes`, `confs`, `cxs`) still on device.
2. A new `.to_dicts()` method on `TrackedDetections` converts to the old `list[dict]` format for backward compatibility.
3. The `.cpu().numpy()` calls move to `.to_dicts()` — only called at the end when writing CSV, not during the hot physics loop.

---

#### [MODIFY] [depth_loader.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/depth_loader.py)

**Tensor-aware depth loading**:

1. New `load_depth_tensor()` — returns `torch.Tensor` on GPU instead of numpy array.
2. New `batch_median_depth()` — takes a depth tensor + N bounding boxes (as a [(N, 4)](file:///e:/capstone/CompositePerceptionEngine/tools/run_perception.py#37-73) tensor) and computes all median depths in one vectorised operation using `torch` indexing.
3. Original [load_depth_map()](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/depth_loader.py#41-76) and [median_depth_in_box()](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/depth_loader.py#78-98) stay unchanged for backward compat.

---

#### [MODIFY] [physics.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/physics.py)

**Vectorised batch physics**:

1. New `batch_compute_bearing(cx_tensor, frame_width)` — operates on a 1D tensor of all centre-x values at once.
2. New `batch_kinetic_score(distances, velocities, severities)` — fully vectorised kinetic score for the entire frame's detections.
3. Original scalar functions stay for backward compat.

---

#### [MODIFY] [csv_writer.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/csv_writer.py)

**Streaming-compatible writer**:

1. New `StreamingCSVWriter` context manager — opens the file once, writes header, then accepts rows streamed from the generator via `.write_rows(rows)`.
2. Original [write_csv()](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/csv_writer.py#32-45) stays unchanged.

---

#### [MODIFY] [__init__.py](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/__init__.py)

Export the new streaming APIs alongside the existing ones.

---

### Shared Module

#### [MODIFY] [fact_sheet.py](file:///e:/capstone/CompositePerceptionEngine/src/shared/fact_sheet.py)

**Deduplicate severity**: Replace the local `CLASS_SEVERITY` dict with an import from `src.perception_stack.physics.CLASS_SEVERITY` to eliminate the divergence.

---

### Tools

#### [MODIFY] [run_perception.py](file:///e:/capstone/CompositePerceptionEngine/tools/run_perception.py)

Add `--batch_size` and `--prefetch` CLI args. Use the streaming API with `StreamingCSVWriter`.

---

## User Review Required

> [!IMPORTANT]
> The streaming generator changes the return type of the pipeline from `list[dict]` to a generator. The old [run_perception()](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/pipeline.py#68-181) function is preserved as a wrapper that materialises the generator into a list, so **existing code won't break**. But if you have any other scripts importing [run_perception](file:///e:/capstone/CompositePerceptionEngine/src/perception_stack/pipeline.py#68-181) directly, let me know.

> [!WARNING]
> The tensor-based flow requires CUDA. If you sometimes run on CPU-only machines, the code will auto-detect and fall back to the old numpy path. Confirm this is acceptable.

---

## Verification Plan

### Automated Tests

Since the `tests/unit/` and `tests/integration/` directories are currently empty, I will create a focused integration test:

#### New test: `tests/integration/test_pipeline_streaming.py`
- Generates 10 synthetic RGB frames (solid colours with cv2) + matching dummy depth maps.
- Runs both `run_perception()` (old materialised list) and `run_perception_stream()` (new generator) on the same input.
- Asserts identical CSV output field-by-field.
- Run command: `python -m pytest tests/integration/test_pipeline_streaming.py -v`

### Manual Verification

1. **Run the pipeline on your existing SANPO sample data**:
   ```
   python tools/run_perception.py --rgb_dir data/sanpo/sample/rgb --depth_dir data/sanpo/sample/depth --out data/processed/sanpo_test_NEW.csv --source sanpo --fps 30
   ```
2. **Compare old vs new output** — diff the CSV files to confirm identical results (within floating-point tolerance).
3. **Check streaming performance** — observe tqdm progress bar and note the throughput (frames/sec) compared to before.
