# Changelog — `src/perception_stack/csv_writer.py`

> Defines the canonical Stage 1 CSV schema and provides writers for both batch and streaming output modes.

---

## [v1.1] — 2026-03-30 | StreamingCSVWriter Added

**Session:** *Optimizing CPE Perception Pipeline Performance*

### What Changed

#### New: `StreamingCSVWriter` Context Manager

**Problem:** The original `write_csv(rows, out_path)` function required the entire perception output to be accumulated in a list before writing. For large SANPO sequences (10,000+ frames × multiple detections per frame), this list could consume several gigabytes of RAM unnecessarily.

**Fix:** Added `StreamingCSVWriter` — a context manager that opens the CSV file once on entry, writes batches of rows incrementally, and flushes/closes on exit. Designed to pair with `run_perception_stream()`.

```python
class StreamingCSVWriter:
    def __enter__(self):   # opens file, writes header
    def write_rows(self, rows: list[dict]) -> None:  # appends rows
    def __exit__(self, *exc):  # closes file handle
    # .rows_written   — cumulative count for final reporting
```

**Usage pattern:**
```python
with StreamingCSVWriter(out_path) as writer:
    for frame_rows in run_perception_stream(...):
        writer.write_rows(frame_rows)
print(f"{writer.rows_written} rows written")
```

### Backward Compatibility
- ✅ `write_csv()` and `CSV_FIELDS` are unchanged.

---

## [v1.0] — 2026-03-25 | Initial Implementation

**Session:** *Implementing Argus-REI Perception Pipeline*

### What Was Built

- **`CSV_FIELDS`** — canonical list of column names that forms the contract between Stage 1 (pipeline) and Stage 2 (fact_sheet_builder). Changing this list is a breaking schema change. Fields:
  - `frame_idx` — source frame number
  - `source` — `'sanpo'` or `'uasol'`
  - `track_id` — ByteTrack persistent ID
  - `class` — COCO class label
  - `confidence` — YOLO detection confidence `[0, 1]`
  - `bbox_x1/y1/x2/y2` — bounding box corners
  - `cx_px` — horizontal centre pixel
  - `bearing_deg` — left/right bearing in degrees
  - `distance_m` — median metric depth (empty if no depth map)
  - `velocity_ms` — estimated closing velocity m/s

- **`write_csv(rows, out_path)`** — writes list of row dicts to CSV using `csv.DictWriter`. Creates parent directories if missing.

---

## File Reference
- **Source:** [`src/perception_stack/csv_writer.py`](../../src/perception_stack/csv_writer.py)
- **Uses:** `csv`, `pathlib`
- **Used by:** `pipeline.py` (schema reference), `tools/run_perception.py` (StreamingCSVWriter), `fact_sheet_builder.py` (schema reference)
- **Exports (via `__init__.py`):** `CSV_FIELDS`, `write_csv`, `StreamingCSVWriter`
