# Changelog — `tools/run_perception.py`

> CLI entry point for Stage 1: RGB frames + depth maps → perception CSV. Thin wrapper around `src.perception_stack`.

---

## [v1.1] — 2026-03-30 | Streaming API + Frame Step

**Session:** *Optimizing CPE Perception Pipeline Performance*

### What Changed

#### New `--frame_step` Argument

Added `--frame_step` CLI argument (default `1`, meaning all frames). Setting to `3` processes every 3rd frame, reducing compute by ~67% for SANPO preprocessing where adjacent frames are nearly identical.

```bash
# SANPO — recommended production setting
python tools/run_perception.py \
    --rgb_dir data/sanpo/sample/rgb \
    --depth_dir data/sanpo/sample/depth \
    --out data/processed/sanpo_perception.csv \
    --source sanpo --fps 30 --frame_step 3
```

#### Switched to `StreamingCSVWriter`

**Before:** Called `run_perception()` (returns full list in RAM), then `write_csv(rows, out_path)`.

**After:** Uses the new streaming API — pairs `run_perception_stream()` with `StreamingCSVWriter` context manager. Rows are written to disk incrementally per frame, breaking the RAM bottleneck.

```python
# After
with StreamingCSVWriter(out_path) as writer:
    for frame_rows in run_perception_stream(
        rgb_dir, depth_dir, args.fps, args.source, args.frame_step
    ):
        writer.write_rows(frame_rows)
print(f"✅  {writer.rows_written} detections → {out_path}")
```

#### Updated Imports

```python
# Before
from src.perception_stack import run_perception
from src.perception_stack.csv_writer import write_csv

# After
from src.perception_stack import run_perception_stream, StreamingCSVWriter
from src.perception_stack.depth_loader import DEPTH_SCALES
```

Added `DEPTH_SCALES` import so the startup banner can show the depth scale factor for the selected source.

---

## [v1.0] — 2026-03-25 | Initial Implementation

**Session:** *Implementing Argus-REI Perception Pipeline*

### What Was Built

- `argparse` CLI with `--rgb_dir`, `--depth_dir`, `--out`, `--fps`, `--source` args
- Startup banner logging all parameters
- Called `run_perception()` → `write_csv()` → print final count
- Supported `sanpo` and `uasol` source choices

---

## File Reference
- **Source:** [`tools/run_perception.py`](../../tools/run_perception.py)
- **Uses:** `src.perception_stack.run_perception_stream`, `src.perception_stack.StreamingCSVWriter`
- **Output:** CSV file at `--out` path
- **Next step after running:** `python tools/run_fact_sheets.py --csv <output>`
