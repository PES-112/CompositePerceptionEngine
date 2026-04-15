# Changelog — `src/perception_stack/fact_sheet_builder.py`

> Stage 2 of the CPE pipeline. Reads the perception CSV, computes present and future kinetic scores, and writes JSONL training data for SLM-1 supervised fine-tuning.

---

## [v1.1] — 2026-03-30 | Deduplication — CLASS_SEVERITY Import

**Session:** *Optimizing CPE Perception Pipeline Performance*

### What Changed

**Problem:** `fact_sheet_builder.py` previously maintained its own copy of the `CLASS_SEVERITY` dict. This created a risk of the severity weights diverging between the physics scoring used in the pipeline and the scoring used in Stage 2 fact sheets.

**Fix:** Removed the local copy. Now imports directly from `physics.py`:

```python
from src.perception_stack.physics import (
    kinetic_score, bearing_label,
    CLASS_SEVERITY, DEFAULT_SEVERITY,
)
```

`physics.py` is now the single source of truth for severity weights across both pipeline and training stages.

---

## [v1.0] — 2026-03-25 | Initial Implementation

**Session:** *Implementing Argus-REI Perception Pipeline*

### What Was Built

This is the Stage 2 processor that transforms raw per-frame CSV rows from Stage 1 into structured JSONL records for SLM-1 training.

#### `load_perception_csv(csv_path)` 
Reads the CSV and groups rows into a `dict[frame_idx → list[row_dict]]`. Casts numeric fields from string: `distance_m → float`, `velocity_ms → float`, `bearing_deg → float`. Empty strings become `None`.

#### `build_fact_sheets(frames, fps, lookahead_s, out_path)`
Main builder loop:

1. **K₀ (Present Kinetic Score):** For every object in a frame, compute `kinetic_score(distance_m, velocity_ms, class)`. Sort by K descending. Identify `present_threat` = highest-K scoring object with K > 0.

2. **K₊₂ (Future Look-Ahead):** Look `lookahead_s × fps` frames ahead. Compute kinetic scores for that future frame, find `future_threat_id`. If `present_threat.track_id == future_threat_id`, flag `future_confirmed = True`. This future match becomes a ground-truth alignment signal — the SLM-1 model is rewarded for picking objects that actually *become* high-threat in the near future.

3. **JSONL Record Assembly:** Each record has three fields:
   - `"system"` — the navigation AI persona prompt (constant)
   - `"user"` — `"[SCENARIO FACT SHEET] {rendered objects}"` 
   - `"assistant"` — JSON with `primary_threat`, `track_id`, `class`, `distance_m`, `velocity_ms`, `kinetic_score`, `future_confirmed`, `reason`

#### `_render_fact_sheet(objects)` 
Converts the scored object list to a pipe-separated human-readable string. Format per object:
```
Object_01: person, 3.2m, v=1.2m/s, ahead [K=0.452]
```

#### `_reasoning(obj, future_match)` 
Generates the natural language reasoning string in the assistant's response. Includes `"Confirmed as highest threat 2 seconds later."` when `future_match=True`.

#### System Prompt
Defines SLM-1's persona: pedestrian navigation AI for visually impaired users. Must respond with valid JSON only.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| K₊₂ future look-ahead | Teaches SLM-1 causal reasoning — not just reacting to current state, but anticipating which objects *will* become dangerous |
| JSONL format | Compatible with HuggingFace `SFTTrainer` directly — no format conversion needed for Qwen2.5/ Phi-3 fine-tuning |
| `future_confirmed` flag | Allows weighted loss during training — higher-weight examples where physics ground truth aligns with semantic pick |
| Skip frames with no threat | Frames where all K=0 (no valid depth + no movement) are skipped to reduce noise in training data |

---

## File Reference
- **Source:** [`src/perception_stack/fact_sheet_builder.py`](../../src/perception_stack/fact_sheet_builder.py)
- **Uses:** `csv`, `json`, `physics.py`
- **Used by:** `tools/run_fact_sheets.py`
- **Output format:** JSONL (one JSON object per line), each with `system` / `user` / `assistant` fields
- **Output destination:** `data/training/train.jsonl` (SLM-1 SFT input)
