# Changelog — `tools/run_fact_sheets.py`

> CLI entry point for Stage 2: perception CSV → kinetic-scored JSONL fact sheets for SLM-1 supervised fine-tuning.

---

## [v1.0] — 2026-03-25 | Initial Implementation

**Session:** *Implementing Argus-REI Perception Pipeline*

### What Was Built

- `argparse` CLI with: `--csv`, `--out`, `--fps`, `--lookahead_s`
- Startup banner logging all parameters including computed look-ahead frames count
- Calls `load_perception_csv(csv_path)` → `build_fact_sheets(frames, fps, lookahead_s, out_path)`
- Reports written vs skipped JSONL record counts on completion
- Outputs next-step instruction: upload JSONL to Colab for SFTTrainer + LoRA

### CLI Usage

```bash
python tools/run_fact_sheets.py \
    --csv         data/processed/merged_perception.csv \
    --out         data/training/train.jsonl \
    --fps         30 \
    --lookahead_s 2.0
```

### Design Notes

- This is intentionally a **thin wrapper** — all logic lives in `fact_sheet_builder.py`. The tool only handles argument parsing and path setup.
- `--lookahead_s` defaults to `2.0` seconds. This is the future look-ahead window for K₊₂ kinetic grounding. At 30fps this is 60 frames ahead.
- Parent directories for `--out` are created automatically if missing.

---

## File Reference
- **Source:** [`tools/run_fact_sheets.py`](../../tools/run_fact_sheets.py)
- **Uses:** `src.perception_stack.load_perception_csv`, `src.perception_stack.build_fact_sheets`
- **Input:** Stage 1 CSV from `run_perception.py`
- **Output:** `train.jsonl` for SLM-1 SFT (HuggingFace SFTTrainer compatible)
- **Next step after running:** Upload JSONL to Google Colab → run SFTTrainer + LoRA on Qwen2.5-1.5B-Instruct
