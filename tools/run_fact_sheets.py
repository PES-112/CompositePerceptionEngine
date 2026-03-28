"""
run_fact_sheets.py
==================
CLI entry point for Stage 2: CSV → Kinetic Score + JSONL Fact Sheets.

This is a thin wrapper around src.perception_stack.fact_sheet_builder.
All logic lives in the perception_stack package; this script only handles
argument parsing and I/O paths.

Usage
-----
python tools/run_fact_sheets.py \\
    --csv         data/processed/merged_perception.csv \\
    --out         data/training/train.jsonl \\
    --fps         30 \\
    --lookahead_s 2.0
"""

import argparse
from pathlib import Path

from src.perception_stack import load_perception_csv, build_fact_sheets


def main():
    parser = argparse.ArgumentParser(
        description="CPE Phase 1 — Stage 2: CSV → Kinetic Score + JSONL Fact Sheets"
    )
    parser.add_argument("--csv",          required=True,
                        help="Perception CSV from Stage 1 (run_perception.py output)")
    parser.add_argument("--out",          required=True,
                        help="Output JSONL for SLM-1 SFT (e.g. data/training/train.jsonl)")
    parser.add_argument("--fps",          type=float, default=30.0,
                        help="Framerate used in Stage 1 — needed for look-ahead window")
    parser.add_argument("--lookahead_s",  type=float, default=2.0,
                        help="Seconds ahead for K₊₂ future threat confirmation (default: 2.0)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== CPE Stage 2: CSV → Kinetic Score + JSONL ===")
    print(f"  Input CSV   : {csv_path}")
    print(f"  Output JSONL: {out_path}")
    print(f"  FPS         : {args.fps}")
    print(f"  Look-ahead  : {args.lookahead_s}s ({int(args.fps * args.lookahead_s)} frames)\n")

    frames = load_perception_csv(csv_path)
    written, skipped = build_fact_sheets(frames, args.fps, args.lookahead_s, out_path)

    print(f"\n✅  Stage 2 done — {written} JSONL records written, {skipped} frames skipped")
    print(f"    Upload {out_path} to Colab → run SFTTrainer + LoRA on Qwen2.5-1.5B-Instruct")


if __name__ == "__main__":
    main()
