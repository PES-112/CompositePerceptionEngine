"""
run_perception.py
=================
CLI entry point for Stage 1: Perception → CSV.

This is a thin wrapper around src.perception_stack.pipeline and csv_writer.
All logic lives in the perception_stack package; this script only handles
argument parsing and I/O paths.

Usage
-----
# SANPO (primary, ~80% of data) — every 3rd frame
python tools/run_perception.py \\
    --rgb_dir   data/sanpo/sample/rgb \\
    --depth_dir data/sanpo/sample/depth \\
    --out       data/processed/sanpo_perception.csv \\
    --source    sanpo \\
    --fps       30 \\
    --frame_step 3

# UASOL (secondary, ~20% — stereo depth) — all frames
python tools/run_perception.py \\
    --rgb_dir   data/uasol/sample/rgb \\
    --depth_dir data/uasol/sample/depth \\
    --out       data/processed/uasol_perception.csv \\
    --source    uasol \\
    --fps       30
"""

import argparse
from pathlib import Path

from src.perception_stack import run_perception_stream, StreamingCSVWriter
from src.perception_stack.depth_loader import DEPTH_SCALES


def main():
    parser = argparse.ArgumentParser(
        description="CPE Phase 1 — Stage 1: Perception → CSV"
    )
    parser.add_argument("--rgb_dir",    required=True,
                        help="Folder of sorted RGB frames (JPG/PNG)")
    parser.add_argument("--depth_dir",  default=None,
                        help="Folder of depth maps "
                             "(SANPO: float16.gz | UASOL: 16-bit PNG). "
                             "Omit to run perception-only without depth.")
    parser.add_argument("--out",        required=True,
                        help="Output CSV path (e.g. data/processed/sanpo.csv)")
    parser.add_argument("--fps",        type=float, default=30.0,
                        help="Source video framerate for velocity calculation (default: 30)")
    parser.add_argument("--source",     choices=["sanpo", "uasol"], default="sanpo",
                        help="Dataset source — sets depth scale")
    parser.add_argument("--frame_step", type=int, default=1,
                        help="Process every Nth frame (default: 1 = all). "
                             "Set to 3 for SANPO preprocessing to cut compute ~67%%.")
    args = parser.parse_args()

    rgb_dir   = Path(args.rgb_dir)
    depth_dir = Path(args.depth_dir) if args.depth_dir else None
    out_path  = Path(args.out)

    print(f"\n=== CPE Stage 1: Perception → CSV ===")
    print(f"  Source     : {args.source.upper()}")
    print(f"  Depth scale: {DEPTH_SCALES[args.source]}")
    print(f"  RGB dir    : {rgb_dir}")
    print(f"  Depth dir  : {depth_dir or 'None — distance/velocity will be empty'}")
    print(f"  FPS        : {args.fps}")
    print(f"  Frame step : every {args.frame_step} frame(s)")
    print(f"  Output CSV : {out_path}\n")

    # ── Streaming pipeline — writes rows incrementally ──
    with StreamingCSVWriter(out_path) as writer:
        for frame_rows in run_perception_stream(
            rgb_dir, depth_dir, args.fps, args.source, args.frame_step
        ):
            writer.write_rows(frame_rows)

    print(f"\n✅  Stage 1 done — {writer.rows_written} detections → {out_path}")
    print(f"    Inspect the CSV, then run: python tools/run_fact_sheets.py --csv {out_path}")


if __name__ == "__main__":
    main()
