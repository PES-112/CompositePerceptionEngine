"""
run_perception.py
=================
CLI entry point for Stage 1: Perception → CSV.

This is a thin wrapper around src.perception_stack.pipeline.run_perception
and src.perception_stack.csv_writer.write_csv.
All logic lives in the perception_stack package; this script only handles
argument parsing and I/O paths.

Usage
-----
# SANPO (primary, ~80% of data)
python tools/run_perception.py \\
    --rgb_dir   data/sanpo/sample/rgb \\
    --depth_dir data/sanpo/sample/depth \\
    --out       data/processed/sanpo_perception.csv \\
    --source    sanpo \\
    --fps       30

# UASOL (secondary, ~20% — stereo depth)
python tools/run_perception.py \\
    --rgb_dir   data/uasol/sample/rgb \\
    --depth_dir data/uasol/sample/depth \\
    --out       data/processed/uasol_perception.csv \\
    --source    uasol \\
    --fps       30
"""

import argparse
from pathlib import Path

from src.perception_stack import run_perception, write_csv
from src.perception_stack.depth_loader import DEPTH_SCALES


def main():
    parser = argparse.ArgumentParser(
        description="CPE Phase 1 — Stage 1: Perception → CSV"
    )
    parser.add_argument("--rgb_dir",    required=True,
                        help="Folder of sorted RGB frames (JPG/PNG)")
    parser.add_argument("--depth_dir",  default=None,
                        help="Folder of 16-bit depth PNGs "
                             "(SANPO: lidar GT | UASOL: stereo GT). "
                             "Omit to run perception-only without depth.")
    parser.add_argument("--out",        required=True,
                        help="Output CSV path (e.g. data/processed/sanpo.csv)")
    parser.add_argument("--fps",        type=float, default=30.0,
                        help="Source video framerate for velocity calculation (default: 30)")
    parser.add_argument("--source",     choices=["sanpo", "uasol"], default="sanpo",
                        help="Dataset source — sets depth scale "
                             "(sanpo: 0.001 mm→m | uasol: 0.256 stereo)")
    args = parser.parse_args()

    rgb_dir   = Path(args.rgb_dir)
    depth_dir = Path(args.depth_dir) if args.depth_dir else None
    out_path  = Path(args.out)

    print(f"\n=== CPE Stage 1: Perception → CSV ===")
    print(f"  Source     : {args.source.upper()}")
    print(f"  Depth scale: {DEPTH_SCALES[args.source]} (raw uint16 × scale = metres)")
    print(f"  RGB dir    : {rgb_dir}")
    print(f"  Depth dir  : {depth_dir or 'None — distance/velocity will be empty'}")
    print(f"  FPS        : {args.fps}")
    print(f"  Output CSV : {out_path}\n")

    rows = run_perception(rgb_dir, depth_dir, args.fps, args.source)
    write_csv(rows, out_path)

    print(f"\n✅  Stage 1 done — {len(rows)} detections → {out_path}")
    print(f"    Inspect the CSV, then run: python tools/run_fact_sheets.py --csv {out_path}")


if __name__ == "__main__":
    main()
