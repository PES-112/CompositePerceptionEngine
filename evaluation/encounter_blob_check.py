"""
encounter_blob_check.py
=======================
How much of the K0 ablation's encounter_top1 comes from the untracked obs_*
depth blobs?

src/perception_stack/pipeline.py (detect_unlabeled_obstacles) emits one
`unlabeled_obstacle` per fixed column of the lower 60% of the frame — the
closest depth pixel YOLO didn't claim, with no ground removal — re-IDed every
frame as obs_<frame>_<cx>. This recomputes encounter_top1 with the ablation's
own functions (same labels, same frame skip, frame-weighted, session
bootstrap) for K0, no-velocity and TTC, once on all objects and once with the
blobs removed.

    python evaluation/encounter_blob_check.py --csv-dir data/processed/ablation_30pct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.kinetic_ablation import VARIANTS, bootstrap_ci, encounters, frame_groups, prepare  # noqa: E402

ARMS = ["K0  sev·v²/d", "no-velocity  sev/d", "ttc  -(d-D)/v"]
UNTRACKED_PREFIX = "obs_"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-dir", type=Path, default=PROJECT_ROOT / "data/processed/ablation_30pct")
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    csvs = sorted(args.csv_dir.glob("*.csv"))
    results = {mode: {arm: ([], []) for arm in ARMS} for mode in ("all", "tracked")}
    blob_share = []
    for path in csvs:
        raw = pd.read_csv(path)
        for mode in ("all", "tracked"):
            df = raw if mode == "all" else raw[~raw["track_id"].astype(str).str.startswith(UNTRACKED_PREFIX)]
            df = prepare(df)
            if df.empty:
                continue
            enc = encounters(df)
            if mode == "all":
                blob_share.extend(t.startswith(UNTRACKED_PREFIX) for tids in enc.values() for t in tids)
            groups = frame_groups(df)
            for arm in ARMS:
                hits = []
                for frame_idx, g in groups:
                    if frame_idx not in enc:
                        continue
                    s = VARIANTS[arm](g)
                    if not np.isfinite(s).any():   # same skip as kinetic_ablation.session_metrics
                        continue
                    hits.append(g.iloc[int(np.argsort(-s)[0])]["track_id"] in enc[frame_idx])
                values, weights = results[mode][arm]
                values.append(float(np.mean(hits)) if hits else float("nan"))
                weights.append(len(groups))

    print(f"Sessions: {len(csvs)} · scored frames (>= 2 objects): {sum(results['all'][ARMS[0]][1])}")
    print(f"Share of (frame, encounter track) pairs that are {UNTRACKED_PREFIX}* blobs: {np.mean(blob_share):.3f}")
    for mode in ("all", "tracked"):
        print(f"\nencounter_top1, {mode} objects (95% session-bootstrap CI):")
        for arm in ARMS:
            pt, lo, hi = bootstrap_ci(*results[mode][arm], n_boot=args.n_boot, seed=0)
            print(f"  {arm:22s} {pt:.3f} [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
