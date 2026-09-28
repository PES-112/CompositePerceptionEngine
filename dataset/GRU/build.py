"""
build.py
========
Build the GRU dataset once from Stage-1 per-session CSVs. Both GRU
architectures then train from the same files, so they see identical samples
and the identical session split.

    python -m dataset.GRU.build --csv-dir data/processed/ablation_30pct

Writes dataset/GRU/<name>/:
    train.npz, val.npz, test.npz   per-frame samples (see samples.py for the format)
    manifest.json                  build settings, source fingerprint, session split, label stats
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dataset.GRU.samples import (
    CLASS_VOCAB,
    DEFAULT_HISTORY,
    DEFAULT_MAX_OBJECTS,
    FEATURES,
    SPLITS,
    csv_fingerprint,
    load_corpus,
    save_split,
    split_sessions,
)
from evaluation.kinetic_ablation import D_HAZ_M, HORIZON_FRAMES, THETA_DEG

HERE = Path(__file__).resolve().parent
GITHUB_WARN_MB = 50   # GitHub warns above 50 MB per file and rejects above 100 MB


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-dir", type=Path, required=True, help="Per-session Stage-1 CSVs (<session_id>.csv).")
    p.add_argument("--name", default=None, help="Defaults to <csv-dir name>_h<history>_<label>_m<max>_s<split-seed>.")
    p.add_argument("--out-root", type=Path, default=HERE)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--history", type=int, default=DEFAULT_HISTORY)
    p.add_argument("--max-objects", type=int, default=DEFAULT_MAX_OBJECTS)
    p.add_argument("--future-only", action="store_true",
                   help="Label only on frames after T (default matches the K0 ablation, which includes T).")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    label = "future" if args.future_only else "current"
    name = args.name or f"{args.csv_dir.name}_h{args.history}_{label}_m{args.max_objects}_s{args.split_seed}"
    out = args.out_root / name
    if (out / "manifest.json").exists() and not args.overwrite:
        raise SystemExit(f"{out} already exists (pass --overwrite or a different --name)")

    csvs = sorted(args.csv_dir.glob("*.csv"))
    corpus = load_corpus(args.csv_dir, history=args.history, future_only=args.future_only,
                         max_objects=args.max_objects)
    split = split_sessions(list(corpus), seed=args.split_seed, val_frac=args.val_frac, test_frac=args.test_frac)

    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for part in SPLITS:
        samples = [s for sid in split[part] for s in corpus[sid]]
        save_split(samples, out / f"{part}.npz")
        y = np.concatenate([s["y"] for s in samples])
        stats[part] = {"sessions": len(split[part]), "frames": len(samples), "objects": int(len(y)),
                       "positive_rate": round(float(y.mean()), 4),
                       "frames_with_encounter": sum(s["frame_has_encounter"] for s in samples)}

    manifest = {
        "name": name,
        "created": time.strftime("%Y-%m-%d"),
        "source": {"csv_dir": str(args.csv_dir), "csv_files": len(csvs), "fingerprint": csv_fingerprint(csvs),
                   "sessions_with_samples": len(corpus)},
        "params": {"history": args.history, "max_objects": args.max_objects, "future_only": args.future_only,
                   "split_seed": args.split_seed, "val_frac": args.val_frac, "test_frac": args.test_frac},
        "label": {"definition": "evaluation/kinetic_ablation.py encounters()"
                                + (" restricted to frames after T" if args.future_only else ""),
                  "D_HAZ_M": D_HAZ_M, "THETA_DEG": THETA_DEG, "HORIZON_FRAMES": HORIZON_FRAMES},
        "features": FEATURES,
        "class_vocab": CLASS_VOCAB,
        "stats": stats,
        "split": split,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for part in SPLITS:
        s = stats[part]
        print(f"{part:5s}: {s['sessions']} sessions, {s['frames']} frames, {s['objects']} objects, "
              f"encounter rate {s['positive_rate']:.3f}")
    for f in sorted(out.glob("*.npz")):
        mb = f.stat().st_size / 1e6
        warn = f"  <- over {GITHUB_WARN_MB} MB: don't commit this file to GitHub" if mb > GITHUB_WARN_MB else ""
        print(f"  {f.name}: {mb:.1f} MB{warn}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
