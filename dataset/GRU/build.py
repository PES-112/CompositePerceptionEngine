"""
build.py
========
Build the GRU dataset once from Stage-1 per-session CSVs. Both GRU
architectures then train from the same files, so they see identical samples
and identical session splits.

    # fixed train/val/test split, labels identical to the K0 ablation
    python -m dataset.GRU.build --csv-dir data/processed/ablation_30pct
    # 5-fold cross-validation, tracked objects only, 3 m hazard zone
    python -m dataset.GRU.build --csv-dir data/processed/sanpo_real_462 --tracked-only --hazard-m 3 --folds 5

Writes dataset/GRU/<name>/:
    train.npz, val.npz, test.npz   (fixed split) or all.npz (with --folds)
    manifest.json                  build settings, source fingerprint, split or folds, label stats
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
    fold_sessions,
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
    p.add_argument("--name", default=None, help="Defaults to a name built from the csv-dir and the options below.")
    p.add_argument("--out-root", type=Path, default=HERE)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--history", type=int, default=DEFAULT_HISTORY)
    p.add_argument("--max-objects", type=int, default=DEFAULT_MAX_OBJECTS)
    p.add_argument("--hazard-m", type=float, default=D_HAZ_M,
                   help=f"Encounter distance (default {D_HAZ_M} m, matching the K0 ablation).")
    p.add_argument("--future-only", action="store_true",
                   help="Label only on frames after T (default matches the K0 ablation, which includes T).")
    p.add_argument("--tracked-only", action="store_true",
                   help="Drop the untracked obs_* depth blobs (mostly ground ahead; see samples.UNTRACKED_PREFIX). "
                        "Default keeps them, matching the K0 ablation.")
    p.add_argument("--folds", type=int, default=0,
                   help="Session-level cross-validation folds (0 = one fixed train/val/test split).")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.15, help="Fixed split only.")
    p.add_argument("--test-frac", type=float, default=0.15, help="Fixed split only.")
    return p.parse_args()


def _stats(samples: list[dict], n_sessions: int) -> dict:
    y = np.concatenate([s["y"] for s in samples])
    return {"sessions": n_sessions, "frames": len(samples), "objects": int(len(y)),
            "positive_rate": round(float(y.mean()), 4),
            "sessions_with_encounter": len({s["session"] for s in samples if s["y"].any()}),
            "frames_with_encounter": sum(s["frame_has_encounter"] for s in samples)}


def default_name(args: argparse.Namespace) -> str:
    parts = [args.csv_dir.name]
    if args.tracked_only:
        parts.append("tracked")
    if args.hazard_m != D_HAZ_M:
        parts.append(f"d{args.hazard_m:g}")
    parts += [f"h{args.history}", "future" if args.future_only else "current", f"m{args.max_objects}"]
    if args.folds:
        parts.append(f"f{args.folds}")
    parts.append(f"s{args.split_seed}")
    return "_".join(parts)


def main() -> None:
    args = parse_args()
    name = args.name or default_name(args)
    out = args.out_root / name
    if (out / "manifest.json").exists() and not args.overwrite:
        raise SystemExit(f"{out} already exists (pass --overwrite or a different --name)")

    csvs = sorted(args.csv_dir.glob("*.csv"))
    corpus = load_corpus(args.csv_dir, history=args.history, future_only=args.future_only,
                         max_objects=args.max_objects, tracked_only=args.tracked_only, hazard_m=args.hazard_m)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": name,
        "created": time.strftime("%Y-%m-%d"),
        "source": {"csv_dir": str(args.csv_dir), "csv_files": len(csvs), "fingerprint": csv_fingerprint(csvs),
                   "sessions_with_samples": len(corpus)},
        "params": {"history": args.history, "max_objects": args.max_objects, "hazard_m": args.hazard_m,
                   "future_only": args.future_only, "tracked_only": args.tracked_only, "folds": args.folds,
                   "split_seed": args.split_seed, "val_frac": args.val_frac, "test_frac": args.test_frac},
        "label": {"definition": f"within {args.hazard_m:g} m inside the ±{THETA_DEG:g}° cone within "
                                f"{HORIZON_FRAMES} frames (kinetic_ablation.encounters() at 1.5 m)"
                                + ("; frames after T only" if args.future_only else "")
                                + ("; tracked objects only (obs_* depth blobs dropped)" if args.tracked_only else ""),
                  "hazard_m": args.hazard_m, "THETA_DEG": THETA_DEG, "HORIZON_FRAMES": HORIZON_FRAMES},
        "features": FEATURES,
        "class_vocab": CLASS_VOCAB,
    }

    if args.folds:
        folds = fold_sessions(corpus, args.folds, seed=args.split_seed)
        save_split([s for sid in sorted(corpus) for s in corpus[sid]], out / "all.npz")
        manifest["stats"] = {f"fold{i}": _stats([s for sid in f for s in corpus[sid]], len(f))
                             for i, f in enumerate(folds)}
        manifest["folds"] = folds
    else:
        split = split_sessions(list(corpus), seed=args.split_seed, val_frac=args.val_frac, test_frac=args.test_frac)
        manifest["stats"] = {}
        for part in SPLITS:
            samples = [s for sid in split[part] for s in corpus[sid]]
            save_split(samples, out / f"{part}.npz")
            manifest["stats"][part] = _stats(samples, len(split[part]))
        manifest["split"] = split
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for part, s in manifest["stats"].items():
        print(f"{part:5s}: {s['sessions']} sessions ({s['sessions_with_encounter']} with an encounter), "
              f"{s['frames']} frames, {s['objects']} objects, encounter rate {s['positive_rate']:.3f}")
    for f in sorted(out.glob("*.npz")):
        mb = f.stat().st_size / 1e6
        warn = f"  <- over {GITHUB_WARN_MB} MB: don't commit this file to GitHub" if mb > GITHUB_WARN_MB else ""
        print(f"  {f.name}: {mb:.1f} MB{warn}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
