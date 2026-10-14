"""
resplit.py
==========
Re-split an existing cross-validation GRU dataset (all.npz) onto SANPO's
official session-level train/test split, without the Stage-1 CSVs.

    python -m dataset.GRU.resplit --dataset dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_f5_s0
    # -> dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_official/

Test = the official test sessions (simulation/datasets/sanpo/official_split.json,
written by tools/audit_sanpo_sessions.py). Validation is carved out of the
official train sessions only — dealt by encounter-track count like
fold_sessions(), so it gets its share of the rare sessions with arrivals — and
the rest is train. Test sessions are never used for training, early stopping,
or fitting an ensemble, so numbers on it are a true held-out estimate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dataset.GRU.samples import SPLITS, fold_sessions, load_split, save_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = PROJECT_ROOT / "simulation/datasets/sanpo/official_split.json"


def official_split(by_session: dict[str, list[dict]], split_file: Path, val_frac: float, seed: int) -> dict:
    official = json.loads(split_file.read_text(encoding="utf-8"))
    test = sorted(sid for sid in official["test"] if sid in by_session)
    train_pool = {sid: by_session[sid] for sid in official["train"] if sid in by_session}
    # One "fold" of a k-fold deal over the train pool is a stratified validation set.
    k = max(2, round(1 / val_frac))
    val = sorted(fold_sessions(train_pool, k, seed=seed)[0])
    train = sorted(sid for sid in train_pool if sid not in set(val))
    return {"train": train, "val": val, "test": test}


def _stats(samples: list[dict], n_sessions: int) -> dict:
    import numpy as np
    y = np.concatenate([s["y"] for s in samples])
    return {"sessions": n_sessions, "frames": len(samples), "objects": int(len(y)),
            "positive_rate": round(float(y.mean()), 4),
            "sessions_with_encounter": len({s["session"] for s in samples if s["y"].any()}),
            "frames_with_encounter": sum(s["frame_has_encounter"] for s in samples)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True, help="A cross-validation dataset folder (has all.npz).")
    p.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--val-frac", type=float, default=0.15, help="Share of the official train sessions held for validation.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    src = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    if "folds" not in src:
        raise SystemExit(f"{args.dataset} is not a cross-validation dataset (no all.npz)")
    name = args.name or src["name"].rsplit("_f", 1)[0] + "_official"
    out = args.dataset.parent / name
    if (out / "manifest.json").exists() and not args.overwrite:
        raise SystemExit(f"{out} already exists (pass --overwrite)")

    by_session: dict[str, list[dict]] = {}
    for s in load_split(args.dataset / "all.npz"):
        by_session.setdefault(s["session"], []).append(s)
    split = official_split(by_session, args.split_file, args.val_frac, args.seed)

    out.mkdir(parents=True, exist_ok=True)
    manifest = {k: v for k, v in src.items() if k not in ("folds", "stats", "name", "created")}
    manifest.update(name=name, created=time.strftime("%Y-%m-%d"),
                    resplit_from=src["name"], split_source=str(args.split_file.relative_to(PROJECT_ROOT))
                    if args.split_file.is_relative_to(PROJECT_ROOT) else str(args.split_file),
                    params={**src["params"], "folds": 0, "split": "official", "val_frac": args.val_frac,
                            "split_seed": args.seed},
                    stats={}, split=split)
    for part in SPLITS:
        samples = [s for sid in split[part] for s in by_session[sid]]
        save_split(samples, out / f"{part}.npz")
        manifest["stats"][part] = _stats(samples, len(split[part]))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for part, s in manifest["stats"].items():
        print(f"{part:5s}: {s['sessions']} sessions ({s['sessions_with_encounter']} with an arrival), "
              f"{s['frames']} frames, {s['objects']} objects, positive rate {s['positive_rate']:.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
