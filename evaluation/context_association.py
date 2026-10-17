"""
context_association.py
======================
Do the scene-context flags point the right way? For every (frame, tracked
object) with context (tools/build_scene_context.py), compare the arrival label
(the corridor label: in the walker's path within the horizon) across
in_path / near_boundary / other_surface and across surfaces.

The multipliers in src/threat_engine/priority.py are declared, not fitted
(in path ×1.5, near a boundary ×1.25, other surface ×0.5), so every session
with context can be used here, train included. Objects on the walker's
surface are also closer on average, so each flag's risk ratio is reported raw
and Mantel-Haenszel-adjusted for distance (strata DIST_BINS), with
session-bootstrap 95% CIs.

"Near a boundary" pointed the wrong way on the first run (an object at the
sidewalk's edge is usually beside the path, not entering it), so two derived
flags are reported too: *approaching* (closing on the walker's centre line at
>= APPROACH_MS, from motion alone: the control) and *near a boundary and
approaching*. The second is also adjusted over distance x approaching strata
("adds over motion"): does the boundary add anything once motion is known?
A revised rule is chosen with --splits train and checked once with
--splits val test.

    python -m evaluation.context_association --context-dir data/sanpo_scene_context/gt
    python -m evaluation.context_association --context-dir data/sanpo_scene_context/gt --splits train
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from dataset.GRU.samples import load_dataset
from src.segmentation.context import APPROACH_MS, APPROACH_STEPS, approach_speed
from src.segmentation.labels import CPE_CLASSES
from src.threat_engine.priority import CONTEXT_IN_PATH, CONTEXT_NEAR_BOUNDARY, CONTEXT_OTHER_SURFACE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_official_corridor_w0.75_d3_h20"
DIST_BINS = (0.0, 3.0, 6.0, 10.0, np.inf)
FLAGS = {"in_path": CONTEXT_IN_PATH, "near_boundary": CONTEXT_NEAR_BOUNDARY, "other_surface": CONTEXT_OTHER_SURFACE,
         "approaching": None, "near_boundary_approaching": None}
N_BOOT = 1000


def approach(x: np.ndarray) -> float:
    """approach_speed from one object's GRU history rows (present, distance/10, ..., bearing/35, ...)."""
    rows = x[-(APPROACH_STEPS + 1):]
    seen = rows[:, 0] > 0
    return approach_speed(np.where(seen, rows[:, 1] * 10, np.nan), np.where(seen, rows[:, 3] * 35, np.nan))


def observations(dataset: Path, context_dir: Path, parts: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Columns over every (frame, object) that has context: session, split, y, distance, surface, flags."""
    _, splits = load_dataset(dataset, splits=parts)
    cols = defaultdict(list)
    for part, samples in splits.items():
        by = defaultdict(list)
        for s in samples:
            by[s["session"]].append(s)
        for sid, ss in by.items():
            path = context_dir / f"{sid}.json"
            if not path.exists():
                continue
            frames = json.loads(path.read_text(encoding="utf-8"))["frames"]
            for s in ss:
                f = frames.get(str(s["frame_idx"]))
                if f is None:
                    continue
                for j, tid in enumerate(s["track_ids"]):
                    c = f["tracks"].get(tid)
                    if c is None:
                        continue
                    surface, in_path, near, other = c
                    cols["session"].append(sid)
                    cols["split"].append(part)
                    cols["y"].append(float(s["y"][j]))
                    cols["distance"].append(float(s["distance"][j]))
                    cols["surface"].append(-1 if surface is None else int(surface))
                    cols["in_path"].append(bool(in_path))
                    cols["near_boundary"].append(bool(near))
                    cols["other_surface"].append(bool(other))
                    cols["approach_ms"].append(approach(s["x"][j]))
    obs = {k: np.asarray(v) for k, v in cols.items()}
    obs["approaching"] = obs["approach_ms"] >= APPROACH_MS
    obs["near_boundary_approaching"] = obs["near_boundary"] & obs["approaching"]
    return obs


def risk_ratios(y: np.ndarray, flag: np.ndarray, stratum: np.ndarray) -> tuple[float, float]:
    """(raw RR, Mantel-Haenszel RR across strata) of y for flag vs not flag."""
    a, n1 = y[flag].sum(), flag.sum()
    c, n0 = y[~flag].sum(), (~flag).sum()
    raw = (a / n1) / (c / n0) if n1 and n0 and c else np.nan
    num = den = 0.0
    for s in np.unique(stratum):
        m = stratum == s
        f, yy = flag[m], y[m]
        n = m.sum()
        num += yy[f].sum() * (~f).sum() / n
        den += yy[~f].sum() * f.sum() / n
    return float(raw), float(num / den) if den else np.nan


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--context-dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--name", default=None)
    args = p.parse_args()

    obs = observations(args.dataset, args.context_dir, tuple(args.splits))
    stratum = np.digitize(obs["distance"], DIST_BINS[1:-1])
    stratum_motion = stratum * 2 + obs["approaching"]
    sessions = np.unique(obs["session"])
    index = {sid: np.nonzero(obs["session"] == sid)[0] for sid in sessions}
    rng = np.random.default_rng(0)
    boot_idx = [np.concatenate([index[sessions[j]] for j in rng.integers(0, len(sessions), len(sessions))])
                for _ in range(N_BOOT)]

    def ci(fn) -> tuple[float, float, float]:
        draws = np.array([fn(i) for i in boot_idx], dtype=float)
        return float(fn(np.arange(len(obs["y"])))), float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))

    y = obs["y"]
    flags = {}
    for name, declared in FLAGS.items():
        f = obs[name]
        flags[name] = {
            "declared_multiplier": declared, "share": float(f.mean()),
            "rate_flag": ci(lambda i: y[i][f[i]].mean() if f[i].any() else np.nan),
            "rate_other": ci(lambda i: y[i][~f[i]].mean() if (~f[i]).any() else np.nan),
            "rr_raw": ci(lambda i: risk_ratios(y[i], f[i], stratum[i])[0]),
            "rr_distance_adjusted": ci(lambda i: risk_ratios(y[i], f[i], stratum[i])[1]),
        }
    f = obs["near_boundary_approaching"]
    flags["near_boundary_approaching"]["rr_adds_over_motion"] = ci(
        lambda i: risk_ratios(y[i], f[i], stratum_motion[i])[1])
    surfaces = {}
    for code in sorted(np.unique(obs["surface"])):
        m = obs["surface"] == code
        surfaces[CPE_CLASSES.get(int(code), "none (no ground at its depth)")] = {
            "share": float(m.mean()), "n": int(m.sum()),
            "rate": ci(lambda i: y[i][m[i]].mean() if m[i].any() else np.nan)}

    suffix = "" if set(args.splits) == {"train", "val", "test"} else "_" + "_".join(args.splits)
    name = args.name or f"{args.context_dir.name}{suffix}_{time.strftime('%Y_%m_%d')}"
    out = PROJECT_ROOT / "evaluation/benchmarks/scene_context" / name
    out.mkdir(parents=True, exist_ok=True)
    result = {"context_dir": args.context_dir.as_posix(), "dataset": args.dataset.name,
              "observations": int(len(y)), "sessions": int(len(sessions)),
              "sessions_by_split": {s: int(len(np.unique(obs["session"][obs["split"] == s]))) for s in ("train", "val", "test")},
              "base_rate": float(y.mean()), "flags": flags, "surfaces": surfaces,
              "splits": args.splits, "approach_ms": APPROACH_MS, "approach_steps": APPROACH_STEPS,
              "distance_bins_m": [b for b in DIST_BINS if np.isfinite(b)]}
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def cell(v, fmt="{:.3f}"):
        return f"{fmt.format(v[0])} [{fmt.format(v[1])}, {fmt.format(v[2])}]"

    lines = [f"# Scene-context flags vs the arrival label - {name}", "",
             f"{len(y)} (frame, object) observations in {len(sessions)} sessions "
             f"({result['sessions_by_split']}); context from `{args.context_dir.as_posix()}`. Label: "
             f"in the walking corridor within the horizon (base rate {y.mean():.3f}). 95% CIs: session bootstrap. "
             "Risk ratio = P(arrives | flag) / P(arrives | not flag); adjusted = Mantel-Haenszel over distance bins "
             f"{result['distance_bins_m']} m.", "",
             "| Flag | declared × | share of objects | P(arrive), flag | P(arrive), not | RR | RR, distance-adjusted |",
             "|---|---:|---:|---|---|---|---|"]
    for k, v in flags.items():
        declared = "—" if v["declared_multiplier"] is None else f"{v['declared_multiplier']:g}"
        lines.append(f"| {k} | {declared} | {v['share']:.2f} | {cell(v['rate_flag'])} | "
                     f"{cell(v['rate_other'])} | {cell(v['rr_raw'], '{:.2f}')} | {cell(v['rr_distance_adjusted'], '{:.2f}')} |")
    lines += ["", f"Approaching: closing on the walker's centre line at >= {APPROACH_MS} m/s over the last "
              f"{APPROACH_STEPS} steps (motion only, no segmentation). Near a boundary and approaching, adjusted "
              "over distance x approaching strata (does the boundary add anything once motion is known?): "
              f"{cell(flags['near_boundary_approaching']['rr_adds_over_motion'], '{:.2f}')}."]
    lines += ["", "| Surface under the object | share | n | P(arrive) |", "|---|---:|---:|---|"]
    for k, v in surfaces.items():
        lines.append(f"| {k} | {v['share']:.2f} | {v['n']} | {cell(v['rate'])} |")
    lines += ["", "A declared multiplier above 1 is supported if its RR's CI is above 1, below 1 if below 1. The "
              "multipliers are not fitted to these numbers; fitting them would need train sessions only."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
