"""
summarize.py
============
One markdown table per slice for a training run name: each architecture's
mean ± std across seeds, next to the baselines.

Fixed-split runs are averaged from their metrics.json. Cross-validation runs
are pooled first: each seed's test predictions from every fold are joined, so
every session is scored exactly once, and the metrics (with session-bootstrap
CIs) are computed on that pooled set before averaging over seeds. Baselines are
scored on the same pooled sessions.

The ± across seeds understates uncertainty — the real noise is which sessions
you happened to test on. So cross-validation summaries also report paired
differences (model minus each baseline, averaged over seeds) with a 95% CI
from resampling whole sessions, the same sessions for both sides of each draw.

    python -m models.GRU.summarize --run-name tracked_2026_09_28
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from dataset.GRU.samples import load_dataset
from evaluation.kinetic_ablation import HORIZON_FRAMES
from models.GRU.evaluate import arrival_pairs, auroc, baseline_scores, evaluate_scores, model_scores
from models.GRU.registry import RUN_ROOTS

N_BOOT = 1000

METRICS = ("auroc", "average_precision", "encounter_top1", "soonest_top1", "arrival_order", "flicker_rate")
HEADER = ("| Method | AUROC | Avg. precision | encounter_top1 | soonest_top1 | arrival_order | flicker_rate |\n"
          "|---|---:|---:|---:|---:|---:|---:|")


def _point(m: dict, key: str) -> float:
    v = m.get(key)
    if v is None:
        return float("nan")
    return v[0] if isinstance(v, list | tuple) else v


def pooled_scores(run_dirs: list[Path]) -> tuple[list[dict], list[np.ndarray], list[np.ndarray]]:
    """(samples, arrive scores, soon scores) for one seed, joined across its cross-validation folds."""
    configs = [json.loads((d / "config.json").read_text(encoding="utf-8")) for d in run_dirs]
    samples, arrive, soon = [], [], []
    for d, cfg in sorted(zip(run_dirs, configs), key=lambda dc: dc[1]["fold"]):
        _, split = load_dataset(Path(cfg["dataset"]), splits=("test",), fold=cfg["fold"])
        test = split["test"]
        flat = np.load(d / "test_outputs.npy")
        bounds = np.cumsum([0] + [len(s["y"]) for s in test])
        outputs = [flat[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
        a, n = model_scores(outputs, cfg.get("time_bins", 0), cfg.get("horizon", HORIZON_FRAMES))
        samples += test
        arrive += a
        soon += n
    return samples, arrive, soon


def paired_differences(samples: list[dict], seeds: list[tuple[list, list]],
                       others: dict[str, list[tuple[list, list]]] | None = None, seed: int = 0) -> dict:
    """
    {other: {metric: (mean, lo, hi)}} for model minus each baseline (and minus
    each model in `others`, e.g. the other architecture), averaged over seeds,
    95% CI from resampling whole sessions (paired: one draw, both sides).
    AUROC uses the arrive score; arrival_order the soon score.
    """
    base = [baseline_scores(s) for s in samples]
    names = list(base[0]) + list(others or {})
    by_session: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_session[s["session"]].append(i)
    sessions = sorted(by_session)

    def per_session(arrive_of, soon_of):
        """Per session: (labels, arrive scores) for AUROC and (concordant, pairs) for arrival_order."""
        out = []
        for sid in sessions:
            idx = by_session[sid]
            y = np.concatenate([samples[i]["y"] for i in idx])
            a = np.concatenate([arrive_of(i) for i in idx]).astype(float)
            c = t = 0.0
            if "tte" in samples[0]:
                for i in idx:
                    if len(samples[i]["y"]) >= 2:
                        ci, ti = arrival_pairs(samples[i]["tte"], soon_of(i))
                        c, t = c + ci, t + ti
            out.append((y, a, c, t))
        return out

    def seed_parts(pairs):
        return [per_session(lambda i, a=a: a[i], lambda i, n=n: n[i]) for a, n in pairs]

    model_parts = seed_parts(seeds)
    other_parts = {nm: [per_session(lambda i, nm=nm: base[i][nm], lambda i, nm=nm: base[i][nm])]
                   for nm in base[0]}
    other_parts.update({nm: seed_parts(pairs) for nm, pairs in (others or {}).items()})

    def metrics(parts, pick):
        y = np.concatenate([parts[j][0] for j in pick])
        a = np.concatenate([parts[j][1] for j in pick])
        c, t = sum(parts[j][2] for j in pick), sum(parts[j][3] for j in pick)
        return auroc(a, y), (c / t if t else float("nan"))

    rng = np.random.default_rng(seed)
    everything = np.arange(len(sessions))
    draws = [everything] + [rng.integers(0, len(sessions), len(sessions)) for _ in range(N_BOOT)]
    diffs = {nm: [] for nm in names}
    for pick in draws:
        model = np.mean([metrics(p, pick) for p in model_parts], axis=0)
        for nm in names:
            diffs[nm].append(model - np.mean([metrics(p, pick) for p in other_parts[nm]], axis=0))
    out = {}
    for nm in names:
        d = np.array(diffs[nm])
        out[nm] = {key: (float(d[0, k]), float(np.nanpercentile(d[1:, k], 2.5)), float(np.nanpercentile(d[1:, k], 97.5)))
                   for k, key in enumerate(("auroc", "arrival_order"))}
    return out


def run_dirs(run_name: str) -> dict[str, list[Path]]:
    """{arch: run folders} named exactly <run_name>[_fold<F>]_seed<N>."""
    own = re.compile(re.escape(run_name) + r"_(fold\d+_)?seed\d+")
    found = {arch: sorted(p.parent for p in root.glob(f"{run_name}_*/metrics.json") if own.fullmatch(p.parent.name))
             for arch, root in RUN_ROOTS.items()}
    return {arch: dirs for arch, dirs in found.items() if dirs}


def pooled_runs(run_name: str) -> dict[str, list[tuple[list[dict], list, list]]]:
    """{arch: [(samples, arrive, soon) per seed]} for cross-validation runs, each joined across its folds."""
    out = {}
    for arch, dirs in run_dirs(run_name).items():
        folded = defaultdict(list)
        for d in dirs:
            if "_fold" in d.name:
                folded[d.name.rsplit("_seed", 1)[1]].append(d)
        if folded:
            out[arch] = [pooled_scores(ds) for _, ds in sorted(folded.items(), key=lambda kv: int(kv[0]))]
    return out


def collect(run_name: str) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """({arch: [metrics per seed]}, {arch: paired differences}) — the latter for cross-validation runs only."""
    found = {}
    pooled_by_arch = pooled_runs(run_name)
    for arch, dirs in run_dirs(run_name).items():
        if arch in pooled_by_arch:
            found[arch] = [evaluate_scores(s, (a, n)) for s, a, n in pooled_by_arch[arch]]
        else:
            found[arch] = [json.loads((d / "metrics.json").read_text(encoding="utf-8")) for d in dirs]

    paired = {}
    archs = list(pooled_by_arch)
    for k, arch in enumerate(archs):
        samples = pooled_by_arch[arch][0][0]
        # Each earlier architecture is also compared against, so the attention
        # model gets a paired "vs gru" row. Only valid on identical test frames.
        others = {}
        for other in archs[:k]:
            other_samples = pooled_by_arch[other][0][0]
            if [(s["session"], s["frame_idx"]) for s in other_samples] == [(s["session"], s["frame_idx"]) for s in samples]:
                others[other] = [(a, n) for _, a, n in pooled_by_arch[other]]
        paired[arch] = paired_differences(samples, [(a, n) for _, a, n in pooled_by_arch[arch]], others)
    return found, paired


def _ci(v: tuple) -> str:
    mean, lo, hi = v
    return f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]" if np.isfinite(mean) else "n/a"


def summarize(run_name: str) -> str:
    runs, paired = collect(run_name)
    if not runs:
        raise SystemExit(f"No runs named {run_name}_* under {list(RUN_ROOTS.values())}")
    baselines = next(iter(runs.values()))[0]["methods"]
    lines = []
    for slice_name in ("all", "interaction"):
        lines += [f"### {run_name} - {slice_name} frames", "", HEADER]
        for arch, ms in runs.items():
            cells = []
            for key in METRICS:
                xs = np.array([_point(m["methods"]["model"][slice_name], key) for m in ms], dtype=float)
                cells.append(f"{np.nanmean(xs):.3f} ± {np.nanstd(xs):.3f}" if np.isfinite(xs).any() else "n/a")
            lines.append(f"| {arch} ({len(ms)} seeds) | " + " | ".join(cells) + " |")
        for name, m in baselines.items():
            if name == "model":
                continue
            m = m[slice_name]
            lines.append(f"| {name} | " + " | ".join(
                f"{_point(m, k):.3f}" if np.isfinite(_point(m, k)) else "n/a" for k in METRICS) + " |")
        k0 = baselines["K0"][slice_name]
        lines += ["", f"Sessions {next(iter(runs.values()))[0]['sessions']}, objects {k0['objects']} "
                      f"(encounter rate {k0['positive_rate']:.3f}), frames with >= 2 objects {k0['frames']}, "
                      f"with an encounter {k0['encounter_frames']}", ""]
    if paired:
        lines += ["### Model minus baseline, all frames (mean over seeds, 95% paired session-bootstrap CI)", "",
                  "| Model | vs | AUROC difference | arrival_order difference |", "|---|---|---|---|"]
        for arch, diffs in paired.items():
            for name, d in diffs.items():
                lines.append(f"| {arch} | {name} | {_ci(d['auroc'])} | {_ci(d['arrival_order'])} |")
        lines += ["", "A difference is only credible if its CI excludes 0.", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True, help="The --run-name given to train.py (without _fold/_seed).")
    print(summarize(p.parse_args().run_name))


if __name__ == "__main__":
    main()
