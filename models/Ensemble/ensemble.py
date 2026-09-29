"""
ensemble.py
===========
Level 1 of models/Ensemble/README.md: combine the arrival predictors into one
"who arrives first" ranking, scored exactly like the GRU experiments.

Components, per object, on the same pooled cross-validation test frames:
  gru, gru_attention   saved out-of-fold predictions of a models/GRU run set
                       trained with --time-bins
  nearest_in_cone, TTC the baselines from models/GRU/evaluate.py

Combiners:
  ensemble_rank   the average of each component's percentile rank. Nothing is
                  fitted, and labels are never used.
  ensemble_stack  a softmax regression over the components (plus distance,
                  bearing and TTC features) predicting the arrival slice. It is
                  fitted out-of-fold: fold F's test objects are scored by a
                  stacker trained only on the other folds' objects.

Seeds are paired (GRU seed s with attention seed s) and results averaged over
seeds. Differences against every component and baseline use the paired
session-bootstrap from models/GRU/summarize.py.

    python -m models.Ensemble.ensemble --run-name arrival387_2026_09_29
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from dataset.GRU.samples import arrival_bins
from evaluation.kinetic_ablation import HORIZON_FRAMES, THETA_DEG
from models.GRU.evaluate import baseline_scores, evaluate_scores, model_scores
from models.GRU.summarize import HEADER, METRICS, _ci, _point, paired_differences, pooled_runs, run_dirs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = ("gru", "gru_attention")
TTC_CAP_S = 10.0
FEATURES = ["gru_arrive_logit", "gru_soon", "att_arrive_logit", "att_soon",
            "distance", "abs_bearing", "in_cone", "ttc_capped", "closing"]

Scores = list[np.ndarray]


def _split(flat: np.ndarray, samples: list[dict]) -> Scores:
    bounds = np.cumsum([0] + [len(s["y"]) for s in samples])
    return [flat[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def rank_average(samples: list[dict], components: list[Scores]) -> Scores:
    """Mean percentile rank across components (higher = more urgent). Ties share their average rank."""
    flat = [np.concatenate(c).astype(float) for c in components]
    ranks = np.mean([rankdata(f) / len(f) for f in flat], axis=0)
    return _split(ranks, samples)


def features(samples: list[dict], gru: tuple[Scores, Scores], att: tuple[Scores, Scores]) -> np.ndarray:
    """[objects, len(FEATURES)] stacking inputs, in the pooled sample order."""
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    cols = [logit(np.concatenate(gru[0])), np.concatenate(gru[1]),
            logit(np.concatenate(att[0])), np.concatenate(att[1])]
    d = np.concatenate([s["distance"] for s in samples])
    b = np.abs(np.concatenate([s["bearing"] for s in samples]))
    ttc = np.concatenate([s["ttc"] for s in samples])
    cols += [d, b, (b < THETA_DEG).astype(float), np.minimum(np.maximum(ttc, 0.0), TTC_CAP_S),
             np.isfinite(ttc).astype(float)]
    return np.stack(cols, axis=1).astype(np.float64)


def fit_softmax(x: np.ndarray, target: np.ndarray, n_classes: int, weight_decay: float = 1e-3) -> dict:
    """Standardised multinomial logistic regression, fitted with L-BFGS."""
    mean, std = x.mean(0), x.std(0) + 1e-8
    xt = torch.tensor((x - mean) / std, dtype=torch.float64)
    yt = torch.tensor(target, dtype=torch.long)
    layer = torch.nn.Linear(x.shape[1], n_classes).double()
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    opt = torch.optim.LBFGS(layer.parameters(), max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(layer(xt), yt) + weight_decay * layer.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return {"mean": mean.tolist(), "std": std.tolist(),
            "weight": layer.weight.detach().numpy().tolist(), "bias": layer.bias.detach().numpy().tolist()}


def predict_logits(model: dict, x: np.ndarray) -> np.ndarray:
    z = (x - np.array(model["mean"])) / np.array(model["std"])
    return z @ np.array(model["weight"]).T + np.array(model["bias"])


def stack_out_of_fold(samples: list[dict], x: np.ndarray, fold_of: np.ndarray,
                      time_bins: int, horizon: int) -> tuple[np.ndarray, list[dict]]:
    """Arrival-slice logits for every object, each fold scored by a stacker that never saw it."""
    target = arrival_bins(np.concatenate([s["tte"] for s in samples]), time_bins, horizon)
    logits = np.zeros((len(x), time_bins + 1))
    fitted = []
    for fold in np.unique(fold_of):
        train, test = fold_of != fold, fold_of == fold
        model = fit_softmax(x[train], target[train], time_bins + 1)
        logits[test] = predict_logits(model, x[test])
        fitted.append({"fold": int(fold), **model})
    return logits, fitted


def _table(per_method: dict[str, list[dict]], baselines: dict, slice_name: str) -> list[str]:
    lines = [HEADER]
    for name, ms in per_method.items():
        cells = []
        for key in METRICS:
            xs = np.array([_point(m["methods"]["model"][slice_name], key) for m in ms], dtype=float)
            cells.append(f"{np.nanmean(xs):.3f} ± {np.nanstd(xs):.3f}" if np.isfinite(xs).any() else "n/a")
        lines.append(f"| {name} ({len(ms)} seeds) | " + " | ".join(cells) + " |")
    for name, m in baselines.items():
        if name != "model":
            m = m[slice_name]
            lines.append(f"| {name} | " + " | ".join(
                f"{_point(m, k):.3f}" if np.isfinite(_point(m, k)) else "n/a" for k in METRICS) + " |")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True, help="A models/GRU cross-validation run set trained with --time-bins.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Defaults to evaluation/benchmarks/ensemble_eval/<run-name>/.")
    args = p.parse_args()
    out_dir = args.out_dir or PROJECT_ROOT / "evaluation/benchmarks/ensemble_eval" / args.run_name

    pooled = pooled_runs(args.run_name)
    missing = [c for c in COMPONENTS if c not in pooled]
    if missing:
        raise SystemExit(f"{args.run_name} has no cross-validation runs for {missing}")
    samples = pooled["gru"][0][0]
    key = [(s["session"], s["frame_idx"]) for s in samples]
    if any([(s["session"], s["frame_idx"]) for s in pooled[c][k][0]] != key
           for c in COMPONENTS for k in range(len(pooled[c]))):
        raise SystemExit("Component runs were not scored on identical test frames")
    if "tte" not in samples[0]:
        raise SystemExit("The run's dataset has no arrival times")

    config = json.loads((run_dirs(args.run_name)["gru"][0] / "config.json").read_text(encoding="utf-8"))
    time_bins, horizon = config["time_bins"], config.get("horizon", HORIZON_FRAMES)
    manifest = json.loads((Path(config["dataset"]) / "manifest.json").read_text(encoding="utf-8"))
    session_fold = {sid: f for f, sids in enumerate(manifest["folds"]) for sid in sids}
    fold_of = np.concatenate([np.full(len(s["y"]), session_fold[s["session"]]) for s in samples])

    base = [baseline_scores(s) for s in samples]
    nic = [b["nearest_in_cone"] for b in base]
    ttc = [b["TTC"] for b in base]

    seeds = min(len(pooled[c]) for c in COMPONENTS)
    scores: dict[str, list[tuple[Scores, Scores]]] = {"gru": [], "gru_attention": [],
                                                      "ensemble_rank": [], "ensemble_stack": []}
    stackers = []
    for k in range(seeds):
        gru = pooled["gru"][k][1:]
        att = pooled["gru_attention"][k][1:]
        scores["gru"].append(gru)
        scores["gru_attention"].append(att)
        scores["ensemble_rank"].append((rank_average(samples, [gru[0], att[0], nic, ttc]),
                                        rank_average(samples, [gru[1], att[1], nic, ttc])))
        logits, fitted = stack_out_of_fold(samples, features(samples, gru, att), fold_of, time_bins, horizon)
        scores["ensemble_stack"].append(model_scores(_split(logits, samples), time_bins, horizon))
        stackers.append({"seed_index": k, "folds": fitted})

    metrics = {name: [evaluate_scores(samples, pair) for pair in pairs] for name, pairs in scores.items()}
    baselines = metrics["gru"][0]["methods"]
    # Baselines are always included by paired_differences; add the learned components (and, for
    # the stacker, the simpler rank ensemble) as further comparisons.
    against = {"ensemble_rank": ["gru", "gru_attention"],
               "ensemble_stack": ["gru", "gru_attention", "ensemble_rank"]}
    paired = {name: paired_differences(samples, scores[name], {o: scores[o] for o in others})
              for name, others in against.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Ensemble - {args.run_name}", "",
             f"Pooled out-of-fold test frames: {metrics['gru'][0]['sessions']} sessions, "
             f"{len(fold_of)} objects, {seeds} seeds. Generated {time.strftime('%Y-%m-%d')} by "
             "`python -m models.Ensemble.ensemble`.", ""]
    for slice_name in ("all", "interaction"):
        lines += [f"## {slice_name} frames", ""] + _table({n: metrics[n] for n in scores}, baselines, slice_name) + [""]
    lines += ["## Ensemble minus other (mean over seeds, 95% paired session-bootstrap CI)", "",
              "| Ensemble | vs | AUROC difference | arrival_order difference |", "|---|---|---|---|"]
    for name, diffs in paired.items():
        for other, d in diffs.items():
            lines.append(f"| {name} | {other} | {_ci(d['auroc'])} | {_ci(d['arrival_order'])} |")
    lines += ["", "A difference is only credible if its CI excludes 0.", ""]
    report = "\n".join(lines)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "paired_differences.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    (out_dir / "stacker.json").write_text(json.dumps({"features": FEATURES, "time_bins": time_bins,
                                                      "horizon": horizon, "seeds": stackers}, indent=2),
                                          encoding="utf-8")
    print(report)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
