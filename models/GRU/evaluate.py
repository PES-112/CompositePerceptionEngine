"""
evaluate.py
===========
Score a trained GRU on held-out sessions next to four baselines:

  per object  AUROC and average precision for "does this object become an
              encounter".
  per frame   encounter_top1 and flicker_rate, computed exactly as in
              evaluation/kinetic_ablation.py: frames with >= 2 objects,
              frame-count-weighted across sessions, 95% CI from a bootstrap
              over whole sessions.
  arrival     (datasets with arrival times) soonest_top1 — in frames with >= 2
              objects where a visible object arrives, is the top pick one of the
              first to arrive? — and arrival_order: for pairs of objects in a
              frame where one arrives strictly sooner (or the other never does),
              how often is the sooner one ranked higher? 0.5 = random.

Every method gives each object two scores: "will it arrive" and "how soon".
Baselines and binary models use one score for both; an arrival-time model
(train.py --time-bins) gives P(arrives within the horizon) and minus its
expected arrival time.

Baselines: K0 (the production formula), TTC ((distance - hazard) / closing
speed), nearest object, and nearest object inside the ±30° cone. The last is a
plain rule mirroring the label's own definition — a learned model has to beat
it to show it learned more than the label.

Everything is repeated on an "interaction" slice (>= 2 objects within 5 m, or
a crosswalk in view). Most frames contain no real interaction, so averages
over all frames hide whatever cross-object attention adds.

    python -m models.GRU.evaluate --run-dir models/GRU/GRU_attention/runs/<run>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.stats import rankdata

from dataset.GRU.samples import collate, load_dataset
from evaluation.kinetic_ablation import HORIZON_FRAMES, THETA_DEG, bootstrap_ci

INTERACTION_RADIUS_M = 5.0
N_BOOT = 2000

Scores = list[np.ndarray]   # one array per sample (frame), one value per object


def baseline_scores(s: dict) -> dict[str, np.ndarray]:
    in_cone = np.abs(s["bearing"]) < THETA_DEG
    out = {"K0": s["k0"]}
    if "ttc" in s:
        out["TTC"] = -np.maximum(s["ttc"], 0.0)   # sooner = higher; not closing = -inf
    out["nearest"] = -s["distance"]
    out["nearest_in_cone"] = -s["distance"] - np.where(in_cone, 0.0, 1e3)
    return out


def is_interaction(s: dict) -> bool:
    return int(np.sum(s["distance"] < INTERACTION_RADIUS_M)) >= 2 or "crosswalk" in s["classes"]


@torch.no_grad()
def predict(model: torch.nn.Module, samples: list[dict], device: str, batch_size: int = 128) -> Scores:
    """Raw outputs per sample, padding stripped: [n] logits, or [n, bins + 1] arrival-time logits."""
    model.eval()
    out = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        batch = collate(chunk)
        logits = model(batch["x"].to(device), batch["cls"].to(device), batch["mask"].to(device)).cpu().numpy()
        out.extend(logits[b, :len(s["y"])] for b, s in enumerate(chunk))
    return out


def model_scores(outputs: Scores, time_bins: int, horizon: int = HORIZON_FRAMES) -> tuple[Scores, Scores]:
    """(will-it-arrive, how-soon) scores from raw outputs. Binary models use their logit for both."""
    if time_bins == 0:
        return outputs, outputs
    width = -(-horizon // time_bins)
    lo = np.arange(time_bins) * width
    hi = np.append(lo[1:] - 1, horizon)
    mids = (lo + hi) / 2
    never = horizon + width   # "not within the horizon" counts as one slice past it
    arrive, soon = [], []
    for o in outputs:
        e = np.exp(o - o.max(axis=-1, keepdims=True))
        p = e / e.sum(axis=-1, keepdims=True)
        arrive.append(1.0 - p[:, -1])
        soon.append(-(p[:, :-1] @ mids + p[:, -1] * never))
    return arrive, soon


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)   # average ranks, so ties count as half-correct
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    hits = labels[np.argsort(-scores, kind="stable")] > 0.5
    if not hits.any():
        return float("nan")
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].mean())


def arrival_pairs(tte: np.ndarray, score: np.ndarray) -> tuple[float, int]:
    """(concordant, total) over object pairs where one arrives strictly sooner; ties in score count half."""
    t = np.where(tte >= 0, tte, np.inf).astype(float)
    sooner = t[:, None] < t[None, :]
    right = (score[:, None] > score[None, :]) + 0.5 * (score[:, None] == score[None, :])
    return float(right[sooner].sum()), int(sooner.sum())


def per_object(samples: list[dict], scores: Scores, keep: Callable[[dict], bool] | None = None) -> dict:
    picked = [(s["y"], sc) for s, sc in zip(samples, scores) if keep is None or keep(s)]
    if not picked:
        return {"objects": 0, "positive_rate": float("nan"), "auroc": float("nan"), "average_precision": float("nan")}
    labels = np.concatenate([y for y, _ in picked])
    flat = np.concatenate([sc for _, sc in picked]).astype(float)
    return {
        "objects": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "auroc": auroc(flat, labels),
        "average_precision": average_precision(flat, labels),
    }


def per_frame(samples: list[dict], arrive: Scores, soon: Scores,
              keep: Callable[[dict], bool] | None = None, seed: int = 0) -> dict:
    """encounter_top1/flicker as in kinetic_ablation.session_metrics(), plus the arrival metrics."""
    by_session: dict[str, list[tuple]] = defaultdict(list)
    pairs: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    has_tte = "tte" in samples[0] if samples else False
    for s, a, n in zip(samples, arrive, soon):
        if len(s["y"]) < 2 or (keep is not None and not keep(s)):
            continue
        top_arrive, top_soon = int(np.argmax(a)), int(np.argmax(n))
        hit = bool(s["y"][top_arrive]) if s["frame_has_encounter"] else None
        first = None
        if has_tte and (s["tte"] >= 0).any():
            first = bool(s["tte"][top_soon] == s["tte"][s["tte"] >= 0].min())
            c, t = arrival_pairs(s["tte"], n)
            pairs[s["session"]][0] += c
            pairs[s["session"]][1] += t
        # The announced object is the "how soon" pick, so flicker follows it.
        by_session[s["session"]].append((s["frame_idx"], s["track_ids"][top_soon], hit, first))

    top1, soonest, flicker, weights, encounter_frames, arrival_frames = [], [], [], [], 0, 0
    for rows in by_session.values():
        rows.sort(key=lambda r: r[0])
        tops = [r[1] for r in rows]
        hits = [r[2] for r in rows if r[2] is not None]
        firsts = [r[3] for r in rows if r[3] is not None]
        encounter_frames += len(hits)
        arrival_frames += len(firsts)
        flicker.append(sum(a != b for a, b in zip(tops, tops[1:])) / max(1, len(tops) - 1))
        top1.append(float(np.mean(hits)) if hits else float("nan"))
        soonest.append(float(np.mean(firsts)) if firsts else float("nan"))
        weights.append(len(rows))
    out = {
        "frames": int(sum(weights)),
        "encounter_frames": encounter_frames,
        "encounter_top1": bootstrap_ci(top1, weights, N_BOOT, seed),
        "flicker_rate": bootstrap_ci(flicker, weights, N_BOOT, seed),
    }
    if has_tte:
        ratios = [c / t if t else float("nan") for c, t in pairs.values()]
        out.update(arrival_frames=arrival_frames,
                   soonest_top1=bootstrap_ci(soonest, weights, N_BOOT, seed),
                   arrival_pairs=int(sum(t for _, t in pairs.values())),
                   arrival_order=bootstrap_ci(ratios, [t for _, t in pairs.values()], N_BOOT, seed))
    return out


def evaluate_scores(samples: list[dict], model: tuple[Scores, Scores] | None = None) -> dict:
    """Metrics for the model's (arrive, soon) scores, if given, and for every baseline on the same frames."""
    methods = {} if model is None else {"model": model}
    per_sample = [baseline_scores(s) for s in samples]
    for name in per_sample[0]:
        scores = [b[name] for b in per_sample]
        methods[name] = (scores, scores)
    results = {}
    for name, (arrive, soon) in methods.items():
        results[name] = {
            "all": {**per_object(samples, arrive), **per_frame(samples, arrive, soon)},
            "interaction": {**per_object(samples, arrive, is_interaction),
                            **per_frame(samples, arrive, soon, is_interaction)},
        }
    return {"sessions": len({s["session"] for s in samples}), "methods": results}


def evaluate_samples(model: torch.nn.Module, samples: list[dict], device: str,
                     time_bins: int = 0, horizon: int = HORIZON_FRAMES) -> tuple[dict, Scores]:
    """(metrics, raw test outputs) — the outputs let summarize.py pool cross-validation folds."""
    outputs = predict(model, samples, device)
    return evaluate_scores(samples, model_scores(outputs, time_bins, horizon)), outputs


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (tuple, list)):
        pt, lo, hi = v
        return f"{pt:.3f} [{lo:.3f}, {hi:.3f}]" if np.isfinite(pt) else "n/a"
    return f"{v:.3f}" if np.isfinite(v) else "n/a"


def write_report(metrics: dict, path: Path, title: str) -> None:
    lines = [f"# {title}", "", f"Held-out test sessions: {metrics['sessions']}", ""]
    for slice_name, blurb in (("all", "All frames"),
                              ("interaction", f"Interaction slice (>= 2 objects within {INTERACTION_RADIUS_M:g} m, or a crosswalk)")):
        first = next(iter(metrics["methods"].values()))[slice_name]
        lines += [
            f"## {blurb}", "",
            f"Objects: {first['objects']} (encounter rate {_fmt(first['positive_rate'])}), "
            f"frames with >= 2 objects: {first['frames']}, with an encounter: {first['encounter_frames']}", "",
            "| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) "
            "| arrival_order (95% CI) | flicker_rate (95% CI) |",
            "|---|---:|---:|---|---|---|---|",
        ]
        for name, m in metrics["methods"].items():
            s = m[slice_name]
            lines.append(f"| {name} | {_fmt(s['auroc'])} | {_fmt(s['average_precision'])} | "
                         f"{_fmt(s['encounter_top1'])} | {_fmt(s.get('soonest_top1'))} | "
                         f"{_fmt(s.get('arrival_order'))} | {_fmt(s['flicker_rate'])} |")
        lines.append("")
    lines += ["Higher is better for everything except flicker_rate. arrival_order: 0.5 = random. "
              "CIs resample whole sessions, as in the K0 ablation."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from models.GRU.registry import build_model

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="A run folder containing model.pt.")
    p.add_argument("--dataset", type=Path, default=None,
                   help="Defaults to the dataset folder the run was trained on.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.run_dir / "model.pt", map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = build_model(ckpt["arch"], ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)

    manifest, splits = load_dataset(args.dataset or Path(cfg["dataset"]), splits=("test",), fold=cfg.get("fold"))
    if manifest["source"]["fingerprint"] != cfg["dataset_fingerprint"] or manifest["split"] != cfg["split"]:
        raise SystemExit(f"{manifest['name']} is not the dataset this run was trained on ({cfg['dataset_name']})")

    metrics, _ = evaluate_samples(model, splits["test"], args.device,
                                  cfg.get("time_bins", 0), cfg.get("horizon", HORIZON_FRAMES))
    (args.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics, args.run_dir / "report.md", f"{ckpt['arch']} - {args.run_dir.name}")
    print((args.run_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
