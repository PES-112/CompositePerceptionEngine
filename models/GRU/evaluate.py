"""
evaluate.py
===========
Score a trained GRU on held-out sessions next to three baselines, at two levels:

  per object  AUROC and average precision for "does this object become an
              encounter".
  per frame   encounter_top1 and flicker_rate, computed exactly as in
              evaluation/kinetic_ablation.py: frames with >= 2 objects,
              frame-count-weighted across sessions, 95% CI from a bootstrap
              over whole sessions.

Baselines: K0 (the production formula), nearest object, and nearest object
inside the ±30° encounter cone. The last one is a plain rule that mirrors the
label's own definition — a learned model has to beat it to show it learned
more than the label.

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
from evaluation.kinetic_ablation import THETA_DEG, bootstrap_ci

INTERACTION_RADIUS_M = 5.0
N_BOOT = 2000
BASELINES = ("K0", "nearest", "nearest_in_cone")


def baseline_scores(s: dict) -> dict[str, np.ndarray]:
    in_cone = np.abs(s["bearing"]) < THETA_DEG
    return {
        "K0": s["k0"],
        "nearest": -s["distance"],
        "nearest_in_cone": -s["distance"] - np.where(in_cone, 0.0, 1e3),
    }


def is_interaction(s: dict) -> bool:
    return int(np.sum(s["distance"] < INTERACTION_RADIUS_M)) >= 2 or "crosswalk" in s["classes"]


@torch.no_grad()
def predict(model: torch.nn.Module, samples: list[dict], device: str, batch_size: int = 128) -> list[np.ndarray]:
    """Encounter logits per sample, padding stripped."""
    model.eval()
    out = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        batch = collate(chunk)
        logits = model(batch["x"].to(device), batch["cls"].to(device), batch["mask"].to(device)).cpu().numpy()
        out.extend(logits[b, :len(s["y"])] for b, s in enumerate(chunk))
    return out


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


def per_object(samples: list[dict], scores: list[np.ndarray],
               keep: Callable[[dict], bool] | None = None) -> dict:
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


def per_frame(samples: list[dict], scores: list[np.ndarray],
              keep: Callable[[dict], bool] | None = None, seed: int = 0) -> dict:
    """Same definitions and weighting as kinetic_ablation.session_metrics()."""
    by_session: dict[str, list[tuple]] = defaultdict(list)
    for s, sc in zip(samples, scores):
        if len(s["y"]) < 2 or (keep is not None and not keep(s)):
            continue
        top = int(np.argmax(sc))
        hit = bool(s["y"][top]) if s["frame_has_encounter"] else None
        by_session[s["session"]].append((s["frame_idx"], s["track_ids"][top], hit))

    top1, flicker, weights, encounter_frames = [], [], [], 0
    for rows in by_session.values():
        rows.sort(key=lambda r: r[0])
        tops = [t for _, t, _ in rows]
        hits = [h for _, _, h in rows if h is not None]
        encounter_frames += len(hits)
        flicker.append(sum(a != b for a, b in zip(tops, tops[1:])) / max(1, len(tops) - 1))
        top1.append(float(np.mean(hits)) if hits else float("nan"))
        weights.append(len(rows))
    return {
        "frames": int(sum(weights)),
        "encounter_frames": encounter_frames,
        "encounter_top1": bootstrap_ci(top1, weights, N_BOOT, seed),
        "flicker_rate": bootstrap_ci(flicker, weights, N_BOOT, seed),
    }


def evaluate_samples(model: torch.nn.Module, samples: list[dict], device: str) -> dict:
    methods = {"model": predict(model, samples, device)}
    for name in BASELINES:
        methods[name] = [baseline_scores(s)[name] for s in samples]
    results = {}
    for name, scores in methods.items():
        results[name] = {
            "all": {**per_object(samples, scores), **per_frame(samples, scores)},
            "interaction": {**per_object(samples, scores, is_interaction),
                            **per_frame(samples, scores, is_interaction)},
        }
    return {"sessions": len({s["session"] for s in samples}), "methods": results}


def _fmt(v) -> str:
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
            "| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | flicker_rate (95% CI) |",
            "|---|---:|---:|---|---|",
        ]
        for name, m in metrics["methods"].items():
            s = m[slice_name]
            lines.append(f"| {name} | {_fmt(s['auroc'])} | {_fmt(s['average_precision'])} | "
                         f"{_fmt(s['encounter_top1'])} | {_fmt(s['flicker_rate'])} |")
        lines.append("")
    lines += ["encounter_top1: higher is better. flicker_rate: lower is better. "
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

    manifest, splits = load_dataset(args.dataset or Path(cfg["dataset"]), splits=("test",))
    if manifest["source"]["fingerprint"] != cfg["dataset_fingerprint"] or manifest["split"] != cfg["split"]:
        raise SystemExit(f"{manifest['name']} is not the dataset this run was trained on ({cfg['dataset_name']})")
    samples = splits["test"]

    metrics = evaluate_samples(model, samples, args.device)
    (args.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics, args.run_dir / "report.md", f"{ckpt['arch']} - {args.run_dir.name}")
    print((args.run_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
