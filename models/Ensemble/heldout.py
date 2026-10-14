"""
heldout.py
==========
The held-out version of ensemble.py: GRU, GRU_attention, both ensembles and
every baseline scored once on SANPO's official test sessions.

Train on the official train split first (dataset/GRU/resplit.py builds it):

    python -m dataset.GRU.resplit --dataset dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_f5_s0
    python -m models.GRU.train --arch gru           --dataset dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_official --time-bins 4 --seeds 0 1 2 --run-name official462_2026_10_14
    python -m models.GRU.train --arch gru_attention --dataset dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_official --time-bins 4 --seeds 0 1 2 --run-name official462_2026_10_14
    python -m models.Ensemble.heldout --run-name official462_2026_10_14

Nothing here touches the test sessions before scoring them: the GRUs were
trained and early-stopped on train/val, and the stacker is fitted on the
models' predictions for the validation sessions only. Cross-validation
(ensemble.py) used every session for model selection somewhere, so this is the
number to quote as the generalisation estimate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dataset.GRU.samples import arrival_bins, load_dataset, truncate_history
from evaluation.kinetic_ablation import HORIZON_FRAMES
from models.Ensemble.ensemble import (COMPONENTS, FEATURES, _split, _table, features, fit_softmax,
                                      predict_logits, rank_average)
from models.GRU.evaluate import baseline_scores, evaluate_scores, model_scores, predict
from models.GRU.registry import build_model
from models.GRU.summarize import _ci, paired_differences, run_dirs

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_run(run_dir: Path) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    model = build_model(ckpt["arch"], ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["config"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True, help="A fixed-split models/GRU run set trained with --time-bins.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Defaults to evaluation/benchmarks/heldout_official_split/<run-name>/.")
    args = p.parse_args()
    out_dir = args.out_dir or PROJECT_ROOT / "evaluation/benchmarks/heldout_official_split" / args.run_name

    dirs = run_dirs(args.run_name)
    missing = [c for c in COMPONENTS if c not in dirs]
    if missing:
        raise SystemExit(f"{args.run_name} has no runs for {missing}")
    if any("_fold" in d.name for c in COMPONENTS for d in dirs[c]):
        raise SystemExit(f"{args.run_name} is a cross-validation run set - use models.Ensemble.ensemble")

    _, first_cfg = load_run(dirs["gru"][0])
    dataset = Path(first_cfg["dataset"])
    manifest, splits = load_dataset(dataset, splits=("val", "test"))
    val, test = splits["val"], splits["test"]
    time_bins, horizon = first_cfg["time_bins"], first_cfg.get("horizon", HORIZON_FRAMES)
    if not time_bins:
        raise SystemExit("Runs were trained without --time-bins")

    # (val outputs, test outputs) per component per seed, seeds paired by sorted run name.
    outputs: dict[str, list[tuple[list, list]]] = {}
    for comp in COMPONENTS:
        outputs[comp] = []
        for d in dirs[comp]:
            model, cfg = load_run(d)
            if cfg["dataset_fingerprint"] != manifest["source"]["fingerprint"] or cfg["split"] != manifest["split"]:
                raise SystemExit(f"{d} was not trained on {manifest['name']}")
            k = cfg.get("history_steps") or 0
            v, t = (truncate_history(val, k), truncate_history(test, k)) if k else (val, test)
            outputs[comp].append((predict(model, v, "cpu"), predict(model, t, "cpu")))
    seeds = min(len(v) for v in outputs.values())

    def comp_scores(comp: str, k: int, part: int, samples: list[dict]):
        return model_scores(outputs[comp][k][part], time_bins, horizon)

    base_test = [baseline_scores(s) for s in test]
    nic = [b["nearest_in_cone"] for b in base_test]
    ttc = [b["TTC"] for b in base_test]
    target_val = arrival_bins(np.concatenate([s["tte"] for s in val]), time_bins, horizon)

    scores: dict[str, list] = {"gru": [], "gru_attention": [], "ensemble_rank": [], "ensemble_stack": []}
    stackers = []
    for k in range(seeds):
        gru_v, att_v = comp_scores("gru", k, 0, val), comp_scores("gru_attention", k, 0, val)
        gru_t, att_t = comp_scores("gru", k, 1, test), comp_scores("gru_attention", k, 1, test)
        scores["gru"].append(gru_t)
        scores["gru_attention"].append(att_t)
        scores["ensemble_rank"].append((rank_average(test, [gru_t[0], att_t[0], nic, ttc]),
                                        rank_average(test, [gru_t[1], att_t[1], nic, ttc])))
        stacker = fit_softmax(features(val, gru_v, att_v), target_val, time_bins + 1)
        logits = predict_logits(stacker, features(test, gru_t, att_t))
        scores["ensemble_stack"].append(model_scores(_split(logits, test), time_bins, horizon))
        stackers.append({"seed_index": k, **stacker})

    metrics = {name: [evaluate_scores(test, pair) for pair in pairs] for name, pairs in scores.items()}
    baselines = metrics["gru"][0]["methods"]
    against = {"gru": [], "gru_attention": ["gru"], "ensemble_rank": ["gru", "gru_attention"],
               "ensemble_stack": ["gru", "gru_attention", "ensemble_rank"]}
    paired = {name: paired_differences(test, scores[name], {o: scores[o] for o in others})
              for name, others in against.items()}

    stats = manifest["stats"]
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Held-out test - {args.run_name}", "",
             f"SANPO's official test sessions ({stats['test']['sessions']} with tracked objects, "
             f"{stats['test']['sessions_with_encounter']} with an arrival; {stats['test']['objects']} objects). "
             f"Trained on {stats['train']['sessions']} official-train sessions, early-stopped and stacker-fitted "
             f"on {stats['val']['sessions']} more. Dataset `{manifest['name']}`, {seeds} seeds. "
             f"Generated {time.strftime('%Y-%m-%d')} by `python -m models.Ensemble.heldout`.", "",
             "Time units: one step = 0.2 s (SANPO is 15 FPS, stride 3), so the 20-step arrival horizon is "
             "4 s and the 8-step history 1.6 s. Stored speeds are 2x too high (the CSVs assumed 30 FPS); "
             "every metric here is a ranking, which a uniform speed scale does not change.", ""]
    for slice_name in ("all", "interaction"):
        lines += [f"## {slice_name} frames", ""] + _table({n: metrics[n] for n in scores}, baselines, slice_name) + [""]
    lines += ["## Method minus other (mean over seeds, 95% paired session-bootstrap CI)", "",
              "| Method | vs | AUROC difference | arrival_order difference |", "|---|---|---|---|"]
    for name, diffs in paired.items():
        for other, d in diffs.items():
            lines.append(f"| {name} | {other} | {_ci(d['auroc'])} | {_ci(d['arrival_order'])} |")
    lines += ["", "A difference is only credible if its CI excludes 0. With 71 test sessions the CIs are wide; "
              "the 5-fold cross-validation (ensemble_eval/) is the tighter development estimate.", ""]
    report = "\n".join(lines)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "paired_differences.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    (out_dir / "stacker.json").write_text(json.dumps({"features": FEATURES, "time_bins": time_bins,
                                                      "horizon": horizon, "fitted_on": "val",
                                                      "seeds": stackers}, indent=2), encoding="utf-8")
    print(report)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
