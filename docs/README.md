# CPE Documentation Index

Each document has one primary responsibility. Update the owning document instead of creating a version-specific plan for information that already has a home.

## Documentation Map

Grouped by what question each file answers, not just alphabetically — start here if you're not sure
which doc owns something.

### Where things stand right now

| File | Owns |
|---|---|
| [`progress.md`](./progress.md) | Current completion checklist, active focus, and immediate next steps. Living doc — update as state changes. |
| [`progress_presentation.md`](./progress_presentation.md) | Presentation-ready narrative of progress and key results — slide-outline-shaped, not a checklist. |
| [`pending_work.md`](./pending_work.md) | Everything not yet done, grouped by priority, with the top-3 recommended next actions. |

### How the system works

| File | Owns |
|---|---|
| [`architecture.md`](./architecture.md) | Runtime components, contracts, data flow, routing, safety behavior, the technology audit behind each model choice (§11), and the historical implementation plan (§12). |
| [`hardware_targets.md`](./hardware_targets.md) | Observed GB10 host specs, planned edge target, latency budgets, and measured-versus-simulated status. |

### Research methodology and evidence

| File | Owns |
|---|---|
| [`methodology.md`](./methodology.md) | **Paper-ready methods reference** — dataset construction, detector training/eval, kinetic-score derivation and ablation design, edge-latency simulation protocol, the experimental-design principles behind all of it, and an Appendix of LLM paper-drafting prompts. Start here for anything paper-related. |
| [`kinetic_score_opinion.md`](./kinetic_score_opinion.md) | Full reasoning trail and decision record behind the kinetic-score evaluation strategy (why K1–K5 were dropped, how the ablation/referee design was chosen, the formal options analysis in §10). Source material for `methodology.md` §4. |
| [`ablation_guide.md`](./ablation_guide.md) | Step-by-step run guide for the full K0 verification pass (YOLO → ablation → top-3 VLM check → referee), plus the latest ablation run's results (§6). |
| [`related_work.md`](./related_work.md) | Verified external paper citations, organized by which CPE component or design decision each grounds. Check here before citing anything from memory. |

### Datasets and training

| File | Owns |
|---|---|
| [`yolo_training.md`](./yolo_training.md) | YOLO datasets, checkpoints, training/evaluation commands, metrics, and cleanup policy. |
| [`sanpo_dataset.md`](./sanpo_dataset.md) | SANPO bucket layout, valid-stream intake, local data convention, edge benchmark evidence, and the depth-versus-YOLO gap analysis workflow. |

### History

| File | Owns |
|---|---|
| [`CHANGELOG.md`](./CHANGELOG.md) | Chronological record of repository changes. |

## Current Detector

Preferred checkpoint:

```text
training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt
```

Current cumulative evaluation report:

```text
evaluation/benchmarks/yolo26n_version_comparison/version_improvement_report.md
```

YOLO26n v3 starts from `models/yolo/base_yolo26n/yolo26n.pt`, uses the cleaned 17-class dataset, repairs most v2 retained-class degradation, and is the preferred detector for downstream simulation. As of 2026-08-25 the perception stack defaults to it: `YoloTracker`'s `DEFAULT_MODEL` is `training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt`, so callers no longer have to pass `model_path` to avoid silently running the base COCO taxonomy.

## Artifact and Naming Conventions

| Path | Purpose |
|---|---|
| `training/scripts/` | Reusable download, training, evaluation, comparison, and export scripts. |
| `training/configs/` | Canonical class taxonomy and dataset-source manifests. |
| `training/runs/<run_id>/` | Local Ultralytics outputs and checkpoints. |
| `evaluation/benchmarks/<benchmark_id>/` | Compact CSV, JSON, and Markdown results worth retaining. |
| `evaluation/generate_report_figures.py` | Regenerates every plot/table under `evaluation/benchmarks/figures/` from the current benchmark artifacts. Re-run after any benchmark result changes. |
| `evaluation/benchmarks/figures/` | Generated PNG plots and a `results_summary.md` table digest — output only, safe to delete and regenerate. |
| `models/yolo/<model_id>/README.md` | Human-readable model registry entry. |
| `simulation/datasets/sanpo/valid_streams.json` | Curated SANPO streams for CPE evaluation. |
| `data/` | Local raw and merged datasets; do not commit. |
| `.env` | Local secrets such as `ROBOFLOW_API_KEY`; do not commit. |

Use names that identify the model, task, and version, such as `cpe_yolo26n_hazards_v3_from_base`. Avoid ambiguous names such as `test`, `new`, `final`, or `latest`.

After a successful training run, retain `weights/best.pt`, `args.yaml`, and `results.csv` locally; copy compact reports into `evaluation/benchmarks/`; remove bulky previews only after confirming they are not needed for analysis or a paper figure.

## Maintenance Rules

- Keep this index pointed only at files that exist.
- Put current detector workflow changes in `yolo_training.md`.
- Put benchmark data in `evaluation/benchmarks/`, then summarize only the durable conclusion in the relevant guide.
- Label analytical edge estimates as simulated until the same workload is measured on physical target hardware.
- Update the main `README.md` project tree whenever files, folders, models, datasets, or benchmark artifacts are added or moved.
- When a benchmark result changes, re-run `evaluation/generate_report_figures.py` so the figures and
  `progress_presentation.md` stay consistent with the underlying data.
- When an item in `pending_work.md` is finished, move its status into `progress.md` rather than
  deleting it — `pending_work.md` should only ever list what's actually left.
- Keep `kinetic_score_opinion.md` as permanent historical record even after a decision resolves;
  only `methodology.md` gets rewritten to reflect the current, condensed state.
- **Prefer adding a section to an existing doc over creating a new file.** This index currently holds
  12 files; before adding a 13th, check whether the content actually belongs as a new section in a
  file above with related ownership. Docs merged into others (`verification_guide.md` →
  `ablation_guide.md`, `decisions.md` → `kinetic_score_opinion.md`, `sanpo_gap_analysis.md` →
  `sanpo_dataset.md`, `roadmap.md` → `architecture.md`, `research_paper_prompts.md` →
  `methodology.md`) are listed in `CHANGELOG.md` under 2026-08-27 if you need their original content.
