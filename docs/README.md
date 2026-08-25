# CPE Documentation Index

Each document has one primary responsibility. Update the owning document instead of creating a version-specific plan for information that already has a home.

## Documentation Map

| File | Owns |
|---|---|
| [`progress.md`](./progress.md) | Current completion checklist, active focus, and immediate next steps. |
| [`architecture.md`](./architecture.md) | Runtime components, contracts, data flow, routing, and safety behavior. |
| [`hardware_targets.md`](./hardware_targets.md) | Observed GB10 host specs, planned edge target, latency budgets, and measured-versus-simulated status. |
| [`roadmap.md`](./roadmap.md) | Technology choices, implementation phases, and future work. |
| [`yolo_training.md`](./yolo_training.md) | YOLO datasets, checkpoints, training/evaluation commands, metrics, and cleanup policy. |
| [`sanpo_dataset.md`](./sanpo_dataset.md) | SANPO bucket layout, valid-stream intake, local data convention, and edge benchmark evidence. |
| [`sanpo_gap_analysis.md`](./sanpo_gap_analysis.md) | Operational depth-versus-YOLO blind-spot and hard-example workflow. |
| [`research_paper_prompts.md`](./research_paper_prompts.md) | Paper-drafting prompts aligned with implemented methodology and evidence. |
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
