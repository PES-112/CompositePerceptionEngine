# CPE Documentation Index

Current project documentation for the Composite Perception Engine (CPE).

## Read First

| File | Purpose |
|---|---|
| [`architecture.md`](./architecture.md) | Canonical system architecture, data flow, perception stack, physics verification, SLM strategy, and edge constraints. |
| [`yolo_training.md`](./yolo_training.md) | Current YOLO26n dataset, checkpoint, training, evaluation, and version-comparison guide. |
| [`artifact_structure.md`](./artifact_structure.md) | Naming conventions and cleanup rules for datasets, runs, benchmarks, and local model binaries. |
| [`research_paper_prompts.md`](./research_paper_prompts.md) | Paper-writing prompts aligned with the current CPE methodology and benchmark story. |
| [`plan_gap_analysis.md`](./plan_gap_analysis.md) | SANPO gap-analysis workflow for identifying perception blind spots and hard examples. |
| [`CHANGELOG.md`](./CHANGELOG.md) | Chronological project changes. |

## Current YOLO Status

Preferred detector checkpoint:

```text
training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt
```

Current cumulative evaluation report:

```text
evaluation/benchmarks/yolo26n_version_comparison/version_improvement_report.md
```

Summary: v3 starts from base `models/yolo/base_yolo26n/yolo26n.pt`, uses the cleaned 17-class dataset, repairs most v2 retained-class degradation, and is the preferred checkpoint for downstream simulation.

## Documentation Hygiene

- Keep this index pointed only at files that actually exist.
- Keep current workflow instructions in `yolo_training.md`; do not create new version-specific YOLO plan docs unless a future experiment needs a separate design proposal.
- Store compact benchmark summaries under `evaluation/benchmarks/` and avoid committing bulky generated images or binary checkpoints.
