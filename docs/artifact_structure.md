# Artifact Structure and Naming Conventions

This project keeps source code, compact evaluation summaries, datasets, and model binaries separate so YOLO experiments do not blur together.

## Source-Controlled Files

| Path | Purpose |
|---|---|
| `training/scripts/` | Reusable training, download, evaluation, and comparison scripts. |
| `training/configs/cpe_hazard_classes.yaml` | Canonical 17-class CPE YOLO dataset config. |
| `training/configs/roboflow_universe_sources.example.json` | Safe template for Roboflow source manifests. |
| `docs/plan_yolo_v2_finetune.md` | Checklist for the next v2 fine-tune from v1 `best.pt`. |
| `evaluation/benchmarks/<run_id>/hazard_eval_*` | Compact CSV/JSON/Markdown metric summaries worth keeping. |
| `models/yolo/<model_id>/README.md` | Human-readable model registry note for each local checkpoint. |

## Local-Only Generated Artifacts

| Path | Purpose |
|---|---|
| `data/roboflow_universe/` | Raw Roboflow downloads. Do not commit. |
| `data/yolo_finetune/` | Merged YOLO train/val/test dataset. Do not commit. |
| `training/runs/<run_id>/` | Ultralytics run output. Keep locally; commit only compact summaries if needed. |
| `models/yolo/<model_id>/best.pt` | Local model checkpoint. Do not commit binary weights. |
| `.env` | Local secrets such as `ROBOFLOW_API_KEY`. Do not commit. |

## Run Naming

Use stable names that say model family, task, and version:

```text
cpe_yolo26n_hazards_v1
cpe_yolo26n_hazards_v2
eval_cpe_yolo26n_hazards_v2_test
compare_cpe_yolo26n_hazards_v2_test
```

Avoid names like `test`, `new`, `final`, or `latest`; they become ambiguous as soon as another run exists.

## Cleanup Rule

After each training run:

1. Keep `weights/best.pt`, `args.yaml`, and `results.csv` locally.
2. Copy compact reports into `evaluation/benchmarks/<run_id>/`.
3. Delete bulky generated plots/previews unless they are needed for a paper figure.
4. Record the checkpoint path and decision in the relevant plan or changelog.
