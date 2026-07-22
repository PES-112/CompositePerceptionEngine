# YOLO26n Hazard Training

This is the current source of truth for CPE YOLO26n hazard checkpoints, datasets, training commands, and evaluation artifacts.

## Active Dataset

Canonical dataset config:

```text
training/configs/cpe_hazard_classes.yaml
```

Active dataset root:

```text
data/yolo_finetune_v2_full
```

The detector uses a compact 17-class CPE taxonomy:

```text
person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign, fire hydrant,
pole, bollard, stairs, crosswalk, pothole, puddle, dog, bench
```

Roboflow source manifest:

```text
training/configs/roboflow_universe_sources.json
```

Keep `.env` local with `ROBOFLOW_API_KEY`; do not put secrets in scripts or JSON manifests.

## Current Checkpoints

| Version | Source checkpoint | Result | Use status |
|---|---|---|---|
| v1 | `models/yolo/base_yolo26n/yolo26n.pt` | Good `pole`/`bollard`, weak `stairs`; original compact hazard benchmark. | Historical baseline |
| v2 | `training/runs/cpe_yolo26n_hazards/weights/best.pt` | Added full 17-class data but caused retained-class regression, especially `truck`. | Superseded |
| v3 | `models/yolo/base_yolo26n/yolo26n.pt` | Best current trade-off; repaired most v2 retention degradation and improved mAP50-95. | Preferred |

Preferred checkpoint:

```text
training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt
```

## Training Command

For training runs, run the command directly in your terminal so progress is visible.

```bash
cd /home/student-4/Projects/CompositePerceptionEngine

.venv/bin/python training/scripts/train_yolo26n_hazards.py \
  --weights models/yolo/base_yolo26n/yolo26n.pt \
  --device 0 \
  --epochs 75 \
  --name cpe_yolo26n_hazards_v3_from_base
```

The script defaults are tuned for the GB10 server:

- `cache=ram`
- `batch=-1` AutoBatch
- 16 dataloader workers on the 20-core CPU
- no epoch-time validation or plots during the fast loop
- no automatic export during training

If a new training run is needed, prefer starting from base `models/yolo/base_yolo26n/yolo26n.pt` unless there is a narrow reason to continue from a CPE checkpoint.

## Evaluation Commands

All-class held-out test evaluation:

```bash
.venv/bin/python training/scripts/evaluate_yolo26n_hazards.py \
  --weights training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt \
  --split test \
  --device 0 \
  --skip-previews \
  --name eval_cpe_yolo26n_hazards_v3_from_base_test
```

Retained COCO-class comparison against base YOLO26n:

```bash
.venv/bin/python training/scripts/compare_yolo26n_retention.py \
  --candidate training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt \
  --base models/yolo/base_yolo26n/yolo26n.pt \
  --split test \
  --device 0 \
  --name compare_yolo26n_v3_from_base_retention_test
```


## Edge-Style Real-Time Benchmark

Use this when testing whether v3 is ready for an edge-style real-time loop. It measures streaming frame processing, YOLO26n + ByteTrack latency, optional depth loading, depth post-processing, and `frame_step` behavior.

Detector-only benchmark on held-out test images:

```bash
.venv/bin/python tools/benchmark_edge_realtime.py \
  --rgb-dir data/yolo_finetune_v2_full/images/test \
  --weights training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt \
  --device 0 \
  --fps 30 \
  --frame-step 3 \
  --max-frames 200 \
  --out evaluation/benchmarks/yolo26n_version_comparison/v3_edge_realtime_detector_only.json \
  --csv evaluation/benchmarks/yolo26n_version_comparison/v3_edge_realtime_detector_only.csv
```

Latest detector-only GB10 result: p95 total latency `9.65 ms` per processed frame, with the 30 FPS / `frame_step=3` real-time budget equal to `100 ms` per processed frame. This passes both the real-time streaming budget and the `<50 ms` reflex-path latency budget for the detector/tracker portion.

For a full architecture test, run the same command on prepared SANPO frame folders and include `--depth-dir` so depth loading, obstacle sweep, distance extraction, velocity, and bearing are timed too.

## Latest Metrics

Cumulative version comparison:

```text
evaluation/benchmarks/yolo26n_version_comparison/version_improvement_report.md
```

Overall held-out test metrics:

| Version | mAP50 | mAP50-95 | Notes |
|---|---:|---:|---|
| v1 | 0.651 | 0.432 | Original compact hazard benchmark; many retained COCO classes absent. |
| v2 | 0.599 | 0.424 | Superseded because retained `truck` collapsed versus base YOLO. |
| v3 | 0.641 | 0.459 | Preferred checkpoint; best mAP50-95 and repaired most retention issues. |

Remaining weak retention class: `dog`. If more data is added, prioritize Roboflow links for `dog`, then `stop sign`, `truck`, `bicycle`, and `bus`.

## Artifact Locations

| Artifact | Path |
|---|---|
| v1 benchmark | `evaluation/benchmarks/yolo26n_hazards_v1/` |
| v2 all-class eval | `training/runs/eval_cpe_yolo26n_hazards_v2_full_test/` |
| v2 retention comparison | `training/runs/compare_yolo26n_v2_full_retention_test/` |
| v3 all-class eval | `training/runs/eval_cpe_yolo26n_hazards_v3_from_base_test/` |
| v3 retention comparison | `training/runs/compare_yolo26n_v3_from_base_retention_test/` |
| cumulative comparison | `evaluation/benchmarks/yolo26n_version_comparison/` |

## Cleanup Policy

Keep successful model runs and compact evaluation artifacts. Delete interrupted training probe folders after confirming they are not the selected checkpoint. Do not delete v1/v2/v3 successful checkpoints unless explicitly archiving them elsewhere.
