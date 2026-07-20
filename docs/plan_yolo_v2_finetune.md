# YOLO26n Hazard Fine-Tuning v2 Implementation Plan

**Date:** 2026-07-20  
**Starting checkpoint:** `training/runs/cpe_yolo26n_hazards/weights/best.pt`  
**Goal:** Improve weak `stairs` and moderate `pothole` performance while preserving `pole`, `bollard`, and retained COCO classes.

---

## Current v1 Evidence

Held-out test results from `training/runs/eval_cpe_yolo26n_hazards_test`:

| Class | Precision | Recall | mAP50 | mAP50-95 | Decision |
|---|---:|---:|---:|---:|---|
| `pole` | 0.755 | 0.940 | 0.937 | 0.649 | Keep existing data; no major expansion needed. |
| `bollard` | 0.802 | 0.961 | 0.934 | 0.700 | Keep existing data; no major expansion needed. |
| `pothole` | 0.650 | 0.619 | 0.647 | 0.359 | Add better examples and hard negatives. |
| `stairs` | 0.304 | 0.101 | 0.086 | 0.020 | High-priority dataset repair. |

The current split has no ground-truth images for retained classes (`person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `traffic light`, `stop sign`, `fire hydrant`, `dog`, `bench`), so retention against base YOLO cannot be judged until those examples are added.

---

## GB10 High-Throughput Training Profile

The v2 training script is tuned for the ARM Grace-Blackwell GB10 server with 20 CPU cores and about 120GB unified RAM/VRAM. The current merged YOLO dataset is about **6.7GB** on disk, with images accounting for almost all of that size, so RAM caching fits comfortably within available memory.

Default v2 training behavior in `training/scripts/train_yolo26n_hazards.py` now uses:

- `cache=ram` to hold the YOLO images in the unified memory pool and avoid repeated disk reads.
- `batch=-1` so Ultralytics AutoBatch searches for the largest viable batch on the selected CUDA device. If AutoBatch is too conservative, test static `--batch 256` or `--batch 512` after watching memory and throughput.
- `workers=-1` with `--reserve-cpu-cores 4`, which resolves to 16 data-loader workers on the 20-core GB10.
- `val=False`, `plots=False`, `save_period=-1`, and no default export, so the raw training loop avoids validation, plot generation, preview image writing, and ONNX dependency overhead.

Because validation is disabled during fast training, select the final v2 checkpoint using the explicit end-validation scripts below, not just training-loop plots.

---

## What You Need To Provide

Add these to `training/configs/roboflow_universe_sources.json` or provide the Universe URLs so they can be added:

- **Stairs:** 2-3 Roboflow Universe object-detection datasets with real bounding boxes for `stairs`, `stair`, `staircase`, steps, curb steps, outdoor stairways, partial/occluded stairs.
- **Potholes:** 1-2 additional pothole datasets, preferably street-level/egocentric or dashcam-style, with varied lighting and road textures.
- **Hard negatives:** datasets/images containing roads, sidewalks, shadows, painted markings, ramps, tiles, drainage covers, and flat curb edges that should **not** become `pothole` or `stairs`.
- **Retained classes:** at least 300-500 validation/test images total containing `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `traffic light`, `stop sign`, `fire hydrant`, `dog`, and `bench` in the CPE 17-class label format.
- **Dog:** minimum 200 images, preferred 400+ images; include leashed/unleashed dogs, partial occlusions, small/large breeds, and sidewalk/crosswalk scenes.
- **Bench:** minimum 200 images, preferred 400+ images; include park benches, bus-stop benches, occluded benches, side/front views, and confusing railings or low barriers as negatives.

Universe URL format needed:

```text
https://universe.roboflow.com/<workspace>/<project>/dataset/<version>
```

For each URL, also provide the source label names if they differ from CPE names, e.g. `Steps -> stairs`, `road damage -> pothole`.

---


### Verified Roboflow Source Versions

Checked via the Roboflow SDK on 2026-07-20:

| Source | Latest usable version | Images reported by Roboflow | Status |
|---|---:|---:|---|
| `stairs-n9kkx/stairs-lusiz` | 1 | 4,089 | Active |
| `tom-lai-8bp7n/stairs-i2yia` | 3 | 1,567 | Active |
| `cpcs-432/potholes-detection-j3w8s` | 2 | 2,742 | Active |
| `fyp-46tke/potholes-xbcxz` | 1 | 3,753 | Active |
| `flower-uqajd/bench-nyfso` | 1 | 168 | Active |
| `blind-1vbdv/bench-detect-3g0kf` | 1 | 967 | Active |
| `sky-zfxvm/coco-yrx1j` | 1 | 9,900 | Active |
| `crosswalk-aee0t/crosswalk-ibrdh` | 4 | 926 | Active |
| `walksenses-workspace/puddle-detection-5y0s4` | 1 | 1,500+ | Active |
| `uijin/motorcycle-srwil` | 2 | 1,152 | Active |
| `tiger-dataset/dog-ukbxr` | none | 0 generated versions | Not downloadable until a dataset version is generated |

The previous invalid `stairs_detection-9av4i-lyswf-orcim-duw1m` entry was removed because Roboflow reported version 1 was not found.

## Dataset Cap Policy

You can provide multiple Roboflow Universe URLs per class. The download/merge script now enforces `class_image_limits` from `training/configs/roboflow_universe_sources.json`, or built-in v2 defaults if the manifest omits them. This means a large source dataset will be sampled only until the target class reaches its cap, while smaller datasets can stack together until the cap is reached. The downloader mutates one running image counter during merge, so final class counts should be audited from label files before training.

Current preferred caps are: `person` 500, `bicycle` 400, `car` 500, `motorcycle` 400, `bus` 300, `truck` 300, `traffic light` 300, `stop sign` 250, `fire hydrant` 250, `pole` 1000, `bollard` 1000, `dog` 400, `bench` 400, `stairs` 1500, `crosswalk` 600, `pothole` 1500, and `puddle` 1000 images. Pass `--no-class-limits` only for audits where you intentionally want every matching image copied.

For SANPO-based final simulation, include SANPO-style validation/test data and hard examples, but do not fine-tune only on SANPO. SANPO should be used for domain alignment and final evaluation; Roboflow/COCO-style labeled data should still supply rare object boxes such as `stairs`, `pothole`, `dog`, and `bench`.

---

## Checklist

### Dataset Intake

- [ ] Add new stairs dataset entries to `training/configs/roboflow_universe_sources.json`.
- [ ] Add new pothole dataset entries to `training/configs/roboflow_universe_sources.json`.
- [ ] Add retained-class validation/test data, preferably from SANPO/COCO-style sources remapped into CPE IDs 0-8 plus 15-16.
- [x] Configure the v2 balanced output root as `data/yolo_finetune_v2_full` so the prior v2 merge remains untouched.
- [ ] Run Roboflow download/merge:

```bash
.venv/bin/python training/scripts/download_roboflow_universe.py \
  --manifest training/configs/roboflow_universe_sources.json
```

- [ ] Re-run train/val/test balancing for any train-only source datasets with `training/scripts/rebalance_yolo_splits.py`.
- [ ] Confirm all labels are 5-column YOLO detect rows: `class cx cy w h`.

### Fine-Tuning v2

- [ ] Start from the v1 best checkpoint, not base YOLO, using the GB10 high-throughput defaults:

```bash
.venv/bin/python training/scripts/train_yolo26n_hazards.py \
  --weights training/runs/cpe_yolo26n_hazards/weights/best.pt \
  --device 0 \
  --epochs 50 \
  --name cpe_yolo26n_hazards_v2
```

- [ ] Confirm the startup log shows `cache='ram'`, `batch=-1`, `workers=16`, `val_during_train=False`, and `plots=False`. The script keeps `cudnn_benchmark=False` during AutoBatch search to avoid Ultralytics falling back to batch 16.
- [ ] If AutoBatch chooses a surprisingly small batch, rerun a short 1-epoch probe with `--batch 256`, then `--batch 512` only if GPU memory headroom remains healthy.
- [ ] Keep `--export-formats` omitted during training; export after the winning checkpoint is selected and export dependencies are fixed.
- [ ] Use `--val` only for slower diagnostic runs where epoch-by-epoch validation is worth the extra time.

### End Validation

- [ ] Run all-class mAP validation on the checkpoint reported by the training script. If fast training was run with `--no-val`, this may be `last.pt` rather than `best.pt`:

```bash
.venv/bin/python training/scripts/evaluate_yolo26n_hazards.py \
  --weights training/runs/cpe_yolo26n_hazards_v2/weights/last.pt \
  --split test \
  --device 0
```

- [ ] Compare v2 against base YOLO for retained COCO classes using that same selected checkpoint:

```bash
.venv/bin/python training/scripts/compare_yolo26n_retention.py \
  --candidate training/runs/cpe_yolo26n_hazards_v2/weights/last.pt \
  --base yolo26n.pt \
  --split test \
  --device 0
```

- [ ] Inspect prediction previews under `training/runs/eval_cpe_yolo26n_hazards_v2_test/prediction_previews/`.
- [ ] Accept v2 only if stairs/pothole improve and retained-class precision/recall does not regress severely versus base YOLO.

---

## Acceptance Targets

| Class group | Target |
|---|---|
| `pole`, `bollard` | Do not regress more than 5 percentage points in mAP50. |
| `pothole` | Recall >= 0.75 and mAP50 >= 0.70. |
| `stairs` | Recall >= 0.50 minimum; mAP50 >= 0.50 preferred. |
| Retained COCO classes | Candidate F1 should stay within 10 percentage points of base YOLO on the same retained-class test images. |
| Overall | No empty validation/test classes for any class we claim in the model card. |
