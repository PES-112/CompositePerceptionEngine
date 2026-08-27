# SANPO Dataset, Intake, Edge Evaluation, and Gap Analysis

This note documents the SANPO public Google Cloud Storage layout used by CPE for real-time edge
testing with `valid_streams.json`, plus the empirical depth-versus-YOLO gap analysis workflow used to
prioritize which hazard classes to add (§"Gap Analysis Workflow" below).

## Source

- Browser URL: `https://console.cloud.google.com/storage/browser/gresearch/sanpo_dataset/v0`
- GCS path: `gs://gresearch/sanpo_dataset/v0/`
- HTTP object base: `https://storage.googleapis.com/gresearch/sanpo_dataset/v0/`
- Local valid stream manifest: `simulation/datasets/sanpo/valid_streams.json`

The bucket is public and can be listed through the Google Cloud Storage JSON API with `prefix` and `delimiter` parameters. Google Cloud Storage uses object-name prefixes to simulate folders, so the folder structure below is derived from object prefixes rather than a native filesystem.

## Dataset Contents

SANPO is an egocentric outdoor navigation dataset with real and synthetic subsets. It contains RGB video frames, camera metadata, pose/IMU information, depth maps, and segmentation annotations for supported subsets. The local CPE use case is edge perception testing, so the primary intake is RGB frames plus paired metric depth maps.

Top-level bucket layout:

```text
gs://gresearch/sanpo_dataset/v0/
  labelmap.json
  labeltype.json
  sanpo-real/
  sanpo-synthetic/
```

`labelmap.json` contains 31 semantic classes. CPE-relevant labels include `crosswalk`, `pedestrian`, `rider`, `animal`, `stairs`, `obstacle`, `vehicle`, `traffic sign`, `traffic light`, `pole`, and `bike rack`.

`labeltype.json` marks whether each class is semantic-only or panoptic. CPE-relevant panoptic labels include `crosswalk`, `pedestrian`, `rider`, `animal`, `stairs`, `obstacle`, `vehicle`, `traffic sign`, `traffic light`, `pole`, `bus stop`, `bike rack`, and `tree`.

## SANPO-Real Layout

`valid_streams.json` currently contains 462 relevant stream IDs. The entries sampled so far all resolve under `sanpo-real`, not `sanpo-synthetic`, and have this shape:

```text
gs://gresearch/sanpo_dataset/v0/sanpo-real/<session_id>/
  description.json
  camera_head/
    camera_poses.csv
    fixed_camera_poses.csv
    left/
      video_frames/
        000000.png
        000001.png
        ...
      depth_maps/
        000000.float16.gz
        000001.float16.gz
        ...
      zed_depth_maps/
        000000.float16.gz
        000001.float16.gz
        ...
    right/
      video_frames/
        000000.png
        000001.png
        ...
  camera_chest/
    camera_poses.csv
    fixed_camera_poses.csv
    left/
      video_frames/
      depth_maps/
      zed_depth_maps/
    right/
      video_frames/
```

For CPE, use `camera_head/left/video_frames` and `camera_head/left/depth_maps` because `valid_streams.json` specifies `camera=head`, `view=left`, and `src/perception_stack/depth_loader.py` supports `.float16.gz` depth maps directly.

`description.json` includes session metadata such as traffic level, environment type, weather, visibility, ego motion, camera intrinsics, image size, stereo transform, and IMU sensor details. Example sampled real stream metadata reported 15 FPS, 2208 x 1242 frames, ZED camera models, and urban/road-junction attributes.

## SANPO-Synthetic Layout

Synthetic sessions use hashed IDs and are organized similarly, but generally include segmentation masks:

```text
gs://gresearch/sanpo_dataset/v0/sanpo-synthetic/<session_id>/
  description.json
  camera_head/
    camera_poses.csv
    left/
      video_frames/
      depth_maps/
      segmentation_masks/
      frame_segmentation_annotation_type.json
```

Synthetic data is useful for segmentation-aware correctness checks or mask-derived pseudo-labels. The current `valid_streams.json` points to real-session IDs, so the immediate real-time edge benchmark should use SANPO-Real.

## Sampled Valid Stream Sizes

These are head/left real-stream counts from the first ten entries in `valid_streams.json`. Sizes include only that camera/view branch.

| Session ID | Frames | RGB size | Depth size | ZED depth size | Recommended use |
|---|---:|---:|---:|---:|---|
| `-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG` | 539 | 1660.6 MB | 456.2 MB | 739.7 MB | Larger urban sample |
| `-PqSDmiEe2pXjmYHgxh4YEBsj0T5LU10` | 609 | 2199.8 MB | 507.9 MB | 1107.4 MB | Larger sample |
| `-aAwyxNyh11M5W0DQ8YPcSfdkbvvctaV` | 439 | 1453.0 MB | 367.9 MB | 849.8 MB | Medium sample |
| `0auMnLmZDf_ZGM3VqzxmYPlwhtls73RF` | 590 | 1513.7 MB | 254.1 MB | 1216.9 MB | Larger sample |
| `0pf0qM0kG_5fZ0HLO2CDDXmfuMG8aiyR` | 190 | 599.4 MB | 144.7 MB | 446.4 MB | Small sample |
| `0xCqEk5hjEvrygxu26MZkieSv45D_gaJ` | 83 | 260.8 MB | 57.8 MB | 175.4 MB | Best first smoke test |
| `0zEhKDk1j7KSuUQYR_rmCLmlbb5FCYG6` | 563 | 1671.1 MB | 345.9 MB | 1033.4 MB | Larger sample |
| `0zlY0PpjObYEKA2MIUXUzaBtrG1EeXGm` | 444 | 1569.0 MB | 471.8 MB | 348.9 MB | Medium sample |
| `11tnj2tdX_ElmqXEv6kwSXI34_ZWXv-N` | 402 | 902.5 MB | 205.4 MB | 291.9 MB | Medium sample |
| `12KBw74zlZrX7OftxToP9-jbVlQNC2MO` | 209 | 599.8 MB | 77.4 MB | 411.2 MB | Small sample |

## Download Strategy

Do not download whole sessions by default. A single valid stream can exceed 2 GB for RGB alone, and the full valid set is much larger than needed for latency testing.

Recommended staged intake:

1. Smoke test: download one small valid stream, `0xCqEk5hjEvrygxu26MZkieSv45D_gaJ`, using only `camera_head/left/video_frames` and `camera_head/left/depth_maps`.
2. Latency test: run `tools/benchmark_edge_realtime.py` with `--depth-dir` against the downloaded frames.
3. Broader robustness test: download 5-10 streams from `valid_streams.json`, preferably selected across different `description.json` environment/traffic/weather metadata.
4. Optional semantic correctness: use synthetic sessions with `segmentation_masks` or real sessions if a segmentation annotation branch is present for the selected stream.

## Local Folder Convention

Downloaded SANPO test data should stay local-only under:

```text
data/sanpo/valid_streams/<session_id>/camera_head/left/
  video_frames/
  depth_maps/
  description.json
  camera_poses.csv
  fixed_camera_poses.csv
```

Benchmark outputs should be saved under:

```text
evaluation/benchmarks/sanpo_edge_realtime/
```

## Edge Benchmark Command Pattern

```bash
.venv/bin/python tools/benchmark_edge_realtime.py \
  --rgb-dir data/sanpo/valid_streams/<session_id>/camera_head/left/video_frames \
  --depth-dir data/sanpo/valid_streams/<session_id>/camera_head/left/depth_maps \
  --weights training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt \
  --device 0 \
  --fps 15 \
  --frame-step 3 \
  --source sanpo \
  --max-frames 300 \
  --out evaluation/benchmarks/sanpo_edge_realtime/<session_id>_v3_edge_realtime.json \
  --csv evaluation/benchmarks/sanpo_edge_realtime/<session_id>_v3_edge_realtime.csv
```

Use `--fps 15` for SANPO-Real streams unless a selected `description.json` says otherwise. The benchmark reports both real-time pass/fail and the `<50 ms` detector/reflex-path budget check.

## Smoke-Test Download and Benchmark Result

A first bounded smoke-test stream was downloaded locally:

```text
data/sanpo/valid_streams/0xCqEk5hjEvrygxu26MZkieSv45D_gaJ/camera_head/left/
```

Downloaded content:

- 83 paired RGB/depth frames from `video_frames/` and `depth_maps/`
- `description.json`
- `camera_head/camera_poses.csv`
- `camera_head/fixed_camera_poses.csv`

The first benchmark run exposed a real integration issue: SANPO-Real RGB frames are `2208 x 1242`, but the selected `depth_maps/*.float16.gz` files decompress to `960 x 960` plus two padding float16 values. The perception stack now infers SANPO depth shape and scales RGB-space boxes into native depth-map coordinates before depth rescoring, median-depth sampling, and unlabeled-obstacle sweeping.

Benchmark outputs:

```text
evaluation/benchmarks/sanpo_edge_realtime/0xCqEk5hjEvrygxu26MZkieSv45D_gaJ_v3_edge_realtime_replay.json
evaluation/benchmarks/sanpo_edge_realtime/0xCqEk5hjEvrygxu26MZkieSv45D_gaJ_v3_edge_realtime_preload.json
```

| Mode | What it measures | Mean total | p95 total | Real-time result | Reflex-budget result |
|---|---|---:|---:|---|---|
| Replay | Local PNG read + gzip depth load + YOLO + depth post-processing | 50.06 ms | 51.54 ms | Passes 15 FPS / `frame_step=3` budget of 200 ms | Slightly over 50 ms due mostly to PNG disk read |
| Preload | In-memory RGB/depth tensors + YOLO + depth post-processing | 6.85 ms | 9.04 ms | Passes | Passes |

Interpretation: replay mode is useful for dataset-pipeline throughput, but live edge readiness should be judged primarily from preloaded mode or from an actual camera/depth sensor feed. In replay mode, RGB PNG decoding/read accounted for roughly 36 ms of the 50 ms frame time; model tracking was roughly 9-10 ms p95.

## Edge Device Simulation Profile

`tools/benchmark_edge_realtime.py` supports `--edge-profile` for pass/fail reporting. The default native profile reports measured GB10 latency; `jetson_orin_nano_8gb` is an analytical proxy that scales measured compute latency and adds sensor/memory overhead.

The canonical profile parameters, hardware status, and physical-device validation requirements are in [`hardware_targets.md`](./hardware_targets.md). Results in this document remain dataset- and benchmark-specific.

Command used for the first SANPO edge simulation:

```bash
.venv/bin/python tools/benchmark_edge_realtime.py \
  --rgb-dir data/sanpo/valid_streams/0xCqEk5hjEvrygxu26MZkieSv45D_gaJ/camera_head/left/video_frames \
  --depth-dir data/sanpo/valid_streams/0xCqEk5hjEvrygxu26MZkieSv45D_gaJ/camera_head/left/depth_maps \
  --weights training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt \
  --device 0 \
  --fps 15 \
  --frame-step 3 \
  --source sanpo \
  --preload \
  --edge-profile jetson_orin_nano_8gb
```

Saved output:

```text
evaluation/benchmarks/sanpo_edge_realtime/0xCqEk5hjEvrygxu26MZkieSv45D_gaJ_v3_jetson_orin_nano_8gb_sim.json
```

| Metric | Native GB10/preloaded | Jetson Orin Nano 8GB simulated |
|---|---:|---:|
| Mean total latency | 7.15 ms | 31.62 ms |
| p95 total latency | 8.49 ms | 36.96 ms |
| Simulated processed FPS | n/a | 31.63 |
| Simulated source-equivalent FPS with `frame_step=3` | n/a | 94.88 |
| Reflex budget pass | Yes | Yes |
| Real-time budget pass | Yes | Yes |

Interpretation: under the current conservative Jetson Orin Nano 8GB proxy, v3 remains within the `<50 ms` detector/reflex-path budget on the downloaded SANPO valid stream. The next stronger validation step is to export v3 to ONNX/TensorRT and run the same benchmark on actual Jetson hardware.

## 10-Session Edge Simulation Result

A broader v3 benchmark was run over 10 bounded SANPO-Real valid streams selected as the smallest planned downloads from the first 80 valid streams scanned. Each session used up to 90 paired RGB/depth frames with `frame_step=3`, so the benchmark processed 30 frames per session and 300 frames total.

Saved outputs:

```text
evaluation/benchmarks/sanpo_edge_realtime/ten_session_v3_jetson_orin_nano_8gb/aggregate_summary.json
evaluation/benchmarks/sanpo_edge_realtime/ten_session_v3_jetson_orin_nano_8gb/aggregate_report.md
```

Aggregate Jetson Orin Nano 8GB simulation result:

| Metric | Value |
|---|---:|
| Sessions | 10 |
| Processed frames | 300 |
| Avg native mean latency | 6.44 ms |
| Avg native p95 latency | 7.83 ms |
| Avg simulated mean latency | 28.77 ms |
| Avg simulated p95 latency | 34.31 ms |
| Worst-session simulated p95 latency | 41.96 ms |
| Avg simulated processed FPS | 35.12 |
| Avg simulated source-equivalent FPS with `frame_step=3` | 105.35 |
| Simulated reflex-budget pass | Yes, all sessions |
| Simulated real-time-budget pass | Yes, all sessions |

Interpretation: under the conservative `jetson_orin_nano_8gb` profile, v3 stays below the `<50 ms p95` detector/reflex budget across all 10 sampled SANPO valid streams.

---

## Gap Analysis Workflow

**Author:** CPE Team | **Date:** 2026-07-10 | **Status:** Notebook ready (`notebooks/sanpo_yolo_gap_analysis.ipynb`).

This section documents the empirical depth-versus-YOLO blind-spot workflow used to identify which
physical objects are consistently present at hazard range but not detected by the current YOLO26n
whitelist — the outputs directly inform which new classes to prioritize in `yolo_training.md`.

### Objective

Run the gap analysis notebook across the SANPO dataset to **empirically identify** which physical
objects are consistently present at hazard range (0.5–6m) but are NOT detected by the current
YOLO26n whitelist.

### What the gap analysis does

For every sampled frame:
1. Run YOLO26n with the current CPE detector configuration.
2. Load the `.float16.gz` SANPO depth map.
3. Find all depth "blobs" in the **0.5–6.0m alert zone** that have **no overlapping YOLO bounding box**.
4. Tag each blob with:
   - **Elevation:** `head_level` (top 45% of frame) | `mid_level` | `foot_level`
   - **Column:** which of 5 vertical navigation columns it occupies
   - **Depth:** nearest and mean distance in metres
5. Generate a 3-panel visualization and save it to the output folder.
6. Export all stats to a CSV for summary analysis.

The frequency of unlabelled blobs across all sessions tells us **what the gap classes are**, and the
elevation distribution tells us **how dangerous they are**.

### Data structure

```
data/sanpo/raw/
└── <session_hash>/
    └── camera_head/
        └── left/
            ├── video_frames/        ← RGB PNGs  (000000.png, 000001.png, ...)
            ├── depth_maps/          ← Depth     (000000.float16.gz, ...)
            └── frame_segmentation_annotation_type.json
```

### Key configuration parameters

All parameters are in **Cell 1** of `notebooks/sanpo_yolo_gap_analysis.ipynb`.

| Parameter | Default | Description |
|---|---|---|
| `SESSION_FILTER` | `None` | Set to a list of session hashes to restrict analysis. `None` = all sessions. |
| `MAX_FRAMES_PER_SESSION` | `30` | **Critical — controls output volume.** Frames are sampled evenly across the session (not just the first N). |
| `DEPTH_MIN_M` | `0.5` | Minimum depth to count as a hazard candidate. |
| `DEPTH_MAX_M` | `6.0` | Maximum depth. Objects beyond 6m are informational only. |
| `MIN_BLOB_AREA_PX` | `800` | Minimum blob pixel area to report. Filters out noise/reflections. |
| `OBSTACLE_GRID_COLS` | `5` | Number of vertical navigation columns to sweep. |
| `YOLO_MODEL_PATH` | `models/yolo/base_yolo26n/yolo26n.pt` | Path to YOLO model (relative to repo root). |

### Output folder management

> **This is important to prevent folder bloat.**

The notebook only saves visualizations for frames **that actually have depth gaps** — frames where
YOLO detected everything get skipped automatically. With `MAX_FRAMES_PER_SESSION = 30` and 3
sessions, worst-case output is **90 PNG files**; typical output will be far fewer.

```
notebooks/gap_analysis_output/
├── <session_hash>_<frame_id>.png    ← 3-panel visualization (saved only for gap frames)
└── gap_analysis_stats.csv          ← Per-frame statistics table
```

| Scenario | `MAX_FRAMES_PER_SESSION` | Expected output files |
|---|---|---|
| Quick check (1 session) | 10 | ~5–10 PNGs |
| Standard analysis (all 3 sessions) | 30 | ~20–60 PNGs |
| Full SANPO sweep (all sessions) | 20 | ~500–700 PNGs (large run) |

### Running the notebook — step by step

Prerequisites:

```bash
pip install ultralytics opencv-python matplotlib scipy pandas
```

1. Open `notebooks/sanpo_yolo_gap_analysis.ipynb`.
2. Configure Cell 1 — for a quick first run, set `MAX_FRAMES_PER_SESSION = 10` and
   `SESSION_FILTER = None`; confirm `YOLO_MODEL_PATH` points to `models/yolo/base_yolo26n/yolo26n.pt`.
3. Run all cells in order (Cells 1–9). Cell 7 (the main loop) prints per-frame progress and displays
   inline visualizations.
4. Review Cell 8 (Summary Statistics) — the most important output: which YOLO classes are detected
   most, head-level vs. foot-level gap counts (severity distribution), and the top 10 gap frames.
5. Use Cell 10 to manually inspect interesting frames by changing `FRAME_INDEX`.

### Reading the 3-panel visualization

```
[ RGB + YOLO boxes (blue) + gap dots ]  |  [ Depth heatmap 0–10m ]  |  [ Gap-only mask ]
```

| Visual element | Meaning |
|---|---|
| Blue rectangle | YOLO detected this object (labelled, tracked) |
| Red filled circle | Depth blob at **head/chest level** — highest priority gap |
| Orange filled circle | Depth blob at **mid/waist level** |
| Green filled circle | Depth blob at **foot level** — trip hazard |
| White ring | Outline of any gap blob |
| Label text | Elevation tag + mean depth in metres |

### Interpreting the gap statistics CSV

`gap_analysis_stats.csv` columns:

| Column | Description |
|---|---|
| `session` | Session hash (first 20 chars) |
| `frame` | Frame index |
| `yolo_detections` | Number of YOLO detections in this frame |
| `yolo_classes` | Pipe-separated list of detected class names |
| `depth_gaps_total` | Total unlabelled depth blobs found |
| `gaps_head_level` | Count of head-level gaps |
| `gaps_mid_level` | Count of mid-level gaps |
| `gaps_foot_level` | Count of foot-level gaps |
| `nearest_gap_m` | Distance to the nearest unlabelled obstacle (metres) |
| `largest_gap_area` | Pixel area of the largest gap blob |

Analysis queries to run in pandas after the run:

```python
import pandas as pd
df = pd.read_csv("notebooks/gap_analysis_output/gap_analysis_stats.csv")

print((df.depth_gaps_total > 0).mean())        # fraction of frames with significant gaps
print(df.gaps_head_level.mean())               # avg head-level gaps per frame
urgent = df[df.nearest_gap_m < 2.0]             # frames with a gap closer than 2m
print(urgent.shape[0], "urgent frames")
```

### What to do with the results

1. Export the CSV and sort by `depth_gaps_total` to find the most gap-heavy sessions.
2. Visually inspect the top 20 gap frames (saved as PNGs) and note what objects appear in the gap
   regions — poles/bollards, benches, vegetation/clutter, construction barriers, stairs/kerbs.
3. Feed observations back into `yolo_training.md` — e.g. if 70% of gap frames show a pole in that
   region, `pole` is the highest-priority new class.
4. Update `CHANGELOG.md` once the analysis run is complete and the gap class list is finalized.

### Open questions

1. Should more SANPO sessions be downloaded for a more statistically robust gap analysis, beyond the
   local `data/sanpo/raw/` sample?
2. Should the gap analysis also run on real SANPO sessions (not just synthetic) — real sessions may
   show different gap distributions (e.g. more construction hazards)?
3. SANPO's panoptic segmentation masks (`segmentation_masks/`) could cross-reference consistent gap
   blobs to recover the ground-truth class of the unlabelled blob — a strong potential paper result
   if pursued.
