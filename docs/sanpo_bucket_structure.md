# SANPO GCS Bucket Structure and Edge Test Intake

This note documents the SANPO public Google Cloud Storage layout used by CPE for real-time edge testing with `valid_streams.json`.

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

`tools/benchmark_edge_realtime.py` now supports `--edge-profile` for pass/fail reporting against a target edge profile. The default native profile reports measured GB10 latency. The current default edge simulation target is:

```text
jetson_orin_nano_8gb
```

This profile is intentionally conservative and analytical rather than pretending the GB10 is physically the edge device:

- target device: Jetson Orin Nano 8GB
- measured-compute slowdown: `4.0x`
- per-processed-frame sensor/memory overhead: `3.0 ms`
- reflex budget: `50 ms p95`
- cognitive budget: `500 ms`
- recommended deployment runtime: ONNX/TensorRT FP16 or INT8

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
