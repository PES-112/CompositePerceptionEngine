# Hardware Targets and Runtime Budgets

This file is the source of truth for the hardware used to train and benchmark CPE, the intended edge target, and which results are measured versus simulated. Values marked as observed were queried from the current SSH host on 2026-07-22.

## Training and Development Host

| Item | Observed specification |
|---|---|
| Platform | NVIDIA GB10 system, ARM64 (`aarch64`) |
| GPU | NVIDIA GB10, CUDA available |
| CPU | 20 cores: 10 Cortex-X925 up to 4.004 GHz and 10 Cortex-A725 up to 2.860 GHz |
| Unified/system memory | 119 GiB visible to Linux; approximately 120 GB advertised pool |
| Swap | 15 GiB |
| Operating system | Ubuntu 24.04.3 LTS, Linux 6.11.0-1014-nvidia |
| NVIDIA driver | 580.82.09 |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime reported by PyTorch | 13.0 |
| Ultralytics | 8.4.95 |

This host is used for YOLO26n fine-tuning, held-out evaluation, SANPO replay, and native latency measurement. The training profile uses RAM dataset caching, AutoBatch, 16 data-loader workers, GPU device 0, and deferred export. Unified memory does not imply that every workload can use all 119 GiB as GPU memory; batch size must still be validated by AutoBatch or a short probe.

## Edge Target Status

The current planning target is a Jetson Orin Nano 8GB-class NVIDIA edge device, but no physical edge device has been benchmarked yet. TensorRT engines are hardware-specific and should be built and measured on the actual deployment device.

| Item | Current status |
|---|---|
| Target class | Jetson Orin Nano 8GB |
| Physical device result | Not available |
| Detector runtime target | TensorRT or ONNX FP16/INT8 |
| Input assumption | SANPO RGB plus paired metric depth, 15 FPS for current real-stream samples |
| Frame policy | Process every third frame (`frame_step=3`) |
| Reflex-path budget | p95 below 50 ms |
| Cognitive-path budget | response below 500 ms, with stale-response rejection |

The earlier Snapdragon 8 Gen 3 section in `architecture.md` was a design possibility, not a confirmed deployment target. It has been removed from the canonical architecture to avoid mixing two unvalidated hardware targets.

## Jetson Simulation Profile

`tools/benchmark_edge_realtime.py` defines `jetson_orin_nano_8gb` as an analytical proxy:

| Parameter | Value |
|---|---|
| Source measurement | Native GB10 PyTorch latency |
| Compute latency scale | 4.0x |
| Added sensor/memory overhead | 3.0 ms per processed frame |
| Reflex budget | 50 ms |
| Cognitive budget | 500 ms |
| Recommended runtime | ONNX or TensorRT FP16/INT8 |

This profile is useful for early pass/fail screening, but it is not an emulator and cannot predict thermal throttling, power draw, TensorRT kernel behavior, camera-copy overhead, or shared-memory contention on a real Jetson.

## Current Benchmark Evidence

The 10-session SANPO-Real benchmark processed 300 RGB+depth frames with `frame_step=3`:

| Metric | Result |
|---|---:|
| Average native GB10 p95 | 7.83 ms |
| Worst native GB10 p95 | 9.74 ms |
| Average simulated Jetson p95 | 34.31 ms |
| Worst simulated Jetson p95 | 41.96 ms |
| Simulated sessions passing the 50 ms reflex budget | 10/10 |

The machine-readable aggregate is stored at `evaluation/benchmarks/sanpo_edge_realtime/ten_session_v3_jetson_orin_nano_8gb/aggregate_summary.json`.

## Required Physical Edge Validation

Before claiming real edge performance:

1. Export YOLO26n v3 to ONNX FP16 and build a TensorRT engine on the target device.
2. Replay the same 10 SANPO sessions from preloaded or sensor-memory frames.
3. Report warm-up separately, then measure mean, median, p95, p99, throughput, peak memory, temperature, and power mode.
4. Benchmark the detector/reflex path under concurrent SLM and audio load.
5. Re-run after a sustained workload to expose thermal throttling.

Do not describe simulated Jetson numbers as measurements from an edge device in papers, demos, or reports.
