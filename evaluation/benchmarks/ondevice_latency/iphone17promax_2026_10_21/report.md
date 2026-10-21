# On-device latency (CPE Timing app)

## `cpe_timing_benchmark_2026-10-21T12-09-47Z.csv`

`device=iPhone18,2 os=Version 26.6.2 (Build 23G90) detector=Ultralytics YOLO26n model trained on /home/student-4/Projects/CompositePerceptionEngine/training/configs/cpe_hazard_classes.yaml gru=CPE arrival GRU (corridor_h8_seed0): P(arrival slice | history), 4 slices over 20 steps of 0.2 s + never frames=300 mode=benchmark`

| stage | run | n | median ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|---:|
| pre | benchmark | 1500 | 24.30 | 27.56 | 29.88 |
| det | benchmark | 1500 | 5.49 | 5.70 | 5.86 |
| det_post | benchmark | 1500 | 2.07 | 2.39 | 2.53 |
| gru | benchmark | 1500 | 0.48 | 0.53 | 0.55 |
| total | benchmark | 1500 | 32.26 | 35.78 | 37.90 |

## `cpe_timing_benchmark_2026-10-21T12-12-48Z.csv`

`device=iPhone18,2 os=Version 26.6.2 (Build 23G90) detector=Ultralytics YOLO26n model trained on /home/student-4/Projects/CompositePerceptionEngine/training/configs/cpe_hazard_classes.yaml gru=CPE arrival GRU (corridor_h8_seed0): P(arrival slice | history), 4 slices over 20 steps of 0.2 s + never frames=300 mode=benchmark`

| stage | run | n | median ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|---:|
| pre | benchmark | 1500 | 24.02 | 27.60 | 29.70 |
| det | benchmark | 1500 | 5.52 | 5.89 | 6.11 |
| det_post | benchmark | 1500 | 2.21 | 2.59 | 2.71 |
| gru | benchmark | 1500 | 0.48 | 0.57 | 0.61 |
| total | benchmark | 1500 | 32.28 | 35.89 | 38.07 |

## `cpe_timing_sustained30min_2026-10-21T12-15-39Z.csv`

`device=iPhone18,2 os=Version 26.6.2 (Build 23G90) detector=Ultralytics YOLO26n model trained on /home/student-4/Projects/CompositePerceptionEngine/training/configs/cpe_hazard_classes.yaml gru=CPE arrival GRU (corridor_h8_seed0): P(arrival slice | history), 4 slices over 20 steps of 0.2 s + never frames=300 mode=sustained30min`

| stage | run | n | median ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|---:|
| pre | sustained | 27000 | 24.10 | 28.91 | 30.23 |
| det | sustained | 27000 | 5.47 | 5.73 | 5.96 |
| det_post | sustained | 27000 | 2.01 | 2.41 | 2.62 |
| gru | sustained | 27000 | 0.47 | 0.53 | 0.58 |
| total | sustained | 27000 | 31.95 | 36.87 | 38.25 |

### Sustained run, by 5-minute window

| window (min) | n | median total ms | p95 total ms | thermal (share of frames) | battery |
|---|---:|---:|---:|---|---:|
| 0-5 | 4500 | 32.18 | 36.46 | nominal 100% | 0.85 |
| 5-10 | 4500 | 32.04 | 37.04 | nominal 100% | 0.85 |
| 10-15 | 4500 | 31.73 | 37.27 | nominal 100% | 0.85 |
| 15-20 | 4500 | 31.79 | 37.09 | nominal 100% | 0.85 |
| 20-25 | 4500 | 31.81 | 36.95 | nominal 100% | 0.85 |
| 25-30 | 4500 | 31.95 | 36.28 | nominal 100% | 0.85 |

Battery: 0.85 -> 0.85 (0% per hour).
