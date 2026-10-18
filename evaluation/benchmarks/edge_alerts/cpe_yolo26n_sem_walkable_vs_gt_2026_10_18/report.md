# Edge-ahead alerts - cpe_yolo26n_sem_walkable_vs_gt_2026_10_18

53 sessions with head-camera masks, 26.6 min at 2.5 Hz; class maps from `training/runs/cpe_yolo26n_sem_walkable/weights/best.pt`. Rule: announce ['curb', 'stairs', 'road'] within 5 m, seen in 2 consecutive frames, one alert per edge episode (2 s clear). Step-off: the ground just ahead turns from walkable to an announced kind after 2 walkable frames and stays off walkable ground for 1 frame(s), on SANPO masks. 95% CIs: session bootstrap.

| alerts / min | step-offs | warned (alert ≤ 10 s before) | median lead (s) | alerts followed by a step-off |
|---|---:|---|---|---|
| 0.83 [0.49, 1.22] | 45 | 0.22 [0.10, 0.37] | 2.0 [0.4, 4.0] | 0.45 [0.24, 0.69] |

| Kind | alerts | alerts / min | followed | median distance at alert (m) | step-offs | warned |
|---|---:|---:|---:|---:|---:|---:|
| curb | 9 | 0.34 | 0.22 | 3.6 | 0 | — |
| stairs | 4 | 0.15 | 0.50 | 2.7 | 3 | 0.67 |
| road | 9 | 0.34 | 0.67 | 3.1 | 42 | 0.19 |

| Official split | sessions | alerts / min | step-offs | warned | followed |
|---|---:|---:|---:|---:|---:|
| train | 43 | 0.79 | 38 | 0.24 | 0.47 |
| test | 10 | 1.00 | 7 | 0.14 | 0.40 |

An alert not followed by a step-off is not necessarily false: the walker may have stopped, turned or walked along the edge. Warned is per step-off; a step-off with no alert before it is a miss.
