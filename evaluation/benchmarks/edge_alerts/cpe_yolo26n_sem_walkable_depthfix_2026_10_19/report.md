# Edge-ahead alerts - cpe_yolo26n_sem_walkable_depthfix_2026_10_19

53 sessions with head-camera masks, 26.6 min at 2.5 Hz; class maps from `training/runs/cpe_yolo26n_sem_walkable/weights/best.pt`. Rule: announce ['curb', 'stairs', 'road'] within 5 m, seen in 2 consecutive frames, one alert per edge episode (2 s clear). Step-off: the ground just ahead turns from walkable to an announced kind after 2 walkable frames and stays off walkable ground for 1 frame(s), on SANPO masks. 95% CIs: session bootstrap.

| alerts / min | step-offs | warned (alert ≤ 10 s before) | median lead (s) | alerts followed by a step-off |
|---|---:|---|---|---|
| 0.56 [0.31, 0.87] | 45 | 0.20 [0.08, 0.36] | 1.2 [0.4, 3.6] | 0.60 [0.30, 0.86] |

| Kind | alerts | alerts / min | followed | median distance at alert (m) | step-offs | warned |
|---|---:|---:|---:|---:|---:|---:|
| curb | 7 | 0.26 | 0.29 | 3.4 | 0 | — |
| stairs | 3 | 0.11 | 0.67 | 3.9 | 3 | 0.67 |
| road | 5 | 0.19 | 1.00 | 3.6 | 42 | 0.17 |

| Official split | sessions | alerts / min | step-offs | warned | followed |
|---|---:|---:|---:|---:|---:|
| train | 43 | 0.51 | 38 | 0.21 | 0.64 |
| test | 10 | 0.80 | 7 | 0.14 | 0.50 |

An alert not followed by a step-off is not necessarily false: the walker may have stopped, turned or walked along the edge. Warned is per step-off; a step-off with no alert before it is a miss.
