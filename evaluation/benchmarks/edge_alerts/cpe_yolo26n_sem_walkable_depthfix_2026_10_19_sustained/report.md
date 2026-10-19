# Edge-ahead alerts - cpe_yolo26n_sem_walkable_depthfix_2026_10_19_sustained

53 sessions with head-camera masks, 26.6 min at 2.5 Hz; class maps from `training/runs/cpe_yolo26n_sem_walkable/weights/best.pt`. Rule: announce ['curb', 'stairs', 'road'] within 5 m, seen in 2 consecutive frames, one alert per edge episode (2 s clear). Step-off: the ground just ahead turns from walkable to an announced kind after 2 walkable frames and stays off walkable ground for 2 frame(s), on SANPO masks. 95% CIs: session bootstrap.

| alerts / min | step-offs | warned (alert ≤ 10 s before) | median lead (s) | alerts followed by a step-off |
|---|---:|---|---|---|
| 0.56 [0.31, 0.87] | 26 | 0.23 [0.06, 0.45] | 1.2 [1.2, 8.0] | 0.40 [0.10, 0.68] |

| Kind | alerts | alerts / min | followed | median distance at alert (m) | step-offs | warned |
|---|---:|---:|---:|---:|---:|---:|
| curb | 7 | 0.26 | 0.00 | 3.4 | 0 | — |
| stairs | 3 | 0.11 | 0.67 | 3.9 | 3 | 0.67 |
| road | 5 | 0.19 | 0.80 | 3.6 | 23 | 0.17 |

| Official split | sessions | alerts / min | step-offs | warned | followed |
|---|---:|---:|---:|---:|---:|
| train | 43 | 0.51 | 22 | 0.27 | 0.55 |
| test | 10 | 0.80 | 4 | 0.00 | 0.00 |

An alert not followed by a step-off is not necessarily false: the walker may have stopped, turned or walked along the edge. Warned is per step-off; a step-off with no alert before it is a miss.
