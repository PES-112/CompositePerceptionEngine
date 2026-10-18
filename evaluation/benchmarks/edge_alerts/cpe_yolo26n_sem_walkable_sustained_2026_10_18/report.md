# Edge-ahead alerts - cpe_yolo26n_sem_walkable_sustained_2026_10_18

53 sessions with head-camera masks, 26.6 min at 2.5 Hz; class maps from `training/runs/cpe_yolo26n_sem_walkable/weights/best.pt`. Rule: announce ['curb', 'stairs', 'road'] within 5 m, seen in 2 consecutive frames, one alert per edge episode (2 s clear). Step-off: the ground just ahead turns from walkable to an announced kind after 2 walkable frames and stays off walkable ground for 2 frame(s), on the same class maps as the alerts (self-referenced). 95% CIs: session bootstrap.

| alerts / min | step-offs | warned (alert ≤ 10 s before) | median lead (s) | alerts followed by a step-off |
|---|---:|---|---|---|
| 0.83 [0.49, 1.22] | 24 | 0.17 [0.00, 0.39] | 2.0 [1.6, 2.8] | 0.18 [0.00, 0.38] |

| Kind | alerts | alerts / min | followed | median distance at alert (m) | step-offs | warned |
|---|---:|---:|---:|---:|---:|---:|
| curb | 9 | 0.34 | 0.00 | 3.6 | 0 | — |
| stairs | 4 | 0.15 | 0.50 | 2.7 | 3 | 0.67 |
| road | 9 | 0.34 | 0.22 | 3.1 | 21 | 0.10 |

| Official split | sessions | alerts / min | step-offs | warned | followed |
|---|---:|---:|---:|---:|---:|
| train | 43 | 0.79 | 19 | 0.21 | 0.24 |
| test | 10 | 1.00 | 5 | 0.00 | 0.00 |

An alert not followed by a step-off is not necessarily false: the walker may have stopped, turned or walked along the edge. Warned is per step-off; a step-off with no alert before it is a miss.
