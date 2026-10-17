# Edge-ahead alerts - gt_2026_10_17

53 sessions with head-camera masks, 26.6 min at 2.5 Hz; class maps from `gt`. Rule: announce ['curb', 'stairs', 'road'] within 5 m, seen in 2 consecutive frames, one alert per edge episode (2 s clear). Step-off: the ground just ahead turns from walkable to an announced kind after 2 walkable frames and stays off walkable ground for 1 frame(s). 95% CIs: session bootstrap.

| alerts / min | step-offs | warned (alert ≤ 10 s before) | median lead (s) | alerts followed by a step-off |
|---|---:|---|---|---|
| 0.79 [0.49, 1.12] | 45 | 0.29 [0.13, 0.47] | 1.2 [0.4, 2.8] | 0.48 [0.24, 0.69] |

| Kind | alerts | alerts / min | followed | median distance at alert (m) | step-offs | warned |
|---|---:|---:|---:|---:|---:|---:|
| curb | 8 | 0.30 | 0.12 | 3.7 | 0 | — |
| stairs | 3 | 0.11 | 0.67 | 3.6 | 3 | 1.00 |
| road | 10 | 0.38 | 0.70 | 2.7 | 42 | 0.24 |

| Official split | sessions | alerts / min | step-offs | warned | followed |
|---|---:|---:|---:|---:|---:|
| train | 43 | 0.79 | 38 | 0.26 | 0.47 |
| test | 10 | 0.80 | 7 | 0.43 | 0.50 |

An alert not followed by a step-off is not necessarily false: the walker may have stopped, turned or walked along the edge. Warned is per step-off; a step-off with no alert before it is a miss.
