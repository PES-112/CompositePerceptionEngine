# Edge-ahead alerts - gt_sustained_2026_10_17

53 sessions with head-camera masks, 26.6 min at 2.5 Hz; class maps from `gt`. Rule: announce ['curb', 'stairs', 'road'] within 5 m, seen in 2 consecutive frames, one alert per edge episode (2 s clear). Step-off: the ground just ahead turns from walkable to an announced kind after 2 walkable frames and stays off walkable ground for 2 frame(s). 95% CIs: session bootstrap.

| alerts / min | step-offs | warned (alert ≤ 10 s before) | median lead (s) | alerts followed by a step-off |
|---|---:|---|---|---|
| 0.79 [0.49, 1.12] | 26 | 0.35 [0.14, 0.59] | 1.6 [0.8, 6.0] | 0.33 [0.12, 0.52] |

| Kind | alerts | alerts / min | followed | median distance at alert (m) | step-offs | warned |
|---|---:|---:|---:|---:|---:|---:|
| curb | 8 | 0.30 | 0.00 | 3.7 | 0 | — |
| stairs | 3 | 0.11 | 0.67 | 3.6 | 3 | 1.00 |
| road | 10 | 0.38 | 0.50 | 2.7 | 23 | 0.26 |

| Official split | sessions | alerts / min | step-offs | warned | followed |
|---|---:|---:|---:|---:|---:|
| train | 43 | 0.79 | 22 | 0.32 | 0.35 |
| test | 10 | 0.80 | 4 | 0.50 | 0.25 |

An alert not followed by a step-off is not necessarily false: the walker may have stopped, turned or walked along the edge. Warned is per step-off; a step-off with no alert before it is a miss.
