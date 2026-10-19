# Edge-ahead alerts - gt_depthfix_trainsweep_2026_10_19

53 sessions with head-camera masks, 26.6 min at 2.5 Hz; class maps from `gt`. Rule: announce ['curb', 'stairs', 'road'] within 5 m, seen in 2 consecutive frames, one alert per edge episode (2 s clear). Step-off: the ground just ahead turns from walkable to an announced kind after 2 walkable frames and stays off walkable ground for 1 frame(s), on SANPO masks. 95% CIs: session bootstrap.

| alerts / min | step-offs | warned (alert ≤ 10 s before) | median lead (s) | alerts followed by a step-off |
|---|---:|---|---|---|
| 0.68 [0.40, 1.01] | 45 | 0.18 [0.06, 0.33] | 0.8 [0.4, 1.6] | 0.33 [0.13, 0.54] |

| Kind | alerts | alerts / min | followed | median distance at alert (m) | step-offs | warned |
|---|---:|---:|---:|---:|---:|---:|
| curb | 8 | 0.30 | 0.00 | 3.8 | 0 | — |
| stairs | 3 | 0.11 | 0.67 | 4.2 | 3 | 1.00 |
| road | 7 | 0.26 | 0.57 | 4.0 | 42 | 0.12 |

| Official split | sessions | alerts / min | step-offs | warned | followed |
|---|---:|---:|---:|---:|---:|
| train | 43 | 0.70 | 38 | 0.18 | 0.33 |
| test | 10 | 0.60 | 7 | 0.14 | 0.33 |

An alert not followed by a step-off is not necessarily false: the walker may have stopped, turned or walked along the edge. Warned is per step-off; a step-off with no alert before it is a miss.
