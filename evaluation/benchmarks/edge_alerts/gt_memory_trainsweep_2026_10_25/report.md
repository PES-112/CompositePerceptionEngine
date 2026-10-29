# Edge-ahead alerts - gt_memory_trainsweep_2026_10_25

53 sessions with head-camera masks, 26.6 min at 2.5 Hz; class maps from `gt`. Rule (edge memory): announce ['curb', 'stairs', 'road'] seen 3 time(s) at the same ground-plane spot (camera pose), once it is on the walking line within 5 m or 4 s at walking speed, even out of view; at most one alert per 2 s. Step-off: the ground just ahead turns from walkable to an announced kind after 2 walkable frames and stays off walkable ground for 1 frame(s), on SANPO masks. 95% CIs: session bootstrap.

| alerts / min | step-offs | warned (alert ≤ 10 s before) | median lead (s) | alerts followed by a step-off |
|---|---:|---|---|---|
| 0.68 [0.37, 1.03] | 45 | 0.16 [0.05, 0.30] | 1.2 [0.4, 6.4] | 0.33 [0.11, 0.57] |

| Kind | alerts | alerts / min | followed | median distance at alert (m) | step-offs | warned |
|---|---:|---:|---:|---:|---:|---:|
| curb | 5 | 0.19 | 0.00 | 4.1 | 0 | — |
| stairs | 5 | 0.19 | 0.60 | 4.8 | 3 | 1.00 |
| road | 8 | 0.30 | 0.38 | 4.7 | 42 | 0.10 |

| Official split | sessions | alerts / min | step-offs | warned | followed |
|---|---:|---:|---:|---:|---:|
| train | 43 | 0.74 | 38 | 0.16 | 0.31 |
| test | 10 | 0.40 | 7 | 0.14 | 0.50 |

An alert not followed by a step-off is not necessarily false: the walker may have stopped, turned or walked along the edge. Warned is per step-off; a step-off with no alert before it is a miss.
