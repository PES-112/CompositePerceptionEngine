# Scene-context flags vs the arrival label - model_headfx_val_test_2026_10_20

10794 (frame, object) observations in 133 sessions ({'train': 0, 'val': 48, 'test': 85}); context from `data/sanpo_scene_context_v2/cpe_yolo26n_sem_walkable`. Label: in the walking corridor within the horizon (base rate 0.150). 95% CIs: session bootstrap. Risk ratio = P(arrives | flag) / P(arrives | not flag); adjusted = Mantel-Haenszel over distance bins [0.0, 3.0, 6.0, 10.0] m.

| Flag | declared × | share of objects | P(arrive), flag | P(arrive), not | RR | RR, distance-adjusted |
|---|---:|---:|---|---|---|---|
| in_path | 1.5 | 0.66 | 0.148 [0.066, 0.238] | 0.154 [0.082, 0.239] | 0.96 [0.65, 1.23] | 1.11 [0.82, 1.30] |
| near_boundary | 1 | 0.19 | 0.055 [0.028, 0.094] | 0.172 [0.083, 0.263] | 0.32 [0.19, 0.55] | 0.59 [0.39, 0.82] |
| other_surface | 0.5 | 0.15 | 0.083 [0.041, 0.146] | 0.162 [0.077, 0.252] | 0.51 [0.30, 0.93] | 0.86 [0.61, 1.25] |
| approaching | — | 0.29 | 0.128 [0.067, 0.197] | 0.159 [0.077, 0.248] | 0.80 [0.68, 0.97] | 0.78 [0.68, 0.91] |
| near_boundary_approaching | — | 0.06 | 0.058 [0.029, 0.089] | 0.156 [0.076, 0.242] | 0.37 [0.20, 0.73] | 0.59 [0.36, 0.92] |

Approaching: closing on the walker's centre line at >= 0.3 m/s over the last 3 steps (motion only, no segmentation). Near a boundary and approaching, adjusted over distance x approaching strata (does the boundary add anything once motion is known?): 0.69 [0.44, 1.07].

| Surface under the object | share | n | P(arrive) |
|---|---:|---:|---|
| none (no ground at its depth) | 0.12 | 1285 | 0.275 [0.127, 0.407] |
| walkable | 0.66 | 7094 | 0.150 [0.068, 0.240] |
| crosswalk | 0.02 | 220 | 0.077 [0.014, 0.151] |
| road | 0.12 | 1248 | 0.046 [0.020, 0.077] |
| curb | 0.00 | 40 | 0.025 [0.000, 0.103] |
| stairs | 0.01 | 86 | 0.302 [0.027, 0.465] |
| non-walkable ground | 0.08 | 821 | 0.124 [0.054, 0.223] |

A declared multiplier above 1 is supported if its RR's CI is above 1, below 1 if below 1. The multipliers are not fitted to these numbers; fitting them would need train sessions only.
