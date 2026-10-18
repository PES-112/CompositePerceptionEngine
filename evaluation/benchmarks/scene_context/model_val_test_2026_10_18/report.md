# Scene-context flags vs the arrival label - model_val_test_2026_10_18

6748 (frame, object) observations in 112 sessions ({'train': 0, 'val': 41, 'test': 71}); context from `data/sanpo_scene_context/cpe_yolo26n_sem_walkable`. Label: in the walking corridor within the horizon (base rate 0.146). 95% CIs: session bootstrap. Risk ratio = P(arrives | flag) / P(arrives | not flag); adjusted = Mantel-Haenszel over distance bins [0.0, 3.0, 6.0, 10.0] m.

| Flag | declared × | share of objects | P(arrive), flag | P(arrive), not | RR | RR, distance-adjusted |
|---|---:|---:|---|---|---|---|
| in_path | 1.5 | 0.65 | 0.171 [0.063, 0.253] | 0.100 [0.047, 0.159] | 1.71 [0.77, 3.23] | 1.12 [0.78, 1.55] |
| near_boundary | 1 | 0.29 | 0.086 [0.032, 0.150] | 0.171 [0.074, 0.242] | 0.50 [0.25, 0.97] | 1.04 [0.61, 1.58] |
| other_surface | 0.5 | 0.22 | 0.054 [0.016, 0.110] | 0.173 [0.079, 0.246] | 0.31 [0.13, 0.69] | 0.89 [0.40, 1.69] |
| approaching | — | 0.24 | 0.140 [0.063, 0.212] | 0.148 [0.068, 0.214] | 0.95 [0.78, 1.18] | 0.89 [0.78, 1.04] |
| near_boundary_approaching | — | 0.07 | 0.096 [0.029, 0.197] | 0.150 [0.067, 0.217] | 0.64 [0.28, 1.50] | 1.17 [0.61, 1.79] |

Approaching: closing on the walker's centre line at >= 0.3 m/s over the last 3 steps (motion only, no segmentation). Near a boundary and approaching, adjusted over distance x approaching strata (does the boundary add anything once motion is known?): 1.28 [0.61, 2.15].

| Surface under the object | share | n | P(arrive) |
|---|---:|---:|---|
| none (no ground at its depth) | 0.06 | 372 | 0.183 [0.067, 0.267] |
| walkable | 0.64 | 4339 | 0.176 [0.066, 0.259] |
| crosswalk | 0.02 | 124 | 0.105 [0.019, 0.214] |
| road | 0.07 | 441 | 0.125 [0.019, 0.260] |
| curb | 0.00 | 6 | 0.167 [0.000, 1.000] |
| stairs | 0.01 | 78 | 0.538 [0.111, 0.762] |
| non-walkable ground | 0.21 | 1388 | 0.032 [0.009, 0.060] |

A declared multiplier above 1 is supported if its RR's CI is above 1, below 1 if below 1. The multipliers are not fitted to these numbers; fitting them would need train sessions only.
