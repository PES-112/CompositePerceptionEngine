# Scene-context flags vs the arrival label - model_val_test_headmasked_2026_10_18

1561 (frame, object) observations in 15 sessions ({'train': 0, 'val': 5, 'test': 10}); context from `data/sanpo_scene_context/cpe_yolo26n_sem_walkable_headmasked`. Label: in the walking corridor within the horizon (base rate 0.176). 95% CIs: session bootstrap. Risk ratio = P(arrives | flag) / P(arrives | not flag); adjusted = Mantel-Haenszel over distance bins [0.0, 3.0, 6.0, 10.0] m.

| Flag | declared × | share of objects | P(arrive), flag | P(arrive), not | RR | RR, distance-adjusted |
|---|---:|---:|---|---|---|---|
| in_path | 1.5 | 0.68 | 0.248 [0.004, 0.394] | 0.024 [0.000, 0.073] | 10.47 [1.12, 42.11] | 2.68 [2.20, 10.34] |
| near_boundary | 1 | 0.31 | 0.086 [0.000, 0.253] | 0.217 [0.002, 0.344] | 0.39 [0.03, 1.25] | 1.15 [0.14, 5.97] |
| other_surface | 0.5 | 0.26 | 0.017 [0.000, 0.063] | 0.233 [0.005, 0.375] | 0.07 [0.00, 0.18] | 0.37 [0.00, 0.44] |
| approaching | — | 0.23 | 0.130 [0.000, 0.278] | 0.189 [0.004, 0.332] | 0.69 [0.00, 0.86] | 0.74 [0.00, 0.87] |
| near_boundary_approaching | — | 0.07 | 0.036 [0.000, 0.162] | 0.186 [0.003, 0.329] | 0.20 [0.00, 0.52] | 0.99 [0.00, 1.57] |

Approaching: closing on the walker's centre line at >= 0.3 m/s over the last 3 steps (motion only, no segmentation). Near a boundary and approaching, adjusted over distance x approaching strata (does the boundary add anything once motion is known?): 1.49 [0.00, 2.04].

| Surface under the object | share | n | P(arrive) |
|---|---:|---:|---|
| none (no ground at its depth) | 0.04 | 70 | 0.057 [0.000, 0.133] |
| walkable | 0.66 | 1036 | 0.252 [0.006, 0.396] |
| crosswalk | 0.01 | 20 | 0.100 [0.000, 0.500] |
| road | 0.02 | 29 | 0.000 [0.000, 0.000] |
| curb | 0.00 | 3 | 0.000 [0.000, 0.000] |
| stairs | 0.00 | 1 | 0.000 [0.000, 0.000] |
| non-walkable ground | 0.26 | 402 | 0.017 [0.000, 0.067] |

A declared multiplier above 1 is supported if its RR's CI is above 1, below 1 if below 1. The multipliers are not fitted to these numbers; fitting them would need train sessions only.
