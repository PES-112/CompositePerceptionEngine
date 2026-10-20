# Scene-context flags vs the arrival label - model_headfx_val_test_headmasked_2026_10_20

1753 (frame, object) observations in 18 sessions ({'train': 0, 'val': 8, 'test': 10}); context from `data/sanpo_scene_context_v2/cpe_yolo26n_sem_walkable_headmasked`. Label: in the walking corridor within the horizon (base rate 0.220). 95% CIs: session bootstrap. Risk ratio = P(arrives | flag) / P(arrives | not flag); adjusted = Mantel-Haenszel over distance bins [0.0, 3.0, 6.0, 10.0] m.

| Flag | declared × | share of objects | P(arrive), flag | P(arrive), not | RR | RR, distance-adjusted |
|---|---:|---:|---|---|---|---|
| in_path | 1.5 | 0.73 | 0.224 [0.008, 0.437] | 0.209 [0.010, 0.458] | 1.07 [0.31, 1.75] | 1.19 [0.46, 1.76] |
| near_boundary | 1 | 0.20 | 0.097 [0.003, 0.249] | 0.251 [0.011, 0.473] | 0.39 [0.23, 1.74] | 0.75 [0.38, 2.06] |
| other_surface | 0.5 | 0.14 | 0.183 [0.000, 0.413] | 0.226 [0.009, 0.441] | 0.81 [0.00, 3.54] | 0.85 [0.00, 2.94] |
| approaching | — | 0.28 | 0.210 [0.013, 0.433] | 0.224 [0.007, 0.441] | 0.94 [0.66, 2.87] | 0.96 [0.48, 2.35] |
| near_boundary_approaching | — | 0.05 | 0.073 [0.000, 0.208] | 0.227 [0.010, 0.446] | 0.32 [0.00, 2.27] | 0.60 [0.00, 2.16] |

Approaching: closing on the walker's centre line at >= 0.3 m/s over the last 3 steps (motion only, no segmentation). Near a boundary and approaching, adjusted over distance x approaching strata (does the boundary add anything once motion is known?): 0.61 [0.00, 2.46].

| Surface under the object | share | n | P(arrive) |
|---|---:|---:|---|
| none (no ground at its depth) | 0.09 | 154 | 0.338 [0.000, 0.623] |
| walkable | 0.72 | 1255 | 0.227 [0.008, 0.447] |
| crosswalk | 0.02 | 37 | 0.054 [0.000, 0.571] |
| road | 0.07 | 125 | 0.032 [0.000, 0.133] |
| curb | 0.00 | 8 | 0.000 [0.000, 0.000] |
| stairs | 0.00 | 2 | 0.500 [0.000, 1.000] |
| non-walkable ground | 0.10 | 172 | 0.244 [0.000, 0.509] |

A declared multiplier above 1 is supported if its RR's CI is above 1, below 1 if below 1. The multipliers are not fitted to these numbers; fitting them would need train sessions only.
