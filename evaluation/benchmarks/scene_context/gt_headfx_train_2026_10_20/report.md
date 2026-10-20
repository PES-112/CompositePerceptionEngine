# Scene-context flags vs the arrival label - gt_headfx_train_2026_10_20

4145 (frame, object) observations in 35 sessions ({'train': 35, 'val': 0, 'test': 0}); context from `data/sanpo_scene_context_v2/gt`. Label: in the walking corridor within the horizon (base rate 0.151). 95% CIs: session bootstrap. Risk ratio = P(arrives | flag) / P(arrives | not flag); adjusted = Mantel-Haenszel over distance bins [0.0, 3.0, 6.0, 10.0] m.

| Flag | declared × | share of objects | P(arrive), flag | P(arrive), not | RR | RR, distance-adjusted |
|---|---:|---:|---|---|---|---|
| in_path | 1.5 | 0.74 | 0.145 [0.051, 0.242] | 0.167 [0.041, 0.311] | 0.87 [0.47, 2.12] | 1.18 [0.86, 2.33] |
| near_boundary | 1 | 0.17 | 0.059 [0.010, 0.130] | 0.169 [0.067, 0.271] | 0.35 [0.11, 0.59] | 0.60 [0.23, 0.83] |
| other_surface | 0.5 | 0.10 | 0.100 [0.008, 0.253] | 0.156 [0.063, 0.253] | 0.64 [0.08, 1.17] | 1.01 [0.16, 1.38] |
| approaching | — | 0.30 | 0.142 [0.046, 0.240] | 0.154 [0.059, 0.254] | 0.92 [0.58, 1.20] | 0.86 [0.59, 0.99] |
| near_boundary_approaching | — | 0.06 | 0.055 [0.000, 0.141] | 0.156 [0.060, 0.254] | 0.35 [0.00, 0.63] | 0.51 [0.00, 0.84] |

Approaching: closing on the walker's centre line at >= 0.3 m/s over the last 3 steps (motion only, no segmentation). Near a boundary and approaching, adjusted over distance x approaching strata (does the boundary add anything once motion is known?): 0.56 [0.00, 0.94].

| Surface under the object | share | n | P(arrive) |
|---|---:|---:|---|
| none (no ground at its depth) | 0.09 | 358 | 0.288 [0.069, 0.477] |
| walkable | 0.70 | 2920 | 0.151 [0.055, 0.250] |
| crosswalk | 0.05 | 210 | 0.038 [0.004, 0.185] |
| road | 0.09 | 372 | 0.113 [0.005, 0.278] |
| curb | 0.00 | 15 | 0.000 [0.000, 0.000] |
| stairs | 0.01 | 51 | 0.000 [0.000, 0.000] |
| non-walkable ground | 0.05 | 219 | 0.142 [0.006, 0.389] |

A declared multiplier above 1 is supported if its RR's CI is above 1, below 1 if below 1. The multipliers are not fitted to these numbers; fitting them would need train sessions only.
