# Scene-context flags vs the arrival label - gt_train_2026_10_17

4121 (frame, object) observations in 32 sessions ({'train': 32, 'val': 0, 'test': 0}); context from `data/sanpo_scene_context/gt`. Label: in the walking corridor within the horizon (base rate 0.078). 95% CIs: session bootstrap. Risk ratio = P(arrives | flag) / P(arrives | not flag); adjusted = Mantel-Haenszel over distance bins [0.0, 3.0, 6.0, 10.0] m.

| Flag | declared × | share of objects | P(arrive), flag | P(arrive), not | RR | RR, distance-adjusted |
|---|---:|---:|---|---|---|---|
| in_path | 1.5 | 0.66 | 0.098 [0.035, 0.164] | 0.038 [0.005, 0.085] | 2.56 [1.36, 15.01] | 1.29 [0.85, 6.48] |
| near_boundary | 1.25 | 0.38 | 0.048 [0.003, 0.110] | 0.096 [0.035, 0.153] | 0.50 [0.06, 0.90] | 0.75 [0.18, 0.96] |
| other_surface | 0.5 | 0.20 | 0.043 [0.000, 0.102] | 0.087 [0.030, 0.145] | 0.50 [0.00, 0.93] | 0.89 [0.00, 1.24] |
| approaching | — | 0.26 | 0.089 [0.030, 0.155] | 0.074 [0.023, 0.130] | 1.20 [0.93, 1.66] | 1.00 [0.76, 1.29] |
| near_boundary_approaching | — | 0.09 | 0.056 [0.004, 0.121] | 0.080 [0.026, 0.136] | 0.70 [0.11, 1.23] | 0.80 [0.21, 1.15] |

Approaching: closing on the walker's centre line at >= 0.3 m/s over the last 3 steps (motion only, no segmentation). Near a boundary and approaching, adjusted over distance x approaching strata (does the boundary add anything once motion is known?): 0.75 [0.21, 1.48].

| Surface under the object | share | n | P(arrive) |
|---|---:|---:|---|
| none (no ground at its depth) | 0.07 | 271 | 0.044 [0.000, 0.125] |
| walkable | 0.66 | 2714 | 0.099 [0.036, 0.163] |
| crosswalk | 0.02 | 62 | 0.065 [0.000, 0.200] |
| road | 0.07 | 303 | 0.013 [0.000, 0.038] |
| curb | 0.00 | 15 | 0.000 [0.000, 0.000] |
| stairs | 0.01 | 58 | 0.000 [0.000, 0.000] |
| non-walkable ground | 0.17 | 698 | 0.047 [0.000, 0.128] |

A declared multiplier above 1 is supported if its RR's CI is above 1, below 1 if below 1. The multipliers are not fitted to these numbers; fitting them would need train sessions only.
