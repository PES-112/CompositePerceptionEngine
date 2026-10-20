# Scene-context flags vs the arrival label - gt_headfx_val_test_2026_10_20

1753 (frame, object) observations in 18 sessions ({'train': 0, 'val': 8, 'test': 10}); context from `data/sanpo_scene_context_v2/gt`. Label: in the walking corridor within the horizon (base rate 0.220). 95% CIs: session bootstrap. Risk ratio = P(arrives | flag) / P(arrives | not flag); adjusted = Mantel-Haenszel over distance bins [0.0, 3.0, 6.0, 10.0] m.

| Flag | declared × | share of objects | P(arrive), flag | P(arrive), not | RR | RR, distance-adjusted |
|---|---:|---:|---|---|---|---|
| in_path | 1.5 | 0.77 | 0.246 [0.007, 0.471] | 0.133 [0.011, 0.313] | 1.85 [0.23, 2.93] | 1.55 [0.42, 2.03] |
| near_boundary | 1 | 0.20 | 0.118 [0.005, 0.309] | 0.246 [0.010, 0.460] | 0.48 [0.29, 2.46] | 0.80 [0.38, 3.37] |
| other_surface | 0.5 | 0.13 | 0.128 [0.000, 0.291] | 0.235 [0.009, 0.456] | 0.54 [0.00, 4.86] | 0.67 [0.00, 2.73] |
| approaching | — | 0.28 | 0.210 [0.013, 0.433] | 0.224 [0.007, 0.441] | 0.94 [0.66, 2.87] | 0.96 [0.48, 2.35] |
| near_boundary_approaching | — | 0.05 | 0.096 [0.007, 0.281] | 0.226 [0.009, 0.442] | 0.43 [0.19, 3.51] | 0.69 [0.13, 2.65] |

Approaching: closing on the walker's centre line at >= 0.3 m/s over the last 3 steps (motion only, no segmentation). Near a boundary and approaching, adjusted over distance x approaching strata (does the boundary add anything once motion is known?): 0.71 [0.11, 2.23].

| Surface under the object | share | n | P(arrive) |
|---|---:|---:|---|
| none (no ground at its depth) | 0.06 | 102 | 0.235 [0.000, 0.482] |
| walkable | 0.76 | 1330 | 0.250 [0.007, 0.481] |
| crosswalk | 0.02 | 33 | 0.000 [0.000, 0.000] |
| road | 0.08 | 141 | 0.043 [0.000, 0.143] |
| curb | 0.00 | 3 | 0.000 [0.000, 0.000] |
| non-walkable ground | 0.08 | 144 | 0.167 [0.000, 0.381] |

A declared multiplier above 1 is supported if its RR's CI is above 1, below 1 if below 1. The multipliers are not fitted to these numbers; fitting them would need train sessions only.
