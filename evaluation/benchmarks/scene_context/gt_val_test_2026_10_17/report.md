# Scene-context flags vs the arrival label - gt_val_test_2026_10_17

1561 (frame, object) observations in 15 sessions ({'train': 0, 'val': 5, 'test': 10}); context from `data/sanpo_scene_context/gt`. Label: in the walking corridor within the horizon (base rate 0.176). 95% CIs: session bootstrap. Risk ratio = P(arrives | flag) / P(arrives | not flag); adjusted = Mantel-Haenszel over distance bins [0.0, 3.0, 6.0, 10.0] m.

| Flag | declared × | share of objects | P(arrive), flag | P(arrive), not | RR | RR, distance-adjusted |
|---|---:|---:|---|---|---|---|
| in_path | 1.5 | 0.72 | 0.234 [0.000, 0.386] | 0.025 [0.000, 0.069] | 9.27 [0.00, 26.27] | 2.64 [0.00, 7.64] |
| near_boundary | 1 | 0.31 | 0.087 [0.000, 0.246] | 0.216 [0.002, 0.350] | 0.40 [0.03, 1.35] | 1.04 [0.12, 6.22] |
| other_surface | 0.5 | 0.23 | 0.019 [0.000, 0.062] | 0.223 [0.003, 0.372] | 0.09 [0.00, 0.86] | 0.38 [0.00, 1.67] |
| approaching | — | 0.23 | 0.130 [0.000, 0.278] | 0.189 [0.004, 0.332] | 0.69 [0.00, 0.86] | 0.74 [0.00, 0.87] |
| near_boundary_approaching | — | 0.07 | 0.035 [0.000, 0.141] | 0.186 [0.003, 0.332] | 0.19 [0.00, 0.44] | 0.87 [0.00, 1.31] |

Approaching: closing on the walker's centre line at >= 0.3 m/s over the last 3 steps (motion only, no segmentation). Near a boundary and approaching, adjusted over distance x approaching strata (does the boundary add anything once motion is known?): 1.29 [0.00, 1.92].

| Surface under the object | share | n | P(arrive) |
|---|---:|---:|---|
| none (no ground at its depth) | 0.03 | 52 | 0.077 [0.000, 0.158] |
| walkable | 0.71 | 1111 | 0.237 [0.000, 0.390] |
| crosswalk | 0.01 | 14 | 0.000 [0.000, 0.000] |
| road | 0.03 | 40 | 0.025 [0.000, 0.130] |
| non-walkable ground | 0.22 | 344 | 0.017 [0.000, 0.061] |

A declared multiplier above 1 is supported if its RR's CI is above 1, below 1 if below 1. The multipliers are not fitted to these numbers; fitting them would need train sessions only.
