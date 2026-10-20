# Scene-context flags vs the arrival label - gt_headfx_2026_10_20

5898 (frame, object) observations in 53 sessions ({'train': 35, 'val': 8, 'test': 10}); context from `data/sanpo_scene_context_v2/gt`. Label: in the walking corridor within the horizon (base rate 0.171). 95% CIs: session bootstrap. Risk ratio = P(arrives | flag) / P(arrives | not flag); adjusted = Mantel-Haenszel over distance bins [0.0, 3.0, 6.0, 10.0] m.

| Flag | declared × | share of objects | P(arrive), flag | P(arrive), not | RR | RR, distance-adjusted |
|---|---:|---:|---|---|---|---|
| in_path | 1.5 | 0.75 | 0.176 [0.054, 0.294] | 0.158 [0.053, 0.277] | 1.11 [0.59, 2.14] | 1.32 [0.93, 2.09] |
| near_boundary | 1 | 0.18 | 0.078 [0.021, 0.157] | 0.192 [0.066, 0.306] | 0.41 [0.23, 0.59] | 0.69 [0.42, 0.87] |
| other_surface | 0.5 | 0.11 | 0.110 [0.020, 0.230] | 0.179 [0.061, 0.291] | 0.61 [0.22, 1.02] | 0.89 [0.38, 1.22] |
| approaching | — | 0.29 | 0.161 [0.050, 0.273] | 0.175 [0.060, 0.288] | 0.92 [0.70, 1.16] | 0.88 [0.67, 0.97] |
| near_boundary_approaching | — | 0.05 | 0.066 [0.013, 0.146] | 0.177 [0.061, 0.289] | 0.37 [0.15, 0.62] | 0.55 [0.21, 0.81] |

Approaching: closing on the walker's centre line at >= 0.3 m/s over the last 3 steps (motion only, no segmentation). Near a boundary and approaching, adjusted over distance x approaching strata (does the boundary add anything once motion is known?): 0.59 [0.24, 0.89].

| Surface under the object | share | n | P(arrive) |
|---|---:|---:|---|
| none (no ground at its depth) | 0.08 | 460 | 0.276 [0.093, 0.447] |
| walkable | 0.72 | 4250 | 0.182 [0.054, 0.305] |
| crosswalk | 0.04 | 243 | 0.033 [0.004, 0.164] |
| road | 0.09 | 513 | 0.094 [0.011, 0.211] |
| curb | 0.00 | 18 | 0.000 [0.000, 0.000] |
| stairs | 0.01 | 51 | 0.000 [0.000, 0.000] |
| non-walkable ground | 0.06 | 363 | 0.152 [0.021, 0.310] |

A declared multiplier above 1 is supported if its RR's CI is above 1, below 1 if below 1. The multipliers are not fitted to these numbers; fitting them would need train sessions only.
