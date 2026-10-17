# Scene-context flags vs the arrival label - oracle_gt_2026_10_17

5682 (frame, object) observations in 47 sessions ({'train': 32, 'val': 5, 'test': 10}); context from `data/sanpo_scene_context/gt`. Label: in the walking corridor within the horizon (base rate 0.105). 95% CIs: session bootstrap. Risk ratio = P(arrives | flag) / P(arrives | not flag); adjusted = Mantel-Haenszel over distance bins [0.0, 3.0, 6.0, 10.0] m.

| Flag | declared × | share of objects | P(arrive), flag | P(arrive), not | RR | RR, distance-adjusted |
|---|---:|---:|---|---|---|---|
| in_path | 1.5 | 0.68 | 0.138 [0.049, 0.231] | 0.035 [0.007, 0.075] | 3.91 [1.73, 12.95] | 1.56 [0.98, 4.17] |
| near_boundary | 1.25 | 0.36 | 0.057 [0.010, 0.116] | 0.131 [0.051, 0.215] | 0.44 [0.17, 0.71] | 0.83 [0.41, 0.99] |
| other_surface | 0.5 | 0.21 | 0.036 [0.003, 0.079] | 0.123 [0.043, 0.210] | 0.29 [0.06, 0.67] | 0.70 [0.20, 1.06] |

| Surface under the object | share | n | P(arrive) |
|---|---:|---:|---|
| none (no ground at its depth) | 0.06 | 323 | 0.050 [0.008, 0.119] |
| walkable | 0.67 | 3825 | 0.139 [0.050, 0.232] |
| crosswalk | 0.01 | 76 | 0.053 [0.000, 0.139] |
| road | 0.06 | 343 | 0.015 [0.003, 0.039] |
| curb | 0.00 | 15 | 0.000 [0.000, 0.000] |
| stairs | 0.01 | 58 | 0.000 [0.000, 0.000] |
| non-walkable ground | 0.18 | 1042 | 0.037 [0.001, 0.096] |

A declared multiplier above 1 is supported if its RR's CI is above 1, below 1 if below 1. The multipliers are not fitted to these numbers; fitting them would need train sessions only.
