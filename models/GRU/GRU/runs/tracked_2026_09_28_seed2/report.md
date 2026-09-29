# gru - tracked_2026_09_28_seed2

Held-out test sessions: 17

## All frames

Objects: 1218 (encounter rate 0.034), frames with >= 2 objects: 309, with an encounter: 45

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|
| model | 0.830 | 0.198 | 0.270 [0.000, 0.326] | 0.566 [0.471, 0.719] |
| K0 | 0.481 | 0.031 | 0.289 [0.000, 0.349] | 0.625 [0.556, 0.668] |
| nearest | 0.937 | 0.513 | 0.366 [0.000, 0.442] | 0.701 [0.654, 0.732] |
| nearest_in_cone | 0.940 | 0.515 | 0.366 [0.000, 0.442] | 0.678 [0.610, 0.708] |

## Interaction slice (>= 2 objects within 5 m, or a crosswalk)

Objects: 394 (encounter rate 0.094), frames with >= 2 objects: 94, with an encounter: 41

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|
| model | 0.687 | 0.243 | 0.351 [0.000, 0.359] | 0.451 [0.333, 1.000] |
| K0 | 0.482 | 0.111 | 0.351 [0.000, 0.359] | 0.656 [0.333, 1.000] |
| nearest | 0.895 | 0.544 | 0.451 [0.000, 0.462] | 0.763 [0.333, 1.000] |
| nearest_in_cone | 0.897 | 0.546 | 0.451 [0.000, 0.462] | 0.763 [0.333, 1.000] |

encounter_top1: higher is better. flicker_rate: lower is better. CIs resample whole sessions, as in the K0 ablation.
