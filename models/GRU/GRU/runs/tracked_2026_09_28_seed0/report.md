# gru - tracked_2026_09_28_seed0

Held-out test sessions: 17

## All frames

Objects: 1218 (encounter rate 0.034), frames with >= 2 objects: 309, with an encounter: 45

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.936 | 0.463 | 0.366 [0.000, 0.442] | n/a | n/a | 0.589 [0.556, 0.643] |
| K0 | 0.481 | 0.031 | 0.289 [0.000, 0.349] | n/a | n/a | 0.625 [0.556, 0.668] |
| nearest | 0.937 | 0.513 | 0.366 [0.000, 0.442] | n/a | n/a | 0.701 [0.654, 0.732] |
| nearest_in_cone | 0.940 | 0.515 | 0.366 [0.000, 0.442] | n/a | n/a | 0.678 [0.610, 0.708] |

## Interaction slice (>= 2 objects within 5 m, or a crosswalk)

Objects: 394 (encounter rate 0.094), frames with >= 2 objects: 94, with an encounter: 41

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.879 | 0.537 | 0.451 [0.000, 0.462] | n/a | n/a | 0.559 [0.333, 1.000] |
| K0 | 0.482 | 0.111 | 0.351 [0.000, 0.359] | n/a | n/a | 0.656 [0.333, 1.000] |
| nearest | 0.895 | 0.544 | 0.451 [0.000, 0.462] | n/a | n/a | 0.763 [0.333, 1.000] |
| nearest_in_cone | 0.897 | 0.546 | 0.451 [0.000, 0.462] | n/a | n/a | 0.763 [0.333, 1.000] |

Higher is better for everything except flicker_rate. arrival_order: 0.5 = random. CIs resample whole sessions, as in the K0 ablation.
