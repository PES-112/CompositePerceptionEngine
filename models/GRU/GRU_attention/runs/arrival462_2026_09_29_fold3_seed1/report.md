# gru_attention - arrival462_2026_09_29_fold3_seed1

Held-out test sessions: 73

## All frames

Objects: 3407 (encounter rate 0.193), frames with >= 2 objects: 771, with an encounter: 334

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.887 | 0.725 | 0.512 [0.342, 0.688] | 0.593 [0.469, 0.786] | 0.646 [0.630, 0.714] | 0.608 [0.496, 0.682] |
| K0 | 0.578 | 0.248 | 0.391 [0.235, 0.515] | 0.420 [0.310, 0.539] | 0.558 [0.435, 0.585] | 0.644 [0.566, 0.701] |
| TTC | 0.676 | 0.559 | 0.453 [0.298, 0.593] | 0.536 [0.433, 0.645] | 0.598 [0.455, 0.618] | 0.662 [0.582, 0.719] |
| nearest | 0.853 | 0.721 | 0.529 [0.380, 0.673] | 0.679 [0.604, 0.790] | 0.663 [0.650, 0.750] | 0.701 [0.618, 0.757] |
| nearest_in_cone | 0.846 | 0.720 | 0.526 [0.379, 0.669] | 0.674 [0.602, 0.773] | 0.673 [0.657, 0.721] | 0.693 [0.606, 0.751] |

## Interaction slice (>= 2 objects within 5 m, or a crosswalk)

Objects: 559 (encounter rate 0.578), frames with >= 2 objects: 148, with an encounter: 134

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.805 | 0.862 | 0.752 [0.613, 0.837] | 0.760 [0.720, 1.000] | 0.705 [0.694, 1.000] | 0.686 [0.458, 0.791] |
| K0 | 0.591 | 0.615 | 0.668 [0.520, 0.745] | 0.598 [0.449, 0.816] | 0.597 [0.475, 0.654] | 0.714 [0.542, 0.822] |
| TTC | 0.701 | 0.814 | 0.731 [0.598, 0.790] | 0.715 [0.603, 0.862] | 0.663 [0.513, 0.721] | 0.777 [0.619, 0.861] |
| nearest | 0.770 | 0.861 | 0.787 [0.733, 0.868] | 0.798 [0.747, 1.000] | 0.706 [0.688, 0.875] | 0.763 [0.561, 0.822] |
| nearest_in_cone | 0.782 | 0.866 | 0.794 [0.734, 0.871] | 0.806 [0.761, 1.000] | 0.723 [0.709, 0.889] | 0.763 [0.561, 0.822] |

Higher is better for everything except flicker_rate. arrival_order: 0.5 = random. CIs resample whole sessions, as in the K0 ablation.
