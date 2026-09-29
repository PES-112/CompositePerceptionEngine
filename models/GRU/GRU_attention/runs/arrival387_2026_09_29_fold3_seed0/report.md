# gru_attention - arrival387_2026_09_29_fold3_seed0

Held-out test sessions: 60

## All frames

Objects: 4914 (encounter rate 0.108), frames with >= 2 objects: 1062, with an encounter: 380

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.817 | 0.547 | 0.308 [0.198, 0.405] | 0.451 [0.405, 0.510] | 0.617 [0.542, 0.668] | 0.674 [0.646, 0.708] |
| K0 | 0.602 | 0.150 | 0.285 [0.176, 0.381] | 0.423 [0.365, 0.512] | 0.568 [0.507, 0.615] | 0.681 [0.644, 0.711] |
| TTC | 0.663 | 0.420 | 0.316 [0.214, 0.410] | 0.479 [0.423, 0.558] | 0.601 [0.522, 0.662] | 0.700 [0.653, 0.741] |
| nearest | 0.794 | 0.558 | 0.298 [0.216, 0.380] | 0.484 [0.384, 0.642] | 0.606 [0.490, 0.703] | 0.731 [0.698, 0.758] |
| nearest_in_cone | 0.786 | 0.557 | 0.318 [0.231, 0.404] | 0.520 [0.429, 0.665] | 0.630 [0.552, 0.710] | 0.731 [0.698, 0.758] |

## Interaction slice (>= 2 objects within 5 m, or a crosswalk)

Objects: 840 (encounter rate 0.307), frames with >= 2 objects: 157, with an encounter: 125

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.809 | 0.760 | 0.566 [0.339, 0.737] | 0.651 [0.589, 0.775] | 0.719 [0.641, 0.894] | 0.676 [0.579, 0.773] |
| K0 | 0.642 | 0.433 | 0.467 [0.265, 0.609] | 0.519 [0.395, 0.621] | 0.590 [0.458, 0.686] | 0.744 [0.683, 0.810] |
| TTC | 0.718 | 0.703 | 0.495 [0.314, 0.647] | 0.544 [0.451, 0.619] | 0.634 [0.480, 0.717] | 0.804 [0.701, 0.907] |
| nearest | 0.776 | 0.757 | 0.542 [0.349, 0.701] | 0.604 [0.501, 0.784] | 0.690 [0.506, 0.881] | 0.777 [0.633, 0.886] |
| nearest_in_cone | 0.767 | 0.757 | 0.567 [0.366, 0.746] | 0.646 [0.530, 0.845] | 0.720 [0.607, 0.925] | 0.769 [0.609, 0.876] |

Higher is better for everything except flicker_rate. arrival_order: 0.5 = random. CIs resample whole sessions, as in the K0 ablation.
