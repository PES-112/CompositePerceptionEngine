# gru_attention - arrival462_2026_09_29_fold1_seed0

Held-out test sessions: 73

## All frames

Objects: 4251 (encounter rate 0.175), frames with >= 2 objects: 1009, with an encounter: 383

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.923 | 0.813 | 0.471 [0.301, 0.639] | 0.692 [0.588, 0.795] | 0.689 [0.659, 0.766] | 0.714 [0.677, 0.748] |
| K0 | 0.542 | 0.216 | 0.354 [0.192, 0.542] | 0.510 [0.356, 0.652] | 0.566 [0.548, 0.603] | 0.676 [0.632, 0.709] |
| TTC | 0.632 | 0.525 | 0.388 [0.221, 0.571] | 0.567 [0.413, 0.700] | 0.602 [0.585, 0.649] | 0.697 [0.649, 0.739] |
| nearest | 0.903 | 0.782 | 0.480 [0.315, 0.647] | 0.712 [0.581, 0.828] | 0.691 [0.663, 0.782] | 0.739 [0.710, 0.764] |
| nearest_in_cone | 0.904 | 0.801 | 0.481 [0.302, 0.662] | 0.710 [0.567, 0.830] | 0.729 [0.676, 0.781] | 0.742 [0.709, 0.770] |

## Interaction slice (>= 2 objects within 5 m, or a crosswalk)

Objects: 870 (encounter rate 0.594), frames with >= 2 objects: 202, with an encounter: 182

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.868 | 0.914 | 0.779 [0.602, 0.853] | 0.819 [0.751, 0.879] | 0.701 [0.663, 0.881] | 0.778 [0.704, 0.850] |
| K0 | 0.504 | 0.596 | 0.653 [0.373, 0.781] | 0.592 [0.427, 0.677] | 0.566 [0.538, 0.664] | 0.700 [0.582, 0.786] |
| TTC | 0.608 | 0.774 | 0.710 [0.487, 0.811] | 0.678 [0.573, 0.727] | 0.606 [0.590, 0.712] | 0.799 [0.709, 0.853] |
| nearest | 0.810 | 0.880 | 0.766 [0.564, 0.853] | 0.801 [0.713, 0.880] | 0.704 [0.675, 0.838] | 0.815 [0.770, 0.869] |
| nearest_in_cone | 0.839 | 0.907 | 0.792 [0.564, 0.888] | 0.834 [0.715, 0.894] | 0.758 [0.737, 0.871] | 0.820 [0.775, 0.894] |

Higher is better for everything except flicker_rate. arrival_order: 0.5 = random. CIs resample whole sessions, as in the K0 ablation.
