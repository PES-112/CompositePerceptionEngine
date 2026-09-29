# gru - arrival387_2026_09_29_fold1_seed0

Held-out test sessions: 61

## All frames

Objects: 3291 (encounter rate 0.203), frames with >= 2 objects: 793, with an encounter: 357

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.897 | 0.798 | 0.425 [0.232, 0.612] | 0.618 [0.449, 0.765] | 0.684 [0.613, 0.770] | 0.611 [0.539, 0.662] |
| K0 | 0.556 | 0.258 | 0.355 [0.178, 0.550] | 0.463 [0.332, 0.586] | 0.562 [0.538, 0.610] | 0.665 [0.597, 0.711] |
| TTC | 0.647 | 0.561 | 0.389 [0.209, 0.579] | 0.542 [0.411, 0.645] | 0.596 [0.567, 0.651] | 0.702 [0.629, 0.753] |
| nearest | 0.900 | 0.783 | 0.402 [0.236, 0.575] | 0.629 [0.500, 0.728] | 0.665 [0.567, 0.756] | 0.725 [0.678, 0.764] |
| nearest_in_cone | 0.907 | 0.816 | 0.424 [0.253, 0.603] | 0.671 [0.545, 0.767] | 0.719 [0.620, 0.767] | 0.727 [0.684, 0.765] |

## Interaction slice (>= 2 objects within 5 m, or a crosswalk)

Objects: 800 (encounter rate 0.592), frames with >= 2 objects: 190, with an encounter: 170

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.838 | 0.897 | 0.723 [0.486, 0.821] | 0.743 [0.580, 0.814] | 0.693 [0.662, 0.818] | 0.697 [0.586, 0.762] |
| K0 | 0.529 | 0.617 | 0.621 [0.323, 0.763] | 0.602 [0.425, 0.692] | 0.558 [0.534, 0.607] | 0.692 [0.596, 0.736] |
| TTC | 0.626 | 0.788 | 0.698 [0.455, 0.809] | 0.690 [0.590, 0.753] | 0.597 [0.589, 0.681] | 0.779 [0.656, 0.825] |
| nearest | 0.802 | 0.867 | 0.720 [0.492, 0.829] | 0.753 [0.641, 0.812] | 0.680 [0.631, 0.801] | 0.760 [0.616, 0.808] |
| nearest_in_cone | 0.840 | 0.909 | 0.760 [0.510, 0.877] | 0.800 [0.669, 0.867] | 0.742 [0.696, 0.811] | 0.765 [0.636, 0.818] |

Higher is better for everything except flicker_rate. arrival_order: 0.5 = random. CIs resample whole sessions, as in the K0 ablation.
