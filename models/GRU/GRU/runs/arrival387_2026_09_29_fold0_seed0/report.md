# gru - arrival387_2026_09_29_fold0_seed0

Held-out test sessions: 61

## All frames

Objects: 3401 (encounter rate 0.260), frames with >= 2 objects: 806, with an encounter: 374

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.890 | 0.797 | 0.652 [0.451, 0.815] | 0.635 [0.482, 0.776] | 0.709 [0.621, 0.745] | 0.607 [0.558, 0.649] |
| K0 | 0.562 | 0.317 | 0.577 [0.344, 0.739] | 0.524 [0.380, 0.645] | 0.610 [0.551, 0.640] | 0.641 [0.594, 0.687] |
| TTC | 0.653 | 0.601 | 0.603 [0.402, 0.758] | 0.590 [0.464, 0.700] | 0.629 [0.553, 0.653] | 0.658 [0.609, 0.701] |
| nearest | 0.876 | 0.791 | 0.656 [0.474, 0.802] | 0.701 [0.585, 0.812] | 0.696 [0.639, 0.727] | 0.743 [0.696, 0.777] |
| nearest_in_cone | 0.874 | 0.806 | 0.657 [0.472, 0.804] | 0.699 [0.576, 0.811] | 0.715 [0.631, 0.753] | 0.734 [0.687, 0.768] |

## Interaction slice (>= 2 objects within 5 m, or a crosswalk)

Objects: 975 (encounter rate 0.649), frames with >= 2 objects: 239, with an encounter: 216

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.811 | 0.898 | 0.888 [0.817, 0.910] | 0.820 [0.794, 0.906] | 0.743 [0.729, 0.880] | 0.661 [0.619, 0.726] |
| K0 | 0.573 | 0.697 | 0.799 [0.537, 0.862] | 0.666 [0.534, 0.722] | 0.629 [0.581, 0.760] | 0.675 [0.635, 0.789] |
| TTC | 0.658 | 0.818 | 0.829 [0.610, 0.877] | 0.745 [0.624, 0.824] | 0.661 [0.619, 0.817] | 0.695 [0.611, 0.771] |
| nearest | 0.789 | 0.886 | 0.868 [0.822, 0.885] | 0.850 [0.818, 0.956] | 0.734 [0.716, 0.886] | 0.751 [0.673, 0.816] |
| nearest_in_cone | 0.821 | 0.911 | 0.878 [0.829, 0.896] | 0.864 [0.828, 0.978] | 0.763 [0.734, 0.893] | 0.738 [0.660, 0.807] |

Higher is better for everything except flicker_rate. arrival_order: 0.5 = random. CIs resample whole sessions, as in the K0 ablation.
