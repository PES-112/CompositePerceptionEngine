# gru - arrival462_2026_09_29_fold2_seed2

Held-out test sessions: 73

## All frames

Objects: 4152 (encounter rate 0.147), frames with >= 2 objects: 908, with an encounter: 385

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.841 | 0.649 | 0.438 [0.307, 0.603] | 0.597 [0.453, 0.784] | 0.581 [0.489, 0.702] | 0.616 [0.561, 0.663] |
| K0 | 0.577 | 0.200 | 0.386 [0.249, 0.512] | 0.458 [0.339, 0.570] | 0.584 [0.528, 0.630] | 0.677 [0.636, 0.709] |
| TTC | 0.643 | 0.465 | 0.407 [0.268, 0.541] | 0.509 [0.375, 0.625] | 0.601 [0.545, 0.647] | 0.689 [0.646, 0.727] |
| nearest | 0.820 | 0.631 | 0.454 [0.332, 0.590] | 0.640 [0.526, 0.791] | 0.591 [0.508, 0.690] | 0.736 [0.691, 0.772] |
| nearest_in_cone | 0.833 | 0.662 | 0.487 [0.368, 0.624] | 0.682 [0.574, 0.828] | 0.623 [0.562, 0.715] | 0.735 [0.688, 0.775] |

## Interaction slice (>= 2 objects within 5 m, or a crosswalk)

Objects: 870 (encounter rate 0.421), frames with >= 2 objects: 198, with an encounter: 172

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.771 | 0.787 | 0.750 [0.578, 0.843] | 0.752 [0.594, 0.871] | 0.637 [0.530, 0.787] | 0.705 [0.625, 0.764] |
| K0 | 0.566 | 0.501 | 0.620 [0.466, 0.693] | 0.560 [0.435, 0.628] | 0.579 [0.492, 0.652] | 0.737 [0.669, 0.857] |
| TTC | 0.638 | 0.693 | 0.684 [0.502, 0.764] | 0.652 [0.484, 0.741] | 0.607 [0.516, 0.704] | 0.750 [0.679, 0.873] |
| nearest | 0.748 | 0.774 | 0.772 [0.657, 0.850] | 0.774 [0.635, 0.896] | 0.642 [0.534, 0.750] | 0.757 [0.631, 0.844] |
| nearest_in_cone | 0.775 | 0.812 | 0.802 [0.720, 0.873] | 0.805 [0.697, 0.927] | 0.678 [0.619, 0.785] | 0.757 [0.625, 0.843] |

Higher is better for everything except flicker_rate. arrival_order: 0.5 = random. CIs resample whole sessions, as in the K0 ablation.
