# gru_attention - arrival462_2026_09_29_fold0_seed2

Held-out test sessions: 73

## All frames

Objects: 4176 (encounter rate 0.208), frames with >= 2 objects: 1001, with an encounter: 352

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.927 | 0.823 | 0.606 [0.348, 0.790] | 0.634 [0.515, 0.747] | 0.703 [0.583, 0.741] | 0.668 [0.620, 0.706] |
| K0 | 0.561 | 0.267 | 0.578 [0.331, 0.750] | 0.532 [0.383, 0.676] | 0.612 [0.554, 0.664] | 0.612 [0.569, 0.647] |
| TTC | 0.649 | 0.566 | 0.567 [0.315, 0.745] | 0.553 [0.402, 0.689] | 0.628 [0.536, 0.662] | 0.635 [0.590, 0.673] |
| nearest | 0.904 | 0.804 | 0.605 [0.384, 0.761] | 0.683 [0.555, 0.783] | 0.694 [0.618, 0.731] | 0.743 [0.705, 0.772] |
| nearest_in_cone | 0.903 | 0.819 | 0.609 [0.391, 0.769] | 0.690 [0.563, 0.793] | 0.722 [0.628, 0.765] | 0.739 [0.701, 0.768] |

## Interaction slice (>= 2 objects within 5 m, or a crosswalk)

Objects: 955 (encounter rate 0.653), frames with >= 2 objects: 238, with an encounter: 209

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.809 | 0.895 | 0.886 [0.688, 0.928] | 0.803 [0.723, 0.869] | 0.744 [0.681, 0.803] | 0.681 [0.502, 0.731] |
| K0 | 0.600 | 0.715 | 0.816 [0.565, 0.868] | 0.683 [0.581, 0.752] | 0.615 [0.526, 0.695] | 0.662 [0.630, 0.746] |
| TTC | 0.670 | 0.827 | 0.832 [0.623, 0.877] | 0.750 [0.679, 0.819] | 0.646 [0.560, 0.745] | 0.680 [0.581, 0.728] |
| nearest | 0.778 | 0.884 | 0.839 [0.642, 0.880] | 0.813 [0.723, 0.875] | 0.718 [0.675, 0.789] | 0.740 [0.638, 0.797] |
| nearest_in_cone | 0.811 | 0.908 | 0.851 [0.673, 0.894] | 0.830 [0.745, 0.903] | 0.751 [0.696, 0.831] | 0.726 [0.619, 0.788] |

Higher is better for everything except flicker_rate. arrival_order: 0.5 = random. CIs resample whole sessions, as in the K0 ablation.
