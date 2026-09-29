# gru_attention - arrival387_2026_09_29_fold4_seed0

Held-out test sessions: 60

## All frames

Objects: 2811 (encounter rate 0.214), frames with >= 2 objects: 600, with an encounter: 283

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.896 | 0.775 | 0.511 [0.330, 0.726] | 0.582 [0.472, 0.742] | 0.627 [0.523, 0.668] | 0.727 [0.633, 0.780] |
| K0 | 0.575 | 0.267 | 0.430 [0.240, 0.552] | 0.460 [0.277, 0.549] | 0.605 [0.459, 0.627] | 0.678 [0.598, 0.721] |
| TTC | 0.657 | 0.554 | 0.440 [0.240, 0.578] | 0.497 [0.289, 0.594] | 0.620 [0.459, 0.640] | 0.680 [0.600, 0.721] |
| nearest | 0.855 | 0.743 | 0.559 [0.406, 0.739] | 0.642 [0.547, 0.772] | 0.629 [0.547, 0.670] | 0.745 [0.670, 0.778] |
| nearest_in_cone | 0.856 | 0.744 | 0.558 [0.415, 0.738] | 0.644 [0.553, 0.774] | 0.640 [0.552, 0.685] | 0.744 [0.652, 0.784] |

## Interaction slice (>= 2 objects within 5 m, or a crosswalk)

Objects: 538 (encounter rate 0.552), frames with >= 2 objects: 129, with an encounter: 118

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.810 | 0.872 | 0.773 [0.706, 0.823] | 0.720 [0.668, 1.000] | 0.681 [0.645, 0.773] | 0.719 [0.569, 0.818] |
| K0 | 0.549 | 0.589 | 0.747 [0.662, 0.775] | 0.673 [0.610, 0.897] | 0.622 [0.600, 0.658] | 0.767 [0.658, 0.822] |
| TTC | 0.661 | 0.782 | 0.799 [0.724, 0.831] | 0.762 [0.737, 0.897] | 0.659 [0.639, 0.714] | 0.791 [0.715, 0.833] |
| nearest | 0.781 | 0.862 | 0.792 [0.734, 0.884] | 0.773 [0.721, 1.000] | 0.678 [0.660, 0.759] | 0.741 [0.600, 0.842] |
| nearest_in_cone | 0.787 | 0.865 | 0.792 [0.734, 0.884] | 0.773 [0.721, 1.000] | 0.685 [0.660, 0.759] | 0.734 [0.600, 0.842] |

Higher is better for everything except flicker_rate. arrival_order: 0.5 = random. CIs resample whole sessions, as in the K0 ablation.
