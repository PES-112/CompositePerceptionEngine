# gru - arrival462h96_2026_09_29_fold4_seed1

Held-out test sessions: 72

## All frames

Objects: 5279 (encounter rate 0.128), frames with >= 2 objects: 1208, with an encounter: 450

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.865 | 0.666 | 0.374 [0.214, 0.505] | 0.565 [0.410, 0.697] | 0.711 [0.627, 0.800] | 0.553 [0.508, 0.605] |
| K0 | 0.606 | 0.179 | 0.293 [0.152, 0.416] | 0.429 [0.335, 0.519] | 0.595 [0.502, 0.615] | 0.682 [0.650, 0.716] |
| TTC | 0.682 | 0.492 | 0.328 [0.182, 0.450] | 0.488 [0.386, 0.577] | 0.628 [0.516, 0.657] | 0.692 [0.656, 0.728] |
| nearest | 0.857 | 0.662 | 0.326 [0.213, 0.449] | 0.533 [0.425, 0.670] | 0.657 [0.560, 0.708] | 0.724 [0.697, 0.746] |
| nearest_in_cone | 0.850 | 0.659 | 0.343 [0.234, 0.461] | 0.575 [0.474, 0.705] | 0.665 [0.597, 0.716] | 0.721 [0.692, 0.741] |

## Interaction slice (>= 2 objects within 5 m, or a crosswalk)

Objects: 952 (encounter rate 0.389), frames with >= 2 objects: 192, with an encounter: 165

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.868 | 0.857 | 0.651 [0.374, 0.816] | 0.721 [0.650, 0.820] | 0.776 [0.710, 0.952] | 0.561 [0.444, 0.659] |
| K0 | 0.618 | 0.494 | 0.557 [0.257, 0.715] | 0.610 [0.494, 0.696] | 0.641 [0.602, 0.715] | 0.734 [0.637, 0.779] |
| TTC | 0.713 | 0.747 | 0.597 [0.294, 0.760] | 0.643 [0.542, 0.736] | 0.682 [0.621, 0.719] | 0.761 [0.654, 0.808] |
| nearest | 0.841 | 0.845 | 0.609 [0.346, 0.755] | 0.665 [0.555, 0.867] | 0.739 [0.673, 0.904] | 0.718 [0.567, 0.790] |
| nearest_in_cone | 0.827 | 0.836 | 0.609 [0.346, 0.755] | 0.678 [0.562, 0.867] | 0.738 [0.684, 0.911] | 0.713 [0.567, 0.787] |

Higher is better for everything except flicker_rate. arrival_order: 0.5 = random. CIs resample whole sessions, as in the K0 ablation.
