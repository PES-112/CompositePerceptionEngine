# gru_attention - arrival387_2026_09_29_fold2_seed2

Held-out test sessions: 60

## All frames

Objects: 3765 (encounter rate 0.130), frames with >= 2 objects: 934, with an encounter: 314

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.895 | 0.709 | 0.418 [0.243, 0.565] | 0.651 [0.537, 0.765] | 0.646 [0.514, 0.686] | 0.649 [0.604, 0.698] |
| K0 | 0.560 | 0.182 | 0.337 [0.157, 0.498] | 0.423 [0.261, 0.554] | 0.580 [0.467, 0.611] | 0.662 [0.632, 0.691] |
| TTC | 0.631 | 0.483 | 0.356 [0.162, 0.530] | 0.451 [0.271, 0.593] | 0.597 [0.469, 0.627] | 0.674 [0.647, 0.700] |
| nearest | 0.890 | 0.747 | 0.451 [0.297, 0.591] | 0.733 [0.632, 0.832] | 0.693 [0.650, 0.774] | 0.726 [0.692, 0.759] |
| nearest_in_cone | 0.890 | 0.755 | 0.466 [0.309, 0.612] | 0.745 [0.646, 0.842] | 0.706 [0.670, 0.801] | 0.723 [0.689, 0.756] |

## Interaction slice (>= 2 objects within 5 m, or a crosswalk)

Objects: 627 (encounter rate 0.486), frames with >= 2 objects: 164, with an encounter: 150

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | soonest_top1 (95% CI) | arrival_order (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|---|---|
| model | 0.779 | 0.817 | 0.725 [0.585, 0.777] | 0.769 [0.704, 0.806] | 0.672 [0.527, 0.793] | 0.693 [0.602, 0.809] |
| K0 | 0.568 | 0.564 | 0.673 [0.444, 0.726] | 0.661 [0.485, 0.793] | 0.605 [0.515, 0.618] | 0.671 [0.558, 0.854] |
| TTC | 0.628 | 0.729 | 0.707 [0.461, 0.761] | 0.706 [0.514, 0.798] | 0.623 [0.522, 0.635] | 0.708 [0.622, 0.831] |
| nearest | 0.812 | 0.863 | 0.784 [0.676, 0.827] | 0.839 [0.812, 0.933] | 0.715 [0.687, 0.849] | 0.765 [0.639, 0.838] |
| nearest_in_cone | 0.822 | 0.870 | 0.784 [0.676, 0.827] | 0.839 [0.812, 0.933] | 0.728 [0.692, 0.897] | 0.771 [0.647, 0.860] |

Higher is better for everything except flicker_rate. arrival_order: 0.5 = random. CIs resample whole sessions, as in the K0 ablation.
