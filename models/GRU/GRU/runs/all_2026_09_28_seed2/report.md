# gru - all_2026_09_28_seed2

Held-out test sessions: 21

## All frames

Objects: 15656 (encounter rate 0.128), frames with >= 2 objects: 3007, with an encounter: 1215

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|
| model | 1.000 | 1.000 | 0.302 [0.198, 0.420] | 0.969 [0.935, 0.992] |
| K0 | 0.490 | 0.118 | 0.274 [0.171, 0.399] | 0.978 [0.952, 0.996] |
| nearest | 1.000 | 1.000 | 0.302 [0.198, 0.420] | 0.999 [0.998, 1.000] |
| nearest_in_cone | 1.000 | 1.000 | 0.302 [0.198, 0.420] | 0.999 [0.998, 1.000] |

## Interaction slice (>= 2 objects within 5 m, or a crosswalk)

Objects: 14619 (encounter rate 0.137), frames with >= 2 objects: 2790, with an encounter: 1171

| Method | AUROC | Avg. precision | encounter_top1 (95% CI) | flicker_rate (95% CI) |
|---|---:|---:|---|---|
| model | 1.000 | 1.000 | 0.318 [0.211, 0.441] | 0.968 [0.931, 0.992] |
| K0 | 0.490 | 0.127 | 0.288 [0.180, 0.418] | 0.976 [0.949, 0.996] |
| nearest | 1.000 | 1.000 | 0.318 [0.211, 0.441] | 1.000 [1.000, 1.000] |
| nearest_in_cone | 1.000 | 1.000 | 0.318 [0.211, 0.441] | 1.000 [1.000, 1.000] |

encounter_top1: higher is better. flicker_rate: lower is better. CIs resample whole sessions, as in the K0 ablation.
