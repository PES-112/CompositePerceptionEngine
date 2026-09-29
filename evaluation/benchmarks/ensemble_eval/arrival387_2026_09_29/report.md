# Ensemble - arrival387_2026_09_29

Pooled out-of-fold test frames: 302 sessions, 18182 objects, 3 seeds. Generated 2026-09-29 by `python -m models.Ensemble.ensemble`.

## all frames

| Method | AUROC | Avg. precision | encounter_top1 | soonest_top1 | arrival_order | flicker_rate |
|---|---:|---:|---:|---:|---:|---:|
| gru (3 seeds) | 0.877 ± 0.002 | 0.732 ± 0.004 | 0.470 ± 0.001 | 0.608 ± 0.008 | 0.669 ± 0.002 | 0.593 ± 0.009 |
| gru_attention (3 seeds) | 0.886 ± 0.004 | 0.729 ± 0.005 | 0.463 ± 0.006 | 0.597 ± 0.016 | 0.656 ± 0.003 | 0.664 ± 0.007 |
| ensemble_rank (3 seeds) | 0.874 ± 0.001 | 0.687 ± 0.001 | 0.474 ± 0.001 | 0.619 ± 0.007 | 0.675 ± 0.001 | 0.653 ± 0.002 |
| ensemble_stack (3 seeds) | 0.889 ± 0.003 | 0.750 ± 0.004 | 0.472 ± 0.007 | 0.622 ± 0.010 | 0.681 ± 0.004 | 0.652 ± 0.009 |
| K0 | 0.569 | 0.210 | 0.395 | 0.460 | 0.586 | 0.666 |
| TTC | 0.651 | 0.507 | 0.420 | 0.515 | 0.610 | 0.683 |
| nearest | 0.871 | 0.733 | 0.465 | 0.633 | 0.656 | 0.733 |
| nearest_in_cone | 0.870 | 0.746 | 0.477 | 0.652 | 0.680 | 0.731 |

## interaction frames

| Method | AUROC | Avg. precision | encounter_top1 | soonest_top1 | arrival_order | flicker_rate |
|---|---:|---:|---:|---:|---:|---:|
| gru (3 seeds) | 0.817 ± 0.004 | 0.863 ± 0.003 | 0.765 ± 0.004 | 0.773 ± 0.007 | 0.717 ± 0.005 | 0.667 ± 0.004 |
| gru_attention (3 seeds) | 0.805 ± 0.012 | 0.846 ± 0.006 | 0.748 ± 0.007 | 0.758 ± 0.012 | 0.699 ± 0.002 | 0.681 ± 0.012 |
| ensemble_rank (3 seeds) | 0.772 ± 0.002 | 0.819 ± 0.001 | 0.758 ± 0.003 | 0.753 ± 0.007 | 0.716 ± 0.001 | 0.697 ± 0.007 |
| ensemble_stack (3 seeds) | 0.822 ± 0.006 | 0.871 ± 0.004 | 0.776 ± 0.004 | 0.793 ± 0.006 | 0.734 ± 0.004 | 0.668 ± 0.017 |
| K0 | 0.564 | 0.571 | 0.670 | 0.626 | 0.600 | 0.704 |
| TTC | 0.653 | 0.754 | 0.714 | 0.693 | 0.635 | 0.749 |
| nearest | 0.804 | 0.854 | 0.751 | 0.772 | 0.702 | 0.758 |
| nearest_in_cone | 0.820 | 0.874 | 0.767 | 0.794 | 0.733 | 0.755 |

## Ensemble minus other (mean over seeds, 95% paired session-bootstrap CI)

| Ensemble | vs | AUROC difference | arrival_order difference |
|---|---|---|---|
| ensemble_rank | K0 | +0.304 [+0.250, +0.340] | +0.090 [+0.060, +0.117] |
| ensemble_rank | TTC | +0.222 [+0.183, +0.250] | +0.065 [+0.038, +0.092] |
| ensemble_rank | nearest | +0.003 [-0.007, +0.019] | +0.019 [+0.007, +0.032] |
| ensemble_rank | nearest_in_cone | +0.004 [-0.008, +0.019] | -0.005 [-0.020, +0.013] |
| ensemble_rank | gru | -0.003 [-0.010, +0.007] | +0.006 [-0.007, +0.021] |
| ensemble_rank | gru_attention | -0.013 [-0.021, -0.003] | +0.019 [+0.005, +0.035] |
| ensemble_stack | K0 | +0.320 [+0.259, +0.359] | +0.096 [+0.058, +0.129] |
| ensemble_stack | TTC | +0.238 [+0.192, +0.269] | +0.072 [+0.037, +0.104] |
| ensemble_stack | nearest | +0.019 [+0.008, +0.032] | +0.026 [+0.008, +0.041] |
| ensemble_stack | nearest_in_cone | +0.019 [+0.008, +0.032] | +0.001 [-0.010, +0.012] |
| ensemble_stack | gru | +0.012 [+0.005, +0.021] | +0.012 [-0.006, +0.030] |
| ensemble_stack | gru_attention | +0.003 [-0.003, +0.009] | +0.026 [+0.007, +0.044] |
| ensemble_stack | ensemble_rank | +0.016 [+0.008, +0.021] | +0.006 [-0.011, +0.018] |

A difference is only credible if its CI excludes 0.
