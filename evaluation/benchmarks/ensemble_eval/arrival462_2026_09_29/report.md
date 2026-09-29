# Ensemble - arrival462_2026_09_29

Pooled out-of-fold test frames: 364 sessions, 21265 objects, 3 seeds. Generated 2026-09-29 by `python -m models.Ensemble.ensemble`.

## all frames

| Method | AUROC | Avg. precision | encounter_top1 | soonest_top1 | arrival_order | flicker_rate |
|---|---:|---:|---:|---:|---:|---:|
| gru (3 seeds) | 0.887 ± 0.001 | 0.741 ± 0.001 | 0.481 ± 0.001 | 0.629 ± 0.003 | 0.676 ± 0.004 | 0.590 ± 0.010 |
| gru_attention (3 seeds) | 0.893 ± 0.001 | 0.737 ± 0.002 | 0.464 ± 0.003 | 0.602 ± 0.002 | 0.666 ± 0.005 | 0.655 ± 0.012 |
| ensemble_rank (3 seeds) | 0.879 ± 0.000 | 0.687 ± 0.001 | 0.473 ± 0.001 | 0.620 ± 0.006 | 0.674 ± 0.002 | 0.651 ± 0.006 |
| ensemble_stack (3 seeds) | 0.894 ± 0.001 | 0.752 ± 0.002 | 0.478 ± 0.002 | 0.630 ± 0.004 | 0.688 ± 0.003 | 0.636 ± 0.008 |
| K0 | 0.571 | 0.202 | 0.394 | 0.469 | 0.585 | 0.660 |
| TTC | 0.656 | 0.504 | 0.422 | 0.528 | 0.612 | 0.676 |
| nearest | 0.872 | 0.727 | 0.469 | 0.644 | 0.657 | 0.730 |
| nearest_in_cone | 0.871 | 0.739 | 0.481 | 0.663 | 0.681 | 0.727 |

## interaction frames

| Method | AUROC | Avg. precision | encounter_top1 | soonest_top1 | arrival_order | flicker_rate |
|---|---:|---:|---:|---:|---:|---:|
| gru (3 seeds) | 0.825 ± 0.000 | 0.867 ± 0.001 | 0.773 ± 0.001 | 0.774 ± 0.005 | 0.724 ± 0.005 | 0.657 ± 0.006 |
| gru_attention (3 seeds) | 0.816 ± 0.003 | 0.857 ± 0.000 | 0.765 ± 0.003 | 0.770 ± 0.005 | 0.714 ± 0.005 | 0.682 ± 0.020 |
| ensemble_rank (3 seeds) | 0.774 ± 0.001 | 0.821 ± 0.001 | 0.760 ± 0.003 | 0.745 ± 0.001 | 0.717 ± 0.002 | 0.693 ± 0.008 |
| ensemble_stack (3 seeds) | 0.825 ± 0.003 | 0.872 ± 0.001 | 0.776 ± 0.001 | 0.786 ± 0.003 | 0.740 ± 0.003 | 0.664 ± 0.007 |
| K0 | 0.567 | 0.573 | 0.668 | 0.612 | 0.599 | 0.707 |
| TTC | 0.657 | 0.759 | 0.714 | 0.688 | 0.638 | 0.749 |
| nearest | 0.797 | 0.851 | 0.756 | 0.771 | 0.701 | 0.758 |
| nearest_in_cone | 0.813 | 0.869 | 0.772 | 0.792 | 0.731 | 0.755 |

## Ensemble minus other (mean over seeds, 95% paired session-bootstrap CI)

| Ensemble | vs | AUROC difference | arrival_order difference |
|---|---|---|---|
| ensemble_rank | K0 | +0.308 [+0.262, +0.342] | +0.089 [+0.060, +0.115] |
| ensemble_rank | TTC | +0.223 [+0.188, +0.251] | +0.061 [+0.035, +0.085] |
| ensemble_rank | nearest | +0.007 [-0.004, +0.019] | +0.016 [+0.003, +0.028] |
| ensemble_rank | nearest_in_cone | +0.008 [-0.004, +0.020] | -0.007 [-0.022, +0.007] |
| ensemble_rank | gru | -0.008 [-0.014, +0.000] | -0.002 [-0.015, +0.011] |
| ensemble_rank | gru_attention | -0.014 [-0.020, -0.006] | +0.008 [-0.004, +0.022] |
| ensemble_stack | K0 | +0.323 [+0.272, +0.361] | +0.104 [+0.071, +0.131] |
| ensemble_stack | TTC | +0.239 [+0.199, +0.269] | +0.076 [+0.046, +0.103] |
| ensemble_stack | nearest | +0.022 [+0.013, +0.034] | +0.031 [+0.013, +0.047] |
| ensemble_stack | nearest_in_cone | +0.023 [+0.013, +0.034] | +0.007 [-0.008, +0.021] |
| ensemble_stack | gru | +0.007 [+0.001, +0.014] | +0.012 [-0.002, +0.028] |
| ensemble_stack | gru_attention | +0.001 [-0.003, +0.006] | +0.022 [+0.009, +0.037] |
| ensemble_stack | ensemble_rank | +0.015 [+0.009, +0.021] | +0.015 [+0.003, +0.025] |

A difference is only credible if its CI excludes 0.
