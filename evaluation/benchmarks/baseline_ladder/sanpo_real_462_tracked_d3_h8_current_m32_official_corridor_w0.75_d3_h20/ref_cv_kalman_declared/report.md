# Baseline ladder - sanpo_real_462_tracked_d3_h8_current_m32_official_corridor_w0.75_d3_h20

Label: in the walking corridor (width 0.75 m + class half-width, 0 < forward <= 3 m along the direction of travel from fixed_camera_poses.csv) within 20 steps (4 s); tracked objects only.

SANPO official test split: 71 sessions (20 with an arrival), 4721 objects, positive rate 0.169. Learned methods are averaged over their seeds; tabular baselines are fitted on the 252 official-train sessions. Generated 2026-10-17 by `python -m models.GRU.benchmark` (177 s).

## All objects

| Method | AUROC | AUPRC | arrival_order | soonest_top1 | flicker ↓ | AUROC static | AUROC dynamic | AUPRC static | AUPRC dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gru | 0.875 | 0.713 | 0.722 | 0.679 | 0.582 | 0.986 | 0.867 | 0.901 | 0.707 |
| gbt_now | 0.865 | 0.693 | 0.702 | 0.595 | 0.604 | 0.974 | 0.860 | 0.852 | 0.688 |
| cv_kalman | 0.862 | 0.675 | 0.721 | 0.703 | 0.639 | 0.985 | 0.858 | 0.896 | 0.670 |
| gbt_history | 0.859 | 0.676 | 0.705 | 0.592 | 0.623 | 0.975 | 0.853 | 0.824 | 0.671 |
| logreg_now | 0.852 | 0.557 | 0.601 | 0.600 | 0.588 | 0.975 | 0.842 | 0.837 | 0.546 |
| nearest_in_corridor | 0.851 | 0.696 | 0.733 | 0.659 | 0.654 | 0.985 | 0.845 | 0.929 | 0.689 |
| nearest | 0.850 | 0.626 | 0.620 | 0.652 | 0.724 | 0.954 | 0.846 | 0.538 | 0.639 |
| nearest_in_cone | 0.846 | 0.626 | 0.635 | 0.649 | 0.714 | 0.954 | 0.843 | 0.539 | 0.640 |
| cv_last2 | 0.829 | 0.639 | 0.681 | 0.656 | 0.762 | 0.965 | 0.823 | 0.887 | 0.633 |
| cv_kalman_declared | 0.822 | 0.553 | 0.625 | 0.599 | 0.649 | 0.972 | 0.816 | 0.811 | 0.546 |
| TTC_in_cone | 0.754 | 0.487 | 0.621 | 0.538 | 0.673 | 0.840 | 0.751 | 0.372 | 0.500 |
| TTC | 0.663 | 0.434 | 0.570 | 0.498 | 0.680 | 0.727 | 0.661 | 0.316 | 0.459 |
| K0 | 0.583 | 0.202 | 0.537 | 0.463 | 0.656 | 0.612 | 0.581 | 0.116 | 0.208 |

Strata: static classes 303 objects (positive rate 0.086), dynamic 4418 (0.174).

## Method minus cv_kalman_declared (95% paired session-bootstrap CI; two-sided p from the bootstrap SE; Holm within each stratum, 24 tests)

| Stratum | Method | ΔAUROC | p (Holm) | ΔAUPRC | p (Holm) |
|---|---|---|---|---|---|
| all | gru | +0.052 [+0.013, +0.078] | 0.020 | +0.160 [+0.111, +0.202] | 0.000 |
| all | gbt_now | +0.043 [+0.013, +0.061] | 0.008 | +0.140 [+0.092, +0.184] | 0.000 |
| all | cv_kalman | +0.040 [+0.023, +0.054] | 0.000 | +0.122 [+0.086, +0.175] | 0.000 |
| all | gbt_history | +0.037 [+0.004, +0.054] | 0.039 | +0.123 [+0.071, +0.170] | 0.000 |
| all | logreg_now | +0.029 [-0.003, +0.057] | 0.431 | +0.005 [-0.099, +0.102] | 0.931 |
| all | nearest_in_corridor | +0.029 [+0.004, +0.051] | 0.158 | +0.143 [+0.104, +0.184] | 0.000 |
| all | nearest | +0.027 [+0.009, +0.041] | 0.009 | +0.073 [-0.012, +0.137] | 0.431 |
| all | nearest_in_cone | +0.024 [-0.001, +0.050] | 0.431 | +0.073 [-0.014, +0.148] | 0.431 |
| all | cv_last2 | +0.006 [-0.007, +0.020] | 0.806 | +0.086 [+0.045, +0.137] | 0.006 |
| all | TTC_in_cone | -0.069 [-0.117, -0.011] | 0.134 | -0.066 [-0.175, +0.036] | 0.663 |
| all | TTC | -0.159 [-0.225, -0.079] | 0.001 | -0.119 [-0.213, -0.016] | 0.158 |
| all | K0 | -0.240 [-0.317, -0.143] | 0.000 | -0.351 [-0.471, -0.211] | 0.000 |
| static | gru | +0.014 [-0.015, +0.029] | 1.000 | +0.089 [-0.386, +0.121] | 1.000 |
| static | gbt_now | +0.003 [-0.047, +0.022] | 1.000 | +0.041 [-0.658, +0.154] | 1.000 |
| static | cv_kalman | +0.014 [-0.007, +0.028] | 1.000 | +0.084 [-0.412, +0.106] | 1.000 |
| static | gbt_history | +0.004 [-0.045, +0.029] | 1.000 | +0.013 [-0.667, +0.165] | 1.000 |
| static | logreg_now | +0.004 [-0.080, +0.021] | 1.000 | +0.025 [-0.687, +0.063] | 1.000 |
| static | nearest_in_corridor | +0.013 [-0.006, +0.028] | 1.000 | +0.118 [+0.000, +0.131] | 0.020 |
| static | nearest | -0.018 [-0.077, +0.007] | 1.000 | -0.273 [-0.759, +0.008] | 1.000 |
| static | nearest_in_cone | -0.017 [-0.076, +0.007] | 1.000 | -0.272 [-0.759, +0.009] | 1.000 |
| static | cv_last2 | -0.006 [-0.017, +0.005] | 1.000 | +0.076 [-0.017, +0.118] | 1.000 |
| static | TTC_in_cone | -0.132 [-0.370, -0.076] | 1.000 | -0.440 [-0.984, -0.150] | 0.913 |
| static | TTC | -0.245 [-0.705, -0.156] | 1.000 | -0.496 [-0.992, -0.172] | 0.538 |
| static | K0 | -0.360 [-0.705, -0.273] | 0.053 | -0.696 [-0.992, -0.503] | 0.000 |
| dynamic | gru | +0.051 [+0.008, +0.076] | 0.043 | +0.161 [+0.115, +0.204] | 0.000 |
| dynamic | gbt_now | +0.044 [+0.012, +0.061] | 0.008 | +0.142 [+0.093, +0.185] | 0.000 |
| dynamic | cv_kalman | +0.041 [+0.024, +0.056] | 0.000 | +0.124 [+0.089, +0.183] | 0.000 |
| dynamic | gbt_history | +0.037 [+0.001, +0.054] | 0.055 | +0.125 [+0.072, +0.174] | 0.000 |
| dynamic | logreg_now | +0.026 [-0.009, +0.054] | 0.473 | -0.000 [-0.097, +0.098] | 1.000 |
| dynamic | nearest_in_corridor | +0.029 [+0.002, +0.052] | 0.197 | +0.143 [+0.105, +0.184] | 0.000 |
| dynamic | nearest | +0.030 [+0.012, +0.044] | 0.003 | +0.093 [+0.030, +0.156] | 0.073 |
| dynamic | nearest_in_cone | +0.027 [+0.002, +0.054] | 0.359 | +0.094 [+0.027, +0.168] | 0.160 |
| dynamic | cv_last2 | +0.007 [-0.007, +0.022] | 1.000 | +0.087 [+0.045, +0.138] | 0.005 |
| dynamic | TTC_in_cone | -0.065 [-0.115, -0.003] | 0.197 | -0.046 [-0.135, +0.053] | 1.000 |
| dynamic | TTC | -0.155 [-0.224, -0.072] | 0.002 | -0.087 [-0.181, +0.003] | 0.378 |
| dynamic | K0 | -0.236 [-0.315, -0.139] | 0.000 | -0.338 [-0.451, -0.200] | 0.000 |

A difference is credible only if its Holm-adjusted p < 0.05. Static = traffic light, stop sign, fire hydrant, pole, bollard, stairs, crosswalk, pothole, puddle, bench; dynamic = the rest.
