# Held-out test - official462_2026_10_14

SANPO's official test sessions (71 with tracked objects, 25 with an arrival; 4721 objects). Trained on 252 official-train sessions, early-stopped and stacker-fitted on 41 more. Dataset `sanpo_real_462_tracked_d3_h8_current_m32_official`, 3 seeds. Generated 2026-10-14 by `python -m models.Ensemble.heldout`.

Time units: one step = 0.2 s (SANPO is 15 FPS, stride 3), so the 20-step arrival horizon is 4 s and the 8-step history 1.6 s. Stored speeds are 2x too high (the CSVs assumed 30 FPS); every metric here is a ranking, which a uniform speed scale does not change.

## all frames

| Method | AUROC | Avg. precision | encounter_top1 | soonest_top1 | arrival_order | flicker_rate |
|---|---:|---:|---:|---:|---:|---:|
| gru (3 seeds) | 0.888 ± 0.002 | 0.790 ± 0.002 | 0.529 ± 0.008 | 0.717 ± 0.007 | 0.697 ± 0.009 | 0.561 ± 0.009 |
| gru_attention (3 seeds) | 0.892 ± 0.003 | 0.786 ± 0.007 | 0.503 ± 0.007 | 0.689 ± 0.005 | 0.670 ± 0.004 | 0.645 ± 0.012 |
| ensemble_rank (3 seeds) | 0.882 ± 0.000 | 0.762 ± 0.003 | 0.522 ± 0.002 | 0.683 ± 0.005 | 0.696 ± 0.001 | 0.638 ± 0.006 |
| ensemble_stack (3 seeds) | 0.893 ± 0.000 | 0.807 ± 0.003 | 0.516 ± 0.002 | 0.698 ± 0.006 | 0.708 ± 0.005 | 0.628 ± 0.015 |
| K0 | 0.571 | 0.259 | 0.454 | 0.573 | 0.563 | 0.656 |
| TTC | 0.661 | 0.559 | 0.471 | 0.609 | 0.597 | 0.680 |
| TTC_in_cone | 0.765 | 0.606 | 0.492 | 0.648 | 0.683 | 0.673 |
| nearest | 0.874 | 0.777 | 0.495 | 0.694 | 0.676 | 0.724 |
| nearest_in_cone | 0.875 | 0.791 | 0.500 | 0.696 | 0.711 | 0.714 |

## interaction frames

| Method | AUROC | Avg. precision | encounter_top1 | soonest_top1 | arrival_order | flicker_rate |
|---|---:|---:|---:|---:|---:|---:|
| gru (3 seeds) | 0.833 ± 0.007 | 0.901 ± 0.005 | 0.777 ± 0.003 | 0.791 ± 0.003 | 0.731 ± 0.012 | 0.685 ± 0.011 |
| gru_attention (3 seeds) | 0.822 ± 0.008 | 0.887 ± 0.007 | 0.772 ± 0.016 | 0.762 ± 0.009 | 0.694 ± 0.007 | 0.707 ± 0.057 |
| ensemble_rank (3 seeds) | 0.810 ± 0.004 | 0.878 ± 0.003 | 0.793 ± 0.008 | 0.769 ± 0.010 | 0.725 ± 0.001 | 0.736 ± 0.010 |
| ensemble_stack (3 seeds) | 0.846 ± 0.004 | 0.910 ± 0.003 | 0.791 ± 0.005 | 0.786 ± 0.004 | 0.753 ± 0.005 | 0.668 ± 0.014 |
| K0 | 0.586 | 0.646 | 0.667 | 0.621 | 0.568 | 0.717 |
| TTC | 0.667 | 0.800 | 0.726 | 0.708 | 0.616 | 0.782 |
| TTC_in_cone | 0.760 | 0.847 | 0.773 | 0.761 | 0.714 | 0.775 |
| nearest | 0.795 | 0.881 | 0.767 | 0.799 | 0.698 | 0.777 |
| nearest_in_cone | 0.821 | 0.899 | 0.795 | 0.832 | 0.745 | 0.777 |

## Method minus other (mean over seeds, 95% paired session-bootstrap CI)

| Method | vs | AUROC difference | arrival_order difference |
|---|---|---|---|
| gru | K0 | +0.317 [+0.219, +0.384] | +0.134 [+0.068, +0.205] |
| gru | TTC | +0.227 [+0.154, +0.279] | +0.100 [+0.062, +0.170] |
| gru | TTC_in_cone | +0.123 [+0.075, +0.151] | +0.015 [-0.021, +0.051] |
| gru | nearest | +0.014 [-0.020, +0.041] | +0.021 [-0.024, +0.031] |
| gru | nearest_in_cone | +0.013 [-0.018, +0.046] | -0.013 [-0.050, +0.026] |
| gru_attention | K0 | +0.321 [+0.211, +0.385] | +0.106 [+0.043, +0.184] |
| gru_attention | TTC | +0.231 [+0.149, +0.280] | +0.072 [+0.050, +0.148] |
| gru_attention | TTC_in_cone | +0.127 [+0.070, +0.155] | -0.013 [-0.051, +0.033] |
| gru_attention | nearest | +0.018 [-0.020, +0.042] | -0.006 [-0.045, +0.018] |
| gru_attention | nearest_in_cone | +0.017 [-0.018, +0.045] | -0.041 [-0.084, +0.008] |
| gru_attention | gru | +0.004 [-0.018, +0.016] | -0.028 [-0.044, -0.003] |
| ensemble_rank | K0 | +0.312 [+0.220, +0.374] | +0.133 [+0.088, +0.193] |
| ensemble_rank | TTC | +0.222 [+0.152, +0.271] | +0.099 [+0.064, +0.164] |
| ensemble_rank | TTC_in_cone | +0.117 [+0.082, +0.140] | +0.013 [+0.001, +0.037] |
| ensemble_rank | nearest | +0.008 [-0.013, +0.031] | +0.020 [-0.011, +0.033] |
| ensemble_rank | nearest_in_cone | +0.007 [-0.011, +0.035] | -0.015 [-0.032, +0.009] |
| ensemble_rank | gru | -0.006 [-0.017, +0.015] | -0.001 [-0.018, +0.028] |
| ensemble_rank | gru_attention | -0.010 [-0.018, +0.016] | +0.026 [-0.005, +0.055] |
| ensemble_stack | K0 | +0.322 [+0.215, +0.391] | +0.145 [+0.035, +0.202] |
| ensemble_stack | TTC | +0.232 [+0.150, +0.287] | +0.111 [+0.043, +0.167] |
| ensemble_stack | TTC_in_cone | +0.128 [+0.078, +0.156] | +0.026 [-0.042, +0.045] |
| ensemble_stack | nearest | +0.019 [-0.012, +0.035] | +0.032 [-0.065, +0.067] |
| ensemble_stack | nearest_in_cone | +0.018 [-0.006, +0.035] | -0.002 [-0.077, +0.014] |
| ensemble_stack | gru | +0.005 [-0.015, +0.016] | +0.011 [-0.051, +0.043] |
| ensemble_stack | gru_attention | +0.001 [-0.011, +0.013] | +0.039 [-0.031, +0.085] |
| ensemble_stack | ensemble_rank | +0.011 [-0.007, +0.019] | +0.012 [-0.067, +0.038] |

A difference is only credible if its CI excludes 0. With 71 test sessions the CIs are wide; the 5-fold cross-validation (ensemble_eval/) is the tighter development estimate.
