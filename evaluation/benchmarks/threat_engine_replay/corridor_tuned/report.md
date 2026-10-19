# Threat-engine replay - corridor_tuned

Label: in the walking corridor (width 0.75 m + class half-width, 0 < forward <= 3 m along the direction of travel from fixed_camera_poses.csv) within 20 steps (4 s); tracked objects only.

SANPO official test split: 83 sessions, 94 arrivals (a track first seen before it entered the zone). Operating points chosen on 47 validation sessions, then scored once on test. GRU: `models/GRU/GRU/runs/corridor_h8_seed0`. Generated 2026-10-19 by `python tools/replay_threat_engine.py` (1581 s).

## Opportunity window (first sighting -> arrival), test

Median 2.2 s; 29% of arrivals were first seen less than 1 s before arriving, 46% less than 2 s, 26% at least 4 s before. Lead time can't exceed the window, so lead is reported conditionally below.

## Operating point `budget_2` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||||
| nearest_in_corridor | 0.95 / cd 2 / sm 1 / st 0.6 | 0.07 [0.02, 0.17] | 3.59 [1.86, 5.63] | 0.37 [0.21, 0.51] | 0.32 [0.16, 0.49] | 5.71 [2.98, 8.67] | 3.68 [1.23, 6.55] | 0.10 [0.03, 0.25] | 0.08 [0.00, 0.18] | 0.00 [0.00, 1.00] | 0.03 [0.00, 0.09] | 0.02 [0.00, 0.07] |
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | — | no setting meets the rule on validation |||||||||||
| v2_kalman_nopose | — | no setting meets the rule on validation |||||||||||
| v2_gru | 0.9 / cd 4 / sm 1 / st 0.6 | 0.15 [0.08, 0.28] | 3.87 [2.01, 6.17] | 0.64 [0.49, 0.77] | 0.60 [0.42, 0.74] | 10.87 [5.88, 16.07] | 6.91 [2.55, 11.64] | 0.18 [0.09, 0.36] | 0.14 [0.05, 0.26] | 0.00 [0.00, 0.20] | 0.04 [0.00, 0.11] | 0.02 [0.00, 0.07] |
| v2_gru_nopose | 0.8 | 0.13 [0.06, 0.24] | 3.32 [1.39, 5.70] | 0.68 [0.52, 0.83] | 0.67 [0.50, 0.82] | 10.50 [5.38, 15.37] | 6.72 [2.50, 10.94] | 0.15 [0.07, 0.31] | 0.12 [0.04, 0.23] | 0.00 [0.00, 0.10] | 0.02 [0.00, 0.06] | 0.00 [0.00, 0.00] |

## Operating point `budget_5` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||||
| nearest_in_corridor | 0.5 / cd 8 / sm 2 / st 0.6 | 0.17 [0.09, 0.30] | 7.37 [4.43, 10.34] | 0.44 [0.32, 0.55] | 0.39 [0.27, 0.49] | 13.08 [7.98, 18.45] | 7.74 [3.34, 12.30] | 0.22 [0.12, 0.43] | 0.18 [0.08, 0.32] | 0.00 [0.00, 0.40] | 0.07 [0.03, 0.16] | 0.03 [0.00, 0.09] |
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | 0.7 / cd 4 / sm 1 / st 0.6 | 0.27 [0.14, 0.46] | 7.18 [4.70, 9.85] | 0.54 [0.43, 0.63] | 0.43 [0.32, 0.52] | 15.66 [10.10, 21.48] | 9.76 [4.75, 15.04] | 0.30 [0.14, 0.53] | 0.27 [0.11, 0.52] | 0.50 [0.00, 1.00] | 0.17 [0.08, 0.28] | 0.07 [0.02, 0.14] |
| v2_kalman_nopose | 0.7 | 0.24 [0.14, 0.47] | 6.08 [3.38, 8.86] | 0.56 [0.45, 0.67] | 0.49 [0.37, 0.62] | 13.82 [8.49, 19.09] | 8.01 [3.73, 12.50] | 0.24 [0.13, 0.47] | 0.20 [0.08, 0.47] | 0.00 [0.00, 0.40] | 0.07 [0.02, 0.17] | 0.01 [0.00, 0.04] |
| v2_gru | 0.5 / cd 4 / sm 1 / st 1.2 | 0.36 [0.26, 0.56] | 8.01 [5.40, 10.88] | 0.55 [0.43, 0.64] | 0.44 [0.33, 0.54] | 17.68 [11.48, 23.86] | 11.33 [5.79, 17.09] | 0.31 [0.22, 0.53] | 0.25 [0.14, 0.44] | 0.20 [0.00, 1.00] | 0.19 [0.11, 0.36] | 0.06 [0.01, 0.13] |
| v2_gru_nopose | 0.5 | 0.36 [0.26, 0.57] | 5.99 [3.76, 8.47] | 0.61 [0.49, 0.71] | 0.50 [0.38, 0.60] | 15.20 [9.34, 20.75] | 9.67 [4.85, 14.66] | 0.31 [0.23, 0.54] | 0.25 [0.18, 0.43] | 0.20 [0.00, 0.80] | 0.18 [0.11, 0.33] | 0.04 [0.00, 0.10] |

## Operating point `budget_10` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||||
| v1_k0_announcer | 12 | 0.39 [0.28, 0.59] | 10.22 [7.68, 12.65] | 0.39 [0.25, 0.50] | 0.27 [0.17, 0.36] | 16.86 [12.26, 21.41] | 11.70 [7.36, 16.07] | 0.39 [0.27, 0.60] | 0.33 [0.20, 0.51] | 0.50 [0.00, 0.80] | 0.20 [0.11, 0.31] | 0.11 [0.05, 0.17] |
| nearest_in_corridor | 0.3 / cd 8 / sm 2 / st 0.6 | 0.27 [0.16, 0.44] | 10.96 [7.93, 14.36] | 0.39 [0.30, 0.47] | 0.32 [0.22, 0.40] | 18.05 [12.47, 23.97] | 11.15 [6.27, 16.26] | 0.31 [0.20, 0.54] | 0.29 [0.17, 0.50] | 0.20 [0.10, 0.80] | 0.16 [0.08, 0.28] | 0.07 [0.03, 0.13] |
| v2_physics | 0.95 | 0.26 [0.16, 0.46] | 13.17 [9.98, 16.63] | 0.36 [0.27, 0.44] | 0.29 [0.20, 0.37] | 20.63 [15.22, 26.27] | 13.63 [8.77, 18.53] | 0.28 [0.16, 0.61] | 0.25 [0.14, 0.53] | 0.20 [0.00, 1.00] | 0.15 [0.07, 0.30] | 0.06 [0.01, 0.14] |
| v2_physics_nopose | 0.95 | 0.30 [0.19, 0.51] | 10.13 [7.31, 13.16] | 0.42 [0.33, 0.52] | 0.35 [0.25, 0.45] | 17.59 [12.67, 22.63] | 11.05 [6.68, 15.58] | 0.30 [0.16, 0.62] | 0.25 [0.11, 0.54] | 0.10 [0.00, 0.60] | 0.14 [0.07, 0.29] | 0.04 [0.00, 0.11] |
| v2_kalman | 0.5 / cd 8 / sm 1 / st 0.6 | 0.31 [0.19, 0.53] | 11.33 [8.63, 14.09] | 0.44 [0.34, 0.52] | 0.35 [0.25, 0.43] | 20.26 [14.67, 26.02] | 13.45 [8.35, 18.87] | 0.33 [0.18, 0.65] | 0.27 [0.12, 0.60] | 0.60 [0.10, 1.00] | 0.19 [0.10, 0.31] | 0.09 [0.03, 0.15] |
| v2_kalman_nopose | 0.5 | 0.33 [0.21, 0.61] | 9.86 [6.91, 12.93] | 0.46 [0.35, 0.56] | 0.37 [0.25, 0.47] | 18.15 [13.00, 23.41] | 11.05 [6.65, 15.36] | 0.33 [0.19, 0.75] | 0.27 [0.12, 0.75] | 0.10 [0.00, 0.80] | 0.15 [0.06, 0.33] | 0.05 [0.00, 0.15] |
| v2_gru | 0.5 / cd 4 / sm 1 / st 1.2 | 0.36 [0.26, 0.56] | 8.01 [5.40, 10.88] | 0.55 [0.43, 0.64] | 0.44 [0.33, 0.54] | 17.68 [11.48, 23.86] | 11.33 [5.79, 17.09] | 0.31 [0.22, 0.53] | 0.25 [0.14, 0.44] | 0.20 [0.00, 1.00] | 0.19 [0.11, 0.36] | 0.06 [0.01, 0.13] |
| v2_gru_nopose | 0.5 | 0.36 [0.26, 0.57] | 5.99 [3.76, 8.47] | 0.61 [0.49, 0.71] | 0.50 [0.38, 0.60] | 15.20 [9.34, 20.75] | 9.67 [4.85, 14.66] | 0.31 [0.23, 0.54] | 0.25 [0.18, 0.43] | 0.20 [0.00, 0.80] | 0.18 [0.11, 0.33] | 0.04 [0.00, 0.10] |

## Operating point `early_budget_2` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||||
| nearest_in_corridor | 0.95 / cd 2 / sm 1 / st 0.6 | 0.07 [0.02, 0.17] | 3.59 [1.86, 5.63] | 0.37 [0.21, 0.51] | 0.32 [0.16, 0.49] | 5.71 [2.98, 8.67] | 3.68 [1.23, 6.55] | 0.10 [0.03, 0.25] | 0.08 [0.00, 0.18] | 0.00 [0.00, 1.00] | 0.03 [0.00, 0.09] | 0.02 [0.00, 0.07] |
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | — | no setting meets the rule on validation |||||||||||
| v2_kalman_nopose | — | no setting meets the rule on validation |||||||||||
| v2_gru | 0.9 / cd 4 / sm 1 / st 0.6 | 0.15 [0.08, 0.28] | 3.87 [2.01, 6.17] | 0.64 [0.49, 0.77] | 0.60 [0.42, 0.74] | 10.87 [5.88, 16.07] | 6.91 [2.55, 11.64] | 0.18 [0.09, 0.36] | 0.14 [0.05, 0.26] | 0.00 [0.00, 0.20] | 0.04 [0.00, 0.11] | 0.02 [0.00, 0.07] |
| v2_gru_nopose | 0.7 | 0.20 [0.11, 0.34] | 3.87 [1.60, 6.51] | 0.67 [0.52, 0.82] | 0.63 [0.49, 0.77] | 11.70 [6.21, 16.78] | 7.28 [2.92, 11.79] | 0.18 [0.09, 0.36] | 0.14 [0.05, 0.29] | 0.00 [0.00, 0.80] | 0.06 [0.02, 0.14] | 0.01 [0.00, 0.04] |

## Operating point `early_budget_5` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||||
| nearest_in_corridor | 0.5 / cd 8 / sm 2 / st 0.6 | 0.17 [0.09, 0.30] | 7.37 [4.43, 10.34] | 0.44 [0.32, 0.55] | 0.39 [0.27, 0.49] | 13.08 [7.98, 18.45] | 7.74 [3.34, 12.30] | 0.22 [0.12, 0.43] | 0.18 [0.08, 0.32] | 0.00 [0.00, 0.40] | 0.07 [0.03, 0.16] | 0.03 [0.00, 0.09] |
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | 0.7 / cd 4 / sm 1 / st 0.6 | 0.27 [0.14, 0.46] | 7.18 [4.70, 9.85] | 0.54 [0.43, 0.63] | 0.43 [0.32, 0.52] | 15.66 [10.10, 21.48] | 9.76 [4.75, 15.04] | 0.30 [0.14, 0.53] | 0.27 [0.11, 0.52] | 0.50 [0.00, 1.00] | 0.17 [0.08, 0.28] | 0.07 [0.02, 0.14] |
| v2_kalman_nopose | 0.7 | 0.24 [0.14, 0.47] | 6.08 [3.38, 8.86] | 0.56 [0.45, 0.67] | 0.49 [0.37, 0.62] | 13.82 [8.49, 19.09] | 8.01 [3.73, 12.50] | 0.24 [0.13, 0.47] | 0.20 [0.08, 0.47] | 0.00 [0.00, 0.40] | 0.07 [0.02, 0.17] | 0.01 [0.00, 0.04] |
| v2_gru | 0.5 / cd 4 / sm 1 / st 0.6 | 0.37 [0.26, 0.57] | 7.00 [4.75, 9.47] | 0.58 [0.47, 0.66] | 0.48 [0.35, 0.57] | 16.67 [10.77, 22.79] | 10.41 [5.33, 15.92] | 0.33 [0.22, 0.54] | 0.27 [0.15, 0.45] | 0.20 [0.00, 0.80] | 0.19 [0.11, 0.36] | 0.06 [0.01, 0.13] |
| v2_gru_nopose | 0.5 | 0.36 [0.26, 0.57] | 5.99 [3.76, 8.47] | 0.61 [0.49, 0.71] | 0.50 [0.38, 0.60] | 15.20 [9.34, 20.75] | 9.67 [4.85, 14.66] | 0.31 [0.23, 0.54] | 0.25 [0.18, 0.43] | 0.20 [0.00, 0.80] | 0.18 [0.11, 0.33] | 0.04 [0.00, 0.10] |

## Operating point `early_budget_10` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||||
| v1_k0_announcer | 12 | 0.39 [0.28, 0.59] | 10.22 [7.68, 12.65] | 0.39 [0.25, 0.50] | 0.27 [0.17, 0.36] | 16.86 [12.26, 21.41] | 11.70 [7.36, 16.07] | 0.39 [0.27, 0.60] | 0.33 [0.20, 0.51] | 0.50 [0.00, 0.80] | 0.20 [0.11, 0.31] | 0.11 [0.05, 0.17] |
| nearest_in_corridor | 0.3 / cd 8 / sm 2 / st 0.6 | 0.27 [0.16, 0.44] | 10.96 [7.93, 14.36] | 0.39 [0.30, 0.47] | 0.32 [0.22, 0.40] | 18.05 [12.47, 23.97] | 11.15 [6.27, 16.26] | 0.31 [0.20, 0.54] | 0.29 [0.17, 0.50] | 0.20 [0.10, 0.80] | 0.16 [0.08, 0.28] | 0.07 [0.03, 0.13] |
| v2_physics | 0.95 | 0.26 [0.16, 0.46] | 13.17 [9.98, 16.63] | 0.36 [0.27, 0.44] | 0.29 [0.20, 0.37] | 20.63 [15.22, 26.27] | 13.63 [8.77, 18.53] | 0.28 [0.16, 0.61] | 0.25 [0.14, 0.53] | 0.20 [0.00, 1.00] | 0.15 [0.07, 0.30] | 0.06 [0.01, 0.14] |
| v2_physics_nopose | 0.95 | 0.30 [0.19, 0.51] | 10.13 [7.31, 13.16] | 0.42 [0.33, 0.52] | 0.35 [0.25, 0.45] | 17.59 [12.67, 22.63] | 11.05 [6.68, 15.58] | 0.30 [0.16, 0.62] | 0.25 [0.11, 0.54] | 0.10 [0.00, 0.60] | 0.14 [0.07, 0.29] | 0.04 [0.00, 0.11] |
| v2_kalman | 0.5 / cd 2 / sm 1 / st 0.6 | 0.32 [0.20, 0.56] | 12.07 [9.25, 15.12] | 0.45 [0.34, 0.54] | 0.35 [0.25, 0.44] | 22.01 [15.97, 28.23] | 13.45 [8.13, 18.99] | 0.34 [0.19, 0.70] | 0.29 [0.14, 0.67] | 0.60 [0.00, 1.00] | 0.19 [0.10, 0.31] | 0.09 [0.03, 0.15] |
| v2_kalman_nopose | 0.6 | 0.31 [0.20, 0.56] | 8.20 [5.33, 11.16] | 0.50 [0.39, 0.60] | 0.41 [0.31, 0.51] | 16.30 [11.02, 21.67] | 9.76 [5.51, 14.07] | 0.31 [0.19, 0.65] | 0.25 [0.12, 0.67] | 0.00 [0.00, 0.50] | 0.13 [0.06, 0.27] | 0.04 [0.00, 0.12] |
| v2_gru | 0.4 / cd 2 / sm 1 / st 1.2 | 0.34 [0.21, 0.61] | 10.32 [7.74, 13.35] | 0.51 [0.39, 0.60] | 0.38 [0.26, 0.48] | 20.91 [14.37, 27.11] | 12.62 [7.05, 18.24] | 0.30 [0.18, 0.62] | 0.25 [0.12, 0.55] | 0.20 [0.00, 0.90] | 0.22 [0.10, 0.46] | 0.05 [0.01, 0.12] |
| v2_gru_nopose | 0.4 | 0.37 [0.24, 0.64] | 7.46 [5.25, 10.02] | 0.56 [0.44, 0.67] | 0.45 [0.31, 0.57] | 17.04 [11.50, 22.46] | 10.68 [5.99, 15.86] | 0.33 [0.21, 0.65] | 0.29 [0.19, 0.62] | 0.20 [0.00, 0.40] | 0.22 [0.11, 0.48] | 0.03 [0.00, 0.09] |

## Operating point `recall_50` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | 0.75 | 0.66 [0.57, 0.83] | 88.89 [74.49, 105.34] | 0.23 [0.12, 0.33] | 0.13 [0.06, 0.20] | 114.77 [98.89, 131.31] | 44.67 [35.99, 52.48] | 0.63 [0.48, 0.82] | 0.55 [0.33, 0.75] | 1.10 [0.60, 1.60] | 0.46 [0.37, 0.60] | 0.24 [0.15, 0.36] |
| v1_k0_announcer | 0.5 | 0.59 [0.49, 0.77] | 32.61 [28.99, 36.80] | 0.24 [0.13, 0.33] | 0.13 [0.06, 0.20] | 42.74 [37.90, 47.56] | 32.88 [27.90, 37.46] | 0.54 [0.41, 0.74] | 0.49 [0.31, 0.70] | 1.10 [0.80, 2.00] | 0.40 [0.31, 0.55] | 0.23 [0.14, 0.35] |
| nearest_in_corridor | — | no setting meets the rule on validation |||||||||||
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | — | no setting meets the rule on validation |||||||||||
| v2_kalman_nopose | — | no setting meets the rule on validation |||||||||||
| v2_gru | — | no setting meets the rule on validation |||||||||||
| v2_gru_nopose | — | no setting meets the rule on validation |||||||||||

## Kaplan-Meier: share of arrivals not yet warned, by time since first sighting (test, `budget_5`)

| Arm | 0.5 s | 1 s | 2 s | 3 s | 4 s |
|---|---:|---:|---:|---:|---:|
| v1_k0 | n/a |
| v1_k0_announcer | n/a |
| nearest_in_corridor | 0.99 | 0.95 | 0.87 | 0.85 | 0.85 |
| v2_physics | n/a |
| v2_physics_nopose | n/a |
| v2_kalman | 0.94 | 0.89 | 0.83 | 0.79 | 0.74 |
| v2_kalman_nopose | 0.97 | 0.90 | 0.83 | 0.79 | 0.71 |
| v2_gru | 0.85 | 0.80 | 0.72 | 0.68 | 0.64 |
| v2_gru_nopose | 0.86 | 0.80 | 0.68 | 0.65 | 0.58 |

## Kaplan-Meier: share of arrivals not yet warned, by time since first sighting (test, `early_budget_5`)

| Arm | 0.5 s | 1 s | 2 s | 3 s | 4 s |
|---|---:|---:|---:|---:|---:|
| v1_k0 | n/a |
| v1_k0_announcer | n/a |
| nearest_in_corridor | 0.99 | 0.95 | 0.87 | 0.85 | 0.85 |
| v2_physics | n/a |
| v2_physics_nopose | n/a |
| v2_kalman | 0.94 | 0.89 | 0.83 | 0.79 | 0.74 |
| v2_kalman_nopose | 0.97 | 0.90 | 0.83 | 0.79 | 0.71 |
| v2_gru | 0.85 | 0.80 | 0.72 | 0.68 | 0.61 |
| v2_gru_nopose | 0.86 | 0.80 | 0.68 | 0.65 | 0.58 |


## Operating curves (test; every knob value)

Recall (warned) against false alerts per minute; the CSV next to this report has every column.

| Arm | knob | warned | warned ≥ 0.2 s before | false/min | precision | alerts/min |
|---|---:|---:|---:|---:|---:|---:|
| v1_k0 | 0.05 | 0.83 | 0.60 | 137.1 | 0.22 | 175.8 |
| v1_k0 | 0.1 | 0.83 | 0.60 | 129.1 | 0.23 | 166.8 |
| v1_k0 | 0.2 | 0.82 | 0.60 | 118.5 | 0.23 | 154.7 |
| v1_k0 | 0.35 | 0.78 | 0.55 | 110.2 | 0.23 | 143.6 |
| v1_k0 | 0.5 | 0.73 | 0.52 | 102.7 | 0.23 | 132.6 |
| v1_k0 | 0.75 | 0.66 | 0.46 | 88.9 | 0.23 | 114.8 |
| v1_k0 | 1 | 0.59 | 0.43 | 77.3 | 0.23 | 100.1 |
| v1_k0 | 1.5 | 0.51 | 0.36 | 64.9 | 0.23 | 84.7 |
| v1_k0 | 2 | 0.48 | 0.34 | 53.9 | 0.25 | 71.8 |
| v1_k0 | 3 | 0.43 | 0.27 | 40.3 | 0.28 | 55.7 |
| v1_k0 | 5 | 0.43 | 0.23 | 27.0 | 0.35 | 41.5 |
| v1_k0 | 8 | 0.40 | 0.21 | 21.7 | 0.39 | 35.9 |
| v1_k0 | 12 | 0.39 | 0.20 | 18.5 | 0.43 | 32.5 |
| v1_k0_announcer | 0.05 | 0.61 | 0.44 | 39.9 | 0.21 | 50.8 |
| v1_k0_announcer | 0.1 | 0.61 | 0.44 | 38.3 | 0.22 | 49.2 |
| v1_k0_announcer | 0.2 | 0.60 | 0.44 | 35.8 | 0.23 | 46.4 |
| v1_k0_announcer | 0.35 | 0.57 | 0.41 | 34.4 | 0.24 | 45.0 |
| v1_k0_announcer | 0.5 | 0.59 | 0.40 | 32.6 | 0.24 | 42.7 |
| v1_k0_announcer | 0.75 | 0.56 | 0.35 | 30.0 | 0.24 | 39.3 |
| v1_k0_announcer | 1 | 0.51 | 0.34 | 28.0 | 0.24 | 36.8 |
| v1_k0_announcer | 1.5 | 0.47 | 0.29 | 24.5 | 0.25 | 32.7 |
| v1_k0_announcer | 2 | 0.46 | 0.28 | 21.1 | 0.27 | 28.9 |
| v1_k0_announcer | 3 | 0.43 | 0.27 | 17.9 | 0.29 | 25.1 |
| v1_k0_announcer | 5 | 0.43 | 0.23 | 13.3 | 0.35 | 20.3 |
| v1_k0_announcer | 8 | 0.40 | 0.21 | 11.2 | 0.37 | 18.0 |
| v1_k0_announcer | 12 | 0.39 | 0.20 | 10.2 | 0.39 | 16.9 |
| v2_physics | 0.01 | 0.28 | 0.18 | 18.0 | 0.29 | 25.4 |
| v2_physics | 0.03 | 0.28 | 0.18 | 18.0 | 0.29 | 25.4 |
| v2_physics | 0.05 | 0.28 | 0.18 | 18.0 | 0.29 | 25.4 |
| v2_physics | 0.1 | 0.28 | 0.18 | 18.0 | 0.29 | 25.4 |
| v2_physics | 0.15 | 0.28 | 0.18 | 18.0 | 0.29 | 25.4 |
| v2_physics | 0.2 | 0.28 | 0.18 | 18.0 | 0.29 | 25.4 |
| v2_physics | 0.3 | 0.28 | 0.17 | 17.7 | 0.29 | 25.1 |
| v2_physics | 0.4 | 0.28 | 0.17 | 17.4 | 0.30 | 24.8 |
| v2_physics | 0.5 | 0.26 | 0.15 | 17.0 | 0.30 | 24.2 |
| v2_physics | 0.6 | 0.26 | 0.16 | 16.4 | 0.30 | 23.6 |
| v2_physics | 0.7 | 0.27 | 0.15 | 15.8 | 0.32 | 23.0 |
| v2_physics | 0.8 | 0.28 | 0.16 | 15.1 | 0.33 | 22.5 |
| v2_physics | 0.9 | 0.28 | 0.15 | 13.9 | 0.35 | 21.5 |
| v2_physics | 0.95 | 0.26 | 0.15 | 13.2 | 0.36 | 20.6 |
| v2_physics_nopose | 0.01 | 0.30 | 0.19 | 14.6 | 0.33 | 21.8 |
| v2_physics_nopose | 0.03 | 0.30 | 0.19 | 14.6 | 0.33 | 21.8 |
| v2_physics_nopose | 0.05 | 0.30 | 0.19 | 14.6 | 0.33 | 21.8 |
| v2_physics_nopose | 0.1 | 0.30 | 0.19 | 14.6 | 0.33 | 21.8 |
| v2_physics_nopose | 0.15 | 0.30 | 0.19 | 14.6 | 0.33 | 21.8 |
| v2_physics_nopose | 0.2 | 0.30 | 0.19 | 14.6 | 0.33 | 21.8 |
| v2_physics_nopose | 0.3 | 0.31 | 0.19 | 14.2 | 0.34 | 21.6 |
| v2_physics_nopose | 0.4 | 0.31 | 0.18 | 13.7 | 0.35 | 21.2 |
| v2_physics_nopose | 0.5 | 0.31 | 0.18 | 13.7 | 0.35 | 21.2 |
| v2_physics_nopose | 0.6 | 0.31 | 0.18 | 12.9 | 0.37 | 20.4 |
| v2_physics_nopose | 0.7 | 0.32 | 0.18 | 12.3 | 0.38 | 19.9 |
| v2_physics_nopose | 0.8 | 0.33 | 0.19 | 12.0 | 0.39 | 19.6 |
| v2_physics_nopose | 0.9 | 0.32 | 0.17 | 10.8 | 0.41 | 18.4 |
| v2_physics_nopose | 0.95 | 0.30 | 0.14 | 10.1 | 0.42 | 17.6 |
| v2_kalman_nopose | 0.01 | 0.33 | 0.26 | 44.9 | 0.16 | 53.4 |
| v2_kalman_nopose | 0.03 | 0.33 | 0.26 | 38.2 | 0.18 | 46.8 |
| v2_kalman_nopose | 0.05 | 0.32 | 0.24 | 33.1 | 0.20 | 41.5 |
| v2_kalman_nopose | 0.1 | 0.29 | 0.22 | 27.4 | 0.23 | 35.6 |
| v2_kalman_nopose | 0.15 | 0.33 | 0.24 | 22.0 | 0.28 | 30.5 |
| v2_kalman_nopose | 0.2 | 0.32 | 0.21 | 19.7 | 0.30 | 28.0 |
| v2_kalman_nopose | 0.3 | 0.33 | 0.22 | 15.0 | 0.36 | 23.5 |
| v2_kalman_nopose | 0.4 | 0.33 | 0.19 | 11.4 | 0.43 | 20.1 |
| v2_kalman_nopose | 0.5 | 0.33 | 0.15 | 9.9 | 0.46 | 18.1 |
| v2_kalman_nopose | 0.6 | 0.31 | 0.13 | 8.2 | 0.50 | 16.3 |
| v2_kalman_nopose | 0.7 | 0.24 | 0.07 | 6.1 | 0.56 | 13.8 |
| v2_kalman_nopose | 0.8 | 0.18 | 0.06 | 5.3 | 0.56 | 12.0 |
| v2_kalman_nopose | 0.9 | 0.13 | 0.03 | 4.6 | 0.58 | 10.9 |
| v2_kalman_nopose | 0.95 | 0.12 | 0.04 | 3.9 | 0.60 | 9.8 |
| v2_gru_nopose | 0.01 | 0.40 | 0.30 | 41.0 | 0.20 | 50.9 |
| v2_gru_nopose | 0.03 | 0.40 | 0.28 | 31.7 | 0.24 | 41.9 |
| v2_gru_nopose | 0.05 | 0.39 | 0.26 | 28.1 | 0.27 | 38.2 |
| v2_gru_nopose | 0.1 | 0.39 | 0.28 | 20.9 | 0.33 | 31.0 |
| v2_gru_nopose | 0.15 | 0.39 | 0.28 | 17.2 | 0.36 | 26.9 |
| v2_gru_nopose | 0.2 | 0.41 | 0.30 | 14.1 | 0.41 | 24.0 |
| v2_gru_nopose | 0.3 | 0.40 | 0.28 | 10.3 | 0.49 | 20.2 |
| v2_gru_nopose | 0.4 | 0.37 | 0.22 | 7.5 | 0.56 | 17.0 |
| v2_gru_nopose | 0.5 | 0.36 | 0.18 | 6.0 | 0.61 | 15.2 |
| v2_gru_nopose | 0.6 | 0.32 | 0.10 | 5.0 | 0.63 | 13.4 |
| v2_gru_nopose | 0.7 | 0.20 | 0.06 | 3.9 | 0.67 | 11.7 |
| v2_gru_nopose | 0.8 | 0.13 | 0.02 | 3.3 | 0.68 | 10.5 |
| v2_gru_nopose | 0.9 | 0.10 | 0.02 | 2.7 | 0.70 | 8.8 |
| v2_gru_nopose | 0.95 | 0.06 | 0.02 | 2.3 | 0.70 | 7.6 |

Classes: true = the object enters the zone later within the horizon; late = it is already in the zone; false = it doesn't enter within the horizon. precision = (true + late) / all. warned = an announcement about the arriving track in the 4 s up to and including arrival.
Open-loop: the recorded walker never heard these alerts. Labels and inputs share SANPO's depth.
