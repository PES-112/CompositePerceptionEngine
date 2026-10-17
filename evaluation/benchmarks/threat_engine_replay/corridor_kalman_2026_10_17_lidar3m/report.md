# Threat-engine replay - corridor_kalman_2026_10_17_lidar3m

Label: in the walking corridor (width 0.75 m + class half-width, 0 < forward <= 3 m along the direction of travel from fixed_camera_poses.csv) within 20 steps (4 s); tracked objects only.

SANPO official test split: 71 sessions, 63 arrivals (a track first seen before it entered the zone). Operating points chosen on 41 validation sessions, then scored once on test. GRU: `models/GRU/GRU/runs/corridor_h8_seed0`. Generated 2026-10-17 by `python tools/replay_threat_engine.py` (205 s).

Depth: LiDAR-degraded inputs (`--lidar-range 3`, seed 0): rows beyond 3 m and 5% at random dropped, noise N(0, (0.01 + 0.01·d)·(1 + 3·min(1, |bearing|/30°)²)) m, closing speed re-fitted over 6 steps. Labels, arrivals and windows use SANPO depth.

## Opportunity window (first sighting -> arrival), test

Median 2.6 s; 25% of arrivals were first seen less than 1 s before arriving, 38% less than 2 s, 27% at least 4 s before. Lead time can't exceed the window, so lead is reported conditionally below.

## Operating point `budget_2` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | 1.5 | 0.06 [0.00, 0.13] | 0.41 [0.00, 1.04] | 0.88 [0.73, 1.00] | 0.73 [0.57, 1.00] | 3.59 [0.67, 7.16] | 2.34 [0.30, 4.89] | 0.09 [0.00, 0.18] | 0.05 [0.00, 0.08] | 0.80 [0.60, 1.00] | 0.06 [0.00, 0.13] | 0.03 [0.00, 0.09] |
| v1_k0_announcer | 0.35 | 0.06 [0.00, 0.13] | 0.41 [0.00, 1.04] | 0.89 [0.73, 1.00] | 0.68 [0.50, 1.00] | 3.86 [0.72, 7.90] | 2.90 [0.30, 6.54] | 0.09 [0.00, 0.18] | 0.05 [0.00, 0.08] | 1.10 [0.00, 1.20] | 0.05 [0.00, 0.13] | 0.05 [0.00, 0.13] |
| nearest_in_corridor | 0.9 | 0.03 [0.00, 0.07] | 0.14 [0.00, 0.43] | 0.90 [0.75, 1.00] | 0.80 [0.50, 1.00] | 1.38 [0.15, 2.88] | 0.69 [0.00, 1.80] | 0.04 [0.00, 0.10] | 0.03 [0.00, 0.11] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | — | no setting meets the rule on validation |||||||||||
| v2_kalman_nopose | — | no setting meets the rule on validation |||||||||||
| v2_gru | 0.8 | 0.17 [0.10, 0.34] | 0.41 [0.00, 0.88] | 0.92 [0.82, 1.00] | 0.84 [0.77, 0.93] | 5.24 [2.18, 8.46] | 3.59 [1.14, 6.33] | 0.17 [0.10, 0.31] | 0.18 [0.09, 0.33] | 0.00 [0.00, 0.40] | 0.02 [0.00, 0.04] | 0.00 [0.00, 0.00] |
| v2_gru_nopose | 0.7 | 0.24 [0.16, 0.44] | 1.10 [0.20, 2.09] | 0.84 [0.73, 0.95] | 0.78 [0.68, 0.91] | 6.90 [3.09, 10.92] | 5.10 [1.58, 9.11] | 0.21 [0.14, 0.38] | 0.23 [0.13, 0.45] | 0.00 [0.00, 0.60] | 0.03 [0.00, 0.09] | 0.00 [0.00, 0.00] |

## Operating point `budget_5` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | 0.2 | 0.08 [0.00, 0.18] | 1.66 [0.00, 4.07] | 0.81 [0.67, 1.00] | 0.63 [0.47, 1.00] | 8.69 [1.36, 18.46] | 4.97 [0.45, 11.50] | 0.11 [0.00, 0.24] | 0.08 [0.00, 0.15] | 1.00 [0.60, 1.10] | 0.06 [0.00, 0.13] | 0.05 [0.00, 0.13] |
| v1_k0_announcer | 0.35 | 0.06 [0.00, 0.13] | 0.41 [0.00, 1.04] | 0.89 [0.73, 1.00] | 0.68 [0.50, 1.00] | 3.86 [0.72, 7.90] | 2.90 [0.30, 6.54] | 0.09 [0.00, 0.18] | 0.05 [0.00, 0.08] | 1.10 [0.00, 1.20] | 0.05 [0.00, 0.13] | 0.05 [0.00, 0.13] |
| nearest_in_corridor | 0.01 | 0.41 [0.30, 0.71] | 2.34 [0.73, 4.13] | 0.77 [0.68, 0.89] | 0.69 [0.55, 0.87] | 10.21 [5.64, 15.01] | 6.76 [2.71, 11.10] | 0.36 [0.22, 0.69] | 0.36 [0.20, 0.67] | 0.00 [0.00, 0.10] | 0.05 [0.00, 0.13] | 0.00 [0.00, 0.00] |
| v2_physics | 0.01 | 0.38 [0.25, 0.71] | 2.07 [0.78, 3.44] | 0.79 [0.68, 0.90] | 0.71 [0.59, 0.84] | 9.93 [5.69, 14.33] | 6.62 [2.58, 10.90] | 0.32 [0.18, 0.69] | 0.31 [0.17, 0.65] | 0.00 [0.00, 0.00] | 0.03 [0.00, 0.09] | 0.00 [0.00, 0.00] |
| v2_physics_nopose | 0.01 | 0.44 [0.33, 0.72] | 2.21 [0.95, 3.55] | 0.76 [0.67, 0.86] | 0.69 [0.58, 0.82] | 9.38 [5.47, 13.12] | 5.93 [2.64, 9.21] | 0.45 [0.32, 0.71] | 0.46 [0.35, 0.67] | 0.00 [0.00, 0.00] | 0.06 [0.00, 0.12] | 0.00 [0.00, 0.00] |
| v2_kalman | 0.2 | 0.40 [0.27, 0.73] | 4.14 [2.16, 6.44] | 0.66 [0.55, 0.76] | 0.59 [0.49, 0.72] | 12.14 [6.84, 17.43] | 8.14 [3.63, 12.95] | 0.38 [0.25, 0.76] | 0.38 [0.27, 0.71] | 0.00 [0.00, 0.00] | 0.03 [0.00, 0.05] | 0.00 [0.00, 0.00] |
| v2_kalman_nopose | 0.4 | 0.37 [0.26, 0.63] | 3.03 [1.58, 4.42] | 0.71 [0.62, 0.79] | 0.62 [0.53, 0.75] | 10.48 [6.09, 14.85] | 6.90 [3.11, 10.92] | 0.36 [0.26, 0.64] | 0.36 [0.26, 0.57] | 0.00 [0.00, 0.00] | 0.05 [0.00, 0.09] | 0.00 [0.00, 0.00] |
| v2_gru | 0.3 | 0.41 [0.31, 0.65] | 3.45 [1.71, 5.22] | 0.71 [0.57, 0.82] | 0.65 [0.54, 0.75] | 11.86 [6.66, 17.18] | 8.41 [3.65, 13.38] | 0.45 [0.33, 0.69] | 0.49 [0.38, 0.71] | 0.00 [0.00, 0.00] | 0.05 [0.00, 0.09] | 0.00 [0.00, 0.00] |
| v2_gru_nopose | 0.3 | 0.41 [0.31, 0.65] | 3.45 [1.71, 5.22] | 0.70 [0.56, 0.81] | 0.64 [0.53, 0.74] | 11.59 [6.57, 16.61] | 8.14 [3.63, 12.82] | 0.45 [0.33, 0.69] | 0.49 [0.38, 0.71] | 0.00 [0.00, 0.00] | 0.05 [0.00, 0.09] | 0.00 [0.00, 0.00] |

## Operating point `budget_10` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | 0.2 | 0.08 [0.00, 0.18] | 1.66 [0.00, 4.07] | 0.81 [0.67, 1.00] | 0.63 [0.47, 1.00] | 8.69 [1.36, 18.46] | 4.97 [0.45, 11.50] | 0.11 [0.00, 0.24] | 0.08 [0.00, 0.15] | 1.00 [0.60, 1.10] | 0.06 [0.00, 0.13] | 0.05 [0.00, 0.13] |
| v1_k0_announcer | 0.35 | 0.06 [0.00, 0.13] | 0.41 [0.00, 1.04] | 0.89 [0.73, 1.00] | 0.68 [0.50, 1.00] | 3.86 [0.72, 7.90] | 2.90 [0.30, 6.54] | 0.09 [0.00, 0.18] | 0.05 [0.00, 0.08] | 1.10 [0.00, 1.20] | 0.05 [0.00, 0.13] | 0.05 [0.00, 0.13] |
| nearest_in_corridor | 0.01 | 0.41 [0.30, 0.71] | 2.34 [0.73, 4.13] | 0.77 [0.68, 0.89] | 0.69 [0.55, 0.87] | 10.21 [5.64, 15.01] | 6.76 [2.71, 11.10] | 0.36 [0.22, 0.69] | 0.36 [0.20, 0.67] | 0.00 [0.00, 0.10] | 0.05 [0.00, 0.13] | 0.00 [0.00, 0.00] |
| v2_physics | 0.01 | 0.38 [0.25, 0.71] | 2.07 [0.78, 3.44] | 0.79 [0.68, 0.90] | 0.71 [0.59, 0.84] | 9.93 [5.69, 14.33] | 6.62 [2.58, 10.90] | 0.32 [0.18, 0.69] | 0.31 [0.17, 0.65] | 0.00 [0.00, 0.00] | 0.03 [0.00, 0.09] | 0.00 [0.00, 0.00] |
| v2_physics_nopose | 0.01 | 0.44 [0.33, 0.72] | 2.21 [0.95, 3.55] | 0.76 [0.67, 0.86] | 0.69 [0.58, 0.82] | 9.38 [5.47, 13.12] | 5.93 [2.64, 9.21] | 0.45 [0.32, 0.71] | 0.46 [0.35, 0.67] | 0.00 [0.00, 0.00] | 0.06 [0.00, 0.12] | 0.00 [0.00, 0.00] |
| v2_kalman | 0.1 | 0.40 [0.27, 0.73] | 4.55 [2.53, 6.83] | 0.63 [0.53, 0.73] | 0.57 [0.46, 0.68] | 12.41 [7.17, 17.94] | 8.28 [3.63, 13.15] | 0.38 [0.25, 0.76] | 0.38 [0.27, 0.71] | 0.00 [0.00, 0.00] | 0.03 [0.00, 0.05] | 0.00 [0.00, 0.00] |
| v2_kalman_nopose | 0.15 | 0.38 [0.27, 0.68] | 4.97 [2.75, 7.54] | 0.59 [0.46, 0.71] | 0.51 [0.40, 0.64] | 12.00 [7.46, 16.68] | 7.72 [3.66, 12.13] | 0.36 [0.24, 0.68] | 0.36 [0.24, 0.63] | 0.00 [0.00, 0.00] | 0.05 [0.00, 0.09] | 0.00 [0.00, 0.00] |
| v2_gru | 0.3 | 0.41 [0.31, 0.65] | 3.45 [1.71, 5.22] | 0.71 [0.57, 0.82] | 0.65 [0.54, 0.75] | 11.86 [6.66, 17.18] | 8.41 [3.65, 13.38] | 0.45 [0.33, 0.69] | 0.49 [0.38, 0.71] | 0.00 [0.00, 0.00] | 0.05 [0.00, 0.09] | 0.00 [0.00, 0.00] |
| v2_gru_nopose | 0.3 | 0.41 [0.31, 0.65] | 3.45 [1.71, 5.22] | 0.70 [0.56, 0.81] | 0.64 [0.53, 0.74] | 11.59 [6.57, 16.61] | 8.14 [3.63, 12.82] | 0.45 [0.33, 0.69] | 0.49 [0.38, 0.71] | 0.00 [0.00, 0.00] | 0.05 [0.00, 0.09] | 0.00 [0.00, 0.00] |

## Operating point `recall_50` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||||
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
| v1_k0 | 0.98 | 0.97 | 0.92 | 0.92 | 0.92 |
| v1_k0_announcer | 0.98 | 0.97 | 0.94 | 0.94 | 0.94 |
| nearest_in_corridor | 0.92 | 0.81 | 0.77 | 0.68 | 0.53 |
| v2_physics | 0.92 | 0.81 | 0.77 | 0.70 | 0.55 |
| v2_physics_nopose | 0.93 | 0.84 | 0.78 | 0.71 | 0.51 |
| v2_kalman | 0.93 | 0.84 | 0.80 | 0.70 | 0.55 |
| v2_kalman_nopose | 0.95 | 0.86 | 0.79 | 0.72 | 0.59 |
| v2_gru | 0.97 | 0.89 | 0.82 | 0.72 | 0.54 |
| v2_gru_nopose | 0.97 | 0.89 | 0.82 | 0.72 | 0.54 |

## Operating curves (test; every knob value)

Recall (warned) against false alerts per minute; the CSV next to this report has every column.

| Arm | knob | warned | false/min | precision | alerts/min |
|---|---:|---:|---:|---:|---:|
| v1_k0 | 0.05 | 0.08 | 2.8 | 0.76 | 11.4 |
| v1_k0 | 0.1 | 0.08 | 2.2 | 0.78 | 10.1 |
| v1_k0 | 0.2 | 0.08 | 1.7 | 0.81 | 8.7 |
| v1_k0 | 0.35 | 0.08 | 1.1 | 0.84 | 6.8 |
| v1_k0 | 0.5 | 0.08 | 1.0 | 0.81 | 5.1 |
| v1_k0 | 0.75 | 0.06 | 0.7 | 0.85 | 4.6 |
| v1_k0 | 1 | 0.06 | 0.6 | 0.87 | 4.1 |
| v1_k0 | 1.5 | 0.06 | 0.4 | 0.88 | 3.6 |
| v1_k0 | 2 | 0.06 | 0.3 | 0.92 | 3.3 |
| v1_k0 | 3 | 0.06 | 0.3 | 0.90 | 2.9 |
| v1_k0 | 5 | 0.06 | 0.3 | 0.90 | 2.9 |
| v1_k0 | 8 | 0.06 | 0.3 | 0.90 | 2.9 |
| v1_k0 | 12 | 0.06 | 0.3 | 0.90 | 2.8 |
| v1_k0_announcer | 0.05 | 0.06 | 1.0 | 0.81 | 5.1 |
| v1_k0_announcer | 0.1 | 0.06 | 0.8 | 0.82 | 4.7 |
| v1_k0_announcer | 0.2 | 0.06 | 0.6 | 0.86 | 4.0 |
| v1_k0_announcer | 0.35 | 0.06 | 0.4 | 0.89 | 3.9 |
| v1_k0_announcer | 0.5 | 0.06 | 0.4 | 0.88 | 3.3 |
| v1_k0_announcer | 0.75 | 0.06 | 0.4 | 0.87 | 3.2 |
| v1_k0_announcer | 1 | 0.06 | 0.4 | 0.86 | 3.0 |
| v1_k0_announcer | 1.5 | 0.06 | 0.4 | 0.86 | 2.9 |
| v1_k0_announcer | 2 | 0.06 | 0.3 | 0.90 | 2.8 |
| v1_k0_announcer | 3 | 0.06 | 0.3 | 0.89 | 2.5 |
| v1_k0_announcer | 5 | 0.06 | 0.3 | 0.89 | 2.5 |
| v1_k0_announcer | 8 | 0.06 | 0.3 | 0.89 | 2.5 |
| v1_k0_announcer | 12 | 0.06 | 0.3 | 0.88 | 2.3 |
| nearest_in_corridor | 0.01 | 0.41 | 2.3 | 0.77 | 10.2 |
| nearest_in_corridor | 0.03 | 0.41 | 2.3 | 0.77 | 10.2 |
| nearest_in_corridor | 0.05 | 0.41 | 2.3 | 0.77 | 10.2 |
| nearest_in_corridor | 0.1 | 0.41 | 2.3 | 0.77 | 10.2 |
| nearest_in_corridor | 0.15 | 0.41 | 2.3 | 0.77 | 10.2 |
| nearest_in_corridor | 0.2 | 0.41 | 2.3 | 0.77 | 10.2 |
| nearest_in_corridor | 0.3 | 0.41 | 2.3 | 0.77 | 10.1 |
| nearest_in_corridor | 0.4 | 0.33 | 1.5 | 0.82 | 8.4 |
| nearest_in_corridor | 0.5 | 0.25 | 1.4 | 0.78 | 6.3 |
| nearest_in_corridor | 0.6 | 0.16 | 1.1 | 0.77 | 4.8 |
| nearest_in_corridor | 0.7 | 0.10 | 0.8 | 0.76 | 3.4 |
| nearest_in_corridor | 0.8 | 0.08 | 0.3 | 0.88 | 2.2 |
| nearest_in_corridor | 0.9 | 0.03 | 0.1 | 0.90 | 1.4 |
| nearest_in_corridor | 0.95 | 0.00 | 0.3 | 0.78 | 1.2 |
| v2_physics | 0.01 | 0.38 | 2.1 | 0.79 | 9.9 |
| v2_physics | 0.03 | 0.38 | 2.1 | 0.79 | 9.9 |
| v2_physics | 0.05 | 0.38 | 2.1 | 0.79 | 9.9 |
| v2_physics | 0.1 | 0.38 | 2.1 | 0.79 | 9.9 |
| v2_physics | 0.15 | 0.38 | 2.1 | 0.79 | 9.9 |
| v2_physics | 0.2 | 0.38 | 2.1 | 0.79 | 9.9 |
| v2_physics | 0.3 | 0.38 | 2.1 | 0.79 | 9.9 |
| v2_physics | 0.4 | 0.38 | 2.1 | 0.79 | 9.9 |
| v2_physics | 0.5 | 0.38 | 2.1 | 0.79 | 9.9 |
| v2_physics | 0.6 | 0.38 | 1.9 | 0.80 | 9.8 |
| v2_physics | 0.7 | 0.38 | 1.9 | 0.80 | 9.8 |
| v2_physics | 0.8 | 0.35 | 1.9 | 0.79 | 9.1 |
| v2_physics | 0.9 | 0.35 | 1.9 | 0.79 | 9.1 |
| v2_physics | 0.95 | 0.35 | 1.9 | 0.79 | 9.1 |
| v2_physics_nopose | 0.01 | 0.44 | 2.2 | 0.76 | 9.4 |
| v2_physics_nopose | 0.03 | 0.44 | 2.2 | 0.76 | 9.4 |
| v2_physics_nopose | 0.05 | 0.44 | 2.2 | 0.76 | 9.4 |
| v2_physics_nopose | 0.1 | 0.44 | 2.2 | 0.76 | 9.4 |
| v2_physics_nopose | 0.15 | 0.44 | 2.2 | 0.76 | 9.4 |
| v2_physics_nopose | 0.2 | 0.44 | 2.2 | 0.76 | 9.4 |
| v2_physics_nopose | 0.3 | 0.44 | 2.2 | 0.76 | 9.4 |
| v2_physics_nopose | 0.4 | 0.44 | 2.2 | 0.76 | 9.4 |
| v2_physics_nopose | 0.5 | 0.44 | 2.2 | 0.76 | 9.4 |
| v2_physics_nopose | 0.6 | 0.44 | 2.2 | 0.76 | 9.4 |
| v2_physics_nopose | 0.7 | 0.44 | 2.1 | 0.78 | 9.2 |
| v2_physics_nopose | 0.8 | 0.41 | 2.1 | 0.76 | 8.6 |
| v2_physics_nopose | 0.9 | 0.41 | 2.1 | 0.76 | 8.6 |
| v2_physics_nopose | 0.95 | 0.41 | 1.9 | 0.77 | 8.4 |
| v2_kalman | 0.01 | 0.40 | 4.8 | 0.62 | 12.7 |
| v2_kalman | 0.03 | 0.40 | 4.7 | 0.63 | 12.6 |
| v2_kalman | 0.05 | 0.40 | 4.7 | 0.63 | 12.6 |
| v2_kalman | 0.1 | 0.40 | 4.6 | 0.63 | 12.4 |
| v2_kalman | 0.15 | 0.40 | 4.4 | 0.64 | 12.3 |
| v2_kalman | 0.2 | 0.40 | 4.1 | 0.66 | 12.1 |
| v2_kalman | 0.3 | 0.38 | 3.7 | 0.69 | 11.9 |
| v2_kalman | 0.4 | 0.37 | 3.3 | 0.71 | 11.3 |
| v2_kalman | 0.5 | 0.37 | 2.5 | 0.76 | 10.3 |
| v2_kalman | 0.6 | 0.35 | 2.1 | 0.78 | 9.5 |
| v2_kalman | 0.7 | 0.30 | 1.4 | 0.83 | 8.3 |
| v2_kalman | 0.8 | 0.29 | 0.8 | 0.88 | 7.0 |
| v2_kalman | 0.9 | 0.24 | 1.0 | 0.86 | 6.8 |
| v2_kalman | 0.95 | 0.21 | 0.8 | 0.87 | 6.2 |
| v2_kalman_nopose | 0.01 | 0.38 | 5.2 | 0.57 | 12.3 |
| v2_kalman_nopose | 0.03 | 0.38 | 5.2 | 0.57 | 12.3 |
| v2_kalman_nopose | 0.05 | 0.38 | 5.1 | 0.58 | 12.1 |
| v2_kalman_nopose | 0.1 | 0.38 | 5.1 | 0.58 | 12.1 |
| v2_kalman_nopose | 0.15 | 0.38 | 5.0 | 0.59 | 12.0 |
| v2_kalman_nopose | 0.2 | 0.38 | 4.7 | 0.60 | 11.7 |
| v2_kalman_nopose | 0.3 | 0.38 | 4.1 | 0.64 | 11.4 |
| v2_kalman_nopose | 0.4 | 0.37 | 3.0 | 0.71 | 10.5 |
| v2_kalman_nopose | 0.5 | 0.37 | 2.3 | 0.76 | 9.8 |
| v2_kalman_nopose | 0.6 | 0.35 | 1.9 | 0.79 | 9.1 |
| v2_kalman_nopose | 0.7 | 0.33 | 1.7 | 0.80 | 8.1 |
| v2_kalman_nopose | 0.8 | 0.32 | 1.2 | 0.82 | 6.9 |
| v2_kalman_nopose | 0.9 | 0.24 | 1.1 | 0.81 | 5.8 |
| v2_kalman_nopose | 0.95 | 0.22 | 1.0 | 0.83 | 5.7 |
| v2_gru | 0.01 | 0.43 | 4.7 | 0.65 | 13.2 |
| v2_gru | 0.03 | 0.43 | 4.6 | 0.65 | 13.1 |
| v2_gru | 0.05 | 0.43 | 4.4 | 0.66 | 13.0 |
| v2_gru | 0.1 | 0.43 | 4.3 | 0.67 | 12.8 |
| v2_gru | 0.15 | 0.44 | 3.4 | 0.72 | 12.3 |
| v2_gru | 0.2 | 0.44 | 3.4 | 0.72 | 12.3 |
| v2_gru | 0.3 | 0.41 | 3.4 | 0.71 | 11.9 |
| v2_gru | 0.4 | 0.40 | 3.2 | 0.71 | 10.9 |
| v2_gru | 0.5 | 0.37 | 2.9 | 0.72 | 10.2 |
| v2_gru | 0.6 | 0.30 | 1.9 | 0.78 | 8.8 |
| v2_gru | 0.7 | 0.24 | 1.1 | 0.85 | 7.2 |
| v2_gru | 0.8 | 0.17 | 0.4 | 0.92 | 5.2 |
| v2_gru | 0.9 | 0.13 | 0.6 | 0.85 | 3.7 |
| v2_gru | 0.95 | 0.11 | 0.0 | 1.00 | 2.5 |
| v2_gru_nopose | 0.01 | 0.43 | 4.7 | 0.64 | 13.0 |
| v2_gru_nopose | 0.03 | 0.43 | 4.6 | 0.65 | 12.8 |
| v2_gru_nopose | 0.05 | 0.43 | 4.4 | 0.65 | 12.7 |
| v2_gru_nopose | 0.1 | 0.43 | 4.3 | 0.66 | 12.6 |
| v2_gru_nopose | 0.15 | 0.44 | 3.4 | 0.71 | 12.0 |
| v2_gru_nopose | 0.2 | 0.44 | 3.4 | 0.71 | 12.0 |
| v2_gru_nopose | 0.3 | 0.41 | 3.4 | 0.70 | 11.6 |
| v2_gru_nopose | 0.4 | 0.40 | 3.2 | 0.70 | 10.6 |
| v2_gru_nopose | 0.5 | 0.37 | 2.9 | 0.71 | 9.9 |
| v2_gru_nopose | 0.6 | 0.30 | 1.9 | 0.77 | 8.6 |
| v2_gru_nopose | 0.7 | 0.24 | 1.1 | 0.84 | 6.9 |
| v2_gru_nopose | 0.8 | 0.17 | 0.6 | 0.89 | 5.1 |
| v2_gru_nopose | 0.9 | 0.13 | 0.6 | 0.83 | 3.3 |
| v2_gru_nopose | 0.95 | 0.11 | 0.0 | 1.00 | 2.1 |

Classes: true = the object enters the zone later within the horizon; late = it is already in the zone; false = it doesn't enter within the horizon. precision = (true + late) / all. warned = an announcement about the arriving track in the 4 s up to and including arrival.
Open-loop: the recorded walker never heard these alerts. Labels and inputs share SANPO's depth.
