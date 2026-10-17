# Threat-engine replay - corridor_kalman_2026_10_17_lidar5m

Label: in the walking corridor (width 0.75 m + class half-width, 0 < forward <= 3 m along the direction of travel from fixed_camera_poses.csv) within 20 steps (4 s); tracked objects only.

SANPO official test split: 71 sessions, 63 arrivals (a track first seen before it entered the zone). Operating points chosen on 41 validation sessions, then scored once on test. GRU: `models/GRU/GRU/runs/corridor_h8_seed0`. Generated 2026-10-17 by `python tools/replay_threat_engine.py` (361 s).

Depth: LiDAR-degraded inputs (`--lidar-range 5`, seed 0): rows beyond 5 m and 5% at random dropped, noise N(0, (0.01 + 0.01·d)·(1 + 3·min(1, |bearing|/30°)²)) m, closing speed re-fitted over 6 steps. Labels, arrivals and windows use SANPO depth.

## Opportunity window (first sighting -> arrival), test

Median 2.6 s; 25% of arrivals were first seen less than 1 s before arriving, 38% less than 2 s, 27% at least 4 s before. Lead time can't exceed the window, so lead is reported conditionally below.

## Operating point `budget_2` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||||
| nearest_in_corridor | 0.8 | 0.02 [0.00, 0.04] | 1.10 [0.27, 2.08] | 0.62 [0.50, 0.73] | 0.52 [0.35, 0.71] | 2.90 [0.75, 5.50] | 1.79 [0.17, 3.98] | 0.02 [0.00, 0.06] | 0.03 [0.00, 0.08] | 1.40 [1.40, 1.40] | 0.02 [0.00, 0.04] | 0.02 [0.00, 0.04] |
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | — | no setting meets the rule on validation |||||||||||
| v2_kalman_nopose | — | no setting meets the rule on validation |||||||||||
| v2_gru | 0.9 | 0.16 [0.03, 0.36] | 1.38 [0.34, 2.57] | 0.72 [0.56, 0.88] | 0.56 [0.35, 0.77] | 4.97 [2.16, 8.08] | 3.31 [0.99, 5.97] | 0.17 [0.04, 0.36] | 0.18 [0.00, 0.41] | 0.30 [0.00, 1.40] | 0.08 [0.00, 0.15] | 0.03 [0.00, 0.09] |
| v2_gru_nopose | 0.9 | 0.14 [0.03, 0.35] | 1.10 [0.26, 2.11] | 0.73 [0.50, 0.92] | 0.57 [0.35, 0.79] | 4.14 [1.79, 6.39] | 2.62 [0.94, 4.38] | 0.15 [0.04, 0.33] | 0.15 [0.00, 0.38] | 0.20 [0.00, 1.00] | 0.06 [0.00, 0.11] | 0.02 [0.00, 0.04] |

## Operating point `budget_5` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | 12 | 0.29 [0.18, 0.41] | 2.48 [0.50, 4.59] | 0.76 [0.57, 0.90] | 0.61 [0.46, 0.80] | 10.21 [3.12, 18.21] | 5.66 [0.93, 11.60] | 0.21 [0.09, 0.35] | 0.18 [0.07, 0.32] | 0.20 [0.00, 1.20] | 0.10 [0.00, 0.22] | 0.06 [0.00, 0.15] |
| v1_k0_announcer | 2 | 0.30 [0.19, 0.45] | 3.17 [1.21, 5.17] | 0.64 [0.41, 0.76] | 0.47 [0.31, 0.57] | 8.83 [3.15, 15.03] | 6.62 [1.50, 12.57] | 0.23 [0.09, 0.40] | 0.21 [0.07, 0.41] | 0.60 [0.00, 1.20] | 0.13 [0.00, 0.27] | 0.06 [0.00, 0.15] |
| nearest_in_corridor | 0.5 | 0.22 [0.14, 0.41] | 2.62 [0.96, 4.51] | 0.67 [0.57, 0.77] | 0.53 [0.43, 0.68] | 7.86 [3.48, 12.51] | 5.52 [2.11, 9.27] | 0.23 [0.15, 0.42] | 0.23 [0.13, 0.41] | 0.00 [0.00, 0.80] | 0.08 [0.03, 0.14] | 0.02 [0.00, 0.04] |
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | 0.95 | 0.38 [0.26, 0.62] | 2.76 [1.21, 4.31] | 0.70 [0.60, 0.79] | 0.61 [0.47, 0.76] | 9.10 [5.01, 13.19] | 6.07 [2.55, 9.57] | 0.40 [0.25, 0.67] | 0.44 [0.29, 0.67] | 0.00 [0.00, 0.00] | 0.06 [0.00, 0.09] | 0.02 [0.00, 0.03] |
| v2_kalman | 0.8 | 0.30 [0.19, 0.58] | 3.31 [1.21, 5.62] | 0.65 [0.54, 0.78] | 0.51 [0.39, 0.72] | 9.38 [4.65, 14.14] | 6.48 [2.42, 10.99] | 0.32 [0.17, 0.68] | 0.31 [0.12, 0.70] | 0.00 [0.00, 0.20] | 0.10 [0.03, 0.23] | 0.03 [0.00, 0.08] |
| v2_kalman_nopose | 0.6 | 0.38 [0.25, 0.63] | 3.31 [1.42, 5.38] | 0.68 [0.59, 0.81] | 0.55 [0.41, 0.73] | 10.34 [6.00, 14.89] | 6.21 [2.58, 9.97] | 0.40 [0.24, 0.70] | 0.41 [0.26, 0.65] | 0.00 [0.00, 0.20] | 0.14 [0.03, 0.33] | 0.02 [0.00, 0.07] |
| v2_gru | 0.5 | 0.37 [0.27, 0.63] | 4.00 [1.97, 6.04] | 0.65 [0.53, 0.76] | 0.52 [0.43, 0.65] | 11.31 [5.81, 16.89] | 8.14 [3.18, 13.40] | 0.38 [0.27, 0.70] | 0.41 [0.28, 0.70] | 0.00 [0.00, 0.40] | 0.11 [0.00, 0.23] | 0.03 [0.00, 0.08] |
| v2_gru_nopose | 0.5 | 0.37 [0.27, 0.63] | 3.45 [1.52, 5.51] | 0.68 [0.54, 0.81] | 0.56 [0.43, 0.69] | 10.90 [5.59, 16.15] | 8.00 [3.18, 12.97] | 0.38 [0.27, 0.70] | 0.41 [0.28, 0.70] | 0.00 [0.00, 0.40] | 0.11 [0.00, 0.23] | 0.02 [0.00, 0.07] |

## Operating point `budget_10` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | 1 | 0.35 [0.24, 0.47] | 5.52 [2.02, 9.08] | 0.68 [0.51, 0.77] | 0.52 [0.41, 0.62] | 17.38 [5.40, 30.44] | 10.07 [2.42, 19.73] | 0.30 [0.15, 0.44] | 0.23 [0.10, 0.41] | 0.30 [0.00, 1.00] | 0.14 [0.03, 0.27] | 0.06 [0.00, 0.15] |
| v1_k0_announcer | 0.5 | 0.30 [0.19, 0.45] | 4.69 [2.01, 7.34] | 0.56 [0.35, 0.67] | 0.42 [0.26, 0.50] | 10.62 [3.96, 17.87] | 7.86 [2.04, 14.85] | 0.23 [0.09, 0.40] | 0.21 [0.07, 0.41] | 0.60 [0.00, 1.20] | 0.13 [0.00, 0.27] | 0.06 [0.00, 0.15] |
| nearest_in_corridor | 0.3 | 0.43 [0.32, 0.72] | 6.34 [3.49, 9.05] | 0.55 [0.45, 0.65] | 0.44 [0.33, 0.58] | 14.21 [8.48, 19.67] | 9.52 [4.65, 14.18] | 0.40 [0.26, 0.76] | 0.38 [0.23, 0.73] | 0.00 [0.00, 0.00] | 0.11 [0.03, 0.24] | 0.03 [0.00, 0.08] |
| v2_physics | 0.95 | 0.33 [0.25, 0.55] | 5.38 [3.02, 7.59] | 0.57 [0.45, 0.67] | 0.48 [0.39, 0.57] | 12.55 [7.05, 17.75] | 8.97 [4.08, 13.80] | 0.34 [0.24, 0.61] | 0.36 [0.23, 0.67] | 0.00 [0.00, 0.00] | 0.08 [0.03, 0.12] | 0.03 [0.00, 0.08] |
| v2_physics_nopose | 0.95 | 0.38 [0.26, 0.62] | 2.76 [1.21, 4.31] | 0.70 [0.60, 0.79] | 0.61 [0.47, 0.76] | 9.10 [5.01, 13.19] | 6.07 [2.55, 9.57] | 0.40 [0.25, 0.67] | 0.44 [0.29, 0.67] | 0.00 [0.00, 0.00] | 0.06 [0.00, 0.09] | 0.02 [0.00, 0.03] |
| v2_kalman | 0.5 | 0.41 [0.29, 0.72] | 7.17 [4.13, 10.04] | 0.52 [0.42, 0.61] | 0.43 [0.34, 0.53] | 15.03 [8.58, 21.00] | 10.34 [4.86, 15.67] | 0.43 [0.29, 0.75] | 0.38 [0.23, 0.72] | 0.00 [0.00, 0.00] | 0.13 [0.06, 0.25] | 0.05 [0.00, 0.11] |
| v2_kalman_nopose | 0.5 | 0.37 [0.24, 0.68] | 5.10 [2.89, 7.09] | 0.59 [0.51, 0.67] | 0.44 [0.33, 0.57] | 12.41 [7.54, 17.08] | 7.59 [3.95, 11.13] | 0.34 [0.21, 0.65] | 0.36 [0.23, 0.65] | 0.00 [0.00, 0.40] | 0.14 [0.03, 0.35] | 0.02 [0.00, 0.07] |
| v2_gru | 0.4 | 0.40 [0.29, 0.66] | 4.55 [2.24, 6.89] | 0.63 [0.52, 0.74] | 0.50 [0.40, 0.62] | 12.41 [6.68, 18.42] | 8.97 [3.74, 14.36] | 0.43 [0.31, 0.72] | 0.46 [0.34, 0.72] | 0.00 [0.00, 0.40] | 0.14 [0.05, 0.29] | 0.03 [0.00, 0.08] |
| v2_gru_nopose | 0.4 | 0.41 [0.31, 0.66] | 4.00 [1.74, 6.42] | 0.66 [0.52, 0.79] | 0.53 [0.42, 0.66] | 11.86 [6.36, 17.59] | 8.69 [3.63, 14.00] | 0.45 [0.33, 0.72] | 0.49 [0.38, 0.72] | 0.00 [0.00, 0.40] | 0.14 [0.05, 0.29] | 0.02 [0.00, 0.07] |

## Operating point `recall_50` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | 0.5 | 0.37 [0.26, 0.47] | 8.14 [3.36, 13.10] | 0.62 [0.45, 0.68] | 0.45 [0.35, 0.52] | 21.24 [6.88, 37.14] | 12.41 [3.01, 23.62] | 0.32 [0.18, 0.44] | 0.26 [0.12, 0.42] | 0.80 [0.00, 2.00] | 0.17 [0.07, 0.27] | 0.11 [0.00, 0.16] |
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
| v1_k0 | 0.90 | 0.79 | 0.77 | 0.72 | 0.69 |
| v1_k0_announcer | 0.90 | 0.79 | 0.77 | 0.70 | 0.67 |
| nearest_in_corridor | 0.98 | 0.91 | 0.87 | 0.81 | 0.77 |
| v2_physics | n/a |
| v2_physics_nopose | 0.97 | 0.89 | 0.85 | 0.75 | 0.56 |
| v2_kalman | 0.95 | 0.88 | 0.84 | 0.71 | 0.61 |
| v2_kalman_nopose | 0.94 | 0.86 | 0.82 | 0.72 | 0.59 |
| v2_gru | 0.95 | 0.88 | 0.81 | 0.66 | 0.59 |
| v2_gru_nopose | 0.95 | 0.88 | 0.81 | 0.66 | 0.59 |

## Operating curves (test; every knob value)

Recall (warned) against false alerts per minute; the CSV next to this report has every column.

| Arm | knob | warned | false/min | precision | alerts/min |
|---|---:|---:|---:|---:|---:|
| v1_k0 | 0.05 | 0.44 | 14.1 | 0.55 | 31.3 |
| v1_k0 | 0.1 | 0.43 | 12.6 | 0.57 | 29.0 |
| v1_k0 | 0.2 | 0.41 | 10.3 | 0.60 | 25.8 |
| v1_k0 | 0.35 | 0.40 | 9.4 | 0.60 | 23.3 |
| v1_k0 | 0.5 | 0.37 | 8.1 | 0.62 | 21.2 |
| v1_k0 | 0.75 | 0.35 | 6.5 | 0.66 | 19.2 |
| v1_k0 | 1 | 0.35 | 5.5 | 0.68 | 17.4 |
| v1_k0 | 1.5 | 0.35 | 4.7 | 0.70 | 15.6 |
| v1_k0 | 2 | 0.32 | 3.9 | 0.73 | 14.1 |
| v1_k0 | 3 | 0.32 | 3.0 | 0.74 | 11.6 |
| v1_k0 | 5 | 0.30 | 2.6 | 0.76 | 10.9 |
| v1_k0 | 8 | 0.30 | 2.6 | 0.75 | 10.6 |
| v1_k0 | 12 | 0.29 | 2.5 | 0.76 | 10.2 |
| v1_k0_announcer | 0.05 | 0.37 | 6.5 | 0.52 | 13.4 |
| v1_k0_announcer | 0.1 | 0.35 | 6.2 | 0.52 | 13.0 |
| v1_k0_announcer | 0.2 | 0.33 | 5.2 | 0.55 | 11.6 |
| v1_k0_announcer | 0.35 | 0.32 | 5.0 | 0.54 | 10.9 |
| v1_k0_announcer | 0.5 | 0.30 | 4.7 | 0.56 | 10.6 |
| v1_k0_announcer | 0.75 | 0.30 | 4.0 | 0.60 | 9.9 |
| v1_k0_announcer | 1 | 0.30 | 3.9 | 0.61 | 9.8 |
| v1_k0_announcer | 1.5 | 0.30 | 3.6 | 0.61 | 9.2 |
| v1_k0_announcer | 2 | 0.30 | 3.2 | 0.64 | 8.8 |
| v1_k0_announcer | 3 | 0.30 | 2.5 | 0.68 | 7.9 |
| v1_k0_announcer | 5 | 0.30 | 2.1 | 0.72 | 7.4 |
| v1_k0_announcer | 8 | 0.30 | 2.1 | 0.72 | 7.4 |
| v1_k0_announcer | 12 | 0.29 | 1.9 | 0.73 | 7.2 |
| nearest_in_corridor | 0.01 | 0.41 | 9.4 | 0.46 | 17.4 |
| nearest_in_corridor | 0.03 | 0.41 | 9.4 | 0.46 | 17.4 |
| nearest_in_corridor | 0.05 | 0.41 | 9.4 | 0.46 | 17.4 |
| nearest_in_corridor | 0.1 | 0.41 | 9.4 | 0.46 | 17.4 |
| nearest_in_corridor | 0.15 | 0.41 | 9.4 | 0.46 | 17.4 |
| nearest_in_corridor | 0.2 | 0.41 | 9.1 | 0.47 | 17.1 |
| nearest_in_corridor | 0.3 | 0.43 | 6.3 | 0.55 | 14.2 |
| nearest_in_corridor | 0.4 | 0.30 | 3.9 | 0.64 | 10.8 |
| nearest_in_corridor | 0.5 | 0.22 | 2.6 | 0.67 | 7.9 |
| nearest_in_corridor | 0.6 | 0.16 | 2.3 | 0.64 | 6.5 |
| nearest_in_corridor | 0.7 | 0.11 | 1.2 | 0.72 | 4.4 |
| nearest_in_corridor | 0.8 | 0.02 | 1.1 | 0.62 | 2.9 |
| nearest_in_corridor | 0.9 | 0.02 | 1.1 | 0.60 | 2.8 |
| nearest_in_corridor | 0.95 | 0.02 | 0.7 | 0.67 | 2.1 |
| v2_physics | 0.01 | 0.37 | 7.9 | 0.50 | 15.6 |
| v2_physics | 0.03 | 0.37 | 7.9 | 0.50 | 15.6 |
| v2_physics | 0.05 | 0.37 | 7.9 | 0.50 | 15.6 |
| v2_physics | 0.1 | 0.37 | 7.9 | 0.50 | 15.6 |
| v2_physics | 0.15 | 0.37 | 7.9 | 0.50 | 15.6 |
| v2_physics | 0.2 | 0.37 | 7.9 | 0.50 | 15.6 |
| v2_physics | 0.3 | 0.37 | 7.7 | 0.50 | 15.4 |
| v2_physics | 0.4 | 0.35 | 7.4 | 0.50 | 15.0 |
| v2_physics | 0.5 | 0.35 | 7.4 | 0.50 | 15.0 |
| v2_physics | 0.6 | 0.35 | 7.3 | 0.51 | 14.9 |
| v2_physics | 0.7 | 0.35 | 7.2 | 0.51 | 14.8 |
| v2_physics | 0.8 | 0.33 | 6.3 | 0.53 | 13.5 |
| v2_physics | 0.9 | 0.33 | 6.1 | 0.54 | 13.2 |
| v2_physics | 0.95 | 0.33 | 5.4 | 0.57 | 12.6 |
| v2_physics_nopose | 0.01 | 0.38 | 4.0 | 0.63 | 10.9 |
| v2_physics_nopose | 0.03 | 0.38 | 4.0 | 0.63 | 10.9 |
| v2_physics_nopose | 0.05 | 0.38 | 4.0 | 0.63 | 10.9 |
| v2_physics_nopose | 0.1 | 0.38 | 4.0 | 0.63 | 10.9 |
| v2_physics_nopose | 0.15 | 0.38 | 4.0 | 0.63 | 10.9 |
| v2_physics_nopose | 0.2 | 0.38 | 4.0 | 0.63 | 10.9 |
| v2_physics_nopose | 0.3 | 0.40 | 4.0 | 0.63 | 10.9 |
| v2_physics_nopose | 0.4 | 0.40 | 4.0 | 0.63 | 10.9 |
| v2_physics_nopose | 0.5 | 0.40 | 4.0 | 0.63 | 10.9 |
| v2_physics_nopose | 0.6 | 0.41 | 4.0 | 0.63 | 10.8 |
| v2_physics_nopose | 0.7 | 0.41 | 3.7 | 0.64 | 10.5 |
| v2_physics_nopose | 0.8 | 0.38 | 3.3 | 0.66 | 9.7 |
| v2_physics_nopose | 0.9 | 0.38 | 3.0 | 0.68 | 9.4 |
| v2_physics_nopose | 0.95 | 0.38 | 2.8 | 0.70 | 9.1 |
| v2_kalman | 0.01 | 0.37 | 18.9 | 0.27 | 25.9 |
| v2_kalman | 0.03 | 0.37 | 18.6 | 0.27 | 25.7 |
| v2_kalman | 0.05 | 0.37 | 18.5 | 0.28 | 25.5 |
| v2_kalman | 0.1 | 0.37 | 17.4 | 0.29 | 24.6 |
| v2_kalman | 0.15 | 0.35 | 15.7 | 0.32 | 23.2 |
| v2_kalman | 0.2 | 0.40 | 13.7 | 0.37 | 21.7 |
| v2_kalman | 0.3 | 0.43 | 10.5 | 0.44 | 18.6 |
| v2_kalman | 0.4 | 0.40 | 8.8 | 0.47 | 16.6 |
| v2_kalman | 0.5 | 0.41 | 7.2 | 0.52 | 15.0 |
| v2_kalman | 0.6 | 0.40 | 4.1 | 0.65 | 11.9 |
| v2_kalman | 0.7 | 0.35 | 3.0 | 0.70 | 10.2 |
| v2_kalman | 0.8 | 0.30 | 3.3 | 0.65 | 9.4 |
| v2_kalman | 0.9 | 0.25 | 2.3 | 0.71 | 8.1 |
| v2_kalman | 0.95 | 0.22 | 2.2 | 0.71 | 7.6 |
| v2_kalman_nopose | 0.01 | 0.33 | 19.2 | 0.25 | 25.7 |
| v2_kalman_nopose | 0.03 | 0.33 | 19.2 | 0.25 | 25.7 |
| v2_kalman_nopose | 0.05 | 0.33 | 19.0 | 0.25 | 25.5 |
| v2_kalman_nopose | 0.1 | 0.33 | 17.7 | 0.27 | 24.1 |
| v2_kalman_nopose | 0.15 | 0.33 | 15.6 | 0.30 | 22.2 |
| v2_kalman_nopose | 0.2 | 0.38 | 12.8 | 0.36 | 20.1 |
| v2_kalman_nopose | 0.3 | 0.37 | 9.7 | 0.44 | 17.1 |
| v2_kalman_nopose | 0.4 | 0.37 | 7.4 | 0.50 | 14.9 |
| v2_kalman_nopose | 0.5 | 0.37 | 5.1 | 0.59 | 12.4 |
| v2_kalman_nopose | 0.6 | 0.38 | 3.3 | 0.68 | 10.3 |
| v2_kalman_nopose | 0.7 | 0.32 | 2.5 | 0.72 | 8.8 |
| v2_kalman_nopose | 0.8 | 0.29 | 1.9 | 0.75 | 7.6 |
| v2_kalman_nopose | 0.9 | 0.22 | 1.2 | 0.80 | 6.2 |
| v2_kalman_nopose | 0.95 | 0.19 | 1.1 | 0.81 | 5.9 |
| v2_gru | 0.01 | 0.38 | 18.9 | 0.30 | 26.9 |
| v2_gru | 0.03 | 0.38 | 18.6 | 0.30 | 26.6 |
| v2_gru | 0.05 | 0.38 | 17.5 | 0.31 | 25.5 |
| v2_gru | 0.1 | 0.40 | 14.9 | 0.35 | 23.0 |
| v2_gru | 0.15 | 0.38 | 11.2 | 0.43 | 19.6 |
| v2_gru | 0.2 | 0.38 | 9.0 | 0.49 | 17.5 |
| v2_gru | 0.3 | 0.41 | 6.6 | 0.55 | 14.8 |
| v2_gru | 0.4 | 0.40 | 4.6 | 0.63 | 12.4 |
| v2_gru | 0.5 | 0.37 | 4.0 | 0.65 | 11.3 |
| v2_gru | 0.6 | 0.30 | 2.8 | 0.72 | 9.9 |
| v2_gru | 0.7 | 0.27 | 2.1 | 0.75 | 8.3 |
| v2_gru | 0.8 | 0.22 | 1.4 | 0.79 | 6.5 |
| v2_gru | 0.9 | 0.16 | 1.4 | 0.72 | 5.0 |
| v2_gru | 0.95 | 0.14 | 0.8 | 0.77 | 3.6 |
| v2_gru_nopose | 0.01 | 0.37 | 18.8 | 0.30 | 26.9 |
| v2_gru_nopose | 0.03 | 0.37 | 18.5 | 0.31 | 26.6 |
| v2_gru_nopose | 0.05 | 0.37 | 17.4 | 0.32 | 25.5 |
| v2_gru_nopose | 0.1 | 0.38 | 14.6 | 0.36 | 22.9 |
| v2_gru_nopose | 0.15 | 0.38 | 10.8 | 0.44 | 19.2 |
| v2_gru_nopose | 0.2 | 0.38 | 8.6 | 0.50 | 17.1 |
| v2_gru_nopose | 0.3 | 0.41 | 6.2 | 0.57 | 14.3 |
| v2_gru_nopose | 0.4 | 0.41 | 4.0 | 0.66 | 11.9 |
| v2_gru_nopose | 0.5 | 0.37 | 3.4 | 0.68 | 10.9 |
| v2_gru_nopose | 0.6 | 0.30 | 2.5 | 0.74 | 9.4 |
| v2_gru_nopose | 0.7 | 0.27 | 1.7 | 0.79 | 7.7 |
| v2_gru_nopose | 0.8 | 0.22 | 0.8 | 0.86 | 5.9 |
| v2_gru_nopose | 0.9 | 0.14 | 1.1 | 0.73 | 4.1 |
| v2_gru_nopose | 0.95 | 0.14 | 0.6 | 0.80 | 2.8 |

Classes: true = the object enters the zone later within the horizon; late = it is already in the zone; false = it doesn't enter within the horizon. precision = (true + late) / all. warned = an announcement about the arriving track in the 4 s up to and including arrival.
Open-loop: the recorded walker never heard these alerts. Labels and inputs share SANPO's depth.
