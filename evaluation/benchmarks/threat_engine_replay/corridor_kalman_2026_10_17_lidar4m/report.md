# Threat-engine replay - corridor_kalman_2026_10_17_lidar4m

Label: in the walking corridor (width 0.75 m + class half-width, 0 < forward <= 3 m along the direction of travel from fixed_camera_poses.csv) within 20 steps (4 s); tracked objects only.

SANPO official test split: 71 sessions, 63 arrivals (a track first seen before it entered the zone). Operating points chosen on 41 validation sessions, then scored once on test. GRU: `models/GRU/GRU/runs/corridor_h8_seed0`. Generated 2026-10-17 by `python tools/replay_threat_engine.py` (296 s).

Depth: LiDAR-degraded inputs (`--lidar-range 4`, seed 0): rows beyond 4 m and 5% at random dropped, noise N(0, (0.01 + 0.01·d)·(1 + 3·min(1, |bearing|/30°)²)) m, closing speed re-fitted over 6 steps. Labels, arrivals and windows use SANPO depth.

## Opportunity window (first sighting -> arrival), test

Median 2.6 s; 25% of arrivals were first seen less than 1 s before arriving, 38% less than 2 s, 27% at least 4 s before. Lead time can't exceed the window, so lead is reported conditionally below.

## Operating point `budget_2` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | 5 | 0.24 [0.10, 0.33] | 1.10 [0.22, 2.13] | 0.84 [0.70, 0.92] | 0.68 [0.53, 0.89] | 6.90 [1.71, 12.71] | 4.28 [0.33, 9.01] | 0.19 [0.00, 0.34] | 0.15 [0.00, 0.33] | 0.20 [0.00, 1.10] | 0.08 [0.00, 0.19] | 0.05 [0.00, 0.13] |
| v1_k0_announcer | 2 | 0.24 [0.10, 0.33] | 1.66 [0.50, 2.86] | 0.73 [0.45, 0.83] | 0.57 [0.42, 0.71] | 6.07 [1.51, 11.29] | 4.69 [0.76, 9.44] | 0.19 [0.00, 0.34] | 0.15 [0.00, 0.33] | 0.20 [0.00, 1.10] | 0.08 [0.00, 0.19] | 0.05 [0.00, 0.13] |
| nearest_in_corridor | 0.95 | 0.02 [0.00, 0.04] | 0.55 [0.00, 1.27] | 0.71 [0.50, 1.00] | 0.50 [0.00, 1.00] | 1.93 [0.17, 4.14] | 1.10 [0.00, 2.59] | 0.02 [0.00, 0.06] | 0.03 [0.00, 0.08] | 1.40 [1.40, 1.40] | 0.02 [0.00, 0.04] | 0.02 [0.00, 0.04] |
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | — | no setting meets the rule on validation |||||||||||
| v2_kalman_nopose | — | no setting meets the rule on validation |||||||||||
| v2_gru | 0.8 | 0.21 [0.12, 0.37] | 1.10 [0.26, 2.19] | 0.83 [0.68, 0.94] | 0.70 [0.50, 0.90] | 6.48 [2.57, 10.90] | 4.41 [1.26, 8.14] | 0.19 [0.12, 0.31] | 0.21 [0.10, 0.36] | 0.00 [0.00, 0.80] | 0.06 [0.00, 0.11] | 0.02 [0.00, 0.04] |
| v2_gru_nopose | 0.8 | 0.19 [0.10, 0.36] | 0.83 [0.19, 1.55] | 0.85 [0.72, 0.95] | 0.78 [0.63, 0.92] | 5.66 [2.40, 9.19] | 4.00 [1.17, 7.34] | 0.17 [0.10, 0.31] | 0.18 [0.09, 0.33] | 0.00 [0.00, 0.60] | 0.05 [0.00, 0.08] | 0.00 [0.00, 0.00] |

## Operating point `budget_5` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | 1 | 0.29 [0.13, 0.36] | 2.90 [0.90, 5.03] | 0.74 [0.60, 0.80] | 0.57 [0.43, 0.74] | 11.17 [2.83, 20.55] | 7.45 [1.00, 15.29] | 0.26 [0.08, 0.40] | 0.21 [0.00, 0.37] | 0.20 [0.00, 1.00] | 0.11 [0.00, 0.23] | 0.05 [0.00, 0.13] |
| v1_k0_announcer | 0.2 | 0.29 [0.13, 0.37] | 2.62 [0.91, 4.47] | 0.67 [0.48, 0.76] | 0.52 [0.40, 0.67] | 8.00 [2.44, 14.47] | 5.93 [1.04, 12.06] | 0.26 [0.06, 0.41] | 0.23 [0.06, 0.40] | 0.10 [0.00, 1.00] | 0.10 [0.00, 0.23] | 0.05 [0.00, 0.13] |
| nearest_in_corridor | 0.5 | 0.21 [0.11, 0.41] | 3.03 [0.92, 5.52] | 0.61 [0.51, 0.76] | 0.50 [0.38, 0.71] | 7.72 [3.37, 12.38] | 5.24 [2.06, 8.60] | 0.21 [0.12, 0.42] | 0.21 [0.11, 0.40] | 0.00 [0.00, 0.20] | 0.06 [0.03, 0.12] | 0.02 [0.00, 0.04] |
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | 0.8 | 0.37 [0.27, 0.61] | 2.76 [1.36, 4.16] | 0.71 [0.62, 0.80] | 0.65 [0.58, 0.74] | 9.52 [5.33, 13.64] | 6.62 [3.00, 10.39] | 0.38 [0.28, 0.62] | 0.41 [0.30, 0.64] | 0.00 [0.00, 0.00] | 0.05 [0.00, 0.09] | 0.02 [0.00, 0.03] |
| v2_kalman | 0.7 | 0.33 [0.21, 0.67] | 2.90 [1.08, 4.87] | 0.70 [0.62, 0.81] | 0.60 [0.48, 0.76] | 9.66 [4.95, 14.44] | 6.34 [2.29, 10.81] | 0.34 [0.19, 0.72] | 0.33 [0.17, 0.68] | 0.00 [0.00, 0.40] | 0.08 [0.03, 0.18] | 0.02 [0.00, 0.04] |
| v2_kalman_nopose | 0.6 | 0.40 [0.30, 0.64] | 2.48 [0.91, 4.31] | 0.74 [0.65, 0.85] | 0.63 [0.52, 0.81] | 9.38 [5.15, 13.64] | 5.66 [2.48, 9.14] | 0.40 [0.28, 0.64] | 0.41 [0.31, 0.59] | 0.00 [0.00, 0.20] | 0.11 [0.05, 0.19] | 0.00 [0.00, 0.00] |
| v2_gru | 0.5 | 0.37 [0.27, 0.62] | 4.14 [1.97, 6.20] | 0.63 [0.52, 0.74] | 0.54 [0.41, 0.69] | 11.31 [5.79, 17.00] | 8.14 [3.26, 13.25] | 0.38 [0.28, 0.67] | 0.41 [0.29, 0.68] | 0.00 [0.00, 0.20] | 0.08 [0.00, 0.15] | 0.02 [0.00, 0.04] |
| v2_gru_nopose | 0.4 | 0.40 [0.30, 0.62] | 4.14 [2.00, 6.35] | 0.65 [0.51, 0.77] | 0.56 [0.45, 0.68] | 11.72 [6.36, 17.08] | 8.55 [3.60, 13.77] | 0.43 [0.32, 0.67] | 0.46 [0.37, 0.68] | 0.00 [0.00, 0.00] | 0.10 [0.03, 0.18] | 0.00 [0.00, 0.00] |

## Operating point `budget_10` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | 0.35 | 0.32 [0.13, 0.40] | 4.14 [1.27, 7.41] | 0.72 [0.58, 0.78] | 0.53 [0.45, 0.68] | 14.76 [3.30, 28.47] | 9.52 [1.51, 19.40] | 0.30 [0.08, 0.45] | 0.26 [0.00, 0.42] | 0.30 [0.00, 1.20] | 0.14 [0.00, 0.27] | 0.06 [0.00, 0.18] |
| v1_k0_announcer | 0.2 | 0.29 [0.13, 0.37] | 2.62 [0.91, 4.47] | 0.67 [0.48, 0.76] | 0.52 [0.40, 0.67] | 8.00 [2.44, 14.47] | 5.93 [1.04, 12.06] | 0.26 [0.06, 0.41] | 0.23 [0.06, 0.40] | 0.10 [0.00, 1.00] | 0.10 [0.00, 0.23] | 0.05 [0.00, 0.13] |
| nearest_in_corridor | 0.3 | 0.41 [0.28, 0.73] | 5.52 [2.82, 8.05] | 0.58 [0.50, 0.67] | 0.49 [0.42, 0.62] | 13.10 [7.61, 18.44] | 8.97 [4.31, 13.51] | 0.38 [0.25, 0.71] | 0.38 [0.26, 0.67] | 0.00 [0.00, 0.00] | 0.08 [0.03, 0.14] | 0.02 [0.00, 0.04] |
| v2_physics | 0.95 | 0.33 [0.23, 0.55] | 4.55 [2.56, 6.56] | 0.60 [0.48, 0.70] | 0.50 [0.41, 0.60] | 11.31 [6.35, 16.28] | 7.72 [3.51, 12.09] | 0.32 [0.21, 0.61] | 0.36 [0.23, 0.67] | 0.00 [0.00, 0.00] | 0.06 [0.00, 0.09] | 0.02 [0.00, 0.04] |
| v2_physics_nopose | 0.8 | 0.37 [0.27, 0.61] | 2.76 [1.36, 4.16] | 0.71 [0.62, 0.80] | 0.65 [0.58, 0.74] | 9.52 [5.33, 13.64] | 6.62 [3.00, 10.39] | 0.38 [0.28, 0.62] | 0.41 [0.30, 0.64] | 0.00 [0.00, 0.00] | 0.05 [0.00, 0.09] | 0.02 [0.00, 0.03] |
| v2_kalman | 0.5 | 0.38 [0.25, 0.74] | 5.38 [2.77, 7.77] | 0.59 [0.49, 0.69] | 0.47 [0.38, 0.60] | 13.10 [7.33, 18.90] | 8.69 [3.97, 13.58] | 0.38 [0.25, 0.72] | 0.36 [0.22, 0.68] | 0.00 [0.00, 0.10] | 0.10 [0.04, 0.20] | 0.03 [0.00, 0.08] |
| v2_kalman_nopose | 0.4 | 0.38 [0.28, 0.66] | 6.48 [3.82, 9.18] | 0.52 [0.42, 0.62] | 0.41 [0.32, 0.54] | 13.52 [8.46, 18.37] | 8.83 [4.52, 13.17] | 0.38 [0.27, 0.66] | 0.38 [0.28, 0.61] | 0.00 [0.00, 0.20] | 0.13 [0.05, 0.21] | 0.02 [0.00, 0.03] |
| v2_gru | 0.4 | 0.40 [0.30, 0.62] | 4.83 [2.29, 7.23] | 0.61 [0.50, 0.71] | 0.51 [0.41, 0.63] | 12.41 [6.53, 18.41] | 9.10 [3.87, 14.61] | 0.43 [0.32, 0.67] | 0.46 [0.37, 0.68] | 0.00 [0.00, 0.00] | 0.10 [0.03, 0.18] | 0.02 [0.00, 0.04] |
| v2_gru_nopose | 0.2 | 0.40 [0.29, 0.71] | 8.55 [4.88, 12.04] | 0.49 [0.36, 0.61] | 0.36 [0.27, 0.49] | 16.69 [10.16, 23.01] | 12.41 [6.14, 18.49] | 0.38 [0.28, 0.69] | 0.44 [0.31, 0.75] | 0.00 [0.00, 0.60] | 0.14 [0.08, 0.24] | 0.05 [0.00, 0.14] |

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
| v1_k0 | 0.92 | 0.81 | 0.79 | 0.73 | 0.67 |
| v1_k0_announcer | 0.93 | 0.82 | 0.80 | 0.75 | 0.69 |
| nearest_in_corridor | 0.98 | 0.91 | 0.89 | 0.85 | 0.79 |
| v2_physics | n/a |
| v2_physics_nopose | 0.97 | 0.89 | 0.83 | 0.73 | 0.56 |
| v2_kalman | 0.95 | 0.88 | 0.83 | 0.69 | 0.62 |
| v2_kalman_nopose | 0.93 | 0.86 | 0.79 | 0.70 | 0.58 |
| v2_gru | 0.97 | 0.89 | 0.85 | 0.68 | 0.58 |
| v2_gru_nopose | 0.95 | 0.89 | 0.85 | 0.68 | 0.55 |

## Operating curves (test; every knob value)

Recall (warned) against false alerts per minute; the CSV next to this report has every column.

| Arm | knob | warned | false/min | precision | alerts/min |
|---|---:|---:|---:|---:|---:|
| v1_k0 | 0.05 | 0.37 | 6.8 | 0.69 | 21.9 |
| v1_k0 | 0.1 | 0.37 | 5.8 | 0.71 | 20.0 |
| v1_k0 | 0.2 | 0.35 | 5.0 | 0.72 | 17.9 |
| v1_k0 | 0.35 | 0.32 | 4.1 | 0.72 | 14.8 |
| v1_k0 | 0.5 | 0.30 | 4.0 | 0.71 | 13.8 |
| v1_k0 | 0.75 | 0.30 | 3.3 | 0.73 | 12.4 |
| v1_k0 | 1 | 0.29 | 2.9 | 0.74 | 11.2 |
| v1_k0 | 1.5 | 0.27 | 2.1 | 0.77 | 9.1 |
| v1_k0 | 2 | 0.25 | 1.7 | 0.80 | 8.3 |
| v1_k0 | 3 | 0.25 | 1.4 | 0.81 | 7.3 |
| v1_k0 | 5 | 0.24 | 1.1 | 0.84 | 6.9 |
| v1_k0 | 8 | 0.24 | 1.1 | 0.84 | 6.9 |
| v1_k0 | 12 | 0.22 | 1.1 | 0.83 | 6.6 |
| v1_k0_announcer | 0.05 | 0.32 | 3.7 | 0.62 | 9.8 |
| v1_k0_announcer | 0.1 | 0.32 | 3.2 | 0.65 | 9.0 |
| v1_k0_announcer | 0.2 | 0.29 | 2.6 | 0.67 | 8.0 |
| v1_k0_announcer | 0.35 | 0.25 | 2.6 | 0.65 | 7.4 |
| v1_k0_announcer | 0.5 | 0.25 | 2.6 | 0.65 | 7.4 |
| v1_k0_announcer | 0.75 | 0.25 | 2.5 | 0.65 | 7.2 |
| v1_k0_announcer | 1 | 0.25 | 2.2 | 0.67 | 6.8 |
| v1_k0_announcer | 1.5 | 0.25 | 1.8 | 0.71 | 6.2 |
| v1_k0_announcer | 2 | 0.24 | 1.7 | 0.73 | 6.1 |
| v1_k0_announcer | 3 | 0.24 | 1.4 | 0.76 | 5.7 |
| v1_k0_announcer | 5 | 0.24 | 1.1 | 0.79 | 5.4 |
| v1_k0_announcer | 8 | 0.24 | 1.1 | 0.79 | 5.4 |
| v1_k0_announcer | 12 | 0.22 | 1.1 | 0.79 | 5.4 |
| nearest_in_corridor | 0.01 | 0.38 | 7.3 | 0.50 | 14.8 |
| nearest_in_corridor | 0.03 | 0.38 | 7.3 | 0.50 | 14.8 |
| nearest_in_corridor | 0.05 | 0.38 | 7.3 | 0.50 | 14.8 |
| nearest_in_corridor | 0.1 | 0.38 | 7.3 | 0.50 | 14.8 |
| nearest_in_corridor | 0.15 | 0.38 | 7.3 | 0.50 | 14.8 |
| nearest_in_corridor | 0.2 | 0.38 | 7.3 | 0.50 | 14.8 |
| nearest_in_corridor | 0.3 | 0.41 | 5.5 | 0.58 | 13.1 |
| nearest_in_corridor | 0.4 | 0.32 | 3.3 | 0.67 | 10.1 |
| nearest_in_corridor | 0.5 | 0.21 | 3.0 | 0.61 | 7.7 |
| nearest_in_corridor | 0.6 | 0.16 | 1.9 | 0.68 | 6.1 |
| nearest_in_corridor | 0.7 | 0.13 | 1.2 | 0.72 | 4.4 |
| nearest_in_corridor | 0.8 | 0.06 | 1.1 | 0.62 | 2.9 |
| nearest_in_corridor | 0.9 | 0.02 | 0.7 | 0.67 | 2.1 |
| nearest_in_corridor | 0.95 | 0.02 | 0.6 | 0.71 | 1.9 |
| v2_physics | 0.01 | 0.35 | 5.7 | 0.57 | 13.2 |
| v2_physics | 0.03 | 0.35 | 5.7 | 0.57 | 13.2 |
| v2_physics | 0.05 | 0.35 | 5.7 | 0.57 | 13.2 |
| v2_physics | 0.1 | 0.35 | 5.7 | 0.57 | 13.2 |
| v2_physics | 0.15 | 0.35 | 5.7 | 0.57 | 13.2 |
| v2_physics | 0.2 | 0.35 | 5.7 | 0.57 | 13.2 |
| v2_physics | 0.3 | 0.35 | 5.7 | 0.57 | 13.2 |
| v2_physics | 0.4 | 0.35 | 5.8 | 0.56 | 13.2 |
| v2_physics | 0.5 | 0.35 | 5.8 | 0.56 | 13.2 |
| v2_physics | 0.6 | 0.35 | 5.7 | 0.57 | 13.1 |
| v2_physics | 0.7 | 0.35 | 5.5 | 0.57 | 13.0 |
| v2_physics | 0.8 | 0.32 | 5.2 | 0.56 | 12.0 |
| v2_physics | 0.9 | 0.32 | 5.2 | 0.56 | 12.0 |
| v2_physics | 0.95 | 0.33 | 4.6 | 0.60 | 11.3 |
| v2_physics_nopose | 0.01 | 0.40 | 3.4 | 0.68 | 10.9 |
| v2_physics_nopose | 0.03 | 0.40 | 3.4 | 0.68 | 10.9 |
| v2_physics_nopose | 0.05 | 0.40 | 3.4 | 0.68 | 10.9 |
| v2_physics_nopose | 0.1 | 0.40 | 3.4 | 0.68 | 10.9 |
| v2_physics_nopose | 0.15 | 0.40 | 3.4 | 0.68 | 10.9 |
| v2_physics_nopose | 0.2 | 0.40 | 3.4 | 0.68 | 10.9 |
| v2_physics_nopose | 0.3 | 0.40 | 3.4 | 0.68 | 10.9 |
| v2_physics_nopose | 0.4 | 0.40 | 3.4 | 0.68 | 10.9 |
| v2_physics_nopose | 0.5 | 0.40 | 3.4 | 0.68 | 10.9 |
| v2_physics_nopose | 0.6 | 0.40 | 3.2 | 0.70 | 10.6 |
| v2_physics_nopose | 0.7 | 0.40 | 2.9 | 0.72 | 10.3 |
| v2_physics_nopose | 0.8 | 0.37 | 2.8 | 0.71 | 9.5 |
| v2_physics_nopose | 0.9 | 0.37 | 2.8 | 0.71 | 9.5 |
| v2_physics_nopose | 0.95 | 0.37 | 2.5 | 0.73 | 9.2 |
| v2_kalman | 0.01 | 0.32 | 12.6 | 0.36 | 19.6 |
| v2_kalman | 0.03 | 0.32 | 12.4 | 0.36 | 19.4 |
| v2_kalman | 0.05 | 0.32 | 12.4 | 0.36 | 19.4 |
| v2_kalman | 0.1 | 0.32 | 11.7 | 0.38 | 18.9 |
| v2_kalman | 0.15 | 0.35 | 10.8 | 0.41 | 18.3 |
| v2_kalman | 0.2 | 0.35 | 9.9 | 0.44 | 17.7 |
| v2_kalman | 0.3 | 0.33 | 8.3 | 0.49 | 16.1 |
| v2_kalman | 0.4 | 0.37 | 7.0 | 0.52 | 14.8 |
| v2_kalman | 0.5 | 0.38 | 5.4 | 0.59 | 13.1 |
| v2_kalman | 0.6 | 0.40 | 3.2 | 0.71 | 10.9 |
| v2_kalman | 0.7 | 0.33 | 2.9 | 0.70 | 9.7 |
| v2_kalman | 0.8 | 0.29 | 2.8 | 0.68 | 8.7 |
| v2_kalman | 0.9 | 0.21 | 1.9 | 0.75 | 7.9 |
| v2_kalman | 0.95 | 0.19 | 1.7 | 0.77 | 7.3 |
| v2_kalman_nopose | 0.01 | 0.30 | 13.1 | 0.32 | 19.2 |
| v2_kalman_nopose | 0.03 | 0.30 | 13.1 | 0.32 | 19.2 |
| v2_kalman_nopose | 0.05 | 0.30 | 13.0 | 0.32 | 19.0 |
| v2_kalman_nopose | 0.1 | 0.30 | 13.0 | 0.32 | 19.0 |
| v2_kalman_nopose | 0.15 | 0.33 | 11.3 | 0.37 | 17.9 |
| v2_kalman_nopose | 0.2 | 0.35 | 10.5 | 0.39 | 17.2 |
| v2_kalman_nopose | 0.3 | 0.35 | 8.3 | 0.46 | 15.3 |
| v2_kalman_nopose | 0.4 | 0.38 | 6.5 | 0.52 | 13.5 |
| v2_kalman_nopose | 0.5 | 0.37 | 4.8 | 0.60 | 12.0 |
| v2_kalman_nopose | 0.6 | 0.40 | 2.5 | 0.74 | 9.4 |
| v2_kalman_nopose | 0.7 | 0.35 | 2.2 | 0.73 | 8.3 |
| v2_kalman_nopose | 0.8 | 0.30 | 1.9 | 0.74 | 7.4 |
| v2_kalman_nopose | 0.9 | 0.24 | 1.1 | 0.82 | 6.2 |
| v2_kalman_nopose | 0.95 | 0.19 | 1.0 | 0.83 | 5.7 |
| v2_gru | 0.01 | 0.35 | 13.2 | 0.38 | 21.2 |
| v2_gru | 0.03 | 0.35 | 13.1 | 0.38 | 21.1 |
| v2_gru | 0.05 | 0.35 | 12.8 | 0.38 | 20.8 |
| v2_gru | 0.1 | 0.37 | 11.9 | 0.41 | 20.0 |
| v2_gru | 0.15 | 0.41 | 9.5 | 0.47 | 18.1 |
| v2_gru | 0.2 | 0.38 | 8.7 | 0.50 | 17.2 |
| v2_gru | 0.3 | 0.40 | 7.2 | 0.52 | 15.0 |
| v2_gru | 0.4 | 0.40 | 4.8 | 0.61 | 12.4 |
| v2_gru | 0.5 | 0.37 | 4.1 | 0.63 | 11.3 |
| v2_gru | 0.6 | 0.29 | 2.9 | 0.70 | 9.8 |
| v2_gru | 0.7 | 0.25 | 2.3 | 0.72 | 8.3 |
| v2_gru | 0.8 | 0.21 | 1.1 | 0.83 | 6.5 |
| v2_gru | 0.9 | 0.14 | 1.1 | 0.76 | 4.7 |
| v2_gru | 0.95 | 0.14 | 0.4 | 0.87 | 3.2 |
| v2_gru_nopose | 0.01 | 0.30 | 13.5 | 0.36 | 21.0 |
| v2_gru_nopose | 0.03 | 0.30 | 13.4 | 0.36 | 20.8 |
| v2_gru_nopose | 0.05 | 0.30 | 13.1 | 0.36 | 20.6 |
| v2_gru_nopose | 0.1 | 0.32 | 12.1 | 0.38 | 19.7 |
| v2_gru_nopose | 0.15 | 0.37 | 9.5 | 0.46 | 17.7 |
| v2_gru_nopose | 0.2 | 0.40 | 8.6 | 0.49 | 16.7 |
| v2_gru_nopose | 0.3 | 0.41 | 6.5 | 0.55 | 14.5 |
| v2_gru_nopose | 0.4 | 0.40 | 4.1 | 0.65 | 11.7 |
| v2_gru_nopose | 0.5 | 0.37 | 3.4 | 0.68 | 10.6 |
| v2_gru_nopose | 0.6 | 0.29 | 2.5 | 0.72 | 9.0 |
| v2_gru_nopose | 0.7 | 0.25 | 1.8 | 0.76 | 7.4 |
| v2_gru_nopose | 0.8 | 0.19 | 0.8 | 0.85 | 5.7 |
| v2_gru_nopose | 0.9 | 0.13 | 0.8 | 0.78 | 3.7 |
| v2_gru_nopose | 0.95 | 0.13 | 0.3 | 0.88 | 2.3 |

Classes: true = the object enters the zone later within the horizon; late = it is already in the zone; false = it doesn't enter within the horizon. precision = (true + late) / all. warned = an announcement about the arriving track in the 4 s up to and including arrival.
Open-loop: the recorded walker never heard these alerts. Labels and inputs share SANPO's depth.
