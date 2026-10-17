# Threat-engine replay - corridor_kalman_2026_10_17

Label: in the walking corridor (width 0.75 m + class half-width, 0 < forward <= 3 m along the direction of travel from fixed_camera_poses.csv) within 20 steps (4 s); tracked objects only.

SANPO official test split: 71 sessions, 63 arrivals (a track first seen before it entered the zone). Operating points chosen on 41 validation sessions, then scored once on test. GRU: `models/GRU/GRU/runs/corridor_h8_seed0`. Generated 2026-10-17 by `python tools/replay_threat_engine.py` (827 s).

## Opportunity window (first sighting -> arrival), test

Median 2.6 s; 25% of arrivals were first seen less than 1 s before arriving, 38% less than 2 s, 27% at least 4 s before. Lead time can't exceed the window, so lead is reported conditionally below.

## Operating point `budget_2` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) |
|---|---:|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||
| nearest_in_corridor | 0.95 | 0.03 [0.00, 0.09] | 2.34 [0.85, 3.94] | 0.35 [0.18, 0.50] | 0.31 [0.12, 0.49] | 3.59 [1.30, 6.01] | 1.93 [0.35, 3.70] | 0.04 [0.00, 0.12] | 0.03 [0.00, 0.08] | 0.20 [0.20, 0.20] |
| v2_physics | — | no setting meets the rule on validation |||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||
| v2_kalman | — | no setting meets the rule on validation |||||||||
| v2_kalman_nopose | — | no setting meets the rule on validation |||||||||
| v2_gru | — | no setting meets the rule on validation |||||||||
| v2_gru_nopose | 0.7 | 0.27 [0.17, 0.48] | 1.93 [0.56, 3.69] | 0.77 [0.62, 0.90] | 0.65 [0.51, 0.83] | 8.28 [4.07, 12.66] | 5.93 [2.06, 10.24] | 0.26 [0.16, 0.50] | 0.28 [0.15, 0.56] | 0.00 [0.00, 0.60] |

## Operating point `budget_5` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) |
|---|---:|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||
| nearest_in_corridor | 0.7 | 0.05 [0.00, 0.12] | 4.14 [1.68, 6.79] | 0.40 [0.23, 0.60] | 0.36 [0.18, 0.59] | 6.90 [3.27, 10.60] | 4.14 [1.41, 7.10] | 0.04 [0.00, 0.12] | 0.03 [0.00, 0.08] | 0.20 [0.20, 0.20] |
| v2_physics | — | no setting meets the rule on validation |||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||
| v2_kalman | 0.95 | 0.22 [0.11, 0.44] | 4.69 [2.31, 7.27] | 0.54 [0.39, 0.66] | 0.46 [0.33, 0.59] | 10.21 [5.42, 14.99] | 6.76 [2.91, 10.84] | 0.28 [0.15, 0.54] | 0.28 [0.12, 0.58] | 0.00 [0.00, 0.40] |
| v2_kalman_nopose | 0.6 | 0.27 [0.17, 0.52] | 4.55 [2.55, 6.58] | 0.58 [0.49, 0.68] | 0.48 [0.34, 0.61] | 10.90 [6.65, 14.99] | 6.34 [2.82, 10.09] | 0.28 [0.18, 0.50] | 0.31 [0.20, 0.57] | 0.00 [0.00, 0.80] |
| v2_gru | 0.6 | 0.32 [0.24, 0.50] | 4.14 [2.06, 6.46] | 0.66 [0.55, 0.74] | 0.55 [0.42, 0.70] | 12.00 [6.24, 17.83] | 8.41 [3.46, 13.69] | 0.32 [0.24, 0.50] | 0.33 [0.24, 0.57] | 0.00 [0.00, 0.40] |
| v2_gru_nopose | 0.5 | 0.37 [0.25, 0.63] | 2.90 [1.43, 4.40] | 0.73 [0.64, 0.82] | 0.58 [0.49, 0.72] | 10.90 [5.60, 16.27] | 7.86 [3.30, 12.68] | 0.38 [0.26, 0.68] | 0.44 [0.30, 0.75] | 0.00 [0.00, 0.50] |

## Operating point `budget_10` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) |
|---|---:|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||
| nearest_in_corridor | 0.4 | 0.19 [0.11, 0.32] | 7.17 [3.71, 10.79] | 0.47 [0.37, 0.57] | 0.40 [0.29, 0.53] | 13.52 [7.37, 19.68] | 8.28 [3.46, 13.53] | 0.21 [0.13, 0.35] | 0.21 [0.11, 0.39] | 0.00 [0.00, 0.80] |
| v2_physics | — | no setting meets the rule on validation |||||||||
| v2_physics_nopose | 0.9 | 0.29 [0.20, 0.50] | 5.10 [3.09, 6.89] | 0.53 [0.46, 0.60] | 0.47 [0.39, 0.56] | 10.90 [6.69, 14.91] | 7.03 [3.32, 10.71] | 0.30 [0.22, 0.50] | 0.33 [0.25, 0.56] | 0.00 [0.00, 0.60] |
| v2_kalman | 0.8 | 0.27 [0.16, 0.48] | 6.21 [3.23, 9.04] | 0.49 [0.35, 0.60] | 0.41 [0.28, 0.53] | 12.14 [6.85, 17.39] | 7.72 [3.48, 12.36] | 0.32 [0.20, 0.59] | 0.33 [0.19, 0.63] | 0.00 [0.00, 0.40] |
| v2_kalman_nopose | 0.4 | 0.32 [0.21, 0.61] | 8.41 [5.54, 11.16] | 0.44 [0.33, 0.54] | 0.33 [0.22, 0.45] | 15.03 [10.14, 19.58] | 9.24 [4.93, 13.47] | 0.30 [0.21, 0.54] | 0.36 [0.26, 0.60] | 0.40 [0.00, 1.60] |
| v2_gru | 0.4 | 0.38 [0.26, 0.67] | 5.52 [3.12, 7.89] | 0.62 [0.48, 0.72] | 0.50 [0.41, 0.59] | 14.34 [7.88, 20.66] | 9.52 [4.36, 14.73] | 0.38 [0.26, 0.67] | 0.38 [0.24, 0.67] | 0.00 [0.00, 0.40] |
| v2_gru_nopose | 0.4 | 0.37 [0.24, 0.67] | 4.14 [2.21, 6.17] | 0.66 [0.52, 0.77] | 0.52 [0.39, 0.64] | 12.00 [6.84, 17.08] | 8.69 [4.01, 13.41] | 0.36 [0.24, 0.67] | 0.38 [0.24, 0.67] | 0.00 [0.00, 0.60] |

## Operating point `recall_50` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) |
|---|---:|---|---|---|---|---|---|---|---|---|
| v1_k0 | 1 | 0.62 [0.54, 0.76] | 81.10 [66.45, 94.66] | 0.21 [0.10, 0.32] | 0.11 [0.04, 0.18] | 102.34 [87.16, 117.98] | 53.24 [42.78, 64.40] | 0.62 [0.52, 0.80] | 0.59 [0.48, 0.82] | 1.40 [0.90, 2.60] |
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||
| nearest_in_corridor | — | no setting meets the rule on validation |||||||||
| v2_physics | — | no setting meets the rule on validation |||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||
| v2_kalman | — | no setting meets the rule on validation |||||||||
| v2_kalman_nopose | — | no setting meets the rule on validation |||||||||
| v2_gru | — | no setting meets the rule on validation |||||||||
| v2_gru_nopose | — | no setting meets the rule on validation |||||||||

## Kaplan-Meier: share of arrivals not yet warned, by time since first sighting (test, `budget_5`)

| Arm | 0.5 s | 1 s | 2 s | 3 s | 4 s |
|---|---:|---:|---:|---:|---:|
| v1_k0 | n/a |
| v1_k0_announcer | n/a |
| nearest_in_corridor | 1.00 | 0.98 | 0.93 | 0.93 | 0.93 |
| v2_physics | n/a |
| v2_physics_nopose | n/a |
| v2_kalman | 0.98 | 0.94 | 0.87 | 0.79 | 0.69 |
| v2_kalman_nopose | 0.95 | 0.91 | 0.89 | 0.76 | 0.61 |
| v2_gru | 0.95 | 0.90 | 0.83 | 0.71 | 0.67 |
| v2_gru_nopose | 0.94 | 0.88 | 0.83 | 0.68 | 0.58 |

## Operating curves (test; every knob value)

Recall (warned) against false alerts per minute; the CSV next to this report has every column.

| Arm | knob | warned | false/min | precision | alerts/min |
|---|---:|---:|---:|---:|---:|
| v1_k0 | 0.05 | 0.73 | 125.2 | 0.20 | 155.6 |
| v1_k0 | 0.1 | 0.73 | 117.7 | 0.20 | 146.8 |
| v1_k0 | 0.2 | 0.73 | 109.4 | 0.20 | 137.4 |
| v1_k0 | 0.35 | 0.71 | 101.1 | 0.20 | 127.0 |
| v1_k0 | 0.5 | 0.67 | 94.8 | 0.20 | 119.2 |
| v1_k0 | 0.75 | 0.62 | 86.5 | 0.21 | 109.2 |
| v1_k0 | 1 | 0.62 | 81.1 | 0.21 | 102.3 |
| v1_k0 | 1.5 | 0.57 | 71.3 | 0.20 | 89.5 |
| v1_k0 | 2 | 0.56 | 65.2 | 0.21 | 82.3 |
| v1_k0 | 3 | 0.52 | 55.2 | 0.21 | 69.9 |
| v1_k0 | 5 | 0.52 | 44.3 | 0.24 | 58.2 |
| v1_k0 | 8 | 0.49 | 39.2 | 0.25 | 52.1 |
| v1_k0 | 12 | 0.48 | 35.6 | 0.25 | 47.4 |
| v1_k0_announcer | 0.05 | 0.57 | 46.6 | 0.17 | 55.9 |
| v1_k0_announcer | 0.1 | 0.57 | 45.2 | 0.17 | 54.2 |
| v1_k0_announcer | 0.2 | 0.57 | 44.3 | 0.16 | 53.0 |
| v1_k0_announcer | 0.35 | 0.57 | 42.3 | 0.17 | 50.8 |
| v1_k0_announcer | 0.5 | 0.57 | 41.0 | 0.17 | 49.1 |
| v1_k0_announcer | 0.75 | 0.56 | 38.8 | 0.18 | 47.0 |
| v1_k0_announcer | 1 | 0.56 | 37.5 | 0.18 | 45.7 |
| v1_k0_announcer | 1.5 | 0.56 | 34.5 | 0.19 | 42.6 |
| v1_k0_announcer | 2 | 0.54 | 33.0 | 0.19 | 40.8 |
| v1_k0_announcer | 3 | 0.51 | 29.8 | 0.20 | 37.4 |
| v1_k0_announcer | 5 | 0.51 | 26.3 | 0.22 | 33.9 |
| v1_k0_announcer | 8 | 0.49 | 24.8 | 0.23 | 32.1 |
| v1_k0_announcer | 12 | 0.48 | 23.2 | 0.23 | 30.2 |
| nearest_in_corridor | 0.01 | 0.33 | 26.3 | 0.22 | 33.9 |
| nearest_in_corridor | 0.03 | 0.32 | 23.7 | 0.24 | 31.0 |
| nearest_in_corridor | 0.05 | 0.32 | 21.4 | 0.26 | 28.8 |
| nearest_in_corridor | 0.1 | 0.32 | 18.6 | 0.27 | 25.7 |
| nearest_in_corridor | 0.15 | 0.30 | 16.6 | 0.29 | 23.4 |
| nearest_in_corridor | 0.2 | 0.30 | 13.2 | 0.35 | 20.4 |
| nearest_in_corridor | 0.3 | 0.30 | 10.1 | 0.42 | 17.4 |
| nearest_in_corridor | 0.4 | 0.19 | 7.2 | 0.47 | 13.5 |
| nearest_in_corridor | 0.5 | 0.17 | 6.1 | 0.44 | 10.9 |
| nearest_in_corridor | 0.6 | 0.11 | 5.2 | 0.40 | 8.7 |
| nearest_in_corridor | 0.7 | 0.05 | 4.1 | 0.40 | 6.9 |
| nearest_in_corridor | 0.8 | 0.03 | 3.3 | 0.38 | 5.4 |
| nearest_in_corridor | 0.9 | 0.03 | 3.0 | 0.33 | 4.6 |
| nearest_in_corridor | 0.95 | 0.03 | 2.3 | 0.35 | 3.6 |
| v2_physics | 0.01 | 0.32 | 15.9 | 0.30 | 22.6 |
| v2_physics | 0.03 | 0.32 | 15.9 | 0.30 | 22.6 |
| v2_physics | 0.05 | 0.32 | 15.9 | 0.30 | 22.6 |
| v2_physics | 0.1 | 0.32 | 15.9 | 0.30 | 22.6 |
| v2_physics | 0.15 | 0.32 | 15.7 | 0.30 | 22.5 |
| v2_physics | 0.2 | 0.32 | 15.7 | 0.30 | 22.5 |
| v2_physics | 0.3 | 0.35 | 15.2 | 0.31 | 22.1 |
| v2_physics | 0.4 | 0.35 | 14.3 | 0.33 | 21.4 |
| v2_physics | 0.5 | 0.35 | 13.8 | 0.34 | 21.0 |
| v2_physics | 0.6 | 0.33 | 13.1 | 0.36 | 20.6 |
| v2_physics | 0.7 | 0.32 | 13.0 | 0.36 | 20.3 |
| v2_physics | 0.8 | 0.29 | 12.0 | 0.36 | 18.6 |
| v2_physics | 0.9 | 0.27 | 11.2 | 0.38 | 17.9 |
| v2_physics | 0.95 | 0.27 | 10.3 | 0.40 | 17.2 |
| v2_physics_nopose | 0.01 | 0.24 | 8.8 | 0.40 | 14.6 |
| v2_physics_nopose | 0.03 | 0.24 | 8.8 | 0.40 | 14.6 |
| v2_physics_nopose | 0.05 | 0.24 | 8.8 | 0.40 | 14.6 |
| v2_physics_nopose | 0.1 | 0.24 | 8.8 | 0.40 | 14.6 |
| v2_physics_nopose | 0.15 | 0.24 | 8.8 | 0.40 | 14.6 |
| v2_physics_nopose | 0.2 | 0.24 | 8.8 | 0.40 | 14.6 |
| v2_physics_nopose | 0.3 | 0.25 | 8.3 | 0.42 | 14.3 |
| v2_physics_nopose | 0.4 | 0.24 | 8.0 | 0.43 | 13.9 |
| v2_physics_nopose | 0.5 | 0.24 | 7.7 | 0.43 | 13.7 |
| v2_physics_nopose | 0.6 | 0.24 | 7.3 | 0.44 | 13.1 |
| v2_physics_nopose | 0.7 | 0.25 | 6.9 | 0.46 | 12.8 |
| v2_physics_nopose | 0.8 | 0.25 | 5.9 | 0.49 | 11.7 |
| v2_physics_nopose | 0.9 | 0.29 | 5.1 | 0.53 | 10.9 |
| v2_physics_nopose | 0.95 | 0.29 | 5.0 | 0.54 | 10.8 |
| v2_kalman | 0.01 | 0.30 | 44.1 | 0.14 | 51.4 |
| v2_kalman | 0.03 | 0.32 | 38.8 | 0.16 | 45.9 |
| v2_kalman | 0.05 | 0.32 | 34.2 | 0.17 | 41.4 |
| v2_kalman | 0.1 | 0.30 | 27.7 | 0.21 | 35.0 |
| v2_kalman | 0.15 | 0.32 | 23.7 | 0.23 | 30.9 |
| v2_kalman | 0.2 | 0.30 | 20.8 | 0.26 | 28.1 |
| v2_kalman | 0.3 | 0.30 | 16.1 | 0.32 | 23.6 |
| v2_kalman | 0.4 | 0.29 | 13.0 | 0.36 | 20.4 |
| v2_kalman | 0.5 | 0.30 | 11.2 | 0.39 | 18.3 |
| v2_kalman | 0.6 | 0.27 | 8.1 | 0.47 | 15.3 |
| v2_kalman | 0.7 | 0.25 | 7.3 | 0.46 | 13.7 |
| v2_kalman | 0.8 | 0.27 | 6.2 | 0.49 | 12.1 |
| v2_kalman | 0.9 | 0.27 | 5.0 | 0.56 | 11.2 |
| v2_kalman | 0.95 | 0.22 | 4.7 | 0.54 | 10.2 |
| v2_kalman_nopose | 0.01 | 0.35 | 39.7 | 0.14 | 45.9 |
| v2_kalman_nopose | 0.03 | 0.35 | 33.9 | 0.16 | 40.3 |
| v2_kalman_nopose | 0.05 | 0.35 | 27.4 | 0.18 | 33.7 |
| v2_kalman_nopose | 0.1 | 0.33 | 20.1 | 0.24 | 26.3 |
| v2_kalman_nopose | 0.15 | 0.32 | 17.5 | 0.25 | 23.3 |
| v2_kalman_nopose | 0.2 | 0.33 | 14.1 | 0.30 | 20.1 |
| v2_kalman_nopose | 0.3 | 0.35 | 10.5 | 0.39 | 17.1 |
| v2_kalman_nopose | 0.4 | 0.32 | 8.4 | 0.44 | 15.0 |
| v2_kalman_nopose | 0.5 | 0.32 | 6.2 | 0.51 | 12.7 |
| v2_kalman_nopose | 0.6 | 0.27 | 4.6 | 0.58 | 10.9 |
| v2_kalman_nopose | 0.7 | 0.25 | 3.6 | 0.61 | 9.1 |
| v2_kalman_nopose | 0.8 | 0.24 | 2.3 | 0.68 | 7.3 |
| v2_kalman_nopose | 0.9 | 0.21 | 2.1 | 0.67 | 6.2 |
| v2_kalman_nopose | 0.95 | 0.19 | 1.8 | 0.70 | 6.1 |
| v2_gru | 0.01 | 0.32 | 53.1 | 0.14 | 62.1 |
| v2_gru | 0.03 | 0.33 | 36.0 | 0.20 | 45.0 |
| v2_gru | 0.05 | 0.37 | 29.2 | 0.23 | 38.1 |
| v2_gru | 0.1 | 0.27 | 19.4 | 0.30 | 27.7 |
| v2_gru | 0.15 | 0.37 | 14.5 | 0.39 | 23.7 |
| v2_gru | 0.2 | 0.40 | 12.1 | 0.44 | 21.7 |
| v2_gru | 0.3 | 0.41 | 8.1 | 0.52 | 17.0 |
| v2_gru | 0.4 | 0.38 | 5.5 | 0.62 | 14.3 |
| v2_gru | 0.5 | 0.40 | 4.1 | 0.68 | 13.0 |
| v2_gru | 0.6 | 0.32 | 4.1 | 0.66 | 12.0 |
| v2_gru | 0.7 | 0.30 | 3.3 | 0.68 | 10.3 |
| v2_gru | 0.8 | 0.21 | 2.3 | 0.72 | 8.4 |
| v2_gru | 0.9 | 0.16 | 2.1 | 0.66 | 6.1 |
| v2_gru | 0.95 | 0.14 | 1.8 | 0.59 | 4.4 |
| v2_gru_nopose | 0.01 | 0.37 | 51.3 | 0.15 | 60.1 |
| v2_gru_nopose | 0.03 | 0.38 | 34.3 | 0.20 | 43.2 |
| v2_gru_nopose | 0.05 | 0.38 | 28.0 | 0.23 | 36.6 |
| v2_gru_nopose | 0.1 | 0.32 | 18.2 | 0.30 | 26.1 |
| v2_gru_nopose | 0.15 | 0.40 | 13.1 | 0.40 | 21.8 |
| v2_gru_nopose | 0.2 | 0.43 | 10.8 | 0.45 | 19.6 |
| v2_gru_nopose | 0.3 | 0.40 | 6.3 | 0.57 | 14.6 |
| v2_gru_nopose | 0.4 | 0.37 | 4.1 | 0.66 | 12.0 |
| v2_gru_nopose | 0.5 | 0.37 | 2.9 | 0.73 | 10.9 |
| v2_gru_nopose | 0.6 | 0.29 | 2.6 | 0.73 | 9.8 |
| v2_gru_nopose | 0.7 | 0.27 | 1.9 | 0.77 | 8.3 |
| v2_gru_nopose | 0.8 | 0.17 | 1.0 | 0.84 | 6.1 |
| v2_gru_nopose | 0.9 | 0.13 | 0.7 | 0.80 | 3.4 |
| v2_gru_nopose | 0.95 | 0.13 | 0.3 | 0.88 | 2.3 |

Classes: true = the object enters the zone later within the horizon; late = it is already in the zone; false = it doesn't enter within the horizon. precision = (true + late) / all. warned = an announcement about the arriving track in the 4 s up to and including arrival.
Open-loop: the recorded walker never heard these alerts. Labels and inputs share SANPO's depth.
