# Threat-engine replay - corridor_depthfix

Label: in the walking corridor (width 0.75 m + class half-width, 0 < forward <= 3 m along the direction of travel from fixed_camera_poses.csv) within 20 steps (4 s); tracked objects only.

SANPO official test split: 83 sessions, 96 arrivals (a track first seen before it entered the zone). Operating points chosen on 47 validation sessions, then scored once on test. GRU: `models/GRU/GRU/runs/corridor_h8_seed0`. Generated 2026-10-17 by `python tools/replay_threat_engine.py` (353 s).

## Opportunity window (first sighting -> arrival), test

Median 2.0 s; 27% of arrivals were first seen less than 1 s before arriving, 47% less than 2 s, 25% at least 4 s before. Lead time can't exceed the window, so lead is reported conditionally below.

## Operating point `budget_2` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | 12 | 0.08 [0.04, 0.17] | 3.78 [2.18, 5.43] | 0.49 [0.25, 0.70] | 0.38 [0.20, 0.55] | 7.37 [4.85, 9.69] | 2.30 [1.16, 3.34] | 0.07 [0.03, 0.14] | 0.04 [0.00, 0.10] | 1.00 [0.00, 2.20] | 0.03 [0.00, 0.07] | 0.03 [0.00, 0.07] |
| v1_k0_announcer | 12 | 0.08 [0.04, 0.17] | 3.04 [1.85, 4.23] | 0.40 [0.20, 0.59] | 0.33 [0.18, 0.48] | 5.07 [3.56, 6.31] | 2.21 [1.08, 3.26] | 0.07 [0.03, 0.14] | 0.04 [0.00, 0.10] | 1.00 [0.00, 2.20] | 0.03 [0.00, 0.07] | 0.03 [0.00, 0.07] |
| nearest_in_corridor | — | no setting meets the rule on validation |||||||||||
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | — | no setting meets the rule on validation |||||||||||
| v2_kalman_nopose | — | no setting meets the rule on validation |||||||||||
| v2_gru | — | no setting meets the rule on validation |||||||||||
| v2_gru_nopose | — | no setting meets the rule on validation |||||||||||

## Operating point `budget_5` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | 5 | 0.08 [0.04, 0.17] | 7.37 [4.74, 10.30] | 0.35 [0.17, 0.53] | 0.26 [0.12, 0.39] | 11.33 [8.06, 14.58] | 3.68 [2.51, 4.84] | 0.07 [0.03, 0.14] | 0.04 [0.00, 0.10] | 1.00 [0.00, 2.20] | 0.04 [0.01, 0.09] | 0.03 [0.00, 0.07] |
| v1_k0_announcer | 5 | 0.08 [0.04, 0.17] | 4.79 [3.36, 6.32] | 0.32 [0.17, 0.48] | 0.25 [0.13, 0.37] | 7.09 [5.37, 8.78] | 3.41 [2.23, 4.50] | 0.07 [0.03, 0.14] | 0.04 [0.00, 0.10] | 1.00 [0.00, 2.20] | 0.04 [0.01, 0.09] | 0.03 [0.00, 0.07] |
| nearest_in_corridor | 0.95 | 0.29 [0.22, 0.40] | 7.46 [4.59, 10.70] | 0.44 [0.34, 0.52] | 0.35 [0.24, 0.43] | 13.36 [7.92, 19.24] | 9.03 [3.74, 14.68] | 0.36 [0.28, 0.51] | 0.29 [0.20, 0.44] | 0.00 [0.00, 0.40] | 0.11 [0.05, 0.20] | 0.03 [0.00, 0.06] |
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | — | no setting meets the rule on validation |||||||||||
| v2_kalman_nopose | — | no setting meets the rule on validation |||||||||||
| v2_gru | 0.8 | 0.35 [0.26, 0.53] | 8.47 [5.15, 12.11] | 0.56 [0.46, 0.66] | 0.48 [0.37, 0.58] | 19.34 [11.94, 26.74] | 13.36 [6.35, 20.43] | 0.37 [0.29, 0.55] | 0.31 [0.22, 0.49] | 0.00 [0.00, 0.40] | 0.12 [0.05, 0.22] | 0.04 [0.01, 0.08] |
| v2_gru_nopose | — | no setting meets the rule on validation |||||||||||

## Operating point `budget_10` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | 5 | 0.08 [0.04, 0.17] | 7.37 [4.74, 10.30] | 0.35 [0.17, 0.53] | 0.26 [0.12, 0.39] | 11.33 [8.06, 14.58] | 3.68 [2.51, 4.84] | 0.07 [0.03, 0.14] | 0.04 [0.00, 0.10] | 1.00 [0.00, 2.20] | 0.04 [0.01, 0.09] | 0.03 [0.00, 0.07] |
| v1_k0_announcer | 1.5 | 0.19 [0.10, 0.36] | 8.20 [6.07, 10.41] | 0.28 [0.15, 0.41] | 0.19 [0.10, 0.28] | 11.42 [8.75, 14.07] | 6.54 [4.50, 8.53] | 0.21 [0.10, 0.44] | 0.18 [0.03, 0.50] | 0.80 [0.00, 2.20] | 0.10 [0.05, 0.21] | 0.07 [0.02, 0.15] |
| nearest_in_corridor | 0.4 | 0.40 [0.29, 0.57] | 12.62 [8.83, 16.61] | 0.45 [0.35, 0.52] | 0.36 [0.27, 0.43] | 22.84 [15.26, 30.64] | 15.29 [8.55, 22.09] | 0.47 [0.36, 0.71] | 0.41 [0.29, 0.68] | 0.20 [0.00, 0.40] | 0.18 [0.09, 0.31] | 0.06 [0.02, 0.12] |
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | 0.95 | 0.39 [0.28, 0.58] | 9.76 [5.93, 13.70] | 0.52 [0.41, 0.62] | 0.43 [0.32, 0.52] | 20.36 [13.17, 27.89] | 14.09 [7.21, 21.15] | 0.47 [0.35, 0.78] | 0.43 [0.31, 0.80] | 0.00 [0.00, 0.40] | 0.17 [0.09, 0.30] | 0.07 [0.03, 0.15] |
| v2_kalman_nopose | — | no setting meets the rule on validation |||||||||||
| v2_gru | 0.7 | 0.44 [0.33, 0.65] | 9.03 [5.59, 12.67] | 0.56 [0.47, 0.64] | 0.48 [0.37, 0.57] | 20.54 [13.10, 28.26] | 14.18 [7.28, 21.46] | 0.44 [0.34, 0.69] | 0.39 [0.28, 0.69] | 0.00 [0.00, 0.40] | 0.16 [0.08, 0.28] | 0.06 [0.02, 0.13] |
| v2_gru_nopose | 0.7 | 0.48 [0.39, 0.66] | 10.13 [6.61, 13.96] | 0.54 [0.44, 0.63] | 0.46 [0.34, 0.55] | 22.01 [14.02, 30.03] | 15.47 [7.77, 23.28] | 0.49 [0.40, 0.69] | 0.45 [0.33, 0.69] | 0.10 [0.00, 0.40] | 0.19 [0.10, 0.31] | 0.07 [0.02, 0.14] |

## Operating point `recall_50` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | 0.2 | 0.59 [0.51, 0.74] | 85.39 [71.08, 101.29] | 0.23 [0.12, 0.33] | 0.14 [0.07, 0.21] | 110.99 [94.80, 127.66] | 42.74 [34.44, 50.28] | 0.59 [0.46, 0.74] | 0.53 [0.33, 0.74] | 1.00 [0.60, 1.60] | 0.42 [0.34, 0.54] | 0.22 [0.12, 0.33] |
| v1_k0_announcer | 0.05 | 0.52 [0.44, 0.69] | 32.51 [28.82, 36.92] | 0.21 [0.12, 0.30] | 0.12 [0.06, 0.17] | 41.26 [37.87, 44.47] | 32.70 [29.37, 35.69] | 0.51 [0.42, 0.69] | 0.45 [0.28, 0.69] | 1.00 [0.60, 1.60] | 0.39 [0.29, 0.56] | 0.21 [0.13, 0.35] |
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
| v1_k0 | 0.97 | 0.92 | 0.90 | 0.90 | 0.90 |
| v1_k0_announcer | 0.97 | 0.92 | 0.90 | 0.90 | 0.90 |
| nearest_in_corridor | 1.00 | 0.95 | 0.75 | 0.73 | 0.65 |
| v2_physics | n/a |
| v2_physics_nopose | n/a |
| v2_kalman | n/a |
| v2_kalman_nopose | n/a |
| v2_gru | 0.97 | 0.89 | 0.71 | 0.68 | 0.58 |
| v2_gru_nopose | n/a |

## Operating curves (test; every knob value)

Recall (warned) against false alerts per minute; the CSV next to this report has every column.

| Arm | knob | warned | false/min | precision | alerts/min |
|---|---:|---:|---:|---:|---:|
| v1_k0 | 0.05 | 0.78 | 117.9 | 0.24 | 154.7 |
| v1_k0 | 0.1 | 0.73 | 107.1 | 0.24 | 140.1 |
| v1_k0 | 0.2 | 0.59 | 85.4 | 0.23 | 111.0 |
| v1_k0 | 0.35 | 0.50 | 66.4 | 0.23 | 86.1 |
| v1_k0 | 0.5 | 0.44 | 53.2 | 0.23 | 68.8 |
| v1_k0 | 0.75 | 0.35 | 38.1 | 0.23 | 49.4 |
| v1_k0 | 1 | 0.26 | 28.3 | 0.23 | 36.9 |
| v1_k0 | 1.5 | 0.21 | 17.9 | 0.25 | 23.7 |
| v1_k0 | 2 | 0.12 | 14.8 | 0.24 | 19.5 |
| v1_k0 | 3 | 0.10 | 11.2 | 0.27 | 15.5 |
| v1_k0 | 5 | 0.08 | 7.4 | 0.35 | 11.3 |
| v1_k0 | 8 | 0.08 | 5.1 | 0.43 | 8.8 |
| v1_k0 | 12 | 0.08 | 3.8 | 0.49 | 7.4 |
| v1_k0_announcer | 0.05 | 0.52 | 32.5 | 0.21 | 41.3 |
| v1_k0_announcer | 0.1 | 0.48 | 30.7 | 0.22 | 39.1 |
| v1_k0_announcer | 0.2 | 0.41 | 26.8 | 0.21 | 33.8 |
| v1_k0_announcer | 0.35 | 0.34 | 21.7 | 0.21 | 27.6 |
| v1_k0_announcer | 0.5 | 0.32 | 17.8 | 0.24 | 23.3 |
| v1_k0_announcer | 0.75 | 0.28 | 14.4 | 0.26 | 19.4 |
| v1_k0_announcer | 1 | 0.22 | 11.7 | 0.27 | 15.9 |
| v1_k0_announcer | 1.5 | 0.19 | 8.2 | 0.28 | 11.4 |
| v1_k0_announcer | 2 | 0.11 | 7.0 | 0.26 | 9.5 |
| v1_k0_announcer | 3 | 0.09 | 5.7 | 0.29 | 8.0 |
| v1_k0_announcer | 5 | 0.08 | 4.8 | 0.32 | 7.1 |
| v1_k0_announcer | 8 | 0.08 | 3.9 | 0.36 | 6.1 |
| v1_k0_announcer | 12 | 0.08 | 3.0 | 0.40 | 5.1 |
| nearest_in_corridor | 0.01 | 0.42 | 29.3 | 0.26 | 39.5 |
| nearest_in_corridor | 0.03 | 0.43 | 27.0 | 0.28 | 37.4 |
| nearest_in_corridor | 0.05 | 0.43 | 25.6 | 0.29 | 36.0 |
| nearest_in_corridor | 0.1 | 0.43 | 22.7 | 0.32 | 33.2 |
| nearest_in_corridor | 0.15 | 0.43 | 20.3 | 0.34 | 30.8 |
| nearest_in_corridor | 0.2 | 0.43 | 18.3 | 0.36 | 28.6 |
| nearest_in_corridor | 0.3 | 0.42 | 15.2 | 0.41 | 25.8 |
| nearest_in_corridor | 0.4 | 0.40 | 12.6 | 0.45 | 22.8 |
| nearest_in_corridor | 0.5 | 0.34 | 11.4 | 0.44 | 20.3 |
| nearest_in_corridor | 0.6 | 0.32 | 10.5 | 0.44 | 18.6 |
| nearest_in_corridor | 0.7 | 0.31 | 9.5 | 0.43 | 16.6 |
| nearest_in_corridor | 0.8 | 0.29 | 8.5 | 0.42 | 14.6 |
| nearest_in_corridor | 0.9 | 0.29 | 7.9 | 0.43 | 13.8 |
| nearest_in_corridor | 0.95 | 0.29 | 7.5 | 0.44 | 13.4 |
| v2_physics | 0.01 | 0.42 | 25.6 | 0.29 | 36.3 |
| v2_physics | 0.03 | 0.42 | 25.6 | 0.29 | 36.3 |
| v2_physics | 0.05 | 0.42 | 25.6 | 0.29 | 36.3 |
| v2_physics | 0.1 | 0.42 | 25.6 | 0.29 | 36.3 |
| v2_physics | 0.15 | 0.42 | 25.6 | 0.29 | 36.3 |
| v2_physics | 0.2 | 0.42 | 25.6 | 0.29 | 36.3 |
| v2_physics | 0.3 | 0.42 | 25.1 | 0.30 | 35.7 |
| v2_physics | 0.4 | 0.42 | 24.8 | 0.30 | 35.4 |
| v2_physics | 0.5 | 0.42 | 24.6 | 0.30 | 35.2 |
| v2_physics | 0.6 | 0.42 | 24.2 | 0.30 | 34.7 |
| v2_physics | 0.7 | 0.42 | 23.7 | 0.31 | 34.2 |
| v2_physics | 0.8 | 0.42 | 23.0 | 0.32 | 33.6 |
| v2_physics | 0.9 | 0.42 | 21.9 | 0.32 | 32.4 |
| v2_physics | 0.95 | 0.42 | 20.3 | 0.34 | 30.8 |
| v2_physics_nopose | 0.01 | 0.48 | 29.2 | 0.26 | 39.7 |
| v2_physics_nopose | 0.03 | 0.48 | 29.2 | 0.26 | 39.7 |
| v2_physics_nopose | 0.05 | 0.48 | 29.2 | 0.26 | 39.7 |
| v2_physics_nopose | 0.1 | 0.48 | 29.2 | 0.26 | 39.7 |
| v2_physics_nopose | 0.15 | 0.48 | 29.2 | 0.26 | 39.7 |
| v2_physics_nopose | 0.2 | 0.48 | 29.2 | 0.26 | 39.7 |
| v2_physics_nopose | 0.3 | 0.48 | 28.6 | 0.27 | 39.1 |
| v2_physics_nopose | 0.4 | 0.48 | 28.1 | 0.27 | 38.7 |
| v2_physics_nopose | 0.5 | 0.48 | 27.8 | 0.28 | 38.4 |
| v2_physics_nopose | 0.6 | 0.48 | 27.4 | 0.28 | 38.0 |
| v2_physics_nopose | 0.7 | 0.48 | 27.0 | 0.28 | 37.6 |
| v2_physics_nopose | 0.8 | 0.48 | 26.4 | 0.29 | 37.0 |
| v2_physics_nopose | 0.9 | 0.47 | 25.2 | 0.29 | 35.7 |
| v2_physics_nopose | 0.95 | 0.47 | 23.9 | 0.30 | 34.3 |
| v2_kalman | 0.01 | 0.48 | 51.3 | 0.19 | 63.6 |
| v2_kalman | 0.03 | 0.48 | 44.9 | 0.21 | 57.1 |
| v2_kalman | 0.05 | 0.48 | 40.4 | 0.23 | 52.7 |
| v2_kalman | 0.1 | 0.46 | 36.6 | 0.25 | 48.7 |
| v2_kalman | 0.15 | 0.46 | 32.1 | 0.27 | 43.9 |
| v2_kalman | 0.2 | 0.46 | 29.8 | 0.29 | 41.8 |
| v2_kalman | 0.3 | 0.45 | 24.8 | 0.32 | 36.3 |
| v2_kalman | 0.4 | 0.41 | 20.6 | 0.36 | 32.1 |
| v2_kalman | 0.5 | 0.41 | 18.2 | 0.39 | 29.9 |
| v2_kalman | 0.6 | 0.42 | 15.6 | 0.43 | 27.2 |
| v2_kalman | 0.7 | 0.43 | 13.9 | 0.45 | 25.5 |
| v2_kalman | 0.8 | 0.43 | 11.9 | 0.49 | 23.1 |
| v2_kalman | 0.9 | 0.41 | 10.0 | 0.52 | 20.9 |
| v2_kalman | 0.95 | 0.39 | 9.8 | 0.52 | 20.4 |
| v2_kalman_nopose | 0.01 | 0.49 | 51.6 | 0.19 | 63.8 |
| v2_kalman_nopose | 0.03 | 0.50 | 45.4 | 0.21 | 57.8 |
| v2_kalman_nopose | 0.05 | 0.51 | 42.1 | 0.23 | 54.4 |
| v2_kalman_nopose | 0.1 | 0.51 | 37.4 | 0.25 | 49.7 |
| v2_kalman_nopose | 0.15 | 0.50 | 34.2 | 0.27 | 46.5 |
| v2_kalman_nopose | 0.2 | 0.49 | 31.4 | 0.29 | 43.9 |
| v2_kalman_nopose | 0.3 | 0.51 | 27.1 | 0.31 | 39.5 |
| v2_kalman_nopose | 0.4 | 0.50 | 22.8 | 0.35 | 35.2 |
| v2_kalman_nopose | 0.5 | 0.50 | 20.2 | 0.38 | 32.5 |
| v2_kalman_nopose | 0.6 | 0.52 | 17.3 | 0.42 | 29.8 |
| v2_kalman_nopose | 0.7 | 0.51 | 15.1 | 0.45 | 27.5 |
| v2_kalman_nopose | 0.8 | 0.50 | 13.2 | 0.48 | 25.4 |
| v2_kalman_nopose | 0.9 | 0.50 | 12.1 | 0.50 | 23.9 |
| v2_kalman_nopose | 0.95 | 0.48 | 11.6 | 0.50 | 23.1 |
| v2_gru | 0.01 | 0.49 | 49.1 | 0.21 | 62.1 |
| v2_gru | 0.03 | 0.50 | 38.7 | 0.25 | 51.8 |
| v2_gru | 0.05 | 0.50 | 34.1 | 0.28 | 47.3 |
| v2_gru | 0.1 | 0.48 | 27.2 | 0.33 | 40.3 |
| v2_gru | 0.15 | 0.49 | 22.6 | 0.37 | 35.7 |
| v2_gru | 0.2 | 0.49 | 18.9 | 0.41 | 31.9 |
| v2_gru | 0.3 | 0.47 | 16.0 | 0.45 | 28.9 |
| v2_gru | 0.4 | 0.49 | 13.3 | 0.49 | 25.8 |
| v2_gru | 0.5 | 0.47 | 11.1 | 0.52 | 23.3 |
| v2_gru | 0.6 | 0.46 | 9.9 | 0.55 | 21.9 |
| v2_gru | 0.7 | 0.44 | 9.0 | 0.56 | 20.5 |
| v2_gru | 0.8 | 0.35 | 8.5 | 0.56 | 19.3 |
| v2_gru | 0.9 | 0.31 | 8.1 | 0.53 | 17.2 |
| v2_gru | 0.95 | 0.31 | 7.7 | 0.54 | 16.9 |
| v2_gru_nopose | 0.01 | 0.53 | 49.8 | 0.21 | 63.0 |
| v2_gru_nopose | 0.03 | 0.54 | 39.8 | 0.25 | 53.0 |
| v2_gru_nopose | 0.05 | 0.54 | 35.2 | 0.27 | 48.4 |
| v2_gru_nopose | 0.1 | 0.52 | 28.0 | 0.32 | 41.3 |
| v2_gru_nopose | 0.15 | 0.52 | 23.7 | 0.36 | 37.0 |
| v2_gru_nopose | 0.2 | 0.52 | 20.1 | 0.39 | 33.2 |
| v2_gru_nopose | 0.3 | 0.51 | 17.1 | 0.43 | 30.2 |
| v2_gru_nopose | 0.4 | 0.53 | 14.1 | 0.48 | 26.9 |
| v2_gru_nopose | 0.5 | 0.51 | 12.1 | 0.51 | 24.6 |
| v2_gru_nopose | 0.6 | 0.50 | 10.9 | 0.53 | 23.2 |
| v2_gru_nopose | 0.7 | 0.48 | 10.1 | 0.54 | 22.0 |
| v2_gru_nopose | 0.8 | 0.40 | 9.5 | 0.54 | 20.6 |
| v2_gru_nopose | 0.9 | 0.34 | 9.1 | 0.51 | 18.4 |
| v2_gru_nopose | 0.95 | 0.34 | 8.8 | 0.52 | 18.1 |

Classes: true = the object enters the zone later within the horizon; late = it is already in the zone; false = it doesn't enter within the horizon. precision = (true + late) / all. warned = an announcement about the arriving track in the 4 s up to and including arrival.
Open-loop: the recorded walker never heard these alerts. Labels and inputs share SANPO's depth.
