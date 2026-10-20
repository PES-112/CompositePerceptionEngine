# Threat-engine replay - severity_gamma0_headfx_2026_10_20

Label: in the walking corridor (width 0.75 m + class half-width, 0 < forward <= 3 m along the direction of travel from fixed_camera_poses.csv) within 20 steps (4 s); tracked objects only.

SANPO official test split: 85 sessions, 79 arrivals (a track first seen before it entered the zone). Operating points chosen on 48 validation sessions, then scored once on test. GRU: `models/GRU/GRU/runs/corridor_h8_seed0`. Generated 2026-10-20 by `python tools/replay_threat_engine.py` (282 s).

## Opportunity window (first sighting -> arrival), test

Median 2.6 s; 24% of arrivals were first seen less than 1 s before arriving, 41% less than 2 s, 33% at least 4 s before. Lead time can't exceed the window, so lead is reported conditionally below.

## Operating point `budget_2` (test, 95% session-bootstrap CI)

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

## Operating point `budget_5` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||||
| nearest_in_corridor | 0.95 | 0.23 [0.11, 0.35] | 4.59 [2.70, 6.84] | 0.40 [0.25, 0.54] | 0.29 [0.16, 0.43] | 7.70 [4.54, 11.34] | 5.11 [1.99, 8.67] | 0.25 [0.12, 0.37] | 0.28 [0.14, 0.40] | 0.20 [0.00, 0.40] | 0.10 [0.02, 0.19] | 0.03 [0.00, 0.04] |
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | 0.95 | 0.35 [0.22, 0.52] | 7.18 [4.38, 10.38] | 0.48 [0.37, 0.59] | 0.38 [0.25, 0.51] | 13.84 [8.42, 20.07] | 9.35 [4.42, 14.99] | 0.37 [0.23, 0.54] | 0.38 [0.22, 0.60] | 0.20 [0.00, 0.40] | 0.16 [0.06, 0.30] | 0.05 [0.02, 0.09] |
| v2_kalman_nopose | — | no setting meets the rule on validation |||||||||||
| v2_gru | 0.5 | 0.34 [0.23, 0.46] | 6.14 [3.72, 8.92] | 0.54 [0.40, 0.67] | 0.42 [0.29, 0.56] | 13.24 [8.08, 19.34] | 8.83 [3.95, 14.46] | 0.33 [0.19, 0.47] | 0.30 [0.14, 0.44] | 0.20 [0.00, 0.40] | 0.16 [0.06, 0.29] | 0.04 [0.00, 0.06] |
| v2_gru_nopose | — | no setting meets the rule on validation |||||||||||

## Operating point `budget_10` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||||
| nearest_in_corridor | 0.2 | 0.39 [0.24, 0.57] | 12.03 [8.87, 15.47] | 0.37 [0.27, 0.45] | 0.28 [0.19, 0.36] | 19.04 [13.47, 25.34] | 12.37 [7.31, 17.66] | 0.38 [0.22, 0.58] | 0.43 [0.24, 0.67] | 0.20 [0.10, 0.40] | 0.22 [0.12, 0.33] | 0.06 [0.03, 0.12] |
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | 0.7 | 0.38 [0.24, 0.56] | 10.30 [7.17, 13.69] | 0.41 [0.30, 0.52] | 0.32 [0.21, 0.42] | 17.57 [12.10, 23.70] | 11.85 [6.86, 17.27] | 0.40 [0.26, 0.60] | 0.43 [0.26, 0.69] | 0.40 [0.10, 0.60] | 0.22 [0.12, 0.36] | 0.06 [0.03, 0.11] |
| v2_kalman_nopose | 0.95 | 0.33 [0.24, 0.45] | 7.36 [4.28, 10.81] | 0.48 [0.37, 0.60] | 0.38 [0.24, 0.51] | 14.28 [8.81, 20.59] | 9.69 [4.77, 15.84] | 0.33 [0.26, 0.44] | 0.28 [0.18, 0.41] | 0.40 [0.20, 0.80] | 0.19 [0.12, 0.30] | 0.05 [0.02, 0.10] |
| v2_gru | 0.3 | 0.44 [0.36, 0.60] | 8.48 [5.55, 11.58] | 0.49 [0.36, 0.62] | 0.37 [0.25, 0.49] | 16.61 [11.00, 22.86] | 10.64 [5.64, 16.27] | 0.45 [0.35, 0.67] | 0.40 [0.29, 0.65] | 0.20 [0.00, 0.80] | 0.24 [0.16, 0.38] | 0.06 [0.03, 0.13] |
| v2_gru_nopose | 0.3 | 0.42 [0.34, 0.60] | 8.91 [5.77, 12.67] | 0.47 [0.35, 0.58] | 0.35 [0.23, 0.46] | 16.70 [10.92, 23.33] | 10.30 [5.02, 16.29] | 0.42 [0.33, 0.65] | 0.34 [0.26, 0.57] | 0.40 [0.10, 0.80] | 0.24 [0.17, 0.38] | 0.06 [0.03, 0.13] |

## Operating point `early_budget_2` (test, 95% session-bootstrap CI)

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

## Operating point `early_budget_5` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||||
| nearest_in_corridor | 0.95 | 0.23 [0.11, 0.35] | 4.59 [2.70, 6.84] | 0.40 [0.25, 0.54] | 0.29 [0.16, 0.43] | 7.70 [4.54, 11.34] | 5.11 [1.99, 8.67] | 0.25 [0.12, 0.37] | 0.28 [0.14, 0.40] | 0.20 [0.00, 0.40] | 0.10 [0.02, 0.19] | 0.03 [0.00, 0.04] |
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | 0.95 | 0.35 [0.22, 0.52] | 7.18 [4.38, 10.38] | 0.48 [0.37, 0.59] | 0.38 [0.25, 0.51] | 13.84 [8.42, 20.07] | 9.35 [4.42, 14.99] | 0.37 [0.23, 0.54] | 0.38 [0.22, 0.60] | 0.20 [0.00, 0.40] | 0.16 [0.06, 0.30] | 0.05 [0.02, 0.09] |
| v2_kalman_nopose | — | no setting meets the rule on validation |||||||||||
| v2_gru | 0.5 | 0.34 [0.23, 0.46] | 6.14 [3.72, 8.92] | 0.54 [0.40, 0.67] | 0.42 [0.29, 0.56] | 13.24 [8.08, 19.34] | 8.83 [3.95, 14.46] | 0.33 [0.19, 0.47] | 0.30 [0.14, 0.44] | 0.20 [0.00, 0.40] | 0.16 [0.06, 0.29] | 0.04 [0.00, 0.06] |
| v2_gru_nopose | — | no setting meets the rule on validation |||||||||||

## Operating point `early_budget_10` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||||
| nearest_in_corridor | 0.2 | 0.39 [0.24, 0.57] | 12.03 [8.87, 15.47] | 0.37 [0.27, 0.45] | 0.28 [0.19, 0.36] | 19.04 [13.47, 25.34] | 12.37 [7.31, 17.66] | 0.38 [0.22, 0.58] | 0.43 [0.24, 0.67] | 0.20 [0.10, 0.40] | 0.22 [0.12, 0.33] | 0.06 [0.03, 0.12] |
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | 0.7 | 0.38 [0.24, 0.56] | 10.30 [7.17, 13.69] | 0.41 [0.30, 0.52] | 0.32 [0.21, 0.42] | 17.57 [12.10, 23.70] | 11.85 [6.86, 17.27] | 0.40 [0.26, 0.60] | 0.43 [0.26, 0.69] | 0.40 [0.10, 0.60] | 0.22 [0.12, 0.36] | 0.06 [0.03, 0.11] |
| v2_kalman_nopose | 0.95 | 0.33 [0.24, 0.45] | 7.36 [4.28, 10.81] | 0.48 [0.37, 0.60] | 0.38 [0.24, 0.51] | 14.28 [8.81, 20.59] | 9.69 [4.77, 15.84] | 0.33 [0.26, 0.44] | 0.28 [0.18, 0.41] | 0.40 [0.20, 0.80] | 0.19 [0.12, 0.30] | 0.05 [0.02, 0.10] |
| v2_gru | 0.3 | 0.44 [0.36, 0.60] | 8.48 [5.55, 11.58] | 0.49 [0.36, 0.62] | 0.37 [0.25, 0.49] | 16.61 [11.00, 22.86] | 10.64 [5.64, 16.27] | 0.45 [0.35, 0.67] | 0.40 [0.29, 0.65] | 0.20 [0.00, 0.80] | 0.24 [0.16, 0.38] | 0.06 [0.03, 0.13] |
| v2_gru_nopose | 0.3 | 0.42 [0.34, 0.60] | 8.91 [5.77, 12.67] | 0.47 [0.35, 0.58] | 0.35 [0.23, 0.46] | 16.70 [10.92, 23.33] | 10.30 [5.02, 16.29] | 0.42 [0.33, 0.65] | 0.34 [0.26, 0.57] | 0.40 [0.10, 0.80] | 0.24 [0.17, 0.38] | 0.06 [0.03, 0.13] |

## Operating point `recall_50` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | 0.75 | 0.65 [0.56, 0.74] | 99.77 [84.65, 114.97] | 0.16 [0.08, 0.25] | 0.08 [0.03, 0.14] | 118.55 [101.99, 133.95] | 47.33 [38.37, 56.36] | 0.67 [0.50, 0.78] | 0.64 [0.50, 0.78] | 1.30 [0.60, 2.10] | 0.48 [0.37, 0.63] | 0.28 [0.12, 0.41] |
| v1_k0_announcer | 0.5 | 0.59 [0.53, 0.70] | 36.00 [32.15, 40.34] | 0.16 [0.08, 0.25] | 0.08 [0.03, 0.14] | 42.83 [37.77, 48.34] | 33.40 [28.32, 38.49] | 0.58 [0.49, 0.71] | 0.57 [0.46, 0.71] | 1.00 [0.40, 1.90] | 0.41 [0.33, 0.55] | 0.23 [0.11, 0.33] |
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
| nearest_in_corridor | 0.99 | 0.95 | 0.92 | 0.87 | 0.78 |
| v2_physics | n/a |
| v2_physics_nopose | n/a |
| v2_kalman | 0.95 | 0.87 | 0.84 | 0.76 | 0.66 |
| v2_kalman_nopose | n/a |
| v2_gru | 0.95 | 0.84 | 0.81 | 0.77 | 0.69 |
| v2_gru_nopose | n/a |

## Kaplan-Meier: share of arrivals not yet warned, by time since first sighting (test, `early_budget_5`)

| Arm | 0.5 s | 1 s | 2 s | 3 s | 4 s |
|---|---:|---:|---:|---:|---:|
| v1_k0 | n/a |
| v1_k0_announcer | n/a |
| nearest_in_corridor | 0.99 | 0.95 | 0.92 | 0.87 | 0.78 |
| v2_physics | n/a |
| v2_physics_nopose | n/a |
| v2_kalman | 0.95 | 0.87 | 0.84 | 0.76 | 0.66 |
| v2_kalman_nopose | n/a |
| v2_gru | 0.95 | 0.84 | 0.81 | 0.77 | 0.69 |
| v2_gru_nopose | n/a |


## Operating curves (test; every knob value)

Recall (warned) against false alerts per minute; the CSV next to this report has every column.

| Arm | knob | warned | warned ≥ 0.2 s before | false/min | precision | alerts/min |
|---|---:|---:|---:|---:|---:|---:|
| v1_k0 | 0.05 | 0.84 | 0.63 | 150.6 | 0.15 | 177.6 |
| v1_k0 | 0.1 | 0.82 | 0.62 | 142.9 | 0.16 | 169.2 |
| v1_k0 | 0.2 | 0.82 | 0.62 | 131.7 | 0.16 | 157.0 |
| v1_k0 | 0.35 | 0.75 | 0.52 | 122.4 | 0.16 | 145.9 |
| v1_k0 | 0.5 | 0.71 | 0.52 | 113.7 | 0.16 | 134.7 |
| v1_k0 | 0.75 | 0.65 | 0.48 | 99.8 | 0.16 | 118.5 |
| v1_k0 | 1 | 0.61 | 0.47 | 87.3 | 0.17 | 104.6 |
| v1_k0 | 1.5 | 0.54 | 0.39 | 73.1 | 0.17 | 88.0 |
| v1_k0 | 2 | 0.52 | 0.34 | 62.9 | 0.17 | 75.6 |
| v1_k0 | 3 | 0.46 | 0.27 | 48.2 | 0.19 | 59.4 |
| v1_k0 | 5 | 0.46 | 0.25 | 33.9 | 0.24 | 44.8 |
| v1_k0 | 8 | 0.46 | 0.25 | 28.0 | 0.28 | 38.9 |
| v1_k0 | 12 | 0.43 | 0.23 | 24.3 | 0.30 | 35.0 |
| v1_k0_announcer | 0.05 | 0.65 | 0.46 | 43.0 | 0.15 | 50.4 |
| v1_k0_announcer | 0.1 | 0.65 | 0.46 | 41.6 | 0.15 | 49.1 |
| v1_k0_announcer | 0.2 | 0.63 | 0.44 | 39.3 | 0.16 | 46.6 |
| v1_k0_announcer | 0.35 | 0.61 | 0.39 | 37.9 | 0.16 | 45.2 |
| v1_k0_announcer | 0.5 | 0.59 | 0.41 | 36.0 | 0.16 | 42.8 |
| v1_k0_announcer | 0.75 | 0.57 | 0.39 | 33.3 | 0.16 | 39.7 |
| v1_k0_announcer | 1 | 0.54 | 0.38 | 31.2 | 0.16 | 37.3 |
| v1_k0_announcer | 1.5 | 0.52 | 0.34 | 27.5 | 0.18 | 33.5 |
| v1_k0_announcer | 2 | 0.48 | 0.29 | 24.5 | 0.18 | 30.0 |
| v1_k0_announcer | 3 | 0.46 | 0.27 | 20.8 | 0.20 | 25.9 |
| v1_k0_announcer | 5 | 0.46 | 0.25 | 16.8 | 0.22 | 21.6 |
| v1_k0_announcer | 8 | 0.46 | 0.25 | 14.3 | 0.25 | 19.1 |
| v1_k0_announcer | 12 | 0.43 | 0.23 | 13.2 | 0.26 | 17.9 |
| nearest_in_corridor | 0.01 | 0.39 | 0.25 | 20.2 | 0.26 | 27.2 |
| nearest_in_corridor | 0.03 | 0.39 | 0.25 | 18.1 | 0.28 | 25.2 |
| nearest_in_corridor | 0.05 | 0.39 | 0.25 | 17.4 | 0.29 | 24.5 |
| nearest_in_corridor | 0.1 | 0.39 | 0.25 | 15.6 | 0.31 | 22.6 |
| nearest_in_corridor | 0.15 | 0.39 | 0.23 | 13.4 | 0.34 | 20.3 |
| nearest_in_corridor | 0.2 | 0.39 | 0.22 | 12.0 | 0.37 | 19.0 |
| nearest_in_corridor | 0.3 | 0.39 | 0.22 | 10.0 | 0.41 | 17.0 |
| nearest_in_corridor | 0.4 | 0.34 | 0.15 | 8.4 | 0.44 | 14.9 |
| nearest_in_corridor | 0.5 | 0.24 | 0.11 | 7.5 | 0.40 | 12.5 |
| nearest_in_corridor | 0.6 | 0.23 | 0.10 | 6.8 | 0.40 | 11.4 |
| nearest_in_corridor | 0.7 | 0.23 | 0.10 | 6.5 | 0.38 | 10.5 |
| nearest_in_corridor | 0.8 | 0.23 | 0.10 | 5.8 | 0.36 | 9.1 |
| nearest_in_corridor | 0.9 | 0.23 | 0.10 | 4.7 | 0.40 | 7.8 |
| nearest_in_corridor | 0.95 | 0.23 | 0.10 | 4.6 | 0.40 | 7.7 |
| v2_physics | 0.01 | 0.38 | 0.25 | 18.9 | 0.26 | 25.4 |
| v2_physics | 0.03 | 0.38 | 0.25 | 18.9 | 0.26 | 25.4 |
| v2_physics | 0.05 | 0.38 | 0.25 | 18.9 | 0.26 | 25.4 |
| v2_physics | 0.1 | 0.38 | 0.25 | 18.9 | 0.26 | 25.4 |
| v2_physics | 0.15 | 0.38 | 0.25 | 18.9 | 0.26 | 25.4 |
| v2_physics | 0.2 | 0.38 | 0.25 | 18.9 | 0.26 | 25.4 |
| v2_physics | 0.3 | 0.38 | 0.25 | 18.4 | 0.27 | 25.1 |
| v2_physics | 0.4 | 0.38 | 0.25 | 17.9 | 0.27 | 24.6 |
| v2_physics | 0.5 | 0.38 | 0.25 | 17.4 | 0.28 | 24.1 |
| v2_physics | 0.6 | 0.38 | 0.25 | 16.6 | 0.29 | 23.3 |
| v2_physics | 0.7 | 0.38 | 0.25 | 16.0 | 0.29 | 22.7 |
| v2_physics | 0.8 | 0.38 | 0.25 | 15.4 | 0.30 | 22.2 |
| v2_physics | 0.9 | 0.38 | 0.25 | 14.5 | 0.32 | 21.2 |
| v2_physics | 0.95 | 0.37 | 0.24 | 13.7 | 0.33 | 20.4 |
| v2_physics_nopose | 0.01 | 0.37 | 0.22 | 20.9 | 0.25 | 27.9 |
| v2_physics_nopose | 0.03 | 0.37 | 0.22 | 20.9 | 0.25 | 27.9 |
| v2_physics_nopose | 0.05 | 0.37 | 0.22 | 20.9 | 0.25 | 27.9 |
| v2_physics_nopose | 0.1 | 0.37 | 0.22 | 20.9 | 0.25 | 27.9 |
| v2_physics_nopose | 0.15 | 0.37 | 0.22 | 20.9 | 0.25 | 27.9 |
| v2_physics_nopose | 0.2 | 0.37 | 0.22 | 20.9 | 0.25 | 27.9 |
| v2_physics_nopose | 0.3 | 0.37 | 0.22 | 20.7 | 0.25 | 27.6 |
| v2_physics_nopose | 0.4 | 0.37 | 0.22 | 20.1 | 0.26 | 27.1 |
| v2_physics_nopose | 0.5 | 0.37 | 0.22 | 19.8 | 0.26 | 26.7 |
| v2_physics_nopose | 0.6 | 0.37 | 0.22 | 19.1 | 0.27 | 26.0 |
| v2_physics_nopose | 0.7 | 0.37 | 0.22 | 17.9 | 0.28 | 24.8 |
| v2_physics_nopose | 0.8 | 0.37 | 0.22 | 17.5 | 0.28 | 24.3 |
| v2_physics_nopose | 0.9 | 0.35 | 0.22 | 16.1 | 0.30 | 22.9 |
| v2_physics_nopose | 0.95 | 0.34 | 0.20 | 15.3 | 0.30 | 22.0 |
| v2_kalman | 0.01 | 0.46 | 0.38 | 46.6 | 0.15 | 54.9 |
| v2_kalman | 0.03 | 0.46 | 0.38 | 42.2 | 0.16 | 50.4 |
| v2_kalman | 0.05 | 0.46 | 0.38 | 36.8 | 0.18 | 45.1 |
| v2_kalman | 0.1 | 0.43 | 0.34 | 31.8 | 0.20 | 39.6 |
| v2_kalman | 0.15 | 0.44 | 0.34 | 27.3 | 0.22 | 35.2 |
| v2_kalman | 0.2 | 0.44 | 0.33 | 24.6 | 0.24 | 32.4 |
| v2_kalman | 0.3 | 0.41 | 0.29 | 18.5 | 0.30 | 26.5 |
| v2_kalman | 0.4 | 0.41 | 0.29 | 15.6 | 0.34 | 23.4 |
| v2_kalman | 0.5 | 0.41 | 0.29 | 13.6 | 0.36 | 21.3 |
| v2_kalman | 0.6 | 0.41 | 0.27 | 12.3 | 0.37 | 19.6 |
| v2_kalman | 0.7 | 0.38 | 0.22 | 10.3 | 0.41 | 17.6 |
| v2_kalman | 0.8 | 0.37 | 0.22 | 8.5 | 0.46 | 15.7 |
| v2_kalman | 0.9 | 0.35 | 0.19 | 7.6 | 0.47 | 14.5 |
| v2_kalman | 0.95 | 0.35 | 0.16 | 7.2 | 0.48 | 13.8 |
| v2_kalman_nopose | 0.01 | 0.42 | 0.33 | 44.6 | 0.15 | 52.5 |
| v2_kalman_nopose | 0.03 | 0.42 | 0.33 | 39.4 | 0.17 | 47.3 |
| v2_kalman_nopose | 0.05 | 0.39 | 0.30 | 34.6 | 0.19 | 42.6 |
| v2_kalman_nopose | 0.1 | 0.38 | 0.28 | 30.1 | 0.21 | 37.9 |
| v2_kalman_nopose | 0.15 | 0.42 | 0.28 | 27.3 | 0.22 | 35.1 |
| v2_kalman_nopose | 0.2 | 0.42 | 0.28 | 24.8 | 0.24 | 32.9 |
| v2_kalman_nopose | 0.3 | 0.39 | 0.25 | 21.0 | 0.27 | 28.9 |
| v2_kalman_nopose | 0.4 | 0.39 | 0.23 | 16.8 | 0.32 | 24.6 |
| v2_kalman_nopose | 0.5 | 0.41 | 0.25 | 14.9 | 0.35 | 22.8 |
| v2_kalman_nopose | 0.6 | 0.39 | 0.24 | 13.4 | 0.36 | 20.9 |
| v2_kalman_nopose | 0.7 | 0.35 | 0.22 | 11.2 | 0.40 | 18.8 |
| v2_kalman_nopose | 0.8 | 0.32 | 0.19 | 9.8 | 0.43 | 17.1 |
| v2_kalman_nopose | 0.9 | 0.30 | 0.16 | 8.2 | 0.46 | 15.1 |
| v2_kalman_nopose | 0.95 | 0.33 | 0.19 | 7.4 | 0.48 | 14.3 |
| v2_gru | 0.01 | 0.49 | 0.37 | 40.5 | 0.18 | 49.5 |
| v2_gru | 0.03 | 0.48 | 0.35 | 28.7 | 0.24 | 37.6 |
| v2_gru | 0.05 | 0.46 | 0.34 | 23.7 | 0.28 | 32.8 |
| v2_gru | 0.1 | 0.46 | 0.33 | 17.4 | 0.34 | 26.2 |
| v2_gru | 0.15 | 0.46 | 0.29 | 14.4 | 0.37 | 22.9 |
| v2_gru | 0.2 | 0.47 | 0.30 | 11.2 | 0.43 | 19.6 |
| v2_gru | 0.3 | 0.44 | 0.24 | 8.5 | 0.49 | 16.6 |
| v2_gru | 0.4 | 0.42 | 0.18 | 6.8 | 0.54 | 14.9 |
| v2_gru | 0.5 | 0.34 | 0.16 | 6.1 | 0.54 | 13.2 |
| v2_gru | 0.6 | 0.30 | 0.13 | 5.4 | 0.56 | 12.1 |
| v2_gru | 0.7 | 0.27 | 0.10 | 5.1 | 0.55 | 11.3 |
| v2_gru | 0.8 | 0.24 | 0.10 | 4.8 | 0.54 | 10.6 |
| v2_gru | 0.9 | 0.23 | 0.10 | 4.5 | 0.54 | 9.9 |
| v2_gru | 0.95 | 0.23 | 0.10 | 4.4 | 0.51 | 9.1 |
| v2_gru_nopose | 0.01 | 0.48 | 0.38 | 40.4 | 0.18 | 49.3 |
| v2_gru_nopose | 0.03 | 0.46 | 0.35 | 28.9 | 0.23 | 37.6 |
| v2_gru_nopose | 0.05 | 0.44 | 0.35 | 23.6 | 0.27 | 32.4 |
| v2_gru_nopose | 0.1 | 0.43 | 0.33 | 17.5 | 0.33 | 26.0 |
| v2_gru_nopose | 0.15 | 0.44 | 0.30 | 14.5 | 0.37 | 22.9 |
| v2_gru_nopose | 0.2 | 0.46 | 0.32 | 11.4 | 0.42 | 19.7 |
| v2_gru_nopose | 0.3 | 0.42 | 0.24 | 8.9 | 0.47 | 16.7 |
| v2_gru_nopose | 0.4 | 0.38 | 0.18 | 7.4 | 0.51 | 15.0 |
| v2_gru_nopose | 0.5 | 0.30 | 0.16 | 6.6 | 0.51 | 13.4 |
| v2_gru_nopose | 0.6 | 0.27 | 0.13 | 5.7 | 0.53 | 12.2 |
| v2_gru_nopose | 0.7 | 0.23 | 0.10 | 5.6 | 0.51 | 11.5 |
| v2_gru_nopose | 0.8 | 0.20 | 0.10 | 5.5 | 0.50 | 10.8 |
| v2_gru_nopose | 0.9 | 0.19 | 0.10 | 5.0 | 0.49 | 9.8 |
| v2_gru_nopose | 0.95 | 0.19 | 0.10 | 5.0 | 0.45 | 9.2 |

Classes: true = the object enters the zone later within the horizon; late = it is already in the zone; false = it doesn't enter within the horizon. precision = (true + late) / all. warned = an announcement about the arriving track in the 4 s up to and including arrival.
Open-loop: the recorded walker never heard these alerts. Labels and inputs share SANPO's depth.
