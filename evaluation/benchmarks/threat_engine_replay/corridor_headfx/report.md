# Threat-engine replay - corridor_headfx

Label: in the walking corridor (width 0.75 m + class half-width, 0 < forward <= 3 m along the direction of travel from fixed_camera_poses.csv) within 20 steps (4 s); tracked objects only.

SANPO official test split: 85 sessions, 79 arrivals (a track first seen before it entered the zone). Operating points chosen on 48 validation sessions, then scored once on test. GRU: `models/GRU/GRU/runs/corridor_h8_seed0`. Generated 2026-10-20 by `python tools/replay_threat_engine.py` (2547 s).

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
| nearest_in_corridor | 0.95 / cd 2 / sm 1 / st 0.6 | 0.23 [0.11, 0.35] | 4.93 [3.09, 7.01] | 0.39 [0.25, 0.52] | 0.29 [0.16, 0.42] | 8.13 [5.03, 11.82] | 5.28 [2.23, 8.84] | 0.25 [0.12, 0.37] | 0.28 [0.14, 0.40] | 0.20 [0.00, 0.40] | 0.10 [0.02, 0.19] | 0.03 [0.00, 0.04] |
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | — | no setting meets the rule on validation |||||||||||
| v2_kalman_nopose | — | no setting meets the rule on validation |||||||||||
| v2_gru | 0.6 / cd 8 / sm 1 / st 0.6 | 0.35 [0.23, 0.52] | 5.62 [3.44, 8.09] | 0.55 [0.40, 0.69] | 0.45 [0.29, 0.60] | 12.63 [7.73, 18.35] | 8.31 [3.72, 13.65] | 0.37 [0.22, 0.58] | 0.36 [0.23, 0.60] | 0.00 [0.00, 0.30] | 0.14 [0.06, 0.25] | 0.04 [0.00, 0.07] |
| v2_gru_nopose | — | no setting meets the rule on validation |||||||||||

## Operating point `budget_10` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||||
| nearest_in_corridor | 0.3 / cd 8 / sm 1 / st 0.6 | 0.38 [0.25, 0.56] | 10.12 [6.94, 13.84] | 0.40 [0.30, 0.51] | 0.32 [0.22, 0.43] | 16.96 [11.42, 23.60] | 10.99 [5.95, 16.66] | 0.38 [0.23, 0.62] | 0.40 [0.25, 0.67] | 0.20 [0.00, 0.40] | 0.19 [0.11, 0.30] | 0.06 [0.03, 0.14] |
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | 0.8 / cd 4 / sm 1.25 / st 0.6 | 0.38 [0.25, 0.57] | 9.78 [6.67, 13.20] | 0.43 [0.31, 0.53] | 0.31 [0.20, 0.42] | 17.05 [11.58, 23.19] | 11.25 [6.17, 16.88] | 0.40 [0.27, 0.63] | 0.43 [0.27, 0.77] | 0.40 [0.20, 0.80] | 0.23 [0.14, 0.40] | 0.09 [0.04, 0.19] |
| v2_kalman_nopose | 0.9 | 0.34 [0.22, 0.55] | 9.35 [6.03, 12.94] | 0.43 [0.33, 0.54] | 0.33 [0.21, 0.45] | 16.53 [10.99, 22.72] | 11.34 [6.08, 17.19] | 0.35 [0.24, 0.60] | 0.34 [0.22, 0.64] | 0.40 [0.00, 1.00] | 0.19 [0.10, 0.36] | 0.09 [0.04, 0.20] |
| v2_gru | 0.4 / cd 8 / sm 1 / st 0.6 | 0.46 [0.36, 0.64] | 7.18 [4.42, 10.17] | 0.53 [0.39, 0.65] | 0.39 [0.26, 0.53] | 15.14 [9.75, 21.45] | 10.30 [5.22, 16.05] | 0.47 [0.36, 0.72] | 0.43 [0.30, 0.71] | 0.30 [0.00, 0.60] | 0.23 [0.14, 0.37] | 0.06 [0.03, 0.13] |
| v2_gru_nopose | 0.3 | 0.43 [0.34, 0.62] | 9.35 [6.24, 12.95] | 0.47 [0.35, 0.58] | 0.34 [0.22, 0.44] | 17.48 [11.90, 23.86] | 10.47 [5.29, 16.73] | 0.43 [0.34, 0.68] | 0.36 [0.27, 0.64] | 0.40 [0.10, 0.90] | 0.25 [0.17, 0.42] | 0.09 [0.04, 0.21] |

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
| nearest_in_corridor | 0.95 / cd 2 / sm 1 / st 0.6 | 0.23 [0.11, 0.35] | 4.93 [3.09, 7.01] | 0.39 [0.25, 0.52] | 0.29 [0.16, 0.42] | 8.13 [5.03, 11.82] | 5.28 [2.23, 8.84] | 0.25 [0.12, 0.37] | 0.28 [0.14, 0.40] | 0.20 [0.00, 0.40] | 0.10 [0.02, 0.19] | 0.03 [0.00, 0.04] |
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | — | no setting meets the rule on validation |||||||||||
| v2_kalman_nopose | — | no setting meets the rule on validation |||||||||||
| v2_gru | 0.6 / cd 8 / sm 1 / st 0.6 | 0.35 [0.23, 0.52] | 5.62 [3.44, 8.09] | 0.55 [0.40, 0.69] | 0.45 [0.29, 0.60] | 12.63 [7.73, 18.35] | 8.31 [3.72, 13.65] | 0.37 [0.22, 0.58] | 0.36 [0.23, 0.60] | 0.00 [0.00, 0.30] | 0.14 [0.06, 0.25] | 0.04 [0.00, 0.07] |
| v2_gru_nopose | — | no setting meets the rule on validation |||||||||||

## Operating point `early_budget_10` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||||
| nearest_in_corridor | 0.3 / cd 8 / sm 1 / st 0.6 | 0.38 [0.25, 0.56] | 10.12 [6.94, 13.84] | 0.40 [0.30, 0.51] | 0.32 [0.22, 0.43] | 16.96 [11.42, 23.60] | 10.99 [5.95, 16.66] | 0.38 [0.23, 0.62] | 0.40 [0.25, 0.67] | 0.20 [0.00, 0.40] | 0.19 [0.11, 0.30] | 0.06 [0.03, 0.14] |
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | 0.8 / cd 8 / sm 1.25 / st 0.6 | 0.38 [0.25, 0.57] | 9.43 [6.47, 12.58] | 0.42 [0.31, 0.54] | 0.32 [0.20, 0.43] | 16.35 [11.18, 22.21] | 10.99 [6.08, 16.42] | 0.40 [0.27, 0.63] | 0.43 [0.27, 0.77] | 0.40 [0.20, 0.80] | 0.23 [0.14, 0.40] | 0.09 [0.04, 0.19] |
| v2_kalman_nopose | 0.95 | 0.35 [0.25, 0.54] | 8.39 [5.38, 11.78] | 0.46 [0.35, 0.57] | 0.35 [0.22, 0.47] | 15.49 [10.12, 21.49] | 10.64 [5.60, 16.49] | 0.37 [0.28, 0.57] | 0.32 [0.20, 0.58] | 0.40 [0.10, 0.80] | 0.20 [0.12, 0.34] | 0.08 [0.03, 0.18] |
| v2_gru | 0.3 / cd 8 / sm 1 / st 0.6 | 0.46 [0.36, 0.64] | 8.65 [5.79, 11.89] | 0.49 [0.36, 0.62] | 0.36 [0.24, 0.49] | 16.87 [11.31, 23.27] | 10.73 [5.63, 16.64] | 0.47 [0.36, 0.72] | 0.43 [0.30, 0.71] | 0.20 [0.00, 0.80] | 0.24 [0.16, 0.40] | 0.09 [0.04, 0.21] |
| v2_gru_nopose | 0.3 | 0.43 [0.34, 0.62] | 9.35 [6.24, 12.95] | 0.47 [0.35, 0.58] | 0.34 [0.22, 0.44] | 17.48 [11.90, 23.86] | 10.47 [5.29, 16.73] | 0.43 [0.34, 0.68] | 0.36 [0.27, 0.64] | 0.40 [0.10, 0.90] | 0.25 [0.17, 0.42] | 0.09 [0.04, 0.21] |

## Operating point `recall_50` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v1_k0 | 0.75 | 0.65 [0.56, 0.74] | 99.77 [84.65, 114.97] | 0.16 [0.08, 0.25] | 0.08 [0.03, 0.14] | 118.55 [101.99, 133.95] | 47.33 [38.37, 56.36] | 0.67 [0.50, 0.78] | 0.64 [0.50, 0.78] | 1.30 [0.60, 2.10] | 0.48 [0.37, 0.63] | 0.28 [0.12, 0.41] |
| v1_k0_announcer | 0.5 | 0.59 [0.53, 0.70] | 36.00 [32.15, 40.34] | 0.16 [0.08, 0.25] | 0.08 [0.03, 0.14] | 42.83 [37.77, 48.34] | 33.40 [28.32, 38.49] | 0.58 [0.49, 0.71] | 0.57 [0.46, 0.71] | 1.00 [0.40, 1.90] | 0.41 [0.33, 0.55] | 0.23 [0.11, 0.33] |
| nearest_in_corridor | — | no setting meets the rule on validation |||||||||||
| v2_physics | — | no setting meets the rule on validation |||||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||||
| v2_kalman | — | no setting meets the rule on validation |||||||||||
| v2_kalman_nopose | 0.01 | 0.42 [0.26, 0.70] | 46.12 [41.50, 51.32] | 0.14 [0.08, 0.22] | 0.09 [0.04, 0.15] | 53.91 [49.26, 58.10] | 35.91 [31.11, 40.31] | 0.43 [0.28, 0.81] | 0.43 [0.25, 0.88] | 1.20 [0.60, 1.90] | 0.33 [0.20, 0.60] | 0.16 [0.09, 0.32] |
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
| v2_kalman | n/a |
| v2_kalman_nopose | n/a |
| v2_gru | 0.95 | 0.86 | 0.82 | 0.74 | 0.64 |
| v2_gru_nopose | n/a |

## Kaplan-Meier: share of arrivals not yet warned, by time since first sighting (test, `early_budget_5`)

| Arm | 0.5 s | 1 s | 2 s | 3 s | 4 s |
|---|---:|---:|---:|---:|---:|
| v1_k0 | n/a |
| v1_k0_announcer | n/a |
| nearest_in_corridor | 0.99 | 0.95 | 0.92 | 0.87 | 0.78 |
| v2_physics | n/a |
| v2_physics_nopose | n/a |
| v2_kalman | n/a |
| v2_kalman_nopose | n/a |
| v2_gru | 0.95 | 0.86 | 0.82 | 0.74 | 0.64 |
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
| v2_physics | 0.01 | 0.39 | 0.25 | 18.8 | 0.26 | 25.4 |
| v2_physics | 0.03 | 0.39 | 0.25 | 18.8 | 0.26 | 25.4 |
| v2_physics | 0.05 | 0.39 | 0.25 | 18.8 | 0.26 | 25.4 |
| v2_physics | 0.1 | 0.39 | 0.25 | 18.8 | 0.26 | 25.4 |
| v2_physics | 0.15 | 0.39 | 0.25 | 18.8 | 0.26 | 25.4 |
| v2_physics | 0.2 | 0.39 | 0.25 | 18.8 | 0.26 | 25.4 |
| v2_physics | 0.3 | 0.39 | 0.25 | 18.5 | 0.27 | 25.3 |
| v2_physics | 0.4 | 0.39 | 0.25 | 17.9 | 0.27 | 24.7 |
| v2_physics | 0.5 | 0.39 | 0.25 | 17.2 | 0.28 | 24.0 |
| v2_physics | 0.6 | 0.39 | 0.25 | 16.8 | 0.29 | 23.5 |
| v2_physics | 0.7 | 0.39 | 0.25 | 16.4 | 0.29 | 23.1 |
| v2_physics | 0.8 | 0.39 | 0.25 | 15.6 | 0.31 | 22.4 |
| v2_physics | 0.9 | 0.39 | 0.25 | 14.5 | 0.32 | 21.4 |
| v2_physics | 0.95 | 0.38 | 0.24 | 13.8 | 0.33 | 20.6 |
| v2_physics_nopose | 0.01 | 0.38 | 0.22 | 20.8 | 0.25 | 27.9 |
| v2_physics_nopose | 0.03 | 0.38 | 0.22 | 20.8 | 0.25 | 27.9 |
| v2_physics_nopose | 0.05 | 0.38 | 0.22 | 20.8 | 0.25 | 27.9 |
| v2_physics_nopose | 0.1 | 0.38 | 0.22 | 20.8 | 0.25 | 27.9 |
| v2_physics_nopose | 0.15 | 0.38 | 0.22 | 20.8 | 0.25 | 27.9 |
| v2_physics_nopose | 0.2 | 0.38 | 0.22 | 20.8 | 0.25 | 27.9 |
| v2_physics_nopose | 0.3 | 0.38 | 0.22 | 20.5 | 0.26 | 27.6 |
| v2_physics_nopose | 0.4 | 0.38 | 0.22 | 20.0 | 0.26 | 27.1 |
| v2_physics_nopose | 0.5 | 0.38 | 0.22 | 19.7 | 0.26 | 26.8 |
| v2_physics_nopose | 0.6 | 0.38 | 0.22 | 19.5 | 0.27 | 26.6 |
| v2_physics_nopose | 0.7 | 0.38 | 0.22 | 18.2 | 0.28 | 25.3 |
| v2_physics_nopose | 0.8 | 0.38 | 0.22 | 17.8 | 0.28 | 24.8 |
| v2_physics_nopose | 0.9 | 0.35 | 0.22 | 16.7 | 0.29 | 23.5 |
| v2_physics_nopose | 0.95 | 0.35 | 0.22 | 15.7 | 0.30 | 22.3 |
| v2_kalman_nopose | 0.01 | 0.42 | 0.33 | 46.1 | 0.14 | 53.9 |
| v2_kalman_nopose | 0.03 | 0.42 | 0.33 | 40.1 | 0.16 | 47.9 |
| v2_kalman_nopose | 0.05 | 0.39 | 0.30 | 35.4 | 0.18 | 43.3 |
| v2_kalman_nopose | 0.1 | 0.38 | 0.28 | 31.1 | 0.20 | 38.8 |
| v2_kalman_nopose | 0.15 | 0.42 | 0.28 | 27.5 | 0.22 | 35.3 |
| v2_kalman_nopose | 0.2 | 0.42 | 0.29 | 25.3 | 0.24 | 33.4 |
| v2_kalman_nopose | 0.3 | 0.39 | 0.25 | 21.5 | 0.27 | 29.3 |
| v2_kalman_nopose | 0.4 | 0.41 | 0.24 | 17.6 | 0.31 | 25.5 |
| v2_kalman_nopose | 0.5 | 0.42 | 0.27 | 15.6 | 0.34 | 23.6 |
| v2_kalman_nopose | 0.6 | 0.41 | 0.25 | 13.8 | 0.36 | 21.6 |
| v2_kalman_nopose | 0.7 | 0.34 | 0.22 | 11.9 | 0.38 | 19.4 |
| v2_kalman_nopose | 0.8 | 0.34 | 0.20 | 10.6 | 0.41 | 18.0 |
| v2_kalman_nopose | 0.9 | 0.34 | 0.19 | 9.3 | 0.43 | 16.5 |
| v2_kalman_nopose | 0.95 | 0.35 | 0.20 | 8.4 | 0.46 | 15.5 |
| v2_gru_nopose | 0.01 | 0.48 | 0.38 | 41.8 | 0.18 | 50.7 |
| v2_gru_nopose | 0.03 | 0.46 | 0.35 | 29.9 | 0.22 | 38.5 |
| v2_gru_nopose | 0.05 | 0.44 | 0.35 | 23.7 | 0.27 | 32.5 |
| v2_gru_nopose | 0.1 | 0.43 | 0.33 | 18.0 | 0.32 | 26.7 |
| v2_gru_nopose | 0.15 | 0.44 | 0.32 | 14.9 | 0.37 | 23.5 |
| v2_gru_nopose | 0.2 | 0.46 | 0.33 | 12.2 | 0.41 | 20.9 |
| v2_gru_nopose | 0.3 | 0.43 | 0.25 | 9.3 | 0.47 | 17.5 |
| v2_gru_nopose | 0.4 | 0.42 | 0.23 | 8.0 | 0.49 | 15.7 |
| v2_gru_nopose | 0.5 | 0.35 | 0.19 | 7.3 | 0.50 | 14.5 |
| v2_gru_nopose | 0.6 | 0.32 | 0.14 | 6.1 | 0.53 | 13.1 |
| v2_gru_nopose | 0.7 | 0.24 | 0.10 | 5.8 | 0.51 | 11.9 |
| v2_gru_nopose | 0.8 | 0.20 | 0.10 | 5.5 | 0.51 | 11.1 |
| v2_gru_nopose | 0.9 | 0.19 | 0.10 | 5.0 | 0.50 | 10.0 |
| v2_gru_nopose | 0.95 | 0.19 | 0.10 | 5.1 | 0.46 | 9.5 |

Classes: true = the object enters the zone later within the horizon; late = it is already in the zone; false = it doesn't enter within the horizon. precision = (true + late) / all. warned = an announcement about the arriving track in the 4 s up to and including arrival.
Open-loop: the recorded walker never heard these alerts. Labels and inputs share SANPO's depth.
