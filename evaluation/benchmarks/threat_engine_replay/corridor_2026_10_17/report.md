# Threat-engine replay - corridor_2026_10_17

Label: in the walking corridor (width 0.75 m + class half-width, 0 < forward <= 3 m along the direction of travel from fixed_camera_poses.csv) within 20 steps (4 s); tracked objects only.

SANPO official test split: 71 sessions, 56 arrivals (a track first seen before it entered the zone). Operating points chosen on 41 validation sessions, then scored once on test. GRU: `models/GRU/GRU/runs/corridor_h8_seed0`. Generated 2026-10-17 by `python tools/replay_threat_engine.py` (437 s).

## Opportunity window (first sighting -> arrival), test

Median 2.7 s; 25% of arrivals were first seen less than 1 s before arriving, 36% less than 2 s, 29% at least 4 s before. Lead time can't exceed the window, so lead is reported conditionally below.

## Operating point `budget_2` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) |
|---|---:|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||
| nearest_in_corridor | — | no setting meets the rule on validation |||||||||
| v2_physics | — | no setting meets the rule on validation |||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||
| v2_gru | — | no setting meets the rule on validation |||||||||
| v2_gru_nopose | — | no setting meets the rule on validation |||||||||

## Operating point `budget_5` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) |
|---|---:|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||
| nearest_in_corridor | 0.8 | 0.25 [0.16, 0.43] | 5.66 [3.04, 8.34] | 0.39 [0.23, 0.50] | 0.25 [0.16, 0.33] | 9.24 [4.89, 14.09] | 5.79 [1.86, 10.32] | 0.29 [0.18, 0.47] | 0.25 [0.15, 0.45] | 0.20 [0.00, 2.40] |
| v2_physics | — | no setting meets the rule on validation |||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||
| v2_gru | 0.8 | 0.32 [0.22, 0.55] | 4.55 [2.24, 7.23] | 0.54 [0.35, 0.68] | 0.37 [0.23, 0.49] | 9.79 [4.80, 15.58] | 6.76 [2.17, 12.20] | 0.36 [0.22, 0.65] | 0.33 [0.20, 0.64] | 0.20 [0.00, 1.40] |
| v2_gru_nopose | 0.95 | 0.25 [0.13, 0.47] | 5.93 [2.99, 9.38] | 0.37 [0.23, 0.50] | 0.26 [0.17, 0.35] | 9.38 [4.76, 14.68] | 6.07 [1.79, 11.17] | 0.31 [0.14, 0.63] | 0.33 [0.14, 0.64] | 0.40 [0.00, 2.60] |

## Operating point `budget_10` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) |
|---|---:|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||
| nearest_in_corridor | 0.4 | 0.41 [0.29, 0.72] | 8.00 [4.87, 11.33] | 0.46 [0.34, 0.54] | 0.36 [0.25, 0.45] | 14.90 [8.60, 21.83] | 9.79 [4.12, 16.15] | 0.45 [0.33, 0.78] | 0.42 [0.28, 0.73] | 0.00 [0.00, 0.40] |
| v2_physics | — | no setting meets the rule on validation |||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||
| v2_gru | 0.4 | 0.41 [0.28, 0.76] | 6.76 [3.27, 10.53] | 0.54 [0.41, 0.68] | 0.40 [0.30, 0.51] | 14.76 [7.90, 22.05] | 10.48 [4.11, 17.18] | 0.45 [0.30, 0.84] | 0.42 [0.26, 0.81] | 0.20 [0.00, 0.80] |
| v2_gru_nopose | 0.4 | 0.41 [0.25, 0.70] | 7.72 [4.08, 11.85] | 0.50 [0.37, 0.62] | 0.39 [0.31, 0.47] | 15.59 [8.23, 23.66] | 10.90 [4.15, 18.56] | 0.48 [0.29, 0.84] | 0.47 [0.27, 0.81] | 0.10 [0.00, 0.80] |

## Operating point `recall_50` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) |
|---|---:|---|---|---|---|---|---|---|---|---|
| v1_k0 | 0.75 | 0.62 [0.52, 0.78] | 88.69 [73.84, 101.92] | 0.19 [0.08, 0.30] | 0.10 [0.03, 0.17] | 109.24 [94.00, 125.16] | 56.55 [45.16, 68.79] | 0.60 [0.46, 0.80] | 0.58 [0.45, 0.79] | 2.00 [0.80, 2.70] |
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||
| nearest_in_corridor | — | no setting meets the rule on validation |||||||||
| v2_physics | — | no setting meets the rule on validation |||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||
| v2_gru | — | no setting meets the rule on validation |||||||||
| v2_gru_nopose | — | no setting meets the rule on validation |||||||||

## Kaplan-Meier: share of arrivals not yet warned, by time since first sighting (test, `budget_5`)

| Arm | 0.5 s | 1 s | 2 s | 3 s | 4 s |
|---|---:|---:|---:|---:|---:|
| v1_k0 | n/a |
| v1_k0_announcer | n/a |
| nearest_in_corridor | 0.96 | 0.86 | 0.81 | 0.78 | 0.78 |
| v2_physics | n/a |
| v2_physics_nopose | n/a |
| v2_gru | 0.95 | 0.84 | 0.79 | 0.70 | 0.66 |
| v2_gru_nopose | 0.98 | 0.88 | 0.83 | 0.76 | 0.72 |

## Operating curves (test; every knob value)

Recall (warned) against false alerts per minute; the CSV next to this report has every column.

| Arm | knob | warned | false/min | precision | alerts/min |
|---|---:|---:|---:|---:|---:|
| v1_k0 | 0.05 | 0.77 | 128.4 | 0.17 | 155.6 |
| v1_k0 | 0.1 | 0.77 | 120.7 | 0.18 | 146.8 |
| v1_k0 | 0.2 | 0.77 | 111.9 | 0.19 | 137.4 |
| v1_k0 | 0.35 | 0.75 | 103.6 | 0.18 | 127.0 |
| v1_k0 | 0.5 | 0.71 | 97.1 | 0.19 | 119.2 |
| v1_k0 | 0.75 | 0.62 | 88.7 | 0.19 | 109.2 |
| v1_k0 | 1 | 0.61 | 83.0 | 0.19 | 102.3 |
| v1_k0 | 1.5 | 0.57 | 72.8 | 0.19 | 89.5 |
| v1_k0 | 2 | 0.55 | 66.6 | 0.19 | 82.3 |
| v1_k0 | 3 | 0.52 | 56.3 | 0.20 | 69.9 |
| v1_k0 | 5 | 0.52 | 45.4 | 0.22 | 58.2 |
| v1_k0 | 8 | 0.50 | 40.1 | 0.23 | 52.1 |
| v1_k0 | 12 | 0.48 | 36.6 | 0.23 | 47.4 |
| v1_k0_announcer | 0.05 | 0.57 | 47.9 | 0.14 | 55.9 |
| v1_k0_announcer | 0.1 | 0.57 | 46.6 | 0.14 | 54.2 |
| v1_k0_announcer | 0.2 | 0.57 | 45.5 | 0.14 | 53.0 |
| v1_k0_announcer | 0.35 | 0.57 | 43.6 | 0.14 | 50.8 |
| v1_k0_announcer | 0.5 | 0.57 | 42.2 | 0.14 | 49.1 |
| v1_k0_announcer | 0.75 | 0.55 | 40.0 | 0.15 | 47.0 |
| v1_k0_announcer | 1 | 0.55 | 38.6 | 0.15 | 45.7 |
| v1_k0_announcer | 1.5 | 0.55 | 35.6 | 0.17 | 42.6 |
| v1_k0_announcer | 2 | 0.54 | 34.1 | 0.17 | 40.8 |
| v1_k0_announcer | 3 | 0.52 | 30.6 | 0.18 | 37.4 |
| v1_k0_announcer | 5 | 0.52 | 27.2 | 0.20 | 33.9 |
| v1_k0_announcer | 8 | 0.50 | 25.7 | 0.20 | 32.1 |
| v1_k0_announcer | 12 | 0.48 | 24.0 | 0.21 | 30.2 |
| nearest_in_corridor | 0.01 | 0.43 | 24.0 | 0.24 | 31.7 |
| nearest_in_corridor | 0.03 | 0.45 | 21.4 | 0.27 | 29.1 |
| nearest_in_corridor | 0.05 | 0.43 | 20.1 | 0.27 | 27.7 |
| nearest_in_corridor | 0.1 | 0.41 | 17.8 | 0.30 | 25.4 |
| nearest_in_corridor | 0.15 | 0.41 | 15.3 | 0.33 | 22.9 |
| nearest_in_corridor | 0.2 | 0.45 | 13.1 | 0.37 | 20.8 |
| nearest_in_corridor | 0.3 | 0.41 | 9.4 | 0.43 | 16.6 |
| nearest_in_corridor | 0.4 | 0.41 | 8.0 | 0.46 | 14.9 |
| nearest_in_corridor | 0.5 | 0.36 | 7.0 | 0.44 | 12.6 |
| nearest_in_corridor | 0.6 | 0.30 | 6.1 | 0.45 | 11.0 |
| nearest_in_corridor | 0.7 | 0.25 | 5.5 | 0.44 | 9.8 |
| nearest_in_corridor | 0.8 | 0.25 | 5.7 | 0.39 | 9.2 |
| nearest_in_corridor | 0.9 | 0.25 | 5.2 | 0.38 | 8.4 |
| nearest_in_corridor | 0.95 | 0.25 | 5.0 | 0.38 | 8.0 |
| v2_physics | 0.01 | 0.43 | 21.9 | 0.26 | 29.8 |
| v2_physics | 0.03 | 0.43 | 21.9 | 0.26 | 29.8 |
| v2_physics | 0.05 | 0.43 | 21.9 | 0.26 | 29.8 |
| v2_physics | 0.1 | 0.43 | 21.9 | 0.26 | 29.8 |
| v2_physics | 0.15 | 0.43 | 21.9 | 0.26 | 29.8 |
| v2_physics | 0.2 | 0.43 | 21.9 | 0.26 | 29.8 |
| v2_physics | 0.3 | 0.45 | 21.4 | 0.27 | 29.4 |
| v2_physics | 0.4 | 0.45 | 20.8 | 0.28 | 28.8 |
| v2_physics | 0.5 | 0.45 | 20.6 | 0.28 | 28.6 |
| v2_physics | 0.6 | 0.45 | 20.0 | 0.29 | 28.1 |
| v2_physics | 0.7 | 0.43 | 19.7 | 0.29 | 27.7 |
| v2_physics | 0.8 | 0.41 | 18.9 | 0.28 | 26.3 |
| v2_physics | 0.9 | 0.38 | 17.7 | 0.29 | 25.0 |
| v2_physics | 0.95 | 0.39 | 16.8 | 0.30 | 24.1 |
| v2_physics_nopose | 0.01 | 0.39 | 21.7 | 0.25 | 28.7 |
| v2_physics_nopose | 0.03 | 0.39 | 21.7 | 0.25 | 28.7 |
| v2_physics_nopose | 0.05 | 0.39 | 21.7 | 0.25 | 28.7 |
| v2_physics_nopose | 0.1 | 0.39 | 21.7 | 0.25 | 28.7 |
| v2_physics_nopose | 0.15 | 0.39 | 21.7 | 0.25 | 28.7 |
| v2_physics_nopose | 0.2 | 0.39 | 21.7 | 0.25 | 28.7 |
| v2_physics_nopose | 0.3 | 0.39 | 21.5 | 0.25 | 28.6 |
| v2_physics_nopose | 0.4 | 0.39 | 21.2 | 0.25 | 28.3 |
| v2_physics_nopose | 0.5 | 0.39 | 20.8 | 0.25 | 27.9 |
| v2_physics_nopose | 0.6 | 0.39 | 20.6 | 0.26 | 27.9 |
| v2_physics_nopose | 0.7 | 0.39 | 20.4 | 0.26 | 27.7 |
| v2_physics_nopose | 0.8 | 0.39 | 19.4 | 0.26 | 26.3 |
| v2_physics_nopose | 0.9 | 0.39 | 19.2 | 0.26 | 26.1 |
| v2_physics_nopose | 0.95 | 0.38 | 17.4 | 0.28 | 24.1 |
| v2_gru | 0.01 | 0.45 | 50.3 | 0.16 | 59.7 |
| v2_gru | 0.03 | 0.43 | 32.4 | 0.22 | 41.4 |
| v2_gru | 0.05 | 0.43 | 23.0 | 0.28 | 32.0 |
| v2_gru | 0.1 | 0.43 | 14.6 | 0.36 | 22.8 |
| v2_gru | 0.15 | 0.46 | 11.3 | 0.44 | 20.1 |
| v2_gru | 0.2 | 0.46 | 10.3 | 0.46 | 19.0 |
| v2_gru | 0.3 | 0.45 | 8.3 | 0.51 | 16.8 |
| v2_gru | 0.4 | 0.41 | 6.8 | 0.54 | 14.8 |
| v2_gru | 0.5 | 0.43 | 5.5 | 0.60 | 13.7 |
| v2_gru | 0.6 | 0.43 | 5.4 | 0.58 | 12.8 |
| v2_gru | 0.7 | 0.36 | 5.0 | 0.56 | 11.3 |
| v2_gru | 0.8 | 0.32 | 4.6 | 0.54 | 9.8 |
| v2_gru | 0.9 | 0.29 | 4.4 | 0.49 | 8.7 |
| v2_gru | 0.95 | 0.29 | 4.4 | 0.48 | 8.4 |
| v2_gru_nopose | 0.01 | 0.46 | 50.9 | 0.15 | 60.1 |
| v2_gru_nopose | 0.03 | 0.45 | 33.1 | 0.21 | 41.9 |
| v2_gru_nopose | 0.05 | 0.45 | 23.7 | 0.28 | 32.8 |
| v2_gru_nopose | 0.1 | 0.39 | 15.3 | 0.35 | 23.4 |
| v2_gru_nopose | 0.15 | 0.45 | 12.1 | 0.41 | 20.7 |
| v2_gru_nopose | 0.2 | 0.45 | 11.4 | 0.42 | 19.9 |
| v2_gru_nopose | 0.3 | 0.45 | 8.8 | 0.49 | 17.2 |
| v2_gru_nopose | 0.4 | 0.41 | 7.7 | 0.50 | 15.6 |
| v2_gru_nopose | 0.5 | 0.43 | 6.5 | 0.55 | 14.5 |
| v2_gru_nopose | 0.6 | 0.41 | 6.3 | 0.54 | 13.7 |
| v2_gru_nopose | 0.7 | 0.30 | 6.1 | 0.49 | 11.9 |
| v2_gru_nopose | 0.8 | 0.29 | 6.1 | 0.44 | 10.8 |
| v2_gru_nopose | 0.9 | 0.25 | 5.9 | 0.39 | 9.7 |
| v2_gru_nopose | 0.95 | 0.25 | 5.9 | 0.37 | 9.4 |

Classes: true = the object enters the zone later within the horizon; late = it is already in the zone; false = it doesn't enter within the horizon. precision = (true + late) / all. warned = an announcement about the arriving track in the 4 s up to and including arrival.
Open-loop: the recorded walker never heard these alerts. Labels and inputs share SANPO's depth.
