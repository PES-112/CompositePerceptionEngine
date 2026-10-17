# Threat-engine replay - corridor_2026_10_14

Label: in the walking corridor (width 0.75 m + class half-width, 0 < forward <= 3 m along the direction of travel from fixed_camera_poses.csv) within 20 steps (4 s); tracked objects only.

SANPO official test split: 71 sessions, 63 arrivals (a track first seen before it entered the zone). Operating points chosen on 41 validation sessions, then scored once on test. GRU: `models/GRU/GRU/runs/corridor_h8_seed0`. Generated 2026-10-14 by `python tools/replay_threat_engine.py` (159 s).

## Opportunity window (first sighting -> arrival), test

Median 2.6 s; 25% of arrivals were first seen less than 1 s before arriving, 38% less than 2 s, 27% at least 4 s before. Lead time can't exceed the window, so lead is reported conditionally below.

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
| nearest_in_corridor | 0.6 | 0.24 [0.16, 0.39] | 7.31 [4.53, 10.10] | 0.40 [0.26, 0.48] | 0.30 [0.18, 0.39] | 12.14 [6.77, 17.85] | 8.00 [2.98, 13.72] | 0.26 [0.15, 0.40] | 0.21 [0.09, 0.38] | 1.30 [0.00, 2.50] |
| v2_physics | — | no setting meets the rule on validation |||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||
| v2_gru | 0.7 | 0.32 [0.23, 0.51] | 5.38 [3.05, 8.01] | 0.56 [0.40, 0.67] | 0.40 [0.28, 0.53] | 12.28 [6.54, 18.72] | 8.55 [3.05, 14.59] | 0.36 [0.23, 0.56] | 0.33 [0.18, 0.57] | 0.40 [0.00, 1.30] |
| v2_gru_nopose | 0.7 | 0.27 [0.15, 0.47] | 6.48 [3.40, 9.99] | 0.49 [0.34, 0.60] | 0.42 [0.30, 0.55] | 12.69 [6.53, 19.77] | 8.83 [2.57, 15.77] | 0.34 [0.19, 0.60] | 0.33 [0.15, 0.62] | 0.50 [0.00, 1.40] |

## Operating point `budget_10` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) |
|---|---:|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||
| nearest_in_corridor | 0.4 | 0.35 [0.24, 0.63] | 9.24 [6.13, 12.36] | 0.43 [0.31, 0.50] | 0.33 [0.20, 0.43] | 16.28 [10.07, 22.97] | 10.90 [5.21, 17.12] | 0.40 [0.29, 0.70] | 0.36 [0.22, 0.67] | 0.20 [0.00, 0.70] |
| v2_physics | — | no setting meets the rule on validation |||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||
| v2_gru | 0.4 | 0.35 [0.23, 0.63] | 7.86 [4.50, 11.34] | 0.50 [0.39, 0.60] | 0.38 [0.26, 0.49] | 15.86 [8.75, 23.66] | 11.31 [4.59, 18.57] | 0.38 [0.25, 0.65] | 0.36 [0.20, 0.67] | 0.60 [0.00, 1.20] |
| v2_gru_nopose | 0.4 | 0.37 [0.23, 0.63] | 8.55 [4.87, 12.38] | 0.49 [0.37, 0.59] | 0.42 [0.33, 0.49] | 16.83 [9.13, 25.32] | 11.86 [4.46, 20.33] | 0.43 [0.27, 0.70] | 0.41 [0.20, 0.73] | 0.00 [0.00, 1.00] |

## Operating point `recall_50` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) |
|---|---:|---|---|---|---|---|---|---|---|---|
| v1_k0 | 1 | 0.62 [0.54, 0.76] | 81.10 [66.45, 94.66] | 0.21 [0.10, 0.32] | 0.11 [0.04, 0.18] | 102.34 [87.16, 117.98] | 53.24 [42.78, 64.40] | 0.62 [0.52, 0.80] | 0.59 [0.48, 0.82] | 1.40 [0.90, 2.60] |
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
| nearest_in_corridor | 0.97 | 0.86 | 0.79 | 0.79 | 0.71 |
| v2_physics | n/a |
| v2_physics_nopose | n/a |
| v2_gru | 0.97 | 0.86 | 0.79 | 0.68 | 0.61 |
| v2_gru_nopose | 0.98 | 0.87 | 0.78 | 0.67 | 0.64 |

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
| nearest_in_corridor | 0.01 | 0.44 | 29.2 | 0.21 | 37.2 |
| nearest_in_corridor | 0.03 | 0.44 | 26.2 | 0.24 | 34.3 |
| nearest_in_corridor | 0.05 | 0.43 | 23.9 | 0.25 | 32.0 |
| nearest_in_corridor | 0.1 | 0.41 | 20.8 | 0.27 | 28.4 |
| nearest_in_corridor | 0.15 | 0.40 | 18.1 | 0.30 | 25.8 |
| nearest_in_corridor | 0.2 | 0.43 | 15.9 | 0.33 | 23.6 |
| nearest_in_corridor | 0.3 | 0.40 | 11.3 | 0.40 | 18.8 |
| nearest_in_corridor | 0.4 | 0.35 | 9.2 | 0.43 | 16.3 |
| nearest_in_corridor | 0.5 | 0.30 | 8.4 | 0.40 | 13.9 |
| nearest_in_corridor | 0.6 | 0.24 | 7.3 | 0.40 | 12.1 |
| nearest_in_corridor | 0.7 | 0.22 | 6.6 | 0.40 | 11.0 |
| nearest_in_corridor | 0.8 | 0.22 | 6.6 | 0.35 | 10.2 |
| nearest_in_corridor | 0.9 | 0.21 | 6.1 | 0.35 | 9.4 |
| nearest_in_corridor | 0.95 | 0.21 | 5.9 | 0.34 | 9.0 |
| v2_physics | 0.01 | 0.38 | 22.9 | 0.26 | 30.8 |
| v2_physics | 0.03 | 0.38 | 22.9 | 0.26 | 30.8 |
| v2_physics | 0.05 | 0.38 | 22.9 | 0.26 | 30.8 |
| v2_physics | 0.1 | 0.38 | 22.9 | 0.26 | 30.8 |
| v2_physics | 0.15 | 0.38 | 22.9 | 0.26 | 30.8 |
| v2_physics | 0.2 | 0.38 | 22.9 | 0.26 | 30.8 |
| v2_physics | 0.3 | 0.38 | 22.9 | 0.26 | 30.8 |
| v2_physics | 0.4 | 0.38 | 22.2 | 0.26 | 30.1 |
| v2_physics | 0.5 | 0.38 | 21.9 | 0.26 | 29.8 |
| v2_physics | 0.6 | 0.38 | 21.7 | 0.27 | 29.5 |
| v2_physics | 0.7 | 0.40 | 21.4 | 0.27 | 29.4 |
| v2_physics | 0.8 | 0.41 | 20.0 | 0.29 | 28.0 |
| v2_physics | 0.9 | 0.40 | 19.0 | 0.30 | 27.0 |
| v2_physics | 0.95 | 0.40 | 17.7 | 0.32 | 25.8 |
| v2_physics_nopose | 0.01 | 0.37 | 22.9 | 0.27 | 31.4 |
| v2_physics_nopose | 0.03 | 0.37 | 22.9 | 0.27 | 31.4 |
| v2_physics_nopose | 0.05 | 0.37 | 22.9 | 0.27 | 31.4 |
| v2_physics_nopose | 0.1 | 0.37 | 22.9 | 0.27 | 31.4 |
| v2_physics_nopose | 0.15 | 0.37 | 22.9 | 0.27 | 31.4 |
| v2_physics_nopose | 0.2 | 0.37 | 22.9 | 0.27 | 31.4 |
| v2_physics_nopose | 0.3 | 0.37 | 22.8 | 0.27 | 31.3 |
| v2_physics_nopose | 0.4 | 0.37 | 22.5 | 0.28 | 31.0 |
| v2_physics_nopose | 0.5 | 0.37 | 22.2 | 0.28 | 30.8 |
| v2_physics_nopose | 0.6 | 0.37 | 22.1 | 0.28 | 30.6 |
| v2_physics_nopose | 0.7 | 0.37 | 21.8 | 0.28 | 30.3 |
| v2_physics_nopose | 0.8 | 0.37 | 20.6 | 0.29 | 28.8 |
| v2_physics_nopose | 0.9 | 0.37 | 20.0 | 0.29 | 28.1 |
| v2_physics_nopose | 0.95 | 0.35 | 18.1 | 0.31 | 26.3 |
| v2_gru | 0.01 | 0.41 | 54.6 | 0.15 | 64.6 |
| v2_gru | 0.03 | 0.41 | 37.4 | 0.21 | 47.2 |
| v2_gru | 0.05 | 0.44 | 28.8 | 0.26 | 38.9 |
| v2_gru | 0.1 | 0.41 | 17.4 | 0.35 | 26.8 |
| v2_gru | 0.15 | 0.44 | 12.4 | 0.44 | 22.3 |
| v2_gru | 0.2 | 0.38 | 10.8 | 0.47 | 20.3 |
| v2_gru | 0.3 | 0.38 | 8.6 | 0.51 | 17.5 |
| v2_gru | 0.4 | 0.35 | 7.9 | 0.50 | 15.9 |
| v2_gru | 0.5 | 0.37 | 6.6 | 0.56 | 15.2 |
| v2_gru | 0.6 | 0.33 | 6.3 | 0.56 | 14.3 |
| v2_gru | 0.7 | 0.32 | 5.4 | 0.56 | 12.3 |
| v2_gru | 0.8 | 0.27 | 5.5 | 0.52 | 11.6 |
| v2_gru | 0.9 | 0.25 | 5.4 | 0.49 | 10.6 |
| v2_gru | 0.95 | 0.25 | 5.1 | 0.45 | 9.2 |
| v2_gru_nopose | 0.01 | 0.40 | 55.3 | 0.15 | 65.1 |
| v2_gru_nopose | 0.03 | 0.41 | 37.8 | 0.21 | 47.6 |
| v2_gru_nopose | 0.05 | 0.44 | 29.7 | 0.25 | 39.7 |
| v2_gru_nopose | 0.1 | 0.41 | 18.9 | 0.33 | 28.3 |
| v2_gru_nopose | 0.15 | 0.43 | 13.2 | 0.43 | 23.0 |
| v2_gru_nopose | 0.2 | 0.41 | 11.9 | 0.44 | 21.2 |
| v2_gru_nopose | 0.3 | 0.41 | 9.8 | 0.48 | 18.8 |
| v2_gru_nopose | 0.4 | 0.37 | 8.6 | 0.49 | 16.8 |
| v2_gru_nopose | 0.5 | 0.37 | 7.6 | 0.52 | 15.9 |
| v2_gru_nopose | 0.6 | 0.33 | 7.3 | 0.51 | 15.0 |
| v2_gru_nopose | 0.7 | 0.27 | 6.5 | 0.49 | 12.7 |
| v2_gru_nopose | 0.8 | 0.25 | 6.3 | 0.48 | 12.1 |
| v2_gru_nopose | 0.9 | 0.22 | 6.2 | 0.44 | 11.2 |
| v2_gru_nopose | 0.95 | 0.22 | 6.1 | 0.40 | 10.1 |

Classes: true = the object enters the zone later within the horizon; late = it is already in the zone; false = it doesn't enter within the horizon. precision = (true + late) / all. warned = an announcement about the arriving track in the 4 s up to and including arrival.
Open-loop: the recorded walker never heard these alerts. Labels and inputs share SANPO's depth.
