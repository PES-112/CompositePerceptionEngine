# Threat-engine replay - oracle_context_neutral_boundary_2026_10_17

Label: in the walking corridor (width 0.75 m + class half-width, 0 < forward <= 3 m along the direction of travel from fixed_camera_poses.csv) within 20 steps (4 s); tracked objects only.

SANPO official test split: 10 sessions, 20 arrivals (a track first seen before it entered the zone). Operating points chosen on 5 validation sessions, then scored once on test. GRU: `models/GRU/GRU/runs/corridor_h8_seed0`. Generated 2026-10-17 by `python tools/replay_threat_engine.py` (31 s).

## Opportunity window (first sighting -> arrival), test

Median 2.5 s; 20% of arrivals were first seen less than 1 s before arriving, 30% less than 2 s, 30% at least 4 s before. Lead time can't exceed the window, so lead is reported conditionally below.

## Operating point `budget_2` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) |
|---|---:|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||
| nearest_in_corridor | 0.2 | 0.40 [0.27, 1.00] | 11.44 [1.91, 18.95] | 0.50 [0.20, 0.61] | 0.39 [0.20, 0.50] | 22.89 [3.61, 40.88] | 17.41 [0.72, 33.94] | 0.50 [0.36, 1.00] | 0.50 [0.40, 1.00] | 0.70 [0.00, 2.60] |
| v2_physics | — | no setting meets the rule on validation |||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||
| v2_gru | 0.4 | 0.35 [0.20, 1.00] | 9.45 [1.51, 16.16] | 0.54 [0.25, 0.63] | 0.41 [0.25, 0.60] | 20.40 [3.30, 37.88] | 14.93 [0.36, 30.31] | 0.38 [0.18, 1.00] | 0.36 [0.20, 1.00] | 0.70 [0.00, 2.60] |
| v2_gru_nopose | 0.4 | 0.50 [0.40, 1.00] | 10.45 [1.53, 18.58] | 0.52 [0.25, 0.63] | 0.41 [0.25, 0.60] | 21.89 [3.41, 41.83] | 16.92 [0.45, 35.61] | 0.56 [0.45, 1.00] | 0.57 [0.50, 1.00] | 0.40 [0.00, 1.40] |
| v2_physics_ctx | — | no setting meets the rule on validation |||||||||
| v2_gru_ctx | 0.4 | 0.40 [0.27, 1.00] | 9.95 [1.51, 17.31] | 0.52 [0.12, 0.60] | 0.40 [0.12, 0.50] | 20.90 [2.51, 40.01] | 15.42 [0.36, 31.62] | 0.44 [0.27, 1.00] | 0.43 [0.30, 1.00] | 1.40 [0.00, 3.20] |

## Operating point `budget_5` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) |
|---|---:|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||
| nearest_in_corridor | 0.2 | 0.40 [0.27, 1.00] | 11.44 [1.91, 18.95] | 0.50 [0.20, 0.61] | 0.39 [0.20, 0.50] | 22.89 [3.61, 40.88] | 17.41 [0.72, 33.94] | 0.50 [0.36, 1.00] | 0.50 [0.40, 1.00] | 0.70 [0.00, 2.60] |
| v2_physics | — | no setting meets the rule on validation |||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||
| v2_gru | 0.4 | 0.35 [0.20, 1.00] | 9.45 [1.51, 16.16] | 0.54 [0.25, 0.63] | 0.41 [0.25, 0.60] | 20.40 [3.30, 37.88] | 14.93 [0.36, 30.31] | 0.38 [0.18, 1.00] | 0.36 [0.20, 1.00] | 0.70 [0.00, 2.60] |
| v2_gru_nopose | 0.4 | 0.50 [0.40, 1.00] | 10.45 [1.53, 18.58] | 0.52 [0.25, 0.63] | 0.41 [0.25, 0.60] | 21.89 [3.41, 41.83] | 16.92 [0.45, 35.61] | 0.56 [0.45, 1.00] | 0.57 [0.50, 1.00] | 0.40 [0.00, 1.40] |
| v2_physics_ctx | — | no setting meets the rule on validation |||||||||
| v2_gru_ctx | 0.4 | 0.40 [0.27, 1.00] | 9.95 [1.51, 17.31] | 0.52 [0.12, 0.60] | 0.40 [0.12, 0.50] | 20.90 [2.51, 40.01] | 15.42 [0.36, 31.62] | 0.44 [0.27, 1.00] | 0.43 [0.30, 1.00] | 1.40 [0.00, 3.20] |

## Operating point `budget_10` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) |
|---|---:|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||
| nearest_in_corridor | 0.2 | 0.40 [0.27, 1.00] | 11.44 [1.91, 18.95] | 0.50 [0.20, 0.61] | 0.39 [0.20, 0.50] | 22.89 [3.61, 40.88] | 17.41 [0.72, 33.94] | 0.50 [0.36, 1.00] | 0.50 [0.40, 1.00] | 0.70 [0.00, 2.60] |
| v2_physics | 0.01 | 0.35 [0.27, 1.00] | 18.91 [11.13, 25.35] | 0.34 [0.07, 0.44] | 0.29 [0.07, 0.38] | 28.86 [12.66, 44.54] | 19.90 [6.03, 35.27] | 0.44 [0.36, 1.00] | 0.50 [0.40, 1.00] | 0.80 [0.00, 1.80] |
| v2_physics_nopose | 0.8 | 0.35 [0.27, 1.00] | 18.41 [8.62, 27.60] | 0.35 [0.09, 0.44] | 0.26 [0.09, 0.32] | 28.36 [10.19, 47.82] | 19.90 [5.06, 36.26] | 0.44 [0.36, 1.00] | 0.50 [0.40, 1.00] | 1.40 [0.00, 2.60] |
| v2_gru | 0.4 | 0.35 [0.20, 1.00] | 9.45 [1.51, 16.16] | 0.54 [0.25, 0.63] | 0.41 [0.25, 0.60] | 20.40 [3.30, 37.88] | 14.93 [0.36, 30.31] | 0.38 [0.18, 1.00] | 0.36 [0.20, 1.00] | 0.70 [0.00, 2.60] |
| v2_gru_nopose | 0.4 | 0.50 [0.40, 1.00] | 10.45 [1.53, 18.58] | 0.52 [0.25, 0.63] | 0.41 [0.25, 0.60] | 21.89 [3.41, 41.83] | 16.92 [0.45, 35.61] | 0.56 [0.45, 1.00] | 0.57 [0.50, 1.00] | 0.40 [0.00, 1.40] |
| v2_physics_ctx | 0.01 | 0.35 [0.27, 1.00] | 19.40 [11.95, 25.60] | 0.34 [0.07, 0.44] | 0.29 [0.07, 0.38] | 29.35 [13.19, 44.74] | 19.90 [6.03, 35.27] | 0.44 [0.36, 1.00] | 0.50 [0.40, 1.00] | 0.80 [0.00, 1.80] |
| v2_gru_ctx | 0.4 | 0.40 [0.27, 1.00] | 9.95 [1.51, 17.31] | 0.52 [0.12, 0.60] | 0.40 [0.12, 0.50] | 20.90 [2.51, 40.01] | 15.42 [0.36, 31.62] | 0.44 [0.27, 1.00] | 0.43 [0.30, 1.00] | 1.40 [0.00, 3.20] |

## Operating point `recall_50` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) |
|---|---:|---|---|---|---|---|---|---|---|---|
| v1_k0 | — | no setting meets the rule on validation |||||||||
| v1_k0_announcer | — | no setting meets the rule on validation |||||||||
| nearest_in_corridor | — | no setting meets the rule on validation |||||||||
| v2_physics | — | no setting meets the rule on validation |||||||||
| v2_physics_nopose | — | no setting meets the rule on validation |||||||||
| v2_gru | — | no setting meets the rule on validation |||||||||
| v2_gru_nopose | — | no setting meets the rule on validation |||||||||
| v2_physics_ctx | — | no setting meets the rule on validation |||||||||
| v2_gru_ctx | — | no setting meets the rule on validation |||||||||

## Context on minus context off (test, same sessions, paired session-bootstrap 95% CI)

Context from `data/sanpo_scene_context/gt`. Each arm at its own validation-chosen knob.

| Arm | rule | Δ warned | Δ false alerts/min | Δ precision | Δ median lead, window ≥ 1 s (s) |
|---|---|---|---|---|---|
| v2_physics_ctx | budget_10 | 0.00 [0.00, 0.00] | 0.50 [0.00, 1.52] | -0.01 [-0.02, 0.00] | 0.00 [0.00, 0.00] |
| v2_gru_ctx | budget_2 | 0.05 [0.00, 0.07] | 0.50 [0.00, 1.35] | -0.01 [-0.19, 0.00] | 0.70 [0.00, 1.80] |
| v2_gru_ctx | budget_5 | 0.05 [0.00, 0.07] | 0.50 [0.00, 1.35] | -0.01 [-0.19, 0.00] | 0.70 [0.00, 1.80] |
| v2_gru_ctx | budget_10 | 0.05 [0.00, 0.07] | 0.50 [0.00, 1.35] | -0.01 [-0.19, 0.00] | 0.70 [0.00, 1.80] |

## Kaplan-Meier: share of arrivals not yet warned, by time since first sighting (test, `budget_5`)

| Arm | 0.5 s | 1 s | 2 s | 3 s | 4 s |
|---|---:|---:|---:|---:|---:|
| v1_k0 | n/a |
| v1_k0_announcer | n/a |
| nearest_in_corridor | 1.00 | 0.83 | 0.63 | 0.47 | 0.47 |
| v2_physics | n/a |
| v2_physics_nopose | n/a |
| v2_gru | 0.95 | 0.78 | 0.72 | 0.55 | 0.55 |
| v2_gru_nopose | 0.95 | 0.73 | 0.66 | 0.42 | 0.42 |
| v2_physics_ctx | n/a |
| v2_gru_ctx | 0.95 | 0.78 | 0.72 | 0.55 | 0.55 |

## Operating curves (test; every knob value)

Recall (warned) against false alerts per minute; the CSV next to this report has every column.

| Arm | knob | warned | false/min | precision | alerts/min |
|---|---:|---:|---:|---:|---:|
| v1_k0 | 0.05 | 0.70 | 127.9 | 0.18 | 156.2 |
| v1_k0 | 0.1 | 0.70 | 117.9 | 0.19 | 146.3 |
| v1_k0 | 0.2 | 0.70 | 110.9 | 0.20 | 138.8 |
| v1_k0 | 0.35 | 0.70 | 105.0 | 0.20 | 131.3 |
| v1_k0 | 0.5 | 0.65 | 97.5 | 0.19 | 119.9 |
| v1_k0 | 0.75 | 0.55 | 88.1 | 0.19 | 109.0 |
| v1_k0 | 1 | 0.50 | 81.1 | 0.19 | 100.5 |
| v1_k0 | 1.5 | 0.45 | 69.2 | 0.18 | 84.1 |
| v1_k0 | 2 | 0.40 | 64.2 | 0.17 | 77.6 |
| v1_k0 | 3 | 0.35 | 53.2 | 0.18 | 64.7 |
| v1_k0 | 5 | 0.35 | 39.3 | 0.22 | 50.2 |
| v1_k0 | 8 | 0.35 | 34.3 | 0.24 | 45.3 |
| v1_k0 | 12 | 0.35 | 31.3 | 0.26 | 42.3 |
| v1_k0_announcer | 0.05 | 0.45 | 44.8 | 0.17 | 54.2 |
| v1_k0_announcer | 0.1 | 0.45 | 42.8 | 0.17 | 51.2 |
| v1_k0_announcer | 0.2 | 0.45 | 42.8 | 0.17 | 51.2 |
| v1_k0_announcer | 0.35 | 0.45 | 42.3 | 0.17 | 50.7 |
| v1_k0_announcer | 0.5 | 0.45 | 42.3 | 0.16 | 50.2 |
| v1_k0_announcer | 0.75 | 0.40 | 38.8 | 0.17 | 46.8 |
| v1_k0_announcer | 1 | 0.40 | 36.8 | 0.18 | 44.8 |
| v1_k0_announcer | 1.5 | 0.40 | 34.3 | 0.19 | 42.3 |
| v1_k0_announcer | 2 | 0.35 | 32.8 | 0.17 | 39.8 |
| v1_k0_announcer | 3 | 0.35 | 29.4 | 0.19 | 36.3 |
| v1_k0_announcer | 5 | 0.35 | 25.9 | 0.21 | 32.8 |
| v1_k0_announcer | 8 | 0.35 | 23.9 | 0.23 | 30.8 |
| v1_k0_announcer | 12 | 0.35 | 21.9 | 0.24 | 28.9 |
| nearest_in_corridor | 0.01 | 0.30 | 22.9 | 0.32 | 33.8 |
| nearest_in_corridor | 0.03 | 0.35 | 18.9 | 0.36 | 29.4 |
| nearest_in_corridor | 0.05 | 0.35 | 16.4 | 0.39 | 26.9 |
| nearest_in_corridor | 0.1 | 0.35 | 14.4 | 0.43 | 25.4 |
| nearest_in_corridor | 0.15 | 0.35 | 12.9 | 0.46 | 23.9 |
| nearest_in_corridor | 0.2 | 0.40 | 11.4 | 0.50 | 22.9 |
| nearest_in_corridor | 0.3 | 0.35 | 9.5 | 0.53 | 19.9 |
| nearest_in_corridor | 0.4 | 0.40 | 8.0 | 0.56 | 17.9 |
| nearest_in_corridor | 0.5 | 0.30 | 7.0 | 0.52 | 14.4 |
| nearest_in_corridor | 0.6 | 0.20 | 7.5 | 0.44 | 13.4 |
| nearest_in_corridor | 0.7 | 0.15 | 7.0 | 0.39 | 11.4 |
| nearest_in_corridor | 0.8 | 0.15 | 8.0 | 0.33 | 11.9 |
| nearest_in_corridor | 0.9 | 0.15 | 7.5 | 0.35 | 11.4 |
| nearest_in_corridor | 0.95 | 0.15 | 6.5 | 0.35 | 10.0 |
| v2_physics | 0.01 | 0.35 | 18.9 | 0.34 | 28.9 |
| v2_physics | 0.03 | 0.35 | 18.9 | 0.34 | 28.9 |
| v2_physics | 0.05 | 0.35 | 18.9 | 0.34 | 28.9 |
| v2_physics | 0.1 | 0.35 | 18.9 | 0.34 | 28.9 |
| v2_physics | 0.15 | 0.35 | 18.9 | 0.34 | 28.9 |
| v2_physics | 0.2 | 0.35 | 18.9 | 0.34 | 28.9 |
| v2_physics | 0.3 | 0.35 | 18.9 | 0.34 | 28.9 |
| v2_physics | 0.4 | 0.35 | 18.9 | 0.34 | 28.9 |
| v2_physics | 0.5 | 0.35 | 18.9 | 0.34 | 28.9 |
| v2_physics | 0.6 | 0.35 | 18.4 | 0.36 | 28.9 |
| v2_physics | 0.7 | 0.30 | 18.4 | 0.36 | 28.9 |
| v2_physics | 0.8 | 0.30 | 17.9 | 0.36 | 27.9 |
| v2_physics | 0.9 | 0.30 | 16.9 | 0.37 | 26.9 |
| v2_physics | 0.95 | 0.35 | 15.9 | 0.38 | 25.9 |
| v2_physics_nopose | 0.01 | 0.35 | 20.9 | 0.31 | 30.3 |
| v2_physics_nopose | 0.03 | 0.35 | 20.9 | 0.31 | 30.3 |
| v2_physics_nopose | 0.05 | 0.35 | 20.9 | 0.31 | 30.3 |
| v2_physics_nopose | 0.1 | 0.35 | 20.9 | 0.31 | 30.3 |
| v2_physics_nopose | 0.15 | 0.35 | 20.9 | 0.31 | 30.3 |
| v2_physics_nopose | 0.2 | 0.35 | 20.9 | 0.31 | 30.3 |
| v2_physics_nopose | 0.3 | 0.35 | 20.9 | 0.31 | 30.3 |
| v2_physics_nopose | 0.4 | 0.35 | 20.9 | 0.31 | 30.3 |
| v2_physics_nopose | 0.5 | 0.35 | 19.4 | 0.33 | 28.9 |
| v2_physics_nopose | 0.6 | 0.35 | 18.4 | 0.36 | 28.9 |
| v2_physics_nopose | 0.7 | 0.35 | 18.4 | 0.36 | 28.9 |
| v2_physics_nopose | 0.8 | 0.35 | 18.4 | 0.35 | 28.4 |
| v2_physics_nopose | 0.9 | 0.35 | 17.9 | 0.36 | 27.9 |
| v2_physics_nopose | 0.95 | 0.35 | 17.4 | 0.36 | 27.4 |
| v2_gru | 0.01 | 0.30 | 47.8 | 0.17 | 57.7 |
| v2_gru | 0.03 | 0.25 | 29.9 | 0.24 | 39.3 |
| v2_gru | 0.05 | 0.35 | 19.4 | 0.37 | 30.8 |
| v2_gru | 0.1 | 0.30 | 14.4 | 0.42 | 24.9 |
| v2_gru | 0.15 | 0.45 | 10.0 | 0.56 | 22.4 |
| v2_gru | 0.2 | 0.45 | 9.5 | 0.57 | 21.9 |
| v2_gru | 0.3 | 0.35 | 10.0 | 0.52 | 20.9 |
| v2_gru | 0.4 | 0.35 | 9.5 | 0.54 | 20.4 |
| v2_gru | 0.5 | 0.35 | 8.5 | 0.56 | 19.4 |
| v2_gru | 0.6 | 0.40 | 8.0 | 0.56 | 17.9 |
| v2_gru | 0.7 | 0.25 | 7.0 | 0.52 | 14.4 |
| v2_gru | 0.8 | 0.20 | 6.5 | 0.50 | 12.9 |
| v2_gru | 0.9 | 0.15 | 6.0 | 0.40 | 10.0 |
| v2_gru | 0.95 | 0.15 | 6.0 | 0.40 | 10.0 |
| v2_gru_nopose | 0.01 | 0.40 | 48.3 | 0.17 | 58.2 |
| v2_gru_nopose | 0.03 | 0.35 | 30.8 | 0.23 | 40.3 |
| v2_gru_nopose | 0.05 | 0.45 | 19.9 | 0.37 | 31.3 |
| v2_gru_nopose | 0.1 | 0.40 | 14.9 | 0.41 | 25.4 |
| v2_gru_nopose | 0.15 | 0.50 | 11.4 | 0.51 | 23.4 |
| v2_gru_nopose | 0.2 | 0.50 | 10.9 | 0.52 | 22.9 |
| v2_gru_nopose | 0.3 | 0.50 | 10.4 | 0.52 | 21.9 |
| v2_gru_nopose | 0.4 | 0.50 | 10.4 | 0.52 | 21.9 |
| v2_gru_nopose | 0.5 | 0.50 | 9.5 | 0.55 | 20.9 |
| v2_gru_nopose | 0.6 | 0.50 | 9.0 | 0.54 | 19.4 |
| v2_gru_nopose | 0.7 | 0.35 | 8.5 | 0.47 | 15.9 |
| v2_gru_nopose | 0.8 | 0.30 | 9.0 | 0.40 | 14.9 |
| v2_gru_nopose | 0.9 | 0.25 | 8.5 | 0.29 | 11.9 |
| v2_gru_nopose | 0.95 | 0.25 | 8.5 | 0.29 | 11.9 |
| v2_physics_ctx | 0.01 | 0.35 | 19.4 | 0.34 | 29.4 |
| v2_physics_ctx | 0.03 | 0.35 | 19.4 | 0.34 | 29.4 |
| v2_physics_ctx | 0.05 | 0.35 | 19.4 | 0.34 | 29.4 |
| v2_physics_ctx | 0.1 | 0.35 | 19.4 | 0.34 | 29.4 |
| v2_physics_ctx | 0.15 | 0.35 | 19.4 | 0.34 | 29.4 |
| v2_physics_ctx | 0.2 | 0.35 | 19.4 | 0.34 | 29.4 |
| v2_physics_ctx | 0.3 | 0.35 | 19.4 | 0.34 | 29.4 |
| v2_physics_ctx | 0.4 | 0.35 | 19.4 | 0.34 | 29.4 |
| v2_physics_ctx | 0.5 | 0.35 | 18.9 | 0.33 | 28.4 |
| v2_physics_ctx | 0.6 | 0.35 | 18.9 | 0.33 | 28.4 |
| v2_physics_ctx | 0.7 | 0.35 | 18.9 | 0.33 | 28.4 |
| v2_physics_ctx | 0.8 | 0.35 | 18.4 | 0.35 | 28.4 |
| v2_physics_ctx | 0.9 | 0.35 | 17.9 | 0.36 | 27.9 |
| v2_physics_ctx | 0.95 | 0.35 | 17.4 | 0.36 | 27.4 |
| v2_gru_ctx | 0.01 | 0.25 | 48.8 | 0.16 | 58.2 |
| v2_gru_ctx | 0.03 | 0.25 | 32.8 | 0.22 | 42.3 |
| v2_gru_ctx | 0.05 | 0.30 | 23.4 | 0.32 | 34.3 |
| v2_gru_ctx | 0.1 | 0.35 | 14.9 | 0.43 | 26.4 |
| v2_gru_ctx | 0.15 | 0.30 | 14.4 | 0.42 | 24.9 |
| v2_gru_ctx | 0.2 | 0.45 | 11.4 | 0.50 | 22.9 |
| v2_gru_ctx | 0.3 | 0.45 | 9.5 | 0.56 | 21.4 |
| v2_gru_ctx | 0.4 | 0.40 | 10.0 | 0.52 | 20.9 |
| v2_gru_ctx | 0.5 | 0.35 | 9.5 | 0.53 | 19.9 |
| v2_gru_ctx | 0.6 | 0.35 | 9.5 | 0.53 | 19.9 |
| v2_gru_ctx | 0.7 | 0.35 | 8.5 | 0.55 | 18.9 |
| v2_gru_ctx | 0.8 | 0.40 | 9.0 | 0.53 | 18.9 |
| v2_gru_ctx | 0.9 | 0.40 | 8.0 | 0.56 | 17.9 |
| v2_gru_ctx | 0.95 | 0.40 | 7.5 | 0.57 | 17.4 |

Classes: true = the object enters the zone later within the horizon; late = it is already in the zone; false = it doesn't enter within the horizon. precision = (true + late) / all. warned = an announcement about the arriving track in the 4 s up to and including arrival.
Open-loop: the recorded walker never heard these alerts. Labels and inputs share SANPO's depth.
