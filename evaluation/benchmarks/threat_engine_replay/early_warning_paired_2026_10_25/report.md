# Threat-engine replay - early_warning_paired_2026_10_25

Label: in the walking corridor (width 0.75 m + class half-width, 0 < forward <= 3 m along the direction of travel from fixed_camera_poses.csv) within 20 steps (4 s); tracked objects only.

SANPO official test split: 85 sessions, 79 arrivals (a track first seen before it entered the zone). Operating points chosen on 48 validation sessions, then scored once on test. GRU: `models/GRU/GRU/runs/corridor_h8_seed0`. Generated 2026-10-25 by `python tools/replay_threat_engine.py` (393 s).

## Opportunity window (first sighting -> arrival), test

Median 2.6 s; 24% of arrivals were first seen less than 1 s before arriving, 41% less than 2 s, 33% at least 4 s before. Lead time can't exceed the window, so lead is reported conditionally below.

## Operating point `budget_2` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v2_gru | — | no setting meets the rule on validation |||||||||||
| v2_gru_fut_e1 | — | no setting meets the rule on validation |||||||||||

## Operating point `budget_5` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v2_gru | 0.6 / cd 8 / sm 1 / st 0.6 | 0.35 [0.23, 0.52] | 5.62 [3.44, 8.09] | 0.55 [0.40, 0.69] | 0.45 [0.29, 0.60] | 12.63 [7.73, 18.35] | 8.31 [3.72, 13.65] | 0.37 [0.22, 0.58] | 0.36 [0.23, 0.60] | 0.00 [0.00, 0.30] | 0.14 [0.06, 0.25] | 0.04 [0.00, 0.07] |
| v2_gru_fut_e1 | 0.5 / cd 2 / sm 1 / st 0.6 | 0.37 [0.26, 0.51] | 5.97 [3.54, 8.58] | 0.56 [0.38, 0.71] | 0.42 [0.24, 0.57] | 13.50 [7.82, 20.23] | 8.83 [3.75, 15.02] | 0.35 [0.21, 0.52] | 0.34 [0.18, 0.48] | 0.20 [0.00, 0.40] | 0.18 [0.07, 0.32] | 0.03 [0.00, 0.04] |

## Operating point `budget_10` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v2_gru | 0.4 / cd 8 / sm 1 / st 0.6 | 0.46 [0.36, 0.64] | 7.18 [4.42, 10.17] | 0.53 [0.39, 0.65] | 0.39 [0.26, 0.53] | 15.14 [9.75, 21.45] | 10.30 [5.22, 16.05] | 0.47 [0.36, 0.72] | 0.43 [0.30, 0.71] | 0.30 [0.00, 0.60] | 0.23 [0.14, 0.37] | 0.06 [0.03, 0.13] |
| v2_gru_fut_e1 | 0.3 / cd 8 / sm 1 / st 1.2 | 0.48 [0.37, 0.63] | 10.56 [7.34, 14.17] | 0.44 [0.29, 0.57] | 0.31 [0.19, 0.41] | 18.69 [12.47, 25.70] | 12.20 [6.52, 18.80] | 0.48 [0.33, 0.67] | 0.47 [0.26, 0.68] | 0.40 [0.00, 0.80] | 0.29 [0.19, 0.42] | 0.09 [0.03, 0.15] |

## Operating point `early_budget_2` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v2_gru | — | no setting meets the rule on validation |||||||||||
| v2_gru_fut_e1 | — | no setting meets the rule on validation |||||||||||

## Operating point `early_budget_5` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v2_gru | 0.6 / cd 8 / sm 1 / st 0.6 | 0.35 [0.23, 0.52] | 5.62 [3.44, 8.09] | 0.55 [0.40, 0.69] | 0.45 [0.29, 0.60] | 12.63 [7.73, 18.35] | 8.31 [3.72, 13.65] | 0.37 [0.22, 0.58] | 0.36 [0.23, 0.60] | 0.00 [0.00, 0.30] | 0.14 [0.06, 0.25] | 0.04 [0.00, 0.07] |
| v2_gru_fut_e1 | 0.5 / cd 2 / sm 1 / st 0.6 | 0.37 [0.26, 0.51] | 5.97 [3.54, 8.58] | 0.56 [0.38, 0.71] | 0.42 [0.24, 0.57] | 13.50 [7.82, 20.23] | 8.83 [3.75, 15.02] | 0.35 [0.21, 0.52] | 0.34 [0.18, 0.48] | 0.20 [0.00, 0.40] | 0.18 [0.07, 0.32] | 0.03 [0.00, 0.04] |

## Operating point `early_budget_10` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v2_gru | 0.3 / cd 8 / sm 1 / st 0.6 | 0.46 [0.36, 0.64] | 8.65 [5.79, 11.89] | 0.49 [0.36, 0.62] | 0.36 [0.24, 0.49] | 16.87 [11.31, 23.27] | 10.73 [5.63, 16.64] | 0.47 [0.36, 0.72] | 0.43 [0.30, 0.71] | 0.20 [0.00, 0.80] | 0.24 [0.16, 0.40] | 0.09 [0.04, 0.21] |
| v2_gru_fut_e1 | 0.3 / cd 8 / sm 1 / st 1.2 | 0.48 [0.37, 0.63] | 10.56 [7.34, 14.17] | 0.44 [0.29, 0.57] | 0.31 [0.19, 0.41] | 18.69 [12.47, 25.70] | 12.20 [6.52, 18.80] | 0.48 [0.33, 0.67] | 0.47 [0.26, 0.68] | 0.40 [0.00, 0.80] | 0.29 [0.19, 0.42] | 0.09 [0.03, 0.15] |

## Operating point `recall_50` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v2_gru | — | no setting meets the rule on validation |||||||||||
| v2_gru_fut_e1 | — | no setting meets the rule on validation |||||||||||

## Operating point `crc_budget_2` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v2_gru | — | no setting meets the rule on validation |||||||||||
| v2_gru_fut_e1 | — | no setting meets the rule on validation |||||||||||

## Operating point `crc_budget_5` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v2_gru | 0.5 / cd 8 / sm 1 / st 0.6 | 0.39 [0.28, 0.54] | 6.58 [4.05, 9.46] | 0.52 [0.38, 0.65] | 0.41 [0.27, 0.55] | 13.76 [8.67, 19.88] | 9.26 [4.41, 14.58] | 0.40 [0.26, 0.60] | 0.38 [0.23, 0.62] | 0.20 [0.00, 0.40] | 0.19 [0.09, 0.32] | 0.06 [0.03, 0.12] |
| v2_gru_fut_e1 | 0.5 / cd 2 / sm 1 / st 0.6 | 0.37 [0.26, 0.51] | 5.97 [3.54, 8.58] | 0.56 [0.38, 0.71] | 0.42 [0.24, 0.57] | 13.50 [7.82, 20.23] | 8.83 [3.75, 15.02] | 0.35 [0.21, 0.52] | 0.34 [0.18, 0.48] | 0.20 [0.00, 0.40] | 0.18 [0.07, 0.32] | 0.03 [0.00, 0.04] |

## Operating point `crc_budget_10` (test, 95% session-bootstrap CI)

| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ | warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) | warned ≥ 0.2 s before | warned ≥ 1 s before |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v2_gru | 0.4 / cd 8 / sm 1 / st 0.6 | 0.46 [0.36, 0.64] | 7.18 [4.42, 10.17] | 0.53 [0.39, 0.65] | 0.39 [0.26, 0.53] | 15.14 [9.75, 21.45] | 10.30 [5.22, 16.05] | 0.47 [0.36, 0.72] | 0.43 [0.30, 0.71] | 0.30 [0.00, 0.60] | 0.23 [0.14, 0.37] | 0.06 [0.03, 0.13] |
| v2_gru_fut_e1 | 0.2 / cd 8 / sm 1 / st 1.2 | 0.53 [0.43, 0.68] | 14.28 [10.39, 18.09] | 0.37 [0.24, 0.50] | 0.25 [0.16, 0.35] | 22.76 [16.09, 29.89] | 15.32 [9.46, 21.76] | 0.53 [0.40, 0.73] | 0.53 [0.34, 0.77] | 0.40 [0.20, 1.00] | 0.37 [0.26, 0.50] | 0.13 [0.05, 0.23] |

## Kaplan-Meier: share of arrivals not yet warned, by time since first sighting (test, `budget_5`)

| Arm | 0.5 s | 1 s | 2 s | 3 s | 4 s |
|---|---:|---:|---:|---:|---:|
| v2_gru | 0.95 | 0.86 | 0.82 | 0.74 | 0.64 |
| v2_gru_fut_e1 | 0.95 | 0.84 | 0.80 | 0.77 | 0.69 |

## Kaplan-Meier: share of arrivals not yet warned, by time since first sighting (test, `early_budget_5`)

| Arm | 0.5 s | 1 s | 2 s | 3 s | 4 s |
|---|---:|---:|---:|---:|---:|
| v2_gru | 0.95 | 0.86 | 0.82 | 0.74 | 0.64 |
| v2_gru_fut_e1 | 0.95 | 0.84 | 0.80 | 0.77 | 0.69 |

## Paired differences (test, same sessions, paired session-bootstrap 95% CI)

Each arm at its own validation-chosen knob.

| A - B | rule | Δ warned | Δ warned ≥ 0.2 s | Δ warned ≥ 1 s | Δ false alerts/min | Δ precision |
|---|---|---|---|---|---|---|
| v2_gru_fut_e1 - v2_gru | budget_5 | 0.01 [-0.12, 0.09] | 0.04 [-0.02, 0.10] | -0.01 [-0.06, 0.00] | 0.35 [-0.29, 0.92] | 0.00 [-0.05, 0.03] |
| v2_gru_fut_e1 - v2_gru | budget_10 | 0.03 [-0.10, 0.09] | 0.06 [-0.08, 0.13] | 0.03 [-0.05, 0.08] | 3.37 [1.98, 4.93] | -0.09 [-0.15, -0.05] |
| v2_gru_fut_e1 - v2_gru | early_budget_5 | 0.01 [-0.12, 0.09] | 0.04 [-0.02, 0.10] | -0.01 [-0.06, 0.00] | 0.35 [-0.29, 0.92] | 0.00 [-0.05, 0.03] |
| v2_gru_fut_e1 - v2_gru | early_budget_10 | 0.03 [-0.10, 0.09] | 0.05 [-0.09, 0.11] | 0.00 [-0.12, 0.06] | 1.90 [0.66, 3.24] | -0.05 [-0.11, -0.01] |
| v2_gru_fut_e1 - v2_gru | crc_budget_5 | -0.03 [-0.14, 0.04] | -0.01 [-0.08, 0.03] | -0.04 [-0.11, 0.00] | -0.61 [-1.39, 0.17] | 0.04 [-0.03, 0.07] |
| v2_gru_fut_e1 - v2_gru | crc_budget_10 | 0.08 [-0.03, 0.14] | 0.14 [0.02, 0.21] | 0.06 [-0.03, 0.15] | 7.10 [4.91, 9.26] | -0.15 [-0.22, -0.10] |


## Operating curves (test; every knob value)

Recall (warned) against false alerts per minute; the CSV next to this report has every column.

| Arm | knob | warned | warned ≥ 0.2 s before | false/min | precision | alerts/min |
|---|---:|---:|---:|---:|---:|---:|

Classes: true = the object enters the zone later within the horizon; late = it is already in the zone; false = it doesn't enter within the horizon. precision = (true + late) / all. warned = an announcement about the arriving track in the 4 s up to and including arrival.
Open-loop: the recorded walker never heard these alerts. Labels and inputs share SANPO's depth.
