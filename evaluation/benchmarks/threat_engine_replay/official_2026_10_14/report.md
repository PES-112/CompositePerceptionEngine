# Threat-engine replay - SANPO official test split

71 official test sessions with tracked objects, 88 arrivals (a track first seen before it came within 3 m in the ±30° cone). Thresholds chosen on 41 validation sessions. GRU: `models/GRU/GRU/runs/official462_2026_10_14_seed0`. Generated 2026-10-14 by `python tools/replay_threat_engine.py` (38 s).

| Arm | ignore threshold | warned ↑ | precision ↑ | median lead (s) ↑ | alerts/min | overrides/min | switches/min ↓ |
|---|---:|---|---|---|---|---|---|
| v1_k0 | — | 0.65 [0.58, 0.81] | 0.24 [0.12, 0.38] | 0.60 [0.00, 0.80] | 119.17 [103.52, 134.92] | 38.90 [27.81, 50.37] | 61.38 [47.65, 75.18] |
| v1_k0_announcer | — | 0.53 [0.46, 0.73] | 0.22 [0.10, 0.34] | 0.20 [0.00, 0.80] | 49.10 [43.16, 55.30] | 26.76 [19.55, 34.78] | 39.86 [32.90, 46.73] |
| nearest_in_cone | 0.6 | 0.33 [0.24, 0.54] | 0.15 [0.09, 0.22] | 0.20 [0.00, 0.60] | 52.00 [48.11, 57.75] | 6.48 [3.59, 10.05] | 39.86 [36.35, 44.00] |
| v2_physics | 0.6 | 0.30 [0.20, 0.54] | 0.27 [0.17, 0.37] | 0.20 [0.00, 2.60] | 29.24 [23.63, 35.33] | 6.48 [3.59, 10.05] | 21.66 [15.57, 27.71] |
| v2_physics_nopose | 0.6 | 0.32 [0.21, 0.59] | 0.31 [0.18, 0.40] | 0.00 [0.00, 1.00] | 32.69 [25.87, 39.99] | 9.38 [4.57, 15.05] | 23.03 [16.33, 30.19] |
| v2_gru | 0.4 | 0.26 [0.18, 0.49] | 0.59 [0.47, 0.69] | 0.00 [0.00, 0.00] | 16.97 [10.36, 24.34] | 6.48 [3.59, 10.05] | 12.00 [5.44, 19.02] |
| v2_gru_nopose | 0.6 | 0.23 [0.15, 0.43] | 0.53 [0.43, 0.61] | 0.00 [0.00, 0.20] | 16.55 [9.13, 24.92] | 9.38 [4.57, 15.05] | 11.72 [4.43, 19.98] |

95% CIs resample whole sessions. warned: arrivals with an announcement about that track in the 4 s before it arrived. precision: announcements whose object is within 3 m in the cone now or within the next 4 s (the arrival label at that frame).

## Threshold selection (validation sessions, F1 of warned and precision)

| Arm | 0.05 | 0.1 | 0.15 | 0.25 | 0.4 | 0.6 |
|---|---:|---:|---:|---:|---:|---:|
| nearest_in_cone | 0.172 | 0.172 | 0.172 | 0.172 | 0.188 | 0.191 |
| v2_physics | 0.249 | 0.249 | 0.249 | 0.250 | 0.252 | 0.255 |
| v2_physics_nopose | 0.326 | 0.326 | 0.326 | 0.327 | 0.328 | 0.334 |
| v2_gru | 0.311 | 0.334 | 0.379 | 0.339 | 0.404 | 0.384 |
| v2_gru_nopose | 0.335 | 0.360 | 0.403 | 0.418 | 0.427 | 0.455 |

Open-loop: the recorded walker never heard these alerts. Labels and inputs share SANPO's depth.
Only YOLO-tracked objects are replayed (depth-sweep pseudo-objects have no identity to warn about).
