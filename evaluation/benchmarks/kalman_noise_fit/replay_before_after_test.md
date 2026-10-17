# Kalman noise: declared vs fitted, paired (test, 71 sessions)

after = train-fitted NoiseModel(), before = NoiseModel.declared(); difference after - before with a 95% paired session-bootstrap CI. PhysicsArrivalPredictor (mean-state corridor entry), same announcer.

| arm | knob | metric | before | after | after - before [95% CI] |
|---|---:|---|---:|---:|---|
| v2_physics | 0.5 | warned | 0.38 | 0.35 | -0.03 [-0.17, +0.11] |
| v2_physics | 0.5 | precision | 0.26 | 0.34 | +0.08 [+0.01, +0.15] |
| v2_physics | 0.5 | false_per_min | 21.93 | 13.79 | -8.14 [-11.28, -5.43] |
| v2_physics | 0.5 | alerts_per_min | 29.79 | 20.97 | -8.83 [-12.16, -5.70] |
| v2_physics | 0.95 | warned | 0.40 | 0.27 | -0.13 [-0.33, -0.04] |
| v2_physics | 0.95 | precision | 0.32 | 0.40 | +0.08 [+0.01, +0.17] |
| v2_physics | 0.95 | false_per_min | 17.66 | 10.34 | -7.31 [-10.73, -4.38] |
| v2_physics | 0.95 | alerts_per_min | 25.79 | 17.24 | -8.55 [-11.65, -5.62] |
| v2_physics_nopose | 0.5 | warned | 0.37 | 0.24 | -0.13 [-0.27, -0.02] |
| v2_physics_nopose | 0.5 | precision | 0.28 | 0.43 | +0.16 [+0.08, +0.26] |
| v2_physics_nopose | 0.5 | false_per_min | 22.21 | 7.72 | -14.48 [-17.65, -11.46] |
| v2_physics_nopose | 0.5 | alerts_per_min | 30.76 | 13.66 | -17.10 [-21.26, -13.12] |
| v2_physics_nopose | 0.95 | warned | 0.35 | 0.29 | -0.06 [-0.24, +0.06] |
| v2_physics_nopose | 0.95 | precision | 0.31 | 0.54 | +0.22 [+0.13, +0.37] |
| v2_physics_nopose | 0.95 | false_per_min | 18.07 | 4.97 | -13.10 [-16.30, -10.11] |
| v2_physics_nopose | 0.95 | alerts_per_min | 26.34 | 10.76 | -15.59 [-20.07, -11.68] |
