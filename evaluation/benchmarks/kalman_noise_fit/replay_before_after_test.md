# Kalman noise: declared vs fitted, paired (test, 83 sessions)

after = train-fitted NoiseModel(), before = NoiseModel.declared(); difference after - before with a 95% paired session-bootstrap CI. PhysicsArrivalPredictor (mean-state corridor entry), same announcer.

| arm | knob | metric | before | after | after - before [95% CI] |
|---|---:|---|---:|---:|---|
| v2_physics | 0.5 | warned | 0.41 | 0.26 | -0.16 [-0.26, -0.07] |
| v2_physics | 0.5 | precision | 0.26 | 0.30 | +0.04 [+0.00, +0.07] |
| v2_physics | 0.5 | false_per_min | 24.13 | 17.04 | -7.09 [-9.25, -4.80] |
| v2_physics | 0.5 | alerts_per_min | 32.61 | 24.22 | -8.38 [-10.90, -5.56] |
| v2_physics | 0.95 | warned | 0.39 | 0.26 | -0.14 [-0.23, -0.04] |
| v2_physics | 0.95 | precision | 0.31 | 0.36 | +0.05 [+0.01, +0.09] |
| v2_physics | 0.95 | false_per_min | 18.70 | 13.17 | -5.53 [-7.41, -3.49] |
| v2_physics | 0.95 | alerts_per_min | 27.08 | 20.63 | -6.45 [-8.78, -3.89] |
| v2_physics_nopose | 0.5 | warned | 0.47 | 0.31 | -0.16 [-0.23, -0.09] |
| v2_physics_nopose | 0.5 | precision | 0.24 | 0.35 | +0.11 [+0.05, +0.17] |
| v2_physics_nopose | 0.5 | false_per_min | 27.36 | 13.72 | -13.63 [-16.38, -10.85] |
| v2_physics_nopose | 0.5 | alerts_per_min | 36.11 | 21.19 | -14.92 [-17.60, -11.98] |
| v2_physics_nopose | 0.95 | warned | 0.45 | 0.30 | -0.15 [-0.21, -0.07] |
| v2_physics_nopose | 0.95 | precision | 0.28 | 0.42 | +0.14 [+0.08, +0.22] |
| v2_physics_nopose | 0.95 | false_per_min | 22.57 | 10.13 | -12.43 [-14.42, -10.43] |
| v2_physics_nopose | 0.95 | alerts_per_min | 31.41 | 17.59 | -13.82 [-15.95, -11.34] |
