# Kalman noise: declared vs fitted, paired (test, 83 sessions)

after = train-fitted NoiseModel(), before = NoiseModel.declared(); difference after - before with a 95% paired session-bootstrap CI. PhysicsArrivalPredictor (mean-state corridor entry), same announcer.

| arm | knob | metric | before | after | after - before [95% CI] |
|---|---:|---|---:|---:|---|
| v2_physics | 0.5 | warned | 0.41 | 0.42 | +0.01 [-0.07, +0.08] |
| v2_physics | 0.5 | precision | 0.27 | 0.30 | +0.03 [+0.00, +0.06] |
| v2_physics | 0.5 | false_per_min | 24.41 | 24.59 | +0.18 [-1.52, +1.85] |
| v2_physics | 0.5 | alerts_per_min | 33.34 | 35.19 | +1.84 [-0.10, +3.72] |
| v2_physics | 0.95 | warned | 0.36 | 0.42 | +0.05 [-0.02, +0.15] |
| v2_physics | 0.95 | precision | 0.31 | 0.34 | +0.03 [-0.01, +0.07] |
| v2_physics | 0.95 | false_per_min | 19.62 | 20.26 | +0.64 [-1.13, +2.39] |
| v2_physics | 0.95 | alerts_per_min | 28.37 | 30.76 | +2.39 [+0.21, +4.49] |
| v2_physics_nopose | 0.5 | warned | 0.48 | 0.48 | +0.00 [-0.06, +0.05] |
| v2_physics_nopose | 0.5 | precision | 0.26 | 0.28 | +0.02 [-0.00, +0.04] |
| v2_physics_nopose | 0.5 | false_per_min | 27.36 | 27.82 | +0.46 [-1.62, +2.57] |
| v2_physics_nopose | 0.5 | alerts_per_min | 36.75 | 38.41 | +1.66 [-0.79, +4.25] |
| v2_physics_nopose | 0.95 | warned | 0.45 | 0.47 | +0.02 [-0.03, +0.09] |
| v2_physics_nopose | 0.95 | precision | 0.29 | 0.30 | +0.02 [-0.01, +0.04] |
| v2_physics_nopose | 0.95 | false_per_min | 22.94 | 23.86 | +0.92 [-1.01, +2.81] |
| v2_physics_nopose | 0.95 | alerts_per_min | 32.24 | 34.26 | +2.03 [-0.30, +4.43] |
