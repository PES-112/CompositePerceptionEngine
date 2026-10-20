# Kalman noise: declared vs fitted, paired (test, 85 sessions)

after = train-fitted NoiseModel(), before = NoiseModel.declared(); difference after - before with a 95% paired session-bootstrap CI. PhysicsArrivalPredictor (mean-state corridor entry), same announcer.

| arm | knob | metric | before | after | after - before [95% CI] |
|---|---:|---|---:|---:|---|
| v2_physics | 0.5 | warned | 0.42 | 0.39 | -0.03 [-0.13, +0.06] |
| v2_physics | 0.5 | precision | 0.24 | 0.28 | +0.04 [+0.01, +0.07] |
| v2_physics | 0.5 | false_per_min | 21.55 | 17.22 | -4.33 [-6.22, -2.54] |
| v2_physics | 0.5 | alerts_per_min | 28.30 | 23.97 | -4.33 [-6.35, -2.41] |
| v2_physics | 0.95 | warned | 0.42 | 0.38 | -0.04 [-0.13, +0.03] |
| v2_physics | 0.95 | precision | 0.29 | 0.33 | +0.04 [+0.01, +0.08] |
| v2_physics | 0.95 | false_per_min | 16.87 | 13.76 | -3.12 [-4.68, -1.54] |
| v2_physics | 0.95 | alerts_per_min | 23.71 | 20.59 | -3.12 [-4.80, -1.35] |
| v2_physics_nopose | 0.5 | warned | 0.48 | 0.38 | -0.10 [-0.15, -0.05] |
| v2_physics_nopose | 0.5 | precision | 0.22 | 0.26 | +0.05 [+0.02, +0.07] |
| v2_physics_nopose | 0.5 | false_per_min | 25.44 | 19.73 | -5.71 [-7.98, -3.43] |
| v2_physics_nopose | 0.5 | alerts_per_min | 32.45 | 26.82 | -5.62 [-7.76, -3.47] |
| v2_physics_nopose | 0.95 | warned | 0.47 | 0.35 | -0.11 [-0.16, -0.07] |
| v2_physics_nopose | 0.95 | precision | 0.26 | 0.30 | +0.04 [+0.02, +0.07] |
| v2_physics_nopose | 0.95 | false_per_min | 20.42 | 15.66 | -4.76 [-6.35, -3.20] |
| v2_physics_nopose | 0.95 | alerts_per_min | 27.43 | 22.32 | -5.11 [-6.62, -3.61] |
