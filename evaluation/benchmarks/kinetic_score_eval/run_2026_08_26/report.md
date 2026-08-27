# Kinetic Score Ablation

Sessions: 139 · scored frames (≥2 objects): 19402

Point estimate is frame-count-weighted across sessions; CI is a 95% percentile bootstrap resampling **whole sessions** (frames within a session are autocorrelated — frame-level CIs would be far too narrow).

## flicker_rate

_lower better — how often the announced top threat changes identity_

| Variant | Value | 95% CI |
|---|---:|---|
| `K0  sev·v²/d` | 0.975 | [0.964, 0.983] |
| `linear  sev·v/d` | 0.975 | [0.964, 0.983] |
| `no-severity  v²/d` | 0.975 | [0.964, 0.983] |
| `no-velocity  sev/d` | 0.994 | [0.991, 0.997] |
| `size  sev·v²·s^½/d` | 0.974 | [0.963, 0.983] |
| `lam=0.25  weak mass` | 0.975 | [0.964, 0.983] |
| `lam=1.0  full KE` | 0.975 | [0.964, 0.983] |
| `ttc  -(d-D)/v` | 0.572 | [0.516, 0.610] |

## tie_rate

_lower better — frames where the top two are within 5%_

| Variant | Value | 95% CI |
|---|---:|---|
| `K0  sev·v²/d` | 0.001 | [0.000, 0.001] |
| `linear  sev·v/d` | 0.001 | [0.000, 0.002] |
| `no-severity  v²/d` | 0.001 | [0.000, 0.001] |
| `no-velocity  sev/d` | 0.979 | [0.970, 0.985] |
| `size  sev·v²·s^½/d` | 0.001 | [0.000, 0.002] |
| `lam=0.25  weak mass` | 0.001 | [0.000, 0.001] |
| `lam=1.0  full KE` | 0.001 | [0.000, 0.001] |
| `ttc  -(d-D)/v` | 0.000 | [0.000, 0.000] |

## smoothness

_lower better — mean relative frame-to-frame change per track_

| Variant | Value | 95% CI |
|---|---:|---|
| `K0  sev·v²/d` | 1.755 | [1.661, 1.856] |
| `linear  sev·v/d` | 1.627 | [1.521, 1.740] |
| `no-severity  v²/d` | 1.753 | [1.660, 1.854] |
| `no-velocity  sev/d` | 0.339 | [0.295, 0.388] |
| `size  sev·v²·s^½/d` | 1.753 | [1.659, 1.850] |
| `lam=0.25  weak mass` | 1.754 | [1.661, 1.855] |
| `lam=1.0  full KE` | 1.757 | [1.663, 1.857] |
| `ttc  -(d-D)/v` | 6.178 | [6.178, 6.178] |

## future_consistency

_higher better — pick at T still the pick at T+H_

| Variant | Value | 95% CI |
|---|---:|---|
| `K0  sev·v²/d` | 0.002 | [0.001, 0.004] |
| `linear  sev·v/d` | 0.002 | [0.001, 0.004] |
| `no-severity  v²/d` | 0.002 | [0.001, 0.004] |
| `no-velocity  sev/d` | 0.001 | [0.000, 0.002] |
| `size  sev·v²·s^½/d` | 0.003 | [0.001, 0.004] |
| `lam=0.25  weak mass` | 0.002 | [0.001, 0.004] |
| `lam=1.0  full KE` | 0.002 | [0.001, 0.004] |
| `ttc  -(d-D)/v` | 0.076 | [0.038, 0.138] |

## encounter_top1

_higher better — pick was an object that measurably closed in (§6: eliminates, never selects)_

| Variant | Value | 95% CI |
|---|---:|---|
| `K0  sev·v²/d` | 0.307 | [0.263, 0.355] |
| `linear  sev·v/d` | 0.307 | [0.263, 0.355] |
| `no-severity  v²/d` | 0.307 | [0.263, 0.355] |
| `no-velocity  sev/d` | 0.342 | [0.295, 0.388] |
| `size  sev·v²·s^½/d` | 0.307 | [0.263, 0.355] |
| `lam=0.25  weak mass` | 0.307 | [0.263, 0.355] |
| `lam=1.0  full KE` | 0.307 | [0.263, 0.355] |
| `ttc  -(d-D)/v` | 0.067 | [0.014, 0.125] |

## rank_stability_2pct

_higher better — Kendall τ under 2% depth noise_

| Variant | Value | 95% CI |
|---|---:|---|
| `K0  sev·v²/d` | 1.000 | [1.000, 1.000] |
| `linear  sev·v/d` | 1.000 | [1.000, 1.000] |
| `no-severity  v²/d` | 1.000 | [1.000, 1.000] |
| `no-velocity  sev/d` | 0.524 | [0.500, 0.550] |
| `size  sev·v²·s^½/d` | 0.999 | [0.996, 1.000] |
| `lam=0.25  weak mass` | 1.000 | [1.000, 1.000] |
| `lam=1.0  full KE` | 1.000 | [1.000, 1.000] |
| `ttc  -(d-D)/v` | 1.000 | [1.000, 1.000] |

## rank_stability_5pct

_higher better — Kendall τ under 5% depth noise_

| Variant | Value | 95% CI |
|---|---:|---|
| `K0  sev·v²/d` | 1.000 | [1.000, 1.000] |
| `linear  sev·v/d` | 0.999 | [0.996, 1.000] |
| `no-severity  v²/d` | 1.000 | [1.000, 1.000] |
| `no-velocity  sev/d` | 0.387 | [0.362, 0.414] |
| `size  sev·v²·s^½/d` | 0.998 | [0.996, 1.000] |
| `lam=0.25  weak mass` | 0.998 | [0.993, 1.000] |
| `lam=1.0  full KE` | 1.000 | [1.000, 1.000] |
| `ttc  -(d-D)/v` | 1.000 | [1.000, 1.000] |

## rank_stability_10pct

_higher better — Kendall τ under 10% depth noise_

| Variant | Value | 95% CI |
|---|---:|---|
| `K0  sev·v²/d` | 0.998 | [0.996, 1.000] |
| `linear  sev·v/d` | 0.998 | [0.996, 0.999] |
| `no-severity  v²/d` | 0.997 | [0.993, 1.000] |
| `no-velocity  sev/d` | 0.283 | [0.259, 0.309] |
| `size  sev·v²·s^½/d` | 0.999 | [0.999, 1.000] |
| `lam=0.25  weak mass` | 0.997 | [0.992, 1.000] |
| `lam=1.0  full KE` | 0.997 | [0.992, 1.000] |
| `ttc  -(d-D)/v` | 0.258 | [0.000, 0.500] |

## What this does not answer

Whether `v²` and the severity weights are *right* is a value judgment. No volume of unlabelled video settles it — see `disagreements.json` and `evaluation/vlm_referee.py`.
