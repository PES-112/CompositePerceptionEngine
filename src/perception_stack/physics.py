"""
physics.py
==========
CPE Perception Stack — Physics calculations for perceived objects.

Functions:
    compute_bearing     : pixel x-coordinate → bearing in degrees
    compute_velocity    : rolling window depth history → closing velocity m/s
    kinetic_score       : K = severity × v² / max(d, ε)
    bearing_label       : bearing degrees → human-readable direction string
"""

# ── Class severity weights ─────────────────────────────────────────────────────
# Severity is a *constant per class*, derived from the object's real-world mass
# rather than hand-tuned. Collision consequence scales with delivered energy, so
# mass is the physical quantity severity is a proxy for.
#
#     severity(c) = behaviour(c) × (mass(c) / mass(person)) ** SEVERITY_LAMBDA
#
# LAMBDA is the compression knob. λ=1 is literal kinetic energy (bus 171× a
# person) which lets a far-off bus outrank a pedestrian about to be hit; λ=0
# throws mass away entirely. λ=0.5 — severity ∝ √mass — keeps a real destructive
# bias (car 4.6× a person, bus 13×, bus 2.8× a car) without the runaway, and is
# the geometric midpoint of those two extremes. Note the compression argument is
# weaker than it looks: with v² in the score, kinematics dominate severity in
# most reorderings, so λ mostly breaks ties between objects of similar motion.
# Fitting the project's old hand-tuned table to log-mass recovers λ ≈ 0.18
# (R²=0.50) — evidence the hand table was *under*-weighting mass, not that 0.18
# is right.
#
# FROZEN at 0.5 (decided 2026-08-25). The λ sweep arms were removed from
# evaluation/kinetic_ablation.py: with Tier-B blinded human labelling dropped
# from the evaluation plan, nothing left in the pipeline can adjudicate between
# λ values — the label-free metrics score arrival time, which λ barely moves.
# λ is therefore a declared design choice, not a measured result, and must be
# reported as a limitation rather than defended as a finding. Same applies to
# BEHAVIOUR_MULTIPLIER below.
CLASS_MASS_KG: dict[str, float] = {
    "person":              70.0,
    "bicycle":             15.0,
    "car":               1500.0,
    "motorcycle":         200.0,
    "bus":              12000.0,
    "truck":             8000.0,
    "traffic light":       50.0,
    "stop sign":           15.0,
    "fire hydrant":        80.0,
    "dog":                 25.0,
    "bench":               50.0,
    "pole":               100.0,
    "bollard":             60.0,
    "unlabeled_obstacle":  50.0,
}
REFERENCE_MASS_KG = 70.0    # person — severity 1.0 by definition
SEVERITY_LAMBDA   = 0.5

# Mass is not the whole hazard. These few classes carry a *secondary* hazard the
# mass law cannot see: erratic motion, unpredictability, or being mounted out of
# the walking path. Sparse by design — every entry here is a claim that needs
# defending, so keep the list short.
BEHAVIOUR_MULTIPLIER: dict[str, float] = {
    "motorcycle":    1.3,   # erratic, fast, quiet
    "dog":           1.5,   # unpredictable trajectory
    "bicycle":       1.2,   # silent and fast for its mass
    "bollard":       1.4,   # low, hard, trip-height
    "pole":          1.2,   # narrow, easily missed
    "traffic light": 0.4,   # mounted above head height — rarely a collision body
    "stop sign":     0.4,   # same
}

# Trip hazards with no meaningful mass. Energy transfer is not the mechanism —
# the pedestrian supplies the energy — so these keep hand-set values and are
# excluded from the mass law rather than fudged into it.
STATIC_HAZARD_SEVERITY: dict[str, float] = {
    "stairs":    2.0,
    "pothole":   1.8,
    "puddle":    1.1,
    "crosswalk": 0.5,
}

DEFAULT_SEVERITY = 1.0
EPSILON = 0.5   # metres — prevents division by zero for very close objects

# Relative-size term (off by default — see kinetic_score). Apparent size
# A_px / d is the user-facing "how big does it loom" quantity; SIZE_REFERENCE
# normalises it so the term is ~1 for a person at conversational distance.
# ponytail: single global reference constant, not a per-camera intrinsic. Swap
# for A_px · d² / f² if you ever need true physical cross-section in m².
SIZE_REFERENCE_PX_PER_M = 12000.0
SIZE_EXPONENT = 0.0   # 0 disables the term entirely; the ablation sweeps it


def class_severity(class_name: str) -> float:
    """
    Constant severity weight for a class, from real-world mass.

    Static trip hazards use their explicit table; unknown classes fall back to
    DEFAULT_SEVERITY (= person).
    """
    if class_name in STATIC_HAZARD_SEVERITY:
        return STATIC_HAZARD_SEVERITY[class_name]
    mass = CLASS_MASS_KG.get(class_name)
    if mass is None:
        return DEFAULT_SEVERITY
    ratio = (mass / REFERENCE_MASS_KG) ** SEVERITY_LAMBDA
    return BEHAVIOUR_MULTIPLIER.get(class_name, 1.0) * ratio


# Materialised once so callers (and the CSV/fact-sheet layers) can read weights
# without recomputing. Regenerate if you change LAMBDA at runtime.
CLASS_SEVERITY: dict[str, float] = {
    name: round(class_severity(name), 3)
    for name in list(CLASS_MASS_KG) + list(STATIC_HAZARD_SEVERITY)
}


def compute_bearing(cx_px: float, frame_width: int, hfov_deg: float = 70.0) -> float:
    """
    Convert the pixel x-coordinate of an object's centre to a bearing in degrees.

    Returns:
        Negative = object is to the LEFT of ego path.
        Positive = object is to the RIGHT.
        0        = directly ahead.

    Assumes a horizontal field of view of 70° (typical phone / dashcam lens).
    """
    normalised = (cx_px - frame_width / 2) / (frame_width / 2)   # normalise to [-1, 1]
    return normalised * (hfov_deg / 2)


def compute_velocity(
    depth_history: list[tuple[int, float]],
    fps: float,
    *,
    use_least_squares: bool = True,
) -> float:
    """
    Estimate closing velocity (m/s) from a rolling window of (frame_idx, distance_m) pairs.

    Positive return value means the object is APPROACHING (depth decreasing).
    Returns 0.0 if fewer than 2 history samples exist or the object is moving away.

    Velocity is a *differentiated* depth estimate, so depth noise is amplified before
    the kinetic score ever sees it — and kinetic_score() then squares it. Taking the
    least-squares slope across the whole window uses every sample instead of just the
    two endpoints, which cuts that variance without adding state or tuning knobs.

    Args:
        depth_history:      List of (frame_idx, distance_m) in chronological order.
        fps:                Video framerate — converts frame delta to seconds.
        use_least_squares:  False reverts to the endpoint difference (d0 - d1) / dt.
    """
    if len(depth_history) < 2:
        return 0.0

    if not use_least_squares or len(depth_history) == 2:
        (f0, d0) = depth_history[0]
        (f1, d1) = depth_history[-1]
        dt = (f1 - f0) / fps
        if dt <= 0:
            return 0.0
        raw_v = (d0 - d1) / dt   # positive = object closing in
        return max(0.0, raw_v)   # clamp: don't report negative (retreating) velocities

    # ponytail: least-squares slope, not a Kalman filter. Upgrade to a constant-velocity
    # Kalman (architecture.md §10.1) if residual jitter still causes false reflex overrides.
    times     = [f / fps for f, _ in depth_history]
    distances = [d for _, d in depth_history]
    n = len(times)
    mean_t = sum(times) / n
    mean_d = sum(distances) / n
    denom = sum((t - mean_t) ** 2 for t in times)
    if denom <= 0:
        return 0.0
    slope = sum((t - mean_t) * (d - mean_d) for t, d in zip(times, distances)) / denom
    return max(0.0, -slope)   # depth shrinking → positive closing velocity


def kinetic_score(
    distance_m: float,
    velocity_ms: float,
    class_name: str,
    *,
    bbox_area_px: float | None = None,
    gamma: float = 2.0,
    size_exponent: float = SIZE_EXPONENT,
) -> float:
    """
    Compute the kinetic threat score for one tracked object.

    Formula (K0):  K = class_severity × v^γ / max(d, ε) × relative_size^μ

    With the defaults (γ=2, μ=0) this is exactly the shipped K0,
    `sev × v² / max(d, ε)`. The exponents are parameters so the ablation in
    `evaluation/kinetic_ablation.py` can knock out one term at a time without a
    second copy of the score drifting out of sync with production.

    Args:
        distance_m:    Metric depth of the object in metres.
        velocity_ms:   Closing velocity in m/s (positive = approaching ego).
        class_name:    COCO class name string (e.g. 'car', 'person').
        bbox_area_px:  Bounding-box area in pixels. Only needed when
                       size_exponent != 0.
        gamma:         Velocity exponent. 2.0 = K0.
        size_exponent: Weight on apparent size (A_px / d). 0.0 = term disabled.
                       This term is ~99% predictable from class + depth for
                       labelled objects — projective geometry already fixes the
                       area — so it earns its place mainly on unlabelled
                       obstacles. Let the ablation decide, not the intuition.
    """
    d = max(distance_m, EPSILON)
    score = class_severity(class_name) * (velocity_ms ** gamma) / d
    if size_exponent and bbox_area_px:
        relative_size = (bbox_area_px / d) / SIZE_REFERENCE_PX_PER_M
        score *= relative_size ** size_exponent
    return score


def bearing_label(deg: float) -> str:
    """Convert a bearing (degrees) to a human-readable direction for the Fact Sheet."""
    if deg < -30:
        return "far-left"
    if deg < -10:
        return "left"
    if deg < 10:
        return "ahead"
    if deg < 30:
        return "right"
    return "far-right"


# ── Vectorised batch operations (torch tensors) ──────────────────────────────

def batch_compute_bearing(
    cx_tensor,
    frame_width: int,
    hfov_deg: float = 70.0,
):
    """
    Vectorised bearing for N detections at once.

    Args:
        cx_tensor: 1-D tensor/array of centre-x pixel coordinates (N,).
        frame_width: Frame width in pixels.
        hfov_deg:    Horizontal field-of-view in degrees.

    Returns:
        Tensor/array of bearing values in degrees (N,).
    """
    import torch
    if not isinstance(cx_tensor, torch.Tensor):
        cx_tensor = torch.tensor(cx_tensor, dtype=torch.float32)
    normalised = (cx_tensor - frame_width / 2) / (frame_width / 2)
    return normalised * (hfov_deg / 2)


def batch_kinetic_score(
    distances,
    velocities,
    severity_weights,
):
    """
    Vectorised kinetic score for N detections: K = severity × v² / max(d, ε).

    Args:
        distances:        1-D tensor/array of distances in metres (N,).
        velocities:       1-D tensor/array of closing velocities m/s (N,).
        severity_weights: 1-D tensor/array of class severity weights (N,).

    Returns:
        Tensor/array of kinetic scores (N,).
    """
    import torch
    if not isinstance(distances, torch.Tensor):
        distances = torch.tensor(distances, dtype=torch.float32)
    if not isinstance(velocities, torch.Tensor):
        velocities = torch.tensor(velocities, dtype=torch.float32)
    if not isinstance(severity_weights, torch.Tensor):
        severity_weights = torch.tensor(severity_weights, dtype=torch.float32)
    return severity_weights * (velocities ** 2) / torch.clamp(distances, min=EPSILON)


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Runnable self-check: python src/perception_stack/physics.py"""
    fps = 30.0

    # Clean constant approach: 5.0 m closing at 1 m/s over 10 frames.
    clean = [(i, 5.0 - 1.0 * i / fps) for i in range(10)]
    assert abs(compute_velocity(clean, fps) - 1.0) < 1e-6
    assert abs(compute_velocity(clean, fps, use_least_squares=False) - 1.0) < 1e-6

    # Retreating objects clamp to zero, not negative.
    assert compute_velocity([(i, 2.0 + 0.1 * i) for i in range(10)], fps) == 0.0

    # Too few samples.
    assert compute_velocity([], fps) == 0.0
    assert compute_velocity([(0, 5.0)], fps) == 0.0

    # The point of least squares: with one corrupted depth sample, using the whole
    # window must beat differencing the two endpoints.
    noisy = list(clean)
    noisy[-1] = (noisy[-1][0], noisy[-1][1] - 0.5)      # spike on the final sample
    ls_err       = abs(compute_velocity(noisy, fps) - 1.0)
    endpoint_err = abs(compute_velocity(noisy, fps, use_least_squares=False) - 1.0)
    assert ls_err < endpoint_err, (ls_err, endpoint_err)

    # K0 monotonicity: closer and faster both raise the score.
    assert kinetic_score(2.0, 3.0, "car") > kinetic_score(8.0, 3.0, "car")
    assert kinetic_score(5.0, 6.0, "car") > kinetic_score(5.0, 2.0, "car")
    assert kinetic_score(5.0, 3.0, "bus") > kinetic_score(5.0, 3.0, "person")

    # Mass-derived severity: heavier class ⇒ higher severity, and the person
    # reference is exactly 1.0 by construction.
    assert class_severity("person") == 1.0
    assert class_severity("bus") > class_severity("car") > class_severity("bicycle")
    assert class_severity("unknown-thing") == DEFAULT_SEVERITY
    assert class_severity("pothole") == STATIC_HAZARD_SEVERITY["pothole"]
    # λ must compress the 171× mass range, not pass it through, and must not
    # flatten it away either — the whole point is a destructive-mass bias.
    # Measured on the pure mass law; behaviour multipliers deliberately widen
    # the final table (a mounted stop sign is meant to sit far below a bus).
    pure = [(m / REFERENCE_MASS_KG) ** SEVERITY_LAMBDA for m in CLASS_MASS_KG.values()]
    assert max(CLASS_MASS_KG.values()) / min(CLASS_MASS_KG.values()) > 100
    assert 5.0 < max(pure) / min(pure) < 60.0, max(pure) / min(pure)
    # The behaviour multiplier is a real term, not decoration.
    assert class_severity("motorcycle") > (200.0 / 70.0) ** SEVERITY_LAMBDA

    # Exponent parameters default to production K0.
    assert kinetic_score(4.0, 2.0, "car", gamma=2.0) == kinetic_score(4.0, 2.0, "car")
    assert kinetic_score(4.0, 2.0, "car", gamma=1.0) < kinetic_score(4.0, 2.0, "car")

    # Size term: off by default even when an area is supplied; on, a bigger box
    # at the same distance scores higher.
    assert kinetic_score(4.0, 2.0, "car", bbox_area_px=90_000) == kinetic_score(4.0, 2.0, "car")
    big   = kinetic_score(4.0, 2.0, "car", bbox_area_px=90_000, size_exponent=0.5)
    small = kinetic_score(4.0, 2.0, "car", bbox_area_px=10_000, size_exponent=0.5)
    assert big > small

    # Static objects score zero — see docs/kinetic_score_opinion.md §7.
    assert kinetic_score(1.0, 0.0, "stairs") == 0.0

    # EPSILON guards division at zero distance.
    assert kinetic_score(0.0, 1.0, "person") == 1.0 / EPSILON

    assert compute_bearing(0, 640) == -35.0
    assert compute_bearing(320, 640) == 0.0
    assert bearing_label(-40) == "far-left" and bearing_label(0) == "ahead"

    print("physics.py self-check OK")


if __name__ == "__main__":
    _demo()
