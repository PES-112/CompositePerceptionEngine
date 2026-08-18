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
# Higher = more dangerous when combined with velocity/distance in kinetic score.
CLASS_SEVERITY: dict[str, float] = {
    "person":              1.0,
    "bicycle":             1.2,
    "car":                 2.0,
    "motorcycle":          1.8,
    "bus":                 2.5,
    "truck":               2.5,
    "traffic light":       0.4,
    "stop sign":           0.4,
    "fire hydrant":        1.0,
    "dog":                 1.4,
    "bench":               0.9,
    "pole":                1.3,
    "bollard":             1.4,
    "stairs":              2.0,
    "crosswalk":           0.5,
    "pothole":             1.8,
    "puddle":              1.1,
    "unlabeled_obstacle":  1.2,
}
DEFAULT_SEVERITY = 1.0
EPSILON = 0.5   # metres — prevents division by zero for very close objects


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


def kinetic_score(distance_m: float, velocity_ms: float, class_name: str) -> float:
    """
    Compute the kinetic threat score for one tracked object.

    Formula (K0):  K = class_severity × (velocity_ms²) / max(distance_m, ε)
    Higher K → higher threat level.

    Args:
        distance_m:  Metric depth of the object in metres.
        velocity_ms: Closing velocity in m/s (positive = approaching ego).
        class_name:  COCO class name string (e.g. 'car', 'person').
    """
    severity = CLASS_SEVERITY.get(class_name, DEFAULT_SEVERITY)
    return severity * (velocity_ms ** 2) / max(distance_m, EPSILON)


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
