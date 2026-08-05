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


def compute_velocity(depth_history: list[tuple[int, float]], fps: float) -> float:
    """
    Estimate closing velocity (m/s) from a rolling window of (frame_idx, distance_m) pairs.

    Positive return value means the object is APPROACHING (depth decreasing).
    Returns 0.0 if fewer than 2 history samples exist or the object is moving away.

    Args:
        depth_history:  List of (frame_idx, distance_m) in chronological order.
        fps:            Video framerate — used to convert frame delta to seconds.
    """
    if len(depth_history) < 2:
        return 0.0
    (f0, d0) = depth_history[0]
    (f1, d1) = depth_history[-1]
    dt = (f1 - f0) / fps
    if dt <= 0:
        return 0.0
    raw_v = (d0 - d1) / dt   # positive = object closing in
    return max(0.0, raw_v)   # clamp: don't report negative (retreating) velocities


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


# ── Kinetic Score Candidate Formulas (K1–K5) ──────────────────────────────────
# These are evaluated-only alternatives to the production K0 formula.
# Run evaluation/kinetic_score_comparison.py to benchmark all candidates.
# DO NOT replace kinetic_score() (K0) until the benchmark validates a winner.

import math as _math


def kinetic_score_k1_ttc(
    distance_m: float, velocity_ms: float, class_name: str
) -> float:
    """
    K1 — TTC-primary: sev × clamp(1/TTC, 0, 10).

    Directly models the time budget available to react.
    Objects with TTC > 10 s are clamped to 0; stationary objects return 0.
    Advantage: linearly proportional to urgency, no quadratic explosion.
    """
    if velocity_ms <= 0:
        return 0.0
    ttc = distance_m / velocity_ms
    severity = CLASS_SEVERITY.get(class_name, DEFAULT_SEVERITY)
    return severity * min(1.0 / ttc, 10.0)


def kinetic_score_k2_linear(
    distance_m: float, velocity_ms: float, class_name: str
) -> float:
    """
    K2 — Linear velocity: sev × v / max(d, ε).

    Same shape as K0 but linear in velocity instead of quadratic.
    Less explosive at high speeds; slower objects penalised more fairly.
    """
    severity = CLASS_SEVERITY.get(class_name, DEFAULT_SEVERITY)
    return severity * velocity_ms / max(distance_m, EPSILON)


def kinetic_score_k3_quad_distance(
    distance_m: float, velocity_ms: float, class_name: str
) -> float:
    """
    K3 — Quadratic distance decay: sev × v² / (d² + ε).

    Decays much faster with distance than K0.  Creates a wider 'safe' zone
    at moderate distances and concentrates high scores in the close range.
    """
    severity = CLASS_SEVERITY.get(class_name, DEFAULT_SEVERITY)
    return severity * (velocity_ms ** 2) / (distance_m ** 2 + EPSILON)


def kinetic_score_k4_hybrid(
    distance_m: float, velocity_ms: float, class_name: str
) -> float:
    """
    K4 — Hybrid: 0.5 × K0 + 0.5 × (sev × clamp(10/TTC, 0, 10)).

    Blends the momentum signal (K0) with a time-budget signal (K1 variant).
    Balances between energy-based danger and reaction-time urgency.
    Recommended candidate for pedestrian navigation.
    """
    momentum = kinetic_score(distance_m, velocity_ms, class_name)  # K0
    severity = CLASS_SEVERITY.get(class_name, DEFAULT_SEVERITY)
    if velocity_ms <= 0:
        ttc_term = 0.0
    else:
        ttc = distance_m / velocity_ms
        ttc_term = severity * min(10.0 / ttc, 10.0)
    return 0.5 * momentum + 0.5 * ttc_term


def kinetic_score_k5_sigmoid(
    distance_m: float,
    velocity_ms: float,
    class_name: str,
    *,
    beta: float = 2.0,
) -> float:
    """
    K5 — Sigmoid-normalised: sev × sigmoid(v/max(d,ε) − β).

    Output is bounded to [0, severity] — no unbounded explosion.
    β is a calibration offset (default 2.0); tune via the benchmark.
    Best for scenarios where a hard ceiling on K is desirable.
    """
    severity = CLASS_SEVERITY.get(class_name, DEFAULT_SEVERITY)
    x = velocity_ms / max(distance_m, EPSILON) - beta
    return severity / (1.0 + _math.exp(-x))


# ── Formula dispatcher ────────────────────────────────────────────────────────

_KINETIC_FORMULAS: dict = {
    "K0": kinetic_score,
    "K1": kinetic_score_k1_ttc,
    "K2": kinetic_score_k2_linear,
    "K3": kinetic_score_k3_quad_distance,
    "K4": kinetic_score_k4_hybrid,
    "K5": kinetic_score_k5_sigmoid,
}


def kinetic_score_all(
    distance_m: float, velocity_ms: float, class_name: str
) -> dict[str, float]:
    """
    Compute all candidate kinetic scores for one object.

    Returns a dict mapping formula name → score, e.g.:
        {"K0": 8.4, "K1": 3.2, "K2": 1.1, "K3": 12.0, "K4": 5.8, "K5": 0.9}

    Used by evaluation/kinetic_score_comparison.py and notebooks.
    """
    return {
        name: fn(distance_m, velocity_ms, class_name)
        for name, fn in _KINETIC_FORMULAS.items()
    }


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
