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
    "person":     1.0,
    "bicycle":    1.2,
    "car":        2.0,
    "motorcycle": 1.8,
    "bus":        2.5,
    "truck":      2.5,
    "dog":        0.8,
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

    Formula:  K = class_severity × (velocity_ms²) / max(distance_m, ε)
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
