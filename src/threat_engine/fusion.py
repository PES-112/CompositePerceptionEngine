"""
fusion.py
=========
Sensor fusion & tracking (docs/architecture.md §6.2): one constant-velocity
Kalman filter per track, on the ground plane, with the walker's own motion
removed using the camera pose.

Why it exists. Stage 1's closing speed is a least-squares slope over 5 raw
depth samples, measured in the camera frame. Every head turn therefore looks
like every object moving sideways, every step forward looks like every static
object approaching, and the speed is clamped at 0 so retreat is invisible.
Here each detection is placed in a world-fixed ground frame using the pose,
filtered there (a parked car stays still even while the head turns), and only
then expressed relative to the walker:

    camera measurement (depth, bearing)  ──pose──>  world ground point
    Kalman filter per track (x, z, vx, vz) in the world ground plane
    relative to walker: position, signed closing speed, time to zone, CPA

Coordinates. Camera ground frame: x right, z forward (metres). World ground
frame: whatever the pose source uses, projected onto the plane perpendicular
to "up" — the code only ever works with the pose's ground-plane position and
its right/forward unit vectors, so handedness never needs to be assumed.
Without a pose the world frame is the camera frame and the walker is treated
as stationary, which reproduces v1's camera-relative velocities (smoothed and
signed).

Depth here is z-depth along the optical axis, as SANPO's depth maps and the
Stage-1 box median are, so a detection at bearing θ sits at x = z·tan θ.

Cost: a 4-state filter per track, a few 4×4 products per frame (< 1 ms for
dozens of tracks in numpy; trivially less on device).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ── Tuning (declared, not fitted; see docs/methodology.md §A) ────────────────
ACCEL_STD = 1.5          # m/s², white-noise acceleration: pedestrians and cyclists change pace
DEPTH_STD_BASE = 0.05    # m, depth noise at 0 m
DEPTH_STD_QUAD = 0.01    # m per m², stereo depth noise grows ~ distance²
BEARING_STD_DEG = 1.0    # box-centre jitter
INIT_SPEED_STD = 3.0     # m/s, velocity is unknown on the first sighting
GATE_NIS = 11.62         # chi² 2-dof, 99.7%: a depth jump past this is an outlier, not motion
MAX_GATED = 3            # this many outliers in a row = the track really moved (re-ID); restart it
MAX_MISSES = 5           # frames without a measurement before a track is dropped
MIN_HITS = 3             # updates before a track counts as confirmed
EGO_SMOOTHING = 0.5      # EMA weight of the newest walker-velocity sample
MIN_WALK_SPEED = 0.3     # m/s; below this the walking direction falls back to the camera heading


@dataclass(frozen=True)
class EgoPose:
    """Camera pose on the ground plane: position and the camera's right/forward unit vectors (world)."""

    position: tuple[float, float]
    right: tuple[float, float]
    forward: tuple[float, float]

    @classmethod
    def identity(cls) -> "EgoPose":
        return cls((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

    @classmethod
    def from_sanpo(cls, pos_x: float, pos_y: float, pos_z: float,
                   q_x: float, q_y: float, q_z: float, q_w: float,
                   up: tuple[float, float, float] = (0.0, 1.0, 0.0)) -> "EgoPose":
        """
        One row of SANPO's camera_poses.csv. The camera axes follow the ZED
        convention (x right, y up, looking along its own -z). The *world* up
        axis is not the same in every session, so pass the session's `up`
        (sanpo_session_poses() estimates it); the ground plane is the plane
        perpendicular to it.
        """
        rot = _quat_to_matrix(q_x, q_y, q_z, q_w)
        e1, e2 = _ground_basis(np.asarray(up, dtype=float))
        pos = np.array([pos_x, pos_y, pos_z], dtype=float)
        right = rot @ np.array([1.0, 0.0, 0.0])
        forward = rot @ np.array([0.0, 0.0, -1.0])
        return cls((float(pos @ e1), float(pos @ e2)),
                   _unit((float(right @ e1), float(right @ e2))),
                   _unit((float(forward @ e1), float(forward @ e2))))

    def basis(self) -> np.ndarray:
        """2×2 matrix whose columns are the camera's right and forward vectors in world coordinates."""
        return np.column_stack([self.right, self.forward])

    def to_world(self, x: float, z: float) -> np.ndarray:
        return np.asarray(self.position) + self.basis() @ np.array([x, z])


@dataclass(frozen=True)
class Measurement:
    """One perceived object in one frame (the Stage-1 row fields fusion needs)."""

    track_id: str
    class_name: str
    depth_m: float | None
    bearing_deg: float
    confidence: float = 1.0
    row: dict | None = None      # the raw perception row, passed through for the GRU's features


@dataclass(frozen=True)
class FusedTrack:
    """One track after fusion, relative to the walker and their walking direction."""

    track_id: str
    class_name: str
    confidence: float
    forward_m: float             # along the walking direction
    lateral_m: float             # + = right of the walking direction
    distance_m: float
    bearing_deg: float           # relative to the walking direction, + = right
    closing_speed_ms: float      # signed: + approaching, - retreating
    lateral_speed_ms: float      # relative, + = moving right
    rel_velocity: tuple[float, float]   # (lateral, forward) relative velocity, m/s
    position_std_m: float
    speed_std_ms: float
    hits: int
    misses: int
    confirmed: bool
    measured: bool               # updated by a measurement this frame (False = predicted only)
    row: dict | None = None

    @property
    def ttc_s(self) -> float | None:
        """v1's time-to-collision, distance / closing speed; None if not closing."""
        if self.closing_speed_ms <= 0:
            return None
        return self.distance_m / self.closing_speed_ms

    def time_to_zone(self, radius_m: float) -> float | None:
        """Seconds until the constant-velocity relative path first comes within radius_m; 0 if already inside."""
        return time_to_radius(np.array([self.lateral_m, self.forward_m]), np.array(self.rel_velocity), radius_m)

    def closest_approach(self) -> tuple[float, float]:
        """(time ≥ 0, distance) of the closest point of approach on the relative path."""
        p, v = np.array([self.lateral_m, self.forward_m]), np.array(self.rel_velocity)
        vv = float(v @ v)
        t = 0.0 if vv < 1e-9 else max(0.0, -float(p @ v) / vv)
        return t, float(np.linalg.norm(p + v * t))


@dataclass
class _Filter:
    x: np.ndarray                # [X, Z, VX, VZ] world ground plane
    P: np.ndarray
    class_name: str
    confidence: float
    hits: int = 1
    misses: int = 0
    gated: int = 0
    row: dict | None = None
    measured: bool = True


@dataclass
class GroundTracker:
    """Stateful per-track Kalman filtering. Call step() once per processed frame, in order."""

    accel_std: float = ACCEL_STD
    max_misses: int = MAX_MISSES
    min_hits: int = MIN_HITS
    _filters: dict[str, _Filter] = field(default_factory=dict)
    _last_t: float | None = None
    _last_pose: EgoPose | None = None
    _ego_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @property
    def ego_velocity(self) -> np.ndarray:
        return self._ego_velocity.copy()

    def walking_basis(self, pose: EgoPose) -> np.ndarray:
        """Columns: the walker's right and forward directions (world). Forward = direction of travel when walking."""
        speed = float(np.linalg.norm(self._ego_velocity))
        if speed < MIN_WALK_SPEED:
            return pose.basis()
        fwd = self._ego_velocity / speed
        # Right = forward turned the same way the camera's right is from its forward,
        # so the result has the same handedness as the pose.
        cam = pose.basis()
        sign = 1.0 if np.linalg.det(cam) > 0 else -1.0
        right = sign * np.array([fwd[1], -fwd[0]])
        return np.column_stack([right, fwd])

    def step(self, t: float, measurements: list[Measurement], pose: EgoPose | None = None) -> list[FusedTrack]:
        """Advance every track to time t (seconds), fold in this frame's measurements, return all live tracks."""
        pose = pose or EgoPose.identity()
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._update_ego(pose, dt)

        for f in self._filters.values():
            self._predict(f, dt)
            f.measured = False

        for m in measurements:
            if m.depth_m is None or not math.isfinite(m.depth_m) or m.depth_m <= 0:
                continue   # depth missing this frame: the track coasts on its prediction
            z, R = self._measure(m, pose)
            f = self._filters.get(m.track_id)
            if f is None:
                self._filters[m.track_id] = self._new_filter(z, R, m)
                continue
            if not self._update(f, z, R):
                f.gated += 1
                if f.gated >= MAX_GATED:
                    self._filters[m.track_id] = self._new_filter(z, R, m)
                continue
            f.hits += 1
            f.misses = 0
            f.gated = 0
            f.measured = True
            f.class_name, f.confidence, f.row = m.class_name, m.confidence, m.row

        for tid in [tid for tid, f in self._filters.items() if not f.measured]:
            self._filters[tid].misses += 1
            if self._filters[tid].misses > self.max_misses:
                del self._filters[tid]

        self._last_t, self._last_pose = t, pose
        return [self._relative(tid, f, pose) for tid, f in self._filters.items()]

    # ── internals ────────────────────────────────────────────────────────────

    def _update_ego(self, pose: EgoPose, dt: float) -> None:
        if self._last_pose is None or dt <= 0:
            return
        sample = (np.asarray(pose.position) - np.asarray(self._last_pose.position)) / dt
        self._ego_velocity = EGO_SMOOTHING * sample + (1 - EGO_SMOOTHING) * self._ego_velocity

    def _measure(self, m: Measurement, pose: EgoPose) -> tuple[np.ndarray, np.ndarray]:
        """World ground-plane position of a detection and its covariance."""
        depth = float(m.depth_m)
        theta = math.radians(m.bearing_deg)
        x_cam, z_cam = depth * math.tan(theta), depth
        sd_depth = DEPTH_STD_BASE + DEPTH_STD_QUAD * depth ** 2
        sd_lat = math.hypot(depth * math.radians(BEARING_STD_DEG), sd_depth * abs(math.tan(theta)))
        B = pose.basis()
        R = B @ np.diag([sd_lat ** 2, sd_depth ** 2]) @ B.T
        return pose.to_world(x_cam, z_cam), R

    def _new_filter(self, z: np.ndarray, R: np.ndarray, m: Measurement) -> _Filter:
        P = np.zeros((4, 4))
        P[:2, :2] = R
        P[2:, 2:] = np.eye(2) * INIT_SPEED_STD ** 2
        # Start at the walker's own velocity: most objects are static, and a static
        # object moves at -ego velocity relative to the walker, i.e. 0 in the world.
        return _Filter(x=np.array([z[0], z[1], 0.0, 0.0]), P=P, class_name=m.class_name,
                       confidence=m.confidence, row=m.row)

    def _predict(self, f: _Filter, dt: float) -> None:
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.accel_std ** 2
        Q1 = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * q
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = Q1
        Q[np.ix_([1, 3], [1, 3])] = Q1
        f.x = F @ f.x
        f.P = F @ f.P @ F.T + Q

    def _update(self, f: _Filter, z: np.ndarray, R: np.ndarray) -> bool:
        """Kalman update; False (and no change) if the measurement fails the gate."""
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        y = z - H @ f.x
        S = H @ f.P @ H.T + R
        S_inv = np.linalg.inv(S)
        if float(y @ S_inv @ y) > GATE_NIS:
            return False
        K = f.P @ H.T @ S_inv
        f.x = f.x + K @ y
        I_KH = np.eye(4) - K @ H
        f.P = I_KH @ f.P @ I_KH.T + K @ R @ K.T   # Joseph form: stays symmetric positive-definite
        return True

    def _relative(self, tid: str, f: _Filter, pose: EgoPose) -> FusedTrack:
        W = self.walking_basis(pose)
        p = W.T @ (f.x[:2] - np.asarray(pose.position))           # (lateral, forward)
        v = W.T @ (f.x[2:] - self._ego_velocity)
        dist = float(np.linalg.norm(p))
        closing = -float(p @ v) / dist if dist > 1e-6 else 0.0
        lateral_speed = float(v[0])
        pos_cov = W.T @ f.P[:2, :2] @ W
        vel_cov = W.T @ f.P[2:, 2:] @ W
        return FusedTrack(
            track_id=tid,
            class_name=f.class_name,
            confidence=f.confidence,
            forward_m=float(p[1]),
            lateral_m=float(p[0]),
            distance_m=dist,
            bearing_deg=math.degrees(math.atan2(p[0], p[1])),
            closing_speed_ms=closing,
            lateral_speed_ms=lateral_speed,
            rel_velocity=(float(v[0]), float(v[1])),
            position_std_m=float(math.sqrt(max(np.trace(pos_cov), 0.0))),
            speed_std_ms=float(math.sqrt(max(np.trace(vel_cov), 0.0))),
            hits=f.hits,
            misses=f.misses,
            confirmed=f.hits >= self.min_hits,
            measured=f.measured,
            row=f.row,
        )


def sanpo_session_poses(rows: list[dict]) -> list[EgoPose]:
    """
    EgoPose per row of one session's camera_poses.csv (dicts with tracking_state,
    pos_*, q_*). World up = the mean of the camera's up axis over the session
    (the walker's head is upright on average). Rows where visual tracking was
    lost (anything but TrackingState.READY) repeat the last good pose, so the
    walker is treated as standing still rather than teleporting.
    """
    keys = ("pos_x", "pos_y", "pos_z", "q_x", "q_y", "q_z", "q_w")
    vals = [[float(r[k]) for k in keys] for r in rows]
    ready = [str(r.get("tracking_state", "")).endswith("READY") and "NOT" not in str(r.get("tracking_state", ""))
             for r in rows]
    ups = [_quat_to_matrix(*v[3:]) @ np.array([0.0, 1.0, 0.0]) for v, ok in zip(vals, ready) if ok]
    if not ups:
        return [EgoPose.identity()] * len(rows)
    up = np.mean(ups, axis=0)
    up = tuple(up / np.linalg.norm(up))
    out, last = [], None
    for v, ok in zip(vals, ready):
        if ok or last is None:
            last = EgoPose.from_sanpo(*v, up=up)
        out.append(last)
    return out


def time_to_radius(p: np.ndarray, v: np.ndarray, radius: float) -> float | None:
    """Smallest t ≥ 0 with |p + v·t| ≤ radius, or None if the path never gets that close."""
    c = float(p @ p) - radius ** 2
    if c <= 0:
        return 0.0
    a, b = float(v @ v), 2.0 * float(p @ v)
    if a < 1e-12 or b >= 0:
        return None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    return (-b - math.sqrt(disc)) / (2 * a)


def _ground_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane perpendicular to `up`, right-handed with it."""
    up = up / np.linalg.norm(up)
    seed = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = seed - (seed @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, up)
    return e1, e2


def _unit(v: tuple[float, float]) -> tuple[float, float]:
    n = math.hypot(*v)
    if n < 1e-9:
        raise ValueError("camera axis is vertical; cannot project it onto the ground plane")
    return (v[0] / n, v[1] / n)


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
