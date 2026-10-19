"""
kalman_arrival.py
=================
Constant-velocity Kalman arrival predictor: the probability that a track's
relative path enters the walking corridor within the horizon, integrated over
the filter's own uncertainty (docs/architecture.md §6.4; the "Kalman CV
extrapolation" rung of the baseline ladder).

PhysicsArrivalPredictor (priority.py) extrapolates the mean state only, so its
P(arrive) is ~0 or ~1 and its ignore threshold barely moves the operating
point. This predictor draws N fixed samples from the track's relative state
covariance (FusedTrack.rel_cov), runs each forward at constant velocity, and
reports the share that enter the zone the arrival label uses:

    0 < forward <= hazard_m  and  |lateral| <= width_m / 2 + half_width(class)

within horizon_s; expected arrival time is the mean entry time of the samples
that enter. A track whose velocity is still uncertain gets a middling
probability instead of a confident yes or no, so the threshold becomes a real
knob. The draws are fixed at construction (seeded), so replays are
deterministic.
"""

from __future__ import annotations

import numpy as np

from src.threat_engine.envelope import CORRIDOR_WIDTH_M, half_width
from src.threat_engine.fusion import FusedTrack
from src.threat_engine.priority import HAZARD_M, HORIZON_S, Arrival

N_SAMPLES = 128


class KalmanArrivalPredictor:
    def __init__(self, hazard_m: float = HAZARD_M, horizon_s: float = HORIZON_S, width_m: float = CORRIDOR_WIDTH_M,
                 step_s: float = 0.2, n_samples: int = N_SAMPLES, seed: int = 0):
        self.hazard_m, self.horizon_s, self.width_m = hazard_m, horizon_s, width_m
        self.steps = np.arange(0.0, horizon_s + 1e-9, step_s)
        self._z = np.random.default_rng(seed).standard_normal((n_samples, 4))
        self._z[0] = 0.0   # one draw is always the mean, as the point predictor would see it

    def entry(self, t: FusedTrack) -> tuple[float, float]:
        """(P(enters the corridor within the horizon), expected entry time in s)."""
        mean = np.array([t.lateral_m, t.forward_m, *t.rel_velocity])
        if t.rel_cov is None:
            states = mean[None]
        else:
            L = np.linalg.cholesky(np.asarray(t.rel_cov) + np.eye(4) * 1e-9)
            states = mean + self._z @ L.T
        lat = states[:, 0:1] + states[:, 2:3] * self.steps
        fwd = states[:, 1:2] + states[:, 3:4] * self.steps
        inside = (fwd > 0) & (fwd <= self.hazard_m) & (np.abs(lat) <= self.width_m / 2 + half_width(t.class_name))
        hit = inside.any(axis=1)
        never = self.horizon_s * 1.25
        if not hit.any():
            return 0.0, never
        first = self.steps[inside[hit].argmax(axis=1)]
        return float(hit.mean()), float(first.mean())

    def predict(self, frame_idx: int, tracks: list[FusedTrack]) -> dict[str, Arrival]:
        return {t.track_id: Arrival(*self.entry(t)) for t in tracks}
