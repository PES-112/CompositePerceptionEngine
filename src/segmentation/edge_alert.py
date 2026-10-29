"""
edge_alert.py
=============
When to say "Curb ahead." (§6.3, edge ahead). Each segmentation frame gives
the nearest edge straight ahead (context.edge_ahead): where walkable ground
gives way to curb, stairs, road or non-walkable ground. EdgeAlerter turns
that per-frame reading into at most one alert per edge.

Rule (declared, not fitted; constants below):
  announce   curb, stairs and road; non-walkable ground (grass, terrain) is
             reported by edge_ahead but not announced
  confirm    the same announced kind within ALERT_M in CONFIRM_FRAMES
             consecutive segmentation frames
  one alert  per episode: an episode lasts while any announced edge keeps
             being seen; it ends after CLEAR_S without one, and only then can
             the next alert fire

The alert is a scripted phrase on the normal (non-override) lane; the
announcement manager decides when it plays relative to object alerts.

SurfaceTracker does the same for the ground under the walker ("Now on the
road.", "Walking on the sidewalk."): a debounced walker surface, which the
announcement manager speaks on a change and every few seconds of silence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.segmentation.context import EdgeAhead
from src.segmentation.labels import CPE_CLASSES, CURB, ROAD, STAIRS

ALERT_KINDS = (CURB, STAIRS, ROAD)
ALERT_M = 5.0                   # ~3.5 s ahead at 1.4 m/s walking pace
CONFIRM_FRAMES = 2              # 0.4 s at 5 Hz, 0.8 s at 2.5 Hz
CLEAR_S = 2.0


@dataclass(frozen=True)
class EdgeAlert:
    t: float
    kind: str                   # CPE class name: "curb", "stairs" or "road"
    distance_m: float


class EdgeAlerter:
    def __init__(self, kinds: tuple[int, ...] = ALERT_KINDS, alert_m: float = ALERT_M,
                 confirm_frames: int = CONFIRM_FRAMES, clear_s: float = CLEAR_S):
        self.kinds, self.alert_m, self.confirm_frames, self.clear_s = kinds, alert_m, confirm_frames, clear_s
        self._armed = True
        self._last_seen = float("-inf")
        self._run_kind: int | None = None
        self._run = 0

    def step(self, t: float, edge: EdgeAhead | None) -> EdgeAlert | None:
        """One segmentation frame at time t (s)."""
        announced = edge is not None and edge.kind in self.kinds
        if announced:
            self._last_seen = t
        elif t - self._last_seen >= self.clear_s:
            self._armed = True
        near = announced and edge.distance_m <= self.alert_m
        if near and edge.kind == self._run_kind:
            self._run += 1
        elif near:
            self._run_kind, self._run = edge.kind, 1
        else:
            self._run_kind, self._run = None, 0
        if self._armed and self._run >= self.confirm_frames:
            self._armed = False
            return EdgeAlert(t, CPE_CLASSES[edge.kind], edge.distance_m)
        return None


SURFACE_CONFIRM_FRAMES = 3      # 0.6 s at 5 Hz: a one-frame flicker of the walker surface is not a change


class SurfaceTracker:
    """
    The walker's surface (context.walker_surface) for narration, debounced: a
    new class replaces the current one only after SURFACE_CONFIRM_FRAMES
    consecutive frames; frames with no ground visible keep the current one.
    The announcement manager says it on a change and as a reminder in silence.
    """

    def __init__(self, confirm_frames: int = SURFACE_CONFIRM_FRAMES):
        self.confirm_frames = confirm_frames
        self._current: int | None = None
        self._candidate: int | None = None
        self._run = 0

    def step(self, walker: int | None) -> str | None:
        """One segmentation frame; the confirmed surface's CPE class name, or None before the first."""
        if walker is not None and walker != self._current:
            self._run = self._run + 1 if walker == self._candidate else 1
            self._candidate = walker
            if self._run >= self.confirm_frames:
                self._current, self._candidate, self._run = walker, None, 0
        elif walker is not None:
            self._candidate, self._run = None, 0
        return None if self._current is None else CPE_CLASSES[self._current]


# ── Edge memory (methodology §4.24) ──────────────────────────────────────────
#
# A head-mounted camera sees no ground closer than ~2.6 m, and edges are usually first seen 6-10 m
# ahead, so a rule that needs the edge confirmed inside ALERT_M has a window of ~1.5 s and mostly
# misses it (§6.4: 18% of step-offs warned with SANPO's own masks). EdgeMemoryAlerter anchors each
# sighting on the ground plane with the camera pose, merges repeat sightings, and dead-reckons the
# walker toward it, so an edge can be confirmed far away and announced by time-to-reach, including
# after it has left the image.
MEMORY_MAX_M = 10.0             # sightings up to this far are remembered (context.MAX_EDGE_M)
MATCH_M = 1.5                   # a sighting within this of a remembered edge of the same kind is that edge
TTR_ALERT_S = 3.0               # announce when the walker would reach it within this ...
LATERAL_M = 1.5                 # ... and it lies within this of the walking line
MIN_WALK_MS = 0.3               # slower: standing, so only the distance rule (ALERT_M) applies
FORGET_S = 8.0                  # an edge not seen for this long is dropped
VELOCITY_SMOOTHING = 0.5        # EMA weight of the newest pose-difference velocity


@dataclass
class _Remembered:
    kind: int
    xy: np.ndarray
    sightings: int
    last_seen: float
    announced: bool = False


class EdgeMemoryAlerter:
    """
    EdgeAlerter with a ground-plane memory. Per segmentation frame: the edge ahead (as EdgeAlerter)
    and the camera pose (fusion.EgoPose). An edge is confirmed by confirm_frames sightings within
    MATCH_M of each other in the world (not necessarily consecutive), and announced once when it is
    ahead on the walking line and either within alert_m or reached within ttr_s at the current speed.
    At most one alert per clear_s, as EdgeAlerter's one-per-episode rule.
    """

    def __init__(self, kinds: tuple[int, ...] = ALERT_KINDS, alert_m: float = ALERT_M,
                 confirm_frames: int = CONFIRM_FRAMES, ttr_s: float = TTR_ALERT_S, clear_s: float = CLEAR_S):
        self.kinds, self.alert_m, self.confirm_frames, self.ttr_s, self.clear_s = (
            kinds, alert_m, confirm_frames, ttr_s, clear_s)
        self._memory: list[_Remembered] = []
        self._last: tuple[float, np.ndarray] | None = None
        self._velocity = None
        self._last_alert = float("-inf")

    def _update_velocity(self, t: float, pos: np.ndarray) -> None:
        if self._last is not None and t > self._last[0]:
            v = (pos - self._last[1]) / (t - self._last[0])
            self._velocity = v if self._velocity is None else \
                VELOCITY_SMOOTHING * v + (1 - VELOCITY_SMOOTHING) * self._velocity
        self._last = (t, pos)
        if self._velocity is None:
            self._velocity = np.zeros(2)

    def step(self, t: float, edge: EdgeAhead | None, pose) -> EdgeAlert | None:
        pos, forward = np.asarray(pose.position, float), np.asarray(pose.forward, float)
        self._update_velocity(t, pos)
        if edge is not None and edge.kind in self.kinds and edge.distance_m <= MEMORY_MAX_M:
            xy = pos + edge.distance_m * forward
            near = [m for m in self._memory if m.kind == edge.kind and np.linalg.norm(m.xy - xy) <= MATCH_M]
            if near:
                m = min(near, key=lambda m: np.linalg.norm(m.xy - xy))
                m.xy = (m.xy * m.sightings + xy) / (m.sightings + 1)
                m.sightings += 1
                m.last_seen = t
            else:
                self._memory.append(_Remembered(edge.kind, xy, 1, t))
        self._memory = [m for m in self._memory if t - m.last_seen <= FORGET_S]

        speed = float(np.linalg.norm(self._velocity))
        heading = self._velocity / speed if speed >= MIN_WALK_MS else forward
        due = []
        for m in self._memory:
            if m.announced or m.sightings < self.confirm_frames:
                continue
            rel = m.xy - pos
            along = float(rel @ heading)
            lateral = abs(float(rel[0] * heading[1] - rel[1] * heading[0]))
            if along <= 0 or lateral > LATERAL_M:
                continue
            if along <= self.alert_m or (speed >= MIN_WALK_MS and along / speed <= self.ttr_s):
                due.append((along, m))
        if not due or t - self._last_alert < self.clear_s:
            return None
        along, m = min(due, key=lambda d: d[0])
        for _, other in due:
            other.announced = True                    # one alert covers every edge due now
        self._last_alert = t
        return EdgeAlert(t, CPE_CLASSES[m.kind], along)
