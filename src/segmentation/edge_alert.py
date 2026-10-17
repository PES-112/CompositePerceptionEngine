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
"""

from __future__ import annotations

from dataclasses import dataclass

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
