"""
announcer.py
============
The announcement manager (docs/architecture.md §6.5): decides *whether* to
speak this frame, separately from *what* is most dangerous.

Rules, in order:
  1. Override preempts. A safety-envelope override is always announced at once
     and interrupts anything playing. It is only rate-limited per track
     (OVERRIDE_REPEAT_S), so a looming object isn't re-announced every frame.
  2. Busy. While a clip is playing, non-override messages wait; one that still
     can't start within STALE_S of being chosen is dropped, never queued — a
     late hazard message is worse than none.
  3. Hysteresis. The announced object only changes if the new top threat beats
     the current one's priority by SWITCH_MARGIN (×). This is where the GRU's
     steadier top pick pays off.
  4. Cooldown. The same track isn't repeated within COOLDOWN_S unless its
     priority has risen by ESCALATION (×) since it was last announced.

Times are seconds on the replay (or device) clock. The manager never looks at
labels, so the same parameters run in replay and on the phone; they are
declared values, to be tuned on train sessions only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.threat_engine.envelope import Override
from src.threat_engine.priority import RankedThreat

COOLDOWN_S = 4.0
ESCALATION = 1.5
SWITCH_MARGIN = 1.25
OVERRIDE_REPEAT_S = 2.0
STALE_S = 0.6
CLIP_S = 1.2            # typical pre-rendered phrase length

Kind = Literal["override", "threat"]


@dataclass(frozen=True)
class Announcement:
    t: float
    kind: Kind
    track_id: str
    class_name: str
    priority: float
    reason: str


@dataclass
class AnnouncementManager:
    cooldown_s: float = COOLDOWN_S
    escalation: float = ESCALATION
    switch_margin: float = SWITCH_MARGIN
    override_repeat_s: float = OVERRIDE_REPEAT_S
    stale_s: float = STALE_S
    clip_s: float = CLIP_S
    _busy_until: float = -1e9
    _current: str | None = None
    _last_said: dict[str, tuple[float, float]] = field(default_factory=dict)   # track -> (t, priority)
    _last_override: dict[str, float] = field(default_factory=dict)
    _pending: tuple[float, RankedThreat] | None = None

    def step(self, t: float, overrides: list[Override], ranked: list[RankedThreat]) -> Announcement | None:
        """At most one announcement for time t."""
        for o in overrides:
            tid = o.track.track_id
            if t - self._last_override.get(tid, -1e9) >= self.override_repeat_s:
                self._last_override[tid] = t
                self._busy_until = t + self.clip_s            # interrupts whatever was playing
                self._pending = None
                self._current = tid
                return Announcement(t, "override", tid, o.track.class_name, float("inf"), o.reason)

        candidate = self._choose(t, ranked)
        if candidate is not None and (self._pending is None or candidate.track.track_id != self._pending[1].track.track_id):
            self._pending = (t, candidate)
        if self._pending is None:
            return None
        chosen_at, threat = self._pending
        if t - chosen_at > self.stale_s:
            self._pending = None                              # couldn't play in time: drop, don't queue
            return None
        if t < self._busy_until:
            return None
        self._pending = None
        tid = threat.track.track_id
        self._busy_until = t + self.clip_s
        self._current = tid
        self._last_said[tid] = (t, threat.priority)
        return Announcement(t, "threat", tid, threat.track.class_name, threat.priority,
                            f"priority {threat.priority:.2f}, P(arrive) {threat.arrival.p_arrive:.2f}, "
                            f"~{threat.arrival.expected_s:.1f} s")

    def _choose(self, t: float, ranked: list[RankedThreat]) -> RankedThreat | None:
        if not ranked:
            return None
        top = ranked[0]
        tid = top.track.track_id
        if self._current is not None and tid != self._current:
            current = next((r for r in ranked if r.track.track_id == self._current), None)
            if current is not None and top.priority < current.priority * self.switch_margin:
                top, tid = current, self._current             # hysteresis: stay on the current object
        last = self._last_said.get(tid)
        if last is not None and t - last[0] < self.cooldown_s and top.priority < last[1] * self.escalation:
            return None
        return top
