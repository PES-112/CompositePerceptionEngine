"""
envelope.py
===========
The physics safety envelope (docs/architecture.md §2 principle 1, §6.4).

    override  ⇔  confirmed track  ∧  inside the forward walking cone
                 ∧  closing  ∧  distance / closing speed ≤ OVERRIDE_TTC_S

It is v1's Rule 1 (PhysicsVerification: TTC ≤ 1 s → override) with the two
fixes the v1 backlog asked for: the cone check, so an object passing beside
the walker can't fire it, and fused (Kalman-smoothed, ego-motion-free)
velocity, so one noisy depth sample can't either. Nothing learned is
consulted: no model output can suppress or delay an override.

Edge cases are decided here, deliberately and tested
(tests/test_threat_engine.py):
  - TTC exactly at the threshold fires (≤, not <).
  - Zero or negative closing speed never fires (retreating or static-relative).
  - A track coasting through missing depth may still fire for up to
    MAX_COAST_FRAMES predicted frames: losing depth for one frame must not
    silence an imminent collision.
  - Unconfirmed tracks (fewer than MIN_HITS updates) never fire: a single
    detection has no velocity yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.threat_engine.fusion import FusedTrack

OVERRIDE_TTC_S = 1.0       # unchanged from v1 (PhysicsVerification.OVERRIDE_TTC_S)
CONE_HALF_ANGLE_DEG = 30.0 # same cone as the arrival label (evaluation/kinetic_ablation.THETA_DEG)
MIN_CLOSING_MS = 0.2       # below this, "closing" is filter noise
MAX_COAST_FRAMES = 2       # predicted-only frames an override may still be raised on


@dataclass(frozen=True)
class Override:
    track: FusedTrack
    ttc_s: float
    reason: str


def in_cone(track: FusedTrack, half_angle_deg: float = CONE_HALF_ANGLE_DEG) -> bool:
    return track.forward_m > 0 and abs(track.bearing_deg) <= half_angle_deg


def check(track: FusedTrack, *, ttc_threshold_s: float = OVERRIDE_TTC_S,
          half_angle_deg: float = CONE_HALF_ANGLE_DEG) -> Override | None:
    """The override for one track, or None."""
    if not track.confirmed or track.misses > MAX_COAST_FRAMES:
        return None
    if track.closing_speed_ms < MIN_CLOSING_MS or not in_cone(track, half_angle_deg):
        return None
    ttc = track.distance_m / track.closing_speed_ms
    if ttc > ttc_threshold_s:
        return None
    return Override(track, ttc, f"OVERRIDE: {track.class_name} {track.distance_m:.1f} m ahead, "
                                f"closing {track.closing_speed_ms:.1f} m/s, TTC {ttc:.2f} s")


def overrides(tracks: list[FusedTrack], **kwargs) -> list[Override]:
    """Every override this frame, soonest first."""
    found = [o for o in (check(t, **kwargs) for t in tracks) if o is not None]
    return sorted(found, key=lambda o: (o.ttc_s, o.track.distance_m))
