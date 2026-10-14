"""
envelope.py
===========
The physics safety envelope (docs/architecture.md §2 principle 1, §6.4).

    override  ⇔  confirmed track  ∧  ahead and closing
                 ∧  distance / closing speed ≤ OVERRIDE_TTC_S
                 ∧  in the walking corridor now or at that predicted moment

It is v1's Rule 1 (PhysicsVerification: TTC ≤ 1 s → override) with the two
fixes the v1 backlog asked for: a walking-corridor check (1.5 m wide plus the
object's half-width, now or at the predicted moment of contact), so an object
passing beside the walker can't fire it, and fused (Kalman-smoothed,
ego-motion-free) velocity, so one noisy depth sample can't either. Nothing learned is
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
CONE_HALF_ANGLE_DEG = 30.0 # the v1 label's cone; with SANPO's 51-60 deg FOV it spans almost the whole image
# The walking corridor. The arrival label uses 0.75 m (Fruin's comfortable lateral
# spacing); the envelope uses twice that (FHWA's 1.5 m passing width) so it errs
# toward firing. Each object adds its own half-width.
CORRIDOR_WIDTH_M = 0.75
ENVELOPE_CORRIDOR_WIDTH_M = 1.5
CLASS_HALF_WIDTH_M = {
    "person": 0.25, "bicycle": 0.3, "car": 0.9, "motorcycle": 0.4, "bus": 1.25, "truck": 1.25,
    "traffic light": 0.15, "stop sign": 0.15, "fire hydrant": 0.2, "pole": 0.1, "bollard": 0.1,
    "stairs": 1.0, "crosswalk": 1.5, "pothole": 0.3, "puddle": 0.5, "dog": 0.2, "bench": 0.75,
}
DEFAULT_HALF_WIDTH_M = 0.3
MIN_CLOSING_MS = 0.2       # below this, "closing" is filter noise
MAX_COAST_FRAMES = 2       # predicted-only frames an override may still be raised on


@dataclass(frozen=True)
class Override:
    track: FusedTrack
    ttc_s: float
    reason: str


def in_cone(track: FusedTrack, half_angle_deg: float = CONE_HALF_ANGLE_DEG) -> bool:
    return track.forward_m > 0 and abs(track.bearing_deg) <= half_angle_deg


def half_width(class_name: str) -> float:
    return CLASS_HALF_WIDTH_M.get(class_name, DEFAULT_HALF_WIDTH_M)


def in_corridor(lateral_m: float, forward_m: float, class_name: str, width_m: float = CORRIDOR_WIDTH_M) -> bool:
    return forward_m > 0 and abs(lateral_m) <= width_m / 2 + half_width(class_name)


def check(track: FusedTrack, *, ttc_threshold_s: float = OVERRIDE_TTC_S,
          width_m: float = ENVELOPE_CORRIDOR_WIDTH_M) -> Override | None:
    """
    The override for one track, or None. The object must be ahead and closing
    with TTC <= threshold, and inside the walking corridor either now or where
    it will be at that moment (constant relative velocity), so an object
    crossing into the path fires and one passing beside it doesn't.
    """
    if not track.confirmed or track.misses > MAX_COAST_FRAMES:
        return None
    if track.closing_speed_ms < MIN_CLOSING_MS or track.forward_m <= 0:
        return None
    ttc = track.distance_m / track.closing_speed_ms
    if ttc > ttc_threshold_s:
        return None
    lateral_at_ttc = track.lateral_m + track.rel_velocity[0] * ttc
    if not (in_corridor(track.lateral_m, track.forward_m, track.class_name, width_m)
            or abs(lateral_at_ttc) <= width_m / 2 + half_width(track.class_name)):
        return None
    return Override(track, ttc, f"OVERRIDE: {track.class_name} {track.distance_m:.1f} m ahead, "
                                f"closing {track.closing_speed_ms:.1f} m/s, TTC {ttc:.2f} s")


def overrides(tracks: list[FusedTrack], **kwargs) -> list[Override]:
    """Every override this frame, soonest first."""
    found = [o for o in (check(t, **kwargs) for t in tracks) if o is not None]
    return sorted(found, key=lambda o: (o.ttc_s, o.track.distance_m))
