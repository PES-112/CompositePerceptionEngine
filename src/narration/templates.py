"""
templates.py
============
Deterministic, template-based narrator — "phase 1" of SLM-2 per
docs/architecture.md §11.3: turn a NarratorEvent into a short warning with zero
model inference, so the narration layer has a safe, fast, always-available
fallback before any SLM narration is added on top of it.

No ML dependency. This is the layer critical-path alerts should always be able
to fall back to, per the orchestration lanes in architecture.md §11.6 — it must
stay fast and dependency-free so it can never become the reason an alert is late.
"""

from __future__ import annotations

from src.physics_verification.physics_verification import NarratorEvent

# Thresholds are initial defaults, not calibrated — same status as the routing
# thresholds in src/threat_prioritizer/events.py (docs/pending_work.md §10.2).
FAST_CLOSING_MS = 3.0     # m/s — "fast" wording kicks in above this
NEAR_DISTANCE_M = 3.0     # m — "close" wording kicks in below this
BEARING_CENTER_DEG = 15.0
BEARING_FAR_DEG = 45.0


def direction(bearing_deg: float) -> str:
    """Mirrors DetectedObject.bearing's ahead/left/right/far-left/far-right convention
    (src/shared/fact_sheet.py) but as narration-ready words, not a label."""
    mag = abs(bearing_deg)
    side = "left" if bearing_deg < 0 else "right"
    if mag <= BEARING_CENTER_DEG:
        return "ahead"
    if mag >= BEARING_FAR_DEG:
        return f"far {side}"
    return side


def _class_words(object_class: str) -> str:
    return object_class[:1].upper() + object_class[1:]


def narrate(event: NarratorEvent) -> str:
    """
    Produce a short (<=10 word, matching NarratorEvent.to_prompt()'s own budget
    for SLM-2) deterministic warning. Never returns an empty string — the
    critical path always has something to say.
    """
    cls = _class_words(event.object_class)
    d = direction(event.bearing_deg)

    if event.is_override:
        return f"Stop! {cls} very close {d}."
    if event.closing_velocity_ms >= FAST_CLOSING_MS:
        return f"{cls} fast from your {d}."
    if event.distance_m <= NEAR_DISTANCE_M:
        return f"{cls} close, {d}."
    return f"{cls} {d}, {event.distance_m:.0f} meters."


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Runnable self-check: python -m src.narration.templates --self-check"""
    # The exact example from docs/architecture.md §2.7.
    fast_left = NarratorEvent(
        track_id="t1", object_class="motorcycle", distance_m=6.0,
        closing_velocity_ms=8.0, bearing_deg=-30.0, reason="fast closing", is_override=False,
    )
    assert narrate(fast_left) == "Motorcycle fast from your left.", narrate(fast_left)

    override = NarratorEvent(
        track_id="t2", object_class="car", distance_m=1.2,
        closing_velocity_ms=4.0, bearing_deg=2.0, reason="OVERRIDE", is_override=True,
    )
    msg = narrate(override)
    assert msg.startswith("Stop!") and "ahead" in msg, msg

    near_slow = NarratorEvent(
        track_id="t3", object_class="person", distance_m=2.0,
        closing_velocity_ms=0.5, bearing_deg=40.0, reason="near static", is_override=False,
    )
    msg = narrate(near_slow)
    assert "close" in msg and "right" in msg, msg

    far_away = NarratorEvent(
        track_id="t4", object_class="bus", distance_m=25.0,
        closing_velocity_ms=1.0, bearing_deg=50.0, reason="context", is_override=False,
    )
    msg = narrate(far_away)
    assert "far right" in msg and "25" in msg, msg

    assert direction(0.0) == "ahead"
    assert direction(-10.0) == "ahead"       # within center cone
    assert direction(-20.0) == "left"
    assert direction(20.0) == "right"
    assert direction(-50.0) == "far left"
    assert direction(50.0) == "far right"

    # Every message must be non-empty and reasonably short (narration budget).
    for ev in (fast_left, override, near_slow, far_away):
        msg = narrate(ev)
        assert 0 < len(msg.split()) <= 10, msg

    print("templates.py self-check OK")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _demo()
    else:
        print(__doc__)
