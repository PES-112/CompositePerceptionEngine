"""
stub.py
=======
Rule-based stand-in for SLM-1, used to prove the full runtime pipeline composes
end-to-end (docs/pending_work.md §5) before a trained model exists. This is
explicitly NOT the Cognitive Layer described in architecture.md §2.5 — it does
no semantic reasoning, no trajectory/intent inference, and reads only the same
kinetic_score the physics side already computes, so it should track the
physics ranking closely by construction. Its purpose is to be a correctly-typed
`SemanticEval` producer so `PhysicsVerification.adjudicate()`,
`src.pipeline.orchestrator`, and `src.narration` can all be exercised together
on real (or synthetic) frames without waiting for training data or a runtime
model. Replace this module, don't extend it, when SLM-1 exists.
"""

from __future__ import annotations

from src.physics_verification.physics_verification import SemanticEval
from src.threat_prioritizer import ThreatFrame

STUB_CONFIDENCE = 0.8   # matches SemanticEval's documented default


def stub_semantic_eval(threat_frame: ThreatFrame) -> SemanticEval | None:
    """
    Picks the highest-kinetic-score object among this frame's cognitive-route
    events as the "semantic" primary threat. Returns None when there are no
    cognitive-route events — PhysicsVerification.adjudicate() already handles
    a missing semantic_eval by falling back to physics alone.
    """
    cognitive = threat_frame.cognitive_events()
    if not cognitive:
        return None

    ranked = sorted(cognitive, key=lambda event: event.kinetic_score, reverse=True)
    primary = ranked[0]
    secondary = ranked[1].track_id if len(ranked) > 1 else None
    scene_state = "object_approaching" if primary.velocity_ms > 0 else "object_static"

    return SemanticEval(
        primary_threat_id=primary.track_id,
        reason=f"[stub] {primary.class_name} at {primary.distance_m or 0:.1f}m, "
               f"K={primary.kinetic_score:.2f} — {primary.reason}",
        secondary_threat_id=secondary,
        scene_state=scene_state,
        confidence=STUB_CONFIDENCE,
    )


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Runnable self-check: python -m src.cognitive_layer.stub --self-check"""
    from src.threat_prioritizer import build_threat_frame

    def row(**overrides):
        base = {
            "frame_idx": 1, "source": "sanpo", "track_id": "p1", "class": "pothole",
            "confidence": 0.9, "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 10, "bbox_y2": 10,
            "cx_px": 5, "bearing_deg": 0, "distance_m": 1.2, "velocity_ms": 0.0,
        }
        base.update(overrides)
        return base

    # No cognitive events -> None, not an error.
    empty_frame = build_threat_frame([row(track_id="b1", **{"class": "bench"}, distance_m=6.0)])
    assert stub_semantic_eval(empty_frame) is None

    # One cognitive event (near-static pothole) -> picks it.
    frame = build_threat_frame([row()])
    eval_ = stub_semantic_eval(frame)
    assert eval_ is not None
    assert eval_.primary_threat_id == "p1"
    assert eval_.secondary_threat_id is None
    assert eval_.scene_state == "object_static"
    assert eval_.confidence == STUB_CONFIDENCE

    # Two cognitive events -> picks the higher-K one as primary, other as secondary.
    two = build_threat_frame([
        row(track_id="p1", distance_m=1.4, velocity_ms=0.0),
        row(track_id="p2", **{"class": "puddle"}, distance_m=1.0, velocity_ms=0.0),
    ])
    eval_ = stub_semantic_eval(two)
    assert eval_.primary_threat_id in ("p1", "p2")
    assert eval_.secondary_threat_id in ("p1", "p2")
    assert eval_.primary_threat_id != eval_.secondary_threat_id

    print("stub.py self-check OK")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _demo()
    else:
        print(__doc__)
