"""
orchestrator.py
================
The one thing that did not exist anywhere in the repo before this: a single
function that runs one processed frame through every runtime component that
has been built — Threat Prioritizer -> (stub) Cognitive Layer -> Physics
Verification -> Narration — and returns what would actually reach the user.

`src/reflex_layer/reflex.py` deliberately only calls PhysicsVerification with
reflex-route events (its docstring: "Cognitive events are intentionally left
for the SLM path"). Nothing before this module ever built that other path, so
a frame containing only cognitive-route events (e.g. a near-static pothole)
previously produced no PhysicsVerification call and no narrator event at all,
regardless of what a Cognitive Layer might have said. This module is that
missing composition: it builds the physics ranking from every non-ignore
object in the frame (reflex- and cognitive-route alike, matching
architecture.md §2.6's "Compares SLM-1 semantic eval vs Raw Kinetic Score"),
gets a SemanticEval from the Cognitive Layer (stub for now — see
src/cognitive_layer/stub.py) only when the frame has cognitive-route events,
and adjudicates exactly once per frame.

Ignored objects never enter PhysicsVerification here, on purpose — that is
the entire point of the "ignore" lane, and `ThreatFrame.reflex_results`
(unlike this module) does not filter them out, so do not substitute that
property in here without re-adding the filter below.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.cognitive_layer.stub import stub_semantic_eval
from src.narration.pipeline import NarrationOutput, NarrationPipeline
from src.physics_verification.physics_verification import (
    NarratorEvent,
    PhysicsVerification,
    SemanticEval,
)
from src.threat_prioritizer import DEFAULT_REFLEX_TTC_S, ThreatFrame


@dataclass(frozen=True)
class FrameResult:
    frame_idx: int
    semantic_eval: SemanticEval | None
    narrator_event: NarratorEvent | None
    reward: float
    narration: NarrationOutput | None

    @property
    def has_alert(self) -> bool:
        return self.narrator_event is not None


def process_frame(
    threat_frame: ThreatFrame,
    narration: NarrationPipeline,
    *,
    target_lang: str = "en",
    reflex_ttc_s: float = DEFAULT_REFLEX_TTC_S,
) -> FrameResult:
    non_ignore_ids = {event.track_id for event in threat_frame.events if event.route != "ignore"}
    reflex_results = tuple(
        obj.to_reflex_result(override_ttc_s=reflex_ttc_s)
        for obj in threat_frame.objects
        if obj.track_id in non_ignore_ids
    )

    semantic_eval = stub_semantic_eval(threat_frame)

    judge = PhysicsVerification(threat_frame.object_registry)
    narrator_event, reward = judge.adjudicate(list(reflex_results), semantic_eval)

    narration_output = narration.process(narrator_event, target_lang=target_lang) if narrator_event else None

    return FrameResult(
        frame_idx=threat_frame.frame_idx,
        semantic_eval=semantic_eval,
        narrator_event=narrator_event,
        reward=reward,
        narration=narration_output,
    )


def process_frames(
    frames: list[ThreatFrame] | tuple[ThreatFrame, ...],
    narration: NarrationPipeline,
    *,
    target_lang: str = "en",
    reflex_ttc_s: float = DEFAULT_REFLEX_TTC_S,
) -> list[FrameResult]:
    return [
        process_frame(frame, narration, target_lang=target_lang, reflex_ttc_s=reflex_ttc_s)
        for frame in frames
    ]


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Runnable self-check: python -m src.pipeline.orchestrator --self-check"""
    from src.narration.tts import CachedClipTTS
    from src.threat_prioritizer import build_threat_frame

    def row(**overrides):
        base = {
            "frame_idx": 1, "source": "sanpo", "track_id": "7", "class": "bicycle",
            "confidence": 0.9, "bbox_x1": 10, "bbox_y1": 20, "bbox_x2": 60, "bbox_y2": 100,
            "cx_px": 35, "bearing_deg": -5, "distance_m": 3.0, "velocity_ms": 1.0,
        }
        base.update(overrides)
        return base

    narration = NarrationPipeline(tts=CachedClipTTS(fallback=b"BEEP"),
                                   critical_tts=CachedClipTTS(fallback=b"CRITICAL_BEEP"))

    # Reflex-only frame (TTC override) — must alert, no cognitive/semantic path
    # touched at all.
    reflex_frame = build_threat_frame([row(distance_m=2.0, velocity_ms=3.0)])
    result = process_frame(reflex_frame, narration)
    assert result.has_alert
    assert result.semantic_eval is None
    assert result.narrator_event.is_override
    assert result.narration is not None and result.narration.lane == "critical"

    # Cognitive-only frame (near-static pothole) — this is exactly the gap
    # described in the module docstring: must now alert via the stub semantic
    # path, where before this module existed, nothing in the repo would call
    # PhysicsVerification for it at all.
    cognitive_frame = build_threat_frame([
        row(track_id="p1", **{"class": "pothole"}, distance_m=1.2, velocity_ms=0.0)
    ])
    result = process_frame(cognitive_frame, narration)
    assert result.semantic_eval is not None
    assert result.semantic_eval.primary_threat_id == "p1"
    assert result.has_alert, "cognitive-only frame produced no alert — the gap this module fixes regressed"
    assert result.narrator_event.track_id == "p1"
    assert result.narration is not None and result.narration.lane == "narration"

    # All-ignore frame (far bench) — must stay silent, never call the judge
    # with a below-threshold object.
    ignore_frame = build_threat_frame([
        row(track_id="b1", **{"class": "bench"}, distance_m=6.0, velocity_ms=0.0)
    ])
    result = process_frame(ignore_frame, narration)
    assert not result.has_alert
    assert result.narration is None

    print("orchestrator.py self-check OK")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _demo()
    else:
        print(__doc__)
