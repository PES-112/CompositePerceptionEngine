"""
Deterministic Reflex Layer bridge for CPE runtime events.

The Reflex Layer is intentionally small: it only consumes ThreatFrame objects
that already contain YOLO + depth + tracking + physics enrichment, then forwards
reflex-route events into PhysicsVerification. It never waits for SLM output.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.physics_verification.physics_verification import (
    NarratorEvent,
    PhysicsVerification,
    ReflexResult,
    SemanticEval,
)
from src.threat_prioritizer import DEFAULT_REFLEX_TTC_S, ThreatFrame


@dataclass(frozen=True)
class ReflexCycleResult:
    """Output from one Reflex Layer cycle."""

    narrator_event: NarratorEvent | None
    reward: float
    reflex_results: tuple[ReflexResult, ...]
    object_registry: dict[str, dict]
    override_active: bool
    reason: str

    @property
    def has_alert(self) -> bool:
        return self.narrator_event is not None


def process_reflex_frame(
    threat_frame: ThreatFrame,
    *,
    semantic_eval: SemanticEval | None = None,
    reflex_ttc_s: float = DEFAULT_REFLEX_TTC_S,
) -> ReflexCycleResult:
    """
    Process one ThreatFrame through the hard real-time reflex path.

    Only `route="reflex"` events are passed into PhysicsVerification. Cognitive
    events are intentionally left for the SLM path so static nearby hazards like
    potholes do not produce immediate collision overrides.
    """
    reflex_track_ids = {event.track_id for event in threat_frame.reflex_events()}
    if not reflex_track_ids:
        return ReflexCycleResult(
            narrator_event=None,
            reward=0.0,
            reflex_results=(),
            object_registry=threat_frame.object_registry,
            override_active=False,
            reason="no reflex-route events",
        )

    reflex_results = tuple(
        obj.to_reflex_result(override_ttc_s=reflex_ttc_s)
        for obj in threat_frame.objects
        if obj.track_id in reflex_track_ids
    )
    override_active = any(result.override for result in reflex_results)

    judge = PhysicsVerification(threat_frame.object_registry)
    narrator_event, reward = judge.adjudicate(list(reflex_results), semantic_eval)
    return ReflexCycleResult(
        narrator_event=narrator_event,
        reward=reward,
        reflex_results=reflex_results,
        object_registry=threat_frame.object_registry,
        override_active=override_active,
        reason="override" if override_active else "physics fallback",
    )


def process_reflex_frames(
    frames: list[ThreatFrame] | tuple[ThreatFrame, ...],
    *,
    reflex_ttc_s: float = DEFAULT_REFLEX_TTC_S,
) -> list[ReflexCycleResult]:
    """Process multiple ThreatFrame objects in order."""
    return [
        process_reflex_frame(frame, reflex_ttc_s=reflex_ttc_s)
        for frame in frames
    ]
