"""
physics_verification.py  —  The Judge

Arbitrates between:
  A) Reflex Layer raw kinetic score + TTC (deterministic physics)
  B) Cognitive Layer SLM-1 semantic evaluation (neural reasoning)

Selects the NarratorEvent to be sent to SLM-2.
Also generates the RL reward signal for SLM-1 training.
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ReflexResult:
    """Output from the Reflex Layer."""
    track_id: str
    kinetic_score: float
    ttc_s: Optional[float]
    override: bool          # True if TTC < OVERRIDE_THRESHOLD


@dataclass
class SemanticEval:
    """Output from the Cognitive Layer (SLM-1)."""
    primary_threat_id: str
    reason: str
    secondary_threat_id: Optional[str]
    scene_state: str
    confidence: float       # [0, 1] — parsed from SLM-1 output or default 0.8


@dataclass
class NarratorEvent:
    """Structured event passed to SLM-2 (Narrator)."""
    track_id: str
    object_class: str
    distance_m: float
    closing_velocity_ms: float
    bearing_deg: float
    reason: str
    is_override: bool

    def to_prompt(self) -> str:
        """Format for SLM-2 input."""
        direction = "left" if self.bearing_deg < -15 else "right" if self.bearing_deg > 15 else "center"
        return (
            f"Object: {self.object_class} | {self.distance_m:.1f}m away | "
            f"closing at {self.closing_velocity_ms:.1f}m/s | direction: {direction}.\n"
            f"Context: {self.reason}\n"
            f"Generate a short navigation warning (max 10 words):"
        )


# Reward constants
REWARD_CORRECT       = +100.0
REWARD_NOVEL_CATCH   = +50.0
PENALTY_MISSED       = -200.0
PENALTY_OVERRIDE_MISS= -500.0


class PhysicsVerification:
    """
    The Judge — merges deterministic physics with SLM-1 semantic evaluation.
    """

    # Thresholds (calibrate from dataset)
    OVERRIDE_TTC_S = 1.0            # Hard override threshold
    HIGH_K_THRESHOLD = 5.0          # High kinetic score threshold

    def __init__(self, object_registry: dict):
        """
        Args:
            object_registry: maps track_id → {class, distance_m, velocity_ms, bearing_deg}
                             Populated by the Perception Stack each frame.
        """
        self.object_registry = object_registry

    def adjudicate(
        self,
        reflex_results: list[ReflexResult],
        semantic_eval: Optional[SemanticEval],
        future_reflex_results: Optional[list[ReflexResult]] = None,
    ) -> tuple[NarratorEvent, float]:
        """
        Core adjudication logic.

        Args:
            reflex_results: Physics score at T=0
            semantic_eval: SLM-1 output at T=0
            future_reflex_results: Physics score at T=future (e.g. +2s).
                                   Used only during PPO training to teach semantics.

        Returns:
            (NarratorEvent, rl_reward)
            NarratorEvent → sent to SLM-2 (Narrator)
            rl_reward     → logged for SLM-1 PPO training
        """
        # Sort reflex results by kinetic score descending
        sorted_reflex = sorted(reflex_results, key=lambda r: r.kinetic_score, reverse=True)
        top_physics = sorted_reflex[0] if sorted_reflex else None

        # Determine the ground truth for REWARDS (Future kinetic grounding)
        if future_reflex_results:
            sorted_future = sorted(future_reflex_results, key=lambda r: r.kinetic_score, reverse=True)
            reward_target = sorted_future[0] if sorted_future else top_physics
        else:
            reward_target = top_physics

        # --- RULE 1: Override (TTC < 1.0s) ---
        override_event = next((r for r in sorted_reflex if r.override), None)
        if override_event:
            logger.warning("OVERRIDE triggered: track=%s TTC=%.2f",
                           override_event.track_id, override_event.ttc_s or -1)
            event = self._build_event(override_event.track_id, "OVERRIDE: immediate collision risk", is_override=True)
            # RL penalty if SLM-1 missed this
            reward = REWARD_CORRECT
            if semantic_eval and semantic_eval.primary_threat_id != override_event.track_id:
                reward = PENALTY_OVERRIDE_MISS
                logger.error("SLM-1 MISSED an active OVERRIDE — reward=%.0f", reward)
            return event, reward

        # --- If no semantic eval available, fall back to physics ---
        if semantic_eval is None or top_physics is None or reward_target is None:
            if top_physics:
                event = self._build_event(top_physics.track_id, "Physics fallback — no SLM output", False)
                return event, 0.0
            return None, 0.0  # type: ignore
        
        # Type checker assertions
        assert reward_target is not None
        assert semantic_eval is not None

        # --- RULE 2: Perfect agreement (evaluated against reward_target) ---
        if semantic_eval.primary_threat_id == reward_target.track_id:
            event = self._build_event(top_physics.track_id, semantic_eval.reason, False)
            return event, REWARD_CORRECT

        # --- RULE 3: Divergence — SLM-1 picked a different object ---
        slm_k_future = self._get_k_for(semantic_eval.primary_threat_id, future_reflex_results or reflex_results)
        physics_k_future = reward_target.kinetic_score

        logger.info("Divergence: SLM=%s (Future K=%.2f) vs Physics=%s (Future K=%.2f)",
                    semantic_eval.primary_threat_id, slm_k_future,
                    reward_target.track_id, physics_k_future)

        if slm_k_future > 0 and slm_k_future * semantic_eval.confidence > physics_k_future * 0.8:
            # SLM found something physics undervalued at T=0, but became dangerous at T=future
            event = self._build_event(semantic_eval.primary_threat_id, semantic_eval.reason, False)
            return event, REWARD_NOVEL_CATCH
        else:
            # Physics wins; SLM missed the true future threat
            event = self._build_event(top_physics.track_id, "Physics override: higher kinetic risk", False)
            reward = PENALTY_MISSED if physics_k_future > self.HIGH_K_THRESHOLD else 0.0
            return event, reward

    def _get_k_for(self, track_id: str, reflex_results: list[ReflexResult]) -> float:
        for r in reflex_results:
            if r.track_id == track_id:
                return r.kinetic_score
        return 0.0

    def _build_event(self, track_id: str, reason: str, is_override: bool) -> NarratorEvent:
        obj = self.object_registry.get(track_id, {})
        return NarratorEvent(
            track_id=track_id,
            object_class=obj.get("class", "unknown"),
            distance_m=obj.get("distance_m", 0.0),
            closing_velocity_ms=obj.get("velocity_ms", 0.0),
            bearing_deg=obj.get("bearing_deg", 0.0),
            reason=reason,
            is_override=is_override,
        )
