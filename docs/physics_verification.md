# Physics Verification — Component Doc

**File:** `src/physics_verification/physics_verification.py`
**Role:** The Judge — arbitrates between the Reflex Layer's deterministic physics and SLM-1's semantic evaluation to select the `NarratorEvent` to send to SLM-2, and generates the RL reward signal for SLM-1 PPO training.

**Status:** Implementation complete. Integration with live Reflex + Cognitive layers is pending Phase 2.

---

## Architecture Position

```
Reflex Layer
  └─→ ReflexResult (track_id, kinetic_score, ttc_s, override: bool)
                                    │
                         PhysicsVerification.adjudicate()
                                    │
Cognitive Layer                     │──→ NarratorEvent → SLM-2
  └─→ SemanticEval                  │──→ rl_reward → PPO trainer
        (primary_threat_id, reason,
         secondary_threat_id,
         scene_state, confidence)
```

---

## Data Classes

```python
@dataclass ReflexResult:
    track_id: str
    kinetic_score: float
    ttc_s: Optional[float]
    override: bool          # TTC < OVERRIDE_TTC_S (1.0s)

@dataclass SemanticEval:
    primary_threat_id: str
    reason: str
    secondary_threat_id: Optional[str]
    scene_state: str
    confidence: float       # [0, 1]

@dataclass NarratorEvent:
    track_id, object_class, distance_m,
    closing_velocity_ms, bearing_deg,
    reason, is_override
    → .to_prompt() formats for SLM-2 input
```

---

## Adjudication Rules (in priority order)

| Rule | Trigger | Action | RL Reward |
|---|---|---|---|
| **1. Override** | Any `ReflexResult.override == True` (TTC < 1.0s) | Bypass SLM-1, send override alarm | +100 if SLM agreed, **−500** if it missed |
| **2. Physics fallback** | No `semantic_eval` or no physics data | Use top-K physics object | 0.0 |
| **3. Perfect agreement** | `semantic_eval.primary_threat_id == reward_target.track_id` | Emit event, SLM-1 reason included | **+100** |
| **4. Divergence** | SLM-1 picked different object than physics | Compare `slm_k_future × confidence` vs `physics_k_future × 0.8`. Winner takes event. | **+50** (SLM novel catch) or **−200** (SLM missed high-K) |

---

## Future Kinetic Grounding

`adjudicate()` accepts optional `future_reflex_results` — physics scores at T=future (e.g. +2s). When provided, reward comparisons use **future K scores** rather than present. This teaches SLM-1 causal reasoning: identify objects that *will become* dangerous, not just those with high K right now.

This is the same mechanic as `K₊₂` in `fact_sheet_builder.py` — dataset depth is the unbiased ground truth teacher.

---

## RL Reward Constants

```python
REWARD_CORRECT        = +100.0   # SLM-1 picked the right object
REWARD_NOVEL_CATCH    = +50.0    # SLM-1 found semantic threat physics undervalued
PENALTY_MISSED        = -200.0   # SLM-1 missed a high-K object
PENALTY_OVERRIDE_MISS = -500.0   # SLM-1 said "safe" while TTC < 1.0s
```

---

## Calibration Targets (Pending Dataset Testing)

| Parameter | Current Value | Notes |
|---|---|---|
| `OVERRIDE_TTC_S` | 1.0s | Hard override threshold |
| `HIGH_K_THRESHOLD` | 5.0 | High kinetic score threshold for penalty |

---

## Object Registry

`PhysicsVerification` is initialised with an `object_registry: dict` that maps `track_id → {class, distance_m, velocity_ms, bearing_deg}`. This must be populated each frame by the Perception Stack before calling `adjudicate()`. Full integration (auto-wiring from pipeline output) is a Phase 2 task.
