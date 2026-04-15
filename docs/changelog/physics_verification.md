# Changelog — `src/physics_verification/physics_verification.py`

> **The Judge** — arbitrates between the Reflex Layer's deterministic physics and SLM-1's semantic evaluation to select the NarratorEvent and generate RL reward signals.

---

## [v1.0] — 2026-03-25 | Initial Implementation

**Session:** *Architecting Dual-SLM Navigation System*

### What Was Built

This module is the architectural centrepiece of the CPE system, implementing the arbitration logic between two independent intelligence streams.

#### Data Classes

```python
@dataclass ReflexResult:
    track_id: str
    kinetic_score: float
    ttc_s: Optional[float]
    override: bool          # True if TTC < 1.0s

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
    .to_prompt() -> str     # formats for SLM-2 input
```

#### `PhysicsVerification.adjudicate(reflex_results, semantic_eval, future_reflex_results)`

Implements the 4-rule adjudication hierarchy:

| Rule | Trigger | Action | RL Reward |
|---|---|---|---|
| **1. Override** | Any `ReflexResult.override == True` (TTC < 1.0s) | Bypass SLM-1, send override alarm to Narrator | +100 if SLM-1 agreed, -500 if it missed |
| **2. Physics fallback** | `semantic_eval is None` or no physics data | Use top physics object, reward=0 | 0.0 |
| **3. Perfect agreement** | `semantic_eval.primary_threat_id == reward_target.track_id` | Emit event for top physics object with SLM reason | +100 |
| **4. Divergence** | SLM-1 picked a different object than physics | Compare `slm_k_future × confidence` vs `physics_k_future × 0.8` — winner takes event | +50 (SLM novel catch) or -200 (SLM missed high-K) |

#### Future Kinetic Grounding
`future_reflex_results` (optional) provides physics scores at T=future. When provided, the reward comparison uses **future K scores** rather than present scores. This teaches SLM-1 to identify objects that *will become* dangerous, not just those that are currently high-K. Same mechanic as K₊₂ in `fact_sheet_builder.py`.

#### Reward Constants
```python
REWARD_CORRECT        = +100.0
REWARD_NOVEL_CATCH    = +50.0
PENALTY_MISSED        = -200.0
PENALTY_OVERRIDE_MISS = -500.0
```

#### `_build_event(track_id, reason, is_override)`
Looks up the object's physical properties from `object_registry` (populated each frame by the Perception Stack) and assembles a `NarratorEvent`.

### Architecture Position

```
Reflex Layer ─────────────────────┐
  (TTC + Kinetic Score)           │
                                  ▼
                       PhysicsVerification.adjudicate()
                                  │
Cognitive Layer ─────────────────┘    → NarratorEvent → SLM-2
  (SLM-1 SemanticEval)               → rl_reward → PPO trainer
```

### Open Items / Future Work
- `object_registry` currently must be manually populated each frame by the calling code. A future version should accept the perception stack's output directly.
- Calibration of `HIGH_K_THRESHOLD = 5.0` and `OVERRIDE_TTC_S = 1.0` is pending dataset testing.
- The Reflex Layer proper (`src/reflex_layer/`) is scaffolded but not yet implemented — currently the perception pipeline feeds `ReflexResult` objects directly.

---

## File Reference
- **Source:** [`src/physics_verification/physics_verification.py`](../../src/physics_verification/physics_verification.py)
- **Uses:** `dataclasses`, `typing`, `logging`
- **Upstream:** `reflex_layer` (ReflexResult), `cognitive_layer` (SemanticEval)
- **Downstream:** `narrator_slm` (NarratorEvent), PPO trainer (rl_reward)
- **Status:** Implementation complete. Integration with live reflex + cognitive layers pending Phase 2.
