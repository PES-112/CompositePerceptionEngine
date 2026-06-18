# Physics Verification — Arbitration Implementation

## Overview

Physics Verification arbitrates between two independent intelligence streams:
- **Kinetic Score `K`** — continuous, physics-grounded, deterministic
- **SLM-1 output** — semantic, intent-aware, context-sensitive

The key constraint: these two streams are **not on the same scale** and cannot be naively compared as raw numbers. The design below resolves this.

---

## SLM-1 Output Schema (Updated)

SLM-1 must return a **ranked list with categorical confidence**, not a single threat ID or a raw numerical score.

```json
{
  "threats": [
    {
      "id": "track_007",
      "rank": 1,
      "confidence": "high",
      "reason": "Motorcycle cutting across, path intersects in ~1.5s"
    },
    {
      "id": "track_012",
      "rank": 2,
      "confidence": "medium",
      "reason": "Pedestrian distracted, slow drift toward curb"
    }
  ],
  "scene_state": "crossing_intersection"
}
```

### Why a Ranked List, Not a Number

SLMs are poorly calibrated on absolute numerical outputs — `0.87` vs `0.91` is meaningless and inconsistent across scenes. However, SLMs are reliably strong at **relative ordering** and **categorical judgment**. The ranked list exploits this strength.

Categorical confidence levels (`high / medium / low`) are used instead of decimal probabilities for the same reason.

The `reason` string serves double duty: it feeds directly into SLM-2 narration with no extra processing.

---

## Arbitration Logic

### Step 1 — Normalize Kinetic Scores

Cap each K score against a predefined `MAX_EXPECTED_K` to produce a `[0, 1]` scale:

```python
MAX_EXPECTED_K = 20.0  # tune from dataset; represents a "certainly fatal" threat

K_norm[i] = min(1.0, K[i] / MAX_EXPECTED_K)
```

> **Why not divide by `sum(K)`?** Sum-normalisation strips absolute magnitude. A single harmless leaf blowing in the wind (`K = 0.1`) would normalise to `K_norm = 1.0` — indistinguishable from a truck at `K = 80`. Capping against `MAX_EXPECTED_K` preserves the real danger signal: a lone low-K object stays near `0.0`, not `1.0`.

`MAX_EXPECTED_K` should be set during Phase 2 calibration by running the dataset through the Threat Prioritizer and observing the empirical 99th-percentile K value.

---

### Step 2 — Convert SLM Ranking to Score Vector

```python
RANK_WEIGHT = {1: 1.0, 2: 0.55, 3: 0.25}  # unranked = 0.0
CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.65, "low": 0.3}

for threat in slm_output["threats"]:
    SLM_score[threat["id"]] = (
        RANK_WEIGHT.get(threat["rank"], 0.0)
        * CONFIDENCE_WEIGHT[threat["confidence"]]
    )
# all objects not mentioned by SLM get SLM_score = 0.0
```

---

### Step 3 — Dynamic Alpha Blend

```python
verdict_score[i] = α * K_norm[i] + (1 - α) * SLM_score[i]
primary_threat = argmax(verdict_score)
```

**`α` is not a fixed hyperparameter.** It is set dynamically per frame based on the agreement state between the two streams:

| Situation | α | Reasoning |
|---|---|---|
| OVERRIDE active (TTC < 1.0s) | `1.0` | Physics wins unconditionally |
| SLM rank-1 == highest-K object | `0.7` | Agreement; slight physics bias for safety |
| SLM diverges, confidence = `high` | `0.4` | SLM sees something K misses — trust it |
| SLM diverges, confidence = `low` | `0.85` | SLM uncertain; lean on physics |
| SLM says no threats, K is high | `1.0` | Hallucination guard — physics overrides |

---

### Step 4 — Divergence Detection

Divergence is when SLM rank-1 ≠ highest-K object.

```python
slm_primary = slm_output["threats"][0]["id"]
physics_primary = max(K_norm, key=K_norm.get)

is_divergent = (slm_primary != physics_primary)
```

Divergence events must be **logged separately** — they are the primary source of RL reward signal and the most informative cases for debugging.

---

## Full Arbitration Pseudocode

```python
MAX_EXPECTED_K = 20.0  # 99th-percentile K from dataset calibration

RANK_WEIGHT       = {1: 1.0, 2: 0.55, 3: 0.25}
CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.65, "low": 0.3}

def arbitrate(slm_output, K_scores, ttc_min, ttc_scores) -> NarratorEvent:

    # ── Guard: nothing to track ────────────────────────────────────────────
    if not K_scores:
        return NarratorEvent(threat_id=None, source="NO_OBJECTS", alpha=None,
                             reason="No objects in scene.")

    # ── Step 0: Hard override — no SLM involvement ─────────────────────────
    if ttc_min < 1.0:
        return NarratorEvent(
            threat_id = min(ttc_scores, key=ttc_scores.get),
            source    = "REFLEX_OVERRIDE",
            alpha     = 1.0,
            reason    = "Kinetic override: object approaching at critical speed."
        )

    # ── Step 1: Normalize K against absolute ceiling ───────────────────────
    # Dividing by sum() would strip absolute magnitude — a lone leaf at K=0.1
    # would score 1.0, same as a truck at K=80. Cap against MAX_EXPECTED_K
    # so a genuinely low-threat scene stays near 0.0.
    K_norm = {k: min(1.0, v / MAX_EXPECTED_K) for k, v in K_scores.items()}

    # ── Step 2: SLM score vector ────────────────────────────────────────────
    SLM_score = {obj_id: 0.0 for obj_id in K_scores}
    for threat in slm_output.get("threats", []):
        if threat["id"] in SLM_score:          # ignore stale/hallucinated IDs
            SLM_score[threat["id"]] = (
                RANK_WEIGHT.get(threat["rank"], 0.0)
                * CONFIDENCE_WEIGHT.get(threat["confidence"], 0.0)
            )

    # ── Step 3: Determine alpha ─────────────────────────────────────────────
    # IMPORTANT: check slm_says_safe BEFORE indexing threats[0] to avoid
    # IndexError when the SLM returns an empty threats list.
    slm_threats   = slm_output.get("threats", [])
    slm_says_safe = len(slm_threats) == 0
    physics_primary = max(K_norm, key=K_norm.get)

    if slm_says_safe:
        # K_norm = min(1.0, K / MAX_EXPECTED_K), so K_norm * MAX_EXPECTED_K just
        # recovers the original K score (capped at MAX_EXPECTED_K). Compare
        # directly against K_scores to avoid the round-trip.
        if K_scores[physics_primary] > HIGH_THRESHOLD:
            alpha = 1.0          # hallucination guard — physics overrides
        else:
            alpha = 1.0          # nothing dangerous either way; physics leads
        slm_primary    = None
        slm_confidence = None
        is_divergent   = False   # no SLM opinion to diverge from
    else:
        slm_primary    = slm_threats[0]["id"]
        slm_confidence = slm_threats[0]["confidence"]
        is_divergent   = (slm_primary != physics_primary)

        if not is_divergent:
            alpha = 0.7          # agreement; slight physics bias for safety
        elif slm_confidence == "high":
            alpha = 0.4          # SLM sees something K misses — trust it
        else:
            alpha = 0.85         # SLM uncertain; lean on physics

    # ── Step 4: Blend and select ────────────────────────────────────────────
    verdict = {
        i: alpha * K_norm[i] + (1 - alpha) * SLM_score[i]
        for i in K_scores
    }
    winner = max(verdict, key=verdict.get)

    # ── Step 5: Resolve reason for SLM-2 ───────────────────────────────────
    # winner may differ from slm_threats[0] when physics overrides the SLM.
    # Always look up whether the winner appears anywhere in the SLM ranked list.
    # If it does not, fall back to a physics-derived reason so SLM-2 never
    # narrates the wrong object.
    slm_reason_map = {t["id"]: t["reason"] for t in slm_threats}
    if winner in slm_reason_map:
        reason = slm_reason_map[winner]
    else:
        reason = f"Kinetic override: {winner} approaching rapidly."

    # ── Step 6: Log divergence for RL ──────────────────────────────────────
    if is_divergent:
        log_divergence(slm_primary, physics_primary, alpha, winner, slm_output)

    return NarratorEvent(
        threat_id = winner,
        source    = "ARBITRATED",
        alpha     = alpha,
        reason    = reason
    )
```

---

## Worked Example: Where the Ranked List Wins

**Scene:**
- `track_007`: Fast car in adjacent lane. `K = 8.0` — high, but trajectory does **not** cross the user's path.
- `track_012`: Pedestrian drifting into path. `K = 2.0` — moderate.

**Kinetic-only verdict:** `track_007` (highest K). Wrong.

**SLM-1 output:**
```json
{
  "threats": [
    {"id": "track_012", "rank": 1, "confidence": "high",
     "reason": "Trajectory intersects user path in ~1.5s"},
    {"id": "track_007", "rank": 2, "confidence": "low",
     "reason": "Fast but adjacent lane, no crossing risk"}
  ]
}
```

**Arbitration:**
- Divergence detected → check SLM confidence → `high` → `α = 0.4`
- `verdict_score[track_012]` wins after blend
- Logged as divergence → PPO `+50 semantic bonus`

This is the core case that justifies the SLM layer. A raw number from SLM would not have surfaced the trajectory reasoning.

---

## RL Reward Signal Integration

The arbitration output maps directly to the existing PPO reward structure:

| Outcome | Reward | Trigger |
|---|---|---|
| SLM rank-1 == highest-K object | `+100` | Agreement |
| SLM rank-1 catches K-missed semantic threat | `+50` | Divergence + correct verdict confirmed post-hoc |
| SLM rank-1 wrong, correct object at rank-2 | `-30` | Partial credit — smoother loss surface for PPO |
| SLM rank-1 misses object with `K > HIGH_THRESHOLD` | `-200` | Dangerous miss |
| SLM says safe, OVERRIDE active | `-500` | Fatal hallucination |

The `-30` partial credit reward (rank-2 hit) is new. It gives PPO a **gradient to climb** rather than a binary cliff, which improves convergence stability during the Phase 3 training loop.

---

## Changes Required in Other Components

### SLM-1 Prompt (Phase 1 / SFT)
The supervised fine-tuning target in `train.jsonl` must be updated to the ranked list schema:

```json
{
  "assistant": "{\"threats\": [{\"id\": \"Object_02\", \"rank\": 1, \"confidence\": \"high\", \"reason\": \"Distracted pedestrian drifting into path.\"}], \"scene_state\": \"sidewalk\"}"
}
```

### NarratorEvent (Physics Verification → SLM-2)
`NarratorEvent` should carry the `reason` string from SLM-1 rank-1 threat so SLM-2 can use it directly as narration context.

### Divergence Log Schema
```json
{
  "frame_id": 1042,
  "slm_primary": "track_012",
  "physics_primary": "track_007",
  "alpha_used": 0.4,
  "verdict_winner": "track_012",
  "reason_source": "slm",
  "slm_confidence": "high",
  "reward_applied": 50
}
```

`reason_source` is either `"slm"` (winner found in SLM ranked list) or `"kinetic_fallback"` (physics pulled in an object SLM did not mention). Tracking this ratio over the dataset gives a direct measure of how often physics is correcting SLM blind spots.

---

## Literature Review: Related Approaches to Priority Arbitration

### Overview

CPE's arbitration problem — combining a deterministic kinetic signal with a semantic language model judgment — has not been solved in exactly this form before, because the LLM-as-semantic-judge architecture is recent. However, every individual component has deep precedents in the literature, and understanding them lets us verify CPE's choices and flag where alternative approaches might offer improvements.

---

### 1. AEB-P: Hierarchical TTC Thresholds (2019)

**Paper:** *Research on Longitudinal Active Collision Avoidance of Autonomous Emergency Braking Pedestrian System (AEB-P)* — Sensors, MDPI, 2019.

**What they do:** AEB-P establishes a three-tiered warning hierarchy based on TTC intervals: a safe class with no intervention, a collision warning level where alarms fire, and an impending collision level where the brake system automatically intervenes.

**Similarity to CPE:** This is almost exactly the Reflex Layer's logic — CPE's `TTC < 1.0s` override is the equivalent of AEB-P's third tier. The conceptual lineage is validated.

**Key difference:** AEB-P is a pure physics system with no semantic layer. When the vehicle enters the third level, false alarms are likely to occur due to frequent changes in vehicle deceleration, so the warning system continues to issue the brake signal without recalculating TTC. CPE avoids this by having the Cognitive Layer handle contextual cases that would otherwise generate false positives. The override-lock problem AEB-P describes is exactly what the CPE arbitration layer is designed to prevent above the reflex threshold.

**Verdict for CPE:** CPE's design is more sophisticated. AEB-P validates the hard-override concept but demonstrates why a semantic correction layer is needed above the reflexive threshold.

---

### 2. Multi-Metric Collision Warning (Vision-Based, PMC Review)

**Paper:** *Vision-Based Collision Warning Systems with Deep Learning: A Systematic Review* — PMC, 2025.

**What they do:** Zhang et al. base their warnings on relative distance and velocity, and also consider additional factors such as the type of obstacle (pedestrian or vehicle), its lane (same or adjacent lane), and the type of environment (structured or unstructured) when issuing warnings. The review concludes that having a single threat metric makes a system less robust — using multiple threat metrics improves accuracy across different scenarios.

**Similarity to CPE:** CPE's Kinetic Score `K = (W_mass · Class_severity) × V²_closing / max(d, ε)` encodes exactly these factors — mass/class severity, velocity, and distance — into a single scalar. The review validates that this multi-factor approach is stronger than pure TTC alone.

**Key difference:** The literature combines these metrics into one score and uses it as the sole decision basis. CPE goes one step further: rather than encoding everything into K, it separates the *physics signal* from the *semantic signal* and arbitrates between them. This is a meaningful architectural advance because it means the system can catch cases where physics metrics lie (adjacent lane car) while still having a hard safety floor.

**Verdict for CPE:** The multi-metric K score design is well-grounded. CPE's separation of physics and semantics is novel relative to this literature.

---

### 3. TrafficRiskGPT: LLM Fine-tuned on Traffic Risk with Physics Indicators (2025)

**Paper:** *Large language model based system with causal inference and Chain-of-Thoughts reasoning for traffic scene risk assessment* — ScienceDirect / Knowledge-Based Systems, 2025.

**What they do:** TrafficRiskGPT is based on the LLaMA3-8B model with LoRA fine-tuning on traffic risk datasets. The system uses indicators such as TTC, DRAC (Deceleration Rate to Avoid a Crash), and DSS (Difference between Space Distance and Stopping Distance) as structured inputs to the LLM for risk reasoning.

**Similarity to CPE:** This is the closest paper to CPE's SLM-1 design — an SLM fine-tuned on structured scene facts including physics indicators, producing a risk judgment. The LoRA approach mirrors CPE's Phase 1 SFT plan exactly.

**Critical difference:** TrafficRiskGPT feeds raw TTC values directly into the LLM and asks it to output a risk level. CPE deliberately separates this: SLM-1 never sees raw TTC (that belongs to the Reflex Layer). SLM-1 sees the structured Fact Sheet and produces a *ranked semantic judgment*, while the Physics Judge computes TTC independently. This separation is intentional — it prevents the SLM from learning to pattern-match TTC numbers (which it will do poorly) and instead forces it to learn trajectory and intent reasoning.

**Verdict for CPE:** CPE's separation of physics input from SLM input is an improvement over TrafficRiskGPT's design. The risk of the TrafficRiskGPT approach is that the SLM absorbs TTC as a feature and becomes a noisy re-implementation of the physics layer rather than a complementary one.

---

### 4. LLM-Guided Collision Evaluation with TTC/MDC (2025)

**Paper:** *From Words to Collisions: LLM-Guided Evaluation and Adversarial Generation of Safety-Critical Driving Scenarios* — arXiv, 2025.

**What they do:** The framework assigns the LLM the role of a collision evaluation expert, with contextual prompting that defines key metrics such as TTC and MDC and quantifies safety-criticality by assigning risk scores based on threshold values. A safety-critical metrics parser converts structured metric data into natural language context for the LLM.

**Similarity to CPE:** The "metrics parser → natural language → LLM evaluator" pipeline is structurally identical to CPE's Fact Sheet → SLM-1 path. The paper also validates the approach of using LLMs as domain experts rather than raw classifiers.

**Key difference:** In this paper, the LLM is the *primary* judge. There is no independent physics arbiter to check LLM outputs. The framework identifies potential ego-attackers by analyzing motion and safety-critical metrics provided through prompts, then modifies ego-attackers' trajectories adversarially to induce collisions. The LLM can hallucinate a threat that isn't there — and the system has no cross-check. CPE's Physics Verification layer is exactly the guard this architecture lacks.

**Verdict for CPE:** This paper validates CPE's LLM-as-reasoner concept but also serves as a cautionary example for why Physics Verification (the Judge) is non-optional. The CPE design adds the safety net that this paper lacks.

---

### 5. Dual-Stage LLM for ADAS Semantic Risk Interpretation (2025)

**Paper:** *Dual-Stage LLM Framework for Scenario-Centric Semantic Interpretation in Driving Assistance* — arXiv, 2025.

**What they do:** Results reveal systematic inter-model divergence in severity assignment, high-risk escalation, evidence use, and causal attribution. Disagreement extends to the interpretation of vulnerable road user presence, indicating that variability often reflects intrinsic semantic indeterminacy rather than isolated model failure.

**Why this matters for CPE:** This paper is empirical evidence that directly justifies CPE's decision to use *categorical confidence* (`high/medium/low`) instead of numerical probability outputs from SLM-1. The paper shows that LLMs are inconsistent at absolute severity scores, even across runs of the same model. CPE's design sidesteps this by using relative ranking and categorical confidence — both of which are robust to this inter-run variance.

**Verdict for CPE:** This paper is essentially a peer-reviewed justification of CPE's SLM output schema choice. Cite it when defending the ranked-list design.

---

### 6. Dempster-Shafer (D-S) Evidence Theory for Sensor Fusion

**Canonical reference:** Murphy, R.R. (1998). *Dempster-Shafer theory for sensor fusion in autonomous mobile robots.* IEEE Transactions on Robotics and Automation, 14(2), 197–206.

**What D-S does:** Dempster-Shafer theory allows specifying a degree of ignorance instead of being forced to supply prior probabilities that add to unity. It combines evidence from different sources and arrives at a degree of belief that takes into account all available evidence. In sensor fusion, it is used to combine outputs from multiple sensors where each sensor assigns probability mass over a hypothesis set, and conflicting evidence is handled via a conflict term K.

**Similarity to CPE:** D-S theory is the formal mathematical framework most commonly used to solve exactly CPE's problem — combining two independent, heterogeneous evidence streams (physics + SLM) about the same set of objects. The "conflict term K" in D-S is analogous to CPE's divergence detection.

**Is D-S more efficient?** For CPE's specific case, **no**, for three reasons:

- D-S requires both sources to output probability mass functions over the full hypothesis space (all tracked objects). Getting a well-calibrated mass function from an SLM is exactly the problem CPE avoids by using categorical ranking. When a high degree of conflict exists between evidence pieces, Dempster's combination rule may yield unreasonable combination results and unreasonable weight assignments. An SLM producing inconsistent numerical masses would trigger exactly this failure mode.
- D-S is computationally heavier — it operates on the full power set of hypotheses, which scales exponentially with the number of tracked objects. CPE's alpha-blend is O(n) in the number of objects.
- D-S gives no mechanism for the asymmetric safety override (TTC < 1.0s → physics wins unconditionally). D-S is symmetric by construction; CPE's override is intentionally asymmetric.

**Where D-S would be useful:** If CPE ever replaces the SLM with a proper probabilistic model (Bayesian network or calibrated classifier) that outputs genuine mass functions, D-S becomes the correct arbitration framework. For the current SLM-based design, the alpha-blend with dynamic weighting is the right choice.

---

### 7. Predictive Collision Risk Area Estimation (LSTM Trajectory Prediction, 2022)

**Paper:** *A novel method of predictive collision risk area estimation for proactive pedestrian accident prevention* — Transportation Research Part C, 2022.

**What they do:** The system predicts trajectories using deep LSTM networks and infers collision risk areas statistically. Severity levels are divided as danger, warning, and caution. The proposed risk-aware method outperforms baseline policies, with the low TTC ratio metric stabilizing near 0.08 even in high jaywalker volume scenarios.

**Similarity to CPE:** The LSTM trajectory predictor is doing what CPE's "Future Kinetic Score K+2s" (Phase 1, Step 5) approximates — looking ahead to predict future risk state. The three-tier severity classification (danger/warning/caution) mirrors CPE's HIGH_THRESHOLD / LOW_THRESHOLD routing.

**Is their LSTM trajectory approach more efficient?** For CPE's edge deployment constraint, no. LSTM trajectory prediction requires per-object sequence modeling and introduces significant latency. CPE's look-ahead approximation using pre-computed dataset frames during training (rather than real-time trajectory prediction) is a deliberate tradeoff for edge hardware. However, if CPE is later deployed on a device with more headroom, replacing the K+2s heuristic with a lightweight trajectory predictor would be a meaningful upgrade.

---

### Summary Table

| Paper | Core Method | Validates in CPE | Gap CPE Fills |
|---|---|---|---|
| AEB-P (2019) | Tiered TTC override | Reflex Layer's hard override | SLM needed above reflexive threshold |
| Multi-Metric Review (2025) | Combined distance/velocity/class score | K score formula design | Physics + semantics separation |
| TrafficRiskGPT (2025) | LLM + LoRA on traffic physics metrics | SLM-1 SFT approach | Don't feed TTC to SLM; keep layers separate |
| LLM Collision Evaluator (2025) | LLM as primary risk judge with TTC context | Fact Sheet → SLM pipeline | Physics Verification cross-check is non-optional |
| Dual-Stage LLM ADAS (2025) | Multi-LLM severity divergence analysis | Categorical confidence over numeric scores | Ranked-list schema design |
| Dempster-Shafer (1998–2023) | Formal uncertainty combination from multiple sources | Conceptual arbitration framing | Not suitable while SLM outputs are categorical, not mass functions |
| LSTM Trajectory Prediction (2022) | Future risk prediction via sequence modeling | K+2s future scoring concept | Too heavy for current edge target; valid future upgrade |