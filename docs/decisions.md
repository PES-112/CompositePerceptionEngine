# Decision Records

Open design decisions that need team discussion before implementation. Each entry captures the problem, the options considered, and their tradeoffs — decisions themselves are filled in once made.

---

## Ground Truth Strategy for Kinetic Score Formula Selection

**Status:** Undecided — revisit with team
**Owner:** TBD
**Related:** `docs/architecture.md` §"Kinetic Score", `evaluation/kinetic_score_comparison.py`, `evaluation/threat_score_eval.py`, `tools/manual_score_inspector.py`

### Problem

We have six candidate kinetic-score formulas (K0–K5, in `src/perception_stack/physics.py` and
mirrored in `evaluation/kinetic_score_comparison.py`) and need to pick one for production with
evidence defensible in a research paper.

The current evaluation harness is **circular**: `evaluation/threat_score_eval.py` grades each
formula's routing decision against a "ground truth" computed by *re-running that same formula*
on a future frame (`ground_truth_label()` calls `compute_k(same_track, fn)`). Each formula sits
an exam it wrote itself. Likewise `kinetic_score_comparison.py` correlates K against `1/TTC`,
but TTC is `d/v` — the same two inputs K consumes — and K1 is *defined* as
`sev·min(1/TTC, 10)`, so K1 wins by construction. `docs/architecture.md` currently states
"K4 (hybrid momentum + TTC) is the leading candidate," but no benchmark results exist anywhere
in the repo (`evaluation/benchmarks/kinetic_score_eval/` has only a README) — that claim has no
evidence and needs to be removed or caveated until real results exist.

We also found a **scale-comparability bug**: K5 is bounded to `[0, severity]` (max ≈ 2.5), so it
can structurally never cross `high_k=5.0` and therefore can never trigger a K-based reflex route
on its own. K0/K3 are unbounded. Comparing routing % / F1 across formulas at identical absolute
thresholds is comparing formulas with incompatible scales, not comparing discriminative quality.

### The core question: what counts as "ground truth danger"?

Any fix requires a formula-independent definition of "this object was actually dangerous,"
derived from something other than a kinetic-score formula. Three options, not mutually exclusive:

#### Tier A — Kinematic "encounter" ground truth (automatic, formula-free)
Define an object as a **true encounter** at frame T if, within a lookahead horizon H seconds,
its track's *measured* `distance_m` drops below a hazard threshold (e.g. 1.5 m) while inside a
forward bearing cone (e.g. ±30°). This uses only `distance_m` and `bearing_deg` — never a K
score, never a class-severity weight. Fully automatic, runs over every session, gives large-N
statistical power. **Cannot** validate whether the severity weights themselves are right (why is
`bus=2.5` vs `person=1.0`?) — severity is a value judgment, not a physical quantity, so no
kinematic label can certify it.

#### Tier B — Blinded human judgment (small-N, validates severity weights)
On ~300–500 sampled frames, a human is shown the scene (RGB + object list) **with K scores and
routing hidden** and asked "which object, if any, is the top threat here?" This is the only tier
that can validate severity weighting and scene-level prioritization, which are inherently human
judgments. `tools/manual_score_inspector.py` already renders ~80% of this UI (bboxes, per-object
table, ✅/❌/⏭ stamps with localStorage persistence) — it needs a blinding mode (hide K/route
columns) and a "select top threat" interaction instead of frame-level approve/flag.

#### Tier C — VLM-as-annotator (scale-up, only after calibration)
Use a VLM to label many more frames than humans can, cheaply — but only report/use it *after*
measuring its agreement (Cohen's κ) against the Tier B human gold set. The VLM's reliability
becomes a *reported, measured result*, not an assumed one. Caveats: VLMs can't see metric depth
or closing velocity from a single still frame (they'd need frame pairs or explicit distance/
velocity text injected into the prompt — i.e., they'd be judging the Fact Sheet, not the image),
and running several VLMs and majority-voting does **not** avoid the calibration step, because
VLM errors are correlated (shared training data/priors) — voting does not equal independence.

### Options considered

| Option | What it buys | What it can't do | Cost |
|---|---|---|---|
| **A only** | Automatic, reproducible, all sessions, large N | Can't validate severity weights or scene-level prioritization | Low — script only |
| **A + B (recommended)** | A's scale + B's ability to validate severity/prioritization; the standard defensible pair for a paper | B is small-N, manual labeling time | Medium — B needs ~2-4 hrs human labeling for 300-500 frames |
| **A + B + C** | Scales human judgment to thousands of frames via a calibrated VLM; VLM reliability becomes a citable number | Extra engineering (VLM prompting/pipeline) + API cost; still needs B first to calibrate | High |
| **Multi-VLM voting as primary ground truth** | Cheapest scale-up | Relocates the ground-truth question rather than answering it; VLMs lack metric depth/velocity from stills; correlated errors overstate reliability; not defensible as primary evidence in a paper | Low build, but weak evidence |

### Answering the original question directly

> "Would manual verification of a video stream + kinetic score help, or running multiple VLMs
> to verify each frame?"

Manual verification (Tier B) is necessary but doesn't scale to full-corpus evidence by itself.
Multiple VLMs voting, used as the *primary* ground truth, is not recommended — it swaps one
unvalidated oracle for another and cannot see the physical quantities (depth, velocity) the
score is built from. The recommended combination is: **Tier A for scale + Tier B to validate
the human-judgment parts (severity, prioritization) + optionally Tier C to scale B once its
agreement with B is measured and reported.**

### Other open parameters (not yet decided)

- Lookahead horizon H, hazard distance threshold, and bearing cone width are knobs — should
  probably be reported as a sensitivity grid, not defended as single "correct" values.
- Dataset slice: SANPO-Synthetic has exact ground-truth depth (removes "your labels come from
  the same noisy depth estimator you're testing" objection) vs SANPO-Real (representative of
  deployment, but depth-estimation error is correlated across time in the encounter labels).
  Recommended: run on both, treat the synthetic-vs-real gap as its own robustness result.
- Session-level bootstrap CIs require multiple independent sessions (frames within one session
  are autocorrelated) — need to confirm how many SANPO sessions have a generated Stage 1
  perception CSV available before committing to a session count.

### Decision

**Decided 2026-08-18. Full reasoning in `docs/kinetic_score_opinion.md`.**

1. **Keep K0** (`sev · v² / max(d, ε)`). Its `v²` term is the only element in the design encoding
   *consequence* rather than *arrival time* — the reflex TTC gate (`events.py:217`, priority 100)
   already covers arrival time, so a formula collinear with `v/d` would add a parameter, not a
   signal.
2. **Removed K1–K5 and both evaluation scripts.** The candidates were dummies and not independent
   ones: within a class K1/K2/K5 are exactly rank-identical to 1/TTC (ρ = 1.0000) and K3 is
   ρ = 0.9998. Publishing a win over acknowledged strawmen is a liability, not evidence.
3. **K0 is defended by ablation of its own terms** (drop `v²`→`v`, drop severity, drop velocity,
   compare against plain time-to-hazard) **and by complementarity with SLM-1** — the latter being
   the actual architectural claim, and measurable with zero labels.
4. **Ground-truth tiers: A + B, as recommended above.** Tier A (automatic encounter labels)
   **eliminates but never selects** — it tests a necessary condition only. Tier B (blinded human
   judgment) is restricted to the ~5% of frames where formulas disagree, cutting the budget to
   ~100–300 frames.
5. **Multi-VLM polling rejected as primary evidence**, and the design inverted: the VLM becomes a
   blinded *referee* (one model, no formula, judging the scene) rather than a formula-holding
   contestant. Any VLM number requires Cohen's κ against a human gold set reported alongside it.

**Explicitly still open:** the class severity weights themselves. Measurement showed the exponent
choice is a *definition* rather than a discovery — scoring against a proximity notion of danger
versus a consequence notion yields opposite optima — so severity weights and the `v²` exponent are
the same kind of value judgment and must be settled together, by Tier B, not by more unlabelled data.
