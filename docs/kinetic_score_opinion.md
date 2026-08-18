# Kinetic Score: How to Choose and Defend It

**Status:** Opinion / recommendation — input to the open decision in `docs/decisions.md`
**Date:** 2026-08-18
**Related:** `src/perception_stack/physics.py`, `src/threat_prioritizer/events.py`,
`docs/decisions.md`, `docs/architecture.md` §2.7 §3

---

## 0. TL;DR

1. **Keep K0.** Its `v²` term is the only element in the design that encodes *consequence* rather
   than *arrival time* — precisely what the reflex TTC gate structurally cannot express.
2. **Stop comparing K0 against K1–K5.** They were written as dummies, and measurement confirms they
   are not even independent dummies. Beating a strawman you wrote yourself is a liability in review.
3. **Defend K0 by ablation**, not by beauty contest.
4. **You are not without ground truth.** Six metrics need no labels at all, and one more (the
   future) is fully automatic. Only *one* question genuinely requires a human.
5. **The VLM idea is right, but wired backwards.** Make the VLM a blinded *referee*, not a
   formula-holding *contestant*.
6. **Build the depth smoothing first.** K0 squares the noisiest input in the system.

---

## 1. The question is not "which of six formulas"

K1–K5 were written as dummies; only K0 (`sev · v² / max(d, ε)`) is research-derived. That matters,
because a benchmark showing **K0 beats five strawmen is worth nothing** — it is the single most
predictable reviewer objection, and it would be correct.

Measured this session, the dummies are not even *independent* strawmen. Within a class, K1, K2 and
K5 are **exactly rank-identical** to 1/TTC (Spearman ρ = 1.0000, identical rank vectors over 500
random points), and K3 is ρ = 0.9998:

| Formula | Rank-identical to 1/TTC within class? | ρ |
|---|---|---|
| K1 `sev·min(v/d, 10)` | **yes, exactly** | 1.0000 |
| K2 `sev·v/d` | **yes, exactly** | 1.0000 |
| K5 `sev·σ(v/d − β)` | **yes, exactly** | 1.0000 |
| K3 `sev·v²/(d²+ε)` | effectively | 0.9998 |
| K4 hybrid | near | 0.9960 |
| K0 `sev·v²/d` | **no — genuinely distinct** | 0.9514 |

The reason is structural: `min()`, squaring and the sigmoid are all strictly increasing functions of
`v/d`, and a strictly increasing transform cannot reorder anything. Cross-class pairwise rank
agreement is ≥ 0.945 for every pair except K5.

**Similarity to TTC is not the problem** — that is an accepted design property. The problem is that
beating a dummy is not evidence.

### Replace the beauty contest with ablations of K0

Ablation is the honest form of "we tried the alternatives." Every term in K0 is a claim that can be
knocked out and measured:

| Variant | The claim it tests |
|---|---|
| `sev · v²/d` | K0 as-is (baseline) |
| `sev · v/d` | Is the **v² exponent** doing work, or would linear do? |
| `v²/d` | Is **class severity** doing work at all? |
| `sev/d` | Is **velocity** doing work, or is this proximity + class? |
| `-(d − D_haz)/v` | Is the whole score beaten by plain **time-to-hazard**? |

Four ablations, each a one-line change, each answering a question a reviewer will actually ask.
Strictly stronger evidence than six formulas, and less work.

---

## 2. What K is for — and why that changes the metric

K is not a stand-alone threat detector. Per `architecture.md` §3, it is the **deterministic,
physics-grounded half of an arbitration** against SLM-1's semantic pick, refereed by
`PhysicsVerification` (`+100` when SLM-1's primary threat equals the highest-K object; `−200` when
SLM-1 misses an object with K above threshold). TTC already handles the immediate reflex trip
independently at `events.py:217`, at priority 100 — *above* any K-based route.

So K's value is **not accuracy. It is complementarity.**

A formula that is 5% less accurate but fails on *different* scenes than the SLM is more useful than
one that is 5% more accurate and fails on the *same* scenes — because a backstop that fails when the
primary fails is not a backstop.

This is the central architectural claim, and it is measurable **with no ground truth whatsoever**:
run both rankers over the corpus and measure how often they disagree, and on what kind of scene.
Nobody has to say who was right in order to establish that the two systems are decorrelated.

**Recommendation:** report complementarity as a first-class result, not accuracy alone.

---

## 3. How to compare two formulas with no ground truth

You never need an absolute "correct threat score." You only need **relative judgments on the cases
where the formulas disagree.** This is the standard construction for evaluating rankers without
labels — forced-choice pairwise comparison, the same machinery behind Bradley-Terry / Elo
leaderboards.

**Procedure:**

1. Run formula A and formula B over the corpus.
2. Keep only frames where `argmax K_A != argmax K_B`. Everything else is uninformative — measured
   pairwise rank agreement is ≥ 0.945, so this discards ~95% of frames and concentrates all of the
   signal. **This is what makes labelling affordable: ~100–300 frames, not thousands.**
3. Present each disagreement frame to a blinded referee: the RGB scene plus a neutral object list
   (class, distance, closing speed). **No K scores, no formula names, object order randomised.**
4. Ask one question: *"Which object is the top threat?"*
5. The formula whose `argmax` matches wins that frame. Aggregate to a win rate with CIs.

Output: *"A beats B on 63% (±6%) of the cases where they differ"* — a defensible claim requiring no
ground-truth score anywhere.

Bootstrap at the **session** level, not the frame level: frames within a session are autocorrelated
and frame-level CIs will be far too narrow.

---

## 4. On the VLM polling idea

The instinct is right; the wiring is backwards.

**Giving each VLM a formula makes the VLMs contestants, and then "pick the best VLM" has no
arbiter.** You would be judging accuracy against either another formula (circular), or VLM consensus
(agreement is not truth), or a human (in which case the VLMs added nothing). Three further problems:

- **Correlated errors.** VLMs share training data and priors. Majority voting only reduces error
  when errors are independent. Four models agreeing tells you much less than four independent judges
  agreeing — you would be measuring a shared prior and reporting it as truth.
- **The VLM cannot see the inputs.** A still frame contains neither metric depth nor closing
  velocity. You must inject `d` and `v` as text, at which point the model grades your Fact Sheet
  rather than the scene.
- **Injection biases the verdict.** Whichever formula's framing you put in the prompt shapes the
  answer.

### Invert it: the VLM is the referee, not the contestant

One VLM, **no formula**, blinded, answering *"which object is the top threat?"* on the disagreement
frames from §3. Judging a scene is a far easier and more reliable task than simulating a formula,
and it is a legitimate oracle.

Two conditions before any VLM number goes in a paper:

- **Calibrate first.** Have a human label 100–150 of the same disagreement frames. Report Cohen's κ
  between VLM and human. The VLM's reliability becomes a *measured, cited result* rather than an
  assumption. If κ is poor, you have learned something important and cheaply.
- **Ensemble for reliability, not for truth.** Multiple VLM referees are fine for estimating
  labelling noise. That does not substitute for the human calibration set.

---

## 5. Six metrics that need no ground truth at all

Computed from formula outputs alone. No annotation, no oracle, no VLM. Together these discriminate
between formulas far more than a TTC-correlation metric ever could.

### 5.1 Flicker rate — the most discriminating free metric

```
flicker(K) = fraction of consecutive frame pairs where argmax K changes identity
```

K feeds a narrator speaking to a blind user. A score whose top threat oscillates several times per
second is unusable regardless of how "accurate" it is. K0 squares velocity, and velocity is a
*differentiated* depth estimate, so K0 should flicker measurably more than a linear variant. This is
a falsifiable prediction of a real, reportable weakness, and it costs nothing to measure.

### 5.2 Rank stability under input perturbation

```
stability(K) = mean Kendall_tau( rank(K | d), rank(K | d*(1+e)) ),   e ~ N(0, sigma)
```

Sweep sigma in {2%, 5%, 10%}. Already measured analytically: at 10% depth error, K0's score error is
**+741%** versus **+190%** for a linear-velocity variant. This converts that into a ranking-level
number.

### 5.3 Temporal smoothness of the score

```
smoothness(K) = mean |K(t) - K(t-1)| / mean K,   per track
```

Distinguishes "K0 is jumpy" from "K0 changes its mind" — flicker alone conflates the two.

### 5.4 Discriminability / tie rate

Fraction of frames where the top two objects fall within 5% of each other. A score that cannot
separate the top two is not producing a usable priority. This exposes saturation failures directly.

### 5.5 Complementarity with SLM-1

```
disagreement(K, SLM) = fraction of frames where argmax K != SLM primary_threat
```

The architectural claim (§2), measured directly. Zero labels. If it approaches 0, K is redundant
with the SLM and the arbitration in `PhysicsVerification` is theatre. If it approaches 1, one of
them is broken. A healthy backstop lives in between — and you want the *shape* of the disagreement:
which classes, which distances, which speeds.

### 5.6 Self-supervised future consistency

The future is not an opinion. For each frame T, does `argmax K` at T match `argmax K` at T+H, once
the situation has resolved and the score is better informed? A formula whose own later self
disagrees with its earlier self is unstable in a way that needs no oracle to detect.

### Which metric settles which question

| Question | Settled by | Cost |
|---|---|---|
| Is K0 usable in a real-time narrator? | 5.1, 5.3, 5.4 | free |
| Is K0 robust to your depth noise? | 5.2 | free |
| Does K0 earn its place beside SLM-1? | 5.5 | free |
| Does K0 catch real physical encounters? | §6 future-as-label | free, automatic |
| **Are K0's `v²` and severity weights *right*?** | **§3 blinded referee — nothing else** | ~100–300 frames |

Only the last row needs a human. Everything else can run the day the Stage-1 CSVs exist.

---

## 6. The future is ground truth

You are not label-less. You do not need anyone to *say* an object was dangerous — you look at what
its **measured** distance actually did a few seconds later.

> An object is a **true encounter** at frame T if, within horizon H, its track's measured
> `distance_m` drops below `D_haz` while `|bearing_deg| < θ`.

This uses only `distance_m` and `bearing_deg` — never velocity, never severity, never any K. It is
not circular, because the prediction at frame T uses only frame-T data while the label comes from
frames the formula never saw. This is the standard construction for any forecasting problem.

`(H, D_haz, θ)` are knobs, not truths. Report a **sensitivity grid** (e.g. H in {1,2,3}s,
D_haz in {1.0,1.5,2.0}m, θ in {20,30,45}°). If the conclusion flips across the grid, that *is* the
result.

**Important limitation:** this tests a *necessary* condition only. A formula that misses real
encounters is broken; passing does not make it best. Use it to **eliminate**, never to **select**.

---

## 7. Why one question irreducibly needs a human

Over 8,000 simulated scenes where the truth was controlled by the simulator, scoring `argmax K`
against two different notions of danger gives **opposite** optima:

| exponent γ in `sev·v^γ/d` | vs "who gets close first" | vs "who delivers most energy" |
|---|---|---|
| 0.5 | **88.2%** | 71.6% |
| 1.0 | 84.3% | 79.0% |
| **2.0 — K0** | 76.5% | 83.2% |
| 3.0 | 72.9% | **84.4%** |

And against a pure proximity notion of danger, plain time-to-hazard beats every kinetic score, with
severity actively *hurting*:

```
direct time-to-hazard  -(d - 1.5)/v : 100.0%
gamma=1  v/d, no severity           :  91.4%
gamma=1  sev*v/d                    :  84.3%   <- severity HURTS
gamma=2  sev*v^2/d   (K0)           :  76.5%
```

**The reading:** if "danger" means proximity, K is redundant on top of the TTC already computed one
line above it in `_route_object`, and severity degrades it. **K0's `v²` is a bet that consequence
matters independently of arrival time** — that a bus and a pedestrian arriving in the same 2 seconds
are not equal threats.

That bet is correct and worth defending. But it is a *value judgment*, so no volume of unlabelled
video can confirm it, and no VLM poll can either. This is exactly why the §3 blinded referee is not
optional — and equally why it is the *only* place human effort is required.

---

## 8. Recommendation

1. **Keep K0.** Its `v²` is the only element encoding consequence rather than arrival time. Do not
   replace it on the strength of a dummy-formula benchmark.
2. **Delete K1–K5** from `physics.py` and drop the two circular evaluation scripts. Publishing a win
   over acknowledged dummies is a liability.
3. **Defend K0 by ablation** (§1), not by beauty contest.
4. **Report complementarity with SLM-1 as the headline** (§2, §5.5) — it is the actual architectural
   claim and it costs no labels.
5. **Run the label-free metrics first** (§5). They may settle the question before any human labels.
6. **Then the blinded referee on disagreement frames** (§3) — human first, VLM second with κ
   reported (§4). Budget ~100–300 frames.
7. **Pre-register the effect size that counts** before looking at results. Given ≥0.945 baseline
   rank agreement, a 2% gap is noise. If the honest answer is "the exponent is second-order and the
   severity weights dominate," that is a publishable finding — and a more robust one than a narrow
   win.

---

## 9. Known blockers

- **No input data exists.** There is no `data/` directory; no CSV in the repo carries `distance_m`,
  `velocity_ms` or `track_id` — only 30-row latency traces. Everything above needs Stage-1 CSVs from
  `tools/run_perception.py` over the 10 sessions in `selected_10_sessions.json`, run on the machine
  holding SANPO (artifacts reference `/home/student-4/...`).
- **The prescribed depth smoothing is not implemented.** `architecture.md` §10.1 calls for a Kalman
  filter on `d` before `v = Δd/Δt`; nothing of the sort exists in `src/` beyond a range filter at
  `depth_loader.py:26`. This matters concretely: at 10% depth error, K0's score error is **+741%**
  versus **+190%** for a linear variant, because K0 squares the noisiest input in the system.
  Choosing or defending a formula on the unfiltered signal benchmarks a system you do not intend to
  ship. **Build the filter first.**
