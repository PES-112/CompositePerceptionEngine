# Methodology

Consolidated, paper-ready record of the methods used so far and the reasoning behind each one. This
is the distilled "Methods section" reference — each subsection below can be adapted close to
verbatim for a paper draft. It draws on and cross-references the fuller working documents
(`kinetic_score_opinion.md`, `ablation_guide.md`, `architecture.md`), which remain
the place to find the full reasoning trail and open questions; this document states only what was
**decided and done**, not the alternatives considered.

For LLM-assisted drafting prompts built from this same evidence, see the Appendix at the end of this
document.

---

## 1. System overview

The Composite Perception Engine (CPE) is an asynchronous, physics-semantic dual-track architecture
for real-time hazard alerting: a deterministic perception/physics path guarantees a hard latency
bound on critical threats, while an advisory semantic path (not yet implemented — see
`pending_work.md`) is designed to add contextual richness without ever gating the safety path. Full
component diagram and contracts: `architecture.md`.

This document covers the methodology for the two components that are implemented and measured:
the **YOLO26n hazard detector** and the **kinetic score threat-ranking formula**, plus the
**edge-latency simulation methodology** used to argue real-time feasibility.

---

## 2. Dataset construction methodology

### 2.1 Primary corpus: SANPO

SANPO (Google Research egocentric outdoor navigation dataset) supplies synchronized RGB frames and
metric depth maps from real pedestrian sessions (`sanpo-real`) and a smaller synthetic subset
(`sanpo-synthetic`). CPE uses `camera_head/left` RGB + `.float16.gz` depth exclusively, because that
branch is what a phone-mounted camera setup can approximate and because
`src/perception_stack/depth_loader.py` supports that depth format directly. Depth ground truth is
taken directly from the dataset rather than predicted, so the kinetic-score inputs are grounded, not
estimated — a deliberate choice to decouple "is the formula right" from "is the depth network right"
during formula development. Full bucket layout and per-session sizes: `sanpo_dataset.md`.

**Sampling protocol.** `valid_streams.json` curates 462 real-session IDs. For the kinetic-score
ablation, a **30% sample (139 sessions)** was drawn by sorting the session list and seeding the
sample with `--seed 20260819`, so the sample is a deterministic function of the seed alone. Sampling
is done at the **session** level, not the frame level, because the downstream statistical procedure
(§4.5) bootstraps over sessions — 139 diverse sessions give more independent evidence than the same
number of frames drawn from fewer sessions. Frames are streamed directly from the public GCS bucket
one session at a time and discarded after that session's CSV is written, so no local copy of SANPO
is required (`tools/stream_sanpo_perception.py`).

### 2.2 Hazard-class augmentation: Roboflow Universe

SANPO's native taxonomy does not cover several navigation-critical classes (`pole`, `bollard`,
`stairs`, `crosswalk`, `pothole`, `puddle`). These were sourced from Roboflow Universe and merged
into a fixed **17-class CPE taxonomy** (`person, bicycle, car, motorcycle, bus, truck, traffic
light, stop sign, fire hydrant, pole, bollard, stairs, crosswalk, pothole, puddle, dog, bench`),
combining retained COCO navigation classes with the added hazard classes. Source labels are remapped
into this taxonomy and merged with session/source-level split boundaries to prevent train/test
leakage. Manifest: `training/configs/roboflow_universe_sources.json`; taxonomy:
`training/configs/cpe_hazard_classes.yaml`. Full intake and dataset-count targets: `yolo_training.md`.

### 2.3 Empirical class prioritization: depth-vs-detection gap analysis

Rather than choosing hazard classes by intuition, an empirical gap-analysis pipeline
(`tools/gap_analysis_experiments.py`, `notebooks/sanpo_yolo_gap_analysis.ipynb`) cross-references
SANPO depth maps against YOLO detections: for every sampled frame, it finds depth "blobs" inside a
configurable hazard-range window (e.g. 0.5–6.0 m) that have no overlapping YOLO box, tags each by
elevation (head/mid/foot level) and navigation column, and logs frame/session identifiers to
`hard_examples.json` rather than exporting heavy images. Running this whitelist-agnostic (all 80 COCO
classes, not a fixed subset) across multiple depth ranges surfaces which physical objects are
consistently missed at hazard range, which is the evidence used to prioritize which new classes to
add. Full parameters and workflow: `sanpo_dataset.md` §"Gap Analysis Workflow".

---

## 3. Detector training and evaluation methodology

### 3.1 Model and training protocol

YOLO26n (nano backbone) is fine-tuned rather than replaced with a larger detector, because the
system is edge-latency constrained, not leaderboard-mAP constrained. Training keeps the nano
architecture fixed, freezes early backbone layers by default, and replaces the broad 80-class COCO
head with the compact 17-class CPE head (`training/scripts/train_yolo26n_hazards.py`,
`training/configs/cpe_hazard_classes.yaml`).

Three checkpoint iterations were trained and evaluated on the same held-out test split
(`data/yolo_finetune_v2_full`):

| Version | Protocol | Outcome |
|---|---|---|
| v1 | Original compact hazard-only benchmark | Good `pole`/`bollard`, weak `stairs`; retained COCO classes not evaluated |
| v2 | Continued from v1 checkpoint, full 17-class data | Improved custom hazards but caused retained-class regression (notably `truck`) |
| v3 | **Restarted from base** `models/yolo/base_yolo26n/yolo26n.pt`, same cleaned data | Repaired most v2 retention regression, best mAP50-95 |

The v2→v3 restart is itself a methodological finding: continuing fine-tuning from an already
domain-adapted checkpoint compounded catastrophic forgetting of retained classes, while restarting
from the original base checkpoint on the same data did not. This is reportable as an ablation of
*training protocol*, not just of data.

### 3.2 Evaluation protocol

Two separate evaluations are run per checkpoint, both against the same held-out test split, to
separate two different questions:

1. **All-class held-out evaluation** (`training/scripts/evaluate_yolo26n_hazards.py`) — standard
   mAP50 / mAP50-95 / mAP75 and per-class precision/recall over all 17 CPE classes.
2. **Retained-class regression test against base YOLO26n**
   (`training/scripts/compare_yolo26n_retention.py`) — precision/recall/F1 for the COCO classes CPE
   retains (`person`, `car`, `bus`, `dog`, etc.), computed against the **unmodified base checkpoint**
   on the same test images, with explicit class-name remapping so COCO class IDs and CPE class IDs
   are never compared positionally. This is the check that caught the v2 `truck` regression.

Reported metrics (§ current numbers): `yolo_training.md`, `evaluation/benchmarks/yolo26n_version_comparison/`.

---

## 4. Kinetic score formulation and ablation methodology

This is the primary research contribution and the most methodologically load-bearing part of the
project — full reasoning trail lives in `kinetic_score_opinion.md` (§10 has the formal problem
statement and options analysis); this section
states the finished method.

### 4.1 Formula

```
K0 = severity(class) · v^2 / max(d, ε)
severity(c) = behaviour(c) · (mass(c) / 70 kg)^λ,   λ = 0.5
```

`v` is closing velocity (m/s, from frame-to-frame depth differencing), `d` is metric distance (m,
from dataset depth at the bbox centroid). `λ = 0.5` (severity ∝ √mass) was chosen as a deliberate
partial compression of a 171× class-mass range: it preserves a real destructive-mass bias (car 4.6×
a person, bus 13× a person, bus 2.8× a car) without the runaway of `λ = 1` (171×), under which a
distant bus could outrank an imminent pedestrian. A sparse behaviour multiplier separately covers
hazards mass cannot express (erratic motion, trip height, head-height mounting), and four massless
trip hazards (stairs, pothole, puddle, crosswalk) carry explicit weights outside the mass law.
Bounding-box apparent size (`A_px/d`) is implemented as an optional term but **disabled by default**
in production, because box area is ~99% predictable from class and metric depth by projective
geometry — it is retained only as an ablation arm (§4.4), not a shipped signal.

**`v²` is a stated design claim, not a measured result:** it encodes a bet that *consequence* matters
independently of *arrival time*. The reflex layer's separate TTC gate (`d/v`, priority 100 in
`events.py`) already handles arrival time, so a kinetic score collinear with `1/TTC` would be a
redundant parameter rather than an independent signal — this is the reason the `v²` exponent was
kept rather than reduced to `v¹`.

### 4.2 Why five other candidate formulas were discarded before evaluation

An earlier design considered six candidate formulas (K0–K5) and planned a formula-vs-formula
"beauty contest." Before running that contest, a rank-correlation check was run: within a class,
K1, K2, and K5 are **exactly rank-identical** to `1/TTC` (Spearman ρ = 1.0000 over 500 random
points), and K3 is ρ = 0.9998 — because `min()`, squaring, and a sigmoid are all strictly increasing
transforms of `v/d`, and a strictly increasing transform cannot reorder anything. K1–K5 were
therefore not five independent hypotheses but restatements of one, and a formula "beating" them would
be beating a dummy it was designed to beat. **Methodological conclusion: K1–K5 and the
circular evaluation harness that scored each formula against a ground truth computed by re-running
that same formula were deleted**, and replaced with the ablation design in §4.4. This negative result
is itself worth stating in a paper as a demonstration of a correlation check that should precede any
multi-formula comparison.

### 4.3 The evaluation problem this design solves

No ground-truth "this object was the true top threat" label exists for arbitrary video. Rather than
manufacture one, the evaluation is split into three independent tiers, each answering a question the
others cannot (full option analysis: `kinetic_score_opinion.md` §10):

- **Automatic, formula-free "encounter" labels** (§4.6) — necessary-condition test only, at full
  corpus scale, zero human effort.
- **Label-free ranking-quality metrics** (§4.5) — no ground truth needed at all, computed from
  formula output alone.
- **Blinded forced-choice referee judgment** (§4.7) — the only tier that can validate the
  human-judgment components (class severity weights, the `v²` exponent), restricted to the small
  slice of frames where it can actually change the answer.

### 4.4 Ablation design

Rather than compare independent formulas, each term of K0 is knocked out one at a time and the
ranking-quality impact is measured — the standard ablation construction, and strictly stronger
evidence than a multi-formula contest:

| Arm | Formula | Question it answers |
|---|---|---|
| K0 (baseline) | `sev · v²/d` | — |
| `linear` | `sev · v/d` | Is the `v²` exponent doing work, or would `v¹` do? |
| `no-severity` | `v²/d` | Does class severity change any ranking? |
| `no-velocity` | `sev/d` | Sanity arm — velocity must matter, or the harness itself is broken |
| `size` | `sev · v² · (A/d)^½ / d` | Does apparent bbox size add anything beyond class + depth? |
| `ttc` | `-(d - D_haz)/v` | Is K beaten by plain time-to-hazard? |
| `λ=0.25` / `λ=1.0` | mass-exponent sweep | Is the mass exponent too weak or too strong? *(dropped as a swept arm 2026-08-25 — see §4.8)* |

Implementation: `evaluation/kinetic_ablation.py`.

### 4.5 Label-free ranking-quality metrics

Computed from formula output alone, no annotation required:

| Metric | Definition | What it exposes |
|---|---|---|
| Flicker rate | Fraction of consecutive frame pairs where `argmax K` changes identity | Whether the score is usable for a narrator speaking to a person in real time |
| Rank stability | Kendall τ between the clean ranking and the ranking under synthetic depth perturbation (σ ∈ {2%, 5%, 10%}) | Robustness to the noisiest input (depth) |
| Temporal smoothness | Mean `\|K(t) − K(t−1)\| / mean(K)` per track | Distinguishes "jumpy" from "changes its mind" (flicker alone conflates these) |
| Tie rate | Fraction of frames where the top two objects are within 5% of each other | Whether the score actually discriminates a top threat |
| Complementarity with SLM-1 | Disagreement rate between `argmax K` and the cognitive layer's pick | The literal architectural claim that K is a backstop, not a duplicate signal — measurable with zero labels once SLM-1 exists |
| Future self-consistency | Does `argmax K` at frame T match `argmax K` at T+H once the scene has resolved? | Self-supervised stability check requiring no external oracle |

### 4.6 Automatic "encounter" ground truth

An object is labeled a **true encounter** at frame T if, within a lookahead horizon H, its **measured**
future `distance_m` drops below a hazard threshold `D_haz` while inside a forward bearing cone
`|bearing_deg| < θ`. This uses only distance and bearing — never velocity, severity, or any K score —
so it cannot structurally flatter any arm, and it is not circular: the prediction at frame T uses only
frame-T data, and the label comes from frames the formula never saw. `(H, D_haz, θ)` are reported as a
sensitivity grid, not defended as single correct values. **This tier eliminates, it never selects:** a
formula that misses real encounters is disqualified, but passing does not certify a formula as best.

### 4.7 Blinded referee evaluation (human + VLM)

For the one question that is irreducibly a value judgment — whether the `v²` exponent and severity
weights are *right* — the procedure is a forced-choice pairwise comparison, the same construction
behind Bradley-Terry/Elo ranking:

1. Run two formula variants (or K0 vs. an ablation arm) over the corpus; keep only the ~5% of frames
   where their `argmax` disagrees (baseline pairwise rank agreement is ≥ 0.945, so this concentrates
   essentially all the signal into a small, affordable set — ~100–300 frames instead of thousands).
2. Present each disagreement frame to a referee blind to which variant picked what: RGB scene plus a
   neutral, randomly ordered object list (class, distance, closing speed) — no scores, no formula
   names.
3. Ask one question: *"Which object is the top threat?"*
4. The variant whose `argmax` matches the referee's pick wins that frame; aggregate to a win rate
   with a confidence interval.

**Referee sourcing.** Three VLMs from three distinct model families (not three checkpoints of one
family, since same-family models share a prior rather than independently verifying) serve as
referees, run locally as OpenAI-compatible servers. VLM referee agreement is **not** treated as
ground truth on its own: a human labels the same 100–150 disagreement frames blind, and Cohen's κ is
reported between each VLM and the human label set before any VLM number is used as evidence. κ < 0.4
means the referees are not measuring what a human means by "top threat," and the VLM numbers stay out
of the paper; κ > 0.6 licenses using VLM win rates as a stand-in for a larger human-labeled set.
Multi-VLM majority voting is deliberately **not** used as a substitute for the human calibration step,
because VLM errors are correlated through shared training priors — agreement across models is not
evidence of correctness. Implementation: `evaluation/vlm_referee.py`; step-by-step: `ablation_guide.md`.

### 4.8 Statistical procedure

All confidence intervals are a **percentile bootstrap resampling whole sessions**, never frames.
Frames inside one walking session are autocorrelated (the same pedestrians and vehicles reappear
across consecutive frames), so a frame-level bootstrap would treat non-independent observations as
independent and produce intervals several times too narrow — making every arm look more different
from every other than the data supports.

**λ (the mass exponent) is a frozen design choice, not a swept experimental arm, as of 2026-08-25.**
It was originally planned as an ablation arm, but dropping the Tier-B blinded human-labeling step (for
resourcing reasons) removed the only tier capable of adjudicating between λ values — the label-free
metrics in §4.5 score arrival time, which λ barely moves, so sweeping it without a human referee would
produce a tie that looks like evidence but isn't. λ = 0.5 and the behaviour multipliers are therefore
reported in the paper as a stated limitation, not a measured result.

### 4.9 Results obtained to date

139 sessions, 19,402 scored frames (≥ 2 objects/frame), run 2026-08-26. `no-velocity` and `ttc` lose
decisively (rank stability collapses to 0.26–0.52 without velocity; plain TTC has 60% higher flicker
and 4× worse encounter recall than K0) — these two verdicts are solid regardless of the caveats below.
`linear`, `no-severity`, and `size` **tie** with K0 on every corpus-wide label-free metric — meaning,
on this corpus, kinematics (velocity/distance) dominate most reorderings. Full table and per-arm
verdicts: `ablation_guide.md` §6.

**One open caveat before treating the tie verdicts as settled**, tracked in `pending_work.md` §1:

1. **A corpus-wide tie cannot distinguish "the term never matters" from "the term matters only in a
   minority of frames diluted into 19,402."** `evaluation/kinetic_ablation_stratified.py` isolates just
   the frames with a severity-differentiated, kinematically-close pair of objects — the scenario
   severity was designed for — and re-runs the same metrics restricted to that subset. This has not
   been run against real data yet (self-checks pass; needs `data/processed/ablation_30pct/`, the raw
   per-session CSVs, which are large and still only exist on the machine that ran Stage 1).

**Resolved 2026-08-27, no longer open caveats:**

- The raw run artifacts (`metrics.json`, `report.md`, `disagreements.json`, `disagreements_key.json`)
  are now committed (`evaluation/benchmarks/kinetic_score_eval/run_2026_08_26/`) and independently
  auditable — the "CIs fully overlap" claims behind every tie verdict have been checked directly
  against them and hold (e.g. K0 and `linear` flicker rate both report the identical 95% CI
  `[0.964, 0.983]`).
- The `lam=0.25`/`lam=1.0` numbers are root-caused, not just flagged: they came from an earlier
  version of `kinetic_ablation.py` (commit `e045310`) that monkey-patched `SEVERITY_LAMBDA` per arm,
  before that plumbing was intentionally removed on 2026-08-25. The raw numbers are real and preserved
  in the committed run artifacts, but are struck from `ablation_guide.md`'s summary table (not merely
  annotated) since they can't be reproduced from the code currently in the repo. Full account:
  `ablation_guide.md` §6.

Blinded referee adjudication of the tied arms (§4.7) has not yet been run — the 219 disagreement
frames are committed and ready (`disagreements.json`), but running `evaluation/vlm_referee.py` against
them needs local GPU-served VLMs, which this environment doesn't have.

---

## 5. Edge-latency simulation methodology

### 5.1 Measurement vs. simulation — kept strictly separate

Two latency numbers exist in the repo and must not be conflated in a paper:

- **Native measurement**: PyTorch/CUDA on the NVIDIA GB10 training host — a real, measured number,
  but not the deployment target.
- **Simulated edge estimate**: an analytical Jetson Orin Nano 8GB proxy that scales the *measured*
  GB10 compute latency by a fixed factor (4.0×) and adds a fixed sensor/memory overhead (3.0 ms) per
  processed frame. This is a conservative screening heuristic, not an emulator — it cannot predict
  thermal throttling, TensorRT kernel behavior, or camera-copy overhead on real hardware.

Every reported edge number in this project is one or the other, labeled as such; no result currently
in the repo is a physical-device measurement (`hardware_targets.md`).

### 5.2 Benchmark protocol

`tools/benchmark_edge_realtime.py` streams RGB(+depth) frames through the full detector/tracker(+depth
post-processing) path at a fixed `frame_step` (3, i.e. every third frame is detected, the rest
interpolated), and reports mean/p95 latency against two budgets: a **reflex-path budget** (p95 < 50
ms, the hard real-time bound for detector/tracker work) and a **real-time streaming budget** (derived
from source FPS and `frame_step`). Preloaded in-memory frames are used for the primary readiness
number, because on-disk PNG decode was found to dominate replay-mode latency (~36 ms of a 50 ms frame
in an early test) and is a dataset-pipeline artifact, not a property of a live camera feed. Evaluated
on a seeded 10-session SANPO-Real subset, 30 frames/session (300 frames total), selected as the
smallest-download streams from the first 80 scanned valid streams. Current results:
`hardware_targets.md`, `sanpo_dataset.md`.

---

## 6. General experimental-design principles applied throughout

These recur across §3–§5 and are worth stating explicitly in a paper's methods section as a coherent
design philosophy, not restated per-experiment:

1. **Never grade a method against itself.** The K1–K5 harness was discarded specifically because it
   computed "ground truth" by re-running the method under test.
2. **Prefer relative (forced-choice) judgments over absolute scores when no ground truth exists.**
   Absolute "how good is this score" questions are unanswerable without labels; "which of these two
   is worse" is answerable by a blinded human or referee and is the standard construction behind
   ranking systems that have no ground truth (Bradley-Terry/Elo).
3. **Spend human/VLM labeling budget only where it can change the answer.** Restricting review to
   disagreement frames (~5% of the corpus) is what makes any labeling budget affordable at all.
4. **Report simulated and measured results as visibly distinct categories, never merged.**
5. **Bootstrap at the level of statistical independence, not the level of data volume.** Session-level,
   not frame-level, throughout.
6. **State design choices as limitations rather than dressing them up as findings** when the evidence
   to defend them was deliberately not collected (λ, behaviour multipliers).
7. **Sensitivity grids over single "correct" knob values** wherever a knob (H, D_haz, θ for the
   encounter label) has no principled single setting.

---

## Appendix: LLM Drafting Prompts

Prompts for drafting each paper section with an LLM (e.g. Gemini or Claude), built from the same
evidence documented above. Formerly the standalone `research_paper_prompts.md`; merged here since
both files serve the same paper-writing purpose. Feed the prompt as-is, or paste in the relevant
section of this document (§1–6 above) as additional grounding context first.

### A.1 Abstract & Introduction

> "I am writing a research paper on an edge-optimized AI system designed to provide real-time,
> semantically rich hazard alerts for the visually impaired. The system, called the Composite
> Perception Engine (CPE), uses a hybrid architecture. It combines a lightweight object detector
> (YOLO26n fine-tuned on street hazards), egocentric depth estimation, and a Knowledge-Distilled
> Small Language Model (SLM) for semantic scene narration. It uses a deterministic 'Physics
> Verification' layer to arbitrate between neural network outputs and raw kinetic physics
> (Time-to-Collision) to prevent hallucinations and guarantee <50ms response times for critical
> threats.
>
> Please write a compelling Academic Abstract (250 words max) and a comprehensive Introduction. The
> Introduction should highlight the limitations of current assistive technologies (either too
> computationally heavy for edge devices, or lacking semantic context) and clearly state our
> contributions: (1) N-frame skip with depth interpolation for 67% compute reduction, (2) Edge-YOLO
> fine-tuned on domain-specific hazards, and (3) SLM Knowledge Distillation from a massive offline
> teacher model."

### A.2 Related Work

> "Based on our edge-AI assistive technology system (CPE) for the visually impaired, write a
> 'Related Work' section. Compare our approach to three main existing paradigms:
> 1. Traditional sensory substitution devices (e.g., ultrasonic canes) which lack semantic
>    understanding.
> 2. Cloud-based LLM/VLM assistive tech (e.g., Be My Eyes, GPT-4V) which suffer from high latency and
>    require constant network connectivity, making them unsafe for real-time hazard avoidance.
> 3. Embedded CNN-based obstacle detectors (standard YOLO on mobile) which identify objects but fail
>    to prioritize physical threats (like a fast-moving bicycle vs a parked car) or provide natural
>    language context.
> Emphasize how our hybrid approach (deterministic physics + knowledge-distilled SLM) bridges the gap
> between semantic richness and edge-critical latency."

### A.3 Methodology: Perception Stack & N-Frame Skip

> "Write the Methodology subsection detailing our 'Hybrid Perception Stack'. Explain how we process
> RGB and depth data on the edge. Include the following technical details:
> - We fine-tune YOLO26n specifically for street hazards (potholes, bollards, stairs, crosswalks,
>   poles, puddles, dogs, and benches) rather than using standard COCO classes, as COCO is
>   insufficient for navigation.
> - The training workflow keeps the nano architecture fixed, freezes early backbone layers during
>   transfer learning, replaces the broad COCO head with a compact 17-class CPE hazard taxonomy, and
>   exports ONNX/TensorRT FP16 or INT8 edge artifacts with modern quantization flags to avoid
>   increasing runtime latency.
> - We evaluated multiple YOLO26n fine-tuning variants and selected a v3-from-base protocol: start
>   from pretrained `models/yolo/base_yolo26n/yolo26n.pt`, train the compact 17-class CPE head with
>   GB10 high-throughput settings (RAM dataset cache, AutoBatch, CPU-saturated data loading, and
>   deferred validation), then validate all classes and compare retained COCO-class precision/recall
>   against base YOLO26n with explicit class-name remapping. This repaired the v2 retained-class
>   regression while preserving custom hazard improvements.
> - We implement an N-frame detection skip (e.g., every 3rd frame). Between detection frames, we rely
>   on depth-guided tracking and interpolation, reducing compute load by ~67% while maintaining high
>   safety margins.
> - We use the depth map to extract precise distance (d) at the bounding box centroid. Object bearing
>   is computed directly from the YOLO bounding box centroid pixel coordinate:
>   `bearing_deg = ((cx_px − frame_width/2) / (frame_width/2)) × (hfov_deg/2)`, requiring no
>   additional hardware beyond the phone camera.
> - We compute a 'Kinetic Score' K = f(class_severity, velocity, distance). An earlier design
>   considered six candidate formulas (K0–K5) but a rank-correlation check showed K1, K2, and K5 are
>   exactly rank-identical to 1/TTC within a class and K3 is 0.9998-correlated — they were not
>   independent hypotheses. K1–K5 and the circular evaluation harness that scored each formula
>   against a ground truth computed by re-running that same formula were deleted. K0
>   (`sev × v² / max(d, ε)`) is instead defended by ablating its own terms (§A.5 below) against
>   label-free ranking metrics and a blinded referee, not a multi-formula contest.
> - We supplement egocentric SANPO pseudo-labels with carefully remapped Roboflow Universe datasets
>   for rare hazard classes, enforcing a fixed 17-class taxonomy and session/source-level validation
>   splits to prevent leakage.
> - We implement an empirical 'YOLO Gap Analysis' pipeline that cross-references depth map structures
>   with YOLO detections using morphological component labeling. The pipeline runs without class
>   whitelist constraints to evaluate general perception failures, compares multi-range depth
>   configurations (from immediate hazards to far-range path planning), compiles comparison metrics,
>   and logs hard examples in structured JSON files for downstream annotation workflows."

### A.4 Methodology: Knowledge Distillation & SLM

> "Write the Methodology subsection detailing our 'Cognitive Layer and SLM Knowledge Distillation'.
> Explain the following pipeline:
> - Because running large Vision-Language Models on edge devices is impossible, we use a
>   Teacher-Student distillation approach.
> - Offline, we use a massive Teacher model (e.g., Grounding DINO / GPT-4V) on street-view datasets
>   to generate high-quality pseudo-labels and rich semantic descriptions of hazards (e.g., 'pothole
>   at foot level').
> - We use this generated dataset to fine-tune a Small Language Model (SLM), with Qwen3-1.7B
>   non-thinking mode as the primary cognitive candidate and Qwen2.5-1.5B as fallback.
> - At runtime, the SLM receives a structured 'Symbolic Fact Sheet' — a JSON object containing only
>   fields derivable from a phone camera: track ID, object class, bearing (computed from pixel
>   centroid), distance, closing velocity, TTC, kinetic score, and the routing decision. Intent
>   labels (from HEADSUP dataset) appear only in the SFT training data to help the SLM learn to infer
>   intent from scene context, not as runtime inputs.
> - The SLM-1 response is also structured JSON: primary_threat_id, reason (one sentence), scene_state,
>   confidence, and future_confirmed (K₊₂ ground-truth validation flag). This structured output
>   enables deterministic reward computation in the Physics Verification RL loop.
>
> **Note when drafting:** the Cognitive Layer (SLM-1) has not been implemented yet as of this
> writing — write this subsection as planned methodology, not as a reported result, until
> `pending_work.md` §2 is complete."

### A.5 Methodology: Physics Verification (The Judge)

> "Write the Methodology subsection on the 'Physics Verification Layer'. This is the safety-critical
> core of the architecture. Explain that:
> - The system converts each YOLO+tracking+depth row into a formal `ThreatEvent` with distance,
>   closing velocity, bearing, Kinetic Score, TTC, route (`ignore`, `cognitive`, or `reflex`),
>   priority, and reason.
> - The SLM runs in soft real-time (~500ms) only for cognitive-route events, but language models can
>   hallucinate.
> - The implemented Reflex Layer consumes reflex-route `ThreatEvent`s, converts them into
>   deterministic `ReflexResult`s, bypasses all SLM/TTS models, and hands immediate override or
>   high-K physics fallback candidates to Physics Verification in hard real-time (<50ms).
> - The Physics Verification acts as an adjudicator: If TTC < 1.0s, it completely bypasses the SLM
>   and triggers an immediate 'Reflex' alarm. If the SLM hallucinates a safe scene but the Kinetic
>   Score is high, the Physics Verification overrides the SLM. This guarantees that critical physical
>   threats are never missed due to neural network latency or hallucination."

### A.6 Methodology: Kinetic Score Formulation and Its Evaluation

> "Write the Methodology subsection defending the Kinetic Score `K = severity(c) * v^gamma /
> max(d, eps)`. Explain that:
> - `gamma = 2` is a deliberate bet that *consequence* matters independently of *arrival time*: the
>   reflex layer's TTC gate already handles arrival time, so a K that merely reproduced TTC ranking
>   would be redundant. State this as a design claim, not a measured result.
> - Class severity is **derived from real-world mass**, not hand-tuned:
>   `severity(c) = behaviour(c) * (mass(c)/70 kg)^lambda`, lambda = 0.5. Justify lambda as *partial*
>   compression, not flattening: the class mass range spans 171x, and lambda=1 (literal kinetic
>   energy) lets a distant bus outrank an imminent pedestrian, while lambda=0.5 — severity
>   proportional to sqrt(mass) — retains a genuine destructive-mass bias (car 4.6x a person, bus 13x,
>   bus 2.8x a car) at the geometric midpoint between discarding mass and passing it through. Lambda
>   is frozen at 0.5 and is NOT an ablation arm: dropping Tier-B blinded human labelling removed the
>   only evidence source that could adjudicate between lambda values, so state plainly in the
>   limitations section that lambda and the behaviour multipliers are declared design choices with no
>   measurement behind them. Note that fitting the project's earlier hand-tuned table against log
>   mass recovers lambda ~ 0.18 (R^2 = 0.50), which is evidence the hand table under-weighted mass
>   rather than evidence for 0.18. Note also that with v^2 in the score, kinematics dominate severity
>   in most reorderings, so lambda mainly breaks ties between objects of similar motion — do not
>   oversell it. Report the sparse behaviour multipliers (erratic motion, trip height, head-height
>   mounting) and the four massless trip hazards held outside the law as an explicit, acknowledged
>   exception.
> - Apparent bounding-box size (`A_px/d`) is carried as an optional term with exponent mu, **disabled
>   in the shipped configuration**, because box area is ~99% predictable from class and metric depth
>   by projective geometry. Present its inclusion as an ablation arm and report the measured result
>   either way.
> - Evaluation avoids circularity: no formula is graded against another formula's output. Report
>   (a) the ablation of K0's own terms — gamma=1, severity removed, velocity removed, apparent size
>   added, and plain time-to-hazard as an external reference; (b) six label-free metrics (flicker
>   rate, rank stability under 2/5/10% depth perturbation, temporal smoothness, tie rate,
>   complementarity with SLM-1, future self-consistency); (c) automatic encounter labels derived only
>   from *measured* future distance and bearing, used to eliminate variants rather than to select one.
> - All confidence intervals come from a **session-level** bootstrap, because frames within a walking
>   session are autocorrelated and frame-level intervals would be dishonestly narrow. State the
>   sample: a seeded 30% sample (139 of 462) of the SANPO-Real valid streams, 19,402 scored frames.
> - Report the actual 2026-08-26 result: `no-velocity` and `ttc` lose decisively (sanity check
>   passes; K0 clearly beats plain time-to-hazard); `linear`, `no-severity`, `size`, and both lambda
>   arms tie with K0 on every label-free metric. State plainly that this means kinematics dominate
>   every reordering on this corpus and that the tied terms are retained as declared design choices,
>   not proven necessities — this is a defensible, reportable finding, not a null result to hide.
> - The one irreducibly human question — whether gamma and the severity weights are *right* — is
>   settled by a **blinded forced-choice referee** restricted to the frames where two variants pick
>   different top objects (~5% of frames, which is what makes labelling affordable). Three VLM
>   referees from three model families, run locally, estimate labelling noise, not truth; report
>   pairwise Cohen's kappa between them and, critically, kappa against a human-labelled calibration
>   subset. State plainly that if human kappa is poor, the VLM numbers are not evidence. **Note when
>   drafting:** this referee step has not been run yet as of this writing (`pending_work.md` §1) — the
>   219 disagreement frames are exported and waiting."

### A.7 Results, Discussion & Conclusion

> "Using only the supplied measured results, write the Results, Discussion, and Conclusion sections.
> Distinguish native NVIDIA GB10 measurements from the analytical Jetson Orin Nano 8GB proxy (4.0x
> compute-latency scaling plus 3.0 ms overhead). Report the 10-session SANPO result as simulated
> evidence: average p95 34.31 ms, worst-session p95 41.96 ms, with all simulated sessions below the
> 50 ms reflex budget. Report the YOLO26n detector result (v3: mAP50 0.641, mAP50-95 0.459) and the
> v2→v3 training-protocol finding (restarting from the base checkpoint repaired catastrophic
> forgetting of retained COCO classes that continuing from v1 caused). Report the kinetic-score
> ablation result from §A.6. Do not claim physical edge-device FPS, high-kinetic recall under real
> depth noise, or stale-SLM filtering until those experiments have actually been run
> (`pending_work.md` §1.4, §2, §6). Discuss the expected trade-offs between semantic richness, compute
> latency, and future battery-powered deployment."
