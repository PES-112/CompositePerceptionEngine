# Methodology

Consolidated, paper-ready record of the methods used so far and the reasoning behind each one. This
is the distilled "Methods section" reference — each subsection below can be adapted close to
verbatim for a paper draft. It cross-references `architecture.md` (the v2 design) and
`ablation_guide.md` (run steps); this document states only what was **decided and done**, not the
alternatives considered. The full reasoning trail behind the K0 evaluation strategy was in
`kinetic_score_opinion.md`, removed 2026-10-14 and retrievable from git history (commit `a6e37a7`).

For LLM-assisted drafting prompts built from this same evidence, see the Appendix at the end of this
document.

---

## 1. System overview

The Composite Perception Engine (CPE) is a phone-based hazard alert for blind pedestrians (v2
design, adopted 2026-10-14; `architecture.md`). Perception (YOLO26n + ByteTrack + metric depth)
feeds a threat engine that ranks objects by **when** they will reach the walker (a learned GRU
arrival predictor), **how bad** a collision would be (mass-derived class severity), and **what** the
scene says about them (walkable-surface segmentation, planned). The ranking sits inside a
deterministic **physics safety envelope**: an object with time-to-collision ≤ 1 s in the walking path
always triggers an alert, whatever the learned components say. Alerts are scripted phrases played
from pre-rendered audio. There is no language model at runtime; the v1 design's SLM cognitive and
narrator layers were dropped.

The target device is an iPhone 17 Pro Max. All results come from **replaying SANPO-Real sessions**,
with SANPO's recorded depth used as is everywhere (`architecture.md` §4; the product's depth sensor
is an external LiDAR, not modelled); no live deployment or user study is run.

This document covers the methodology for the parts that are implemented and measured: the
**YOLO26n hazard detector** (§3), the **K0 kinetic-score ablation** that motivated the v2 design (§4.1–4.9),
the **learned arrival predictors** that replace K0 as the ranking (§4.10), and the **latency
methodology** (§5).

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

**Camera parameters (audited 2026-10-14, all 462 sessions; `tools/audit_sanpo_sessions.py`).**
Every curated session is 15 FPS. The pipeline uses `camera_head` (ZED-M): horizontal FOV 69.6–77.8°
(fx 1367–1391 px). (Until 2026-10-17 the audit read `camera_chest`'s ZED 2i intrinsics, 51.4–60.1°,
by mistake; `session_camera_details` is parallel to `session_camera_location`.) The Stage-1 CSVs
used so far assumed 30 FPS and a 70° linear FOV. The frame-rate error is recoverable from the data
itself: the stored closing speed equals 10.0 × the per-step least-squares depth slope (IQR
9.99–10.01), i.e. 10 Hz was assumed for frames processed at 5 Hz (15 FPS, stride 3). Hence stored
speeds are exactly 2× too high, one step is 0.2 s, the 20-step horizon is 4 s and the 8-step history
1.6 s. Labels are defined in steps and every metric we report is a ranking, which a uniform speed
scale does not change, so the results stand; only the time units in older text were wrong. Stage 1
now reads fps and fx per session (pinhole bearing `atan(dx / fx)`), and the replay converts old rows
(speed × 0.5, bearing from fx). With the real FOV, the ±30° label cone spans almost the whole image,
so the label is effectively "comes within the hazard distance while in view".

**Official split.** SANPO publishes a session-level split (`sanpo-real/splits/`). Of our 462
sessions, 374 are train and 88 test (`simulation/datasets/sanpo/official_split.json`). All final
numbers use it: models and every tuned parameter see train (validation carved from train) only, and
test is scored once. The 5-fold cross-validation below is kept as the development estimate.

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

K0 was the v1 threat ranking. The ablation below, plus the arrival-prediction results in §4.10
(K0 AUROC 0.57 as an arrival predictor), are why v2 keeps only K0's **severity** term ("how bad") and
takes arrival from a learned model. The evaluation machinery built here — label-free metrics,
formula-free encounter labels, session bootstrap — is reused unchanged for v2. The blinded
referee (§4.7) is not: it was dropped from the paper on 2026-10-20 (evaluation-only, needs GPU-served
VLMs) and is future work.

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
others cannot:

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
| Complementarity with SLM-1 *(v1 only; SLM-1 dropped 2026-10-14, never computed)* | Disagreement rate between `argmax K` and the cognitive layer's pick | Whether K was a backstop rather than a duplicate signal |
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

**Referee sourcing (protocol v1, 2026-08).** Three VLMs from three distinct model families (not
three checkpoints of one family, since same-family models share a prior) serve as referees, run
locally as OpenAI-compatible servers. VLM agreement is never treated as ground truth on its own, and
multi-VLM majority voting is not a substitute for human calibration, because VLM errors are
correlated through shared training priors.

**Protocol v2 (2026-10-17), used for the v2 announcement-order check.** The judge literature showed
that v1's design (object list, one presentation order, Cohen's κ against one human on 100–150
frames) could not support a conclusion. `evaluation/vlm_referee.py` now implements:

1. **Cases.** Frames where v2's top pick differs from K0's or from nearest-in-corridor's
   (`cases.json` + `cases_key.json`, picks per engine). A sample of n = 250 is drawn round-robin
   across sessions with a fixed seed.
2. **Forced choice over numbered boxes.** Objects are numbered 1..k on the image and in the list, in
   a seeded random order, and the referee answers a number.
3. **Order swap.** Each case is asked twice, the second time in reversed order, so the numbers
   change. Only order-consistent picks count, and the consistency rate is reported per referee.
   Position bias is the commonest judge failure: GPT-4 was consistent on 65% of swapped pairs
   (Zheng et al. 2023).
4. **Text-only control.** The same list without the image. If a referee agrees with its text-only
   self as much as with humans, it is reading the distance column, not the scene.
5. **Human ceiling.** Two people label the same ≥ 50 cases independently on the same numbered
   images. Human–human agreement bounds what any referee can reach; a fixed κ > 0.6 gate is
   meaningless if humans agree at 0.5.
6. **Coefficients.**
   - Raw agreement, against the per-frame chance level mean(1/k).
   - Gwet's AC1 for pairs. It is robust when one answer dominates; Cohen's κ collapses at high
     prevalence, the "κ paradox".
   - Krippendorff's α (nominal) across the referee panel; it handles missing ballots.
   - For AC1 and α a pick is coded by its distance rank within the frame, so categories mean the
     same thing across frames.
   - Cohen's κ is kept only for binary reductions ("picked engine E's object?").
   - Every coefficient gets a 95% session-cluster bootstrap CI (1,000 draws).
7. **Sample size.** At p_o ≈ 0.8 with ~3 candidates, 100 cases give a 95% CI of about ±0.12, so a
   point estimate of 0.6 can't clear a 0.6 gate; 250 cases give about ±0.08.
8. **Pre-registration and gate.** The first run writes `protocol.json` (sample, seed, conditions,
   coefficients, gate) and later runs refuse to change it. A referee's results stand in for human
   labels only if the **lower 95% bound of its AC1 against every human is > 0.4**. Agreement on
   disagreement-selected frames is lower than on random frames and is reported as such.

Implementation: `evaluation/vlm_referee.py` (`--self-check` verifies AC1 against the κ paradox and α
against Krippendorff's published worked example, α = 0.743). Step-by-step: `ablation_guide.md` §4.
The BLV listening study (`listening_study_plan.md`) adds a small human check on the same question.

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
decisively on the label-free metrics (rank stability collapses to 0.26–0.52 without velocity; plain
TTC has 60% higher flicker than K0). Their encounter-recall gaps (e.g. TTC "4× worse") do **not**
survive caveat 2 below and must not be cited.
`linear`, `no-severity`, and `size` **tie** with K0 on every corpus-wide label-free metric — meaning,
on this corpus, kinematics (velocity/distance) dominate most reorderings. Full table and per-arm
verdicts: `ablation_guide.md` §6.

**One open caveat before treating the tie verdicts as settled**, tracked in `pending_work.md` ("Carried over from v1"):

1. **A corpus-wide tie cannot distinguish "the term never matters" from "the term matters only in a
   minority of frames diluted into 19,402."** `evaluation/kinetic_ablation_stratified.py` isolates just
   the frames with a severity-differentiated, kinematically-close pair of objects — the scenario
   severity was designed for — and re-runs the same metrics restricted to that subset. This has not
   been run against real data yet (self-checks pass; needs `data/processed/ablation_30pct/`, the raw
   per-session CSVs, which are large and still only exist on the machine that ran Stage 1).
2. **The §4.6 encounter label is dominated by untracked depth blobs (found 2026-09-28).**
   - **Where they come from.** `detect_unlabeled_obstacles` emits one `unlabeled_obstacle` per fixed
     column of the lower 60% of every frame: the closest depth pixel YOLO didn't claim, with no
     ground removal. These are mostly the ground ahead of the walker.
   - **Share of the label.** They are 99.7% of all (frame, encounter-track) pairs.
   - **Why K0 misses them.** They are re-IDed every frame, so they have velocity 0 and K0 = 0.
   - **Unwinnable frames.** Only ~36% of encounter frames have the encounter object visible at T,
     which caps every arm's encounter_top1.
   - **Effect on the arms.** With the blobs removed (`evaluation/encounter_blob_check.py`), K0,
     no-velocity and TTC tie: 0.305 [0.177, 0.427], 0.306 [0.203, 0.409], 0.310 [0.177, 0.440].
     The corpus-wide TTC and no-velocity encounter results are therefore artifacts of blob
     selection, not evidence about real encounters.
   - **Unaffected.** The label-free metrics in §4.5 do not use this label and are not affected by
     this caveat.
   - **Fix.** Ground-plane removal in the obstacle detector, or tracked-object-only encounter
     labels, before encounter_top1 is cited.

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

### 4.10 Learned encounter predictors: GRU vs. GRU + cross-object attention (first run 2026-09-28)

K0 is a closed-form score of one frame. The question this experiment asks is whether a small learned
model, given each object's recent history, predicts the §4.6 encounter label better than K0 — and
whether letting objects in a frame see each other ("scene context") adds anything on top of that.
Code: `dataset/GRU/` (samples, labels, split) and `models/GRU/` (models, training, evaluation); see
their READMEs.

- **Input.** Per object, its last 8 processed frames (1.6 s at the 5 Hz processing rate) of runtime-derivable
  features: distance, closing velocity, bearing, per-step change in distance and bearing (retreat and
  lateral motion, both invisible to K0), YOLO confidence, log box area, track age, and a learned
  class embedding. `kinetic_score` is deliberately **not** an input, so a model cannot learn to copy K0.
- **Label.** Exactly the §4.6 encounter label, imported from `evaluation/kinetic_ablation.py` rather
  than re-implemented (unit-tested to match). A `--future-only` variant excludes frame T itself.
- **Two architectures, one variable.** Both share the same per-object GRU encoder and scoring head
  (`models/GRU/layers.py`). Architecture 1 (`GRU/`) scores each object independently. Architecture 2
  (`GRU_attention/`) adds one residual pre-norm self-attention block across the objects of a frame,
  with no positional encoding (fact sheets are K-sorted, so list position would leak K0's ranking)
  and training-time context dropout (so a single spurious detection cannot dominate its neighbours).
  Same samples, same split, same loop — a metric gap is attributable to the attention block.
- **Capacity control.** The attention model has ~2× the parameters (≈64k vs. ≈30k at hidden=64). If
  it wins narrowly, re-run the plain GRU at `--hidden 96` (≈ matched parameters) before crediting
  scene context.
- **Metrics.** Per object: AUROC and average precision. Per frame: `encounter_top1` and
  `flicker_rate` with the ablation's exact definitions and frame weighting. Everything is reported
  again on an **interaction slice** (≥ 2 objects within 5 m, or a crosswalk in view), because most
  frames contain no interaction and a corpus-wide average would dilute any effect — the same
  dilution concern as §4.9's open caveat.
- **Baselines, on the same held-out sessions.** K0, nearest object, and nearest object inside the
  ±30° encounter cone. The last is a plain rule mirroring the label's own definition: a learned model
  must beat it to show it learned more than the label.
- **Protocol.** Session-level 70/15/15 split fixed once at dataset build time and shared by both
  architectures, ≥ 3 training seeds per architecture, early stopping on validation loss,
  session-level bootstrap CIs (§4.8).
- **Caveat to state when reporting.** The label comes from the same noisy perception stack as the
  inputs (depth noise, ByteTrack ID switches), so "predicts encounters" means "predicts what the
  perception stack will measure", not ground-truth collisions.
- **Two dataset variants.**
  - **All objects:** the ablation's label, dominated by depth blobs (§4.9 caveat 2).
  - **Tracked only** (`--tracked-only`): the blobs are dropped. This is the only variant that tests
    prediction.
- **First results** (139 sessions, 3 seeds; `evaluation/benchmarks/gru_encounter_eval/run_2026_09_28/`).
  - **All objects:** both architectures reduce to the nearest-object rule (AUROC 1.000). This
    variant cannot test prediction.
  - **Tracked only:** the plain GRU reaches AUROC 0.905 ± 0.054, the attention GRU 0.879 ± 0.022,
    nearest 0.937 and K0 0.481.
  - **Stability:** the GRUs' top pick is steadier than nearest's (flicker 0.57–0.58 vs. 0.68–0.70).
  - **Verdict:** neither GRU beats nearest, and attention adds nothing measurable. The result is
    inconclusive rather than negative: validation holds only 7 positive object-frames and the test
    set 45 encounter frames.
  - **Next:** a relaxed tracked-object threshold (e.g. 3 m) and session-level cross-validation.
- **Arrival-time ranking (2026-09-29, 387 sessions;
  `evaluation/benchmarks/gru_encounter_eval/run_2026_09_29_arrival/`).**
  - **Label.** Tracked objects; the hazard distance is relaxed to 3 m to get enough positives
    (263 vs. 45 encounter tracks on 139 sessions). Arrival time = frames until the object first
    enters the zone.
  - **Output.** The model outputs a distribution over 4 slices of the 4 s horizon plus "not within
    it". The how-soon score is minus the expected arrival time.
  - **Metrics.** `arrival_order` (pairwise concordance within a frame) and `soonest_top1`.
    TTC = (d − 3 m)/v joins the baselines as physics' own arrival estimate.
  - **Protocol.** 5-fold session-level cross-validation, stratified so each fold holds ~19
    encounter sessions. Test predictions are pooled across folds. Differences are reported as
    paired session-bootstrap CIs (1,000 resamples).
  - **Results vs. K0 and TTC.** The GRU beats K0 (AUROC +0.31, arrival order +0.08) and TTC
    (+0.23 / +0.06); all CIs exclude 0.
  - **Results vs. nearest-in-cone.** The GRU ties it (+0.007 / −0.011, CIs include 0), with a
    steadier top pick (flicker 0.59 vs. 0.73).
  - **Attention vs. plain GRU.** +0.010 AUROC [+0.002, +0.021]. That survives a size-matched
    control (plain GRU at hidden 96 ≈ 67k parameters), but there is no gain in arrival order or on
    the interaction slice.
  - **Reporting guidance.** Report the K0 result as "K0 is not an arrival ranking" rather than as a
    straw-man win. Report the scene-context result as small and limited to encounter prediction.
- **Ensemble (2026-09-29; `models/Ensemble/`, `evaluation/benchmarks/ensemble_eval/`).**
  - **Components.** The GRU, the attention GRU, nearest-in-cone and TTC.
  - **Combiners.**
    - Rank averaging, with nothing fitted.
    - A softmax-regression stacker predicting the arrival slice, trained on out-of-fold
      predictions. Fold F is scored by a stacker fitted on the other folds only; this is standard
      cross-validation stacking, not fully nested.
  - **Results.** The stacked ensemble gives the best encounter prediction: AUROC 0.889, +0.019
    [+0.008, +0.032] over nearest-in-cone. It still **ties** nearest-in-cone on arrival order
    (+0.001 [−0.010, +0.012]).
  - **Reporting guidance.** Report it as evidence that arrival ordering is limited by the
    perception features, not by the choice of model.
- **Final numbers: all 462 curated sessions (2026-09-29;
  `evaluation/benchmarks/gru_encounter_eval/run_2026_09_29_arrival462/`).** Use these, not the
  387-session ones, in the paper.
  - **Data.** 364 sessions with tracked objects, 109 with encounters.
  - **GRU vs. K0.** AUROC +0.316 [+0.264, +0.353], arrival order +0.091 [+0.054, +0.122].
  - **GRU vs. nearest-in-cone.** AUROC +0.016 [+0.002, +0.029]; arrival order ties (−0.005
    [−0.028, +0.019]).
  - **Attention vs. size-matched GRU.** AUROC +0.007 [−0.000, +0.016], which is **not** credible
    on full data, so scene context adds nothing measurable.
  - **Stacked ensemble.** AUROC 0.894, +0.023 [+0.013, +0.034] over nearest-in-cone; arrival
    order ties (+0.007 [−0.008, +0.021]).

---

### 4.11 Held-out evaluation on SANPO's official split (2026-10-14)

- **Data.** `dataset/GRU/resplit.py` re-splits the 462-session arrival dataset onto the official
  split: 252 train / 41 validation (dealt from official-train sessions, stratified by encounter
  tracks) / 71 test sessions with tracked objects; 25 test sessions contain arrivals.
- **Training.** Both GRUs, `--time-bins 4`, 3 seeds, early stopping on validation loss.
- **Ensembles.** Rank averaging (unfitted) and the softmax stacker, now fitted on the models'
  **validation** predictions only (`models/Ensemble/heldout.py`), so no test label is used anywhere.
- **New baseline: TTC in the cone.** Objects inside the ±30° cone rank above those outside; within
  each group by TTC = (d − 3 m)/v (capped at 10 s), with non-closing objects ordered by distance.
  It gives TTC the same geometric prior nearest-in-cone has, so the learned-vs-physics comparison
  is fair.
- **Statistics.** As §4.8: per-seed metrics averaged over 3 seeds, paired session-bootstrap CIs
  (1,000 resamples) for differences.
- **Results** (`evaluation/benchmarks/heldout_official_split/official462_2026_10_14/report.md`):
  - GRU AUROC 0.888 (CV: 0.887). Beats K0 +0.317 [+0.219, +0.384], TTC +0.227 [+0.154, +0.279],
    cone TTC +0.123 [+0.075, +0.151]. Ties nearest-in-cone (+0.013 [−0.018, +0.046]; arrival order
    −0.013 [−0.050, +0.026]).
  - Flicker 0.561 vs. nearest-in-cone 0.714.
  - Attention: AUROC +0.004 [−0.018, +0.016] but arrival order −0.028 [−0.044, −0.003] vs. the
    plain GRU, so scene context via attention is, if anything, harmful held-out.
  - Stacker vs. GRU: +0.005 [−0.015, +0.016] AUROC; no ensemble gain survives held-out.

### 4.12 End-to-end replay of the runtime engines (2026-10-14)

`tools/replay_threat_engine.py` replays every official validation and test session, frame by frame,
through each runtime engine, rebuilding Stage-1 rows for tracked objects from the arrival dataset
(converted to correct units, §2.1).

- **Engines.** v1 as built (K0 routing → stub → Physics Verification; it speaks whenever the judge
  emits an event); v1 with the v2 announcement manager; and v2 (`src/threat_engine/`) with each
  arrival predictor: nearest-in-cone, physics (time until the fused relative path enters the zone
  inside the cone) and the GRU. v2 runs with and without camera poses (ego-motion ablation).
- **Tuning.** Each v2 engine's ignore threshold is chosen from {0.05, 0.1, 0.15, 0.25, 0.4, 0.6} on
  the validation sessions by the F1 of "warned" and "precision", then applied once to test.
  Announcer constants are declared (`announcer.py`) and not tuned.
- **Metrics.** An *arrival* is a track's first frame inside 3 m in the cone, counted only if the
  track was seen at least one frame earlier (so a warning was possible).
  - *Warned:* fraction of arrivals with an announcement about that track in the 4 s before.
  - *Precision:* fraction of announcements whose object is inside the zone now or within 4 s (the
    arrival label at that frame).
  - *Lead time:* median time from the first such announcement to arrival.
  - *Alerts, overrides and topic switches per minute.*
  - 95% CIs by resampling whole sessions (1,000 draws).
- **Results** (`evaluation/benchmarks/threat_engine_replay/official_2026_10_14/report.md`; 71
  sessions, 88 arrivals): v2 with the GRU reaches precision 0.59 [0.47, 0.69] at 17 alerts/min,
  against v1's 0.24 [0.12, 0.38] at 119/min, with 12 vs. 61 topic switches/min and 6.5 vs. 39
  overrides/min. It warns about 0.26 [0.18, 0.49] of arrivals against v1's 0.65 [0.58, 0.81].
  Lead time is ≈ 0 s for every engine, because objects typically first appear near the 3 m zone.
- **Caveats.** Open-loop; only tracked objects are replayed; inputs and labels share SANPO depth.

### 4.13 Sensor fusion: noise model fitted on train, and the camera-pose check

> **Superseded (2026-10-18).** The numbers in this section were computed on scrambled depth (architecture §9 item 0). The rebuild on correct depth is §4.16; the method described here is unchanged.

`src/threat_engine/fusion.py`: a constant-velocity Kalman filter per track on the ground plane. Tracks
are dropped after 5 missed frames and confirmed after 3 updates.

**Why the noise model was refitted (2026-10-17).** The hand-set values assumed stereo-like depth
noise: σ = 0.05 + 0.01·d² m, acceleration 1.5 m/s², a Gaussian update and a hard χ² gate at 11.62.
The Stage-1 depth is the median of SANPO depth inside the box, and it is far noisier than that, and
heavy-tailed:
- The robust σ from second differences on train sessions is ≈ 0.7 m even under 3 m, against 0.07 m
  assumed.
- Frame-to-frame jumps of 30–50% are common when the box median falls on background.

With the declared values the filter was badly over-confident:
- the mean normalised innovation squared (NIS) on validation was 17.5, where 2 is consistent;
- 26% of measurements were rejected by the gate, restarting tracks;
- so fused velocities were mostly noise.

**Fitting.** `tools/fit_kalman_noise.py` chooses the noise model by **maximum one-step predictive
log-likelihood on the 252 official-train sessions only**. It never looks at an arrival label.
- **Free parameters:**
  - acceleration σ (separately for static classes);
  - depth σ = a + b·d;
  - bearing σ;
  - initial speed σ.
- **Measurement noise:** Gaussian, or Student-t with ν = 4 (Roth et al.'s approximation: R is
  inflated by (ν + NIS)/(ν + 2), so a jump is down-weighted instead of dropped). With ν = 4 the gate
  only rejects jumps too large to be the same object (NIS > 60).
- **Search:** Nelder–Mead on the log-parameters, 100 evaluations per ν.

Results (`evaluation/benchmarks/kalman_noise_fit/fit_pose.json`):

| | log-lik train | log-lik val | NIS mean / median, val | rejected, val |
|---|---:|---:|---:|---:|
| declared (2026-10-14) | −6.60 | −10.15 | 17.5 / 2.66 | 26% |
| Gaussian fit | −4.26 | −3.63 | 1.76 / 0.48 | 3% |
| **Student-t fit (ν = 4), the default** | **−4.13** | **−3.37** | 4.2 / 1.16 | 0.6% |

Fitted values: acceleration σ 0.35 m/s² (static classes 0.098), depth σ 0.30 + 0.22·d m, bearing σ
0.55°, initial speed σ 0.85 m/s. `NoiseModel.declared()` keeps the old values for comparison.

The median 1 s-ahead position error on validation is 2.16 m fitted vs 1.98 m declared. The two
aren't comparable: the declared filter keeps restarting tracks, so it only forecasts the 210 easy,
stable ones, against 344 for the fitted filter.

**Kalman arrival predictor.** `src/threat_engine/kalman_arrival.py` draws 128 fixed samples from
each track's relative state covariance (`FusedTrack.rel_cov`, 4×4) and runs each forward at constant
velocity. P(arrive) is the share of samples that enter the label's corridor within the horizon.
Unlike the mean-state physics rule, which returns ≈ 0 or ≈ 1, it gives an uncertain track a middling
probability, so the ignore threshold becomes a real knob. It is the replay arm `v2_kalman` and the
ladder rung `cv_kalman`.

**Results** (official test split, scored once):
- **Held-out ladder** (§4.14): `cv_kalman` (fitted, probabilistic) vs. the declared mean-state rung:
  AUROC 0.862 vs 0.822, Δ +0.040 [+0.023, +0.054]; AUPRC Δ +0.122 [+0.086, +0.175]; Holm p < 0.001
  for both. It now **ties the GRU**: −0.013 [−0.039, +0.024], against −0.052 [−0.078, −0.013] before.
- **Replay, same physics predictor, paired before/after at fixed thresholds**
  (`kalman_noise_fit/replay_before_after_test.md`): false alerts/min fall by 7.3–14.5 (every CI
  excludes 0) and precision rises by 0.08–0.22.
  - With camera poses at threshold 0.95: 17.7 → 10.3 false alerts/min, Δ −7.3 [−10.7, −4.4].
  - Without poses: 18.1 → 5.0, Δ −13.1 [−16.3, −10.1].
  - Recall drops in two of the four settings: with poses at 0.95, 0.40 → 0.27, Δ −0.13 [−0.33, −0.04].
- **v2_physics still meets no alert budget on validation with poses.**
  - Without poses it now meets `budget_10`: warned 0.29 [0.20, 0.50] at 5.1 [3.1, 6.9] false/min.
  - The probabilistic `v2_kalman` meets `budget_5` and `budget_10`. At `budget_5`: warned 0.22
    [0.11, 0.44] at 4.7 false/min with poses, 0.27 [0.17, 0.52] at 4.6 without. Both are below the GRU
    (0.32 [0.24, 0.50] at 4.1).
  - Report: `evaluation/benchmarks/threat_engine_replay/corridor_kalman_2026_10_17/`.
- **The tracker feeds every v2 arm**, so the GRU and nearest-in-corridor arms changed too:
  - `v2_gru` at `budget_5`: warned 0.32 [0.24, 0.50], 4.1 false/min, precision 0.66 [0.55, 0.74],
    against 0.32, 5.4 and 0.56 with the declared filter.
  - `nearest_in_corridor` (corridor membership from fused positions) fell to 0.05 [0.00, 0.12] warned
    at `budget_5`, from 0.24. Smoother positions enter the 0.75 m corridor later. Compare arms only
    within one tracker version.

**Camera poses.** SANPO's `fixed_camera_poses.csv` is used with OpenCV camera axes in a Z-up world,
NOT_READY rows interpolated (`tools/validate_sanpo_poses.py`). Trajectory-level gates pass, but
object-level reconciliation is still inconclusive. With the fitted filter, the no-pose arm does at
least as well as the pose arm for every v2 predictor at `budget_5`: fewer or equal false alerts at
equal or higher recall. Ego-motion removal remains an ablation, not a default claim.

### 4.14 Baseline ladder on the corridor label (2026-10-17)

> **Superseded (2026-10-18).** The numbers in this section were computed on scrambled depth (architecture §9 item 0). The rebuild on correct depth is §4.16; the method described here is unchanged.

`python -m models.GRU.benchmark` on `…_official_corridor_w0.75_d3_h20`: 71 test sessions (20 with
an arrival), 4,721 objects, positive rate 0.169. Learned models are averaged over 3 seeds. Paired
session bootstrap (1,000 draws), two-sided p from the bootstrap SE, Holm within each stratum
(30 tests). Report: `evaluation/benchmarks/baseline_ladder/<dataset>/report.md`;
`ref_cv_kalman_declared/` holds the same ladder referenced to the old Kalman rung.

| Method | AUROC | AUPRC | Δ AUROC vs GRU [95% CI] | Holm p |
|---|---:|---:|---|---:|
| GRU hist4 | 0.879 | 0.718 | +0.004 [+0.002, +0.009] | 0.43 |
| GRU hist1 | 0.879 | 0.708 | +0.004 [−0.002, +0.012] | 1.00 |
| **GRU (h8, reference)** | **0.875** | **0.713** | — | — |
| GRU attention | 0.873 | 0.640 | −0.002 [−0.020, +0.018] | 1.00 |
| GBT, current step | 0.865 | 0.693 | −0.010 [−0.026, +0.010] | 1.00 |
| Kalman CV (fitted, probabilistic) | 0.862 | 0.675 | −0.013 [−0.039, +0.024] | 1.00 |
| GBT, history | 0.859 | 0.676 | −0.016 [−0.033, −0.001] | 0.74 |
| Logistic regression | 0.852 | 0.557 | −0.023 [−0.047, +0.004] | 0.74 |
| Nearest in corridor | 0.851 | 0.696 | −0.024 [−0.040, +0.002] | 0.43 |
| CVM, last two steps | 0.829 | 0.639 | −0.046 [−0.063, −0.010] | 0.016 |
| Kalman CV (declared, mean state) | 0.822 | 0.553 | −0.052 [−0.078, −0.013] | 0.033 |
| TTC in cone | 0.754 | 0.487 | −0.121 [−0.158, −0.058] | < 0.001 |
| TTC | 0.663 | 0.434 | −0.212 [−0.264, −0.140] | < 0.001 |
| K0 | 0.583 | 0.202 | −0.292 [−0.356, −0.194] | < 0.001 |

**Reading.**
- The learned models **do not significantly beat nearest-in-corridor, GBT on the current step, or
  the fitted Kalman** once the label is the corridor. They beat only:
  - the two-step CVM;
  - the declared Kalman;
  - TTC and K0.
- **History length doesn't matter:** hist1 ≈ hist4 ≈ h8, which is Schöller et al.'s "history is
  largely irrelevant" result.
- **Cross-object attention:** it ties on AUROC, but its AUPRC is −0.073 [−0.116, −0.023] (Holm
  p = 0.050), consistent with the earlier finding that neighbours' histories hurt.
- **What the GRU does buy** is in ranking and announcements, not separability. Its flicker is 0.58
  against 0.65 for nearest-in-corridor and 0.64 for the Kalman, and in the replay it reaches 0.32
  warned at 4.1 false/min, against 0.22 for the Kalman and 0.05 for nearest-in-corridor at
  `budget_5`.
- **Strata.** Static objects (303, positive rate 0.086) are near-trivially separable for every
  geometric method (AUROC ≈ 0.98). All differences live in the dynamic stratum.
- **A statistics caveat.** For hist4 the percentile CI excludes 0, but the SE-based p does not
  survive Holm. The bootstrap distribution is skewed; treat it as a tie.

### 4.15 Lead-based recall (2026-10-17)

> **Superseded (2026-10-18).** The numbers in this section were computed on scrambled depth (architecture §9 item 0). The rebuild on correct depth is §4.16; the method described here is unchanged.

All numbers use SANPO's recorded depth as is. A LiDAR-range simulation that briefly existed on
2026-10-17 was removed: the product uses an external LiDAR, not the phone's, and depth stays as
recorded.

**"Warned" includes lead-0 announcements.** The replay's `warned` counts an announcement made on the
arrival frame itself. Two lead-based columns were added to every operating-point table:
- `warned_before`: warned at least one processed step (0.2 s) before arrival;
- `warned_lead_ge1`: warned at least 1 s before.

**Results**, test, operating points chosen on validation
(`evaluation/benchmarks/threat_engine_replay/corridor_kalman_2026_10_17/`):

| Arm | budget | warned | warned ≥ 0.2 s before | warned ≥ 1 s before | false alerts/min | precision |
|---|---|---|---|---|---|---|
| v2_gru | 5 | 0.32 [0.24, 0.50] | 0.08 [0.00, 0.11] | 0.00 [0.00, 0.00] | 4.1 [2.1, 6.5] | 0.66 [0.55, 0.74] |
| v2_gru | 10 | 0.38 [0.26, 0.67] | 0.13 [0.03, 0.29] | 0.02 [0.00, 0.07] | 5.5 [3.1, 7.9] | 0.62 [0.48, 0.72] |
| v2_gru_nopose | 5 | 0.37 [0.25, 0.63] | 0.11 [0.00, 0.23] | 0.02 [0.00, 0.07] | 2.9 [1.4, 4.4] | 0.73 [0.64, 0.82] |
| v2_kalman | 5 | 0.22 [0.11, 0.44] | 0.06 [0.00, 0.16] | 0.02 [0.00, 0.07] | 4.7 [2.3, 7.3] | 0.54 [0.39, 0.66] |
| v2_physics_nopose | 10 | 0.29 [0.20, 0.50] | 0.08 [0.00, 0.11] | 0.06 [0.00, 0.09] | 5.1 [3.1, 6.9] | 0.53 [0.46, 0.60] |
| nearest_in_corridor | 10 | 0.19 [0.11, 0.32] | 0.08 [0.03, 0.14] | 0.02 [0.00, 0.03] | 7.2 [3.7, 10.8] | 0.47 [0.37, 0.57] |

**Reading.**
- At every arm's alert-budget point, only 0.08–0.14 of arrivals are warned at least one step ahead,
  and at most 0.06 at least 1 s ahead. Most of the headline `warned` is announcements on the arrival
  frame.
- The opportunity window isn't the limit: the median is 2.6 s, and 75% of arrivals are seen ≥ 1 s
  before they arrive. The engines are.
- Operating points were chosen to maximise `warned`, which rewards lead-0 announcements. They should
  be re-chosen on `warned_before` (or lead-conditional recall) before any lead or recall claim goes
  in the paper.

### 4.16 Rebuild on correctly decoded depth (2026-10-18)

> **Partly superseded (2026-10-19).** The labels, ladder, Kalman fit and replay below used bearings converted twice (§4.17). Dataset sizes and the Stage-1 rebuild stand; for numbers, use §4.17.

Every number in §4.11–4.15 used depth maps read as 960×960 instead of a 2-value header plus a
720×1280 image (architecture §9 item 0). Everything downstream of depth was redone with
`notebooks/colab_rebuild.ipynb` (Colab T4), with no change to methods, settings or the split rule:
- Stage 1 on all 462 sessions (300 frames, stride 3), then the arrival dataset, the official
  split and the six corridor labels.
- The Kalman noise refitted on train and written into `NoiseModel`'s defaults.
- The four corridor GRU variants retrained (3 seeds each).
- The ladder, the end-to-end replay (`threat_engine_replay/corridor_depthfix/`) and the Kalman
  before/after comparison re-run, with thresholds chosen on validation and test scored once.

**The dataset changed, not only the numbers.** More sessions now have tracked objects within reach:
287 train / 47 validation / 83 test (32 test sessions with an arrival), against 252 / 41 / 71 (20)
before. The corridor positive rate on test is 0.224 (was 0.169), and there are 96 test arrivals.
The old and new results are therefore not paired and must not be compared as differences.

**Kalman noise** (`kalman_noise_fit/fit_pose.json`). Correct depth is far less noisy, so the fit moved
a lot:

| | log-lik train | log-lik val | NIS mean / median, val | rejected, val | 1 s error, val (median) |
|---|---:|---:|---:|---:|---:|
| declared (2026-10-14) | −3.40 | −4.74 | 8.39 / 0.59 | 10.8% | 0.79 m (676 forecasts) |
| **Student-t fit (ν = 4), the default** | **−1.85** | **−1.82** | 9.91 / 1.27 | 3.5% | 1.14 m (775 forecasts) |

Fitted values: depth σ 0.058 + 0.053·d m (was 0.30 + 0.22·d on scrambled depth), acceleration σ
2.93 m/s² (static classes 2.04), bearing σ 0.37°, initial speed σ 0.90 m/s. Student-t beat Gaussian
(train log-lik −1.85 vs −2.47).

**Contradiction, flagged.** The fitted filter is much better on one-step likelihood but worse where
it matters:
- **Ladder:** `cv_kalman` vs the declared rung: AUROC −0.016 [−0.023, −0.004] (Holm p 0.009), AUPRC
  +0.010 [+0.001, +0.056] (Holm p 1.0).
- **Replay:** paired before/after with the physics predictor shows no significant change in warned,
  false alerts or precision (`kalman_noise_fit/replay_before_after_test.md`). Alerts/min rise by
  2.4 [0.2, 4.5] at threshold 0.95 with poses.
- **Forecasts:** the median 1 s-ahead error is higher (1.14 m vs 0.79 m), on more tracks.

The likely cause is the objective. With accurate depth, the one-step likelihood explains frame-to-frame
jumps (box shifts, residual ego-motion) as acceleration, and 2.9 m/s² makes velocity jumpy over a 4 s
horizon. A fit on multi-step (≈ 1 s) predictive error on train is the next thing to try. Until then,
the fitted filter is not shown to help, and either noise model is defensible.

**Baseline ladder** (`baseline_ladder/…_corridor_w0.75_d3_h20/report.md`; 83 test sessions, 5,974
objects; learned models averaged over 3 seeds; Holm within stratum, 30 tests):

| Method | AUROC | AUPRC | Δ AUROC vs GRU [95% CI] | Holm p |
|---|---:|---:|---|---:|
| GRU attention | 0.915 | 0.793 | +0.006 [−0.005, +0.012] | 1.00 |
| GBT, history | 0.914 | 0.796 | +0.005 [+0.000, +0.010] | 0.49 |
| GBT, current step | 0.914 | 0.798 | +0.005 [−0.001, +0.011] | 1.00 |
| GRU hist1 | 0.910 | 0.788 | +0.001 [−0.004, +0.008] | 1.00 |
| GRU hist4 | 0.909 | 0.790 | +0.000 [−0.003, +0.004] | 1.00 |
| **GRU (h8, reference)** | **0.909** | **0.788** | — | — |
| Nearest | 0.899 | 0.744 | −0.011 [−0.021, +0.006] | 1.00 |
| Logistic regression | 0.898 | 0.736 | −0.011 [−0.020, +0.001] | 0.51 |
| Kalman CV (declared, mean state) | 0.882 | 0.741 | −0.028 [−0.038, −0.013] | 0.001 |
| CVM, last two steps | 0.879 | 0.754 | −0.030 [−0.040, −0.016] | < 0.001 |
| Nearest in corridor | 0.871 | 0.734 | −0.039 [−0.054, −0.021] | < 0.001 |
| Kalman CV (fitted, probabilistic) | 0.866 | 0.751 | −0.043 [−0.056, −0.019] | < 0.001 |
| TTC in cone | 0.771 | 0.592 | −0.138 [−0.171, −0.115] | < 0.001 |
| TTC | 0.678 | 0.533 | −0.232 [−0.289, −0.196] | < 0.001 |
| K0 | 0.549 | 0.243 | −0.360 [−0.412, −0.311] | < 0.001 |

**Reading.**
- Every method separates better on correct depth (GRU 0.909, was 0.875 on a different test set).
- **Learned models and GBT tie.** The four GRU variants, GBT and attention all fall in 0.909–0.915,
  and no difference survives Holm. History length still doesn't matter.
- **Plain distance is close behind.** `nearest` is −0.011 [−0.021, +0.006]; its AUPRC is lower,
  −0.044 [−0.068, −0.020] (Holm p 0.002).
- **Learned models do beat every dynamics rule now:** both Kalman rungs, the two-step CVM,
  nearest-in-corridor, TTC and K0 (Holm p ≤ 0.001). This differs from §4.14, where the fitted
  Kalman and nearest-in-corridor tied the GRU.
- Some percentile CIs sit off-centre or exclude their own point estimate (e.g. declared Kalman ΔAUPRC
  −0.047 [−0.143, −0.037]). The bootstrap distribution is skewed, so use the Holm p.

**End-to-end replay** (`threat_engine_replay/corridor_depthfix/report.md`; operating points chosen
on 47 validation sessions, test scored once; 96 test arrivals). The opportunity window (first
sighting → arrival) has a median of 2.0 s: 27% of arrivals are first seen < 1 s before, and 25%
≥ 4 s before.

| Arm | budget | knob | warned | warned ≥ 0.2 s before | warned ≥ 1 s before | false/min | precision | alerts/min |
|---|---|---:|---|---|---|---|---|---|
| v2_gru | 5 | 0.8 | 0.35 [0.26, 0.53] | 0.12 [0.05, 0.22] | 0.04 [0.01, 0.08] | 8.5 [5.2, 12.1] | 0.56 [0.46, 0.66] | 19.3 |
| nearest_in_corridor | 5 | 0.95 | 0.29 [0.22, 0.40] | 0.11 [0.05, 0.20] | 0.03 [0.00, 0.06] | 7.5 [4.6, 10.7] | 0.44 [0.34, 0.52] | 13.4 |
| v2_gru | 10 | 0.7 | 0.44 [0.33, 0.65] | 0.16 [0.08, 0.28] | 0.06 [0.02, 0.13] | 9.0 [5.6, 12.7] | 0.56 [0.47, 0.64] | 20.5 |
| v2_gru_nopose | 10 | 0.7 | 0.48 [0.39, 0.66] | 0.19 [0.10, 0.31] | 0.07 [0.02, 0.14] | 10.1 [6.6, 14.0] | 0.54 [0.44, 0.63] | 22.0 |
| v2_kalman | 10 | 0.95 | 0.39 [0.28, 0.58] | 0.17 [0.09, 0.30] | 0.07 [0.03, 0.15] | 9.8 [5.9, 13.7] | 0.52 [0.41, 0.62] | 20.4 |
| nearest_in_corridor | 10 | 0.4 | 0.40 [0.29, 0.57] | 0.18 [0.09, 0.31] | 0.06 [0.02, 0.12] | 12.6 [8.8, 16.6] | 0.45 [0.35, 0.52] | 22.8 |
| v1_k0_announcer | 10 | 1.5 | 0.19 [0.10, 0.36] | 0.10 [0.05, 0.21] | 0.07 [0.02, 0.15] | 8.2 [6.1, 10.4] | 0.28 [0.15, 0.41] | 11.4 |

**Reading.**
- **The alert budgets are now hard to meet.** With correct depth, many more objects really are
  close, and the engines speak far more often (~20 alerts/min at the chosen points). No v2 arm meets
  `budget_2` on validation; only `v2_gru` and nearest-in-corridor meet `budget_5`; v2_physics meets
  none.
- **Validation doesn't transfer.** At `budget_5`, v2_gru's test false-alert rate is 8.5 [5.2, 12.1]
  per minute, above the 5/min it was chosen for.
- **Early warning is still the weak point.** At most 0.19 of arrivals are warned ≥ 0.2 s ahead and
  0.07 ≥ 1 s ahead, while 73% of arrivals are seen ≥ 1 s ahead.
- **Between arms, the CIs overlap everywhere.** v2_gru has the highest precision; nearest-in-corridor
  and the Kalman are within the CIs on recall.
- **Next:** re-choose operating points on `warned_before` instead of `warned`, and reconsider the
  announcer's rate limits and the false-alert budgets now that real close-range traffic is this dense.

### 4.17 Units fix, 1 s-ahead Kalman fit and lead-based tuning (2026-10-19)

> **Superseded numbers (2026-10-20).** The Stage-1 CSVs behind this section used the chest camera's fx and a mis-scaled depth lookup in the detector's size check (architecture §9 items 8–9). The methods here stand; the current numbers are in §4.18.

**A units bug in the rebuild, found and fixed.** The rebuilt Stage-1 CSVs already store pinhole
bearings (from the session's fx) and speeds at the true frame rate. But `relabel.py`, the ladder's
rule baselines and tabular features, the replay and both Kalman tools still treated the stored
bearing as the pre-2026-10-14 linear ±35° value and converted it again. That shrank every bearing
by about 10–15% (20° became 17.7°), so objects looked more central: about 13 cm at 3 m and 20°. The
replay also halved the already-correct speeds.
- **Fix:** `dataset/GRU/build.py` records `source.units = "pinhole_bearing_true_fps"` when every
  CSV has a stream sidecar with fx. Every consumer converts only when `legacy_units(manifest)` is
  true. The eight rebuilt manifests were marked. Tests: `tests/test_dataset_units.py`.
- **Redone:** the six corridor labels, all GRU runs (4 variants × 3 seeds), the Kalman fit, the
  ladder, the before/after comparison and the replay.
- **Labels moved a little:** the test positive rate went from 0.224 to 0.212, and test sessions
  with an arrival from 32 to 29. The §4.16 labels, ladder, Kalman fit and replay numbers are
  superseded by this section.

**Kalman noise, fitted on 1 s-ahead likelihood.** `tools/fit_kalman_noise.py --objective ahead`
scores each candidate by the log-likelihood of the position measured 1 s later, under the filter's
own 5-step forecast and its propagated covariance plus the measurement's R. The one-step objective
was refitted for comparison. The default was chosen by validation 1 s-ahead log-likelihood:

| Noise model | val 1 s log-lik ↑ | val NIS mean (2 = calibrated) | val median 1 s error | forecasts |
|---|---:|---:|---:|---:|
| declared (2026-10-14) | −3.43 | 8.6 | 0.77 m | 675 |
| one-step fit, Student-t ν = 4 (`fit_pose.json`) | −3.57 | 12.2 | 1.11 m | 761 |
| **1 s-ahead fit, Gaussian (`fit_pose_ahead.json`), the default** | **−3.08** | **2.3** | 1.09 m | 751 |

- **Fitted values:** acceleration σ 0.50 m/s² (static classes 2.14), depth σ 0.033 + 0.263·d m,
  bearing σ 0.28°, initial speed σ 0.88 m/s, Gaussian update with the χ² gate at 11.62.
- **Not completed:** the Student-t half of the 1 s fit, which was stopped twice by low memory on
  the laptop.
- **Why the old filter's median error is lowest:** it restarts tracks often, so it only forecasts
  the 675 steadiest ones.

**Results** (test, scored once):
- **Ladder** (83 sessions, 29 with an arrival, 5,974 objects, positive rate 0.212):
  - GRU 0.905. It ties attention (0.914, +0.008 [−0.006, +0.014]), GBT (0.911–0.912), the other
    GRU variants and the fitted Kalman (0.893, −0.013 [−0.023, +0.002], Holm p 0.53).
  - The GRU beats `nearest` (−0.016 [−0.026, −0.002]; AUPRC −0.060, Holm p < 0.001), logistic
    regression, the declared Kalman, the two-step CVM, nearest-in-corridor, TTC and K0.
  - **The §4.16 contradiction is resolved:** the 1 s fit beats the declared Kalman by AUROC +0.017
    [+0.009, +0.026] (Holm p 0.003) and AUPRC +0.044 [+0.026, +0.113].
- **Replay, physics predictor, paired before/after at fixed thresholds:** false alerts/min fall
  by 5.5–13.6 and precision rises by 0.04–0.14 (every CI excludes 0). Recall `warned` falls by
  0.14–0.16 (CIs exclude 0): smoother velocities predict fewer entries.

**Lead-based tuning** (`threat_engine_replay/corridor_tuned/`). New operating rules
`early_budget_{2,5,10}` maximise `warned_before` (warned ≥ 0.2 s ahead) within each false-alert
budget on validation. `--announcer-sweep` also tunes the announcer for v2_gru, v2_kalman and
nearest-in-corridor: 10 thresholds × cooldown {2, 4, 8} s × switch margin {1, 1.25, 2} × stale
drop {0.6, 1.2} s. Test has 94 arrivals; the opportunity window has a median of 2.2 s, 29% of
arrivals are seen < 1 s ahead and 26% ≥ 4 s ahead.

| Arm | rule | knob (threshold / announcer) | warned | warned ≥ 0.2 s | warned ≥ 1 s | false/min | precision | alerts/min |
|---|---|---|---|---|---|---|---|---|
| v2_gru | early_budget_5 | 0.5 / cd 4 / sm 1 / st 0.6 | 0.37 [0.26, 0.57] | 0.19 [0.11, 0.36] | 0.06 [0.01, 0.13] | 7.0 [4.8, 9.5] | 0.58 [0.47, 0.66] | 16.7 |
| v2_gru_nopose | early_budget_5 | 0.5 (default announcer) | 0.36 [0.26, 0.57] | 0.18 [0.11, 0.33] | 0.04 [0.00, 0.10] | 6.0 [3.8, 8.5] | 0.61 [0.49, 0.71] | 15.2 |
| v2_kalman | early_budget_5 | 0.7 / cd 4 / sm 1 / st 0.6 | 0.27 [0.14, 0.46] | 0.17 [0.08, 0.28] | 0.07 [0.02, 0.14] | 7.2 [4.7, 9.9] | 0.54 [0.43, 0.63] | 15.7 |
| nearest_in_corridor | early_budget_5 | 0.5 / cd 8 / sm 2 / st 0.6 | 0.17 [0.09, 0.30] | 0.07 [0.03, 0.16] | 0.03 [0.00, 0.09] | 7.4 [4.4, 10.3] | 0.44 [0.32, 0.55] | 13.1 |
| v2_gru | early_budget_10 | 0.4 / cd 2 / sm 1 / st 1.2 | 0.34 [0.21, 0.61] | 0.22 [0.10, 0.46] | 0.05 [0.01, 0.12] | 10.3 [7.7, 13.4] | 0.51 [0.39, 0.60] | 20.9 |
| v2_gru | early_budget_2 | 0.9 / cd 4 / sm 1 / st 0.6 | 0.15 [0.08, 0.28] | 0.04 [0.00, 0.11] | 0.02 [0.00, 0.07] | 3.9 [2.0, 6.2] | 0.64 [0.49, 0.77] | 10.9 |
| v1_k0_announcer | early_budget_10 | 12 | 0.39 [0.28, 0.59] | 0.20 [0.11, 0.31] | 0.11 [0.05, 0.17] | 10.2 [7.7, 12.7] | 0.39 [0.25, 0.50] | 16.9 |

**Reading.**
- **Tuning on lead doesn't create lead.** The `early_*` points are the same as, or next to, the
  `budget_*` points. The best arms warn about 0.19–0.22 of arrivals ≥ 0.2 s ahead and ≤ 0.07 ≥ 1 s
  ahead, while 71% are seen ≥ 1 s ahead. The limit is the predictors: they rarely put an
  object above threshold early enough.
- **The announcer barely matters.** At equal threshold, the best announcer setting beats the
  default (cooldown 4 s, switch margin 1.25, stale 0.6 s) by at most +0.06 `warned_before` on
  validation, and by ≤ +0.03 on test. Switch margin 1.0 is chosen most often. The announcer
  defaults stay as declared.
- **Validation under-predicts test false alerts.** Every point chosen for ≤ 5 false/min on 47
  validation sessions gives 6.0–7.4 on test. Budgets should be read as "about", or chosen with a
  margin.
- **Ego-motion removal still doesn't help:** the no-pose GRU matches the pose GRU on recall and
  lead, with fewer false alerts (6.0 vs 7.0) and higher precision (0.61 vs 0.58), with overlapping CIs.
- **v1's ≥ 1 s rate (0.11) is the highest**, but at precision 0.39. It speaks about everything
  near, so some of it is early by chance.
- **Next, for lead:** train the GRU on an objective that rewards early probability mass (e.g. a
  loss weighted by time-to-arrival, or a "will arrive in ≥ 1 s" target), and give the announcer a
  "rising P(arrive)" trigger instead of a fixed threshold.

### 4.18 Head-camera Stage-1 rerun: current numbers (2026-10-20)

Stage 1 was re-run on all 462 walks with the head camera's fx and the fixed size-check depth lookup
(architecture §9 items 8–9; `notebooks/colab_rebuild.ipynb`, `RUN_TAG = "headfx"`; the replay and
before/after comparison were run locally after the Colab GPU quota ran out). Methods, settings and
the split rule are those of §4.16–4.17. 421 walks have samples; the official split gives 288 train,
48 val and 85 test sessions. Every number in §4.13–4.17 is superseded by this section.

**Kalman noise (1 s-ahead fit).** The fit chose the Student-t update (ν = 4, gate 60): acceleration
σ 0.84 m/s² (static classes 1.58), depth σ 0.012 + 0.074·d m, bearing σ 0.28°, initial speed σ
0.86 m/s. Validation 1 s-ahead log-likelihood −3.09 vs −3.51 for `declared()`, so it is the default.
Caveat: the two are scored on slightly different forecast sets (905 vs 848), because gating changes
which tracks survive long enough to forecast.

**Ladder** (test once; 85 sessions, 24 with an arrival, 6,628 objects, positive rate 0.161):
- **GRU 0.911.** It ties GBT with history (0.917, +0.006 [−0.005, +0.014]), attention (0.916), GBT
  now (0.912) and the other GRU variants; no Holm-adjusted p is below 0.05.
- **The GRU beats** logistic regression (−0.024 [−0.039, −0.010]), `nearest` (−0.025 [−0.041,
  −0.006]; AUPRC −0.122, Holm p < 0.001), the two-step CVM (−0.031), the declared Kalman (−0.035),
  nearest-in-corridor (−0.039), the fitted Kalman (−0.048), TTC and K0.
- **Contradiction with §4.17:** the fitted Kalman no longer beats the declared one. AUROC −0.013
  [−0.022, +0.001] (Holm p 0.30), AUPRC +0.027 [+0.010, +0.082] (Holm p 1.0): no credible
  difference either way. The validation likelihood win does not carry over to the ranking.
- **Static classes** (463 objects): simple rules rank them at least as well as the GRU
  (nearest-in-corridor +0.082 [+0.002, +0.147]), but nothing survives Holm.

**Replay, physics predictor, before/after at fixed thresholds** (fitted − declared): false alerts/min
fall by 3.1–5.7 and precision rises by 0.04–0.05 (all CIs exclude 0). Recall falls by 0.03–0.11; for
the no-pose arms the CIs exclude 0.

**Replay, operating points** (`threat_engine_replay/corridor_headfx/`; 79 test arrivals; opportunity
window median 2.6 s, 24% seen < 1 s ahead, 33% ≥ 4 s ahead):

| Arm | rule | knob (threshold / announcer) | warned | warned ≥ 0.2 s | warned ≥ 1 s | false/min | precision | alerts/min |
|---|---|---|---|---|---|---|---|---|
| v2_gru | early_budget_5 = budget_5 | 0.6 / cd 8 / sm 1 / st 0.6 | 0.35 [0.23, 0.52] | 0.14 [0.06, 0.25] | 0.04 [0.00, 0.07] | 5.6 [3.4, 8.1] | 0.55 [0.40, 0.69] | 12.6 |
| nearest_in_corridor | early_budget_5 = budget_5 | 0.95 / cd 2 / sm 1 / st 0.6 | 0.23 [0.11, 0.35] | 0.10 [0.02, 0.19] | 0.03 [0.00, 0.04] | 4.9 [3.1, 7.0] | 0.39 [0.25, 0.52] | 8.1 |
| v2_gru | budget_10 | 0.4 / cd 8 / sm 1 / st 0.6 | 0.46 [0.36, 0.64] | 0.23 [0.14, 0.37] | 0.06 [0.03, 0.13] | 7.2 [4.4, 10.2] | 0.53 [0.39, 0.65] | 15.1 |
| v2_gru | early_budget_10 | 0.3 / cd 8 / sm 1 / st 0.6 | 0.46 [0.36, 0.64] | 0.24 [0.16, 0.40] | 0.09 [0.04, 0.21] | 8.7 [5.8, 11.9] | 0.49 [0.36, 0.62] | 16.9 |
| v2_kalman | budget_10 | 0.8 / cd 4 / sm 1.25 / st 0.6 | 0.38 [0.25, 0.57] | 0.23 [0.14, 0.40] | 0.09 [0.04, 0.19] | 9.8 [6.7, 13.2] | 0.43 [0.31, 0.53] | 17.1 |

**Reading.**
- **Only v2_gru and nearest-in-corridor meet a 5 false/min budget on validation**; nothing meets
  2/min. The Kalman, physics and no-pose arms only fit within 10/min. v2_gru is the only engine
  with precision above 0.5 at ≤ 5 false/min.
- **Lead is still the limit.** At ≤ 5 false/min, 0.14 of arrivals are warned ≥ 0.2 s ahead and
  0.04 ≥ 1 s ahead, though 76% are seen ≥ 1 s ahead. Relaxing to 10/min roughly doubles the
  ≥ 0.2 s share (0.23–0.24). The §4.17 next steps for lead stand.
- **Validation and test false alerts agree better:** the ≤ 5/min points give 4.9–5.6 on test
  (vs 6.0–7.4 in §4.17).
- **Viewer default:** `tools/build_session_viewer.py` uses threshold 0.6 with cooldown 8 s, switch
  margin 1, stale 0.6 s (the v2_gru `early_budget_5` point).

### 4.19 Scene context from walkable-surface segmentation (2026-10-17)

> **Superseded (depth-decoding bug).** Numbers in this section that use depth were computed on scrambled depth maps and are being re-run (`pending_work.md`, Integration). The scene-context flags and context replays also take object bearings from the Stage-1 datasets, so they are re-run on the head-camera rerun (§4.18): both are done ("Scene context, oracle, …" and "Scene context, model, on correct depth and the head-camera rerun", below). The edge alerts use only masks and depth. The segmentation model's pixel scores don't use depth and stand.


**Taxonomy and training data** (`src/segmentation/labels.py`, `tools/build_sanpo_segmentation.py`).
SANPO's 31 panoptic classes map to 8: walkable (sidewalk, paved trail, other walkable surface),
crosswalk, road (+ railway), curb, stairs, non-walkable ground (inaccessible surface, water,
terrain, vegetation), obstacle (all vertical classes) and sky; unlabeled is ignored (255). Masks are
stored as raw SANPO ids and remapped by the Ultralytics YAML's `label_mapping`, so the scheme can
change without rebuilding. All 237 masked SANPO-Real sessions are used, each with masks on one
camera (head ZED-M or chest ZED 2i): SANPO's official test sessions and the replay's test sessions
form the test split, the replay's validation sessions plus a seeded 15% of other train sessions
form validation. Validation and test hold HUMAN_ANNOTATED frames only (every 6th frame), as in the
SANPO paper; training adds every 6th propagated frame.

**Model and metrics** (`training/scripts/*_sem_walkable.py`, `evaluation/segmentation_metrics.py`).
YOLO26n-sem from Cityscapes weights at imgsz 640, trained on the M4 (MPS); results and the model
choice are below ("Trained model, 2026-10-18"). Reported, per
camera and overall, with session-bootstrap CIs: mIoU and per-class IoU; **false-safe rate** (unsafe
ground, i.e. road, curb, stairs or non-walkable, predicted walkable or crosswalk, over all unsafe
pixels), a generalisation of SENSATION-DS's road-as-sidewalk rate, which is reported too; the
opposite (nuisance) error; **road as walkable** (ground-truth road pixels predicted walkable or
crosswalk, over all ground-truth road pixels); and boundary F at ±3 px for curb and stairs, the thin edges that mIoU
barely weighs.

**From a class map to `SceneContext`** (`src/segmentation/context.py`). The ground under an object
is the majority ground class among pixels in its image columns whose depth is within ±15% (at least
±0.3 m) of the object's own depth. This needs only the object's bearing and distance, so it works on
the GRU dataset without boxes. The walker's surface is the majority ground class in the bottom-centre
of the image. Flags: *in path* = object and walker both on walkable ground; *near a boundary* = the
object's (doubled-width) ground is at least 15% walkable and 15% unsafe, or 10% curb; *other surface*
= walker on walkable ground, object on road or non-walkable ground. All thresholds are declared.

**Oracle check: SANPO's own masks in place of a model.** Head-camera masks exist for 47 of the 364
replay sessions (32 train / 5 val / 10 test); the tracks come from the head camera, so the 130
chest-masked sessions can't be used without re-running Stage 1 on the chest stream.

- *Do the flags point the right way?* (`evaluation/context_association.py`, all 47 sessions, 5,682
  object-frames; `evaluation/benchmarks/scene_context/oracle_gt_2026_10_17/`.) P(arrive within 4 s)
  by flag, as a risk ratio raw and Mantel-Haenszel-adjusted over distance bins (0/3/6/10 m),
  session-bootstrap 95% CI:

  | Flag | declared × | share | RR | RR, distance-adjusted |
  |---|---:|---:|---|---|
  | in path | 1.5 | 0.68 | 3.91 [1.73, 12.95] | 1.56 [0.98, 4.17] |
  | near a boundary | 1.25 | 0.36 | 0.44 [0.17, 0.71] | **0.83 [0.41, 0.99]** |
  | other surface | 0.5 | 0.21 | 0.29 [0.06, 0.67] | 0.70 [0.20, 1.06] |

  "In path" and "other surface" agree in direction with their multipliers; much of their raw
  effect is distance (objects on the walker's surface are closer). **"Near a boundary" points the
  wrong way:** objects at a surface edge arrive *less* often, so its ×1.25 is contradicted. Most
  such objects are at the sidewalk's edge, beside the walker's path, not crossing it; the flag
  can't tell an object about to step off a curb from one standing beside it. "In path" holds 68% of
  objects, so it separates little. The multipliers are not changed on this evidence, because 10 of
  the 47 sessions are test sessions; a revision must be fitted on train sessions only and declared.
- *Does context change the alerts?* (`tools/replay_threat_engine.py --context-dir`, 10 test / 5
  val sessions, 20 arrivals; `evaluation/benchmarks/threat_engine_replay/oracle_context_2026_10_17/`.)
  With the GRU at the validation-chosen threshold, context on vs. off: warned 0.40 vs 0.35, paired
  difference +0.05 [0.00, +0.07], at the same false-alert rate (9.45/min). The physics arm is
  unchanged. Twenty arrivals cannot settle it; the powered test needs the trained model's masks on
  all 71 test sessions.

**Revising "near a boundary", on train sessions only (2026-10-17).** The hypothesis was that the flag
is right only for objects moving toward the walker's path. *Approaching* is a closing speed on the
walker's centre line of at least 0.3 m/s over the last 3 steps (0.6 s). The lateral offset is
distance · sin(bearing) in the camera frame (`context.approach_speed`). The decision rule was fixed
before looking: keep a motion-qualified flag only if its distance-adjusted RR on train is above 1;
otherwise make the flag neutral. On the 32 train sessions (4,121 observations,
`scene_context/gt_train_2026_10_17/`):

| Flag | share | RR, distance-adjusted |
|---|---:|---|
| near a boundary | 0.38 | 0.75 [0.18, 0.96] |
| approaching (motion only) | 0.26 | 1.00 [0.76, 1.29] |
| near a boundary and approaching | 0.09 | 0.80 [0.21, 1.15]; 0.75 [0.21, 1.48] adjusted over distance × approaching |

Neither variant raises the arrival rate, so `CONTEXT_NEAR_BOUNDARY` is now **1.0**. The flag is
still computed and reported, but carries no weight. Motion in the camera frame includes head turns;
fused ground-plane lateral speed may separate better and is a follow-up once the fitted tracker is
merged.
- **Held-out check, once** (5 val + 10 test sessions, 1,561 observations,
  `gt_val_test_2026_10_17/`): near a boundary 1.04 [0.12, 6.22], consistent with neutral.
- **Oracle replay with the neutral weight**
  (`threat_engine_replay/oracle_context_neutral_boundary_2026_10_17/`): GRU context on vs. off,
  warned +0.05 [0.00, +0.07], false alerts/min +0.50 [0.00, +1.35]. Under ×1.25 the false-alert
  change was 0.00. Both are within noise at 20 arrivals.

**Scene context, oracle, on correct depth and the head-camera rerun (2026-10-20).** This supersedes
the 2026-10-17 oracle check and near-boundary revision above, which used scrambled depth and the
chest-camera Stage 1. Same masks, flags, thresholds and decision rule. Head-camera masks now cover
53 of the 421 replay sessions (35 train / 8 val / 10 test). Distance-adjusted RR, session-bootstrap
95% CI (`scene_context/gt_headfx_train_2026_10_20/`, `gt_headfx_val_test_2026_10_20/`,
`gt_headfx_2026_10_20/`):

| Flag | declared × | train (4,145 obs.) | val + test, once (1,753) | all 53 (5,898) |
|---|---:|---|---|---|
| in path | 1.5 | 1.18 [0.86, 2.33] | 1.55 [0.42, 2.03] | 1.32 [0.93, 2.09] |
| near a boundary | 1.0 | 0.60 [0.23, 0.83] | 0.80 [0.38, 3.37] | 0.69 [0.42, 0.87] |
| near a boundary and approaching | — | 0.51 [0.00, 0.84] | 0.69 [0.13, 2.65] | 0.55 [0.21, 0.81] |
| other surface | 0.5 | 1.01 [0.16, 1.38] | 0.67 [0.00, 2.73] | 0.89 [0.38, 1.22] |
| approaching (motion only) | — | 0.86 [0.59, 0.99] | 0.96 [0.48, 2.35] | 0.88 [0.67, 0.97] |

- **Near a boundary stays ×1.0.** Under the declared rule, neither the flag nor its approaching
  variant has an RR above 1 on train. The held-out check is consistent with neutral. Same
  conclusion as on scrambled depth.
- **In path** points the right way but its CI includes 1. **Other surface** is near 1, so its
  ×0.5 is not supported. Changing it would need a declared rule and train sessions; none was
  declared, so the multipliers stay.
- **Oracle replay, context on vs. off** (`threat_engine_replay/oracle_context_headfx_2026_10_20/`;
  10 test sessions, 27 arrivals; GRU `corridor_h8_seed0`, the run behind §4.18). At `budget_5` both
  arms choose threshold 0.4. With context, warned is unchanged (0.30), false alerts/min rise by
  +1.49 [0.00, +3.86] and precision changes by −0.04 [−0.20, +0.02]. At `early_budget_5` the arms
  choose different thresholds (off 0.7, on 0.6), which gives warned +0.19 [0.00, +0.21] at false
  alerts/min +0.50 [0.00, +1.23]. The difference comes from the threshold, not from context. At a
  fixed threshold of 0.6, context gives 0.26 warned and 3.0 false/min, against 0.15 and 2.5 without
  it. In path (×1.5) covers 75% of objects, so context mostly scales scores up, much like a lower
  threshold. Twenty-seven arrivals cannot separate the two. The powered test is the model's context
  on all 85 test sessions.

**Scene context, model, on correct depth and the head-camera rerun (2026-10-20).** This supersedes
the 2026-10-18 model run. The class maps come from YOLO26n-sem @640 (§6.5 of the paper,
`cpe_yolo26n_sem_walkable/weights/best.pt`) instead of SANPO's masks. Everything else is the same as
the oracle paragraph above: frames, depth, flags and multipliers. The model ran on all 133 val/test
sessions (5,829 frames; `data/sanpo_scene_context_v2/cpe_yolo26n_sem_walkable/`). It ran on CPU
(PyTorch, Ultralytics 8.4.178), not on the M4's GPU (MPS) it was trained on or the Core ML export.
Core ML matches PyTorch on 0.99–1.00 of pixels (§6.5), so the masks should differ only at the
margin. Distance-adjusted RR, session-bootstrap 95% CI
(`scene_context/model_headfx_val_test_2026_10_20/`, `model_headfx_val_test_headmasked_2026_10_20/`):

| Flag | declared × | model, all 133 (10,794 obs.) | model, 18 head-masked (1,753) | oracle, same 18 |
|---|---:|---|---|---|
| in path | 1.5 | 1.11 [0.82, 1.30] | 1.19 [0.46, 1.76] | 1.55 [0.42, 2.03] |
| near a boundary | 1.0 | 0.59 [0.39, 0.82] | 0.75 [0.38, 2.06] | 0.80 [0.38, 3.37] |
| near a boundary and approaching | — | 0.59 [0.36, 0.92] | 0.60 [0.00, 2.16] | 0.69 [0.13, 2.65] |
| other surface | 0.5 | 0.86 [0.61, 1.25] | 0.85 [0.00, 2.94] | 0.67 [0.00, 2.73] |
| approaching (motion only) | — | 0.78 [0.68, 0.91] | 0.96 [0.48, 2.35] | 0.96 [0.48, 2.35] |

- **The model reproduces the oracle's context.** On the 18 sessions that have both (970 frames;
  `agreement_vs_gt.json`), the walker's surface agrees in 0.967 of frames and each object's surface
  in 0.908. Per (frame, track), in path agrees 0.894 (κ 0.72), near a boundary 0.845 (κ 0.51) and
  other surface 0.901 (κ 0.58), with the same flag shares (0.73 / 0.20 / 0.14 for the model,
  0.77 / 0.20 / 0.13 for the oracle). The motion-only row is identical to the oracle's, as it should
  be, since it uses no masks.
- **Near a boundary points the wrong way, now with a CI that excludes 1** (all 133 sessions). This
  is held-out data, so it confirms the train-only decision to keep it neutral (×1.0); nothing is
  re-tuned on it. In path and other surface have CIs that include 1, so neither of their
  multipliers is supported or contradicted.
- **Replay, context on vs. off** (`threat_engine_replay/model_context_headfx_2026_10_20/`; all 85
  test sessions, 79 arrivals; GRU `corridor_h8_seed0`). Every test session has model context, so
  the context-off GRU arm is the §4.18 replay and reproduces it (`budget_5`: threshold 0.6, warned
  0.35, 5.71 false/min). Paired session-bootstrap differences, each arm at its own
  validation-chosen knob:

  | Rule | threshold off / on | Δ warned | Δ false alerts/min | Δ precision |
  |---|---|---|---|---|
  | `budget_5` (= `early_budget_5`) | 0.6 / 0.7 | +0.01 [−0.13, +0.08] | +0.52 [0.00, +1.02] | −0.03 [−0.06, −0.01] |
  | `budget_10` | 0.4 / 0.6 | −0.06 [−0.19, −0.01] | −0.35 [−0.83, 0.00] | −0.01 [−0.04, +0.01] |
  | `early_budget_10` | 0.3 / 0.4 | +0.03 [0.00, +0.05] | +0.09 [−0.61, +0.74] | 0.00 [−0.02, +0.02] |

  At every budget the context arm picks a higher threshold: in path (×1.5) covers about two-thirds
  of objects, so context mostly scales scores up. Along the operating curves the two arms overlap
  (for example, off 0.6: 0.35 warned at 5.7 false/min; on 0.8: 0.32 at 5.9; on 0.7: 0.37 at 6.2).
  The deltas are small and change sign between budgets: **with the declared multipliers, context
  does not change the alerts measurably**, as in the oracle replay and the 2026-10-18 run. Physics
  arms meet no budget on validation, with or without context.

**Edge-ahead alerts, oracle, on correct depth (2026-10-19).** This supersedes the 2026-10-17
paragraph below, which used scrambled depth. Same rule, sessions and step-off reference.

| Step-off definition | step-offs | alerts / min | warned | median lead (s) | alerts followed |
|---|---:|---|---|---|---|
| declared | 45 | 0.68 [0.40, 1.01] | 0.18 [0.06, 0.33] | 0.8 [0.4, 1.6] | 0.33 [0.13, 0.54] |
| sustained ≥ 2 frames | 26 | 0.68 [0.40, 1.01] | 0.19 [0.04, 0.40] | 0.8 [0.4, 2.8] | 0.22 [0.06, 0.41] |

Sources: `edge_alerts/gt_depthfix_2026_10_19/`, `gt_depthfix_sustained_2026_10_19/`.
- **By kind.** Stairs: all 3 step-offs warned. Road: 12% of 42 warned. Curb: 8 alerts, no curb
  step-offs (the bottom-centre patch never registers curb).
- **Misses.**
  - 18 of 37: the edge was first seen 5.6–9.5 m ahead and never confirmed inside 5 m. The nearest
    ground a head-mounted camera sees is about 2.6 m ahead, so an edge spends only about 1.5 s
    between 5 m and the bottom of the image.
  - 10: seen inside 5 m but not confirmed, or the episode had already alerted.
  - 7: no edge in the centre band.
  - 2: only non-walkable edges.
- **Train-only sweep** (`--sweep`, `edge_alerts/gt_depthfix_trainsweep_2026_10_19/`). Declared
  before the run:
  - grid: range {5, 7, 10} m × confirm {1, 2} frames;
  - choose the most warned on the 43 official-train sessions (38 step-offs) at ≤ 1 alert/min.

  Only the current rule meets the budget, so it stays: 5 m, 2 frames.

  | Range, confirm | alerts / min | warned |
  |---|---:|---:|
  | 5 m, 2 (kept) | 0.70 | 0.18 |
  | 7 m, 2 | 1.07 | 0.34 |
  | 5 m, 1 | 1.30 | 0.37 |
  | 10 m, 2 | 1.85 | 0.50 |
  | 7 m, 1 | 2.18 | 0.58 |
  | 10 m, 1 | 3.20 | 0.76 |

  Warning more step-offs costs alerts, about one extra alert per minute for each +0.2 warned. The
  right budget is a user question, for the listening study. The 10 test sessions hold only 7
  step-offs (1 warned), too few to compare settings.

**Edge-ahead alerts, oracle (2026-10-17), superseded.** `tools/evaluate_edge_alerts.py` runs every 6th source
frame (2.5 Hz) of the 53 sessions with head-camera masks: 26.6 min, about 30 s per session. The
class maps are SANPO's own masks, so this scores the rule and the edge geometry, not a model.

*Rule* (`src/segmentation/edge_alert.py`, declared before the run):
- curb, stairs or road within 5 m;
- seen in 2 consecutive frames;
- one alert per episode, re-armed after 2 s with no edge.

*Reference: step-off.* SANPO has no annotated "stepped off the curb" event, so the reference is a
step-off: the bottom-centre ground turns from walkable to curb, stairs or road after at least 2
walkable frames. Warned = an alert in the 10 s before.

| Step-off definition | step-offs | alerts / min | warned | median lead (s) | alerts followed by a step-off |
|---|---:|---|---|---|---|
| declared (any duration) | 45 | 0.79 [0.49, 1.12] | 0.29 [0.13, 0.47] | 1.2 [0.4, 2.8] | 0.48 [0.24, 0.69] |
| sustained ≥ 2 frames (sensitivity) | 26 | 0.79 [0.49, 1.12] | 0.35 [0.14, 0.59] | 1.6 [0.8, 6.0] | 0.33 [0.12, 0.52] |

- **Why the sensitivity row exists.** Under the declared definition, 20 of the 32 misses lasted
  one frame: mask flicker at crosswalk/road boundaries, not a step. The sustained variant drops
  these. It was added after seeing the misses and is reported only as a sensitivity check.
- **By kind (declared).**
  - Stairs: 3 alerts, all 3 step-offs warned.
  - Road: 10 alerts, 42 step-offs, 24% warned.
  - Curb: 8 alerts; the bottom-centre patch never registers curb as the walker's surface, so
    curbs have no step-offs to score against.
- **Misses (declared).**
  - 11: the edge was only ever seen beyond 5 m.
  - 8: no edge was seen in the 10 s before at all. The walker turned onto the road, or the edge
    was outside the centre band.
  - 12: a near edge was seen but not confirmed in 2 consecutive frames, or its episode had already
    alerted.
- **Not tuned.** Every session was used, test included, so the rule was not changed on these
  numbers. Widening the band, raising the range or confirming in 1 frame would each trade
  nuisance alerts for warned share; that choice needs a train-only sweep.
- **Model masks.** The same tool takes a trained model (`--source <best.pt>`), which gives the
  model-vs-oracle gap (below). `--reference gt` takes the step-offs from SANPO's masks while the
  alerts come from the model. Without it, a model is graded against its own masks.

**Trained model, 2026-10-18 (branch `v2-seg-training`, MacBook Air M4, MPS).**

*Speed pilot* (`segmentation/pilot_mps_2026_10_17/`; timed while the dataset build was running):

| config | s/iteration | memory |
|---|---:|---:|
| n@640, batch 8 | 0.62 | 3.6 GB |
| s@640, batch 8 | 1.16 | 5.9 GB |
| s@768, batch 8 | 3.09 (swapping) | 7.9 GB |
| s@768, batch 4 | 1.46 | 4.7 GB |

*Choice, declared before training* (commit `6fd1412`):
- **Config.** YOLO26n-sem @640, batch 8, 20 epochs, patience 5, from Cityscapes weights.
- **Why.** The 12 h budget, and n is the variant with a measured iPhone latency (4.9 ms).
- **Run.** It early-stopped at epoch 8 (6.3 h, about 47 min per epoch). Validation mIoU by epoch:
  0.522, 0.543, **0.570**, 0.550, 0.567, 0.554, 0.557, 0.568. Validation CE fell to 0.433 at
  epoch 3, then rose (0.47–0.54). best.pt = epoch 3, chosen on validation mIoU.

*Test, scored once* (48 sessions, 3,850 frames; `segmentation/cpe_yolo26n_sem_walkable_test/`):

| | mIoU | false-safe | road as walkable | boundary F curb / stairs |
|---|---|---|---:|---|
| all | 0.615 [0.540, 0.641] | 0.044 [0.029, 0.062] | 0.173 | 0.295 / 0.062 |
| chest (2,561 frames) | 0.604 | 0.058 | 0.163 | 0.305 / 0.067 |
| head (1,289 frames) | 0.554 | 0.026 | 0.222 | 0.283 / — (no stairs in head test frames) |

- **Road as walkable** = ground-truth road pixels predicted walkable or crosswalk, over all
  ground-truth road pixels.
- **IoU by class:** walkable 0.828, crosswalk 0.590, road 0.494, curb 0.239, stairs 0.474,
  non-walkable 0.686, obstacle 0.741, sky 0.868.

*Deviation from the declared follow-up.* The plan was to try s@640 if test scores showed the model
to be the bottleneck. We kept n instead, for three reasons:
1. Validation mIoU peaked at epoch 3 and validation loss then rose, so the model was overfitting.
   More capacity on the same data is unlikely to help.
2. Thin edges depend on input resolution more than on model size, and s at 640 sees the same
   pixels.
3. Choosing a second model from test scores would make the final model test-informed.

The follow-up is n at a larger imgsz on a GPU machine. It has not been run.

*Core ML.* `weights/best.mlpackage`: FP16 mlprogram with the argmax in the graph, output
(1, 640, 640) class ids, 3.1 MB. Pixel agreement with PyTorch on 4 test frames was 0.991–1.000.
The first export attempt failed inside coremltools 9.0 (`only 0-dimensional arrays can be
converted to Python scalars`; torch 2.14 is untested with it). An identical retry succeeded.

**Superseded (scrambled depth).** *Scene context with the model* (`build_scene_context.py --source best.pt --all-sessions --imgsz 640`;
112 replay val + test sessions, none of them in segmentation training):
- **Flags.** On the oracle's 15 sessions (`scene_context/model_val_test_headmasked_2026_10_18/`,
  1,561 observations), distance-adjusted RR, model vs. oracle:
  - in path: 2.68 [2.20, 10.34] vs 2.64 [0.00, 7.64];
  - near a boundary: 1.15 [0.14, 5.97] vs 1.04 [0.12, 6.22];
  - other surface: 0.37 [0.00, 0.44] vs 0.38 [0.00, 1.67].

  The model matches the oracle. On all 112 sessions (`model_val_test_2026_10_18/`, 6,748
  observations) none is significant: 1.12 [0.78, 1.55], 1.04 [0.61, 1.58] and 0.89 [0.40, 1.69].
- **Replay, context on vs. off** (`threat_engine_replay/model_context_2026_10_18/`; 71 test
  sessions, 56 arrivals; paired). At `budget_10`: warned +0.04 [−0.06, +0.08], false alerts/min
  +0.41 [−0.30, +1.44]. At `budget_2` and `budget_5`, the context arm has no setting that meets
  the rule on validation. No measurable effect.
- **Edge alerts** (`edge_alerts/cpe_yolo26n_sem_walkable_*_2026_10_18/`). 43 of the 53 sessions
  are in segmentation training, so only the 10 official-test sessions score the model fairly.
  With step-offs taken from SANPO's masks for both:

  | | warned | warned, sustained |
  |---|---|---|
  | model | 0.14 (1 of 7) | 0.00 (0 of 4) |
  | oracle | 0.43 (3 of 7) | 0.50 (2 of 4) |

  - Followed by a step-off: model 0.40, oracle 0.50.
  - Alerts per minute: model 1.00, oracle 0.80.
  - **The model's edge alerts are far below the oracle's.**
  - All 53 sessions: 0.22 [0.10, 0.37] vs 0.29 [0.13, 0.47].
  - Self-referenced, the model scores 0.22 (50 step-offs from its own masks). Self-referencing
    hides the gap.

### 4.20 The kinetic score and the severity term: literature check and a γ = 0 ablation (2026-10-20)

Asked whether K0 (`severity · v² / d`) is the right formula. v2 uses only one part of it: class
severity, compressed to `severity^γ` with γ = 0.25, as the "how bad" factor in
`priority = P(arrive) · severity^γ · context` (§4.18; `src/threat_engine/priority.py`).

**K0 as a ranking stays retired.** On the head-camera corridor ladder
(`baseline_ladder/…_corridor_w0.75_d3_h20/`) K0's AUROC is 0.537, against 0.911 for the GRU and
0.720 for TTC in the cone. On static objects it is 0.353, below chance: `v²` is zero for anything
that isn't closing, so K0 ranks a bollard dead ahead last. The geometric signal worth keeping is TTC,
which the safety envelope already uses.

**The priority's structure matches the literature.** Risk is defined as collision probability times
the harm of that collision in the ethics-of-risk planners for automated driving (Geisslinger et al.,
*Philosophy & Technology* 2021; *Nature Machine Intelligence* 2023, open-source code
`TUMFTM/EthicalTrajectoryPlanning`); `P(arrive) · harm` is that form. Criticality reviews
(Westhofen et al., *Arch. Comput. Methods Eng.* 2023) list TTC, Δv and energy-based severity as
separate measures, each valid in a different scenario class, which supports keeping TTC (envelope)
and harm (ranking) apart rather than fusing them into one formula, as K0 did.

**The severity term itself is not literature-backed.**
- *Harm is a function of Δv, not of mass.* Geisslinger's harm model computes the speed change from
  conservation of momentum, `Δv_walker = m_obj / (m_obj + m_walker) · v_rel`, and maps it through
  a logistic injury curve (GIDAS-fitted, P(MAIS 2+)). Mass enters only through that ratio, which
  saturates: for a pedestrian, a car, bus and truck all give Δv ≈ v_rel. Pedestrian injury curves
  are functions of impact speed (Rosén & Sander 2009; Tefft, *AAP* 2013). Our
  `(mass/70)^0.5`, then `^0.25`, is `mass^0.125`: it grows without bound and makes a bus 1.9× a
  person at any speed, which no injury model supports.
- *The mass table leaves out riders.* Bicycle (15 kg) and motorcycle (200 kg) carry the vehicle
  alone, so a cyclist's severity (0.56) is below a pedestrian's (1.0).
- *Vehicle injury curves don't cover the walker's main hazard.* They are fitted to crashes at car
  speeds. Most of what a blind walker meets (people, poles, benches, bollards, curbs) is hit at
  walking speed with the walker supplying the energy, where the harm is a trip or fall, not
  kinetic-energy injury. The static-hazard table (stairs, pothole) is the right idea for those, but
  its values are also declared.

**Ablation: does severity change the alerts?** Replay with `--severity-gamma 0` (rank by arrival
alone; flag added to `tools/replay_threat_engine.py`, default unchanged), all 85 test sessions,
79 arrivals, GRU `corridor_h8_seed0`, default announcer, thresholds chosen on validation
(`threat_engine_replay/severity_gamma0_headfx_2026_10_20/`). The γ = 0.25 arm is the context-off
GRU arm of `model_context_headfx_2026_10_20` (same sessions, same announcer). Not paired, so read
the CIs side by side:

| Rule | γ | threshold | warned | false alerts/min | precision |
|---|---:|---:|---|---|---|
| `budget_5` | 0.25 | 0.6 | 0.35 [0.23, 0.52] | 5.71 [3.47, 8.26] | 0.56 [0.41, 0.69] |
| `budget_5` | 0 | 0.5 | 0.34 [0.23, 0.46] | 6.14 [3.72, 8.92] | 0.54 [0.40, 0.67] |
| `budget_10` | 0.25 | 0.4 | 0.46 [0.36, 0.64] | 7.44 [4.67, 10.43] | 0.52 [0.39, 0.64] |
| `budget_10` | 0 | 0.3 | 0.44 [0.36, 0.60] | 8.48 [5.55, 11.58] | 0.49 [0.36, 0.62] |

The two operating curves overlap at every threshold. **Severity doesn't change the alerts
measurably on SANPO.** That is expected from the data: 73% of test objects are people (severity
1.0 by definition), only 11% of frames hold objects of more than one class, and vehicles rarely
arrive (2.3% of car observations vs. 10% of person observations). The severity weights can be
neither confirmed nor refuted here. They stay declared values and are reported as a limitation.

**Decision for the freeze.** No formula change: γ, λ and the tables were declared before the test
runs, and changing them now would be test-informed. Post-freeze, a principled replacement, declared
on train sessions before any replay:
`harm(c) = h(m_c / (m_c + 70 kg) · v_ref)` with rider-inclusive masses and one fixed reference
closing speed per class (so the GRU stays the only source of motion), plus a separate trip-hazard
weight for static classes.

## 5. Latency methodology

### 5.1 Measurement vs. simulation — kept strictly separate

Three kinds of latency number exist and must never be conflated in a paper:

- **Server measurement:** PyTorch/CUDA on the NVIDIA GB10 development host. Real, but not the target
  device. The 10-session SANPO result is avg p95 7.83 ms, worst session 9.74 ms.
- **Legacy analytical proxy (retired 2026-10-14):** the Jetson Orin Nano 8GB profile scaled GB10
  latency by 4.0× and added 3.0 ms (avg p95 34.31 ms). It was a screening heuristic for a target that
  is no longer planned, so don't report it as edge evidence.
- **On-device measurement (harness built 2026-10-17, not yet run):** `ios/CPETiming` is a SwiftUI app
  built by GitHub Actions on macOS and installed through TestFlight (no team Mac needed). It runs:
  - YOLO26n v3 (FP16 Core ML) on 300 bundled SANPO test frames;
  - the arrival GRU (`training/scripts/export_gru_coreml.py`; FP16, fixed [1, 32, 8, 9] input,
    softmax in the graph) on recorded SANPO test-split features.

  It writes per-stage times (decode/resize, detector, output read, GRU, total) as CSV rows and
  `os_signpost` intervals, with the thermal state and battery. The phone's camera and LiDAR are not
  used: frames and depth-derived inputs come from SANPO. Protocol:
  - a 60-frame warm-up, excluded;
  - 10 blocks of 150 frames paced at 15 FPS;
  - a 20–40 min sustained run.

  Reported: median, p95 and p99 per stage; p95 per 5-minute window; time in each thermal state;
  battery drop per hour (`ios/CPETiming/analyze_timing.py`). This is the only number that supports a
  "runs on the phone" claim. It times two models per frame, not the whole app, and SANPO frames come
  from a different camera than the phone's.

**Latency-aware replay.** The SANPO replay is open-loop, so latency can't change what the walker
does. It does change *when* an alert is heard. `tools/replay_threat_engine.py --latency-csv
<timing CSVs>` delays each announcement by a total drawn from the measured per-frame totals, seeded
per session (`--latency-ms` gives a fixed delay instead). Warned and lead time use the heard time;
precision is still judged at the frame that triggered the alert.

The GB10 development host: NVIDIA GB10, ARM64, 20 CPU cores, ~119 GiB unified memory, Ubuntu 24.04,
PyTorch 2.13 + CUDA 13.0, Ultralytics 8.4.95 (observed 2026-07-22).

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
`sanpo_dataset.md`, `evaluation/benchmarks/sanpo_edge_realtime/`.

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

> "I am writing a research paper on a phone-based hazard alert system for blind pedestrians, called
> the Composite Perception Engine (CPE). A YOLO26n detector fine-tuned on 17 street-hazard classes,
> with ByteTrack and metric depth, feeds a threat engine that ranks objects by three separate
> questions: *when* will it reach the walker (a small GRU arrival predictor over 1.6 s of track
> history), *how bad* would a collision be (class severity derived from real-world mass), and *what*
> does the scene say about it (walkable-surface segmentation). Every learned decision sits inside a
> deterministic physics safety envelope: an object with time-to-collision under 1 s in the walking
> path always triggers an alert. Alerts are scripted phrases from pre-rendered audio; there is no
> language model at runtime. The target device is an iPhone 17 Pro Max; evaluation replays the
> SANPO egocentric dataset with its recorded depth used as is (the product uses an external LiDAR),
> and no human ground-truth
> labels are needed except a small calibration set for a blinded referee.
>
> Please write an Academic Abstract (250 words max) and an Introduction. The Introduction should
> cover the limitations of current assistive technologies (too heavy for phones, too slow, or
> detecting objects without prioritising them) and state our contributions: (1) evidence that a
> classic kinetic threat score is not an arrival predictor (AUROC 0.57) while a 30k-parameter GRU
> reaches 0.89, (2) a when / how-bad / what decomposition of threat inside a physics safety
> envelope, and (3) a label-free, latency-aware replay evaluation on SANPO. Do not claim a live
> deployment or a user study."

### A.2 Related Work

> "Based on our edge-AI assistive technology system (CPE) for the visually impaired, write a
> 'Related Work' section. Compare our approach to three main existing paradigms:
> 1. Traditional sensory substitution devices (e.g., ultrasonic canes) which lack semantic
>    understanding.
> 2. Cloud-based LLM/VLM assistive tech (e.g., Be My Eyes, GPT-4V) which suffer from high latency and
>    require constant network connectivity, making them unsafe for real-time hazard avoidance.
> 3. Embedded CNN-based obstacle detectors (standard YOLO on mobile) which identify objects but fail
>    to prioritize physical threats (like a fast-moving bicycle vs a parked car).
> Emphasize how our approach — small learned predictors for arrival and scene context, combined
> with physics severity inside a deterministic safety envelope — prioritises threats on a phone
> without a language model in the loop. Use only citations from docs/related_work.md."

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
> - We process every 3rd source frame. Between processed frames, tracks are carried by the tracker.
> - Depth is SANPO's recorded depth, used as is; the product will use an external LiDAR sensor,
>   which is not modelled. State this explicitly. Distance (d) is the
>   median depth in the central patch of each box, masking closer overlapping objects. Object bearing
>   is computed from the box centroid pixel and the camera's horizontal field of view:
>   `bearing_deg = ((cx_px − frame_width/2) / (frame_width/2)) × (hfov_deg/2)`.
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

### A.4 Methodology: Threat Engine (when × how bad × what)

> "Write the Methodology subsection on the 'Threat Engine'. Explain that:
> - The engine replaces an earlier design in which a small language model picked the main threat;
>   it was dropped because a language model is slow (~500 ms), large (~1 GB) and unverifiable in a
>   safety loop.
> - Threat is decomposed into three questions answered by separate components: *when* (a GRU
>   arrival predictor reading each track's last 8 processed frames of distance, closing speed,
>   bearing, their per-step changes, confidence, box size, track age and class — never the kinetic
>   score — and outputting a distribution over arrival-time slices), *how bad* (class severity from
>   real-world mass, `behaviour(c) · (mass(c)/70 kg)^0.5`), and *what* (walkable-surface
>   segmentation trained on SANPO labels, giving whether the object is on the walker's path, on the
>   road, or near a curb).
> - The fused priority is `P(arrives within 4 s) × severity^γ × context` (γ = 0.25, declared), ties
>   broken by expected arrival time; the ignore threshold is tuned on validation sessions only.
> - Tracks are first fused per object with a constant-velocity Kalman filter in ground-plane
>   coordinates (§4.13), giving signed closing speed, time to the hazard zone and closest approach;
>   it can remove head motion using camera pose, but that step is not validated on SANPO yet.
> - An announcement manager decides whether to speak: overrides preempt, a new object must beat the
>   current one by 25%, the same object is not repeated within 4 s unless its priority rises 50%,
>   and messages that can't play within 0.6 s are dropped.
>
> **Note when drafting:** the GRU (§4.10–4.11), fusion, fused score and announcement manager are
> built and replay-evaluated (§4.12); segmentation ("what") is planned (`pending_work.md` P1). Write
> segmentation as method, not results, until it is run."

### A.5 Methodology: Physics Safety Envelope

> "Write the Methodology subsection on the 'Physics Safety Envelope'. Explain that:
> - Each tracked object is a fused track with distance, signed closing speed, bearing relative to
>   the walking direction, and a confirmation count.
> - If a confirmed track's TTC (distance / closing speed) is at most 1.0 s and it is inside the
>   forward walking cone (±30°), the system raises an immediate override alert from a cached audio
>   clip. A track keeps that protection for 2 frames of missing depth; zero or negative closing
>   speed never fires; the edge cases are unit-tested. This path contains no learned
>   component and cannot be suppressed or delayed by one.
> - Learned components only reorder the objects that are not in the envelope, so the system is
>   never less safe than its physics rule alone.
> - Override alerts bypass translation and any speech synthesis by construction (enforced in code,
>   not convention)."

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
>   added, and plain time-to-hazard as an external reference; (b) five label-free metrics (flicker
>   rate, rank stability under 2/5/10% depth perturbation, temporal smoothness, tie rate,
>   future self-consistency); (c) automatic encounter labels derived only
>   from *measured* future distance and bearing, used to eliminate variants rather than to select one.
> - All confidence intervals come from a **session-level** bootstrap, because frames within a walking
>   session are autocorrelated and frame-level intervals would be dishonestly narrow. State the
>   sample: a seeded 30% sample (139 of 462) of the SANPO-Real valid streams, 19,402 scored frames.
> - Report the actual 2026-08-26 result: `no-velocity` loses decisively (rank stability collapses
>   to 0.26–0.52), so velocity matters; `ttc` has higher flicker than K0, but its encounter-recall
>   deficit is an artifact of untracked ground blobs (§4.9 caveat 2) and must not be reported as a
>   win for K0. `linear`, `no-severity` and `size` tie with K0 on every label-free metric; the two
>   lambda arms are struck as not reproducible from current code. State plainly that kinematics
>   dominate every reordering on this corpus and that the tied terms are declared design choices,
>   not proven necessities.
> - The one irreducibly human question — whether gamma and the severity weights are *right* — is
>   settled by a **blinded forced-choice referee** restricted to the frames where two variants pick
>   different top objects (~5% of frames, which is what makes labelling affordable). Three VLM
>   referees from three model families, run locally, estimate labelling noise, not truth; report
>   pairwise Cohen's kappa between them and, critically, kappa against a human-labelled calibration
>   subset. State plainly that if human kappa is poor, the VLM numbers are not evidence. **Note when
>   drafting:** this referee step has not been run yet as of this writing (`pending_work.md` P2.4) — the
>   219 disagreement frames are exported and waiting.
> - If the learned-predictor experiment (§4.10) has been run, add a paragraph comparing K0 against two
>   learned encounter predictors trained on the same formula-free labels: a per-object GRU over 1.6 s
>   of track history, and the same GRU plus one self-attention block across the objects in a frame
>   (no positional encoding, so object order — which is K-sorted — cannot leak). Report per-object
>   AUROC/average precision and per-frame encounter_top1/flicker with session-bootstrap CIs, on all
>   frames and on the interaction slice, against the nearest-in-cone rule baseline as well as K0.
>   Attribute any GRU-vs-attention gap to scene context only if it survives a parameter-matched plain
>   GRU. State that the labels come from the same perception stack as the inputs. **Note when
>   drafting:** the first run (2026-09-28) is in §4.10.
>   - It found that the encounter label is 99.7% untracked depth blobs (§4.9 caveat 2).
>   - Do not cite corpus-wide encounter_top1 for any arm, K0 included, without stating that caveat.
>     On tracked objects only, K0, no-velocity and TTC tie."

### A.7 Results, Discussion & Conclusion

> "Using only the supplied measured results, write the Results, Discussion, and Conclusion sections.
> Report latency only as measured, naming the device: the GB10 server result (10 SANPO sessions,
> avg p95 7.83 ms) is not a phone result; report iPhone 17 Pro Max numbers only if they have been
> measured on the device (`pending_work.md` P3). Report the YOLO26n detector result (v3: mAP50 0.641,
> mAP50-95 0.459) and the v2→v3 training-protocol finding. Report the kinetic-score ablation (§A.6)
> and the arrival-prediction result on SANPO's official test split (71 sessions, §4.11): the GRU
> reaches AUROC 0.888 (0.887 in 5-fold cross-validation) vs. K0 0.571, TTC 0.661 and cone-restricted
> TTC 0.765, ties the nearest-in-cone rule (0.875), and has a steadier top pick (flicker 0.56 vs.
> 0.71); attention and ensembles add nothing significant. Report the system replay (§4.12): v2
> announces 17 times a minute with precision 0.59 vs. v1's 119 at 0.24, but warns about 26% of
> arrivals vs. 65%. Discuss that the ceiling is set by the input features, motivating scene context,
> and that the camera-pose check failed (§4.13), so ego-motion removal is not claimed. State the
> limitations: open-loop replay, SANPO depth used as is (external LiDAR not modelled), head-mounted camera, no
> user study, declared severity / announcer parameters. Do not claim a live deployment, phone FPS that
> was not measured, or results for components that have not been run."
