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
with depth read from the dataset rather than from the phone's LiDAR (`architecture.md` §4); no live
deployment or user study is run.

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
formula-free encounter labels, session bootstrap, blinded referee — is reused unchanged for v2.

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

### 4.13 Sensor fusion parameters and the camera-pose check

`src/threat_engine/fusion.py`: constant-velocity Kalman filter per track on the ground plane; white
acceleration noise σ = 1.5 m/s²; depth noise σ = 0.05 + 0.01·d² m; bearing noise 1°; initial
velocity σ = 3 m/s; χ² gate 11.62 (2 dof, 99.7%), with a track restarted after 3 consecutive gated
measurements; tracks dropped after 5 missed frames and confirmed after 3 updates. All are declared
values, not fitted.

Camera poses come from SANPO `camera_poses.csv` (ZED camera axes: x right, y up, looking along −z,
confirmed against walking direction; the alternatives are 80–150° off). The world up axis differs
between sessions, so it is estimated per session as the mean camera-up vector, and frames where
visual tracking was lost repeat the last good pose. **Validation failed so far:** with either
`camera_poses.csv` or `fixed_camera_poses.csv`, near (< 8 m) static objects move more per step in
the world frame (median 0.52 m) than in the camera frame (0.32 m), which should not happen if poses
and box depth agree. No frame offset or axis sign fixes it. Ego-motion removal is therefore reported
only as an ablation until this is explained (`pending_work.md` P0.2).

### 4.14 Scene context from walkable-surface segmentation (2026-10-17)

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
YOLO26s-sem from Cityscapes weights at imgsz 1024; not trained yet (lab server). Reported, per
camera and overall, with session-bootstrap CIs: mIoU and per-class IoU; **false-safe rate** (unsafe
ground, i.e. road, curb, stairs or non-walkable, predicted walkable or crosswalk, over all unsafe
pixels), a generalisation of SENSATION-DS's road-as-sidewalk rate, which is reported too; the
opposite (nuisance) error; and boundary F at ±3 px for curb and stairs, the thin edges that mIoU
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

## 5. Latency methodology

### 5.1 Measurement vs. simulation — kept strictly separate

Three kinds of latency number exist and must never be conflated in a paper:

- **Server measurement:** PyTorch/CUDA on the NVIDIA GB10 development host. Real, but not the target
  device. The 10-session SANPO result is avg p95 7.83 ms, worst session 9.74 ms.
- **Legacy analytical proxy (retired 2026-10-14):** the Jetson Orin Nano 8GB profile scaled GB10
  latency by 4.0× and added 3.0 ms (avg p95 34.31 ms). It was a screening heuristic for a target that
  is no longer planned, so don't report it as edge evidence.
- **On-device measurement (planned):** each runtime model is exported to Core ML and timed on the
  target iPhone 17 Pro Max over SANPO frames. Report median and p95, warm-up separately, and a
  sustained run of several minutes to expose thermal throttling. This is the only number that
  supports a "runs on the phone" claim (`pending_work.md` P3).

**Latency-aware replay.** The SANPO replay is open-loop, so latency cannot change what the walker
does. It does change *when* an alert would have fired. System metrics (warning lead time, missed
arrivals) are therefore computed after delaying each alert by the measured on-device latency.

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
> SANPO egocentric dataset with dataset depth standing in for phone LiDAR, and no human ground-truth
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
> - Depth comes from the SANPO dataset in our replay; on the target iPhone 17 Pro Max it would come
>   from the LiDAR scene depth via ARKit. State this substitution explicitly. Distance (d) is the
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
> limitations: open-loop replay, dataset depth standing in for phone LiDAR, head-mounted camera, no
> user study, declared severity / announcer parameters. Do not claim a live deployment, phone FPS that
> was not measured, or results for components that have not been run."
