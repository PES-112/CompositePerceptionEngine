# Changelog

All notable changes to the Composite Perception Engine (CPE) project will be documented in this file.

## [2026-08-27] - First Full-Pipeline Runtime Composition (Synthetic Data)

Closed the integration gap `pending_work.md` §5 called "the single most valuable integration
milestone remaining": every runtime component built so far — Threat Prioritizer, Cognitive Layer,
Physics Verification, Narration — is now called together, in order, in one process, and actually
produces audio-ready output. Runs on synthetic data only; real SANPO replay is still open.

### Added
- `src/cognitive_layer/stub.py` - rule-based, non-ML stand-in for SLM-1: picks the highest-K object
  among a frame's cognitive-route events as the semantic primary threat, with a canned reason string.
  Explicitly documented as a placeholder to be replaced, not extended, once a real model exists - the
  exact point where `SemanticEval` should come from a trained model instead.
- `src/pipeline/orchestrator.py` - `process_frame()`/`process_frames()`, the composition that did not
  exist anywhere in the repo before this: builds the physics ranking from every non-ignore object in
  a frame (reflex- and cognitive-route alike), gets a `SemanticEval` from the Cognitive Layer,
  adjudicates once via `PhysicsVerification`, and feeds any resulting `NarratorEvent` into
  `NarrationPipeline`. **Real gap found and fixed:** `src/reflex_layer/reflex.py` only ever calls
  `PhysicsVerification` for reflex-route events by design ("Cognitive events are intentionally left
  for the SLM path") - nothing before this module built that other half. A frame containing only
  cognitive-route objects (e.g. a near-static pothole with nothing fast-closing present) previously
  produced no `PhysicsVerification` call and no narrator event at all, regardless of Cognitive Layer
  output. `orchestrator.py`'s self-check specifically regression-tests against this.
- `tools/run_full_pipeline_demo.py` - runs a 6-frame synthetic scenario (a car closing at a constant
  1.5 m/s alongside a static pothole) through the full chain and prints per-frame routing, the
  semantic pick, whether an alert fired, which narration lane it used, and the resulting text.
  Distances were chosen by actually running `physics.kinetic_score()` first, not guessed, to produce
  a genuine ignore -> cognitive -> reflex escalation. Confirmed live: the override frame stays in
  English even when `--lang hi` is passed and a matching phrase-table entry exists for it - the
  critical lane's translation bypass (`architecture.md` §11.6) working as designed, not a missing
  translation.
- **Finding from running the demo:** with real (not guessed) kinetic-score values, a `car` object
  moving at a slow constant 1.5 m/s crosses `LOW_K_THRESHOLD` (cognitive routing) at roughly 20 m,
  driven mostly by its class severity weight rather than proximity or closing speed - relevant to the
  threshold-calibration item in `pending_work.md`'s refinement backlog, since it suggests the routing
  may be more sensitive to vehicle classes than the thresholds were likely eyeballed against.

### Changed
- `docs/pending_work.md` §5, `docs/progress.md`, `docs/progress_presentation.md`, `README.md` -
  updated from "not started" to reflect the above; §5 keeps "real SANPO replay" explicitly open since
  this milestone used synthetic data only.

No existing runtime component (`threat_prioritizer`, `reflex_layer`, `physics_verification`,
`narration`) was modified - this entry is new, additive composition code on top of all of them.

## [2026-08-27] - Narration Layer Phase 1: Template Narrator, Translation/TTS Adapters, Literature Review

Built the end-of-pipeline narration/translation/TTS layer — the parts of the roadmap that don't
depend on the SANPO ablation data — plus a verified external-literature reference doc and a
refinement backlog from a critical pass over what's already built.

### Added
- `src/narration/` — new module: `templates.py` (deterministic template narrator, no ML dependency,
  reproduces `architecture.md` §2.7's worked example exactly), `translation.py`
  (`PhraseTableTranslator`, `EnglishFallbackTranslator` real and tested; `IndicTrans2Translator`
  correct-API adapter, code complete but not run against real model weights), `tts.py`
  (`CachedClipTTS` real and tested; `PiperTTS` real, tested, and latency-measured), `pipeline.py`
  (`NarrationPipeline` — structurally enforces that an override event never reaches translation or
  model-backed TTS, per the orchestration-lane rule in `architecture.md` §11.6). All four modules
  have a runnable `--self-check` (invoke as `python -m src.narration.<module> --self-check`).
- `tools/benchmark_narration_latency.py` - standalone latency benchmark, no SANPO data or GPU needed.
  Measured on this dev machine (CPU only): template generation ~0 ms, Piper voice load ~444 ms
  (one-time), Piper synthesis 30-68 ms/utterance for four representative `NarratorEvent` phrasings.
  Found and flagged a real documentation gap: no formal TTS latency budget exists anywhere in
  `docs/` (only 50 ms reflex / 500 ms cognitive are defined).
- `models/tts/piper/en_US-lessac-low.onnx` (+ config) - downloaded Piper voice used for the above.
- `piper-tts`, `IndicTransToolkit` added to `requirements.txt` (installed and verified importable
  this session; the IndicTrans2 model weights themselves were not downloaded).
- `docs/related_work.md` - ~20 externally-verified paper citations (title/authors/year/venue checked
  against a live search, not recalled from memory), organized by which CPE component or design
  decision each grounds: positioning (Dakopoulos & Bourbakis 2010 ETA survey; Liu & Slade 2026
  "Mobilio" smartphone nav app - the closest published comparable system), perception (ByteTrack,
  YOLO26 technical overview, Depth Anything, Grounding DINO), the SANPO dataset paper, kinetic-score
  grounding in real traffic-safety literature (Rosen & Sander 2009/2011 pedestrian impact-speed risk
  curves - independent empirical support for the `v^2` design choice beyond CPE's own architectural
  argument), evaluation methodology (Bradley-Terry 1952, Cohen's kappa 1960, Zheng et al. 2023
  LLM-as-judge), SLM/distillation (Hinton et al. 2015, Qwen3, Phi-4-mini), RL (Schulman et al. 2017
  PPO; Bai et al. 2022 Constitutional AI/RLAIF as the conceptual anchor for Physics-Verification's
  human-label-free reward design), translation (IndicTrans2), and TTS (FastSpeech2, VITS underlying
  Piper, IndicF5/Indic Parler-TTS).
- `docs/pending_work.md` §10 "Refinement backlog" - a critical pass distinct from "not started" work:
  7 items where something already produces results but the result or code has a specific, named gap
  (detector eval has no uncertainty quantification; reflex/cognitive routing thresholds are still
  uncalibrated defaults despite the ablation corpus now existing to calibrate them against; the
  Jetson simulation's 4.0x scaling factor has no cited basis; safety-critical test coverage is thin
  at 9 tests total; `BEHAVIOUR_MULTIPLIER` has no grounding unlike mass now does; bearing uses a fixed
  uncalibrated 70 degree HFoV; the VLM referees' "three distinct families" independence is assumed,
  not checked).

### Changed
- `docs/architecture.md` §11.3-11.5 - added implementation-status notes with the measured numbers
  above; §11.5 in particular now states FastSpeech2/IndicF5/Indic Parler-TTS remain unimplemented per
  the existing "benchmark before entering the real-time path" guidance, now that Piper has been
  benchmarked and it passed.
- `docs/pending_work.md` §4, `docs/progress.md`, `README.md` (SLM Stack table + project tree) -
  updated from "not started" to reflect the phase-1 status above.

No formula, threshold, or existing pipeline component was changed - this entry is new, additive
narration-layer code plus documentation/research, gated the same way everything else in this repo is
gated: implemented-and-tested is stated separately from implemented-but-unverified.

## [2026-08-27] - Ablation Methodology Audit: Stratified Severity Check, λ Inconsistency Flagged

A critical review of the 2026-08-26 kinetic-score ablation run's methodology surfaced two real gaps
before its "severity/λ/size tie with K0" conclusions should be treated as settled.

### Added
- `evaluation/kinetic_ablation_stratified.py` - re-runs `kinetic_ablation.py`'s label-free metrics
  (imported, not reimplemented) restricted to frames containing a severity-differentiated,
  kinematically-close pair of objects - the specific scenario severity's design intent targets. Also
  reports the coverage number (what fraction of the corpus even qualifies) and the K0-vs-arm argmax
  disagreement rate on the stratified subset vs. the full corpus. Rationale: a corpus-wide tie on
  `no-severity` cannot distinguish "severity never matters" from "severity matters only in a minority
  of frames diluted into 19,402" - this isolates that minority directly. Self-check passes; has not
  yet been run against real data (needs `data/processed/ablation_30pct/` on the remote box that ran
  the original ablation).
- `evaluation/kinetic_ablation.py` - `session_metrics()` now accepts an optional pre-computed `groups`
  argument so the stratified script reuses the exact same metric math instead of a second
  implementation that could drift out of sync. Purely additive; default behavior (no `groups` passed)
  is unchanged, verified against the existing `--self-check`.

### Changed
- `docs/ablation_guide.md` - added a `[!WARNING]` before the §6 results table: the `lam=0.25`/
  `lam=1.0` columns are not reproducible from the currently committed `kinetic_ablation.py`, whose
  lambda-sweep plumbing was removed in commit `0e6e0d7` on 2026-08-25 - the day *before* this run's
  Step 4 timestamp. Both arms' verdicts changed from "TIES" to "unverified" pending reconciliation.
  Also documented that no raw run artifacts (`metrics.json`, `report.md`, `disagreements*.json`) were
  ever committed for this run, so the tie verdicts for every arm are not yet independently auditable
  from this repo - only the transcribed summary table exists. Added the stratified-check command to
  the "Next steps" section.
- `docs/methodology.md` §4.9 - added the same three caveats (dilution risk, λ inconsistency, missing
  raw artifacts) so the paper-ready methods reference doesn't overstate what the corpus-wide ties
  currently prove.
- `docs/pending_work.md` §1 - rewritten as a precise ordered list: push the missing run artifacts from
  the remote box -> reconcile the λ inconsistency -> run the stratified check -> run the VLM referee ->
  human calibration -> top-3 VLM check -> depth smoothing filter. Added an explicit instruction not to
  change `physics.py`'s formula or weights based on the current evidence alone.
- `README.md` - added `evaluation/kinetic_ablation_stratified.py` to the project tree.

No formula or weight in `src/perception_stack/physics.py` was changed - this entry is entirely
methodology/tooling/documentation, pending the remote-box steps in `pending_work.md` §1.

## [2026-08-27] - Documentation Reorganization, Methodology/Progress/Backlog Docs, Result Figures

### Added
- `docs/methodology.md` - consolidated, paper-ready methods reference covering dataset construction, detector training/evaluation protocol, kinetic-score formulation and ablation design, edge-latency simulation methodology, and the experimental-design principles applied throughout (never grading a method against itself, relative/forced-choice judgments in place of absolute scores without ground truth, session-level bootstrapping, etc). Synthesizes `architecture.md`, `kinetic_score_opinion.md`, `decisions.md`, `ablation_guide.md`, `yolo_training.md`, and `hardware_targets.md` into one drafting-ready document without duplicating their full reasoning trails.
- `docs/progress_presentation.md` - presentation-shaped summary of current progress: what's built vs. simulated vs. not started, the v2->v3 detector training-protocol finding, the kinetic-score ablation headline result, the measured-vs-simulated edge latency split, a suggested slide order, and a Q&A number sheet.
- `docs/pending_work.md` - explicit, prioritized backlog (kinetic-score referee/human-calibration run, Cognitive Layer, Physics Verification completion, narration/translation/TTS, full SANPO replay, physical edge validation, detector loose ends, export packaging, paper-writing dependencies) with a top-3 recommended next actions section.
- `evaluation/generate_report_figures.py` - regenerates every result figure and a markdown table digest from the current committed benchmark artifacts (`evaluation/benchmarks/yolo26n_version_comparison/`, `.../sanpo_edge_realtime/ten_session_v3_jetson_orin_nano_8gb/`, `.../kinetic_score_eval/`). Output: `evaluation/benchmarks/figures/` (6 PNGs + `results_summary.md`), safe to delete and regenerate.
- `evaluation/benchmarks/kinetic_score_eval/run_2026_08_26/ablation_summary.csv` - machine-readable transcription of the ablation table already published in `docs/ablation_guide.md` §6 (2026-08-26 run, 139 sessions), so it can be plotted; no new data, just a queryable form of the existing published numbers.
- `matplotlib` added to `requirements.txt` for the new figure-generation script.

### Changed
- `docs/progress.md` - refreshed; the previous version predated the kinetic-score ablation work and did not reflect it. Now includes the ablation result, the blinded-referee gap, and the depth-smoothing blocker, and points to `pending_work.md` for the full backlog instead of duplicating it.
- `docs/decisions.md` - status header corrected: the ground-truth-strategy decision was already resolved 2026-08-18 in the body of the entry, but the header still read "Undecided" - fixed the inconsistency and clarified that resolved decisions stay in this file permanently as historical record rather than being deleted once closed.
- `docs/roadmap.md` - added a status note clarifying that actual work after Day 4 diverged from the 14-day plan to prioritize the kinetic-score defense; the day-by-day schedule is historical intent, not a current plan.
- `docs/README.md` - documentation map reorganized into grouped sections (current status / how the system works / research methodology and evidence / datasets and training / history) instead of one flat table, with the new docs and `evaluation/benchmarks/figures/` added to the artifact conventions and maintenance rules.
- `README.md` - project tree updated with the new docs and `evaluation/generate_report_figures.py` / `evaluation/benchmarks/figures/`.

No existing document content was deleted; all changes above are additive or status/consistency corrections.

### Removed / merged (same-day follow-up)
- `docs/verification_guide.md` - merged into `docs/ablation_guide.md` as new §5b (top-3 VLM agreement check via `topk_threat_validation.py`, which `ablation_guide.md` did not previously document) and §7 (troubleshooting). The two guides covered the same commands from different angles; content preserved, file count reduced.
- `docs/decisions.md` - merged into `docs/kinetic_score_opinion.md` as new §10 (formal problem statement, the three-tier ground-truth options table, and the closed decision record). The header inconsistency noted earlier in this entry is now moot since the file no longer exists standalone.
- `docs/sanpo_gap_analysis.md` - merged into `docs/sanpo_dataset.md` as a new "Gap Analysis Workflow" section; the two files covered adjacent SANPO workflows (intake/benchmarking vs. gap analysis) that belong under one dataset doc.
- `docs/roadmap.md` - merged into `docs/architecture.md` as new §11 (Technology Audit: Model and Runtime Choices - the rationale behind §7's SLM/translation/TTS table, plus tracking and orchestration decisions) and §12 (Historical Implementation Plan, superseded by `pending_work.md`).
- `docs/research_paper_prompts.md` - merged into `docs/methodology.md` as a new Appendix of LLM paper-drafting prompts, updated to reflect the current (not yet fully implemented) state of the Cognitive Layer and the pending referee step.
- All cross-references to the five removed files were updated across `docs/`, `README.md`, and `.agents/AGENTS.md`. No content was deleted in these merges, only relocated and, where two files said the same thing, deduplicated. Net effect: 17 files in `docs/` -> 12.

## [2026-08-25] - v3 Detector Default, Frozen Mass Exponent, Enforced 30% Ablation Sample

### Changed
- `src/perception_stack/yolo_tracker.py` - `DEFAULT_MODEL` now points at the finetuned CPE hazard checkpoint v3 (`training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt`) instead of base `models/yolo/base_yolo26n/yolo26n.pt`. The base COCO checkpoint does not contain the CPE hazard classes at all, so any caller that did not pass `model_path` explicitly was silently detecting a different taxonomy than the one the kinetic score's severity table is written against. The path is now absolute (derived from the module location) so the default survives being run from any working directory.
- `src/perception_stack/physics.py` - `SEVERITY_LAMBDA` **frozen at 0.5**; the value is unchanged, but it is now documented as a declared design choice rather than a swept parameter. Dropping Tier-B blinded human labelling from the evaluation plan removed the only tier that could adjudicate between lambda values - the label-free metrics score arrival time, which lambda barely moves. The same applies to `BEHAVIOUR_MULTIPLIER`, which is retained by decision.
- `evaluation/kinetic_ablation.py` - removed the `lam=0.25` and `lam=1.0` arms and the `lam` plumbing in `_score_k()`, which no longer mutates `physics.SEVERITY_LAMBDA`. Six arms remain: K0, linear, no-severity, no-velocity, size, ttc.
- `evaluation/kinetic_ablation.py` - the run is now **restricted to the seeded sample** of `simulation/datasets/sanpo/valid_streams.json` via `--valid-streams`, `--session-fraction` (default 0.30) and `--sample-seed`, rather than scoring whatever CSVs happen to sit in `--csv-dir`. A stale session CSV from an earlier run would otherwise widen the corpus silently and invalidate the session-level bootstrap CIs. Off-sample CSVs are skipped and sampled sessions with no CSV yet are reported as a warning. `--sample-seed` is deliberately separate from `--seed` (bootstrap) so changing one cannot silently change the other. Pass `--session-fraction 1.0` to score every session on disk.
- `tools/download_sanpo_valid_streams.py` - now owns `sample_sessions()` / `sampled_session_ids()` and the `DEFAULT_SESSION_FRACTION` (0.30) and `DEFAULT_SAMPLE_SEED` constants. Moved here from `tools/stream_sanpo_perception.py` because Stage 1 and the ablation must agree on the subset, and this module is stdlib-only so the analysis side can import it without pulling in ultralytics/torch. 139 of the 462 valid streams are selected at the default fraction and seed.
- `tools/stream_sanpo_perception.py` - imports the shared sampler and constants instead of defining its own copy; the `DEFAULT_MODEL` comment claiming YoloTracker defaults to the base checkpoint is corrected.
- `docs/ablation_guide.md`, `docs/architecture.md`, `docs/research_paper_prompts.md` - lambda documented as frozen and reported as a stated limitation; the two lambda arms removed from the arm table.

### Testing
- `python src/perception_stack/physics.py` self-check passes with lambda unchanged at 0.5 (car 4.63x, bus 13.09x a person).
- `evaluation/kinetic_ablation.py` run end-to-end over six synthetic per-session CSVs named after real sampled session IDs plus one off-sample CSV: the off-sample file is skipped, the 133 not-yet-processed sampled sessions are warned about, all six arms score, and `report.md` / `disagreements.json` are written.

## [2026-08-19b] - Local VLM Referees, Mass Exponent Raised, Ablation Run Guide

### Changed
- `src/perception_stack/physics.py` - `SEVERITY_LAMBDA` raised from 0.25 to **0.5** (severity proportional to sqrt(mass)). 0.25 compressed a 171x mass range to 3.6x, which under-weighted the destructive-mass bias the score exists to express; 0.5 gives car 4.6x a person, bus 13x, bus 2.8x a car, without lambda=1's 171x runaway. The self-check now bounds the pure mass-law spread on *both* sides (5x < spread < 60x) rather than only capping it, so a future edit cannot silently flatten mass away either. Note the original compression argument was overstated: with v^2 in the score, kinematics dominate severity in most reorderings, so lambda mostly breaks ties between objects of similar motion.
- `evaluation/vlm_referee.py` - referees are now **local models**, not hosted APIs. The three vendor-specific client functions collapse into one `ask_local()` that speaks the OpenAI-compatible `/chat/completions` shape over stdlib urllib, which vLLM, Ollama, llama.cpp `llama-server`, LM Studio and SGLang all serve. No API keys, no client SDK dependency, no frame leaving the machine. Defaults are three different pretraining lineages (Qwen2.5-VL-7B, InternVL3-8B, Gemma-3-12B) on ports 8001-8003; `--endpoint NAME=URL` and `--model NAME=ID` retarget them, so a single-GPU box can run the three referees sequentially against one port. `temperature=0` so a re-run reproduces the ballot.
- `evaluation/kinetic_ablation.py` - added `lambda = 0.25` and `lambda = 1.0` ablation arms. `_score_k()` swaps `physics.SEVERITY_LAMBDA` and restores it rather than duplicating the severity law, so the arms cannot drift from production.
- `tools/stream_sanpo_perception.py` - frame downloads now run through a thread pool (`--download-workers`, default 8). One HTTPS GET per frame makes the fetch latency-bound, not bandwidth-bound; failures are re-raised so a half-downloaded session fails loudly instead of producing a short CSV.
- `requirements.txt` - dropped the commented-out hosted-VLM SDKs; the referee needs no client library.

### Added
- `docs/ablation_guide.md` - the run guide: what each ablation arm asks, environment setup and self-checks, the Stage-1 streaming run (sampling, disk, resume, the stride/fps trap), the label-free metrics pass, serving three local VLMs (including the one-GPU sequential path), the blinded judging run, the human calibration subset with kappa interpretation thresholds, and how to write up each of the three possible per-arm outcomes.

### Testing
- `evaluation/vlm_referee.py --self-check` now stands up a throwaway `http.server`, calls `ask_local()` against it, and asserts the request path, model, `temperature`, base64 image part and text part - the payload shape is the one thing that silently breaks when a serving stack is swapped.

## [2026-08-19] - Mass-Derived Severity, 30% SANPO Ablation Pipeline, Multi-VLM Referee

### Changed
- `src/perception_stack/physics.py` - class severity is now **derived from real-world mass** instead of hand-tuned: `severity(c) = behaviour(c) x (mass(c) / 70 kg) ** lambda`, lambda = 0.25, exposed as `class_severity()` with `CLASS_SEVERITY` materialised from it. lambda = 0.25 compresses the 171x class mass range to ~5x severity spread; raw mass (lambda=1) gives ~600x and lets a distant bus outrank an imminent pedestrian. Fitting the previous hand-tuned table against log mass recovers lambda ~ 0.18 (R^2 = 0.50), so the mass law restates beliefs already encoded there without the arbitrariness. A sparse `BEHAVIOUR_MULTIPLIER` covers hazards mass cannot express (motorcycle/dog erratic motion, bollard/pole trip height, traffic light/stop sign mounted above head height), and four massless trip hazards (stairs, pothole, puddle, crosswalk) keep explicit weights in `STATIC_HAZARD_SEVERITY` outside the mass law rather than being fudged into it.
- `src/perception_stack/physics.py` - `kinetic_score()` gained keyword-only `gamma`, `size_exponent` and `bbox_area_px` parameters. The defaults (`gamma=2.0`, `size_exponent=0.0`) reproduce shipped K0 exactly; the ablation drives the same production function rather than maintaining a second implementation that can drift. The size term uses apparent size `A_px / d` normalised by `SIZE_REFERENCE_PX_PER_M` and is **off by default**: bounding-box area is ~99% predictable from class + metric depth by projective geometry, so it is redundant for labelled objects and is carried as an ablation arm, not a production term.

- `tools/visualize_stage1.py` - dropped its private six-class severity table and now calls the production `kinetic_score()`. A visualiser that ranks objects differently from the pipeline it visualises is worse than no visualiser.

### Added
- `tools/stream_sanpo_perception.py` - runs Stage 1 over a seeded 30% sample of the 462 SANPO valid streams **streamed directly from the public GCS bucket**, with no local copy of SANPO. One session's frames are pulled to scratch, converted to a per-session CSV, then deleted, so peak disk stays at one session regardless of run length. Resumable (finished sessions are skipped; CSVs are written to `.partial` and renamed only on success) and failure-isolated, for unattended `nohup` runs on a shared SSH box. Writes `<session>.frames.json` mapping each CSV `frame_idx` back to its GCS object so the referee can re-fetch a handful of images later. Frames are strided at download time and `fps` is scaled by the stride - without which every closing velocity would come out `stride`x too fast.
- `evaluation/kinetic_ablation.py` - ablates K0's terms (gamma=1, no severity, no velocity, apparent-size arm, plain time-to-hazard) and scores each variant on the label-free metrics of `kinetic_score_opinion.md` §5 plus the automatic encounter label of §6. All CIs are a percentile bootstrap resampling **whole sessions**, since frames within a session are autocorrelated. Exports the frames where variants pick different top objects as `disagreements.json` (blind) and `disagreements_key.json` (which formula picked what - never shown to a referee).
- `evaluation/vlm_referee.py` - blinded forced-choice referee over those disagreement frames, using three VLMs from three vendors (Claude / Gemini / GPT) because agreement between models sharing training data is a shared prior, not truth. Referees see the annotated RGB frame and a neutral object list in randomised order - no scores, no formula names. Reports per-variant win rates, pairwise Cohen's kappa between referees, and kappa against a human-labelled calibration subset (`--human-template`), which the report marks as MISSING until it exists. Ballots are written per case so an interrupted API run resumes.
- `requirements.txt` - `scipy` (Kendall tau for rank stability) and the optional referee SDKs.

### Testing
- Extended the `physics.py` self-check with the mass law (person = 1.0 by construction, monotone in mass, lambda compresses the mass range below 6x, the behaviour multiplier is load-bearing, unknown classes fall back) and with the new exponents (defaults reproduce K0; the size term stays inert unless `size_exponent` is set).
- Added `--self-check` self-tests to `evaluation/kinetic_ablation.py` (encounter labels exclude a stationary bus and include a closing pedestrian; every velocity-aware variant ranks the closing pedestrian first; the no-velocity arm ranks the parked bus first - the failure it exists to expose; degenerate bootstrap) and `evaluation/vlm_referee.py` (vote parsing rejects hallucinated and unparseable ids rather than guessing, kappa bounds, baseline/variant/neither tallying).

## [2026-08-18] - Kinetic Score Decision: Keep K0, Remove Dummies

### Documentation
- Added `docs/kinetic_score_opinion.md` — recommendation for choosing and defending the kinetic score. Concludes: **keep K0**, whose `v²` term is the only element encoding *consequence* rather than *arrival time* (which the reflex TTC gate at `events.py:217` already covers). Documents how to evaluate with **no ground truth**: six label-free metrics (flicker rate, rank stability under depth perturbation, temporal smoothness, tie rate, complementarity with SLM-1, future self-consistency), automatic "encounter" labels derived from measured future distance, and forced-choice pairwise comparison with a blinded referee restricted to the ~5% of frames where formulas disagree. Rejects multi-VLM polling as primary evidence (polling measures agreement not accuracy; VLM errors are correlated; a still frame carries neither metric depth nor closing velocity) and reframes the VLM as a blinded *referee* calibrated against a human gold set with Cohen's κ reported.

### Changed
- `src/perception_stack/physics.py` — `compute_velocity()` now takes the **least-squares slope** across the whole depth window instead of differencing only the two endpoints, using every sample to cut variance without adding state or tuning knobs. Reverts to endpoint differencing via `use_least_squares=False`. Motivation: velocity is a differentiated depth estimate and `kinetic_score()` squares it, so at 10% depth error K0's score error is +741% versus +190% for a linear-velocity variant. Partially addresses `architecture.md` §10.1, which prescribes a Kalman filter that was never implemented.
- `docs/architecture.md` — **withdrew the unevidenced claim** that "K4 (hybrid momentum + TTC) is the leading candidate." No benchmark was ever run; `evaluation/benchmarks/kinetic_score_eval/` contained only a README.

### Removed
- Deleted `kinetic_score_k1_ttc`, `kinetic_score_k2_linear`, `kinetic_score_k3_quad_distance`, `kinetic_score_k4_hybrid`, `kinetic_score_k5_sigmoid`, `_KINETIC_FORMULAS` and `kinetic_score_all` from `src/perception_stack/physics.py`. K1–K5 were dummies, and measurement showed they were not even independent ones: within a class K1/K2/K5 are *exactly* rank-identical to 1/TTC (Spearman ρ = 1.0000, identical rank vectors over 500 points) and K3 is ρ = 0.9998 — five labels for two behaviours. Cross-class pairwise rank agreement was ≥ 0.945 for every pair except K5. Nothing outside `physics.py` referenced them.
- Deleted `evaluation/kinetic_score_comparison.py` and `evaluation/threat_score_eval.py` — circular by construction (ground truth computed by re-running the formula being scored) with a saturated discriminating metric that passed five of six candidates at its own ρ > 0.7 gate. K0 is now defended by ablation of its own terms; see the rewritten `evaluation/benchmarks/kinetic_score_eval/README.md`.

### Testing
- Added an `assert`-based `_demo()` self-check to `src/perception_stack/physics.py` (`python src/perception_stack/physics.py`) pinning velocity estimation on clean input, the retreating-object clamp, short-history guards, K0 monotonicity in distance/velocity/severity, the `EPSILON` guard, and — the point of the change — that least-squares beats endpoint differencing when one depth sample is corrupted.

## [2026-08-13] - Kinetic Score Ground Truth Decision Record

### Documentation
- Added `docs/decisions.md` — open design-decision log. First entry: "Ground Truth Strategy for Kinetic Score Formula Selection," documenting a circularity problem in `evaluation/threat_score_eval.py` (ground truth is computed by re-running the same formula being scored) and a scale-comparability bug across K0–K5 (K5 is bounded to `[0, severity]` and can never cross `high_k=5.0`, unlike unbounded K0/K3). Records three candidate ground-truth tiers (automatic kinematic "encounter" labels, blinded human judgment, calibrated VLM-as-annotator) and their tradeoffs, pending a team decision. Also flags that `docs/architecture.md`'s claim of "K4 leading candidate" currently has no benchmark evidence behind it (`evaluation/benchmarks/kinetic_score_eval/` contains only a README).

## [2026-08-05] - Manual Kinetic Score Inspector

### Evaluation
- Added `tools/manual_score_inspector.py` — generates a self-contained interactive HTML file for manual frame-by-frame verification of kinetic scores. For each sampled SANPO frame it renders the RGB image with colour-coded bounding boxes (reflex=red, cognitive=orange, ignore=green), a table of all K0–K5 scores per object with routing badges, and the objective ground-truth column (actual depth at T+lookahead). The user can ✅ approve, ❌ flag, or ⏭ skip each frame; judgements persist in `localStorage` and can be exported as a JSON audit file.


### Evaluation
- Added `evaluation/kinetic_score_comparison.py` — benchmark harness for six kinetic score formula candidates (K0–K5). Produces per-formula distribution statistics (mean, P50, P95, P99, IQR), Pearson/Spearman TTC correlation, routing sensitivity (reflex/cognitive/ignore %), monotonicity check, and a ranked Markdown report. Generates Matplotlib plots when available.
- Added `evaluation/threat_score_eval.py` — routing F1/Precision/Recall evaluator. Uses K₊₂ (same track at T+lookahead) as ground truth to score each formula's routing decisions. Primary safety metric is Reflex Recall (missed reflex = safety failure).
- Added `evaluation/benchmarks/kinetic_score_eval/` — dedicated benchmark output folder following the established benchmark folder convention. Contains README with reproduction steps and decision criteria.
- Registered `kinetic_score_eval/` in `evaluation/benchmarks/README.md` index.

### Physics — Additive Formula Candidates (non-breaking)
- Extended `src/perception_stack/physics.py` with five alternative kinetic score functions: `kinetic_score_k1_ttc`, `kinetic_score_k2_linear`, `kinetic_score_k3_quad_distance`, `kinetic_score_k4_hybrid`, `kinetic_score_k5_sigmoid`. All are evaluation-only; the production `kinetic_score()` (K0) is unchanged.
- Added `kinetic_score_all()` dispatcher returning all six formula scores for one object — used by the benchmark harness.

### Shared Data Contract — FactSheet Redesign
- Rewrote `src/shared/fact_sheet.py` with a lean, phone-derivable `DetectedObject` schema:
  - **Removed:** `intent_label` (HEADSUP training-data only; not available at phone inference), `hallucination_filtered` (internal pipeline flag).
  - **Renamed:** `threat_score` → `kinetic_score` (explicit about what it is), `object_id` → `track_id` (aligns with `events.py`), `is_scene_stable` → `scene_stable`.
  - **Added:** `bearing` (human-readable direction string from `bearing_label()`), `route` (Threat Prioritizer lane: reflex/cognitive/ignore).
  - Added `to_slm_user_message()` method — returns the FactSheet as a structured JSON dict for the SLM-1 SFT training record user slot (structured JSON, not flat string).

### Documentation
- Updated `README.md` project structure with new evaluation scripts and `kinetic_score_eval/` benchmark folder.

### Repository Cleanup
- Deleted empty placeholder directories from `src/`, `simulation/`, and `training/` to declutter the workspace and reflect the current state of implementation.
- Updated the `README.md` project structure tree to match the pruned directory state.

## [2026-07-22] - Documentation Ownership and Hardware Targets
### Documentation
- Added `docs/hardware_targets.md` as the source of truth for observed GB10 training-host specs, the planned Jetson Orin Nano 8GB target, runtime budgets, and measured-versus-simulated evidence.
- Moved `Roadmap.md` to `docs/roadmap.md` and normalized the progress checklist path as `docs/progress.md`.
- Renamed the SANPO documents to `docs/sanpo_dataset.md` and `docs/sanpo_gap_analysis.md` so their names match their responsibilities.
- Merged artifact naming and cleanup rules into `docs/README.md`, then removed the overlapping `docs/artifact_structure.md`.
- Updated the main README project tree and documentation index for the consolidated structure.
- Removed the stale Snapdragon-specific hardware target from the architecture and corrected the paper prompt so simulated Jetson results are not described as physical-device measurements.


## [2026-07-22] - Reflex Layer Bridge and Progress Checklist
### Runtime Pipeline
- Added `src/reflex_layer/reflex.py` to convert reflex-route `ThreatEvent`s into `ReflexResult`s and invoke Physics Verification without waiting for cognitive SLM output.
- Updated `src/reflex_layer/__init__.py` exports for the new Reflex Layer bridge.

### Progress Tracking
- Added `docs/progress.md` as a living project checklist covering detector status, dataset intake, ThreatEvent routing, Reflex Layer status, SLM/TTS gaps, and immediate next steps.

### Tests
- Added `tests/test_reflex_layer.py` for TTC override behavior, non-reflex suppression, and high-K physics fallback behavior.

### Documentation
- Updated README structure, architecture notes, roadmap, research prompts, and changelog to reflect the Reflex Layer bridge and current project progress.

## [2026-07-22] - ThreatEvent Routing Contract
### Runtime Pipeline
- Added `src/threat_prioritizer/events.py` with `PerceivedObject`, `ThreatEvent`, `ThreatFrame`, TTC calculation, kinetic-score routing, Physics Verification registry output, and ReflexResult conversion.
- Added `tools/build_threat_events.py` to convert Stage 1 perception CSV files into non-ignore `ThreatEvent` JSONL plus route/class summaries.

### Tests
- Added `tests/test_threat_prioritizer.py` and `tests/__init__.py` with standard-library unit tests for TTC, reflex routing, cognitive near-static hazard routing, ignore routing, and CSV-to-JSONL conversion.

### Documentation
- Updated architecture, roadmap, README structure, and research prompts to document the formal perception-to-physics event contract.

## [2026-07-22] - Roadmap Technology Audit and Edge Export Step
### Planning
- Updated `docs/roadmap.md` with the current detector status, SANPO 10-session latency result, and revised model recommendations: keep YOLO26n v3, prefer Qwen3-1.7B non-thinking mode for SLM-1, use template-first critical narration with optional Phi-4-mini, and keep IndicTrans2 distilled/phrase-table translation for the prototype.

### Tooling
- Added `training/scripts/export_yolo26n_edge.py` and `models/yolo/cpe_yolo26n_hazards_v3_from_base/` to export the preferred v3 checkpoint to ONNX, TensorRT engine, OpenVINO, or TorchScript with modern Ultralytics `quantize` arguments and a model-registry export manifest.
- Updated the training script export path to use `--export-quantize` instead of deprecated INT8 export aliases.

### Documentation
- Updated README, YOLO training docs, training workflow docs, architecture notes, and research-paper prompts to point at the detector export/runtime comparison as the next step.

## [2026-07-22] - README Structure Maintenance Rule
### Documentation
- Added a `.agents/AGENTS.md` rule requiring the main `README.md` Project Structure section to be reviewed and updated whenever new files, folders, scripts, datasets, model registry entries, benchmark artifacts, or major docs are added.
- Updated the main `README.md` project map to include current docs, SANPO valid-stream metadata, YOLO training scripts, edge benchmark tools, model registry folders, and benchmark artifact locations.

## [2026-07-22] - SANPO 10-Session Edge Latency Benchmark
### Evaluation
- Ran v3 over 10 bounded SANPO-Real valid streams with the `jetson_orin_nano_8gb` simulation profile, processing 300 RGB+depth frames total.
- Saved individual metrics and aggregate reports under `evaluation/benchmarks/sanpo_edge_realtime/ten_session_v3_jetson_orin_nano_8gb/`.
- Average simulated p95 latency was `34.31 ms`; worst-session simulated p95 latency was `41.96 ms`, with all sessions passing the `<50 ms` detector/reflex budget.

## [2026-07-22] - Edge Device Simulation Profile
### Evaluation
- Added edge-profile support to `tools/benchmark_edge_realtime.py`, including a conservative `jetson_orin_nano_8gb` simulation profile with `4.0x` latency scaling and `3.0 ms` per-frame sensor/memory overhead.
- Ran the SANPO valid-stream smoke benchmark under the Jetson Orin Nano 8GB profile; simulated p95 total latency was `36.96 ms`, passing the `<50 ms` detector/reflex budget.

## [2026-07-22] - SANPO Valid-Stream Edge Smoke Test
### Evaluation
- Documented the public SANPO GCS bucket layout, valid-stream local convention, label metadata, and bounded download strategy in `docs/sanpo_dataset.md`.
- Added `tools/download_sanpo_valid_streams.py` to download selected paired SANPO-Real RGB/depth frames from `valid_streams.json` without copying whole multi-GB sessions.
- Downloaded valid stream `0xCqEk5hjEvrygxu26MZkieSv45D_gaJ` and saved v3 full RGB+depth edge benchmarks under `evaluation/benchmarks/sanpo_edge_realtime/`.
- Fixed SANPO depth handling for native `960 x 960` `.float16.gz` maps by inferring shape and scaling RGB-space boxes into depth-map coordinates.

## [2026-07-22] - Edge Real-Time Benchmark Utility
### Evaluation
- Added `tools/benchmark_edge_realtime.py` to measure YOLO26n + ByteTrack streaming latency, optional depth loading, depth post-processing, effective FPS, and real-time/reflex budget pass-fail status.
- Saved the v3 detector-only GB10 benchmark under `evaluation/benchmarks/yolo26n_version_comparison/` with p95 total latency of `9.65 ms` per processed frame at `frame_step=3`.

## [2026-07-22] - Local Artifact Relocation
### Project Structure
- Moved the pretrained YOLO26n base checkpoint into `models/yolo/base_yolo26n/yolo26n.pt` and updated training, comparison, perception, notebook, and documentation references.
- Moved SANPO valid stream metadata into `simulation/datasets/sanpo/valid_streams.json` and updated gap-analysis consumers.

## [2026-07-22] - Documentation Cleanup and YOLO Guide Consolidation
### Documentation
- Replaced stale YOLO v1/v2 planning docs with `docs/yolo_training.md` as the current source of truth for dataset roots, checkpoints, training commands, evaluation commands, and version-comparison artifacts.
- Updated the docs index, architecture, artifact-structure guide, research-paper prompts, training README, and benchmark README to reflect the v3-from-base checkpoint and cumulative evaluation results.
- Removed obsolete links to documentation files that no longer exist.

## [2026-07-22] - YOLO26n Version Comparison Artifacts
### Evaluation
- Ran all-class held-out test evaluation for the v3-from-base checkpoint.
- Added cumulative v1/v2/v3 comparison artifacts under `evaluation/benchmarks/yolo26n_version_comparison/`, including overall metrics, per-class mAP tables, and retained-class base-YOLO deltas.

## [2026-07-22] - v3 From-Base Training Setup
### Training
- Removed interrupted YOLO training run folders while preserving successful v1/v2 checkpoints and evaluation artifacts.
- Documented the v3 retention-repair path that starts from base `models/yolo/base_yolo26n/yolo26n.pt`, reuses the cleaned v2_full dataset, and recommends extra Roboflow data for `truck`, `bicycle`, `dog`, `stop sign`, and `bus`.

## [2026-07-20] - AutoBatch Startup Fix
### Training
- Updated the GB10 training script to keep cuDNN benchmark disabled during `batch=-1` AutoBatch search so Ultralytics can probe large batch sizes instead of falling back to batch 16.

## [2026-07-20] - v2 Motorcycle Source Fill
### Training Data
- Added a dedicated `motorcycle` Roboflow detection source and moved the active clean rebuild target to `data/yolo_finetune_v2_full` after auditing that the retained COCO-style source had zero motorcycle positives.

## [2026-07-20] - v2 Missing-Class Source Fill
### Training Data
- Added verified Roboflow detection sources for `crosswalk` and `puddle` so the active 17-class detector is not trained with zero positive examples for those classes.
- Added `training/scripts/rebalance_yolo_splits.py` to move deterministic train examples into validation/test splits for train-only Roboflow sources before final all-class evaluation.

## [2026-07-20] - v2 Balanced Dataset Rebuild
### Training Data
- Fixed Roboflow merge accounting so class image caps are not double-counted while merging multi-source v2 data.
- Added explicit `pole` and `bollard` caps/minimums and pointed the active YOLO data config at `data/yolo_finetune_v2_balanced` for a clean v2 rebuild.

## [2026-07-20] - Roboflow Source Version Verification
### Training Data
- Verified the provided Roboflow Universe source URLs with the Roboflow SDK, updated the v2 manifest to the latest usable versions, and removed the invalid stairs source whose requested version was not found.
- Marked `tiger-dataset/dog-ukbxr` as unavailable for download because the project currently has no generated dataset versions; dog examples remain sourced through the COCO-style retained dataset until another dog dataset is provided.

## [2026-07-20] - v2 Roboflow Source Manifest Update
### Training Data
- Updated the active Roboflow Universe manifest with additional stairs, pothole, dog, bench, and retained COCO-class sources while preserving the existing pole and bollard datasets for retention.
- Tightened Roboflow label intake to accept only 5-column YOLO detection rows so segmentation polygon labels are not accidentally treated as bounding boxes.

## [2026-07-20] - Roboflow Class-Balanced Intake Caps
### Training Data
- Added per-class image caps to `training/scripts/download_roboflow_universe.py` so multiple Roboflow Universe URLs can be supplied per class without letting oversized source datasets dominate v2 fine-tuning.
- Documented the v2 class cap policy and SANPO domain-alignment guidance in `docs/plan_yolo_v2_finetune.md`, and added class limit/minimum examples to the Roboflow manifest template.

## [2026-07-20] - Retained COCO Class Expansion
### Training & Perception
- Added `dog` and `bench` back into the active CPE YOLO taxonomy as retained COCO navigation-obstacle classes, expanding the active detector head to 17 classes.
- Updated training, dataset-download, evaluation, retention-comparison, perception-filtering, physics severity, and planning docs to use retained class IDs 0-10 and custom hazard IDs 11-16.

## [2026-07-20] - CPE Taxonomy Simplification
### Training & Perception
- Reduced the active CPE YOLO hazard taxonomy from 16 to 15 classes by removing the broad head-height obstacle catch-all from training configs, scripts, perception filters, and docs.
- Kept `puddle` separate from `pothole` because it represents a different navigation risk, while noting that v2 data quality should decide whether it remains active long term.

## [2026-07-20] - GB10 YOLO26n Training Throughput Optimization
### Training
- Retuned `training/scripts/train_yolo26n_hazards.py` for the Grace-Blackwell GB10 lab server with RAM dataset caching, AutoBatch defaults, CPU-saturating worker selection, disabled training-loop validation/plots, skipped default export, and startup throughput reporting.
- Updated `docs/plan_yolo_v2_finetune.md` with the GB10 high-throughput profile, the 6.7GB dataset-cache estimate, and the fast v2 command/checklist.

## [2026-07-20] - Project Branding Cleanup
### Documentation & UI Text
- Removed visible legacy project-branding references from repository documentation and the Stage 1 visualization overlay, standardizing project-facing text on Composite Perception Engine (CPE).

## [2026-07-20] - Repository Artifact Cleanup & Model Registry
### Repository Hygiene
- Removed generated Ultralytics plot/preview artifacts from tracked training runs while preserving compact metrics and checkpoint metadata.
- Added artifact conventions, `training/README.md`, `evaluation/benchmarks/README.md`, and `models/yolo/README.md`; the conventions now live in `docs/README.md`.
- Added a local model registry entry under `models/yolo/cpe_yolo26n_hazards_v1/` and compact v1 benchmark summaries under `evaluation/benchmarks/yolo26n_hazards_v1/`.
- Updated `.gitignore` to keep datasets, secrets, binary weights, training plots, and local Roboflow manifests out of version control.

## [2026-07-20] - YOLO26n v2 Fine-Tuning Plan & Retention Comparison
### Training & Evaluation
- Added `docs/plan_yolo_v2_finetune.md`, a checklist-driven plan for continuing from the v1 `best.pt` checkpoint, repairing weak `stairs`/`pothole` performance, and adding retained-class validation data.
- Added `training/scripts/compare_yolo26n_retention.py` to compare a fine-tuned CPE checkpoint against base `models/yolo/base_yolo26n/yolo26n.pt` for retained COCO classes using class-name remapping, while validating the candidate on all active CPE classes.
- Updated `docs/architecture.md`, `docs/plan_yolo_finetune.md`, and `docs/research_paper_prompts.md` to document the v2 staged fine-tuning and retained-class validation workflow.

## [2026-07-20] - YOLO26n Hazard Checkpoint Evaluation
### Training & Evaluation
- Added `training/scripts/evaluate_yolo26n_hazards.py` to validate a fine-tuned checkpoint on `val` or `test`, export per-class metrics, generate a Markdown report, and save prediction previews for the new navigation-hazard classes.
- Documented the held-out test evaluation command in `docs/plan_yolo_finetune.md`.

## [2026-07-15] - Roboflow Universe Dataset Intake for YOLO26n
### Training Data
- Optimized `training/scripts/train_yolo26n_hazards.py` for GPU fine-tuning with CUDA runtime reporting, TF32/cuDNN benchmark flags, multi-GPU device parsing, integer batch handling, AMP controls, cache mode, and optional torch compile.
- Fixed the YOLO26n training preflight by using integer batch sizes and normalizing Roboflow labels to 5-column detection format before training.
- Added `.env` loading to `training/scripts/download_roboflow_universe.py` so SSH users can keep `ROBOFLOW_API_KEY` outside source code while running Roboflow Universe downloads.
- Added `training/scripts/download_roboflow_universe.py` to download selected Roboflow Universe datasets, remap source labels into the CPE hazard taxonomy, and merge YOLO images/labels into `data/yolo_finetune`.
- Added `training/configs/roboflow_universe_sources.example.json` as a manifest template for Universe workspace/project/version sources and class mappings.

### Perception Stack
- Aligned `src/perception_stack/yolo_tracker.py` and `src/perception_stack/physics.py` with the current hazard fine-tuning taxonomy so newly trained hazard classes are not filtered out at inference.

### Documentation
- Updated `docs/plan_yolo_finetune.md` with the SSH Roboflow workflow and minimum/preferred image-count targets for each new hazard class.

## [2026-07-14] - YOLO26n Hazard Fine-Tuning & Multi-Range Gap Analysis
### Tooling & Evaluation
- **Multi-Range Gap Analysis Script**: Added `tools/gap_analysis_experiments.py` to evaluate and compare 4 distinct depth ranges across 20 valid streams using all YOLO classes (whitelist disabled).
- **JSON Hard Example Logging**: Configured the script to save session ID and frame number mappings of hard examples in a range-specific JSON format (`hard_examples.json`) rather than writing large raw frame images to disk.
- **Experimental Reports**: Configured the script to output range-specific CSV summaries and a master comparison report (`distance_metric_comparison.md`) under the `gap_analysis_output/` folder.
- **Workspace Cleanup**: Cleaned up the workspace by deleting legacy visual output PNGs.

### Training
- Added `training/scripts/train_yolo26n_hazards.py`, a lab-GPU training script that fine-tunes `models/yolo/base_yolo26n/yolo26n.pt` on CPE navigation hazards while preserving the nano architecture, freezing early layers, applying rare-class weighting, and optionally exporting INT8 edge artifacts with an export-only mode for finished checkpoints.
- Added `training/configs/cpe_hazard_classes.yaml` with a CPE hazard taxonomy covering retained COCO safety classes plus non-COCO hazards such as `pole`, `bollard`, `stairs`, `crosswalk`, `pothole`, and `puddle`.

### Documentation
- Updated `docs/architecture.md` and `docs/research_paper_prompts.md` to document the multi-range gap analysis capabilities, whitelist removal, and JSON logging format.
- Updated the architecture and research-paper prompts to describe the edge-preserving YOLO26n fine-tuning workflow.

## [2026-07-12] — YOLO Gap Analysis Expansion & Automation
### Tooling & Inference
- **Expanded YOLO Gap Analysis Notebook**: Overhauled `notebooks/sanpo_yolo_gap_analysis.ipynb` with centralized configuration, robust morphological object extraction, 5-panel visualizations, and automated fine-tuning advice reports.
- **GCS-Aware Stream Fallback**: Integrated streaming logic in `SANPOLoader` to dynamically fetch images and depth maps from Google Cloud Storage anonymously when local files are missing.
- **Automated Verification**: Created verification test scripts to execute and validate the notebook code cells under headless conditions.

### Architecture & Documentation
- **Architecture Doc Update**: Documented the new error analysis and dataset gap analysis pipeline in `docs/architecture.md`.
- **Research Prompt Curation**: Added prompt guidelines for writing methodology sections on connected component extraction and hard example mining in `docs/research_paper_prompts.md`.

## [2026-07-10] — Implementation Plans for YOLO Fine-Tuning & Gap Analysis
### Documentation
- **`docs/plan_yolo_finetune.md`**: Detailed implementation plan for fine-tuning YOLO26n. Includes class curation rationale (dropping non-navigation COCO classes while adding `pole`, `stairs`, `crosswalk`, and surface hazards), SANPO panoptic-to-YOLO label conversion pipeline, dataset split strategy, training script outline, and benchmark targets.
- **`docs/sanpo_gap_analysis.md`**: Step-by-step guide for running the gap analysis notebook across SANPO. Documents all config parameters, output folder management (frame sampling to prevent bloat), visualization legend, CSV column definitions, and post-run analysis queries.

## [2026-07-10] — SANPO Analysis & YOLO Class Curation
### Research & Tooling
- **SANPO dataset clarified**: SANPO is a Google Research egocentric navigation dataset (701 real + 1961 synthetic sessions), NOT an Indian dataset. Confirmed 30-class taxonomy covering sidewalks, poles, crosswalks, pedestrians, and cyclists.
- **YOLO class curation**: Removed region-specific classes (autorickshaw); class list finalized to 14 navigation-critical COCO classes plus depth-detected unlabeled blobs.
- **Gap Analysis Notebook**: Created `notebooks/sanpo_yolo_gap_analysis.ipynb` to empirically identify which scene objects are detected by depth but missed by the current YOLO26n COCO whitelist. Outputs per-frame visualizations (3-panel: RGB+YOLO, depth heatmap, gap-only), a frequency table, and a CSV of gap statistics.

## [2026-07-10]
### Architecture & Documentation
- **Knowledge Distillation Strategy**: Added formal documentation to `architecture.md` outlining the Teacher-Student SLM training pipeline for edge constraints.
- **YOLO Domain Fine-tuning**: Added explicit strategy for fine-tuning YOLO26n on street-level hazards rather than generic COCO classes.
- **Documentation Refactoring**: Consolidated `perception_stack.md`, `physics_verification.md`, `phase_1_breakdown.md`, and `scaffolded_components.md` into the master `architecture.md` for a single source of truth.
- **Research Prompts**: Created `research_paper_prompts.md` to aid in generating the academic paper via LLMs.

## [2026-03-30]
### Perception Stack
- **Centre-patch depth sampling**: Added 20% patch to fix bbox edge bleed (`src/perception_stack/depth_loader.py`)
- **Occlusion exclusion masking**: Added `exclude_boxes` param to `median_depth_in_box` (`src/perception_stack/depth_loader.py`)
- **Sparse depth 3x fallback**: Auto-expands patch if no valid pixels found (`src/perception_stack/depth_loader.py`)
- **Class whitelist filter**: `ALLOWED_CLASSES` set to drop non-navigation COCO classes (`src/perception_stack/yolo_tracker.py`)
- **Geometric validation**: Added `is_valid_detection()` for aspect ratio, area, and edge FP checks (`src/perception_stack/yolo_tracker.py`)
- **3-pass depth rescoring**: Added `depth_rescore()` function for size sanity, conf, and depth NMS (`src/perception_stack/yolo_tracker.py`)
- **Grid-based unlabeled obstacle sweep**: Replaces single-ROI scan with 5 columns (`src/perception_stack/pipeline.py`)
- **Front-to-back occlusion ordering**: Added sorting and cumulative occluder list (`src/perception_stack/pipeline.py`)
- **Nth-frame skip**: Added `frame_step` param (e.g., `frame_step=3` for ~67% compute reduction) (`src/perception_stack/pipeline.py`, `tools/run_perception.py`)
- **Streaming generator architecture**: `run_perception_stream` generator yields per-frame, breaking RAM bottleneck (`src/perception_stack/pipeline.py`)
- **tqdm progress bar**: Replaces print every 50 frames (`src/perception_stack/pipeline.py`)
- **StreamingCSVWriter**: Context manager for incremental disk writes paired with generator (`src/perception_stack/csv_writer.py`)
- **Batch tensor physics**: Vectorized functions `batch_compute_bearing` & `batch_kinetic_score` (`src/perception_stack/physics.py`)
- **CLASS_SEVERITY deduplication**: `fact_sheet_builder` now imports from `physics.py` (`src/perception_stack/fact_sheet_builder.py`)

### Tools
- **CLI updates**: Added `--frame_step` arg and streaming API to `run_perception.py`, using `StreamingCSVWriter` (`tools/run_perception.py`)

## [2026-03-27]
### Perception Stack
- **Unlabeled obstacle detection**: Single central-ROI scan for poles/bollards via depth scan (`src/perception_stack/pipeline.py`)

## [2026-03-25]
### Architecture
- **Dual-SLM system design**: Initial CPE architecture finalized with Reflex, Cognitive, and Physics Verification layers (`architecture.md`)

### Perception Stack
- **YOLO26n + ByteTrack integration**: Basic tracking pipeline (`src/perception_stack/yolo_tracker.py`)
- **SANPO + UASOL depth map loading**: Auto-detects `.float16.gz` vs `.png` (`src/perception_stack/depth_loader.py`)
- **Stage 1 pipeline**: RGB+Depth → CSV batch mode with `run_perception()` (`src/perception_stack/pipeline.py`)
- **CSV canonical schema**: `CSV_FIELDS` contract defined and writer implemented (`src/perception_stack/csv_writer.py`)
- **Physics calculations**: Scalar functions for bearing, velocity, and kinetic score (`src/perception_stack/physics.py`)
- **Stage 2 fact sheet builder**: CSV → JSONL with K0 and K+2 lookahead (`src/perception_stack/fact_sheet_builder.py`)

### Physics Verification
- **Adjudication logic**: Added 4-rule judge and RL rewards via `PhysicsVerification` class (`src/physics_verification/physics_verification.py`)

### Tools
- **run_perception.py**: Initial batch API CLI for Stage 1 (`tools/run_perception.py`)
- **run_fact_sheets.py**: Initial implementation for Stage 2 (`tools/run_fact_sheets.py`)
