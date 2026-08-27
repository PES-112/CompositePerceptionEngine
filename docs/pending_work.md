# Pending Work

Everything not yet done, grouped by priority. This is the backlog counterpart to `progress.md` (what
exists) and `progress_presentation.md` (how to present what exists). Update this file whenever an item
is finished — move it to `progress.md` rather than deleting it from history; the CHANGELOG keeps the
dated record.

---

## 1. Finish the kinetic-score evidence (closest to done)

1. ~~Push the missing run artifacts.~~ **Done 2026-08-27** — ishaan committed `report.md`,
   `metrics.json`, `disagreements.json`, and `disagreements_key.json` from the DGX Spark (commit
   `d9f4330`). All four are now in `evaluation/benchmarks/kinetic_score_eval/run_2026_08_26/` and
   independently auditable — cross-checked directly against `ablation_guide.md`'s summary table and
   they match exactly, including CI bounds (e.g. K0 and `linear` flicker rate both report the
   identical `[0.964, 0.983]`, confirming the "TIES" verdicts weren't just eyeballed).
2. ~~Reconcile the `lam=0.25`/`lam=1.0` inconsistency.~~ **Done 2026-08-27** — same commit root-caused
   it precisely: an earlier version of `kinetic_ablation.py` (commit `e045310`) monkey-patched
   `SEVERITY_LAMBDA` per arm before that plumbing was intentionally removed on 2026-08-25. The raw
   numbers are real (preserved in the committed run artifacts) but not reproducible from current code,
   so they're struck from `ablation_guide.md`'s tables rather than cited. Resolved during the merge of
   local doc work with ishaan's push (`docs/ablation_guide.md` §6 now has the full account).

Everything below still needs `data/processed/ablation_30pct/` (the raw per-session Stage 1 CSVs —
large, still only on the remote box the ablation ran on: `ssh student-machine`, `10.1.24.55`, user
`student-4`, not reachable from this environment) and, for items 3–4, GPU-served local VLMs.

3. **Run the stratified severity check** (`evaluation/kinetic_ablation_stratified.py`, added
   2026-08-27 — self-checks pass, needs the raw CSVs to run for real):
   ```bash
   python evaluation/kinetic_ablation_stratified.py \
       --csv-dir data/processed/ablation_30pct \
       --out-dir evaluation/benchmarks/kinetic_score_eval/run_2026_08_26
   ```
   The corpus-wide `no-severity`/`linear`/`size` ties in the main ablation can't distinguish "severity
   never matters" from "severity only matters in a minority of frames diluted into 19,402." This
   isolates just the frames with a severity-differentiated, kinematically-close pair of objects — the
   scenario severity was designed for — and reports both the coverage (how many such frames even
   exist) and the same metrics restricted to them. Read this before deciding whether to change
   `physics.py`.
4. **Run the blinded VLM referee on the 219 already-exported disagreement frames** (`disagreements.json`
   is committed and ready — this step no longer needs a push, only GPU + the raw CSVs for
   `--frames-dir`). Per-arm breakdown of those 219, now knowable from the committed
   `disagreements_key.json`: `no-velocity` 60 (capped), `ttc` 60 (capped), `size` 51, `linear` 26,
   `no-severity` 11, `lam=0.25` 6, `lam=1.0` 5 — `evaluation/vlm_referee.py` against the `linear`,
   `no-severity`, and `size` arms that tied on label-free metrics. Commands: `ablation_guide.md` §4.
5. **Human calibration set (100–150 frames), not optional.** Until `human_labels.json` exists, VLM
   win rates are not citable evidence — three models can share one prior. `--human-template` /
   `--score-only` flags in `vlm_referee.py`. Report Cohen's κ per model against the human set and
   pairwise κ between VLM referees.
6. **Run `topk_threat_validation.py`.** Documented in `ablation_guide.md` §5b but no output
   directory (`evaluation/benchmarks/topk_threat_eval/`) exists yet — this check (does an independent
   VLM agree with K0's top-3 picks at all) has not been executed.
7. **Implement the depth-smoothing filter before trusting any of the above under real (non-dataset)
   depth noise.** `kinetic_score_opinion.md` §9 flags this as a blocker: K0 squares velocity, which is
   a differentiated depth estimate, so at 10% depth error K0's score error is +741% vs. +190% for a
   linear variant — and no Kalman/smoothing filter exists yet in `src/perception_stack/depth_loader.py`
   beyond a basic range filter. `architecture.md` §10.1 already calls for this.

**Do not change the equation or weights in `physics.py` based on the current evidence alone** — items
1–2 are resolved and the data is now audited, but the corpus-wide "severity ties" result still has the
dilution problem item 3 addresses (untested), and the human-judgment question (item 4) is still
unrun. Revisit after 3–4.

## 2. Cognitive Layer (SLM-1) — not started

- Build the SLM-1 adapter: consumes `ThreatEvent(route="cognitive")`, emits the `SemanticEval` JSON
  contract already specified in `architecture.md` §2.5.
- Primary target Qwen3-1.7B, non-thinking mode; fallback Qwen2.5-1.5B-Instruct (`architecture.md` §11.2).
- Feed only the Symbolic Fact Sheet (no raw images) — schema already frozen in `architecture.md`.
- PPO training loop against the Physics Verification reward (`architecture.md` §3, §8) — entry points
  exist at `training/rl_agent/warmup.py` / `train_ppo.py` but have not been run against real data.
- Latency budget to hit: soft real-time ≤500 ms with stale-response rejection.

## 3. Physics Verification — partial

- Judge logic exists and is unit-tested against the Reflex Layer (`tests/test_reflex_layer.py`), but
  has no cognitive-route SLM output to arbitrate yet — blocked on item 2.
- Stale-response / latency-budget handling between Cognitive Layer and Physics Verification not built.

## 4. Narration, translation, audio — phase 1 done, rest not started

Implemented 2026-08-27 (`src/narration/`, `tools/benchmark_narration_latency.py`) — self-checks pass,
`--self-check` requires `python -m src.narration.<module>` (relative imports, not a direct script
run), the `NarrationPipeline` structurally enforces that an override event never reaches
translation/model-TTS regardless of caller intent (see its self-check), and real latency was measured
on this dev machine with no SANPO data or GPU needed:

| Component | Status | Note |
|---|---|---|
| Deterministic template narrator (`templates.py`) | **Done** | Matches `architecture.md` §2.7's worked example exactly. |
| Phrase-table translator (critical path) | **Done** | `translation.py` `PhraseTableTranslator` — falls back to English on any miss, never raises. |
| English fallback translator | **Done** | Trivial, but real — the documented latency-exceeded fallback. |
| IndicTrans2 adapter | **Code done, untested** | Correct API usage, `IndicTransToolkit`/`transformers` installed; the actual `ai4bharat/indictrans2-en-indic-dist-200M` weights (several hundred MB) were not downloaded this session. Run `_smoke_test_indictrans2()` in `translation.py` once they are. |
| Cached-clip TTS (critical path) | **Done** | Cache miss returns a fallback beep in ~0 ms — never blocks on a model. |
| PiperTTS | **Done and measured** | `en_US-lessac-low` voice downloaded to `models/tts/piper/`; measured load ~444 ms (one-time), synthesis 30–68 ms/utterance on CPU, no GPU. |
| FastSpeech2 / IndicF5 / Indic Parler-TTS | **Not started** | Per `architecture.md` §11.5, gated on benchmarking Piper first — now that Piper is measured, these remain lower priority unless Indic voice quality specifically is needed. |
| System Heartbeat (ambient, independent of Physics Verification) | **Not started** | |
| Formal TTS latency budget | **Missing from docs** | `tools/benchmark_narration_latency.py` found this gap while running: `hardware_targets.md` defines 50 ms reflex / 500 ms cognitive budgets but nothing for narration TTS. Add one once a target-device measurement exists. |

Next for this component: wire `NarrationPipeline` to actually consume live `NarratorEvent`s from
`PhysicsVerification.adjudicate()` in a real run (currently only exercised with synthetic events in
self-checks) — this is naturally part of item 5's full end-to-end replay, not a separate task.

## 5. End-to-end integration

- **Synthetic full-pipeline replay — done (2026-08-27).** `src/pipeline/orchestrator.py` +
  `src/cognitive_layer/stub.py` + `tools/run_full_pipeline_demo.py` compose Threat Prioritizer ->
  (stub) Cognitive Layer -> Physics Verification -> Narration/Translation/TTS in one process and run
  on synthetic frames end-to-end — the composition gap this item used to describe. **Real SANPO
  replay is still not done**: the demo uses a 6-frame synthetic car-approach scenario, not actual
  SANPO perception CSVs, so this doesn't yet prove behavior on real sensor noise, real class mixes, or
  real multi-object frames. Re-run the same orchestrator over `data/processed/ablation_30pct/*.csv`
  (via `src/threat_prioritizer.build_threat_frames_from_csv`) once that's available.
- **Finding from building the demo:** the orchestrator didn't previously exist anywhere in the repo —
  `src/reflex_layer/reflex.py` only ever calls `PhysicsVerification` for reflex-route events by
  design ("Cognitive events are intentionally left for the SLM path"), and nothing built the other
  half of that sentence. Before this session, a frame containing *only* cognitive-route objects (e.g.
  a near-static pothole with no fast-closing object present) produced no `PhysicsVerification` call
  and no narrator event at all, regardless of what a Cognitive Layer might have said. This was a real
  gap, not a hypothetical one — `src/pipeline/orchestrator.py`'s self-check specifically regression-
  tests against it.
- **Finding from running the demo:** with real `kinetic_score()` values (not guessed), a `car` at a
  slow constant 1.5 m/s crosses `LOW_K_THRESHOLD` (cognitive routing) at ~20m purely from its class
  severity weight, before proximity or closing speed alone would suggest concern — worth keeping in
  mind for the threshold-calibration item below, since it means the routing may be more sensitive to
  vehicle classes specifically than to the generic "distance/speed" intuition the thresholds were
  eyeballed against.
- Threshold tuning (`LOW_K_THRESHOLD`, `HIGH_K_THRESHOLD`, near-static hazard distance) against real
  SANPO perception CSVs rather than defaults (`progress.md`, refinement backlog §10.2).

## 6. Physical hardware validation — not started

Everything edge-related in the repo today is either a native GB10 measurement or an analytical Jetson
proxy (4.0× compute scaling + 3 ms overhead) — **no physical edge device has been benchmarked.**
Required before any paper or demo claims real edge performance (`hardware_targets.md`):

1. Export YOLO26n v3 to ONNX FP16 (`training/scripts/export_yolo26n_edge.py` exists, has not been
   run for a committed artifact) and build a TensorRT engine on the actual target device.
2. Replay the same 10 SANPO sessions from preloaded/sensor-memory frames on that device.
3. Report warm-up separately from steady-state; measure mean/median/p95/p99, throughput, peak memory,
   temperature, power mode.
4. Benchmark the detector/reflex path under concurrent SLM + audio load (needs item 2 first).
5. Re-run after a sustained workload to expose thermal throttling.

## 7. Detector loose ends

- `dog` retention is still below base YOLO after v3 — needs more Roboflow data for that class, then
  `stop sign`, `truck`, `bicycle`, `bus` in priority order (`yolo_training.md`).
- Gap-analysis findings have not been looped back into a v4 training round.

## 8. Packaging / export

- Export detector, SLM, translation, and TTS artifacts for on-device deployment — explicitly deferred
  until the Python pipeline behavior is stable end-to-end (item 5).

## 9. Paper-writing tasks that depend on the above

- Results/Discussion sections cannot claim physical edge FPS, high-kinetic recall under real noise, or
  stale-SLM filtering until items 1.4, 2, and 6 exist (`methodology.md` Appendix §A.7 already states
  this constraint explicitly — keep enforcing it as later sections get drafted).
- The blinded-referee kinetic-score result (item 1.4) is likely the single most defensible novel
  result once it exists — prioritize it over new feature work if a paper deadline is the driving
  constraint. The λ root-cause and audited-data findings (items 1.1–1.2, now resolved) are already
  citable as-is.

---

## 10. Refinement backlog (done, but not yet rigorous)

Distinct from everything above: these parts are built and produce results today, but the results or
the code carry a gap that should be closed before being trusted as-is — not "not started," but "not
finished being right."

1. **Detector evaluation has no uncertainty quantification.** `evaluation/benchmarks/yolo26n_version_comparison/`
   reports single point-estimate mAP per version (v3 mAP50-95 0.459 vs. v2's 0.424) with no CI, no
   multi-seed re-run, and no variance estimate — unlike the kinetic-score ablation, which is careful
   about session-level bootstrapping, the detector comparison can't currently say whether a 0.035
   mAP50-95 delta is a real improvement or within training-noise. Several per-class deltas are also
   computed from small image counts (e.g. `crosswalk`/`puddle` have 0 v1 images per
   `per_class_version_metrics.csv`, so their "v3 - v1" columns are blank, not zero — already correctly
   left blank, but worth an explicit caveat in `yolo_training.md` rather than a silent blank cell).
   Refine by: re-running v3 training/eval with 2-3 different seeds and reporting a range, or at minimum
   annotating which per-class deltas rest on <50 test images.
2. **The reflex/cognitive routing thresholds are still uncalibrated defaults.**
   `src/threat_prioritizer/events.py` ships `DEFAULT_LOW_K_THRESHOLD = 0.5`,
   `DEFAULT_HIGH_K_THRESHOLD = 5.0`, `DEFAULT_NEAR_STATIC_DISTANCE_M = 1.5` as initial guesses
   (`architecture.md` §5 already flags these as "calibrate from dataset"). The 139-session ablation
   corpus now exists and has never been used to check where real K values actually fall relative to
   these two thresholds — this is ready to refine *now*, without any new data collection: plot the K
   distribution from `data/processed/ablation_30pct/` against 0.5/5.0 and check whether the routing
   split looks sane (e.g. what fraction of frames route reflex vs. cognitive vs. ignore at these
   defaults) before tuning further.
3. **The Jetson simulation's 4.0× compute-scaling factor has no cited empirical basis.**
   `hardware_targets.md` states the `jetson_orin_nano_8gb` profile scales measured GB10 latency by
   4.0× plus 3 ms overhead, but doesn't say where 4.0× came from — it reads as a conservative round
   number, not a measured or cited ratio. Every "passes the 50 ms budget" claim in the repo rests on
   this constant. Refine by: citing a source for the multiplier, or at minimum running a sensitivity
   check (does the reflex-budget verdict still hold at 5×? at 6×?) before item 6 (physical validation)
   replaces the estimate entirely.
4. **Safety-critical test coverage is thin.** `tests/test_reflex_layer.py` (3 tests) and
   `tests/test_threat_prioritizer.py` (6 tests) — 9 tests total — cover the override/non-reflex/
   physics-fallback happy paths but not edge cases a hard-real-time safety path should have: missing/
   NaN depth mid-track, two objects tying exactly at a threshold boundary, simultaneous reflex and
   cognitive events for different objects in the same frame, negative or zero velocity, and
   division-by-near-zero distance. Refine before claiming the Reflex Layer is "done" rather than "done
   first pass" (as `progress.md` already correctly hedges it).
5. **`BEHAVIOUR_MULTIPLIER` has no grounding, unlike mass now does.** `src/perception_stack/physics.py`
   applies a sparse hand-set multiplier for erratic motion, trip height, and head-height mounting on
   top of the mass-based severity law. The mass exponent now has real accident-literature support
   (`related_work.md` §4), but the behaviour multipliers are still pure design judgment with zero
   literature or data behind them. Refine by either finding supporting injury/hazard literature for
   each specific multiplier or explicitly demoting them to the same "declared limitation" status as λ
   (`ablation_guide.md` §0) rather than leaving them implicitly more credible than they are.
6. **Bearing is computed from a fixed, uncalibrated 70° HFoV assumption.**
   `physics.py: compute_bearing()` and `yolo_tracker.py` both hardcode `hfov_deg=70.0` as "typical
   phone/dashcam lens" — every bearing value (and therefore every reflex-cone and encounter-label
   calculation that uses `bearing_deg`) inherits this uncalibrated constant. Low priority relative to
   items above, but should be an explicit, stated limitation before any bearing-dependent number is
   presented as precise, and should become a per-device calibration step before real deployment.
7. **VLM referee "three distinct families" independence is assumed, not checked.**
   `kinetic_score_opinion.md` §4 argues three model *families* (not three checkpoints of one family)
   reduce correlated-error risk, and `ablation_guide.md` §4a picks Qwen2.5-VL, InternVL3, and Gemma-3
   on that basis — but no one has checked whether these three actually share overlapping pretraining
   image-caption corpora (plausible, since most large-scale VLM pretraining draws from a small number
   of public web-scale sources). Worth a caveat sentence in the eventual paper rather than treating
   "three vendors" as proof of independence.

## Suggested next actions

Given everything above, the highest-leverage next steps, in order:

1. ~~Push the missing run artifacts and reconcile the λ inconsistency~~ — **done 2026-08-27**
   (item 1.1–1.2). Data is real, committed, and audited.
2. **On the remote box: run the stratified severity check, then the VLM referee + human calibration
   on the 219 already-exported disagreement frames** (item 1.3–1.5). The disagreement frames
   themselves no longer need pushing — only the raw CSVs (for the stratified check and the referee's
   `--frames-dir`) and GPU access for the referee are still remote-only.
3. ~~Wire together one full replay end-to-end~~ — **done 2026-08-27** on synthetic data
   (`tools/run_full_pipeline_demo.py`, item 5). Re-run the same orchestrator on real SANPO CSVs once
   available — that part of item 5 is still open.
4. **Start the SLM-1 adapter** (item 2) — now the largest remaining component and the one with the
   longest lead time (data generation, SFT, PPO). `src/cognitive_layer/stub.py` shows exactly where
   it plugs in: replace `stub_semantic_eval()`'s rule-based pick with a real model call, same
   `SemanticEval` return contract, everything else in `src/pipeline/orchestrator.py` stays unchanged.
