# Kinetic Score Ablation — Run Guide

End-to-end procedure for defending the kinetic score `K = severity(c) · v^γ / max(d, ε)`.
Everything below runs on your own hardware: SANPO streams from a public bucket, the referees are
local VLMs. No API keys, no dataset copy, no data leaving the box.

Read `docs/kinetic_score_opinion.md` first if you want the *why*. This file is the *what to type*.

---

## 0. What we are actually testing

Not "which of six formulas is best" — the old K1–K5 contest was circular and was deleted. We ablate
K0's **own terms** and ask, for each, whether removing or changing it measurably degrades the
ranking. An arm that changes nothing is a term we should not be claiming credit for in the paper.

| Arm | Question it answers |
|---|---|
| `K0  sev·v²/d` | baseline |
| `linear  sev·v/d` | is the `v²` exponent doing work, or would `v` do? |
| `no-severity  v²/d` | does class severity change any ranking? |
| `no-velocity  sev/d` | does velocity change any ranking? (sanity arm — it must fail) |
| `size  sev·v²·s^½/d` | does apparent bbox size `A/d` add anything beyond class + depth? |
| `ttc  -(d-D)/v` | is K beaten by plain time-to-hazard, which we already compute? |

> **λ is frozen at 0.5 and is no longer an arm (2026-08-25).** Dropping Tier-B blinded
> human labelling removed the only tier that could adjudicate between λ values — the
> label-free metrics score arrival time, which λ barely moves. λ and the behaviour
> multipliers are declared design choices and must be written up as a limitation, not
> swept. See `src/perception_stack/physics.py`.

The `no-velocity` arm exists to prove the harness can detect a broken formula. If it ever *wins*,
distrust the metrics, not the finding.

---

## 1. Environment (once, on the SSH box)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # includes ultralytics, torch, scipy
python -c "import torch; print(torch.cuda.is_available())"    # want True for Stage 1
mkdir -p logs
```

Self-checks — run these before anything long. Each is a few seconds and needs no data:

```bash
python src/perception_stack/physics.py --self-check
python evaluation/kinetic_ablation.py --self-check
python evaluation/vlm_referee.py --self-check
```

---

## 2. Stage 1 — perception over 30% of the valid streams

```bash
# See what will be processed without touching the network hard
python tools/stream_sanpo_perception.py --dry-run

# Smoke test: 2 sessions, 30 frames each (~2 min)
python tools/stream_sanpo_perception.py --max-sessions 2 --max-frames 30 \
    --out-dir data/processed/smoke

# The real run — hours. Detach it.
nohup python tools/stream_sanpo_perception.py \
    --out-dir data/processed/ablation_30pct > logs/stage1.log 2>&1 &
tail -f logs/stage1.log
```

**Sampling.** 30% of the **462 sessions** (not 30% of frames), seeded at `--seed 20260819`, sorted
before sampling so the sample depends only on the seed. Sessions, not frames, because the bootstrap
resamples sessions — 140 diverse sessions beat 46 sessions' worth of extra frames.

**Disk.** One session at a time is fetched to `data/.sanpo_scratch/<sid>/` and deleted after its CSV
is written. Peak disk is tens of MB regardless of run length. `--download-workers 8` is the default;
raise it on a fast link, lower it if GCS starts throttling.

**Interruptions.** Finished sessions are skipped on restart; CSVs are written `.partial` and renamed
only on success; a failing session is logged and the run continues. Re-running the same command
after a dropped SSH session resumes.

**Do not change `--frame-stride` without `--fps`.** Frames are strided at download time, so the
pipeline's effective rate is `fps / stride`. The script does this for you; a hand-edited stride with
an unchanged fps inflates every closing velocity by exactly `stride`×.

Output: `<session_id>.csv` (the ablation input) and `<session_id>.frames.json` (frame_idx → GCS
object, so the referee can re-fetch the handful of images it needs).

---

## 3. Ablation + label-free metrics

```bash
python evaluation/kinetic_ablation.py \
    --csv-dir data/processed/ablation_30pct \
    --out-dir evaluation/benchmarks/kinetic_score_eval/run_$(date +%Y_%m_%d)
```

Produces `report.md`, `metrics.json`, `disagreements.json`, `disagreements_key.json`.

Metrics, all label-free (§5 of the opinion doc): flicker rate, rank stability (Kendall τ under
2/5/10% depth perturbation), temporal smoothness, tie rate, future self-consistency, plus the
automatic **encounter** label of §6 — an object counts as a true encounter if its *measured* future
distance drops below `D_haz` within `|bearing| < θ`. That label uses only distance and bearing,
never velocity or severity, so it cannot flatter any arm.

All confidence intervals are a percentile bootstrap resampling **whole sessions**. Frames inside a
session are autocorrelated; frame-level CIs would be several times too narrow and would make every
arm look significantly different from every other.

**Read this report before doing anything else.** If an arm's CI already excludes K0 on the encounter
metric, that arm is settled and does not need a referee. Step 4 exists only for the arms that tie.

---

## 4. Local VLM referees

Only ~5% of frames are ones where two arms pick different top objects — that is what makes hand-
checkable refereeing affordable, and it is the only place a referee's opinion can change an outcome.

### 4a. Serve three models

Three *families*, not three checkpoints of one family: two Qwen sizes agreeing measures a shared
prior, not truth. Any OpenAI-compatible server works (vLLM, Ollama, llama.cpp's `llama-server`,
LM Studio, SGLang).

```bash
# One 24 GB GPU holds these one at a time; a 48 GB card holds two.
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001 --max-model-len 8192 &
vllm serve OpenGVLab/InternVL3-8B      --port 8002 --max-model-len 8192 --trust-remote-code &
vllm serve google/gemma-3-12b-it       --port 8003 --max-model-len 8192 &

curl -s localhost:8001/v1/models | head    # confirm before spending an evening
```

Only one GPU? Run the referees sequentially against the same port, swapping models between runs —
ballots are per-referee files, so this is three separate invocations:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001
python evaluation/vlm_referee.py --run-dir $RUN --frames-dir data/processed/ablation_30pct \
    --referee qwen2.5-vl --endpoint qwen2.5-vl=http://localhost:8001/v1
# kill it, serve the next model on the same port, repeat with --referee internvl3, then gemma3
```

### 4b. Judge

```bash
RUN=evaluation/benchmarks/kinetic_score_eval/run_$(date +%Y_%m_%d)

# 5 cases first — check the replies parse before committing hours of GPU time
python evaluation/vlm_referee.py --run-dir $RUN --frames-dir data/processed/ablation_30pct \
    --referee qwen2.5-vl --limit 5

python evaluation/vlm_referee.py --run-dir $RUN --frames-dir data/processed/ablation_30pct \
    --referee qwen2.5-vl --referee internvl3 --referee gemma3
```

Referees see the annotated frame and a neutral object list in randomised order — no scores, no
formula names, no hint which object either arm picked. A reply that does not parse, or names an
object that is not in the list, is recorded as `None` rather than guessed at. Ballots are written
after every case, so an interrupted run resumes.

If a model returns a low valid-vote count (printed at the end of its run), its instruction-following
is the problem, not the formula. Swap the model rather than loosening the parser.

### 4c. Human calibration — not optional

```bash
python evaluation/vlm_referee.py --run-dir $RUN --human-template
# label 100–150 cases in the template, save as $RUN/human_labels.json
python evaluation/vlm_referee.py --run-dir $RUN --score-only
```

Until `human_labels.json` exists the report prints `MISSING` where human κ belongs, and the VLM win
rates are **not evidence** — three models agreeing can be one shared prior three times. Interpret:
κ < 0.4 means the referees are not measuring what a human means by "top threat" and the numbers stay
out of the paper; κ > 0.6 means the VLM win rates can stand in for a much larger human label set.

Report pairwise referee κ too. Low inter-referee κ with high human κ for one model means use that
model alone, and say so.

---

## 5. What to write up

For each arm: win rate vs. K0 on disagreement frames, the label-free metric deltas with session-level
CIs, and the encounter-metric result. Then one of three conclusions, stated plainly:

- **arm loses** → the term is doing work, keep it, report the effect size;
- **arm ties** → the term is not earning its place; either drop it or state that it is retained for
  reasons other than measured ranking quality;
- **arm wins** → change the default in `src/perception_stack/physics.py` and re-run everything
  downstream of it.

The λ arms are the likeliest to tie: with `v²` in the score, kinematics dominate severity in most
reorderings. A tie there is a real finding — it means the mass exponent is a low-sensitivity knob and
the paper should not oversell it.

## 5b. Optional: top-3 VLM agreement check (a different question from the ablation)

This checks something the ablation doesn't: whether an independent VLM agrees with K0's top-3 picks
at all, with **no competing formula involved** — just K0 as shipped, checked against blind judgment.
Useful as a standalone sanity check even before or without running the ablation above.

```bash
RUN=evaluation/benchmarks/topk_threat_eval/run_$(date +%Y_%m_%d)

# build the scenes: picks K0's top-3 per busy scene, hides the answer
python evaluation/topk_threat_validation.py \
    --run-dir $RUN \
    --csv-dir data/processed/ablation_30pct

# 5 scenes first, to confirm replies parse
python evaluation/topk_threat_validation.py --run-dir $RUN --frames-dir data/processed/ablation_30pct \
    --referee qwen2.5-vl --limit 5

# the real run — each VLM independently picks its own top 3, no hints
python evaluation/topk_threat_validation.py --run-dir $RUN --frames-dir data/processed/ablation_30pct \
    --referee qwen2.5-vl --referee internvl3 --referee gemma3
```

Writes `$RUN/topk_report.md` automatically. It reports, per model: how often it picked the same 3
objects as K0, how often it agreed on the single worst one, and how much the 3 VLMs agreed with each
other. It also adds a `vlm_majority` row — not a fourth model, but the objects at least 2 of 3 VLMs
agreed on, combined into one consensus answer per scene. That row is the steadier number to quote,
since it doesn't ride on any single model's quirks.

As with the referee step above, this is not evidence on its own without the human calibration set
(§4c) — read it as "the VLMs agree with K0 this much," not as proof.

---

## 6. Run results — 2026-08-26 (DGX Spark)

### Chronological run log

| Time (IST) | Event |
|---|---|
| 2026-08-25 00:32:33 | **Step 1 — Environment setup.** venv created, `pip install -r requirements.txt` (scipy 1.18.1 installed). CUDA confirmed: NVIDIA GB10. Weights found: `training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt` (5.2 MB). |
| 2026-08-25 00:32:35 | **Self-checks.** `physics.py --self-check` OK, `kinetic_ablation.py --self-check` OK, `vlm_referee.py --self-check` OK. |
| 2026-08-25 00:32:39 | **Dry run.** 139/462 sessions sampled (fraction=0.30, seed=20260819), stride 3 → effective 10.00 fps, ≤300 frames/session. |
| 2026-08-25 00:32:39 | **Step 2 — Smoke test.** 2 sessions, 30 frames each. Session 1: 30 frames → 159 rows (23 s). Session 2: 30 frames → 157 rows (20 s). 2 CSVs written, 0 failed. |
| 2026-08-25 00:33:24 | **Step 3 — Full Stage 1 launched** (nohup, PID 422872). 139 sessions from SANPO GCS bucket. |
| 2026-08-26 ~04:30 | **Stage 1 complete.** `written=139 skipped_existing=0 failed=0`. All 139 CSVs in `data/processed/ablation_30pct/`. |
| 2026-08-26 10:32:44 | **Step 4 — Ablation metrics.** `kinetic_ablation.py` over 139 sessions, 19,402 scored frames. 219 disagreement frames exported. |
| 2026-08-26 10:35:07 | **Step 4 complete.** Report written to `evaluation/benchmarks/kinetic_score_eval/run_2026_08_26/`. |

### Hardware

- **Machine:** NVIDIA DGX Spark (`promaxgb10-7bf0`), NVIDIA GB10 GPU
- **Checkpoint:** `training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt` (5.2 MB)
- **CUDA:** True, PyTorch ≥ 2.2

### Ablation report summary

**139 sessions · 19,402 scored frames (≥2 objects)**

All confidence intervals are 95% percentile bootstrap resampling whole sessions.

> [!NOTE]
> **`lam=0.25` and `lam=1.0` are struck from the tables below (resolved 2026-08-27).** The
> lambda-sweep plumbing (`VARIANTS` entries for those two arms, and the code path that
> mutated `physics.SEVERITY_LAMBDA`) was removed in commit `0e6e0d7` ("froze lambda to
> 0.5") on 2026-08-25, the day *before* this run's Step 4 timestamp — the two columns were
> produced by an earlier version of `evaluation/kinetic_ablation.py` (commit `e045310`)
> that monkey-patched `SEVERITY_LAMBDA` per arm, not by the code currently committed.
> **Not a bug in the run** — Tier-B human labelling was dropped before this run, and
> nothing in the pipeline can adjudicate between λ values (see §0), so the plumbing was
> intentionally removed. The raw numbers are preserved in `run_2026_08_26/metrics.json`
> and `run_2026_08_26/report.md` (now committed, along with `disagreements.json` and
> `disagreements_key.json`) for reference, but are not reproducible from current code and
> should not appear in a paper. λ=0.5 is a declared design choice, not a measured result.

#### Key metrics (K0 = baseline)

| Metric | K0 | linear | no-severity | no-velocity | size | ttc |
|---|---|---|---|---|---|---|
| flicker ↓ | **0.975** | 0.975 | 0.975 | 0.994 | 0.974 | 0.572 |
| tie_rate ↓ | **0.001** | 0.001 | 0.001 | 0.979 | 0.001 | 0.000 |
| smoothness ↓ | **1.755** | 1.627 | 1.753 | 0.339 | 1.753 | 6.178 |
| future_consist ↑ | **0.002** | 0.002 | 0.002 | 0.001 | 0.003 | 0.076 |
| encounter_top1 ↑ | **0.307** | 0.307 | 0.307 | 0.342 | 0.307 | 0.067 |
| rank_stab_2% ↑ | **1.000** | 1.000 | 1.000 | 0.524 | 0.999 | 1.000 |
| rank_stab_5% ↑ | **1.000** | 0.999 | 1.000 | 0.387 | 0.998 | 1.000 |
| rank_stab_10% ↑ | **0.998** | 0.998 | 0.997 | 0.283 | 0.999 | 0.258 |

#### Per-arm verdicts

| Arm | Verdict | Interpretation |
|---|---|---|
| **no-velocity** (sev/d) | **LOSES** | tie_rate 0.979, rank stability collapses (0.283–0.524). Velocity is doing real work. Sanity check passes. |
| **ttc** (-(d-D)/v) | **LOSES** | flicker 0.572 (vs 0.975), encounter 0.067 (vs 0.307), smoothness 6.178 (vs 1.755). K0 dominates plain time-to-contact. |
| **linear** (sev·v/d) | **TIES** | Identical to K0 on every metric (CIs fully overlap). The v² exponent is not measurably earning its place over v¹. |
| **no-severity** (v²/d) | **TIES** | Identical to K0. Severity does not change any ranking — kinematics dominate, as predicted in the λ discussion. |
| **size** (sev·v²·s^½/d) | **TIES** | Identical to K0. Apparent bbox size adds nothing beyond class + depth. |
| ~~**lam=0.25**~~ | ~~TIES~~ | _Struck — not reproducible from current code (see note above)._ |
| ~~**lam=1.0**~~ | ~~TIES~~ | _Struck — not reproducible from current code (see note above)._ |

#### Interpretation

- The only terms that measurably affect ranking quality are **velocity** (catastrophic without it) and the **K0 formula structure** (crushes raw TTC).
- Severity, the v² exponent, and bbox size tie on corpus-wide label-free metrics — kinematics (velocity/distance) dominate most reorderings. This does not by itself prove severity is unnecessary: these metrics are computed over the whole 19,402-frame corpus, which likely contains few frames where a severity-differentiated pair (e.g. bus vs. pedestrian) is also kinematically close enough for severity to be the deciding factor — a real effect restricted to a small, diluted minority of frames can look identical to no effect in an aggregate metric. `evaluation/kinetic_ablation_stratified.py` isolates exactly that minority and re-runs the same metrics on it; run it before concluding severity should be dropped or reweighted.
- The λ arms (`lam=0.25`, `lam=1.0`) also tied in the raw run (see `run_2026_08_26/report.md`), corroborating that λ is a low-sensitivity knob — but those specific numbers are not reproducible from the current code, so cite the conclusion, not the struck figures.
- 219 disagreement frames were exported to `evaluation/benchmarks/kinetic_score_eval/run_2026_08_26/disagreements.json` for VLM referee adjudication (Step 4 of this guide).

### Output files

| File | Path |
|---|---|
| Stage 1 CSVs (139) | `data/processed/ablation_30pct/*.csv` |
| Frame maps (139) | `data/processed/ablation_30pct/*.frames.json` |
| Ablation report | `evaluation/benchmarks/kinetic_score_eval/run_2026_08_26/report.md` |
| Metrics JSON | `evaluation/benchmarks/kinetic_score_eval/run_2026_08_26/metrics.json` |
| Disagreements (blind) | `evaluation/benchmarks/kinetic_score_eval/run_2026_08_26/disagreements.json` |
| Disagreement key | `evaluation/benchmarks/kinetic_score_eval/run_2026_08_26/disagreements_key.json` |
| Stage 1 log | `logs/stage1.log` |
| Run log | `logs/ablation_runlog.txt` |

### Next steps

Per §3 of this guide: read the report before doing anything else. The `no-velocity` and `ttc` arms are
settled — their CIs exclude K0. Before trusting the `linear`/`no-severity`/`size` ties, run:

```bash
python evaluation/kinetic_ablation_stratified.py \
    --csv-dir data/processed/ablation_30pct \
    --out-dir evaluation/benchmarks/kinetic_score_eval/run_2026_08_26
```

This restricts the same metrics to only the frames where a severity-differentiated, kinematically-close
pair of objects actually exists — the scenario severity was designed for — and reports what fraction of
the corpus that even is. A tie there is much stronger evidence than a corpus-wide tie. Then run the VLM
referees (§4) on the disagreement frames to determine whether their picks reveal a qualitative
difference a human would care about. See `pending_work.md` §1 for the full ordered list. The
`lam=0.25`/`lam=1.0` inconsistency itself is resolved (struck above, root-caused, raw numbers
committed) — what's still open is running the referees on the frames that actually tied.

---

## 7. Troubleshooting

- **Stage 1 log doesn't show your finetuned checkpoint path** → the code or checkpoint didn't come
  over correctly, or `--model` got overridden. It should print the finetuned checkpoint path
  (`training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt`), not a base model.
- **A `votes_*.json` file has a lot of `null` answers** → that model isn't returning parseable
  replies; swap the model rather than loosening the parser (§4b).
- **`report.md`'s confidence interval already excludes K0 on a stripped-down arm** → that arm is
  settled without needing the referee step at all (§3, last paragraph).
- Both `topk_report.md` (§5b) and `referee_report.md` (§4) will show a "human agreement" line as
  `MISSING` until `human_labels.json` exists (§4c) — that's expected, not a bug.
