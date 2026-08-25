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

For the top-3-per-scene VLM validation (a different question from this ablation — no competing formula
involved, just K0 checked against independent VLM judgment) see `docs/verification_guide.md`.
