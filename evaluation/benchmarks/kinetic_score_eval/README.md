# Kinetic Score Evaluation Benchmark

Results from evaluating the production kinetic score **K0** (`sev × v² / max(d, ε)`).

> The former K0–K5 comparison harness (`evaluation/kinetic_score_comparison.py`,
> `evaluation/threat_score_eval.py`) has been **deleted**. It was circular — it graded each formula
> against a "ground truth" computed by re-running that same formula on a future frame — and its
> discriminating metric was saturated, passing five of six candidates. K1–K5 were dummies and are
> gone from `src/perception_stack/physics.py`.
>
> See **`docs/kinetic_score_opinion.md`** for the replacement strategy.

## What replaces it

K0 is defended by **ablation of its own terms**, not by a contest against strawmen:

| Variant | Claim under test |
|---|---|
| `sev · v²/d` | K0 as-is (baseline) |
| `sev · v/d` | is the `v²` exponent doing work? |
| `v²/d` | is class severity doing work? |
| `sev/d` | is velocity doing work? |
| `sev · v²·(A/d)^½ / d` | does apparent bounding-box size add anything? |
| `λ = 0.25` / `λ = 1.0` | is the mass exponent too weak or too strong? |
| `-(d − D_haz)/v` | is K beaten by plain time-to-hazard? |

## Metrics requiring no ground truth

Run these first — they may settle the question before any labelling.

1. **Flicker rate** — how often `argmax K` changes identity between consecutive frames.
2. **Rank stability** — Kendall τ between rankings on clean vs. depth-perturbed input.
3. **Temporal smoothness** — mean `|K(t) − K(t−1)| / mean K` per track.
4. **Tie rate** — frames where the top two objects fall within 5%.
5. **Complementarity with SLM-1** — disagreement rate between `argmax K` and SLM-1's pick.
6. **Future self-consistency** — does `argmax K` at T match `argmax K` at T+H?

## Automatic ground truth (eliminates, never selects)

An object is a **true encounter** at frame T if, within horizon H, its *measured* `distance_m` drops
below `D_haz` while `|bearing_deg| < θ`. Uses only `distance_m` and `bearing_deg` — never velocity,
severity, or any K. Report as a sensitivity grid over `(H, D_haz, θ)`, not a single setting.

## The one question needing a human

Whether K0's `v²` and its class severity weights are *right* is a value judgment, settled only by a
**blinded referee on disagreement frames** (§3 of the opinion doc). Budget ~100–300 frames.

Step-by-step commands, model serving, and how to read the output: **`docs/ablation_guide.md`**.

## How to run it

Stage 1 no longer needs a local copy of SANPO — frames stream from the public GCS bucket, one
session at a time, and are deleted after that session's CSV is written.

```bash
# 1. Stage 1 over a seeded 30% sample of the 462 valid streams (long; run under nohup)
nohup python tools/stream_sanpo_perception.py \
    --out-dir data/processed/ablation_30pct > logs/stage1.log 2>&1 &

# 2. Ablation + label-free metrics + disagreement export (no API keys, no labels)
python evaluation/kinetic_ablation.py \
    --csv-dir data/processed/ablation_30pct \
    --out-dir evaluation/benchmarks/kinetic_score_eval/run_<date>

# 3. Blinded referee on the disagreement frames only
python evaluation/vlm_referee.py \
    --run-dir evaluation/benchmarks/kinetic_score_eval/run_<date> \
    --frames-dir data/processed/ablation_30pct \
    --referee qwen2.5-vl --referee internvl3 --referee gemma3
```

Smoke-test step 1 with `--max-sessions 2 --max-frames 30` before committing to the full run, and
step 3 with `--limit 5` before spending an API budget.

**Step 2 may settle the question on its own.** Run it and read the report before paying for step 3.

Bootstrap is at the **session** level throughout — frames within a session are autocorrelated, so
frame-level CIs would be far too narrow.

## The human calibration set is not optional

`evaluation/vlm_referee.py --human-template` writes a blank ballot. Label 100–150 of the same cases
by hand, save as `human_labels.json`, and the report will carry Cohen's κ between each VLM and the
human. Until that file exists the report prints `MISSING` where κ should be — and VLM win rates
without it are not evidence, because three models from three vendors agreeing may still be three
draws from one shared prior.
