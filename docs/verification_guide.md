# Kinetic Score Verification — How To Run It

This is the one guide you need to run the full check on the threat-scoring formula (K0): run your
finetuned YOLO over some real SANPO video, then check the formula two ways — does an independent AI
agree with it, and does each piece of the formula actually matter. Plain-English, in the order you
actually run things.

For the full technical reasoning behind any step, see `docs/ablation_guide.md` and
`docs/kinetic_score_opinion.md`. This file is just the how-to.

## What you need first

- Your finetuned YOLO checkpoint present on the machine
  (`training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt` — the "preferred detector" per
  `models/yolo/cpe_yolo26n_hazards_v3_from_base/README.md`, not the older v1 checkpoint) —
  it's gitignored, so make sure the file itself got copied over, not just the code. If it's not
  there yet, `--model <path>` overrides the default.
- A GPU, to run the 3 AI judge models.
- Internet access (to pull SANPO video frames, and to download the 3 AI models the first time).

## 0. One-time setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

Quick self-checks — a few seconds each, no data needed:

```bash
python src/perception_stack/physics.py --self-check
python evaluation/kinetic_ablation.py --self-check
python evaluation/vlm_referee.py --self-check
python evaluation/topk_threat_validation.py --self-check
```

## 1. Run your finetuned YOLO over some real SANPO video

```bash
# smoke test first — 2 sessions, 30 frames each, just to confirm it works
python tools/stream_sanpo_perception.py --max-sessions 2 --max-frames 30 --out-dir data/processed/smoke

# the real subset — this defaults to your finetuned checkpoint automatically
python tools/stream_sanpo_perception.py --out-dir data/processed/ablation_30pct
```

This pulls real video + depth frames from the public SANPO bucket, runs your finetuned model on them,
tracks each object across frames, and measures how far away and how fast everything is. Output: one CSV
per video session, in `data/processed/ablation_30pct/`. Everything below reads from this same folder —
skip this step if it's already been run with your finetuned model.

Sanity check: the log's `Model:` line should print your finetuned checkpoint's path, not a base model.

## 2. Start the 3 AI judge models

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001 &
vllm serve OpenGVLab/InternVL3-8B      --port 8002 --trust-remote-code &
vllm serve google/gemma-3-12b-it       --port 8003 &
```

Each downloads the first time you run it — expect this to take a while, and to need internet + disk
space. Keep these running; both checks below use them. Only one GPU? Serve one at a time on the same
port and repeat each judging step per model, swapping `--referee` — every command below is safe to run
that way since answers are saved per model.

## 3. Check 1 — does an AI agree with the formula's top-3?

No competing formula involved here — just the formula as shipped, checked against independent judgment.

**Build the scenes** (picks the formula's top-3 per busy scene, hides the answer):
```bash
RUN=evaluation/benchmarks/topk_threat_eval/run_today

python evaluation/topk_threat_validation.py \
    --run-dir $RUN \
    --csv-dir data/processed/ablation_30pct
```

**Ask the AIs, blind** (each independently picks its own top 3, no hints):
```bash
# 5 scenes first, to check the replies actually parse
python evaluation/topk_threat_validation.py --run-dir $RUN --frames-dir data/processed/ablation_30pct \
    --referee qwen2.5-vl --limit 5

# the real run
python evaluation/topk_threat_validation.py --run-dir $RUN --frames-dir data/processed/ablation_30pct \
    --referee qwen2.5-vl --referee internvl3 --referee gemma3
```

This writes `$RUN/topk_report.md` automatically — no separate scoring step. It tells you, per AI model:
how often it picked the same 3 objects as the formula, how often it agreed on the single worst one, and
how much the 3 AIs agreed with each other. It also adds one extra row, `vlm_majority`: not a fourth AI,
but the objects at least 2 of the 3 AIs agreed on, combined into one consensus answer per scene — this
is the steadier number to quote, since it isn't riding on any single model's quirks.

## 4. Check 2 — does every piece of the formula actually matter? (ablation, run last)

This one takes the formula apart — removes the speed term, removes the class weighting, swaps `speed²`
for plain `speed`, and so on — and checks whether the ranking of "most dangerous object" actually
changes when a piece is removed. If it doesn't change, that piece isn't earning its place.

**Score every version:**
```bash
python evaluation/kinetic_ablation.py \
    --csv-dir data/processed/ablation_30pct \
    --out-dir evaluation/benchmarks/kinetic_score_eval/run_today
```
Read `report.md` first. This also picks out the handful of scenes (usually ~5%) where the full formula
and a stripped-down version actually disagreed on the top object — only those go to the AIs next.

**Ask the AIs about just the disagreements:**
```bash
python evaluation/vlm_referee.py \
    --run-dir evaluation/benchmarks/kinetic_score_eval/run_today \
    --frames-dir data/processed/ablation_30pct \
    --referee qwen2.5-vl --referee internvl3 --referee gemma3
```
Writes `referee_report.md`: how often the full formula's pick beat each stripped-down version's pick,
on the scenes where they disagreed.

## What you get at the end

- **`topk_report.md`** — does an independent AI agree with the formula's top-3 picks.
- **`referee_report.md`** + **`report.md`** — does each piece of the formula actually change the outcome.

Both reports will show a "human agreement" line as missing — that's expected, we're not doing the
hand-labeling pass in this run. It just means: read the AI-agreement numbers as "the AIs agree with the
formula this much," not as final proof. If this ever needs to go in a paper, that's the piece to add
back in — `docs/ablation_guide.md` §4c explains why.

## If something looks wrong

- **Stage 1 log doesn't show the finetuned checkpoint path** → the code or checkpoint didn't come over
  correctly, or `--model` got overridden.
- **A `votes_*.json` file has a lot of `null` answers** → that AI isn't returning parseable replies;
  swap the model rather than loosening the parser.
- **`report.md`'s confidence interval already excludes the full formula on a stripped-down version** →
  that one's settled without needing the AI check at all.
