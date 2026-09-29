# GRU encounter predictors

Two learned alternatives to K0, built to answer one question: **does letting objects in a frame see
each other (scene context) improve threat prediction?** Methodology: `docs/methodology.md` §4.10.

```text
models/GRU/
├── GRU/model.py              # Architecture 1: per-object GRU — each object scored from its own history
├── GRU_attention/model.py    # Architecture 2: same GRU + one attention block across the frame's objects
├── layers.py                 # Shared encoder + head (so the attention block is the only difference)
├── train.py                  # Train either architecture, then evaluate on held-out sessions
├── evaluate.py               # Metrics + baselines (K0, nearest, nearest-in-cone)
├── summarize.py              # Mean ± std across seeds for a run name, next to the baselines
└── registry.py               # arch name -> class, and where runs are written
```

**Latest results:** `evaluation/benchmarks/gru_encounter_eval/run_2026_09_29_arrival462/report.md`
(all 462 curated sessions, arrival-time ranking, 5-fold cross-validation).
- **vs. K0 and TTC:** both GRUs beat them.
- **vs. nearest object in the cone:** the GRU slightly beats it at predicting *whether* objects come
  close, ties it at ranking *who arrives first*, and has a much steadier top pick.
- **Attention:** adds nothing measurable on full data.

Read `dataset/README.md` on depth blobs before interpreting any encounter number.

The data both architectures train on lives in [`dataset/GRU/`](../../dataset/README.md).

## The task

For every object in a frame, predict whether it becomes an **encounter**: it comes within the hazard
distance inside the ±30° forward cone within the next 20 frames (2 s at 10 Hz). At the default
1.5 m the label is identical to `evaluation/kinetic_ablation.py`'s, so results compare directly with
K0's `encounter_top1` in the ablation report.

The models have two output modes:
- **Yes/no (default).** One logit per object: will it arrive within 2 s?
- **Arrival time (`--time-bins K`).** Probabilities over K slices of the 2 s horizon plus "not within
  2 s". That gives both P(arrives), which is 1 − P(not within 2 s), and an expected arrival time,
  so objects can be **ranked by how soon they arrive**.

**Input per object:** its last 8 processed frames (0.8 s at the 10 Hz stride), each with:
- distance, closing velocity and bearing;
- per-step change in distance and bearing, which capture retreat and sideways motion that K0 can't see;
- YOLO confidence, log box area and track age;
- a learned class embedding.

`kinetic_score` is deliberately not an input, so neither model can learn to copy K0.

## The two architectures

```text
                          ┌─ GRU/ ──────────── head ─> p(encounter) per object
object history ─> GRU ────┤
 (per track)              └─ GRU_attention/ ── attention across objects (+residual) ── head ─> p(encounter)
```

- **Residual connection.** If other objects don't help, the attention block can learn to pass the GRU
  output straight through.
- **No positional encoding across objects.** Fact sheets are sorted by K, so list position would leak
  K0's ranking. Without it, object order can't matter; this is unit-tested.
- **Context dropout (training only).** Hides random neighbours from each other, so one bad detection
  can't become something every prediction depends on.

## Running it

Run everything from the repo root. First, build the dataset once from the per-session Stage-1 CSVs
the K0 ablation used (see `dataset/README.md`):

```bash
python -m dataset.GRU.build --csv-dir data/processed/ablation_30pct
```

Then train both architectures on that same dataset:

```bash
python -m models.GRU.train --arch gru           --dataset dataset/GRU/ablation_30pct_h8_current_m32_s0 --seeds 0 1 2
python -m models.GRU.train --arch gru_attention --dataset dataset/GRU/ablation_30pct_h8_current_m32_s0 --seeds 0 1 2
```

For arrival-time ranking with cross-validation, build a `--folds` dataset and add `--time-bins`.
Every fold is trained for each seed:

```bash
python -m dataset.GRU.build --csv-dir data/processed/sanpo_real_462 --tracked-only --hazard-m 3 --folds 5
python -m models.GRU.train --arch gru --dataset dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_f5_s0 \
    --time-bins 4 --seeds 0 1 2 --run-name arrival462
```

Each run writes `models/GRU/<GRU|GRU_attention>/runs/<run-name>[_fold<F>]_seed<N>/`:

| File | Contents |
|---|---|
| `model.pt` | Checkpoint (~130–260 KB) |
| `config.json` | Every setting, the session split, and the parameter count |
| `history.csv` | Train and validation loss per epoch |
| `metrics.json`, `report.md` | Test-set results |
| `test_outputs.npy` | Raw test predictions; `summarize.py` pools these across folds |

The session split is fixed inside the dataset, so both architectures' reports are directly
comparable. Each run also records the dataset's fingerprint.

To re-evaluate a saved run on the dataset it was trained on (it refuses a different one):

```bash
python -m models.GRU.evaluate --run-dir models/GRU/GRU_attention/runs/<run>
```

To compare both architectures across seeds for one run name (cross-validation runs are pooled
across folds first, so every session is scored once):

```bash
python -m models.GRU.summarize --run-name tracked_2026_09_28
```

Training runs on a CPU in minutes; a GPU is used automatically if available. Tests:
`python -m unittest tests.test_gru_models`.

## Reading the results

Each `report.md` has two tables: all frames, and an **interaction slice** (≥ 2 objects within 5 m,
or a crosswalk in view). Compare the two architectures on both. Most frames have no real
interaction, so an effect of attention may only show up in the slice.

| Column | Meaning |
|---|---|
| AUROC, Avg. precision | Per object: how well each object's encounter is predicted |
| encounter_top1 | Per frame: in frames where something closed in, was the top-scored object one of them? Same definition as the K0 ablation. Higher is better. |
| soonest_top1 | Per frame (datasets with arrival times): in frames where a visible object arrives, was the top pick one of the first to arrive? Higher is better. |
| arrival_order | For pairs of objects in a frame where one arrives strictly sooner (or the other never does), how often is the sooner one ranked higher? 0.5 = random, 1.0 = perfect. |
| flicker_rate | Per frame: how often the top pick changes identity between frames. Lower is better. |

Every table also includes these baselines on the same test sessions:
- **K0**, the production formula.
- **TTC** (datasets with arrival times), (distance − hazard) / closing speed. This is physics' own
  arrival estimate and the main baseline for arrival order.
- **nearest**, the closest object.
- **nearest_in_cone**, the closest object inside the ±30° cone. This is a plain rule that mirrors the
  label, so a model has to beat it to show it learned something beyond the label's definition.

**Before crediting scene context:**
- Run ≥ 3 seeds per architecture and check whether the CIs overlap.
- The attention model has ~2× the parameters (≈64k vs. ≈30k). If it wins narrowly, retrain the plain
  GRU with `--hidden 96` (≈67k, about the same parameter count) to rule out plain capacity.

## Known limitations

- The labels come from the same perception stack as the inputs, including its depth noise and
  tracker ID switches. "Predicts encounters" therefore means "predicts what the perception stack
  will measure".
- The default label includes frame T itself, matching the ablation, so an object already inside the
  hazard zone counts as a positive. `--future-only` uses frames after T only.
- Rows without depth are dropped, as in the ablation. Frames are capped at the 32 nearest objects.
  The same cap applies to the baselines.
- Objects interact only through their own ego-relative features. Explicit pairwise relative
  positions are a natural next step if attention shows promise.
