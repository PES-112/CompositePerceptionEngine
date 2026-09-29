# Datasets

Model-ready datasets built from the Stage-1 perception CSVs, one folder per model family. `models/`
holds the models, and this folder holds what they train on.

```text
dataset/
└── GRU/                  # Shared by models/GRU/GRU and models/GRU/GRU_attention
    ├── samples.py        # Sample construction, encounter labels, session split, .npz format
    ├── build.py          # CLI: CSVs -> dataset/GRU/<name>/
    └── <name>/           # One built dataset
        ├── train.npz, val.npz, test.npz   # fixed split, or all.npz with --folds
        └── manifest.json # Build settings, source fingerprint, split or folds, label stats
```

## Why both GRU architectures share one dataset

The GRU-vs-GRU-attention comparison is only meaningful if both models see the same samples and the
same train/val/test sessions. Separate datasets would add a second difference besides the attention
layer, so there is deliberately one `dataset/GRU/` for both.

A model family gets its own folder only when its input format really is different. For example, the
SLM-1 dataset will be text prompts (compact fact-sheet history in, predicted outcome out), so it
will live in `dataset/SLM/`.

## Building the GRU dataset

The input is the folder of per-session Stage-1 CSVs used by the K0 ablation
(`data/processed/ablation_30pct/`, produced by `tools/stream_sanpo_perception.py`). Run from the
repo root:

```bash
# Labels identical to the K0 ablation, one fixed split
python -m dataset.GRU.build --csv-dir data/processed/ablation_30pct
# -> dataset/GRU/ablation_30pct_h8_current_m32_s0/

# Tracked objects, 3 m hazard zone, 5-fold cross-validation (what arrival-time training uses)
python -m dataset.GRU.build --csv-dir data/processed/sanpo_real_462 --tracked-only --hazard-m 3 --folds 5
# -> dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_f5_s0/
```

| Option | Default | Meaning |
|---|---|---|
| `--history` | 8 | Frames of history per object (0.8 s at the 10 Hz stride) |
| `--max-objects` | 32 | Keep the N nearest objects per frame |
| `--hazard-m` | 1.5 | Encounter distance; the default matches the K0 ablation |
| `--future-only` | off | Label on frames after T only; the default includes T, matching the K0 ablation |
| `--tracked-only` | off | Drop the untracked `obs_*` depth blobs (see below); the default keeps them, matching the K0 ablation |
| `--folds` | 0 | K > 0: session-level K-fold cross-validation instead of one 70/15/15 split |
| `--split-seed` | 0 | Seed for the split or folds |

**Cross-validation.** With `--folds`, sessions are dealt into folds by how many encounter tracks they
contain, so every fold gets a share of the rare sessions with real encounters. For fold F, the test
set is fold F, validation is fold F+1, and training is the rest. `models/GRU/summarize.py` pools
the test predictions from all folds, so every session is scored exactly once.

**What's in a sample.** One sample is one frame. For each object in the frame it holds:
- the object's history of runtime features (see `FEATURES` in `samples.py`), with no kinetic score;
- its class;
- whether it becomes an encounter: within `--hazard-m` inside the ±30° cone within the next 20
  frames (2 s at 10 Hz). At 1.5 m this is identical to `evaluation/kinetic_ablation.py`'s label,
  which is unit-tested.
- `tte`: frames until it first gets there, or -1 if it doesn't within 2 s. This is what arrival-time
  training (`train.py --time-bins`) learns from.
- the raw distance, bearing, K0 and TTC values, which evaluation uses for the baselines.

**Depth blobs dominate the default label.** In the Stage-1 CSVs, about 90% of "objects" are
`unlabeled_obstacle` rows with IDs like `obs_<frame>_<x>`. `detect_unlabeled_obstacles` in
`src/perception_stack/pipeline.py` produces one per fixed column of the lower 60% of the frame. It
reports the closest depth pixel YOLO didn't claim, and nothing removes the ground.

- **What they are.** On a head-mounted camera they are mostly the ground 1–3 m ahead.
- **No history.** They are re-IDed every frame, so they have no history and K0 is always 0 for them.
- **Their label is trivial.** For a blob, "becomes an encounter" just means "is within 1.5 m in the
  cone right now".
- **They swamp the positives.** In a 43-session sample they were 98.6% of all encounters.

`--tracked-only` removes them, leaving YOLO-tracked objects with real history. At 1.5 m, real
encounters are then rare: 45 encounter tracks in 14 of 139 sessions. At 3 m there are 263 tracks in
35 sessions.

**File format.** The `.npz` files contain plain arrays and never use pickle, so they're safe to share
or upload, for example as a Kaggle Dataset.

**Committing.** A 139-session build should be a few MB, fine to commit next to the models trained on
it. The builder prints each file's size and warns above 50 MB. GitHub rejects files over 100 MB.
