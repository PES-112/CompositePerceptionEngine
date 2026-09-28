# Datasets

Model-ready datasets built from the Stage-1 perception CSVs, one folder per model family. `models/`
holds the models, and this folder holds what they train on.

```text
dataset/
└── GRU/                  # Shared by models/GRU/GRU and models/GRU/GRU_attention
    ├── samples.py        # Sample construction, encounter labels, session split, .npz format
    ├── build.py          # CLI: CSVs -> dataset/GRU/<name>/
    └── <name>/           # One built dataset
        ├── train.npz, val.npz, test.npz
        └── manifest.json # Build settings, source fingerprint, session split, label stats
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
python -m dataset.GRU.build --csv-dir data/processed/ablation_30pct
# -> dataset/GRU/ablation_30pct_h8_current_m32_s0/
```

| Option | Default | Meaning |
|---|---|---|
| `--history` | 8 | Frames of history per object (0.8 s at the 10 Hz stride) |
| `--max-objects` | 32 | Keep the N nearest objects per frame |
| `--future-only` | off | Label on frames after T only; the default includes T, matching the K0 ablation |
| `--split-seed` | 0 | Session-level 70/15/15 split |

**What's in a sample.** One sample is one frame. For each object in the frame it holds:
- the object's history of runtime features (see `FEATURES` in `samples.py`), with no kinetic score;
- its class;
- whether it becomes an encounter (within 1.5 m inside the ±30° cone within 20 source frames). This
  label comes straight from `evaluation/kinetic_ablation.py`.
- the raw distance, bearing and K0 values, which evaluation uses for the baselines.

**File format.** The `.npz` files contain plain arrays and never use pickle, so they're safe to share
or upload, for example as a Kaggle Dataset.

**Committing.** A 139-session build should be a few MB, fine to commit next to the models trained on
it. The builder prints each file's size and warns above 50 MB. GitHub rejects files over 100 MB.
