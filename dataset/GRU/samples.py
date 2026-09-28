"""
samples.py
==========
The GRU dataset: Stage-1 perception CSVs -> per-frame samples, plus the
on-disk format that dataset/GRU/build.py writes and models/GRU/ trains from.

One sample is one frame. For every object in that frame it holds the object's
own last `history` frames, its class, and whether it becomes an encounter. The
label is the formula-free one from evaluation/kinetic_ablation.py (distance
< 1.5 m inside a ±30° cone within 20 source frames), imported rather than
re-implemented so GRU scores stay directly comparable with K0's encounter_top1.

Both GRU architectures read this one dataset. The plain GRU scores each object
from its own history; the attention variant also lets objects in the same
frame see each other. Identical samples and split are what make that the only
difference between them.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation.kinetic_ablation import (
    D_HAZ_M,
    HORIZON_FRAMES,
    THETA_DEG,
    encounters,
    physics,
    prepare,
)

# training/configs/cpe_hazard_classes.yaml, same order. Anything else -> UNKNOWN_CLASS.
CLASS_VOCAB = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light",
    "stop sign", "fire hydrant", "pole", "bollard", "stairs", "crosswalk",
    "pothole", "puddle", "dog", "bench",
]
UNKNOWN_CLASS = len(CLASS_VOCAB)
N_CLASSES = len(CLASS_VOCAB) + 1
_CLASS_INDEX = {name: i for i, name in enumerate(CLASS_VOCAB)}

# Per-timestep inputs, all derivable on-device at runtime. kinetic_score is
# deliberately absent: given K, a model can learn to copy K0 instead of
# learning what actually happens next.
FEATURES = [
    "present",     # 1 if the track was observed at this step, else the step is padding
    "distance",    # distance_m / 10
    "velocity",    # velocity_ms / 5 — closing speed, clamped >= 0 upstream
    "bearing",     # bearing_deg / 35 (half the 70° HFoV)
    "d_distance",  # metres per step since the previous observed step — shows retreat, which velocity_ms can't
    "d_bearing",   # degrees per step / 10 — lateral motion, which K0 ignores entirely
    "confidence",  # YOLO confidence
    "log_area",    # log1p(bbox area px) / 15
    "age",         # steps since the track first appeared, capped at 30, / 30
]
N_FEATURES = len(FEATURES)

DEFAULT_HISTORY = 8        # steps per object (0.8 s at the default 10 Hz stride)
DEFAULT_MAX_OBJECTS = 32   # nearest-N per frame; the same cap applies to every baseline
SPLITS = ("train", "val", "test")

# Per-object arrays stored in each split file, with their on-disk dtype.
_OBJECT_ARRAYS = {"x": np.float32, "cls": np.int64, "y": np.float32,
                  "k0": np.float64, "distance": np.float64, "bearing": np.float64}


# ── Building samples from CSVs ───────────────────────────────────────────────

def encounter_labels(df: pd.DataFrame, future_only: bool = False) -> dict[int, set[str]]:
    """
    Default is exactly kinetic_ablation.encounters(): the window includes frame
    T itself, so an object already inside the hazard zone counts.
    future_only=True looks at (T, T+H] only, so being inside the zone at T is
    not by itself enough.
    """
    if not future_only:
        return encounters(df)
    close = df[(df["distance_m"] < D_HAZ_M) & (df["bearing_deg"].abs() < THETA_DEG)]
    by_track = close.groupby("track_id")["frame_idx"].apply(list).to_dict()
    hits: dict[int, set[str]] = {}
    for frame_idx in df["frame_idx"].unique():
        present = {tid for tid, frames in by_track.items()
                   if any(frame_idx < f <= frame_idx + HORIZON_FRAMES for f in frames)}
        if present:
            hits[int(frame_idx)] = present
    return hits


def _stride(frame_idx: np.ndarray) -> int:
    """Source frames between consecutive processed frames (3 for the 10 Hz ablation run)."""
    diffs = np.diff(np.unique(frame_idx))
    if len(diffs) == 0:
        return 1
    values, counts = np.unique(diffs, return_counts=True)
    return int(values[np.argmax(counts)])


def session_samples(
    raw: pd.DataFrame,
    session_id: str,
    *,
    history: int = DEFAULT_HISTORY,
    future_only: bool = False,
    max_objects: int = DEFAULT_MAX_OBJECTS,
) -> list[dict]:
    """One sample per frame of one session's Stage-1 CSV."""
    df = prepare(raw)   # the ablation's own row filter: depth-less rows dropped
    if df.empty:
        return []
    labels = encounter_labels(df, future_only)
    stride = _stride(df["frame_idx"].to_numpy())

    cols = ["frame_idx", "track_id", "confidence", "bearing_deg", "distance_m", "bbox_area_px", "velocity_ms"]
    obs: dict[str, dict[int, tuple]] = {}
    for frame_idx, tid, conf, bearing, dist, area, vel in df[cols].itertuples(index=False, name=None):
        obs.setdefault(tid, {})[int(frame_idx)] = (conf, bearing, dist, area, vel)
    first_seen = {tid: min(frames) for tid, frames in obs.items()}

    samples = []
    for frame_idx, g in df.groupby("frame_idx", sort=True):
        frame_idx = int(frame_idx)
        if len(g) > max_objects:
            g = g.nsmallest(max_objects, "distance_m")
        tids = g["track_id"].tolist()

        x = np.zeros((len(tids), history, N_FEATURES), dtype=np.float32)
        for i, tid in enumerate(tids):
            track, prev = obs[tid], None
            for k in range(history):
                f = frame_idx - (history - 1 - k) * stride
                if f not in track:
                    continue
                conf, bearing, dist, area, vel = track[f]
                if prev is None:
                    d_dist = d_bearing = 0.0
                else:
                    gap = k - prev[0]
                    d_dist = (dist - prev[1]) / gap
                    d_bearing = (bearing - prev[2]) / gap
                age = min((f - first_seen[tid]) / stride, 30.0)
                x[i, k] = (1.0, dist / 10, vel / 5, bearing / 35, d_dist, d_bearing / 10,
                           conf, np.log1p(max(area, 0.0)) / 15, age / 30)
                prev = (k, dist, bearing)

        enc = labels.get(frame_idx, set())
        samples.append({
            "session": session_id,
            "frame_idx": frame_idx,
            "track_ids": tids,
            "x": x,
            "cls": np.array([_CLASS_INDEX.get(c, UNKNOWN_CLASS) for c in g["class"]], dtype=np.int64),
            "y": np.array([tid in enc for tid in tids], dtype=np.float32),
            # The ablation's encounter_top1 counts a frame whenever *any* track
            # closes in within the horizon — including ones not visible yet —
            # so the evaluation needs this separately from y.
            "frame_has_encounter": bool(enc),
            # Raw values for the baselines in models/GRU/evaluate.py.
            "k0": np.array([physics.kinetic_score(d, v, c) for d, v, c in
                            zip(g["distance_m"], g["velocity_ms"], g["class"])], dtype=float),
            "distance": g["distance_m"].to_numpy(dtype=float),
            "bearing": g["bearing_deg"].to_numpy(dtype=float),
            "classes": g["class"].tolist(),
        })
    return samples


def csv_fingerprint(csvs: list[Path]) -> str:
    """Cheap identity of a CSV folder (names + sizes), recorded so a dataset can be traced to its source."""
    return hashlib.md5("|".join(f"{p.name}:{p.stat().st_size}" for p in csvs).encode()).hexdigest()[:10]


def load_corpus(csv_dir: Path, *, history: int = DEFAULT_HISTORY, future_only: bool = False,
                max_objects: int = DEFAULT_MAX_OBJECTS) -> dict[str, list[dict]]:
    """{session_id: samples} for every <session_id>.csv in csv_dir (the ablation's input layout)."""
    csvs = sorted(Path(csv_dir).glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No per-session CSVs in {csv_dir}")
    corpus = {}
    for path in csvs:
        samples = session_samples(pd.read_csv(path), path.stem, history=history,
                                  future_only=future_only, max_objects=max_objects)
        if samples:
            corpus[path.stem] = samples
    return corpus


def split_sessions(session_ids: list[str], seed: int = 0,
                   val_frac: float = 0.15, test_frac: float = 0.15) -> dict[str, list[str]]:
    """Split by session, never by frame — neighbouring frames are near-duplicates."""
    ids = sorted(session_ids)
    if len(ids) < 3:
        raise ValueError(f"Need at least 3 sessions for a train/val/test split, got {len(ids)}")
    random.Random(seed).shuffle(ids)
    n_test = max(1, round(len(ids) * test_frac))
    n_val = max(1, round(len(ids) * val_frac))
    return {"test": ids[:n_test], "val": ids[n_test:n_test + n_val], "train": ids[n_test + n_val:]}


# ── On-disk format ───────────────────────────────────────────────────────────
# One compressed .npz per split: per-object arrays concatenated across frames,
# plus `offsets` marking where each frame's objects start. Plain arrays only —
# loading never unpickles, so a dataset is safe to pass between machines.

def save_split(samples: list[dict], path: Path) -> None:
    if not samples:
        raise ValueError(f"Refusing to write an empty split to {path}")
    counts = [len(s["y"]) for s in samples]
    arrays = {key: np.concatenate([s[key] for s in samples]).astype(dtype)
              for key, dtype in _OBJECT_ARRAYS.items()}
    np.savez_compressed(
        path,
        offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        session=np.array([s["session"] for s in samples]),
        frame_idx=np.array([s["frame_idx"] for s in samples], dtype=np.int64),
        frame_has_encounter=np.array([s["frame_has_encounter"] for s in samples], dtype=bool),
        track_ids=np.array([t for s in samples for t in s["track_ids"]]),
        classes=np.array([c for s in samples for c in s["classes"]]),
        **arrays,
    )


def load_split(path: Path) -> list[dict]:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    o = d["offsets"]
    samples = []
    for i in range(len(d["frame_idx"])):
        a, b = o[i], o[i + 1]
        sample = {key: d[key][a:b] for key in _OBJECT_ARRAYS}
        sample.update(session=str(d["session"][i]), frame_idx=int(d["frame_idx"][i]),
                      frame_has_encounter=bool(d["frame_has_encounter"][i]),
                      track_ids=d["track_ids"][a:b].tolist(), classes=d["classes"][a:b].tolist())
        samples.append(sample)
    return samples


def load_dataset(dataset_dir: Path, splits: tuple[str, ...] = SPLITS) -> tuple[dict, dict[str, list[dict]]]:
    """(manifest, {split: samples}) for a folder written by dataset/GRU/build.py."""
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["features"] != FEATURES:
        raise ValueError(f"{dataset_dir} was built with a different feature list — rebuild it with dataset/GRU/build.py")
    return manifest, {split: load_split(dataset_dir / f"{split}.npz") for split in splits}


def collate(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Pad frames to the batch's largest object count; `mask` marks real objects."""
    n = max(len(s["y"]) for s in samples)
    history = samples[0]["x"].shape[1]
    x = torch.zeros(len(samples), n, history, N_FEATURES)
    cls = torch.full((len(samples), n), UNKNOWN_CLASS, dtype=torch.long)
    y = torch.zeros(len(samples), n)
    mask = torch.zeros(len(samples), n, dtype=torch.bool)
    for b, s in enumerate(samples):
        k = len(s["y"])
        x[b, :k] = torch.from_numpy(s["x"])
        cls[b, :k] = torch.from_numpy(s["cls"])
        y[b, :k] = torch.from_numpy(s["y"])
        mask[b, :k] = True
    return {"x": x, "cls": cls, "y": y, "mask": mask}
