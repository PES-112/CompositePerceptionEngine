"""
samples.py
==========
The GRU dataset: Stage-1 perception CSVs -> per-frame samples, plus the
on-disk format that dataset/GRU/build.py writes and models/GRU/ trains from.

One sample is one frame. For every object in that frame it holds the object's
own last `history` frames, its class, whether it becomes an encounter, and how
many frames until it does. An encounter is the formula-free label from
evaluation/kinetic_ablation.py: distance < hazard_m inside a ±30° cone within
20 frames (2 s at 10 Hz). At the default hazard_m = 1.5 m the labels are
identical to kinetic_ablation.encounters() (unit-tested), so GRU scores stay
directly comparable with K0's encounter_top1.

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

from evaluation.kinetic_ablation import D_HAZ_M, HORIZON_FRAMES, THETA_DEG, physics, prepare

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

# Depth-only pseudo-detections from src/perception_stack/pipeline.py
# (detect_unlabeled_obstacles): the closest depth pixel in each of 5 fixed
# columns of the lower 60% of the frame, re-IDed every frame as
# obs_<frame>_<cx>. With no ground removal they are mostly the ground just
# ahead of the walker; they are never tracked, so they have no history and
# their velocity (and K0) is always 0.
UNTRACKED_PREFIX = "obs_"

# Per-object arrays stored on disk, with their dtype. tte = frames until the
# object first reaches the hazard zone (-1 if not within the horizon); ttc is
# the physics estimate of the same thing, (d - hazard_m) / v, inf if not closing.
_OBJECT_ARRAYS = {"x": np.float32, "cls": np.int64, "y": np.float32, "tte": np.int64,
                  "k0": np.float64, "ttc": np.float64, "distance": np.float64, "bearing": np.float64}


# ── Building samples from CSVs ───────────────────────────────────────────────

def arrival_times(df: pd.DataFrame, hazard_m: float = D_HAZ_M,
                  future_only: bool = False) -> dict[int, dict[str, int]]:
    """
    {frame T: {track: frames until it is first within hazard_m inside the cone}}
    for every track — visible at T or not — that gets there within
    HORIZON_FRAMES. The window is [T, T+H] like kinetic_ablation.encounters(),
    so an object already inside the zone arrives at 0; future_only=True uses
    (T, T+H], so being inside the zone at T is not by itself enough.
    """
    close = df[(df["distance_m"] < hazard_m) & (df["bearing_deg"].abs() < THETA_DEG)]
    by_track = {tid: np.sort(frames.to_numpy()) for tid, frames in close.groupby("track_id")["frame_idx"]}
    out: dict[int, dict[str, int]] = {}
    for frame_idx in np.unique(df["frame_idx"]):
        frame_idx = int(frame_idx)
        start = frame_idx + 1 if future_only else frame_idx
        hits = {}
        for tid, frames in by_track.items():
            i = np.searchsorted(frames, start)
            if i < len(frames) and frames[i] <= frame_idx + HORIZON_FRAMES:
                hits[tid] = int(frames[i] - frame_idx)
        if hits:
            out[frame_idx] = hits
    return out


def _stride(frame_idx: np.ndarray) -> int:
    """frame_idx step between processed frames (1 for stream_sanpo_perception.py, which renumbers strided frames)."""
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
    tracked_only: bool = False,
    hazard_m: float = D_HAZ_M,
) -> list[dict]:
    """
    One sample per frame of one session's Stage-1 CSV. tracked_only=True drops
    the untracked depth blobs before anything else, so they are neither
    candidates, context, nor encounters.
    """
    if tracked_only:
        raw = raw[~raw["track_id"].astype(str).str.startswith(UNTRACKED_PREFIX)]
    df = prepare(raw)   # the ablation's own row filter: depth-less rows dropped
    if df.empty:
        return []
    arrivals = arrival_times(df, hazard_m, future_only)
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

        hits = arrivals.get(frame_idx, {})
        tte = np.array([hits.get(tid, -1) for tid in tids], dtype=np.int64)
        d = g["distance_m"].to_numpy(dtype=float)
        v = g["velocity_ms"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ttc = np.where(v > 0, (d - hazard_m) / v, np.inf)
        samples.append({
            "session": session_id,
            "frame_idx": frame_idx,
            "track_ids": tids,
            "x": x,
            "cls": np.array([_CLASS_INDEX.get(c, UNKNOWN_CLASS) for c in g["class"]], dtype=np.int64),
            "y": (tte >= 0).astype(np.float32),
            "tte": tte,
            # The ablation's encounter_top1 counts a frame whenever *any* track
            # closes in within the horizon — including ones not visible yet —
            # so the evaluation needs this separately from y.
            "frame_has_encounter": bool(hits),
            # Raw values for the baselines in models/GRU/evaluate.py.
            "k0": np.array([physics.kinetic_score(dd, vv, c) for dd, vv, c in zip(d, v, g["class"])], dtype=float),
            "ttc": ttc,
            "distance": d,
            "bearing": g["bearing_deg"].to_numpy(dtype=float),
            "classes": g["class"].tolist(),
        })
    return samples


def csv_fingerprint(csvs: list[Path]) -> str:
    """Cheap identity of a CSV folder (names + sizes), recorded so a dataset can be traced to its source."""
    return hashlib.md5("|".join(f"{p.name}:{p.stat().st_size}" for p in csvs).encode()).hexdigest()[:10]


def load_corpus(csv_dir: Path, *, history: int = DEFAULT_HISTORY, future_only: bool = False,
                max_objects: int = DEFAULT_MAX_OBJECTS, tracked_only: bool = False,
                hazard_m: float = D_HAZ_M) -> dict[str, list[dict]]:
    """{session_id: samples} for every <session_id>.csv in csv_dir (the ablation's input layout)."""
    csvs = sorted(Path(csv_dir).glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No per-session CSVs in {csv_dir}")
    corpus = {}
    for path in csvs:
        samples = session_samples(pd.read_csv(path), path.stem, history=history, future_only=future_only,
                                  max_objects=max_objects, tracked_only=tracked_only, hazard_m=hazard_m)
        if samples:
            corpus[path.stem] = samples
    return corpus


# ── Session splits ───────────────────────────────────────────────────────────
# Always by session, never by frame — neighbouring frames are near-duplicates.

def split_sessions(session_ids: list[str], seed: int = 0,
                   val_frac: float = 0.15, test_frac: float = 0.15) -> dict[str, list[str]]:
    """One fixed train/val/test split."""
    ids = sorted(session_ids)
    if len(ids) < 3:
        raise ValueError(f"Need at least 3 sessions for a train/val/test split, got {len(ids)}")
    random.Random(seed).shuffle(ids)
    n_test = max(1, round(len(ids) * test_frac))
    n_val = max(1, round(len(ids) * val_frac))
    return {"test": ids[:n_test], "val": ids[n_test:n_test + n_val], "train": ids[n_test + n_val:]}


def fold_sessions(corpus: dict[str, list[dict]], k: int, seed: int = 0) -> list[list[str]]:
    """
    k session-level folds for cross-validation. Encounters cluster in a few
    sessions, so sessions are dealt out in snake order by how many distinct
    encounter tracks they contain — every fold gets a share of them.
    """
    if len(corpus) < k:
        raise ValueError(f"Need at least {k} sessions for {k} folds, got {len(corpus)}")
    ids = sorted(corpus)
    random.Random(seed).shuffle(ids)
    n_tracks = {sid: len({t for s in corpus[sid] for t, y in zip(s["track_ids"], s["y"]) if y}) for sid in ids}
    ids.sort(key=lambda sid: -n_tracks[sid])   # stable: ties keep the shuffled order
    folds: list[list[str]] = [[] for _ in range(k)]
    for i, sid in enumerate(ids):
        lap, pos = divmod(i, k)
        folds[pos if lap % 2 == 0 else k - 1 - pos].append(sid)
    return folds


def fold_split(folds: list[list[str]], fold: int) -> dict[str, list[str]]:
    """Fold `fold` is the test set, the next fold is validation, the rest train."""
    k = len(folds)
    val = (fold + 1) % k
    return {"test": folds[fold], "val": folds[val],
            "train": [sid for i, f in enumerate(folds) if i not in (fold, val) for sid in f]}


# ── On-disk format ───────────────────────────────────────────────────────────
# Compressed .npz files: per-object arrays concatenated across frames, plus
# `offsets` marking where each frame's objects start. A fixed-split dataset has
# train/val/test.npz; a cross-validation dataset has all.npz and its folds in
# manifest.json. Plain arrays only — loading never unpickles, so a dataset is
# safe to pass between machines.

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
        sample = {key: d[key][a:b] for key in _OBJECT_ARRAYS if key in d}   # older datasets lack tte/ttc
        sample.update(session=str(d["session"][i]), frame_idx=int(d["frame_idx"][i]),
                      frame_has_encounter=bool(d["frame_has_encounter"][i]),
                      track_ids=d["track_ids"][a:b].tolist(), classes=d["classes"][a:b].tolist())
        samples.append(sample)
    return samples


def load_dataset(dataset_dir: Path, splits: tuple[str, ...] = SPLITS,
                 fold: int | None = None) -> tuple[dict, dict[str, list[dict]]]:
    """
    (manifest, {split: samples}) for a folder written by dataset/GRU/build.py.
    Cross-validation datasets need `fold`; manifest["split"] is then set to
    that fold's train/val/test sessions.
    """
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["features"] != FEATURES:
        raise ValueError(f"{dataset_dir} was built with a different feature list — rebuild it with dataset/GRU/build.py")
    if "folds" not in manifest:
        if fold is not None:
            raise ValueError(f"{dataset_dir} has a fixed split, not cross-validation folds")
        return manifest, {split: load_split(dataset_dir / f"{split}.npz") for split in splits}
    if fold is None or not 0 <= fold < len(manifest["folds"]):
        raise ValueError(f"{dataset_dir} has {len(manifest['folds'])} folds — pass fold=0..{len(manifest['folds']) - 1}")
    manifest = {**manifest, "fold": fold, "split": fold_split(manifest["folds"], fold)}
    by_session: dict[str, list[dict]] = {}
    for s in load_split(dataset_dir / "all.npz"):
        by_session.setdefault(s["session"], []).append(s)
    return manifest, {split: [s for sid in manifest["split"][split] for s in by_session.get(sid, [])]
                      for split in splits}


def arrival_bins(tte: np.ndarray, n_bins: int, horizon: int = HORIZON_FRAMES) -> np.ndarray:
    """Arrival-time class per object: 0..n_bins-1 = which slice of the horizon it arrives in, n_bins = not within it."""
    width = -(-horizon // n_bins)   # ceil
    return np.where(tte >= 0, np.minimum(tte // width, n_bins - 1), n_bins)


def collate(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Pad frames to the batch's largest object count; `mask` marks real objects."""
    n = max(len(s["y"]) for s in samples)
    history = samples[0]["x"].shape[1]
    x = torch.zeros(len(samples), n, history, N_FEATURES)
    cls = torch.full((len(samples), n), UNKNOWN_CLASS, dtype=torch.long)
    y = torch.zeros(len(samples), n)
    tte = torch.full((len(samples), n), -1, dtype=torch.long)
    mask = torch.zeros(len(samples), n, dtype=torch.bool)
    for b, s in enumerate(samples):
        k = len(s["y"])
        x[b, :k] = torch.from_numpy(s["x"])
        cls[b, :k] = torch.from_numpy(s["cls"])
        y[b, :k] = torch.from_numpy(s["y"])
        if "tte" in s:
            tte[b, :k] = torch.from_numpy(s["tte"])
        mask[b, :k] = True
    return {"x": x, "cls": cls, "y": y, "tte": tte, "mask": mask}
