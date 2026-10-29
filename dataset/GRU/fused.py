"""
fused.py
========
Append the threat engine's fused ground-plane state to an arrival dataset
(methodology §4.23; pending_work.md P1.3).

The stored inputs (dataset.GRU.samples.FEATURES) are Stage-1 camera-frame
values: distance, bearing and closing speed, with head turns mixed in. The
arrival label is defined on the ground plane, in the walker's direction of
travel. This builder replays every session through the runtime tracker
(ThreatEngine + GroundTracker with fixed_camera_poses.csv, exactly as
tools/replay_threat_engine.py does) and appends, per history step,
FUSED_FEATURES (lateral / forward position, lateral / closing speed, position
uncertainty, in-corridor flag) recorded by priority.FusedHistory, the class
the runtime GRU predictor reads them with. Labels, splits and everything else
are copied unchanged.

    python -m dataset.GRU.fused --dataset dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_official_corridor_w0.75_d3_h20

Writes <dataset>_fused/ next to the input.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dataset.GRU.samples import FEATURES, FUSED_FEATURES, SPLITS, legacy_units, load_dataset, save_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT = PROJECT_ROOT / "simulation/datasets/sanpo/session_audit.json"


class _Recorder:
    """An arrival predictor that only records fused history (the engine calls predict once per frame)."""

    def __init__(self):
        from src.threat_engine.priority import FusedHistory
        self.history = FusedHistory()

    def predict(self, frame_idx, tracks):
        self.history.observe(frame_idx, tracks)
        return {}


def fuse_session(args: tuple) -> list[dict]:
    """One session's samples with FUSED_FEATURES appended to x."""
    samples, fx, width, legacy = args
    from src.threat_engine.engine import ThreatEngine
    from src.threat_engine.fusion import GroundTracker
    from tools.replay_threat_engine import STEP_S, STRIDE, session_poses, session_rows

    sid = samples[0]["session"]
    frames = session_rows(samples, fx, width, legacy)
    poses = session_poses(sid)
    rec = _Recorder()
    engine = ThreatEngine(predictor=rec, tracker=GroundTracker())
    by_frame = {s["frame_idx"]: s for s in samples}
    history = samples[0]["x"].shape[1]
    out = []
    for k in sorted(frames):
        engine.step(k, k * STEP_S, frames[k], poses[min(STRIDE * k, len(poses) - 1)])
        s = by_frame[k]
        fused = np.stack([rec.history.features(tid, k, history) for tid in s["track_ids"]]) if s["track_ids"] \
            else np.zeros((0, history, len(FUSED_FEATURES)), np.float32)
        out.append({**s, "x": np.concatenate([s["x"], fused], axis=-1).astype(np.float32)})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    started = time.time()
    manifest, splits = load_dataset(args.dataset)
    if manifest["features"] != FEATURES:
        raise SystemExit(f"{args.dataset} already has fused features")
    legacy = legacy_units(manifest)
    audit = {r["session_id"]: r for r in json.loads(AUDIT.read_text(encoding="utf-8"))["sessions"]}
    out_dir = args.dataset.with_name(args.dataset.name + "_fused")
    out_dir.mkdir(exist_ok=True)
    for split in SPLITS:
        by = defaultdict(list)
        for s in splits[split]:
            by[s["session"]].append(s)
        jobs = [(sorted(v, key=lambda s: s["frame_idx"]), audit[sid]["fx"], audit[sid]["image_width"], legacy)
                for sid, v in sorted(by.items())]
        with ProcessPoolExecutor(args.workers) as pool:
            fused = [s for session in pool.map(fuse_session, jobs) for s in session]
        save_split(fused, out_dir / f"{split}.npz")
        print(f"{split}: {len(jobs)} sessions, {len(fused)} frames ({time.time() - started:.0f} s)", flush=True)
    manifest = {**manifest, "name": out_dir.name, "features": FEATURES + FUSED_FEATURES,
                "fused": {"source": args.dataset.name, "builder": "dataset/GRU/fused.py",
                          "created": time.strftime("%Y-%m-%d"),
                          "note": "FUSED_FEATURES from the runtime GroundTracker with fixed_camera_poses.csv"}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
