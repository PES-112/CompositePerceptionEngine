"""
compare_kalman_noise.py
=======================
Paired before/after of the fusion noise model on the replay: the same engine
arm run with the 2026-10-14 declared noise (NoiseModel.declared()) and with the
train-fitted default (NoiseModel()), on the same sessions and knobs.

For each arm and knob it reports warned, false alerts/min and precision for
both, and the difference (after - before) with a 95% paired session-bootstrap
CI (the same sessions are resampled for both sides). Run on validation by
default; --split test scores the official test sessions (report, don't tune).

    python tools/compare_kalman_noise.py --split test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.GRU.samples import load_dataset
from src.threat_engine.announcer import AnnouncementManager
from src.threat_engine.engine import ThreatEngine
from src.threat_engine.fusion import GroundTracker, NoiseModel
from src.threat_engine.priority import PhysicsArrivalPredictor
from tools.replay_threat_engine import (AUDIT, DEFAULT_DATASET, N_BOOT, STEP_S, STRIDE, point, session_metrics,
                                        session_poses, session_rows)

ARMS = {"v2_physics": True, "v2_physics_nopose": False}
KNOBS = (0.5, 0.95)
METRICS = {"warned": 0, "precision": 1, "false_per_min": 3, "alerts_per_min": 4}


def run(frames, poses, noise: NoiseModel, knob: float):
    engine = ThreatEngine(predictor=PhysicsArrivalPredictor(), ignore_threshold=knob,
                          tracker=GroundTracker(noise=noise), announcer=AnnouncementManager())
    log = []
    for k in sorted(frames):
        pose = poses[min(STRIDE * k, len(poses) - 1)] if poses else None
        f = engine.step(k, k * STEP_S, frames[k], pose)
        if f.announcement:
            log.append((k, f.announcement.track_id, f.announcement.kind))
    return session_metrics(frames, log, 20)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "evaluation/benchmarks/kalman_noise_fit")
    args = p.parse_args()
    started = time.time()
    _, splits = load_dataset(DEFAULT_DATASET, splits=(args.split,))
    audit = {r["session_id"]: r for r in json.loads(AUDIT.read_text(encoding="utf-8"))["sessions"]}
    by = defaultdict(list)
    for s in splits[args.split]:
        by[s["session"]].append(s)
    ids = sorted(by)
    frames = {sid: session_rows(by[sid], audit[sid]["fx"], audit[sid]["image_width"]) for sid in ids}
    with ThreadPoolExecutor(max_workers=16) as pool:
        poses = dict(zip(ids, pool.map(session_poses, ids)))

    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(ids), len(ids)) for _ in range(N_BOOT)]
    lines = [f"# Kalman noise: declared vs fitted, paired ({args.split}, {len(ids)} sessions)", "",
             "after = train-fitted NoiseModel(), before = NoiseModel.declared(); difference after - before with a "
             "95% paired session-bootstrap CI. PhysicsArrivalPredictor (mean-state corridor entry), same announcer.",
             "", "| arm | knob | metric | before | after | after - before [95% CI] |", "|---|---:|---|---:|---:|---|"]
    out = {}
    for arm, use_pose in ARMS.items():
        for knob in KNOBS:
            per = {name: [run(frames[sid], poses[sid] if use_pose else None, noise, knob) for sid in ids]
                   for name, noise in (("before", NoiseModel.declared()), ("after", NoiseModel()))}
            full = {n: point(v) for n, v in per.items()}
            boot = np.array([point([per["after"][j] for j in d]) - point([per["before"][j] for j in d]) for d in draws])
            for metric, i in METRICS.items():
                lo, hi = np.nanpercentile(boot[:, i], [2.5, 97.5])
                diff = full["after"][i] - full["before"][i]
                out[f"{arm}|{knob}|{metric}"] = {"before": full["before"][i], "after": full["after"][i],
                                                  "diff": diff, "ci": [lo, hi]}
                lines.append(f"| {arm} | {knob:g} | {metric} | {full['before'][i]:.2f} | {full['after'][i]:.2f} | "
                             f"{diff:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
            print(f"{arm} {knob}: {time.time() - started:.0f} s", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"replay_before_after_{args.split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / f"replay_before_after_{args.split}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
