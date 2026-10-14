"""
replay_threat_engine.py
=======================
End-to-end SANPO replay of the runtime engines (pending_work.md P2.1):
v1 (K0 routing -> stub -> PhysicsVerification) against the v2 threat engine,
on SANPO's official split, with system-level metrics.

Input. The GRU arrival dataset already stores, for every processed frame, every
tracked object's Stage-1 values (distance, bearing, closing speed, confidence,
box size) and its arrival label, so the replay rebuilds Stage-1 rows from it
(no CSVs needed). Camera poses come from each session's camera_poses.csv
(public bucket; cached under data/sanpo_poses/). Processed frame k is source
frame 3k (stride 3) and is replayed at t = 0.2·k s (15 FPS source).

Units, corrected on the fly for everything except the GRU, which is fed the
units it was trained in:
  speed    stored speeds are 2x too high (Stage 1 assumed 30 FPS) -> x0.5
  bearing  stored bearings assumed a 70 deg linear FOV -> pinhole angle from
           the session's real fx (simulation/datasets/sanpo/session_audit.json)

Arms (all on the same frames):
  v1_k0               the v1 chain as built; it speaks on every frame with an event
  v1_k0_announcer     v1's pick fed through the v2 announcement manager
  nearest_in_cone     v2 engine, P(arrive) from distance rank inside the cone only
  v2_physics          v2 engine, physics arrival predictor, ego-motion removed
  v2_physics_nopose   same, camera-frame tracking (ego-motion ablation)
  v2_gru              v2 engine, GRU arrival predictor, ego-motion removed
  v2_gru_nopose       same, camera-frame tracking

Metrics (per arrival: a track's first frame inside 3 m in the ±30° cone, among
tracks seen at least one frame earlier):
  warned          an announcement about that track in the 4 s before it arrived
  lead_s          median warning lead time over warned arrivals
  precision       announcements whose object is within 3 m in the cone now or
                  within the next 4 s (the arrival label at that frame)
  alerts_per_min, overrides_per_min, switches_per_min (announced track changed)
Session-bootstrap 95% CIs. The ignore threshold of each v2 arm is chosen on the
official validation sessions (best F1 of warned vs. precision) and then applied
once to the test sessions.

    python tools/replay_threat_engine.py --gru-run models/GRU/GRU/runs/official462_2026_10_14_seed0
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.GRU.samples import load_dataset
from src.narration.pipeline import NarrationPipeline
from src.narration.tts import CachedClipTTS
from src.pipeline.orchestrator import process_frame
from src.threat_engine import envelope
from src.threat_engine.announcer import AnnouncementManager
from src.threat_engine.engine import ThreatEngine
from src.threat_engine.fusion import EgoPose, GroundTracker, sanpo_session_poses
from src.threat_engine.priority import Arrival, GRUArrivalPredictor, PhysicsArrivalPredictor, RankedThreat
from src.threat_prioritizer import build_threat_frame
from tools.audit_sanpo_sessions import fetch, object_url
from tools.download_sanpo_valid_streams import BASE_PREFIX

DEFAULT_DATASET = PROJECT_ROOT / "dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_official"
AUDIT = PROJECT_ROOT / "simulation/datasets/sanpo/session_audit.json"
POSE_CACHE = PROJECT_ROOT / "data/sanpo_poses"
STEP_S = 0.2
STRIDE = 3
LEGACY_HALF_FOV = 35.0
SPEED_FIX = 0.5
WARN_WINDOW_S = 4.0
THRESHOLDS = (0.05, 0.1, 0.15, 0.25, 0.4, 0.6)
N_BOOT = 1000


# ── Inputs ───────────────────────────────────────────────────────────────────

def session_poses(sid: str) -> list[EgoPose]:
    POSE_CACHE.mkdir(parents=True, exist_ok=True)
    path = POSE_CACHE / f"{sid}.csv"
    if not path.exists():
        path.write_bytes(fetch(object_url(f"{BASE_PREFIX}/{sid}/camera_head/camera_poses.csv")))
    return sanpo_session_poses(list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))))


def session_rows(samples: list[dict], fx: float, width: float) -> dict[int, list[dict]]:
    """{frame_idx: Stage-1 rows} in corrected units, with the GRU's legacy inputs kept alongside."""
    frames = {}
    for s in samples:
        rows = []
        for j, tid in enumerate(s["track_ids"]):
            last = s["x"][j, -1]
            legacy_bearing = float(s["bearing"][j])
            dx = legacy_bearing / LEGACY_HALF_FOV * width / 2
            legacy_speed = float(last[2]) * 5
            rows.append({
                "frame_idx": s["frame_idx"], "track_id": tid, "class": s["classes"][j],
                "confidence": float(last[6]), "distance_m": float(s["distance"][j]),
                "bearing_deg": math.degrees(math.atan2(dx, fx)),
                "velocity_ms": legacy_speed * SPEED_FIX,
                "bbox_area_px": float(np.expm1(last[7] * 15)),
                "_legacy": {"bearing_deg": legacy_bearing, "velocity_ms": legacy_speed},
                "_tte": int(s["tte"][j]),
            })
        frames[s["frame_idx"]] = rows
    return frames


def legacy_row(row: dict) -> dict:
    """The row in the units the GRU was trained on."""
    return {**row, **row["_legacy"]}


# ── Arms ─────────────────────────────────────────────────────────────────────

class NearestInConePredictor:
    """P(arrive) = 1 for the nearest object in the cone, decaying with distance rank; 0 outside it."""

    def predict(self, frame_idx, tracks):
        inside = sorted((t for t in tracks if envelope.in_cone(t)), key=lambda t: t.distance_m)
        out = {t.track_id: Arrival(0.0, 5.0) for t in tracks}
        for rank_, t in enumerate(inside):
            out[t.track_id] = Arrival(1.0 / (1 + rank_), t.distance_m)
        return out


class LegacyGRU(GRUArrivalPredictor):
    def observe(self, frame_idx, tracks):
        swapped = [t.__class__(**{**t.__dict__, "row": legacy_row(t.row)}) if t.row else t for t in tracks]
        super().observe(frame_idx, swapped)


def run_v2(frames: dict[int, list[dict]], poses: list[EgoPose] | None, predictor, threshold: float):
    engine = ThreatEngine(predictor=predictor, ignore_threshold=threshold, tracker=GroundTracker(),
                          announcer=AnnouncementManager())
    log = []
    for k in sorted(frames):
        pose = poses[min(STRIDE * k, len(poses) - 1)] if poses else None
        f = engine.step(k, k * STEP_S, frames[k], pose)
        if f.announcement:
            log.append((f.t, f.announcement.track_id, f.announcement.kind))
    return log


def run_v1(frames: dict[int, list[dict]], announcer: bool):
    narration = NarrationPipeline(tts=CachedClipTTS(fallback=b""), critical_tts=CachedClipTTS(fallback=b""))
    mgr = AnnouncementManager() if announcer else None
    log = []
    for k in sorted(frames):
        rows = [{**r, "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 0, "bbox_y2": 0, "cx_px": 0, "source": "sanpo"}
                for r in frames[k]]
        result = process_frame(build_threat_frame(rows), narration)
        ev = result.narrator_event
        if ev is None:
            continue
        t = k * STEP_S
        if mgr is None:
            log.append((t, ev.track_id, "override" if ev.is_override else "threat"))
            continue
        dummy = _ranked_stub(ev.track_id, ev.object_class)   # the announcer only needs the id and kind
        overrides = [envelope.Override(dummy.track, 0.0, ev.reason)] if ev.is_override else []
        a = mgr.step(t, overrides, [] if ev.is_override else [dummy])
        if a:
            log.append((t, a.track_id, a.kind))
    return log


def _ranked_stub(track_id: str, class_name: str) -> RankedThreat:
    from src.threat_engine.fusion import FusedTrack
    tr = FusedTrack(track_id, class_name, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, (0.0, 0.0), 0.0, 0.0, 1, 0, True, True)
    return RankedThreat(tr, 1.0, Arrival(1.0, 0.0), 1.0, 1.0)


# ── Metrics ──────────────────────────────────────────────────────────────────

def arrivals(frames: dict[int, list[dict]]) -> dict[str, float]:
    """{track: arrival time} for tracks first seen before they arrive (a warning was possible)."""
    first_seen, arrive = {}, {}
    for k in sorted(frames):
        for r in frames[k]:
            first_seen.setdefault(r["track_id"], k)
            if r["_tte"] == 0 and r["track_id"] not in arrive:
                arrive[r["track_id"]] = k
    return {tid: k * STEP_S for tid, k in arrive.items() if first_seen[tid] < k}


def session_metrics(frames, log) -> dict:
    arr = arrivals(frames)
    duration_min = max(1, len(frames)) * STEP_S / 60
    warned, leads = 0, []
    for tid, ta in arr.items():
        times = [t for t, a, _ in log if a == tid and ta - WARN_WINDOW_S <= t <= ta]
        if times:
            warned += 1
            leads.append(ta - min(times))
    # precision: the announced object is within 3 m in the cone now or within the next 4 s
    # (the arrival label at that frame, y = 1)
    label = {(k, r["track_id"]): r["_tte"] >= 0 for k, rows in frames.items() for r in rows}
    hits = sum(1 for t, a, _ in log if label.get((round(t / STEP_S), a), False))
    switches = sum(1 for (_, a, _), (_, b, _) in zip(log, log[1:]) if a != b)
    return {"arrivals": len(arr), "warned": warned, "leads": leads, "announcements": len(log),
            "hits": hits, "overrides": sum(1 for *_, kind in log if kind == "override"),
            "switches": switches, "minutes": duration_min}


def aggregate(per_session: list[dict], seed: int = 0) -> dict:
    def point(idx):
        s = [per_session[i] for i in idx]
        arr, ann, mins = sum(x["arrivals"] for x in s), sum(x["announcements"] for x in s), sum(x["minutes"] for x in s)
        leads = [v for x in s for v in x["leads"]]
        return np.array([
            sum(x["warned"] for x in s) / arr if arr else np.nan,
            sum(x["hits"] for x in s) / ann if ann else np.nan,
            np.median(leads) if leads else np.nan,
            ann / mins, sum(x["overrides"] for x in s) / mins, sum(x["switches"] for x in s) / mins,
        ])
    rng = np.random.default_rng(seed)
    n = len(per_session)
    full = point(range(n))
    boot = np.array([point(rng.integers(0, n, n)) for _ in range(N_BOOT)])
    keys = ("warned", "precision", "lead_s", "alerts_per_min", "overrides_per_min", "switches_per_min")
    out = {k: (float(full[i]), float(np.nanpercentile(boot[:, i], 2.5)), float(np.nanpercentile(boot[:, i], 97.5)))
           for i, k in enumerate(keys)}
    out["arrivals"] = int(sum(x["arrivals"] for x in per_session))
    return out


def f1(m: dict) -> float:
    w, p = m["warned"][0], m["precision"][0]
    return 0.0 if not (w and p) or not (np.isfinite(w) and np.isfinite(p)) else 2 * w * p / (w + p)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.disable(logging.CRITICAL)   # PhysicsVerification logs every v1 override and stub disagreement
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--gru-run", type=Path, required=True, help="A models/GRU run trained on --dataset's train split.")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "evaluation/benchmarks/threat_engine_replay")
    p.add_argument("--max-sessions", type=int, default=0, help="Per split; 0 = all (smoke testing).")
    args = p.parse_args()

    manifest, splits = load_dataset(args.dataset, splits=("val", "test"))
    cfg = json.loads((args.gru_run / "config.json").read_text(encoding="utf-8"))
    if cfg["split"] != manifest["split"]:
        raise SystemExit(f"{args.gru_run} was not trained on {manifest['name']}'s split")
    audit = {r["session_id"]: r for r in json.loads(AUDIT.read_text(encoding="utf-8"))["sessions"]}

    sessions = {}
    for part, samples in splits.items():
        by = defaultdict(list)
        for s in samples:
            by[s["session"]].append(s)
        ids = sorted(by)[: args.max_sessions or None]
        sessions[part] = {sid: by[sid] for sid in ids}
    all_ids = [sid for part in sessions.values() for sid in part]
    with ThreadPoolExecutor(max_workers=16) as pool:
        poses = dict(zip(all_ids, pool.map(session_poses, all_ids)))

    def arms(threshold):
        return {
            "nearest_in_cone": lambda fr, ps: run_v2(fr, ps, NearestInConePredictor(), threshold),
            "v2_physics": lambda fr, ps: run_v2(fr, ps, PhysicsArrivalPredictor(), threshold),
            "v2_physics_nopose": lambda fr, ps: run_v2(fr, None, PhysicsArrivalPredictor(), threshold),
            "v2_gru": lambda fr, ps: run_v2(fr, ps, LegacyGRU(args.gru_run / "model.pt"), threshold),
            "v2_gru_nopose": lambda fr, ps: run_v2(fr, None, LegacyGRU(args.gru_run / "model.pt"), threshold),
        }

    def evaluate(part: str, arm: str, threshold: float | None) -> dict:
        per = []
        for sid, samples in sessions[part].items():
            a = audit[sid]
            frames = session_rows(samples, a["fx"], a["image_width"])
            if arm == "v1_k0":
                log = run_v1(frames, announcer=False)
            elif arm == "v1_k0_announcer":
                log = run_v1(frames, announcer=True)
            else:
                log = arms(threshold)[arm](frames, poses[sid])
            per.append(session_metrics(frames, log))
        return aggregate(per)

    started = time.time()
    chosen, val_table = {}, {}
    for arm in arms(0.0):
        val_table[arm] = {th: evaluate("val", arm, th) for th in THRESHOLDS}
        chosen[arm] = max(THRESHOLDS, key=lambda th: f1(val_table[arm][th]))
        print(f"val  {arm:18s} threshold {chosen[arm]} (F1 {f1(val_table[arm][chosen[arm]]):.3f})", flush=True)

    test = {"v1_k0": evaluate("test", "v1_k0", None), "v1_k0_announcer": evaluate("test", "v1_k0_announcer", None)}
    for arm, th in chosen.items():
        test[arm] = evaluate("test", arm, th)
        print(f"test {arm:18s} done", flush=True)

    def cell(v):
        pt, lo, hi = v
        return f"{pt:.2f} [{lo:.2f}, {hi:.2f}]" if np.isfinite(pt) else "n/a"

    out = args.out_dir / f"official_{time.strftime('%Y_%m_%d')}"
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Threat-engine replay - SANPO official test split", "",
        f"{len(sessions['test'])} official test sessions with tracked objects, "
        f"{test['v1_k0']['arrivals']} arrivals (a track first seen before it came within 3 m in the ±30° cone). "
        f"Thresholds chosen on {len(sessions['val'])} validation sessions. GRU: `{args.gru_run.as_posix()}`. "
        f"Generated {time.strftime('%Y-%m-%d')} by `python tools/replay_threat_engine.py` "
        f"({time.time() - started:.0f} s).", "",
        "| Arm | ignore threshold | warned ↑ | precision ↑ | median lead (s) ↑ | alerts/min | overrides/min | switches/min ↓ |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    for arm, m in test.items():
        lines.append(f"| {arm} | {chosen.get(arm, '—')} | {cell(m['warned'])} | {cell(m['precision'])} | "
                     f"{cell(m['lead_s'])} | {cell(m['alerts_per_min'])} | {cell(m['overrides_per_min'])} | "
                     f"{cell(m['switches_per_min'])} |")
    lines += ["", "95% CIs resample whole sessions. warned: arrivals with an announcement about that track in the "
              "4 s before it arrived. precision: announcements whose object is within 3 m in the cone now or within "
              "the next 4 s (the arrival label at that frame).", "",
              "## Threshold selection (validation sessions, F1 of warned and precision)", "",
              "| Arm | " + " | ".join(str(t) for t in THRESHOLDS) + " |", "|---|" + "---:|" * len(THRESHOLDS)]
    for arm, by_th in val_table.items():
        lines.append(f"| {arm} | " + " | ".join(f"{f1(by_th[t]):.3f}" for t in THRESHOLDS) + " |")
    lines += ["", "Open-loop: the recorded walker never heard these alerts. Labels and inputs share SANPO's depth.",
              "Only YOLO-tracked objects are replayed (depth-sweep pseudo-objects have no identity to warn about).", ""]
    report = "\n".join(lines)
    (out / "report.md").write_text(report, encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"test": test, "chosen_thresholds": chosen,
                                                  "val": {a: {str(t): m for t, m in v.items()} for a, v in val_table.items()}},
                                                 indent=2), encoding="utf-8")
    print(report)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
