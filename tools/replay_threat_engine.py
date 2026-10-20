"""
replay_threat_engine.py
=======================
End-to-end SANPO replay of the runtime engines: v1 (K0 routing -> stub ->
PhysicsVerification) against the v2 threat engine, on SANPO's official split,
as full operating curves rather than single points.

Input. The GRU arrival dataset stores, for every processed frame, every tracked
object's Stage-1 values (distance, bearing, closing speed, confidence, box
size) and its arrival label, so the replay rebuilds Stage-1 rows from it (no
CSVs needed). Camera poses come from each session's fixed_camera_poses.csv
(public bucket; cached under data/sanpo_poses_fixed/). Processed frame k is
source frame 3k (stride 3) and is replayed at t = 0.2·k s (15 FPS source).

Depth is SANPO's own depth, as recorded, for every engine and every label
(the demo replays pre-recorded video with depth; the product's depth sensor is
an external LiDAR not modelled here).

Units, corrected on the fly for everything except the GRU, which is fed the
units it was trained in:
  speed    stored speeds are 2x too high (Stage 1 assumed 30 FPS) -> x0.5
  bearing  stored bearings assumed a 70 deg linear FOV -> pinhole angle from
           the session's real fx (simulation/datasets/sanpo/session_audit.json)

Arms, each with one sensitivity knob swept densely:
  v1_k0               the v1 chain as built (speaks whenever the judge emits an
                      event); knob = K0's low routing threshold
  v1_k0_announcer     v1's pick through the v2 announcement manager; same knob
  nearest_in_corridor v2 engine, P(arrive) = exp(-distance / 3 m) for objects in
                      the walking corridor, 0 otherwise; knob = ignore threshold
  v2_physics          v2 engine, physics (constant-velocity) arrival predictor
  v2_physics_nopose   same, camera-frame tracking (ego-motion ablation)
  v2_kalman           v2 engine, Kalman arrival predictor: P(enter the corridor) over
                      the fused track's own uncertainty (src/threat_engine/kalman_arrival.py)
  v2_kalman_nopose    same, camera-frame tracking
  v2_gru              v2 engine, GRU arrival predictor
  v2_gru_nopose       same, camera-frame tracking
  v2_physics_ctx,     with --context-dir: the same arms with the scene-context multiplier
  v2_gru_ctx          (priority.SceneContext from tools/build_scene_context.py) on, run on
                      the sessions that have context; the context-off arms run on the same
                      sessions, and the report adds paired session-bootstrap differences

Every announcement is classed by the label of its object at that frame:
  true   the object enters the zone later within the horizon (tte >= 1)
  late   it is already inside the zone (tte == 0)
  false  it doesn't enter the zone within the horizon (tte == -1)
An arrival is a track's first frame inside the zone, counted if the track was
seen at least one frame earlier (a warning was possible). warned = an
announcement about it in the 4 s up to and including arrival.

Operating points are chosen on the official validation sessions, then scored
once on test with session-bootstrap 95% CIs:
  budget_2 / budget_5 / budget_10   most arrivals warned with <= 2 / 5 / 10 false alerts per minute
  early_budget_2 / _5 / _10         most arrivals warned >= 0.2 s *before* arrival (warned_before) within
                                    the same budgets; `warned` also counts announcements on the arrival
                                    frame itself, which the budget_* rules reward
  recall_50                         fewest false alerts with >= 50% of arrivals warned

--announcer-sweep ARM ...: for those v2 arms the knob is (ignore threshold,
announcer cooldown_s, switch_margin, stale_s) over ANNOUNCER_GRID instead of
the threshold alone, so the announcer is tuned on validation too.
Lead time is reported against the opportunity window (first sighting ->
arrival): warned rate and lead among arrivals with a window of >= 1 s and
>= 2 s, lead / window, and a Kaplan-Meier "not yet warned" curve over time
since first sighting (an arrival never warned is censored at arrival).

    python tools/replay_threat_engine.py --gru-run models/GRU/GRU/runs/corridor_h8_seed0
    python tools/replay_threat_engine.py --gru-run ... --context-dir data/sanpo_scene_context_v2/gt --name oracle_context

Latency-aware replay (--latency-csv, or a fixed --latency-ms): each
announcement is heard that long after the frame that triggered it, using the
per-frame totals the CPE Timing app measured on the phone (ios/CPETiming).
Warned / lead time use the heard time; precision is still judged at the frame.

    python tools/replay_threat_engine.py --gru-run ... --latency-csv cpe_timing_sustained30min_*.csv
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
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.GRU.samples import legacy_units, load_dataset
from src.narration.pipeline import NarrationPipeline
from src.narration.tts import CachedClipTTS
from src.pipeline.orchestrator import process_frame
from src.threat_engine import envelope
from src.threat_engine.announcer import AnnouncementManager
from src.threat_engine.engine import ThreatEngine
from src.threat_engine.fusion import EgoPose, FusedTrack, GroundTracker, sanpo_session_poses
from src.threat_engine.kalman_arrival import KalmanArrivalPredictor
from src.threat_engine.priority import (SEVERITY_GAMMA, Arrival, GRUArrivalPredictor, PhysicsArrivalPredictor,
                                        RankedThreat, SceneContext)
from src.threat_prioritizer import build_threat_frame
from tools.audit_sanpo_sessions import fetch, object_url
from tools.download_sanpo_valid_streams import BASE_PREFIX

DEFAULT_DATASET = PROJECT_ROOT / "dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_official_corridor_w0.75_d3_h20"
AUDIT = PROJECT_ROOT / "simulation/datasets/sanpo/session_audit.json"
POSE_CACHE = PROJECT_ROOT / "data/sanpo_poses_fixed"
STEP_S = 0.2
STRIDE = 3
LEGACY_HALF_FOV = 35.0
SPEED_FIX = 0.5
WARN_WINDOW_S = 4.0
V2_THRESHOLDS = (0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
# --announcer-sweep: thresholds x (cooldown_s, switch_margin, stale_s); announcer.py's defaults are (4.0, 1.25, 0.6)
SWEEP_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
ANNOUNCER_GRID = tuple((c, m, st) for c in (2.0, 4.0, 8.0) for m in (1.0, 1.25, 2.0) for st in (0.6, 1.2))
V1_LOW_K = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0)
BUDGETS = (2.0, 5.0, 10.0)
KM_TIMES = (0.5, 1.0, 2.0, 3.0, 4.0)
N_BOOT = 1000


# ── Inputs ───────────────────────────────────────────────────────────────────

def session_poses(sid: str) -> list[EgoPose]:
    POSE_CACHE.mkdir(parents=True, exist_ok=True)
    path = POSE_CACHE / f"{sid}.csv"
    if not path.exists():
        path.write_bytes(fetch(object_url(f"{BASE_PREFIX}/{sid}/camera_head/fixed_camera_poses.csv")))
    return sanpo_session_poses(list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))))


def session_rows(samples: list[dict], fx: float, width: float, legacy: bool = True) -> dict[int, list[dict]]:
    """
    {frame_idx: Stage-1 rows} in correct units, with the stored values (what the GRU was trained on)
    kept alongside. legacy: the dataset stores pre-2026-10-14 units (dataset.GRU.samples.legacy_units).
    """
    frames = {}
    for s in samples:
        rows = []
        for j, tid in enumerate(s["track_ids"]):
            last = s["x"][j, -1]
            legacy_bearing = float(s["bearing"][j])
            legacy_speed = float(last[2]) * 5
            if legacy:
                bearing = math.degrees(math.atan2(legacy_bearing / LEGACY_HALF_FOV * width / 2, fx))
                speed = legacy_speed * SPEED_FIX
            else:
                bearing, speed = legacy_bearing, legacy_speed
            rows.append({
                "frame_idx": s["frame_idx"], "track_id": tid, "class": s["classes"][j],
                "confidence": float(last[6]), "distance_m": float(s["distance"][j]),
                "bearing_deg": bearing,
                "velocity_ms": speed,
                "bbox_area_px": float(np.expm1(last[7] * 15)),
                "_legacy": {"bearing_deg": legacy_bearing, "velocity_ms": legacy_speed},
                "_tte": int(s["tte"][j]),
            })
        frames[s["frame_idx"]] = rows
    return frames


def session_contexts(path: Path) -> dict[int, dict[str, SceneContext]]:
    """{frame_idx: {track_id: SceneContext}} from a tools/build_scene_context.py file."""
    frames = json.loads(path.read_text(encoding="utf-8"))["frames"]
    return {int(k): {tid: SceneContext(*map(bool, v[1:])) for tid, v in f["tracks"].items()}
            for k, f in frames.items()}


def load_latencies(paths: list[Path]) -> np.ndarray:
    """total_ms of every non-warm-up frame in CPE Timing app CSVs (ios/CPETiming/analyze_timing.py format)."""
    vals = []
    for path in paths:
        rows = csv.DictReader(line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#"))
        vals += [float(r["total_ms"]) for r in rows if r["block"] != "warmup"]
    if not vals:
        raise SystemExit("no timing rows in --latency-csv")
    return np.array(vals)


def legacy_row(row: dict) -> dict:
    """The row in the units the GRU was trained on."""
    return {**row, **row["_legacy"]}


# ── Arms ─────────────────────────────────────────────────────────────────────

class NearestInCorridorPredictor:
    """P(arrive) = exp(-distance / 3 m) for objects in the walking corridor ahead; 0 elsewhere."""

    def predict(self, frame_idx, tracks):
        return {t.track_id: Arrival(math.exp(-t.distance_m / 3.0) if envelope.in_corridor(
            t.lateral_m, t.forward_m, t.class_name) else 0.0, t.distance_m) for t in tracks}


class LegacyGRU(GRUArrivalPredictor):
    def observe(self, frame_idx, tracks):
        swapped = [t.__class__(**{**t.__dict__, "row": legacy_row(t.row)}) if t.row else t for t in tracks]
        super().observe(frame_idx, swapped)


class CachedPredictor:
    """Replays the arrivals an inner predictor gave on the first pass (the tracker is deterministic)."""

    def __init__(self, inner, cache: dict):
        self.inner, self.cache = inner, cache

    def predict(self, frame_idx, tracks):
        if frame_idx not in self.cache:
            self.cache[frame_idx] = self.inner.predict(frame_idx, tracks)
        return self.cache[frame_idx]


def run_v2(frames: dict[int, list[dict]], poses: list[EgoPose] | None, predictor, threshold: float,
           announcer: tuple[float, float, float] | None = None,
           contexts: dict[int, dict[str, SceneContext]] | None = None, gamma: float = SEVERITY_GAMMA):
    """announcer: (cooldown_s, switch_margin, stale_s), or None for announcer.py's defaults."""
    mgr = AnnouncementManager() if announcer is None else AnnouncementManager(
        cooldown_s=announcer[0], switch_margin=announcer[1], stale_s=announcer[2])
    engine = ThreatEngine(predictor=predictor, ignore_threshold=threshold, tracker=GroundTracker(), announcer=mgr,
                          gamma=gamma)
    log = []
    for k in sorted(frames):
        pose = poses[min(STRIDE * k, len(poses) - 1)] if poses else None
        f = engine.step(k, k * STEP_S, frames[k], pose, contexts=(contexts or {}).get(k))
        if f.announcement:
            log.append((k, f.announcement.track_id, f.announcement.kind))
    return log


def run_v1(frames: dict[int, list[dict]], announcer: bool, low_k: float):
    narration = NarrationPipeline(tts=CachedClipTTS(fallback=b""), critical_tts=CachedClipTTS(fallback=b""))
    mgr = AnnouncementManager() if announcer else None
    high_k = max(5.0, 2 * low_k)
    log = []
    for k in sorted(frames):
        rows = [{**r, "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 0, "bbox_y2": 0, "cx_px": 0, "source": "sanpo"}
                for r in frames[k]]
        result = process_frame(build_threat_frame(rows, low_k_threshold=low_k, high_k_threshold=high_k), narration)
        ev = result.narrator_event
        if ev is None:
            continue
        if mgr is None:
            log.append((k, ev.track_id, "override" if ev.is_override else "threat"))
            continue
        dummy = _ranked_stub(ev.track_id, ev.object_class)   # the announcer only needs the id and kind
        overrides = [envelope.Override(dummy.track, 0.0, ev.reason)] if ev.is_override else []
        a = mgr.step(k * STEP_S, overrides, [] if ev.is_override else [dummy])
        if a:
            log.append((k, a.track_id, a.kind))
    return log


def _ranked_stub(track_id: str, class_name: str) -> RankedThreat:
    tr = FusedTrack(track_id, class_name, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, (0.0, 0.0), 0.0, 0.0, 1, 0, True, True)
    return RankedThreat(tr, 1.0, Arrival(1.0, 0.0), 1.0, 1.0)


# ── Metrics ──────────────────────────────────────────────────────────────────

def arrivals(frames: dict[int, list[dict]]) -> dict[str, tuple[int, int]]:
    """{track: (first seen frame, arrival frame)} for tracks first seen before they arrive."""
    first_seen, arrive = {}, {}
    for k in sorted(frames):
        for r in frames[k]:
            first_seen.setdefault(r["track_id"], k)
            if r["_tte"] == 0 and r["track_id"] not in arrive:
                arrive[r["track_id"]] = k
    return {tid: (first_seen[tid], k) for tid, k in arrive.items() if first_seen[tid] < k}


def session_metrics(frames, log, horizon_frames: int, delays: np.ndarray | None = None) -> dict:
    """
    delays: seconds added to each announcement in `log` (on-device latency, --latency-*); an
    announcement decided at frame k is heard at k·STEP_S + delay. Labels are judged at frame k.
    """
    arr = arrivals(frames)
    label = {(k, r["track_id"]): r["_tte"] for k, rows in frames.items() for r in rows}
    heard = [k * STEP_S + (float(delays[i]) if delays is not None else 0.0) for i, (k, _, _) in enumerate(log)]
    per_arrival = []
    for tid, (seen, ka) in arr.items():
        ta = ka * STEP_S
        times = [t for t, (_, a, _) in zip(heard, log) if a == tid and ta - WARN_WINDOW_S - 1e-9 <= t <= ta + 1e-9]
        warn = min(times) if times else None
        per_arrival.append({"window_s": (ka - seen) * STEP_S,
                            "lead_s": ta - warn if warn is not None else None,
                            # time from first sighting to first warning (Kaplan-Meier event), or censored at arrival
                            "time_to_warn_s": (warn - seen * STEP_S) if warn is not None else (ka - seen) * STEP_S,
                            "event": warn is not None})
    kinds = {"true": 0, "late": 0, "false": 0}
    for k, a, _ in log:
        tte = label.get((k, a), -1)
        kinds["true" if tte >= 1 else "late" if tte == 0 else "false"] += 1
    switches = sum(1 for (_, a, _), (_, b, _) in zip(log, log[1:]) if a != b)
    return {"arrivals": per_arrival, "announcements": len(log), **kinds,
            "overrides": sum(1 for *_, kind in log if kind == "override"),
            "switches": switches, "minutes": max(1, len(frames)) * STEP_S / 60}


KEYS = ("warned", "precision", "late_share", "false_per_min", "alerts_per_min", "switches_per_min",
        "overrides_per_min", "median_lead_s", "warned_window_ge1", "warned_window_ge2", "lead_ge1_s",
        "lead_over_window", "warned_before", "warned_lead_ge1")
# warned counts an announcement on the arrival frame itself (lead 0). warned_before needs at least
# one processed step (0.2 s) of lead and warned_lead_ge1 at least 1 s.


def point(sessions: list[dict]) -> np.ndarray:
    arr = [a for s in sessions for a in s["arrivals"]]
    ann = sum(s["announcements"] for s in sessions)
    mins = sum(s["minutes"] for s in sessions)
    warned = [a for a in arr if a["event"]]
    ge1 = [a for a in arr if a["window_s"] >= 1.0]
    ge2 = [a for a in arr if a["window_s"] >= 2.0]

    def rate(xs):
        return sum(a["event"] for a in xs) / len(xs) if xs else np.nan

    leads_ge1 = [a["lead_s"] for a in ge1 if a["event"]]
    return np.array([
        rate(arr),
        sum(s["true"] + s["late"] for s in sessions) / ann if ann else np.nan,
        sum(s["late"] for s in sessions) / ann if ann else np.nan,
        sum(s["false"] for s in sessions) / mins,
        ann / mins,
        sum(s["switches"] for s in sessions) / mins,
        sum(s["overrides"] for s in sessions) / mins,
        float(np.median([a["lead_s"] for a in warned])) if warned else np.nan,
        rate(ge1), rate(ge2),
        float(np.median(leads_ge1)) if leads_ge1 else np.nan,
        float(np.median([a["lead_s"] / a["window_s"] for a in warned])) if warned else np.nan,
        sum(a["event"] and a["lead_s"] >= STEP_S - 1e-9 for a in arr) / len(arr) if arr else np.nan,
        sum(a["event"] and a["lead_s"] >= 1.0 - 1e-9 for a in arr) / len(arr) if arr else np.nan,
    ])


def aggregate(per_session: list[dict], boot: bool = True, seed: int = 0) -> dict:
    full = point(per_session)
    out = {"arrivals": sum(len(s["arrivals"]) for s in per_session)}
    if not boot:
        out.update({k: (float(full[i]),) for i, k in enumerate(KEYS)})
        return out
    rng = np.random.default_rng(seed)
    n = len(per_session)
    draws = np.array([point([per_session[j] for j in rng.integers(0, n, n)]) for _ in range(N_BOOT)])
    out.update({k: (float(full[i]), float(np.nanpercentile(draws[:, i], 2.5)), float(np.nanpercentile(draws[:, i], 97.5)))
                for i, k in enumerate(KEYS)})
    return out


def paired_difference(a: list[dict], b: list[dict], seed: int = 0) -> dict:
    """a - b on the same sessions, with a paired session-bootstrap 95% CI."""
    diff = point(a) - point(b)
    rng = np.random.default_rng(seed)
    n = len(a)
    draws = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        draws.append(point([a[j] for j in idx]) - point([b[j] for j in idx]))
    draws = np.array(draws)
    return {k: (float(diff[i]), float(np.nanpercentile(draws[:, i], 2.5)), float(np.nanpercentile(draws[:, i], 97.5)))
            for i, k in enumerate(KEYS)}


def kaplan_meier(per_session: list[dict]) -> dict[float, float]:
    """S(t): share of arrivals not yet warned t seconds after first sighting (never-warned censored at arrival)."""
    rows = sorted((a["time_to_warn_s"], a["event"]) for s in per_session for a in s["arrivals"])
    surv, at_risk, out, i = 1.0, len(rows), {}, 0
    for t_query in KM_TIMES:
        while i < len(rows) and rows[i][0] <= t_query:
            t = rows[i][0]
            events = removed = 0
            while i < len(rows) and rows[i][0] == t:
                events += rows[i][1]
                removed += 1
                i += 1
            if at_risk:
                surv *= 1 - events / at_risk
            at_risk -= removed
        out[t_query] = surv
    return out


def window_profile(per_session: list[dict]) -> dict:
    w = np.array([a["window_s"] for s in per_session for a in s["arrivals"]])
    return {"arrivals": int(len(w)), "median_window_s": float(np.median(w)),
            "share_lt_1s": float(np.mean(w < 1.0)), "share_lt_2s": float(np.mean(w < 2.0)),
            "share_ge_4s": float(np.mean(w >= 4.0))}


def pick(curve: dict, rule: str):
    """Knob value for an operating rule from a {knob: aggregate} validation curve."""
    items = list(curve.items())
    if "budget_" in rule:
        budget = float(rule.rsplit("_", 1)[1])
        target = "warned_before" if rule.startswith("early_") else "warned"
        ok = [(k, m) for k, m in items if m["false_per_min"][0] <= budget]
        if not ok:
            return None
        return max(ok, key=lambda km: (np.nan_to_num(km[1][target][0]), -km[1]["false_per_min"][0]))[0]
    if rule == "recall_50":
        ok = [(k, m) for k, m in items if np.nan_to_num(m["warned"][0]) >= 0.5]
        if not ok:
            return None
        return min(ok, key=lambda km: km[1]["false_per_min"][0])[0]
    raise ValueError(rule)


def knob_str(k) -> str:
    """A knob for tables and the CSV: a number, or threshold / cooldown / switch margin / stale."""
    if isinstance(k, tuple):
        return f"{k[0]:g} / cd {k[1]:g} / sm {k[2]:g} / st {k[3]:g}"
    return f"{k:g}"


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.disable(logging.CRITICAL)   # PhysicsVerification logs every v1 override and stub disagreement
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--gru-run", type=Path, required=True, help="A models/GRU run trained on --dataset's train split.")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "evaluation/benchmarks/threat_engine_replay")
    p.add_argument("--name", default=None, help="Output folder name (default: <dataset label>_<date>).")
    p.add_argument("--max-sessions", type=int, default=0, help="Per split; 0 = all (smoke testing).")
    p.add_argument("--context-dir", type=Path, default=None,
                   help="tools/build_scene_context.py output; adds context-on arms, restricted to its sessions.")
    p.add_argument("--latency-ms", type=float, default=0.0,
                   help="Add this fixed on-device latency to every announcement (ms).")
    p.add_argument("--latency-csv", type=Path, nargs="*", default=[],
                   help="CPE Timing app CSVs (ios/CPETiming): each announcement is delayed by a total_ms drawn "
                        "from them (warm-up excluded), seeded per session.")
    p.add_argument("--announcer-sweep", nargs="*", default=[], metavar="ARM",
                   help="v2 arms whose announcer (cooldown, switch margin, stale) is swept with the threshold.")
    p.add_argument("--severity-gamma", type=float, default=SEVERITY_GAMMA,
                   help="Severity exponent γ in the v2 priority (0 = rank by arrival alone; ablation only).")
    args = p.parse_args()

    manifest, splits = load_dataset(args.dataset, splits=("val", "test"))
    cfg = json.loads((args.gru_run / "config.json").read_text(encoding="utf-8"))
    if cfg["split"] != manifest["split"] or cfg["dataset_name"] != manifest["name"]:
        raise SystemExit(f"{args.gru_run} was not trained on {manifest['name']}")
    horizon = int(manifest["label"]["HORIZON_FRAMES"])
    legacy = legacy_units(manifest)
    audit = {r["session_id"]: r for r in json.loads(AUDIT.read_text(encoding="utf-8"))["sessions"]}

    sessions = {}
    for part, samples in splits.items():
        by = defaultdict(list)
        for s in samples:
            by[s["session"]].append(s)
        ids = sorted(sid for sid in by if args.context_dir is None or (args.context_dir / f"{sid}.json").exists())
        ids = ids[: args.max_sessions or None]
        sessions[part] = {sid: session_rows(by[sid], audit[sid]["fx"], audit[sid]["image_width"], legacy)
                          for sid in ids}
    all_ids = [sid for part in sessions.values() for sid in part]
    contexts = {sid: session_contexts(args.context_dir / f"{sid}.json") for sid in all_ids} if args.context_dir else {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        poses = dict(zip(all_ids, pool.map(session_poses, all_ids)))

    latency_pool = load_latencies(args.latency_csv) if args.latency_csv else None

    def delays_for(sid: str, n: int) -> np.ndarray | None:
        if latency_pool is not None:
            return np.random.default_rng(zlib.crc32(sid.encode())).choice(latency_pool, n) / 1000.0
        return np.full(n, args.latency_ms / 1000.0) if args.latency_ms else None

    caches: dict[tuple, dict] = defaultdict(dict)
    gru_path = args.gru_run / "model.pt"
    v2_arms = {
        "nearest_in_corridor": (True, lambda: NearestInCorridorPredictor()),
        "v2_physics": (True, lambda: PhysicsArrivalPredictor()),
        "v2_physics_nopose": (False, lambda: PhysicsArrivalPredictor()),
        "v2_kalman": (True, lambda: KalmanArrivalPredictor()),
        "v2_kalman_nopose": (False, lambda: KalmanArrivalPredictor()),
        "v2_gru": (True, lambda: LegacyGRU(gru_path)),
        "v2_gru_nopose": (False, lambda: LegacyGRU(gru_path)),
    }
    context_arms = {"v2_physics_ctx": "v2_physics", "v2_gru_ctx": "v2_gru"} if args.context_dir else {}
    for arm, base in context_arms.items():
        v2_arms[arm] = v2_arms[base]
    unknown = set(args.announcer_sweep) - set(v2_arms)
    if unknown:
        raise SystemExit(f"--announcer-sweep: not v2 arms: {sorted(unknown)}")
    sweep = tuple((t, *a) for t in SWEEP_THRESHOLDS for a in ANNOUNCER_GRID)
    arms = {"v1_k0": V1_LOW_K, "v1_k0_announcer": V1_LOW_K,
            **{a: sweep if a in args.announcer_sweep else V2_THRESHOLDS for a in v2_arms}}

    def run(part: str, arm: str, knob: float) -> list[dict]:
        per = []
        for sid, frames in sessions[part].items():
            if arm.startswith("v1"):
                log = run_v1(frames, announcer=arm.endswith("announcer"), low_k=knob)
            else:
                use_pose, make = v2_arms[arm]
                base = context_arms.get(arm, arm)    # context arms reuse their base arm's arrival cache
                predictor = CachedPredictor(make(), caches[(base, sid)])
                thr, ann = (knob[0], knob[1:]) if isinstance(knob, tuple) else (knob, None)
                log = run_v2(frames, poses[sid] if use_pose else None, predictor, thr, ann,
                             contexts[sid] if arm in context_arms else None, args.severity_gamma)
            per.append(session_metrics(frames, log, horizon, delays_for(sid, len(log))))
        return per

    started = time.time()
    rules = [f"budget_{b:g}" for b in BUDGETS] + [f"early_budget_{b:g}" for b in BUDGETS] + ["recall_50"]
    curves = {"val": {}, "test": {}}
    test_runs: dict[str, dict[float, list[dict]]] = {}
    chosen: dict[str, dict[str, float | None]] = {}
    for arm, knobs in arms.items():
        curves["val"][arm] = {k: aggregate(run("val", arm, k), boot=False) for k in knobs}
        test_runs[arm] = {k: run("test", arm, k) for k in knobs}
        curves["test"][arm] = {k: aggregate(v, boot=False) for k, v in test_runs[arm].items()}
        chosen[arm] = {r: pick(curves["val"][arm], r) for r in rules}
        print(f"{arm:20s} {time.time() - started:5.0f} s  chosen on val: "
              f"{ {r: (knob_str(k) if k is not None else None) for r, k in chosen[arm].items()} }", flush=True)

    points = {arm: {r: (aggregate(test_runs[arm][k]) if k is not None else None) for r, k in by_rule.items()}
              for arm, by_rule in chosen.items()}
    km = {arm: {r: (kaplan_meier(test_runs[arm][k]) if k is not None else None) for r, k in by_rule.items()}
          for arm, by_rule in chosen.items()}
    windows = window_profile(next(iter(next(iter(test_runs.values())).values())))

    def cell(v, fmt="{:.2f}"):
        if v is None or not np.isfinite(v[0]):
            return "n/a"
        return (fmt.format(v[0]) + (f" [{fmt.format(v[1])}, {fmt.format(v[2])}]" if len(v) == 3 else ""))

    stats = manifest["stats"]
    name = args.name or f"{manifest['label'].get('kind', 'cone')}_{time.strftime('%Y_%m_%d')}"
    if latency_pool is not None or args.latency_ms:
        name += "_latency"
    out = args.out_dir / name
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Threat-engine replay - {name}", "",
        f"Label: {manifest['label']['definition']}.", "",
        f"SANPO official test split: {len(sessions['test'])} sessions, {windows['arrivals']} arrivals (a track "
        f"first seen before it entered the zone). Operating points chosen on {len(sessions['val'])} validation "
        f"sessions, then scored once on test. GRU: `{args.gru_run.as_posix()}`. Generated "
        f"{time.strftime('%Y-%m-%d')} by `python tools/replay_threat_engine.py` ({time.time() - started:.0f} s).", "",
    ]
    if latency_pool is not None:
        lines += [f"Latency: each announcement is heard after an on-device total drawn from {len(latency_pool)} "
                  f"measured frames ({', '.join(p.name for p in args.latency_csv)}; median "
                  f"{np.median(latency_pool):.1f} ms, p95 {np.percentile(latency_pool, 95):.1f} ms).", ""]
    elif args.latency_ms:
        lines += [f"Latency: every announcement is heard {args.latency_ms:g} ms after the frame that triggered it.", ""]
    lines += [
        "## Opportunity window (first sighting -> arrival), test", "",
        f"Median {windows['median_window_s']:.1f} s; {windows['share_lt_1s']:.0%} of arrivals were first seen less "
        f"than 1 s before arriving, {windows['share_lt_2s']:.0%} less than 2 s, {windows['share_ge_4s']:.0%} at least "
        "4 s before. Lead time can't exceed the window, so lead is reported conditionally below.", "",
    ]
    for rule in rules:
        lines += [f"## Operating point `{rule}` (test, 95% session-bootstrap CI)", "",
                  "| Arm | knob | warned ↑ | false alerts/min ↓ | precision ↑ | late share | alerts/min | switches/min ↓ "
                  "| warned, window ≥ 1 s | warned, window ≥ 2 s | median lead, window ≥ 1 s (s) "
                  "| warned ≥ 0.2 s before | warned ≥ 1 s before |",
                  "|---|---:|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm in arms:
            m = points[arm][rule]
            if m is None:
                lines.append(f"| {arm} | — | no setting meets the rule on validation |||||||||||")
                continue
            lines.append(f"| {arm} | {knob_str(chosen[arm][rule])} | {cell(m['warned'])} | {cell(m['false_per_min'])} | "
                         f"{cell(m['precision'])} | {cell(m['late_share'])} | {cell(m['alerts_per_min'])} | "
                         f"{cell(m['switches_per_min'])} | {cell(m['warned_window_ge1'])} | "
                         f"{cell(m['warned_window_ge2'])} | {cell(m['lead_ge1_s'])} | "
                         f"{cell(m['warned_before'])} | {cell(m['warned_lead_ge1'])} |")
        lines.append("")
    for km_rule in ("budget_5", "early_budget_5"):
        lines += [f"## Kaplan-Meier: share of arrivals not yet warned, by time since first sighting (test, "
                  f"`{km_rule}`)", "",
                  "| Arm | " + " | ".join(f"{t:g} s" for t in KM_TIMES) + " |", "|---|" + "---:|" * len(KM_TIMES)]
        for arm in arms:
            k = km[arm][km_rule]
            lines.append(f"| {arm} | " + (" | ".join(f"{k[t]:.2f}" for t in KM_TIMES) if k else "n/a") + " |")
        lines.append("")
    paired = {}
    for arm, base in context_arms.items():
        for rule in rules:
            ka, kb = chosen[arm][rule], chosen[base][rule]
            if ka is not None and kb is not None:
                paired[(arm, rule)] = paired_difference(test_runs[arm][ka], test_runs[base][kb])
    if context_arms:
        lines += ["## Context on minus context off (test, same sessions, paired session-bootstrap 95% CI)", "",
                  f"Context from `{args.context_dir.as_posix()}`. Each arm at its own validation-chosen knob.", "",
                  "| Arm | rule | Δ warned | Δ false alerts/min | Δ precision | Δ median lead, window ≥ 1 s (s) |",
                  "|---|---|---|---|---|---|"]
        for (arm, rule), d in paired.items():
            lines.append(f"| {arm} | {rule} | {cell(d['warned'])} | {cell(d['false_per_min'])} | "
                         f"{cell(d['precision'])} | {cell(d['lead_ge1_s'])} |")
        lines.append("")
    lines += ["", "## Operating curves (test; every knob value)", "",
              "Recall (warned) against false alerts per minute; the CSV next to this report has every column.", "",
              "| Arm | knob | warned | warned ≥ 0.2 s before | false/min | precision | alerts/min |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    rows_csv = ["split,arm,knob," + ",".join(KEYS)]
    for split in ("val", "test"):
        for arm, curve in curves[split].items():
            for k, m in curve.items():
                rows_csv.append(f"{split},{arm},{knob_str(k)}," + ",".join(f"{m[key][0]:.4f}" for key in KEYS))
                if split == "test" and not isinstance(k, tuple):   # swept arms: see curves.csv
                    lines.append(f"| {arm} | {k:g} | {m['warned'][0]:.2f} | {m['warned_before'][0]:.2f} | "
                                 f"{m['false_per_min'][0]:.1f} | {m['precision'][0]:.2f} | {m['alerts_per_min'][0]:.1f} |")
    lines += ["", "Classes: true = the object enters the zone later within the horizon; late = it is already in the "
              "zone; false = it doesn't enter within the horizon. precision = (true + late) / all. warned = an "
              "announcement about the arriving track in the 4 s up to and including arrival.",
              "Open-loop: the recorded walker never heard these alerts. Labels and inputs share SANPO's depth.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "curves.csv").write_text("\n".join(rows_csv) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"chosen": {a: {r: (knob_str(k) if k is not None else None)
                                                                  for r, k in by.items()} for a, by in chosen.items()},
                                                  "points": points, "kaplan_meier": km,
                                                  "window_profile": windows,
                                                  "paired_context": {f"{a}|{r}": d for (a, r), d in paired.items()}},
                                                 indent=2, default=str), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
