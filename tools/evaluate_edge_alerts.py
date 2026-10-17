"""
evaluate_edge_alerts.py
=======================
Does the "Curb ahead." alert (src/segmentation/edge_alert.py) fire before the
walker reaches unsafe ground, and how often does it fire for nothing?

Every 6th source frame (2.5 Hz, the low end of §6.3's segmentation rate; on
SANPO these are the HUMAN_ANNOTATED frames) of each session with head-camera
masks: class map (SANPO's own masks with --source gt, or a trained model) +
depth -> walker surface and edge ahead (src/segmentation/context.py). The
per-frame track is cached in data/sanpo_edge_track/<source>/; EdgeAlerter then
runs over it.

There is no annotated "stepped off the curb" event in SANPO, so the reference
is the walker-surface change the masks imply: a *step-off* is a frame where
the ground just ahead of the walker (bottom-centre patch) turns from walkable
(walkable, crosswalk) to an announced kind (curb, stairs, road), after at
least STEP_OFF_HOLD frames on walkable ground (--step-off-stay N: and stays off walkable ground for
N frames; 1 = declared, 2 = the sensitivity variant that drops one-frame mask flicker at crosswalk
edges, chosen after seeing the misses). Scores (session-bootstrap 95% CIs):
  alerts / min           every alert, by kind
  step-offs warned       share with an alert in the LOOKBACK_S before it; lead = step-off - alert
  alerts followed        share of alerts followed by a step-off within LOOKBACK_S; the rest
                         are edges the walker turned from, stopped at or walked beside:
                         not all of them are false, so this bounds precision from below
With --source gt the class map is SANPO's annotation, so this scores the rule,
not a model.

    python tools/evaluate_edge_alerts.py --source gt
    python tools/evaluate_edge_alerts.py --source training/runs/cpe_yolo26s_sem_walkable/weights/best.pt --device mps --imgsz 768
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from dataset.GRU.relabel import load_session_poses
from src.segmentation.context import EdgeAhead, edge_ahead, walker_surface, _depth_on
from src.segmentation.edge_alert import ALERT_KINDS, ALERT_M, CLEAR_S, CONFIRM_FRAMES, EdgeAlerter
from src.segmentation.labels import CPE_CLASSES, WALKABLE_CLASSES
from tools.build_scene_context import AUDIT, DOWNSAMPLE, class_map_source, depth_map

SOURCE_FPS = 15
FRAME_STEP = 6                  # 2.5 Hz
LOOKBACK_S = 10.0
STEP_OFF_HOLD = 2               # frames on walkable ground before a step-off counts
OUT_ROOT = PROJECT_ROOT / "data/sanpo_edge_track"
N_BOOT = 1000


def session_track(sid: str, get_class_map) -> list[list]:
    """[[source frame, walker surface, edge kind, edge m], ...] at FRAME_STEP."""
    rows = []
    for f in range(0, len(load_session_poses(sid)), FRAME_STEP):
        cls = get_class_map(sid, f)
        if cls is None:
            continue
        cls = cls[::DOWNSAMPLE, ::DOWNSAMPLE]
        depth = _depth_on(depth_map(sid, f), cls.shape)
        walker = walker_surface(cls)
        edge = edge_ahead(cls, depth, walker)
        rows.append([f, walker, None if edge is None else edge.kind,
                     None if edge is None else round(edge.distance_m, 2)])
    return rows


def stays_off(rows: list[list], i: int, n: int) -> bool:
    """The walker's ground is not walkable for n frames from row i (frames with no ground visible skipped)."""
    seen = [r[1] for r in rows[i:] if r[1] is not None][:n]
    return len(seen) == n and not any(w in WALKABLE_CLASSES for w in seen)


def score_session(rows: list[list], stay: int = 1) -> dict:
    alerter = EdgeAlerter()
    alerts, step_offs = [], []
    walkable_run = 0
    for i, (f, walker, kind, dist) in enumerate(rows):
        t = f / SOURCE_FPS
        a = alerter.step(t, None if kind is None else EdgeAhead(kind, dist))
        if a is not None:
            alerts.append(a)
        if walker in WALKABLE_CLASSES:
            walkable_run += 1
        elif walker in ALERT_KINDS:
            if walkable_run >= STEP_OFF_HOLD and stays_off(rows, i, stay):
                step_offs.append((t, CPE_CLASSES[walker]))
            walkable_run = 0
        # walker None (ground not visible): keep the run
    leads = []
    for t, _ in step_offs:
        before = [a.t for a in alerts if 0.0 <= t - a.t <= LOOKBACK_S]
        leads.append(t - max(before) if before else None)
    followed = [any(0.0 <= t - a.t <= LOOKBACK_S for t, _ in step_offs) for a in alerts]
    minutes = (rows[-1][0] - rows[0][0]) / SOURCE_FPS / 60 if len(rows) > 1 else 0.0
    return {"minutes": minutes, "frames": len(rows),
            "alerts": [[round(a.t, 2), a.kind, round(a.distance_m, 2)] for a in alerts],
            "followed": followed, "step_offs": [[round(t, 2), k] for t, k in step_offs], "leads": leads}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="gt", help="'gt' (SANPO masks) or a YOLO26-sem checkpoint.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--device", default=None)
    p.add_argument("--max-sessions", type=int, default=0)
    p.add_argument("--step-off-stay", type=int, default=1,
                   help="Frames the walker must stay off walkable ground for a step-off (1 = declared).")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    audit = {r["session_id"]: r for r in json.loads(AUDIT.read_text(encoding="utf-8"))["sessions"]}
    ids = sorted(sid for sid, r in audit.items() if r["segmentation_masks"])[: args.max_sessions or None]
    source = "gt" if args.source == "gt" else Path(args.source).parents[1].name
    cache = OUT_ROOT / source
    cache.mkdir(parents=True, exist_ok=True)
    get_class_map = class_map_source(args.source, args.imgsz, args.device)
    started = time.time()

    def one(sid: str) -> list[list]:
        path = cache / f"{sid}.json"
        if not path.exists():
            path.write_text(json.dumps(session_track(sid, get_class_map)), encoding="utf-8")
        return json.loads(path.read_text(encoding="utf-8"))

    tracks = {}
    with ThreadPoolExecutor(max_workers=args.workers if args.source == "gt" else 1) as pool:
        for i, (sid, rows) in enumerate(zip(ids, pool.map(one, ids)), 1):
            tracks[sid] = rows
            print(f"  {i}/{len(ids)} {sid}  {len(rows)} frames  {time.time() - started:.0f} s", flush=True)

    scored = {sid: score_session(rows, args.step_off_stay) for sid, rows in tracks.items() if len(rows) > 1}
    sessions = sorted(scored)
    rng = np.random.default_rng(0)
    draws = [rng.choice(sessions, len(sessions)) for _ in range(N_BOOT)]

    def stats(group: list[str]) -> dict:
        s = [scored[x] for x in group]
        minutes = sum(x["minutes"] for x in s)
        leads = [l for x in s for l in x["leads"]]
        followed = [f for x in s for f in x["followed"]]
        warned = [l for l in leads if l is not None]
        return {"minutes": minutes, "alerts": sum(len(x["alerts"]) for x in s),
                "alerts_per_min": sum(len(x["alerts"]) for x in s) / minutes if minutes else np.nan,
                "step_offs": len(leads), "warned": len(warned) / len(leads) if leads else np.nan,
                "followed": float(np.mean(followed)) if followed else np.nan,
                "median_lead_s": float(np.median(warned)) if warned else np.nan}

    def ci(key: str) -> list[float]:
        vals = np.array([stats(list(d))[key] for d in draws], dtype=float)
        return [stats(sessions)[key], float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]

    overall = stats(sessions)
    result = {"source": args.source, "sessions": len(sessions), "frame_step": FRAME_STEP,
              "rule": {"kinds": [CPE_CLASSES[k] for k in ALERT_KINDS], "alert_m": ALERT_M,
                       "confirm_frames": CONFIRM_FRAMES, "clear_s": CLEAR_S},
              "lookback_s": LOOKBACK_S, "step_off_hold": STEP_OFF_HOLD, "step_off_stay": args.step_off_stay,
              "overall": overall,
              "ci": {k: ci(k) for k in ("alerts_per_min", "warned", "followed", "median_lead_s")},
              "by_kind": {}, "by_official_split": {}, "sessions_detail": scored}
    for kind in (CPE_CLASSES[k] for k in ALERT_KINDS):
        alerts = [(a, f) for x in scored.values() for a, f in zip(x["alerts"], x["followed"]) if a[1] == kind]
        offs = [(o, l) for x in scored.values() for o, l in zip(x["step_offs"], x["leads"]) if o[1] == kind]
        result["by_kind"][kind] = {
            "alerts": len(alerts), "alerts_per_min": len(alerts) / overall["minutes"],
            "followed": float(np.mean([f for _, f in alerts])) if alerts else None,
            "median_alert_m": float(np.median([a[2] for a, _ in alerts])) if alerts else None,
            "step_offs": len(offs), "warned": float(np.mean([l is not None for _, l in offs])) if offs else None}
    for split in ("train", "test"):
        group = [s for s in sessions if audit[s]["split"] == split]
        if group:
            result["by_official_split"][split] = {"sessions": len(group), **stats(group)}

    name = args.name or f"{source}_{time.strftime('%Y_%m_%d')}"
    out = PROJECT_ROOT / "evaluation/benchmarks/edge_alerts" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")

    def cell(v, fmt="{:.2f}"):
        return f"{fmt.format(v[0])} [{fmt.format(v[1])}, {fmt.format(v[2])}]"

    c = result["ci"]
    lines = [f"# Edge-ahead alerts - {name}", "",
             f"{len(sessions)} sessions with head-camera masks, {overall['minutes']:.1f} min at 2.5 Hz; class maps from "
             f"`{args.source}`. Rule: announce {result['rule']['kinds']} within {ALERT_M:g} m, seen in "
             f"{CONFIRM_FRAMES} consecutive frames, one alert per edge episode ({CLEAR_S:g} s clear). Step-off: the "
             f"ground just ahead turns from walkable to an announced kind after {STEP_OFF_HOLD} walkable frames "
             f"and stays off walkable ground for {args.step_off_stay} frame(s). "
             "95% CIs: session bootstrap.", "",
             "| alerts / min | step-offs | warned (alert ≤ 10 s before) | median lead (s) | alerts followed by a step-off |",
             "|---|---:|---|---|---|",
             f"| {cell(c['alerts_per_min'])} | {overall['step_offs']} | {cell(c['warned'])} | "
             f"{cell(c['median_lead_s'], '{:.1f}')} | {cell(c['followed'])} |", "",
             "| Kind | alerts | alerts / min | followed | median distance at alert (m) | step-offs | warned |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in result["by_kind"].items():
        f = lambda x, fmt="{:.2f}": "—" if x is None else fmt.format(x)
        lines.append(f"| {k} | {v['alerts']} | {v['alerts_per_min']:.2f} | {f(v['followed'])} | "
                     f"{f(v['median_alert_m'], '{:.1f}')} | {v['step_offs']} | {f(v['warned'])} |")
    lines += ["", "| Official split | sessions | alerts / min | step-offs | warned | followed |", "|---|---:|---:|---:|---:|---:|"]
    for k, v in result["by_official_split"].items():
        lines.append(f"| {k} | {v['sessions']} | {v['alerts_per_min']:.2f} | {v['step_offs']} | {v['warned']:.2f} | {v['followed']:.2f} |")
    lines += ["", "An alert not followed by a step-off is not necessarily false: the walker may have stopped, turned "
              "or walked along the edge. Warned is per step-off; a step-off with no alert before it is a miss."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
