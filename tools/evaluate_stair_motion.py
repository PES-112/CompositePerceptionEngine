"""
Score src/threat_engine/stairs.StairMotion on SANPO: camera height from fixed_camera_poses.csv
(head camera, Z up, 15 FPS source, replayed at 5 Hz as the phone runs), against the human-annotated
segmentation frames of the same session (data/sanpo_walkable_seg, chest or head).

Ground truth is approximate: a frame counts as "stairs" when its walker patch
(context.walker_surface, the ground 2-3 m ahead) is mostly stairs. Labelled stair frames closer than
GAP_S merge into one episode.
- An episode is found if the detector is on stairs at any time in [start - LEAD_S, end + LAG_S].
- A false start is an UP/DOWN event with no stair label within FALSE_S, counted only in sessions
  that have labels (time covered by labels = from the first to the last labelled frame).
Parameters are the physics defaults in stairs.py, not tuned here; train/val/test are reported apart.

    .venv-seg.nosync/bin/python tools/evaluate_stair_motion.py

Writes evaluation/benchmarks/stairs/stair_motion/{metrics.json,report.md}.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.context import walker_surface
from src.segmentation.labels import STAIRS, read_sanpo_mask
from src.threat_engine.stairs import END, StairMotion

POSES = PROJECT_ROOT / "data/sanpo_poses_fixed"
SEG = PROJECT_ROOT / "data/sanpo_walkable_seg"
OUT = PROJECT_ROOT / "evaluation/benchmarks/stairs/stair_motion"
SOURCE_FPS, STRIDE = 15, 3
GAP_S, LEAD_S, LAG_S, FALSE_S = 3.0, 2.0, 3.0, 5.0


def labels() -> dict[str, tuple[str, dict[float, bool]]]:
    """{session: (split, {t: walker patch is stairs})} for sessions that also have poses."""
    have = {p.stem for p in POSES.glob("*.csv")}
    out: dict[str, tuple[str, dict[float, bool]]] = {}
    for split in ("train", "val", "test"):
        for m in sorted((SEG / "masks" / split).glob("*.png")):
            sid, _, k = m.stem.rsplit("_", 2)
            if sid not in have:
                continue
            out.setdefault(sid, (split, {}))[1][int(k) / SOURCE_FPS] = walker_surface(read_sanpo_mask(m)) == STAIRS
    return out


def run(sid: str) -> tuple[list[tuple[float, str, float]], list[tuple[float, float]], float]:
    """(events, on-stairs intervals, duration s) for one session at 5 Hz."""
    rows = list(csv.DictReader((POSES / f"{sid}.csv").open(encoding="utf-8")))
    det, events, intervals, start = StairMotion(), [], [], None
    t = 0.0
    for i in range(0, len(rows), STRIDE):
        r = rows[i]
        t = i / SOURCE_FPS
        if r["tracking_state"] != "TrackingState.READY":
            continue
        e = det.step(t, float(r["pos_x"]), float(r["pos_y"]), float(r["pos_z"]))
        if e is None:
            continue
        events.append((e.t, e.kind, e.rise_m))
        if e.kind == END and start is not None:
            intervals.append((start, e.t))
            start = None
        elif e.kind != END:
            start = e.t
    if start is not None:
        intervals.append((start, t))
    return events, intervals, t


def episodes(lab: dict[float, bool]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for t in sorted(t for t, s in lab.items() if s):
        if out and t - out[-1][1] <= GAP_S:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [tuple(e) for e in out]


def main() -> None:
    lab = labels()
    per_split = defaultdict(lambda: {"sessions": 0, "episodes": 0, "found": 0, "false_starts": 0,
                                     "labelled_min": 0.0, "events": 0})
    all_events, all_min, detail = 0, 0.0, []
    for p in sorted(POSES.glob("*.csv")):
        sid = p.stem
        events, intervals, dur = run(sid)
        starts = [e for e in events if e[1] != END]
        all_events += len(starts)
        all_min += dur / 60
        if sid not in lab:
            continue
        split, frames = lab[sid]
        eps = episodes(frames)
        s = per_split[split]
        s["sessions"] += 1
        covered = (max(frames) - min(frames)) / 60
        s["labelled_min"] += covered
        found = [any(a <= e1 + LAG_S and b >= e0 - LEAD_S for a, b in intervals) for e0, e1 in eps]
        s["episodes"] += len(eps)
        s["found"] += sum(found)
        stair_ts = [t for t, v in frames.items() if v]
        false = [e for e in starts if min(frames) <= e[0] <= max(frames)
                 and not any(abs(e[0] - t) <= FALSE_S for t in stair_ts)]
        s["false_starts"] += len(false)
        s["events"] += sum(min(frames) <= e[0] <= max(frames) for e in starts)
        if eps or false:
            detail.append({"session": sid, "split": split, "episodes": eps, "found": found,
                           "events": [(round(t, 1), k, round(r, 2)) for t, k, r in events],
                           "false_starts": [(round(t, 1), k) for t, k, _ in false]})
    result = {"params": "stairs.py defaults", "by_split": per_split,
              "all_sessions": {"sessions": len(list(POSES.glob('*.csv'))), "minutes": all_min,
                               "starts": all_events, "starts_per_min": all_events / max(all_min, 1e-9)},
              "detail": detail}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    lines = ["# Stairs from walker motion (stairs.StairMotion), SANPO", "",
             "| split | sessions with labels | labelled min | stair episodes | found | false starts | false / labelled min |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for split in ("train", "val", "test"):
        s = per_split[split]
        lines.append(f"| {split} | {s['sessions']} | {s['labelled_min']:.1f} | {s['episodes']} | {s['found']} | "
                     f"{s['false_starts']} | {s['false_starts'] / max(s['labelled_min'], 1e-9):.3f} |")
    a = result["all_sessions"]
    lines += ["", f"All {a['sessions']} pose sessions ({a['minutes']:.0f} min): {a['starts']} stair starts, "
              f"{a['starts_per_min']:.3f}/min (an upper bound on false starts: most walks have no stairs).", "",
              "Episodes and events:", ""]
    for d in detail:
        lines.append(f"- {d['split']} `{d['session']}`: episodes {[(round(a, 1), round(b, 1)) for a, b in d['episodes']]} "
                     f"found {d['found']}; events {d['events']}; false {d['false_starts']}")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
