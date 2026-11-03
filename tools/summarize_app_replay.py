"""
summarize_app_replay.py
=======================
Report of the CPEWalk app's own pipeline run on SANPO walks (ios/CPEWalk/Tests/SanpoReplayTests.swift,
data from tools/prepare_app_replay.py): what it said, how often, and the LiDAR floor check.

    python tools/summarize_app_replay.py [--replay-dir data.nosync/app_replay] [--out evaluation/benchmarks/app_replay_sanpo]

Stage times come from the iOS Simulator (Core ML on the Mac's CPU), not the phone: they are listed only
as a sanity check, never as on-device latency.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {   # announcement (kind, class) -> group
    "override": "override (TTC ≤ 1 s)", "threat": "object threat", "surface": "surface",
}
EDGE = {"vehicle_left": "crossing", "vehicle_right": "crossing", "vehicles_left": "crossing",
        "vehicles_right": "crossing", "vehicles_both": "crossing", "bollard": "post", "pole": "post",
        "obstacle": "LiDAR obstacle", "stairs_up": "stairs", "stairs_down": "stairs"}
ORDER = ["override (TTC ≤ 1 s)", "object threat", "crossing", "post", "LiDAR obstacle", "curb / edge", "stairs",
         "surface"]


def group(r: dict) -> str:
    if r["kind"] == "edge":
        return EDGE.get(r.get("class", ""), "stairs" if "stair" in r.get("class", "") else "curb / edge")
    return GROUPS.get(r["kind"], r["kind"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replay-dir", type=Path, default=ROOT / "data.nosync/app_replay")
    p.add_argument("--out", type=Path, default=ROOT / "evaluation/benchmarks/app_replay_sanpo")
    a = p.parse_args()
    sessions = json.loads((a.replay_dir / "sessions.json").read_text())
    a.out.mkdir(parents=True, exist_ok=True)

    per, total, said_all, heights, frames_all = [], Counter(), [], [], 0
    timing = {k: [] for k in ("det1_ms", "det2_ms", "rules_ms", "engine_ms", "seg_ms")}
    for sid in sessions:
        f = a.replay_dir / sid / "app_results.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        said = [r for r in rows if "phrase" in r]
        c = Counter(group(r) for r in said)
        total += c
        minutes = len(rows) * 0.2 / 60
        h = [r["cam_height_m"] for r in rows if "cam_height_m" in r]
        heights += h
        frames_all += len(rows)
        for r in rows[10:]:                     # skip model warm-up
            for k in timing:
                if k != "seg_ms" or r[k] > 1:
                    timing[k].append(r[k])
        surf = Counter(r.get("surface") or "—" for r in rows)
        per.append({"session": sid, "frames": len(rows), "minutes": round(minutes, 2), "alerts": len(said),
                    "per_min": round(len(said) / minutes, 1) if minutes else 0, "groups": dict(c),
                    "floor_found": round(len(h) / len(rows), 2), "cam_height_median": round(st.median(h), 2) if h else None,
                    "main_surface": surf.most_common(1)[0][0],
                    "said": [{"t": r["t"], "kind": r["kind"], "class": r.get("class"), "phrase": r["phrase"]} for r in said]})
        said_all += said

    minutes = frames_all * 0.2 / 60
    (a.out / "sessions.json").write_text(json.dumps(per, indent=1, ensure_ascii=False))
    L = ["# The CPEWalk app on SANPO test walks", "",
         "The app's own Swift pipeline (`WalkPipeline.process`: both detectors, IoU tracker, street rules, LiDAR",
         "obstacle scan, threat engine, narration, segmentation after the alert), run in the iOS Simulator on",
         f"{len(per)} SANPO-Real test-split walks ({frames_all} frames at 5 Hz, {minutes:.1f} min).",
         "Data: `tools/prepare_app_replay.py`; run: `Tests/SanpoReplayTests.swift`; this report:",
         "`tools/summarize_app_replay.py`.", "",
         "Inputs as the phone gets them: half-resolution RGB, SANPO stereo depth resized to 256 px wide and cut",
         "at 5 m (the phone's LiDAR range), the replay's camera poses. SANPO is head-mounted, the phone is",
         "chest-held. SANPO has no labels for these alerts, so this shows what the app says, not whether it",
         "is right.", "",
         f"## What it said: {len(said_all)} alerts, {len(said_all) / minutes:.1f} per minute", "",
         "| Alert | Count | Per minute |", "|---|---:|---:|"]
    for g in ORDER + sorted(set(total) - set(ORDER)):
        if total[g]:
            L.append(f"| {g} | {total[g]} | {total[g] / minutes:.2f} |")
    L += ["", "## Per walk", "",
          "| Walk | Min | Alerts | /min | Main surface | Floor found | Cam height (m) | Mix |",
          "|---|---:|---:|---:|---|---:|---:|---|"]
    for s in per:
        mix = ", ".join(f"{g} {n}" for g, n in sorted(s["groups"].items(), key=lambda x: -x[1]))
        L.append(f"| `{s['session'][:10]}` | {s['minutes']} | {s['alerts']} | {s['per_min']} | {s['main_surface']} | "
                 f"{s['floor_found']:.0%} | {s['cam_height_median'] if s['cam_height_median'] is not None else '—'} | {mix} |")
    hm = st.median(heights) if heights else float("nan")
    L += ["", "## LiDAR floor check", "",
          f"The obstacle scan found the floor in {len(heights) / frames_all:.0%} of frames; median camera height",
          f"above it {hm:.2f} m (SANPO's head camera; the phone at chest height should read ~1.2–1.4 m).", "",
          "## Phrases (most common)", ""]
    for ph, n in Counter(r["phrase"] for r in said_all).most_common(15):
        L.append(f"- {n}× “{ph}”")
    L += ["", "## Stage times in the Simulator (Mac CPU, NOT the phone)", "",
          "| Stage | Median ms |", "|---|---:|"]
    for k, v in timing.items():
        if v:
            L.append(f"| {k.removesuffix('_ms')} | {st.median(v):.1f} |")
    L += ["", "On-device times are in `docs/progress.md` §9.2 (measured on the iPhone)."]
    (a.out / "README.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
