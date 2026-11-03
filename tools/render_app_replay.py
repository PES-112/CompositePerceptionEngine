"""
render_app_replay.py
====================
Video of one SANPO walk as the CPEWalk app saw it (Tests/SanpoReplayTests.swift results): the app's own
overlays (grey = tracked, amber = ranked threat, red = override; label = class and LiDAR distance), the
walker surface, and each spoken alert as a caption for 3 s. Real time, 5 frames a second.

Audio: each alert spoken by macOS `say` (Samantha, the iPhone's en-US voice) at the moment the app speaks
it, panned toward the object as the app does (constant power, full at ±60°), and cut off by the next
alert, as Speech.swift does.

    python tools/render_app_replay.py <session_id> [--replay-dir data.nosync/app_replay] [--out <file.mp4>]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
COLORS = {0: (170, 170, 170), 1: (0, 176, 255), 2: (40, 40, 230)}     # BGR: grey, amber, red
GROUP_COLOR = {"override": (40, 40, 230), "edge": (255, 140, 40), "threat": (0, 176, 255), "surface": (200, 200, 200)}


def text(img, s, org, scale=0.6, color=(255, 255, 255), thick=1, bg=(0, 0, 0)):
    (w, h), b = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 4, y - h - 6), (x + w + 4, y + b + 2), bg, -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session")
    p.add_argument("--replay-dir", type=Path, default=ROOT / "data.nosync/app_replay")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    d = a.replay_dir / a.session
    rows = [json.loads(l) for l in (d / "app_results.jsonl").read_text().splitlines() if l.strip()]
    out = a.out or d / f"{a.session}_app.mp4"
    first = cv2.imread(str(d / "frames/0000.jpg"))
    H, W = first.shape[:2]
    tmp = out.with_suffix(".tmp.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 5, (W, H))
    caption, until, ccol = "", -1.0, (255, 255, 255)
    for k, r in enumerate(rows):
        img = cv2.imread(str(d / f"frames/{k:04d}.jpg"))
        if img is None:
            continue
        for (x, y, w, h), lab, c in zip(r.get("boxes", []), r.get("labels", []), r.get("colors", [])):
            col = COLORS.get(c, COLORS[0])
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), col, 2 if c else 1)
            text(img, lab, (int(x) + 2, max(int(y) - 4, 16)), 0.45, (0, 0, 0), 1, col)
        if "phrase" in r:
            caption, until, ccol = r["phrase"], r["t"] + 3.0, GROUP_COLOR.get(r["kind"], (255, 255, 255))
        text(img, f"t {r['t']:5.1f} s   surface: {r.get('surface') or '-'}   tracks: {r['tracks']}"
                  + (f"   LiDAR obstacle {r['obstacle_m']:.1f} m" if "obstacle_m" in r else ""), (12, 26), 0.55)
        text(img, "CPEWalk pipeline on SANPO (iOS Simulator replay)", (12, H - 14), 0.45, (220, 220, 220))
        if r["t"] <= until:
            (tw, _), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            text(img, caption, ((W - tw) // 2, H - 52), 0.9, (0, 0, 0), 2, ccol)
        vw.write(img)
    vw.release()
    said = [r for r in rows if "phrase" in r]
    dur = len(rows) / 5
    with tempfile.TemporaryDirectory() as td:
        inputs, chains = ["-i", str(tmp)], []
        for i, r in enumerate(said):
            wav = Path(td) / f"{i}.aiff"
            subprocess.run(["say", "-v", "Samantha", "-r", "200", "-o", str(wav), r["phrase"]], check=True)
            inputs += ["-i", str(wav)]
            start = r["t"]
            end = said[i + 1]["t"] if i + 1 < len(said) else dur        # the next alert cuts this one
            a = math.radians((max(-1.0, min(1.0, r.get("bearing_deg", 0.0) / 60)) + 1) * 45)
            l, rr = math.cos(a), math.sin(a)                              # constant-power pan
            ms = int(start * 1000)
            chains.append(f"[{i + 1}:a]atrim=0:{max(end - start, 0.05):.3f},aformat=channel_layouts=mono,"
                          f"pan=stereo|c0={l:.3f}*c0|c1={rr:.3f}*c0,adelay={ms}|{ms}[a{i}]")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs]
        if said:
            mix = "".join(f"[a{i}]" for i in range(len(said)))
            graph = ";".join(chains) + f";{mix}amix=inputs={len(said)}:normalize=0,apad[aout]"
            cmd += ["-filter_complex", graph, "-map", "0:v", "-map", "[aout]", "-c:a", "aac", "-b:a", "128k"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-t", f"{dur:.2f}", str(out)]
        subprocess.run(cmd, check=True)
    tmp.unlink()
    print(out)


if __name__ == "__main__":
    main()
