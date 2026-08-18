"""
manual_score_inspector.py
=========================
CPE Evaluation — Interactive Manual Inspection Tool.

Generates a single self-contained HTML file that lets you manually verify
whether kinetic scores make sense for each scene frame. Open the HTML in
any browser — no server required.

For each frame you will see:
  • The actual RGB frame image with bounding boxes drawn per object.
  • A colour-coded table showing K0–K5 scores, TTC, distance, velocity,
    and the routing decision each formula would make.
  • The future state of each object at T+lookahead_s (ground-truth column).
  • ✅ / ❌ stamp buttons — your judgment is saved to an audit JSON file
    next to the HTML so you can resume across multiple sessions.

Usage
-----
python tools/manual_score_inspector.py \\
    --csv   data/processed/merged_perception.csv \\
    --rgb   data/sanpo/sample/rgb \\
    --out   evaluation/benchmarks/kinetic_score_eval/manual_inspection.html \\
    [--fps       30] \\
    [--lookahead 2.0] \\
    [--max-frames 200]

Flags
-----
--max-frames : Limit the number of frames rendered (default: 200).
               Frames are sampled uniformly across the CSV, so you get
               coverage of the whole session, not just the first N seconds.
--fps        : Source framerate — used to compute the lookahead window.
--lookahead  : Seconds in the future used for ground-truth labelling.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception_stack.physics import (
    CLASS_SEVERITY, DEFAULT_SEVERITY, EPSILON,
)
from src.perception_stack.physics import bearing_label

# ── Try importing optional deps (Pillow for image drawing) ───────────────────
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARN] Pillow not installed — frames will be shown without bbox overlays.")
    print("       pip install Pillow   to enable bbox drawing.")


# ══════════════════════════════════════════════════════════════════════════════
# Kinetic Formula Definitions (K0 production + ablation variants)
# ══════════════════════════════════════════════════════════════════════════════

def _sev(c): return CLASS_SEVERITY.get(c, DEFAULT_SEVERITY)

FORMULAS = {
    "K0": lambda d, v, c: _sev(c) * (v**2) / max(d, EPSILON),
    "K1": lambda d, v, c: 0.0 if v <= 0 else _sev(c) * min(1.0 / (d / v), 10.0),
    "K2": lambda d, v, c: _sev(c) * v / max(d, EPSILON),
    "K3": lambda d, v, c: _sev(c) * (v**2) / (d**2 + EPSILON),
    "K4": lambda d, v, c: (
        0.5 * (_sev(c) * (v**2) / max(d, EPSILON)) +
        0.5 * (0.0 if v <= 0 else _sev(c) * min(10.0 / (d / v), 10.0))
    ),
    "K5": lambda d, v, c: _sev(c) / (1.0 + math.exp(-(v / max(d, EPSILON) - 2.0))),
}

REFLEX_DIST_M = 1.5   # hard physical danger zone (meters)
DEFAULT_LOW_K  = 0.5
DEFAULT_HIGH_K = 5.0


def route(k: float, ttc: Optional[float], low_k: float, high_k: float) -> str:
    if ttc is not None and ttc <= 1.0:
        return "reflex"
    if k >= high_k:
        return "reflex"
    if k >= low_k:
        return "cognitive"
    return "ignore"


def route_badge(r: str) -> str:
    colour = {"reflex": "#c0392b", "cognitive": "#e67e22", "ignore": "#27ae60"}[r]
    return f'<span style="background:{colour};color:#fff;padding:1px 6px;border-radius:3px;font-size:0.8em">{r}</span>'


def k_colour(k: float, high_k: float) -> str:
    """Return a red→green background colour proportional to K magnitude."""
    ratio = min(k / max(high_k, 1.0), 1.0)
    r = int(220 * ratio)
    g = int(200 * (1 - ratio))
    return f"rgba({r},{g},40,0.25)"


# ══════════════════════════════════════════════════════════════════════════════
# CSV Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(csv_path: Path) -> dict[int, list[dict]]:
    frames: dict = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["frame_idx"])
            row["frame_idx"]   = idx
            row["distance_m"]  = float(row["distance_m"])  if row["distance_m"]  else None
            row["velocity_ms"] = float(row["velocity_ms"]) if row["velocity_ms"] else 0.0
            row["bearing_deg"] = float(row["bearing_deg"]) if row["bearing_deg"] else 0.0
            row["bbox_x1"]     = int(float(row["bbox_x1"])) if row.get("bbox_x1") else None
            row["bbox_y1"]     = int(float(row["bbox_y1"])) if row.get("bbox_y1") else None
            row["bbox_x2"]     = int(float(row["bbox_x2"])) if row.get("bbox_x2") else None
            row["bbox_y2"]     = int(float(row["bbox_y2"])) if row.get("bbox_y2") else None
            frames[idx].append(row)
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Image Rendering
# ══════════════════════════════════════════════════════════════════════════════

_ROUTE_COLOURS = {
    "reflex":    (192, 57, 43),
    "cognitive": (230, 126, 34),
    "ignore":    (39, 174, 96),
}

def _find_rgb_frame(rgb_dir: Path, frame_idx: int) -> Optional[Path]:
    """Try common naming conventions for SANPO RGB frames."""
    for pattern in [
        f"{frame_idx:06d}.jpg", f"{frame_idx:06d}.png",
        f"{frame_idx:05d}.jpg", f"{frame_idx:05d}.png",
        f"frame_{frame_idx:06d}.jpg", f"frame_{frame_idx:06d}.png",
    ]:
        p = rgb_dir / pattern
        if p.exists():
            return p
    return None


def _render_frame_b64(
    rgb_dir: Optional[Path],
    frame_idx: int,
    rows: list[dict],
    high_k: float,
) -> str:
    """Return a base64-encoded PNG of the frame with bboxes, or empty string."""
    if not HAS_PIL or rgb_dir is None:
        return ""

    img_path = _find_rgb_frame(rgb_dir, frame_idx)
    if img_path is None:
        return ""

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for row in rows:
        if None in (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]):
            continue
        if row["distance_m"] is None:
            continue
        d, v, c = row["distance_m"], row["velocity_ms"], row["class"]
        ttc = d / v if v > 0 else None
        k0  = FORMULAS["K0"](d, v, c)
        r   = route(k0, ttc, DEFAULT_LOW_K, high_k)
        colour = _ROUTE_COLOURS[r]
        x1, y1, x2, y2 = row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
        label = f"{row['class']} K0={k0:.1f}"
        draw.rectangle([x1, y1-16, x1+len(label)*7, y1], fill=colour)
        draw.text((x1+2, y1-15), label, fill=(255, 255, 255))

    buf = BytesIO()
    # Resize to max 720px wide to keep HTML file size manageable
    w, h = img.size
    if w > 720:
        img = img.resize((720, int(h * 720 / w)), Image.LANCZOS)
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


# ══════════════════════════════════════════════════════════════════════════════
# HTML Generation — one card per frame
# ══════════════════════════════════════════════════════════════════════════════

def _score_table(
    rows: list[dict],
    future_rows: list[dict],
    high_k: float,
    low_k: float,
) -> str:
    """Render the per-object score table as HTML."""
    formula_names = list(FORMULAS.keys())
    future_by_track = {str(r["track_id"]): r for r in future_rows}

    html = ['<table class="score-table">']
    html.append(
        "<tr>"
        "<th>Object</th><th>Class</th><th>Dist</th><th>Vel</th><th>TTC</th>"
        + "".join(f"<th>{n}</th>" for n in formula_names)
        + "<th>GT (future dist)</th></tr>"
    )

    for row in rows:
        d   = row["distance_m"]
        v   = row["velocity_ms"]
        c   = row["class"]
        tid = str(row["track_id"])
        brng = bearing_label(row["bearing_deg"])

        if d is None:
            continue

        ttc = d / v if v > 0 else None
        ttc_str = f"{ttc:.1f}s" if ttc is not None else "∞"

        # Future ground truth
        frow = future_by_track.get(tid)
        if frow and frow["distance_m"] is not None:
            fd = frow["distance_m"]
            gt_danger = fd <= REFLEX_DIST_M
            gt_cell = f'<td style="color:{"#c0392b" if gt_danger else "#27ae60"};font-weight:bold">{fd:.2f}m {"⚠️" if gt_danger else "✓"}</td>'
        elif frow is None:
            gt_cell = '<td style="color:#888">track lost</td>'
        else:
            gt_cell = '<td style="color:#888">no depth</td>'

        # K scores
        score_cells = ""
        for name, fn in FORMULAS.items():
            k = fn(d, v, c)
            r = route(k, ttc, low_k, high_k)
            bg = k_colour(k, high_k)
            score_cells += (
                f'<td style="background:{bg}">'
                f'{k:.2f}<br>{route_badge(r)}</td>'
            )

        html.append(
            f"<tr>"
            f'<td>{tid} <small style="color:#888">{brng}</small></td>'
            f"<td>{c}</td>"
            f"<td>{d:.2f}m</td>"
            f"<td>{v:.2f}m/s</td>"
            f"<td>{ttc_str}</td>"
            + score_cells
            + gt_cell
            + "</tr>"
        )

    html.append("</table>")
    return "\n".join(html)


def _frame_card(
    frame_idx: int,
    rows: list[dict],
    future_rows: list[dict],
    b64_img: str,
    high_k: float,
    low_k: float,
    card_idx: int,
) -> str:
    img_html = (
        f'<img src="data:image/png;base64,{b64_img}" '
        f'style="max-width:100%;border-radius:4px;margin-bottom:8px">'
        if b64_img else
        '<div style="background:#1a1a2e;color:#666;padding:20px;text-align:center;border-radius:4px">'
        '⚠️ Image not found (check --rgb path)</div>'
    )
    table = _score_table(rows, future_rows, high_k, low_k)
    return f"""
<div class="frame-card" id="card-{card_idx}" data-frame="{frame_idx}">
  <div class="card-header">
    <span class="frame-label">Frame {frame_idx}</span>
    <span class="obj-count">{len(rows)} objects</span>
    <span class="stamp-area">
      <button class="stamp ok-btn"    onclick="stamp({card_idx},'ok')">✅ Scores look correct</button>
      <button class="stamp flag-btn"  onclick="stamp({card_idx},'flag')">❌ Scores look wrong</button>
      <button class="stamp skip-btn"  onclick="stamp({card_idx},'skip')">⏭ Skip</button>
      <span class="stamp-result" id="stamp-{card_idx}"></span>
    </span>
  </div>
  <div class="card-body">
    <div class="img-col">{img_html}</div>
    <div class="table-col">{table}</div>
  </div>
</div>"""


# ══════════════════════════════════════════════════════════════════════════════
# Full HTML Document
# ══════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CPE Manual Kinetic Score Inspector</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f0f1a; color: #e0e0e0; }}

  /* ── Header ── */
  .top-bar {{
    position: sticky; top: 0; z-index: 100;
    background: #1a1a2e; border-bottom: 1px solid #2d2d4e;
    padding: 10px 20px; display: flex; align-items: center; gap: 20px;
  }}
  .top-bar h1 {{ font-size: 1.1em; color: #7ec8e3; }}
  .progress-wrap {{ flex: 1; background: #0f0f1a; border-radius: 6px; height: 8px; }}
  .progress-bar {{ height: 8px; border-radius: 6px; background: linear-gradient(90deg,#27ae60,#2ecc71); transition: width 0.3s; }}
  .counter {{ font-size: 0.85em; color: #aaa; white-space: nowrap; }}
  .export-btn {{
    background: #2980b9; color: #fff; border: none; padding: 6px 14px;
    border-radius: 5px; cursor: pointer; font-size: 0.85em;
  }}
  .export-btn:hover {{ background: #3498db; }}

  /* ── Legend ── */
  .legend {{
    background: #13132a; border-bottom: 1px solid #2d2d4e;
    padding: 8px 20px; font-size: 0.8em; color: #aaa; display: flex; gap: 16px; flex-wrap: wrap;
  }}
  .legend span {{ display: flex; align-items: center; gap: 4px; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}

  /* ── Frame cards ── */
  .frame-card {{
    margin: 16px 20px; border: 1px solid #2d2d4e; border-radius: 8px;
    background: #16162a; overflow: hidden;
  }}
  .frame-card.stamped-ok   {{ border-color: #27ae60; }}
  .frame-card.stamped-flag {{ border-color: #c0392b; }}
  .frame-card.stamped-skip {{ border-color: #555; opacity: 0.7; }}

  .card-header {{
    background: #1e1e3a; padding: 8px 14px;
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  }}
  .frame-label {{ font-weight: 700; color: #7ec8e3; font-size: 0.9em; }}
  .obj-count   {{ font-size: 0.75em; color: #888; background: #2d2d4e; padding: 2px 7px; border-radius: 10px; }}
  .stamp-area  {{ margin-left: auto; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
  .stamp       {{
    border: none; cursor: pointer; padding: 5px 12px; border-radius: 5px;
    font-size: 0.8em; font-weight: 600; transition: opacity 0.15s;
  }}
  .ok-btn   {{ background: #1e6b3e; color: #2ecc71; }}
  .flag-btn {{ background: #6b1e1e; color: #e74c3c; }}
  .skip-btn {{ background: #2d2d4e; color: #888; }}
  .stamp:hover {{ opacity: 0.8; }}
  .stamp-result {{ font-size: 0.85em; font-weight: 700; min-width: 120px; }}

  .card-body {{
    display: flex; gap: 12px; padding: 12px;
    flex-wrap: wrap;
  }}
  .img-col   {{ flex: 0 0 auto; max-width: 380px; width: 100%; }}
  .table-col {{ flex: 1 1 auto; overflow-x: auto; }}

  /* ── Score table ── */
  .score-table {{
    width: 100%; border-collapse: collapse; font-size: 0.78em;
  }}
  .score-table th {{
    background: #1e1e3a; color: #7ec8e3; padding: 5px 8px;
    text-align: center; white-space: nowrap; border-bottom: 1px solid #2d2d4e;
  }}
  .score-table td {{
    padding: 5px 8px; text-align: center; border-bottom: 1px solid #1a1a2e;
    vertical-align: middle;
  }}
  .score-table tr:hover td {{ background: rgba(255,255,255,0.03); }}

  /* ── Filter bar ── */
  .filter-bar {{
    padding: 8px 20px; background: #12122a; border-bottom: 1px solid #2d2d4e;
    display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
  }}
  .filter-bar label {{ font-size: 0.8em; color: #aaa; }}
  .filter-bar button {{
    background: #2d2d4e; color: #ccc; border: 1px solid #3d3d5e;
    padding: 4px 12px; border-radius: 5px; cursor: pointer; font-size: 0.8em;
  }}
  .filter-bar button.active {{ background: #2980b9; color: #fff; border-color: #3498db; }}

  /* ── Summary panel ── */
  #summary-panel {{
    margin: 16px 20px; background: #16162a; border: 1px solid #2d2d4e;
    border-radius: 8px; padding: 16px; display: none;
  }}
  #summary-panel h2 {{ color: #7ec8e3; margin-bottom: 10px; }}
  #summary-content {{ font-size: 0.85em; line-height: 1.8; }}
</style>
</head>
<body>

<div class="top-bar">
  <h1>📐 CPE Kinetic Score Inspector</h1>
  <div class="progress-wrap"><div class="progress-bar" id="prog-bar" style="width:0%"></div></div>
  <span class="counter" id="counter">0 / {total} reviewed</span>
  <button class="export-btn" onclick="exportAudit()">💾 Export Audit JSON</button>
  <button class="export-btn" style="background:#8e44ad" onclick="showSummary()">📊 Summary</button>
</div>

<div class="legend">
  <span><span class="dot" style="background:#c0392b"></span> Reflex (danger — K ≥ {high_k} or TTC ≤ 1s)</span>
  <span><span class="dot" style="background:#e67e22"></span> Cognitive (K ≥ {low_k})</span>
  <span><span class="dot" style="background:#27ae60"></span> Ignore (K &lt; {low_k})</span>
  <span>· Ground-truth column shows actual distance at T+{lookahead}s — ⚠️ = breached {reflex_dist}m danger zone</span>
</div>

<div class="filter-bar">
  <label>Show:</label>
  <button class="active" id="f-all"    onclick="filter('all')">All</button>
  <button id="f-ok"     onclick="filter('ok')">✅ Verified OK</button>
  <button id="f-flag"   onclick="filter('flag')">❌ Flagged</button>
  <button id="f-unseen" onclick="filter('unseen')">⏳ Unseen</button>
</div>

{cards}

<div id="summary-panel">
  <h2>📊 Audit Summary</h2>
  <div id="summary-content"></div>
</div>

<script>
const TOTAL = {total};
const stamps = {{}};
let currentFilter = 'all';

function stamp(idx, verdict) {{
  stamps[idx] = {{
    verdict: verdict,
    frame_id: document.getElementById('card-' + idx).dataset.frame,
    ts: new Date().toISOString()
  }};
  const card = document.getElementById('card-' + idx);
  card.className = 'frame-card stamped-' + verdict;
  const labels = {{ ok: '✅ Correct', flag: '❌ Flagged', skip: '⏭ Skipped' }};
  document.getElementById('stamp-' + idx).textContent = labels[verdict];
  updateProgress();
  applyFilter();
  // Auto-save to localStorage
  localStorage.setItem('cpe_audit_{session_key}', JSON.stringify(stamps));
}}

function updateProgress() {{
  const reviewed = Object.keys(stamps).length;
  const pct = (reviewed / TOTAL * 100).toFixed(0);
  document.getElementById('prog-bar').style.width = pct + '%';
  document.getElementById('counter').textContent = reviewed + ' / ' + TOTAL + ' reviewed';
}}

function filter(f) {{
  currentFilter = f;
  document.querySelectorAll('.filter-bar button').forEach(b => b.classList.remove('active'));
  document.getElementById('f-' + f).classList.add('active');
  applyFilter();
}}

function applyFilter() {{
  document.querySelectorAll('.frame-card').forEach(card => {{
    const idx = card.id.replace('card-', '');
    const verdict = stamps[idx] ? stamps[idx].verdict : null;
    let show = false;
    if (currentFilter === 'all')    show = true;
    if (currentFilter === 'ok')     show = verdict === 'ok';
    if (currentFilter === 'flag')   show = verdict === 'flag';
    if (currentFilter === 'unseen') show = verdict === null;
    card.style.display = show ? '' : 'none';
  }});
}}

function exportAudit() {{
  const ok   = Object.values(stamps).filter(s => s.verdict === 'ok').length;
  const flag = Object.values(stamps).filter(s => s.verdict === 'flag').length;
  const skip = Object.values(stamps).filter(s => s.verdict === 'skip').length;
  const data = {{
    meta: {{
      total_frames: TOTAL,
      reviewed: Object.keys(stamps).length,
      ok, flag, skip,
      human_accuracy_pct: TOTAL > 0 ? (ok / TOTAL * 100).toFixed(1) : 'N/A',
      exported_at: new Date().toISOString()
    }},
    judgements: stamps
  }};
  const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'cpe_manual_audit.json';
  a.click();
}}

function showSummary() {{
  const ok   = Object.values(stamps).filter(s => s.verdict === 'ok').length;
  const flag = Object.values(stamps).filter(s => s.verdict === 'flag').length;
  const skip = Object.values(stamps).filter(s => s.verdict === 'skip').length;
  const unseen = TOTAL - Object.keys(stamps).length;
  const panel = document.getElementById('summary-panel');
  panel.style.display = 'block';
  document.getElementById('summary-content').innerHTML = `
    <b>Total frames rendered:</b> ${{TOTAL}}<br>
    <b>Reviewed:</b> ${{Object.keys(stamps).length}}<br>
    <b>✅ Scores verified correct:</b> ${{ok}}<br>
    <b>❌ Scores flagged as wrong:</b> ${{flag}}<br>
    <b>⏭ Skipped:</b> ${{skip}}<br>
    <b>⏳ Unseen:</b> ${{unseen}}<br>
    <br>
    <b>Human-verified score accuracy:</b> ${{
      ok + flag > 0 ? (ok / (ok + flag) * 100).toFixed(1) + '%' : 'N/A'
    }} of inspected frames had correct scores.<br>
    <br>
    <i>Click "💾 Export Audit JSON" to save this data for use as ground-truth labels.</i>
  `;
  panel.scrollIntoView({{behavior: 'smooth'}});
}}

// Restore previous session from localStorage
const saved = localStorage.getItem('cpe_audit_{session_key}');
if (saved) {{
  const prev = JSON.parse(saved);
  Object.keys(prev).forEach(idx => {{
    if (prev[idx] && prev[idx].verdict) {{
      stamp(parseInt(idx), prev[idx].verdict);
    }}
  }});
}}
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPE Manual Kinetic Score Inspector — generates interactive HTML review tool"
    )
    parser.add_argument("--csv",        required=True, type=Path,
                        help="Path to Stage 1 perception CSV")
    parser.add_argument("--rgb",        type=Path, default=None,
                        help="Folder of RGB frames (JPG/PNG). Skipped if not provided.")
    parser.add_argument("--out",        type=Path,
                        default=Path("evaluation/benchmarks/kinetic_score_eval/manual_inspection.html"),
                        help="Output HTML path")
    parser.add_argument("--fps",        type=float, default=30.0)
    parser.add_argument("--lookahead",  type=float, default=2.0,
                        help="Seconds ahead used for ground-truth future column")
    parser.add_argument("--max-frames", type=int,   default=200,
                        help="Max frames to render (uniformly sampled)")
    parser.add_argument("--low-k",      type=float, default=DEFAULT_LOW_K)
    parser.add_argument("--high-k",     type=float, default=DEFAULT_HIGH_K)
    args = parser.parse_args()

    print(f"\n=== CPE Manual Score Inspector ===")
    print(f"  CSV       : {args.csv}")
    print(f"  RGB dir   : {args.rgb or 'not provided — no images'}")
    print(f"  Lookahead : {args.lookahead}s ({int(args.fps * args.lookahead)} frames)")
    print(f"  Max frames: {args.max_frames}")

    frames = load_csv(args.csv)
    all_indices = sorted(frames.keys())
    total = len(all_indices)
    print(f"  Total frames in CSV: {total}")

    # Uniform sampling
    if total > args.max_frames:
        step = total / args.max_frames
        sampled = [all_indices[int(i * step)] for i in range(args.max_frames)]
    else:
        sampled = all_indices

    lookahead_frames = int(args.fps * args.lookahead)
    print(f"  Rendering : {len(sampled)} frames → {args.out}")

    cards_html = []
    for card_idx, frame_idx in enumerate(sampled):
        rows        = frames[frame_idx]
        future_rows = frames.get(frame_idx + lookahead_frames, [])
        b64_img     = _render_frame_b64(args.rgb, frame_idx, rows, args.high_k)
        cards_html.append(
            _frame_card(frame_idx, rows, future_rows, b64_img, args.high_k, args.low_k, card_idx)
        )
        if (card_idx + 1) % 20 == 0:
            print(f"  [{card_idx + 1}/{len(sampled)}] frames processed…")

    import hashlib
    session_key = hashlib.md5(str(args.csv).encode()).hexdigest()[:8]

    html = HTML_TEMPLATE.format(
        total=len(sampled),
        high_k=args.high_k,
        low_k=args.low_k,
        lookahead=args.lookahead,
        reflex_dist=REFLEX_DIST_M,
        session_key=session_key,
        cards="\n".join(cards_html),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    size_mb = args.out.stat().st_size / 1_048_576
    print(f"\n✅  Inspector ready: {args.out}  ({size_mb:.1f} MB)")
    print(f"   Open in any browser. Your stamps auto-save to localStorage.")
    print(f"   Click '💾 Export Audit JSON' to save a ground-truth annotation file.")


if __name__ == "__main__":
    main()
