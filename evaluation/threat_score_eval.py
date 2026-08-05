"""
threat_score_eval.py
====================
CPE Evaluation — Routing F1 / Precision / Recall evaluator.

Uses future kinetic ground truth (K₊₂) from a Stage 1 perception CSV to
label whether each frame's top-threat routing decision was correct, then
computes per-formula, per-route precision, recall, and F1 scores.

This answers: "Does our routing decision (reflex/cognitive/ignore) correctly
classify objects that turned out to be genuinely dangerous two seconds later?"

Ground-Truth Definition
-----------------------
For each object at frame T with TTC-derived ground truth:
  - TRUE REFLEX   : At frame T+lookahead, same track_id has d < 1.5m  OR  was
                    the top-K object AND K₊₂ > high_k_threshold
  - TRUE COGNITIVE: K₊₂ in [low_k, high_k] range at T+lookahead
  - TRUE IGNORE   : K₊₂ < low_k at T+lookahead OR track lost

Usage
-----
python evaluation/threat_score_eval.py \\
    --csv         data/processed/merged_perception.csv \\
    --fps         30 \\
    [--lookahead-s 2.0] \\
    [--low-k  0.5] \\
    [--high-k 5.0] \\
    [--out-dir evaluation/benchmarks/kinetic_score_eval]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception_stack.physics import CLASS_SEVERITY, DEFAULT_SEVERITY, EPSILON

# Import all formula candidates from the comparison script (sibling file)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kinetic_score_comparison import (
    FORMULAS,
    load_csv as _load_csv_raw,
    route_label,
    _mean,
    _percentile,
)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_LOW_K     = 0.5
DEFAULT_HIGH_K    = 5.0
DEFAULT_LOOKAHEAD = 2.0
REFLEX_TTC_S      = 1.0
REFLEX_DIST_M     = 1.5   # object within this distance at T+lookahead → true reflex


# ══════════════════════════════════════════════════════════════════════════════
# CSV Loading (extended — keeps all rows for future-frame lookup)
# ══════════════════════════════════════════════════════════════════════════════

def load_csv_all(csv_path: Path) -> dict[int, list[dict]]:
    """Load CSV into frame_idx → list[row] dict. Includes rows with None distance."""
    frames: dict[int, list[dict]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame_idx = int(row.get("frame_idx", 0))
            d = row.get("distance_m", "")
            v = row.get("velocity_ms", "")
            frames[frame_idx].append({
                "frame_idx":   frame_idx,
                "track_id":    row.get("track_id", ""),
                "class_name":  row.get("class", row.get("class_name", "unknown")),
                "distance_m":  float(d) if d and d.strip() else None,
                "velocity_ms": float(v) if v and v.strip() else 0.0,
            })
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Ground-Truth Labelling
# ══════════════════════════════════════════════════════════════════════════════

def compute_k(row: dict, fn: callable) -> float:
    d = row["distance_m"]
    if d is None:
        return 0.0
    return fn(d, row["velocity_ms"], row["class_name"])


def ground_truth_label(
    row: dict,
    future_frames: dict[int, list[dict]],
    lookahead_frames: int,
    fn: callable,
    low_k: float,
    high_k: float,
) -> str:
    """
    Derive the ground-truth routing label for a row using K₊lookahead.

    Returns 'reflex', 'cognitive', or 'ignore'.
    """
    future_frame_idx = row["frame_idx"] + lookahead_frames
    future_rows = future_frames.get(future_frame_idx, [])

    # Find the same track in the future frame
    same_track = next(
        (r for r in future_rows if str(r["track_id"]) == str(row["track_id"])),
        None,
    )

    if same_track is None:
        # Track lost — assume object cleared; treat as ignore
        return "ignore"

    k_future = compute_k(same_track, fn)
    d_future = same_track["distance_m"]

    # Hard reflex: object within REFLEX_DIST_M at lookahead time
    if d_future is not None and d_future <= REFLEX_DIST_M:
        return "reflex"
    if k_future >= high_k:
        return "reflex"
    if k_future >= low_k:
        return "cognitive"
    return "ignore"


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def evaluate_formula(
    fn: callable,
    frames: dict[int, list[dict]],
    lookahead_frames: int,
    low_k: float,
    high_k: float,
) -> dict:
    """Evaluate one formula across all frames."""
    routes = ["reflex", "cognitive", "ignore"]
    counters = {r: {"tp": 0, "fp": 0, "fn": 0} for r in routes}

    for frame_idx, rows in frames.items():
        for row in rows:
            if row["distance_m"] is None:
                continue
            k = compute_k(row, fn)
            ttc_s = row["distance_m"] / row["velocity_ms"] if row["velocity_ms"] > 0 else None
            pred = route_label(k, ttc_s, low_k, high_k, REFLEX_TTC_S)
            gt   = ground_truth_label(row, frames, lookahead_frames, fn, low_k, high_k)

            for r in routes:
                is_pred = (pred == r)
                is_gt   = (gt   == r)
                if is_pred and is_gt:
                    counters[r]["tp"] += 1
                elif is_pred and not is_gt:
                    counters[r]["fp"] += 1
                elif not is_pred and is_gt:
                    counters[r]["fn"] += 1

    result = {}
    macro_f1 = 0.0
    for r in routes:
        c = counters[r]
        m = precision_recall_f1(c["tp"], c["fp"], c["fn"])
        m["tp"] = c["tp"]
        m["fp"] = c["fp"]
        m["fn"] = c["fn"]
        result[r] = m
        macro_f1 += m["f1"]
    result["macro_f1"] = round(macro_f1 / len(routes), 4)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Report Rendering
# ══════════════════════════════════════════════════════════════════════════════

def render_report(all_results: dict, csv_path: Path,
                  low_k: float, high_k: float, lookahead_s: float) -> str:
    lines = [
        "# CPE Threat Score Evaluation Report",
        "",
        f"**Input CSV:** `{csv_path.name}`  ",
        f"**Routing thresholds:** low_k={low_k}, high_k={high_k}  ",
        f"**Ground-truth lookahead:** {lookahead_s}s  ",
        f"**Ground-truth reflex distance:** ≤ {REFLEX_DIST_M}m at T+lookahead",
        "",
        "> Precision / Recall / F1 for each routing lane under each formula.",
        "> Ground truth is derived from K₊lookahead (same track at future frame).",
        "",
        "---",
        "",
        "## 1. Reflex Lane (most safety-critical)",
        "",
        "| Formula | Precision | Recall | F1 | TP | FP | FN |",
        "|---------|-----------|--------|----|----|----|----|",
    ]
    for name, res in all_results.items():
        r = res["reflex"]
        lines.append(
            f"| {name} | {r['precision']} | {r['recall']} | {r['f1']} | "
            f"{r['tp']} | {r['fp']} | {r['fn']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 2. Cognitive Lane",
        "",
        "| Formula | Precision | Recall | F1 | TP | FP | FN |",
        "|---------|-----------|--------|----|----|----|----|",
    ]
    for name, res in all_results.items():
        r = res["cognitive"]
        lines.append(
            f"| {name} | {r['precision']} | {r['recall']} | {r['f1']} | "
            f"{r['tp']} | {r['fp']} | {r['fn']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. Ignore Lane",
        "",
        "| Formula | Precision | Recall | F1 |",
        "|---------|-----------|--------|----|",
    ]
    for name, res in all_results.items():
        r = res["ignore"]
        lines.append(f"| {name} | {r['precision']} | {r['recall']} | {r['f1']} |")

    lines += [
        "",
        "---",
        "",
        "## 4. Macro F1 Ranking (primary ranking metric)",
        "",
        "| Rank | Formula | Macro F1 | Reflex F1 | Cognitive F1 |",
        "|------|---------|----------|-----------|--------------|",
    ]
    ranked = sorted(all_results.items(), key=lambda x: x[1]["macro_f1"], reverse=True)
    for rank, (name, res) in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {name} | **{res['macro_f1']}** | "
            f"{res['reflex']['f1']} | {res['cognitive']['f1']} |"
        )

    best_name, best_res = ranked[0]
    lines += [
        "",
        "---",
        "",
        "## 5. Recommendation",
        "",
        f"> **Best formula by Macro F1:** `{best_name}` (Macro F1 = {best_res['macro_f1']})",
        ">",
        f"> Reflex F1 = {best_res['reflex']['f1']} — **this is the safety-critical metric.**",
        f"> A high Reflex Recall means we rarely miss genuine collision-risk objects.",
        "",
        "> [!IMPORTANT]",
        "> Prefer the formula with the highest **Reflex Recall** over raw Macro F1.",
        "> A false positive in the reflex lane is safe (overcaution);",
        "> a false negative (missed reflex) is a safety failure.",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPE Threat Score Evaluator — F1/Precision/Recall per routing lane."
    )
    parser.add_argument("--csv",          type=Path, required=True,
                        help="Stage 1 perception CSV path.")
    parser.add_argument("--fps",          type=float, default=30.0,
                        help="Source video framerate (used to compute lookahead window).")
    parser.add_argument("--lookahead-s",  type=float, default=DEFAULT_LOOKAHEAD,
                        help="Seconds ahead for K₊lookahead ground truth (default: 2.0).")
    parser.add_argument("--low-k",        type=float, default=DEFAULT_LOW_K)
    parser.add_argument("--high-k",       type=float, default=DEFAULT_HIGH_K)
    parser.add_argument("--out-dir",      type=Path,
                        default=Path("evaluation/benchmarks/kinetic_score_eval"),
                        help="Output directory for report and JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lookahead_frames = int(args.fps * args.lookahead_s)

    print(f"\n=== CPE Threat Score Evaluation ===")
    print(f"  Input CSV     : {args.csv}")
    print(f"  Lookahead     : {args.lookahead_s}s ({lookahead_frames} frames)")
    print(f"  Thresholds    : low_k={args.low_k}, high_k={args.high_k}")
    print(f"  Output dir    : {args.out_dir}\n")

    frames = load_csv_all(args.csv)
    total_rows = sum(len(v) for v in frames.values())
    print(f"  Loaded {len(frames)} frames, {total_rows} total rows.\n")

    all_results = {}
    for name, fn in FORMULAS.items():
        print(f"  Evaluating {name}...", end="", flush=True)
        res = evaluate_formula(fn, frames, lookahead_frames, args.low_k, args.high_k)
        all_results[name] = res
        print(f"  macro_f1={res['macro_f1']}  reflex_f1={res['reflex']['f1']}")

    # Save outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_md = render_report(all_results, args.csv, args.low_k, args.high_k, args.lookahead_s)
    report_path = args.out_dir / "threat_score_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"\n  ✅ Report saved: {report_path}")

    json_path = args.out_dir / "threat_score_results.json"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"  ✅ JSON saved:   {json_path}")

    # Print ranked summary
    print("\n── Macro F1 Ranking ─────────────────────────────────────────")
    ranked = sorted(all_results.items(), key=lambda x: x[1]["macro_f1"], reverse=True)
    for rank, (name, res) in enumerate(ranked, 1):
        print(
            f"  {rank}. {name:<25}  macro_f1={res['macro_f1']}"
            f"  reflex_recall={res['reflex']['recall']}"
        )
    print(f"\n  🏆 Best: {ranked[0][0]}")
    print(f"  ⚠️  Check Reflex Recall — missed reflex = safety failure.")


if __name__ == "__main__":
    main()
