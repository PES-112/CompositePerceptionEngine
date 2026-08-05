"""
kinetic_score_comparison.py
===========================
CPE Evaluation — Benchmark harness for kinetic score formula candidates.

Runs K0 (current) through K5 (sigmoid-normalised) over a Stage 1 perception
CSV, then produces:
  1. Per-formula distribution statistics (mean, P50, P95, P99, IQR)
  2. TTC Pearson/Spearman correlation for rows where TTC is defined
  3. Routing threshold sensitivity — what fraction of objects route to
     reflex/cognitive/ignore under each formula at current thresholds
  4. Monotonicity verification (K increases as d↓ and v↑)
  5. A ranked summary table written to
     evaluation/benchmarks/kinetic_score_eval/kinetic_score_report.md
  6. Matplotlib plots saved to
     evaluation/benchmarks/kinetic_score_eval/plots/

Usage
-----
python evaluation/kinetic_score_comparison.py \
    --csv  data/processed/merged_perception.csv \
    --fps  30 \
    [--low-k  0.5] \
    [--high-k 5.0] \
    [--out-dir evaluation/benchmarks/kinetic_score_eval]

The script is self-contained: it does NOT modify any source files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception_stack.physics import CLASS_SEVERITY, DEFAULT_SEVERITY, EPSILON

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_LOW_K  = 0.5
DEFAULT_HIGH_K = 5.0
SIGMOID_BETA   = 2.0   # calibration offset for K5


# ══════════════════════════════════════════════════════════════════════════════
# Formula Definitions (K0–K5)
# ══════════════════════════════════════════════════════════════════════════════

def _sev(class_name: str) -> float:
    return CLASS_SEVERITY.get(class_name, DEFAULT_SEVERITY)


def k0_current(distance_m: float, velocity_ms: float, class_name: str) -> float:
    """K0 (current): sev × v² / max(d, ε)  — physics kinetic energy proxy."""
    return _sev(class_name) * (velocity_ms ** 2) / max(distance_m, EPSILON)


def k1_ttc_primary(distance_m: float, velocity_ms: float, class_name: str) -> float:
    """K1: sev × clamp(1/TTC, 0, 10)  — directly models time budget."""
    if velocity_ms <= 0:
        return 0.0
    ttc = distance_m / velocity_ms
    return _sev(class_name) * min(1.0 / ttc, 10.0)


def k2_linear_velocity(distance_m: float, velocity_ms: float, class_name: str) -> float:
    """K2: sev × v / max(d, ε)  — linear in velocity, less explosive than K0."""
    return _sev(class_name) * velocity_ms / max(distance_m, EPSILON)


def k3_quadratic_distance(distance_m: float, velocity_ms: float, class_name: str) -> float:
    """K3: sev × v² / (d² + ε)  — quadratic distance decay, wider safe zone."""
    return _sev(class_name) * (velocity_ms ** 2) / (distance_m ** 2 + EPSILON)


def k4_hybrid(distance_m: float, velocity_ms: float, class_name: str) -> float:
    """K4: 0.5×K0 + 0.5×(sev×clamp(10/TTC,0,10))  — momentum + time budget."""
    momentum = k0_current(distance_m, velocity_ms, class_name)
    if velocity_ms <= 0:
        ttc_term = 0.0
    else:
        ttc = distance_m / velocity_ms
        ttc_term = _sev(class_name) * min(10.0 / ttc, 10.0)
    return 0.5 * momentum + 0.5 * ttc_term


def k5_sigmoid(distance_m: float, velocity_ms: float, class_name: str) -> float:
    """K5: sev × sigmoid(v/max(d,ε) − β)  — bounded [0, sev]; β is calibration offset."""
    x = velocity_ms / max(distance_m, EPSILON) - SIGMOID_BETA
    return _sev(class_name) / (1.0 + math.exp(-x))


FORMULAS: dict[str, callable] = {
    "K0_current":          k0_current,
    "K1_ttc_primary":      k1_ttc_primary,
    "K2_linear_velocity":  k2_linear_velocity,
    "K3_quad_distance":    k3_quadratic_distance,
    "K4_hybrid":           k4_hybrid,
    "K5_sigmoid":          k5_sigmoid,
}


# ══════════════════════════════════════════════════════════════════════════════
# CSV Loading
# ══════════════════════════════════════════════════════════════════════════════

def _float_or_none(v: str) -> Optional[float]:
    return float(v) if v and v.strip() else None


def load_csv(csv_path: Path) -> list[dict]:
    """Load Stage 1 perception CSV; cast numeric fields; skip rows without distance."""
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            distance_m = _float_or_none(row.get("distance_m", ""))
            if distance_m is None:
                continue
            velocity_ms = float(row.get("velocity_ms") or 0.0)
            rows.append({
                "frame_idx":   int(row.get("frame_idx", 0)),
                "track_id":    row.get("track_id", ""),
                "class_name":  row.get("class", row.get("class_name", "unknown")),
                "distance_m":  distance_m,
                "velocity_ms": velocity_ms,
                # TTC from raw data if closing
                "ttc_s": distance_m / velocity_ms if velocity_ms > 0 else None,
            })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Statistics Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    idx = (len(sv) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (idx - lo)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Rank-based Spearman correlation."""
    if len(xs) < 2:
        return float("nan")
    def ranks(lst):
        order = sorted(range(len(lst)), key=lambda i: lst[i])
        r = [0.0] * len(lst)
        for rank, idx in enumerate(order):
            r[idx] = rank + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    return _pearson(rx, ry)


# ══════════════════════════════════════════════════════════════════════════════
# Routing Under Each Formula
# ══════════════════════════════════════════════════════════════════════════════

def route_label(k: float, ttc_s: Optional[float],
                low_k: float, high_k: float, reflex_ttc: float = 1.0) -> str:
    if ttc_s is not None and ttc_s <= reflex_ttc:
        return "reflex"
    if k >= high_k:
        return "reflex"
    if k >= low_k:
        return "cognitive"
    return "ignore"


# ══════════════════════════════════════════════════════════════════════════════
# Monotonicity Check
# ══════════════════════════════════════════════════════════════════════════════

def check_monotonicity(formula_fn: callable) -> dict:
    """Verify K increases as d decreases and v increases, ceteris paribus."""
    class_name = "car"
    distances   = [10.0, 5.0, 2.0, 1.0]   # decreasing
    velocities  = [1.0, 3.0, 5.0, 8.0]    # increasing
    fixed_v     = 3.0
    fixed_d     = 5.0

    d_scores = [formula_fn(d, fixed_v, class_name) for d in distances]
    v_scores = [formula_fn(fixed_d, v, class_name) for v in velocities]

    d_monotone = all(d_scores[i] <= d_scores[i+1] for i in range(len(d_scores)-1))
    v_monotone = all(v_scores[i] <= v_scores[i+1] for i in range(len(v_scores)-1))

    return {
        "d_decreasing_increases_K": d_monotone,
        "v_increasing_increases_K": v_monotone,
        "d_scores": [round(s, 3) for s in d_scores],
        "v_scores": [round(s, 3) for s in v_scores],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main Analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis(rows: list[dict], low_k: float, high_k: float) -> dict:
    results = {}
    all_ttc_inverses = [1.0 / r["ttc_s"] for r in rows if r["ttc_s"] is not None]

    for name, fn in FORMULAS.items():
        scores = [fn(r["distance_m"], r["velocity_ms"], r["class_name"]) for r in rows]

        # Distribution stats
        dist_stats = {
            "count":  len(scores),
            "mean":   round(_mean(scores), 4),
            "std":    round(_std(scores), 4),
            "p50":    round(_percentile(scores, 50), 4),
            "p95":    round(_percentile(scores, 95), 4),
            "p99":    round(_percentile(scores, 99), 4),
            "iqr":    round(_percentile(scores, 75) - _percentile(scores, 25), 4),
            "max":    round(max(scores), 4) if scores else 0,
        }

        # TTC correlation
        paired = [(scores[i], 1.0 / rows[i]["ttc_s"])
                  for i in range(len(rows)) if rows[i]["ttc_s"] is not None]
        if paired:
            ks, invttc = zip(*paired)
            pearson  = round(_pearson(list(ks), list(invttc)), 4)
            spearman = round(_spearman(list(ks), list(invttc)), 4)
        else:
            pearson = spearman = float("nan")

        # Routing counts
        routes = defaultdict(int)
        for i, row in enumerate(rows):
            lbl = route_label(scores[i], row["ttc_s"], low_k, high_k)
            routes[lbl] += 1
        total = len(rows)
        routing_pct = {k: round(100 * v / total, 1) for k, v in routes.items()}

        # Monotonicity
        mono = check_monotonicity(fn)

        results[name] = {
            "distribution": dist_stats,
            "ttc_correlation": {"pearson": pearson, "spearman": spearman},
            "routing_pct": routing_pct,
            "monotonicity": mono,
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Report Rendering
# ══════════════════════════════════════════════════════════════════════════════

def render_report(results: dict, rows: list[dict], low_k: float, high_k: float,
                  csv_path: Path) -> str:
    n = len(rows)
    lines = [
        "# CPE Kinetic Score Formula Comparison Report",
        "",
        f"**Input CSV:** `{csv_path.name}`  ",
        f"**Objects analysed:** {n}  ",
        f"**Routing thresholds:** low_k={low_k}, high_k={high_k}",
        "",
        "---",
        "",
        "## 1. Distribution Statistics",
        "",
        "| Formula | Mean | Std | P50 | P95 | P99 | Max | IQR |",
        "|---------|------|-----|-----|-----|-----|-----|-----|",
    ]
    for name, res in results.items():
        d = res["distribution"]
        lines.append(
            f"| {name} | {d['mean']} | {d['std']} | {d['p50']} | "
            f"{d['p95']} | {d['p99']} | {d['max']} | {d['iqr']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 2. TTC Correlation (higher = better alignment with time-to-collision urgency)",
        "",
        "> Pearson r and Spearman ρ between K_score and 1/TTC for rows where TTC is defined.",
        "",
        "| Formula | Pearson r | Spearman ρ | Verdict |",
        "|---------|-----------|------------|---------|",
    ]
    for name, res in results.items():
        tc = res["ttc_correlation"]
        p, s = tc["pearson"], tc["spearman"]
        verdict = "✅ Strong" if (not math.isnan(s) and s > 0.7) else \
                  "⚠️  Moderate" if (not math.isnan(s) and s > 0.4) else "❌ Weak"
        lines.append(f"| {name} | {p} | {s} | {verdict} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Routing Sensitivity (% of objects routed per lane)",
        "",
        "| Formula | Reflex % | Cognitive % | Ignore % |",
        "|---------|----------|-------------|----------|",
    ]
    for name, res in results.items():
        r = res["routing_pct"]
        lines.append(
            f"| {name} | {r.get('reflex', 0.0)}% | "
            f"{r.get('cognitive', 0.0)}% | {r.get('ignore', 0.0)}% |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Monotonicity Check",
        "",
        "Does K strictly increase as distance ↓ (at fixed v=3m/s) and velocity ↑ (at fixed d=5m)?",
        "",
        "| Formula | d↓ → K↑ | v↑ → K↑ | Verdict |",
        "|---------|---------|---------|---------|",
    ]
    for name, res in results.items():
        m = res["monotonicity"]
        d_ok = "✅" if m["d_decreasing_increases_K"] else "❌"
        v_ok = "✅" if m["v_increasing_increases_K"] else "❌"
        verdict = "✅ Monotone" if (m["d_decreasing_increases_K"] and m["v_increasing_increases_K"]) else "⚠️  Partial"
        lines.append(f"| {name} | {d_ok} | {v_ok} | {verdict} |")

    lines += [
        "",
        "---",
        "",
        "## 5. Recommendation",
        "",
        "Score each formula on 4 criteria (1 point each):",
        "- **TTC alignment** (Spearman ρ > 0.7)",
        "- **Monotonicity** (both d↓ and v↑ pass)",
        "- **Range stability** (P99/P50 ratio < 20 — avoids runaway scores)",
        "- **Routing balance** (reflex% between 5–25% on typical data)",
        "",
        "| Formula | TTC | Mono | Range | Balance | **Score** |",
        "|---------|-----|------|-------|---------|-----------|",
    ]
    scored = []
    for name, res in results.items():
        tc = res["ttc_correlation"]
        m  = res["monotonicity"]
        d  = res["distribution"]
        r  = res["routing_pct"]

        ttc_ok     = not math.isnan(tc["spearman"]) and tc["spearman"] > 0.7
        mono_ok    = m["d_decreasing_increases_K"] and m["v_increasing_increases_K"]
        range_ok   = d["p99"] / d["p50"] < 20 if d["p50"] > 0 else False
        balance_ok = 5.0 <= r.get("reflex", 0.0) <= 25.0

        score = sum([ttc_ok, mono_ok, range_ok, balance_ok])
        scored.append((name, score, ttc_ok, mono_ok, range_ok, balance_ok))

        def _icon(b): return "✅" if b else "❌"
        lines.append(
            f"| {name} | {_icon(ttc_ok)} | {_icon(mono_ok)} | "
            f"{_icon(range_ok)} | {_icon(balance_ok)} | **{score}/4** |"
        )

    best = max(scored, key=lambda x: x[1])
    lines += [
        "",
        f"> **Recommended formula:** `{best[0]}` scored **{best[1]}/4**.",
        "> Review the distribution plots in `evaluation/benchmarks/kinetic_score_plots/`",
        "> before committing this formula to `src/perception_stack/physics.py`.",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Optional Plot Generation
# ══════════════════════════════════════════════════════════════════════════════

def try_plot(results: dict, rows: list[dict], out_dir: Path) -> bool:
    """Attempt to generate matplotlib distribution + TTC scatter plots.
    Returns False if matplotlib is not installed (non-fatal)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [plots] matplotlib not available — skipping plots.")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. K-score distribution (box plot per formula)
    fig, ax = plt.subplots(figsize=(12, 5))
    all_scores = []
    labels = []
    for name, fn in FORMULAS.items():
        scores = [fn(r["distance_m"], r["velocity_ms"], r["class_name"]) for r in rows]
        # Clip to P99 for readability
        p99 = _percentile(scores, 99)
        clipped = [min(s, p99 * 1.5) for s in scores]
        all_scores.append(clipped)
        labels.append(name.replace("_", "\n"))

    ax.boxplot(all_scores, labels=labels, patch_artist=True,
               medianprops={"color": "black", "linewidth": 2})
    ax.set_title("K-Score Distribution per Formula (clipped at 1.5×P99)")
    ax.set_ylabel("Kinetic Score K")
    ax.set_xlabel("Formula")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "k_distribution_boxplot.png", dpi=150)
    plt.close(fig)

    # 2. TTC correlation scatter (one subplot per formula)
    paired_rows = [r for r in rows if r["ttc_s"] is not None]
    if paired_rows:
        inv_ttc = [1.0 / r["ttc_s"] for r in paired_rows]
        n_cols = 3
        n_rows = math.ceil(len(FORMULAS) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        for i, (name, fn) in enumerate(FORMULAS.items()):
            ks = [fn(r["distance_m"], r["velocity_ms"], r["class_name"]) for r in paired_rows]
            p99 = _percentile(ks, 99)
            axes[i].scatter(inv_ttc, ks, alpha=0.3, s=8, c="#4A90D9")
            axes[i].set_xlim(0, _percentile(inv_ttc, 99) * 1.2)
            axes[i].set_ylim(0, p99 * 1.5)
            axes[i].set_title(name)
            axes[i].set_xlabel("1/TTC (higher = more urgent)")
            axes[i].set_ylabel("K Score")
            r_val = _pearson(ks, inv_ttc)
            axes[i].text(0.05, 0.92, f"r={r_val:.3f}", transform=axes[i].transAxes,
                         fontsize=10, color="red")
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("K-Score vs 1/TTC Correlation (all formula candidates)", y=1.01)
        fig.tight_layout()
        fig.savefig(out_dir / "k_vs_invttc_scatter.png", dpi=150)
        plt.close(fig)

    # 3. Routing sensitivity bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(FORMULAS))
    width = 0.25
    names = list(results.keys())
    reflex_pct  = [results[n]["routing_pct"].get("reflex", 0.0) for n in names]
    cog_pct     = [results[n]["routing_pct"].get("cognitive", 0.0) for n in names]
    ignore_pct  = [results[n]["routing_pct"].get("ignore", 0.0) for n in names]
    ax.bar(x - width, reflex_pct,  width, label="Reflex",    color="#E74C3C")
    ax.bar(x,         cog_pct,    width, label="Cognitive",  color="#F39C12")
    ax.bar(x + width, ignore_pct, width, label="Ignore",     color="#2ECC71")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("% of objects routed")
    ax.set_title(f"Routing Lane Distribution per Formula (low_k={args_cache['low_k']}, high_k={args_cache['high_k']})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "routing_sensitivity.png", dpi=150)
    plt.close(fig)

    return True


# ── Global cache for plot title (avoids passing args into plot fn) ────────────
args_cache: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPE Kinetic Score Formula Comparison — K0 through K5."
    )
    parser.add_argument("--csv",      type=Path, required=True,
                        help="Stage 1 perception CSV path.")
    parser.add_argument("--fps",      type=float, default=30.0,
                        help="Source video framerate (unused in analysis but logged).")
    parser.add_argument("--low-k",   type=float, default=DEFAULT_LOW_K,
                        help="K threshold for cognitive routing (default: 0.5).")
    parser.add_argument("--high-k",  type=float, default=DEFAULT_HIGH_K,
                        help="K threshold for reflex routing (default: 5.0).")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("evaluation/benchmarks/kinetic_score_eval"),
                        help="Output directory for report and plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args_cache.update({"low_k": args.low_k, "high_k": args.high_k})

    print(f"\n=== CPE Kinetic Score Formula Comparison ===")
    print(f"  Input CSV  : {args.csv}")
    print(f"  Thresholds : low_k={args.low_k}, high_k={args.high_k}")
    print(f"  Output dir : {args.out_dir}\n")

    # Load data
    print("Loading perception CSV...")
    rows = load_csv(args.csv)
    print(f"  → {len(rows)} objects with valid distance loaded.\n")
    if not rows:
        print("ERROR: No valid rows found. Check that 'distance_m' column is populated.")
        sys.exit(1)

    # Run analysis
    print("Running formula analysis...")
    results = run_analysis(rows, args.low_k, args.high_k)

    # Render report
    report_md = render_report(results, rows, args.low_k, args.high_k, args.csv)

    # Save outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "kinetic_score_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"  ✅ Markdown report saved: {report_path}")

    results_path = args.out_dir / "kinetic_score_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  ✅ JSON results saved:    {results_path}")

    # Plots
    plot_dir = args.out_dir / "plots"
    plotted = try_plot(results, rows, plot_dir)
    if plotted:
        print(f"  ✅ Plots saved to:        {plot_dir}/")

    # Print summary to terminal
    print("\n── Summary Table ──────────────────────────────────────────")
    print(f"{'Formula':<25} {'P50':>7} {'P95':>7} {'P99':>7} {'Spearman':>10} {'Reflex%':>8}")
    print("─" * 70)
    for name, res in results.items():
        d = res["distribution"]
        tc = res["ttc_correlation"]
        r  = res["routing_pct"]
        spear = f"{tc['spearman']:.3f}" if not math.isnan(tc["spearman"]) else "  N/A"
        print(f"{name:<25} {d['p50']:>7} {d['p95']:>7} {d['p99']:>7} {spear:>10} {r.get('reflex', 0.0):>7}%")
    print("─" * 70)
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
