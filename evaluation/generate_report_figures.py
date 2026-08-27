"""Regenerate every plot and summary table from the current benchmark artifacts.

Reads only committed, compact benchmark artifacts under `evaluation/benchmarks/` — never raw
per-session dumps — and writes PNGs plus a markdown table digest to
`evaluation/benchmarks/figures/`. Safe to delete that output directory and re-run.

Usage:
    .venv/bin/python evaluation/generate_report_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "evaluation" / "benchmarks"
OUT_DIR = BENCH_DIR / "figures"

VERSION_COMPARISON_DIR = BENCH_DIR / "yolo26n_version_comparison"
EDGE_REALTIME_DIR = BENCH_DIR / "sanpo_edge_realtime" / "ten_session_v3_jetson_orin_nano_8gb"
KINETIC_RUN_DIR = BENCH_DIR / "kinetic_score_eval" / "run_2026_08_26"
# Ground truth as of 2026-08-27: metrics.json/disagreements_key.json were copied from the
# DGX Spark and committed (git log: d9f4330). ablation_summary.csv (a hand transcription of
# docs/ablation_guide.md's table, written before the raw files existed) is kept only as a
# fallback for anyone running this script against an older checkout.
KINETIC_METRICS_JSON = KINETIC_RUN_DIR / "metrics.json"
KINETIC_DISAGREEMENTS_KEY = KINETIC_RUN_DIR / "disagreements_key.json"
KINETIC_ABLATION_CSV = KINETIC_RUN_DIR / "ablation_summary.csv"

# Struck arms (lam=0.25/lam=1.0) are real numbers from an earlier script version that can't
# be reproduced from the currently committed evaluation/kinetic_ablation.py — see the
# [!NOTE] in docs/ablation_guide.md §6. Plotted, but visually distinguished, not hidden.
KINETIC_VERDICTS = {
    "K0  sev·v²/d": "baseline",
    "linear  sev·v/d": "ties",
    "no-severity  v²/d": "ties",
    "no-velocity  sev/d": "loses",
    "size  sev·v²·s^½/d": "ties",
    "lam=0.25  weak mass": "struck",
    "lam=1.0  full KE": "struck",
    "ttc  -(d-D)/v": "loses",
}
KINETIC_SHORT_NAMES = {
    "K0  sev·v²/d": "K0", "linear  sev·v/d": "linear", "no-severity  v²/d": "no-severity",
    "no-velocity  sev/d": "no-velocity", "size  sev·v²·s^½/d": "size",
    "lam=0.25  weak mass": "lam=0.25", "lam=1.0  full KE": "lam=1.0", "ttc  -(d-D)/v": "ttc",
}

PALETTE = {
    "v1": "#8c8c8c",
    "v2": "#e07b39",
    "v3": "#2f6fed",
    "native": "#2f6fed",
    "simulated": "#e07b39",
    "loses": "#d64545",
    "ties": "#8c8c8c",
    "baseline": "#2f6fed",
    "struck": "#c9b8e0",
}
BUDGET_COLOR = "#d64545"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
    }
)


def _save(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")
    return path


def plot_yolo_version_map() -> Path | None:
    csv_path = VERSION_COMPARISON_DIR / "overall_version_metrics.csv"
    if not csv_path.exists():
        print(f"skip: {csv_path} not found")
        return None
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(df))
    width = 0.35
    ax.bar(
        [i - width / 2 for i in x],
        df["map50"],
        width=width,
        label="mAP50",
        color="#2f6fed",
    )
    ax.bar(
        [i + width / 2 for i in x],
        df["map50_95"],
        width=width,
        label="mAP50-95",
        color="#9fbcf5",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["version"])
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 1)
    ax.set_title("YOLO26n hazard detector: held-out mAP by version")
    for i, (m50, m5095) in enumerate(zip(df["map50"], df["map50_95"])):
        ax.text(i - width / 2, m50 + 0.02, f"{m50:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, m5095 + 0.02, f"{m5095:.3f}", ha="center", fontsize=8)
    ax.legend()
    return _save(fig, "yolo_version_map.png")


def plot_yolo_per_class_map50() -> Path | None:
    csv_path = VERSION_COMPARISON_DIR / "per_class_version_metrics.csv"
    if not csv_path.exists():
        print(f"skip: {csv_path} not found")
        return None
    df = pd.read_csv(csv_path).sort_values("v3_map50", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 8))
    y = range(len(df))
    ax.barh(
        [i + 0.27 for i in y], df["v1_map50"], height=0.27, label="v1", color=PALETTE["v1"]
    )
    ax.barh(
        [i for i in y], df["v2_map50"], height=0.27, label="v2", color=PALETTE["v2"]
    )
    ax.barh(
        [i - 0.27 for i in y], df["v3_map50"], height=0.27, label="v3", color=PALETTE["v3"]
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["class_name"])
    ax.set_xlabel("mAP50")
    ax.set_xlim(0, 1)
    ax.set_title("Per-class mAP50 across detector versions\n(blank = class not evaluated in that version)")
    ax.legend(loc="lower right")
    return _save(fig, "yolo_per_class_map50.png")


def plot_yolo_retention_f1_delta() -> Path | None:
    csv_path = VERSION_COMPARISON_DIR / "retention_vs_base_metrics.csv"
    if not csv_path.exists():
        print(f"skip: {csv_path} not found")
        return None
    df = pd.read_csv(csv_path)
    v2 = df[df["version"] == "v2"].set_index("class_name")["delta_f1_vs_base"]
    v3 = df[df["version"] == "v3"].set_index("class_name")["delta_f1_vs_base"]
    classes = list(v3.index)

    fig, ax = plt.subplots(figsize=(7, 5))
    y = range(len(classes))
    ax.barh(
        [i + 0.2 for i in y],
        [v2.get(c, 0) for c in classes],
        height=0.35,
        label="v2 vs base",
        color=PALETTE["v2"],
    )
    ax.barh(
        [i - 0.2 for i in y],
        [v3.get(c, 0) for c in classes],
        height=0.35,
        label="v3 vs base",
        color=PALETTE["v3"],
    )
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(classes)
    ax.set_xlabel("F1 delta vs. base YOLO26n (retained COCO classes)")
    ax.set_title("Retained-class regression check: v2 vs. v3 (v3 restarted from base)")
    ax.legend()
    return _save(fig, "yolo_retention_f1_delta.png")


def plot_edge_latency_sessions() -> Path | None:
    json_path = EDGE_REALTIME_DIR / "aggregate_summary.json"
    if not json_path.exists():
        print(f"skip: {json_path} not found")
        return None
    data = json.loads(json_path.read_text())
    sessions = data["sessions"]
    labels = [s["session_id"][:8] for s in sessions]
    p95 = [s["simulated_p95_ms"] for s in sessions]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, p95, color=PALETTE["simulated"])
    ax.axhline(50, color=BUDGET_COLOR, linestyle="--", linewidth=1.5, label="50 ms reflex budget")
    for bar, val in zip(bars, p95):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.6, f"{val:.1f}", ha="center", fontsize=8)
    ax.set_ylabel("Simulated Jetson Orin Nano 8GB p95 latency (ms)")
    ax.set_xlabel("SANPO session (first 8 chars of ID)")
    ax.set_title("Per-session edge latency vs. reflex budget (simulated, not physical hardware)")
    ax.set_ylim(0, 60)
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, "edge_latency_sessions.png")


def plot_edge_latency_native_vs_sim() -> Path | None:
    json_path = EDGE_REALTIME_DIR / "aggregate_summary.json"
    if not json_path.exists():
        print(f"skip: {json_path} not found")
        return None
    data = json.loads(json_path.read_text())

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["mean", "p95"]
    native = [data["avg_native_mean_ms"], data["avg_native_p95_ms"]]
    simulated = [data["avg_simulated_mean_ms"], data["avg_simulated_p95_ms"]]
    x = range(len(labels))
    width = 0.35
    ax.bar(
        [i - width / 2 for i in x], native, width=width, label="Native GB10 (measured)",
        color=PALETTE["native"],
    )
    ax.bar(
        [i + width / 2 for i in x], simulated, width=width,
        label="Simulated Jetson (analytical proxy)", color=PALETTE["simulated"],
    )
    ax.axhline(50, color=BUDGET_COLOR, linestyle="--", linewidth=1.2, label="50 ms reflex budget")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms), avg over 10 sessions")
    ax.set_title("Native measurement vs. simulated edge estimate")
    ax.legend(fontsize=8)
    return _save(fig, "edge_latency_native_vs_sim.png")


def _load_kinetic_metrics() -> dict | None:
    if KINETIC_METRICS_JSON.exists():
        return json.loads(KINETIC_METRICS_JSON.read_text())
    return None


def plot_kinetic_ablation_metrics() -> Path | None:
    metrics = _load_kinetic_metrics()
    if metrics is None:
        print(f"skip: {KINETIC_METRICS_JSON} not found")
        return None

    arms = list(KINETIC_SHORT_NAMES)
    labels = [KINETIC_SHORT_NAMES[a] for a in arms]
    colors = [PALETTE[KINETIC_VERDICTS[a]] for a in arms]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    panels = [
        ("flicker_rate", "Flicker rate (lower is better)", (0, 1.05)),
        ("rank_stability_10pct", "Rank stability @ 10% depth noise\n(Kendall τ, higher is better)", (0, 1.05)),
        ("encounter_top1", "Encounter top-1 recall\n(higher is better)", (0, 0.5)),
    ]
    for ax, (metric, title, ylim) in zip(axes, panels):
        point = [metrics[a][metric][0] for a in arms]
        lo = [metrics[a][metric][1] for a in arms]
        hi = [metrics[a][metric][2] for a in arms]
        yerr = [[p - l for p, l in zip(point, lo)], [h - p for p, h in zip(point, hi)]]
        ax.bar(labels, point, color=colors, yerr=yerr, capsize=3,
               error_kw={"linewidth": 1, "ecolor": "#444444"})
        ax.set_title(title)
        ax.set_ylim(*ylim)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["baseline"]),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["ties"]),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["loses"]),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["struck"]),
    ]
    fig.legend(
        handles, ["K0 (baseline)", "ties with K0", "loses to K0", "struck (unreproducible, see docs)"],
        loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=9,
    )
    fig.suptitle(
        "K0 term ablation — 139 SANPO sessions, 19,402 frames (2026-08-26)\n"
        "error bars: 95% session-level bootstrap CI",
        y=1.17, fontweight="bold",
    )
    return _save(fig, "kinetic_ablation_metrics.png")


def _df_to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:{floatfmt}}"
        return str(v)

    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def write_results_summary_md(written: list[Path]) -> None:
    lines = [
        "# Results Summary",
        "",
        "Auto-generated by `evaluation/generate_report_figures.py`. Do not hand-edit — re-run the",
        "script after any benchmark artifact changes instead.",
        "",
        "## Figures",
        "",
    ]
    for path in written:
        if path is None:
            continue
        lines.append(f"- `{path.relative_to(REPO_ROOT)}`")
    lines += [
        "",
        "## Detector version comparison",
        "",
    ]
    csv_path = VERSION_COMPARISON_DIR / "overall_version_metrics.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)[["version", "map50", "map50_95", "map75"]]
        lines.append(_df_to_markdown(df))
    lines += [
        "",
        "## Edge latency (10-session SANPO simulation)",
        "",
    ]
    json_path = EDGE_REALTIME_DIR / "aggregate_summary.json"
    if json_path.exists():
        data = json.loads(json_path.read_text())
        lines.append(
            "| Metric | Native GB10 (measured) | Simulated Jetson Orin Nano 8GB |\n"
            "|---|---:|---:|\n"
            f"| Avg mean latency | {data['avg_native_mean_ms']:.2f} ms | {data['avg_simulated_mean_ms']:.2f} ms |\n"
            f"| Avg p95 latency | {data['avg_native_p95_ms']:.2f} ms | {data['avg_simulated_p95_ms']:.2f} ms |\n"
            f"| Worst-session p95 | {data['worst_native_p95_ms']:.2f} ms | {data['worst_simulated_p95_ms']:.2f} ms |\n"
        )
    lines += [
        "",
        "## Kinetic score ablation verdicts",
        "",
        "Source: `evaluation/benchmarks/kinetic_score_eval/run_2026_08_26/metrics.json` "
        "(committed 2026-08-27, copied from the DGX Spark run — see `docs/ablation_guide.md` §6).",
        "",
    ]
    metrics = _load_kinetic_metrics()
    if metrics is not None:
        rows = []
        for arm, short in KINETIC_SHORT_NAMES.items():
            m = metrics[arm]
            rows.append({
                "arm": short,
                "flicker": m["flicker_rate"][0],
                "flicker_ci": f"[{m['flicker_rate'][1]:.3f}, {m['flicker_rate'][2]:.3f}]",
                "rank_stab_10pct": m["rank_stability_10pct"][0],
                "encounter_top1": m["encounter_top1"][0],
                "verdict": KINETIC_VERDICTS[arm],
            })
        lines.append(_df_to_markdown(pd.DataFrame(rows)))
    elif KINETIC_ABLATION_CSV.exists():
        df = pd.read_csv(KINETIC_ABLATION_CSV)[["arm", "formula", "flicker", "rank_stab_10pct", "encounter_top1", "verdict"]]
        lines.append(_df_to_markdown(df))

    if KINETIC_DISAGREEMENTS_KEY.exists():
        key = json.loads(KINETIC_DISAGREEMENTS_KEY.read_text())
        counts: dict[str, int] = {}
        for entry in key:
            counts[entry["variant"]] = counts.get(entry["variant"], 0) + 1
        lines += [
            "",
            "### Disagreement frames exported per arm (of 219 total, capped at 60/arm)",
            "",
            "How often each arm's top pick differed from K0's, out of 19,402 scored frames — "
            "the frames the VLM referee (`evaluation/vlm_referee.py`) still needs to adjudicate.",
            "",
            "| Arm | Disagreement frames exported |",
            "|---|---:|",
        ]
        for arm, short in KINETIC_SHORT_NAMES.items():
            if arm == "K0  sev·v²/d":
                continue
            lines.append(f"| {short} | {counts.get(arm, 0)} |")
    out_path = OUT_DIR / "results_summary.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    written = [
        plot_yolo_version_map(),
        plot_yolo_per_class_map50(),
        plot_yolo_retention_f1_delta(),
        plot_edge_latency_sessions(),
        plot_edge_latency_native_vs_sim(),
        plot_kinetic_ablation_metrics(),
    ]
    write_results_summary_md([p for p in written if p is not None])


if __name__ == "__main__":
    main()
