"""
analyze_timing.py
=================
Summarise CSVs written by the CPE Timing app (ios/CPETiming) into the numbers
the paper reports (docs/methodology.md §5):

  per stage (pre / det / det_post / gru / total): median, p95, p99, with ARKit off and on,
      warm-up excluded; blocks alternate ARKit off/on, so the difference is ARKit's load
  sustained run: p95 of total per 5-minute window, time in each thermal state per window,
      battery drop per hour, ARKit frame rate

    python ios/CPETiming/analyze_timing.py cpe_timing_benchmark_*.csv cpe_timing_sustained30min_*.csv
        --out evaluation/benchmarks/ondevice_latency/<device>_<date>

The total column is what tools/replay_threat_engine.py --latency-csv adds to
alert timing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STAGES = ("pre_ms", "det_ms", "det_post_ms", "gru_ms", "total_ms")
THERMAL = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}
WINDOW_S = 300


def load(path: Path) -> tuple[pd.DataFrame, str]:
    header = path.read_text(encoding="utf-8").splitlines()[0]
    df = pd.read_csv(path, comment="#")
    return df[df["block"] != "warmup"].copy(), header.lstrip("# ")


def stage_table(df: pd.DataFrame) -> list[str]:
    lines = ["| stage | ARKit | n | median ms | p95 ms | p99 ms |", "|---|---|---:|---:|---:|---:|"]
    for stage in STAGES:
        for ar, g in df.groupby("ar_on"):
            v = g[stage].to_numpy()
            lines.append(f"| {stage.removesuffix('_ms')} | {'on' if ar else 'off'} | {len(v)} | "
                         f"{np.median(v):.2f} | {np.percentile(v, 95):.2f} | {np.percentile(v, 99):.2f} |")
    return lines


def sustained_table(df: pd.DataFrame) -> list[str]:
    df = df.assign(window=(df["t_s"] - df["t_s"].min()) // WINDOW_S)
    lines = ["| window (min) | n | median total ms | p95 total ms | thermal (share of frames) | battery | ARKit fps |",
             "|---|---:|---:|---:|---|---:|---:|"]
    for w, g in df.groupby("window"):
        shares = ", ".join(f"{THERMAL.get(int(k), k)} {v:.0%}" for k, v in g["thermal_state"].value_counts(normalize=True).sort_index().items())
        lines.append(f"| {int(w) * 5}-{int(w) * 5 + 5} | {len(g)} | {g['total_ms'].median():.2f} | "
                     f"{np.percentile(g['total_ms'], 95):.2f} | {shares} | {g['battery'].iloc[-1]:.2f} | "
                     f"{g['ar_fps'].median():.0f} |")
    hours = (df["t_s"].max() - df["t_s"].min()) / 3600
    valid = df[df["battery"] >= 0]
    if hours > 0 and len(valid) > 1:
        lines += ["", f"Battery: {valid['battery'].iloc[0]:.2f} -> {valid['battery'].iloc[-1]:.2f} "
                      f"({(valid['battery'].iloc[0] - valid['battery'].iloc[-1]) / hours:.0%} per hour)."]
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lines = ["# On-device latency (CPE Timing app)", ""]
    for path in args.csv:
        df, header = load(path)
        lines += [f"## `{path.name}`", "", f"`{header}`", ""]
        lines += stage_table(df) + [""]
        if (df["block"] == "sustained").any():
            lines += ["### Sustained run, by 5-minute window", ""] + sustained_table(df[df["block"] == "sustained"]) + [""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
