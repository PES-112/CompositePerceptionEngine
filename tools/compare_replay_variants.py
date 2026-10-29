"""
compare_replay_variants.py
==========================
One table from several tools/replay_threat_engine.py runs (one arm each, same val/test sessions),
for the early-warning study (methodology §4.23).

Selection rule, declared before any result was read (2026-10-25): among the variants, choose the
one with the highest *validation* warned_before (warned ≥ 0.2 s ahead) at its own validation-chosen
early_budget_5 operating point; ties → fewer validation false alerts/min. Test is then read once,
for every variant, so nothing is hidden. Paired differences of the chosen variant against the
baseline come from one more replay with both arms and --paired (methodology §4.23).

    python tools/compare_replay_variants.py --root data/replay_423 --base base:v2_gru \\
        --out evaluation/benchmarks/threat_engine_replay/early_warning_2026_10_25
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

RULES = ("early_budget_5", "budget_5", "budget_10", "crc_budget_5", "crc_budget_10")
COLS = ("warned", "warned_before", "warned_lead_ge1", "false_per_min", "false_per_min_walk", "precision",
        "late_share", "switches_per_min")


def val_row(run: Path, arm: str, knob: str) -> dict:
    with (run / "curves.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "val" and r["arm"] == arm and r["knob"] == knob:
                return {k: float(v) for k, v in r.items() if k not in ("split", "arm", "knob")}
    raise KeyError((run, arm, knob))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--base", default="base:v2_gru", help="run:arm of the baseline.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    variants = {}
    for run in sorted(d for d in args.root.iterdir() if (d / "metrics.json").exists()):
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        for arm, by_rule in m["chosen"].items():
            variants[f"{run.name}:{arm}"] = (run, arm, by_rule, m["points"][arm])

    def val_at(name: str, rule: str) -> dict | None:
        run, arm, by_rule, _ = variants[name]
        return val_row(run, arm, by_rule[rule]) if by_rule.get(rule) else None

    scored = {n: val_at(n, "early_budget_5") for n in variants}
    eligible = {n: v for n, v in scored.items() if v is not None}
    chosen = max(eligible, key=lambda n: (eligible[n]["warned_before"], -eligible[n]["false_per_min"]))

    def cell(v):
        if v is None:
            return "n/a"
        if isinstance(v, (list, tuple)):
            return f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" if len(v) == 3 else f"{v[0]:.2f}"
        return f"{v:.2f}"

    lines = ["# Early-warning study: GRU objectives, fused inputs, rising trigger, CRC budgets", "",
             f"Runs under `{args.root.as_posix()}` (tools/replay_threat_engine.py, one arm per run, the same 48 "
             "validation / 85 test sessions, seed-0 checkpoints). Baseline: "
             f"`{args.base}`. Selection rule (declared first): highest validation warned_before at each "
             "variant's validation-chosen `early_budget_5` point.", "",
             f"**Chosen on validation: `{chosen}`.**", "",
             "## Validation, at each variant's `early_budget_5` point (selection data)", "",
             "| Variant | knob | warned | warned ≥ 0.2 s | false/min | precision | late share |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for n, (_, _, by_rule, _) in variants.items():
        v = scored[n]
        lines.append(f"| {n} | {by_rule.get('early_budget_5') or '—'} | " +
                     (" | ".join(f"{v[k]:.2f}" for k in ("warned", "warned_before", "false_per_min", "precision",
                                                          "late_share")) if v else "no point meets the budget | | | |")
                     + " |")
    for rule in RULES:
        lines += ["", f"## Test, `{rule}` (each variant at its validation-chosen knob; 95% session-bootstrap CI)", "",
                  "| Variant | knob | " + " | ".join(COLS) + " |", "|---|---|" + "---|" * len(COLS)]
        for n, (_, _, by_rule, points) in variants.items():
            pt = points.get(rule)
            if pt is None:
                lines.append(f"| {n} | — | no setting meets the rule on validation |" + " |" * (len(COLS) - 1))
                continue
            lines.append(f"| {n} | {by_rule[rule]} | " + " | ".join(cell(pt.get(k)) for k in COLS) + " |")
    lines += ["", "warned_before = warned ≥ 0.2 s before arrival; warned_lead_ge1 = ≥ 1 s. false_per_min pools all "
              "walks; false_per_min_walk is the mean of each walk's own rate, which `crc_budget_*` bounds in "
              "expectation. Open-loop replay; labels and inputs share SANPO's depth.", ""]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps({
        "chosen": chosen, "rule": "max val warned_before at early_budget_5",
        "val_early_budget_5": scored,
        "test": {n: {r: v[3].get(r) for r in RULES} for n, v in variants.items()},
        "knobs": {n: v[2] for n, v in variants.items()}}, indent=2, default=str), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
