# Kinetic Score Evaluation Benchmark

This folder holds results from running the kinetic score formula comparison
and threat score routing evaluation scripts over SANPO perception CSVs.

## Contents

| File / Folder | Description |
|---|---|
| `kinetic_score_report.md` | Ranked comparison of K0–K5 formulas: distribution stats, TTC correlation, routing sensitivity, monotonicity. |
| `kinetic_score_results.json` | Machine-readable results (same data as the report). |
| `threat_score_report.md` | Per-route (reflex/cognitive/ignore) Precision, Recall, F1 for each formula using K₊₂ ground truth. |
| `threat_score_results.json` | Machine-readable routing evaluation results. |
| `plots/` | Matplotlib figures: K-score box plots, K vs 1/TTC scatter, routing sensitivity bar chart. |

## How to Reproduce

```powershell
# Step 1 — K-score distribution + TTC correlation
python evaluation/kinetic_score_comparison.py `
    --csv   data/processed/merged_perception.csv `
    --fps   30 `
    --low-k  0.5 `
    --high-k 5.0 `
    --out-dir evaluation/benchmarks/kinetic_score_eval

# Step 2 — Routing F1 / Precision / Recall
python evaluation/threat_score_eval.py `
    --csv         data/processed/merged_perception.csv `
    --fps         30 `
    --lookahead-s 2.0 `
    --low-k        0.5 `
    --high-k       5.0 `
    --out-dir evaluation/benchmarks/kinetic_score_eval
```

## Decision Criteria

1. **Highest Reflex Recall** — a missed reflex = safety failure.
2. **Spearman ρ > 0.7** with 1/TTC — score aligns with urgency.
3. **Monotone** — K strictly increases as d↓ and v↑.
4. **Range stability** — P99/P50 < 20 (no runaway unbounded scores).

The winning formula replaces `kinetic_score()` in
`src/perception_stack/physics.py` after manual review of these results.
