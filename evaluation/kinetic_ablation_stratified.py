"""
kinetic_ablation_stratified.py
===============================
Does severity matter *in the scenes it was designed for*?

evaluation/kinetic_ablation.py's corpus-wide metrics tied for the no-severity and
linear arms (docs/ablation_guide.md §6). That is expected if v² dominates most
frames — but it cannot distinguish "severity never matters" from "severity matters
only in the minority of frames where a high-mass and low-mass object are closing at
similar rates, and that minority is diluted into 19,402 mostly-unambiguous frames."

This script isolates exactly that minority — frames with at least two objects whose
severity differs meaningfully (§ physics.CLASS_SEVERITY) *and* whose kinematics
(v/d) are close enough that severity is the only thing that could break the tie —
and re-runs the same label-free metrics (imported, not reimplemented, from
kinetic_ablation.py) restricted to that subset. If K0 still ties no-severity even
here, that is real evidence severity is low-impact. If it doesn't, the corpus-wide
tie was a dilution artifact, not a null result.

Input : the same per-session CSV directory as kinetic_ablation.py.
Output: stratified_report.md, stratified_metrics.json — a coverage count (how many
        severity-discriminating frames exist at all) plus K0 vs. no-severity vs.
        linear restricted to that subset.

Usage
-----
    python evaluation/kinetic_ablation_stratified.py \\
        --csv-dir data/processed/ablation_30pct \\
        --out-dir evaluation/benchmarks/kinetic_score_eval/run_2026_08_26
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.download_sanpo_valid_streams import (  # noqa: E402
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SESSION_FRACTION,
    DEFAULT_VALID_STREAMS,
    sampled_session_ids,
)

# Load kinetic_ablation.py directly (not as a package import) so this script has
# no dependency beyond what kinetic_ablation.py itself needs, and so the metric
# math is guaranteed to be the literal same code, not a reimplementation.
_ka_spec = importlib.util.spec_from_file_location(
    "cpe_kinetic_ablation", Path(__file__).resolve().parent / "kinetic_ablation.py"
)
ka = importlib.util.module_from_spec(_ka_spec)
_ka_spec.loader.exec_module(ka)

_phys_spec = importlib.util.spec_from_file_location(
    "cpe_physics", PROJECT_ROOT / "src/perception_stack/physics.py"
)
physics = importlib.util.module_from_spec(_phys_spec)
_phys_spec.loader.exec_module(physics)

# The three arms where severity is the variable under test. `ttc` and
# `no-velocity` already lose decisively on the full corpus (docs/ablation_guide.md
# §6) and aren't re-examined here — this script is specifically about the arms
# that tied.
STRATIFY_VARIANTS = {
    name: ka.VARIANTS[name]
    for name in ("K0  sev·v²/d", "no-severity  v²/d", "linear  sev·v/d")
}

DEFAULT_SEVERITY_RATIO = 1.5   # objects must differ by >= this factor in severity
DEFAULT_KINEMATIC_TOL = 0.15   # ...while their v/d differs by <= this fraction


def _severity(class_name: str) -> float:
    return physics.CLASS_SEVERITY.get(class_name, 1.0)


def is_severity_discriminating(
    g: pd.DataFrame, severity_ratio: float, kinematic_tol: float, eps: float = 1e-6,
) -> bool:
    """
    True if this frame contains at least one pair of objects where severity
    differs by >= severity_ratio but v/d (the kinematic term K0 shares with every
    arm) is within kinematic_tol of each other — i.e. a pair that kinematics alone
    would call a near-tie, so severity is the only thing available to break it.
    This is the literal scenario severity's design intent (kinetic_score_opinion.md
    §"K0's v² is a bet...") targets: a bus and a pedestrian arriving in a
    comparable window should not be scored as equal threats.
    """
    sev = g["class"].map(_severity).to_numpy(dtype=float)
    voverd = (g["velocity_ms"].to_numpy(dtype=float)
              / np.maximum(g["distance_m"].to_numpy(dtype=float), eps))
    n = len(g)
    for i in range(n):
        for j in range(i + 1, n):
            lo, hi = sorted((sev[i], sev[j]))
            if lo <= eps or hi / lo < severity_ratio:
                continue
            base = max(abs(voverd[i]), abs(voverd[j]), eps)
            if abs(voverd[i] - voverd[j]) / base <= kinematic_tol:
                return True
    return False


def stratified_groups(
    df: pd.DataFrame, severity_ratio: float, kinematic_tol: float,
) -> list[tuple[int, pd.DataFrame]]:
    return [
        (idx, g) for idx, g in ka.frame_groups(df)
        if is_severity_discriminating(g, severity_ratio, kinematic_tol)
    ]


def disagreement_rate(
    groups: list[tuple[int, pd.DataFrame]], scorer_a, scorer_b,
) -> tuple[int, int]:
    """(disagreements, total) — how often two variants' argmax differs on `groups`."""
    disagree = 0
    total = 0
    for _, g in groups:
        a = scorer_a(g)
        b = scorer_b(g)
        if not (np.isfinite(a).any() and np.isfinite(b).any()):
            continue
        total += 1
        if g.iloc[int(np.argmax(a))]["track_id"] != g.iloc[int(np.argmax(b))]["track_id"]:
            disagree += 1
    return disagree, total


def write_report(
    out_dir: Path, n_sessions: int, total_frames: int, strat_frames: int,
    per_variant: dict, disagreements: dict, severity_ratio: float, kinematic_tol: float,
) -> None:
    coverage_pct = 100.0 * strat_frames / max(1, total_frames)
    lines = [
        "# Stratified Kinetic Score Check: Severity-Discriminating Scenes Only",
        "",
        f"Sessions: {n_sessions} · total scored frames (≥2 objects): {total_frames} · "
        f"severity-discriminating frames: {strat_frames} ({coverage_pct:.2f}% of total)",
        "",
        f"A frame qualifies if it has >= 2 objects whose class severity differs by "
        f">= {severity_ratio}x while their v/d differs by <= {kinematic_tol * 100:.0f}% "
        "— i.e. kinematics alone would call it a near-tie, so severity is the only "
        "thing that can break it. This is the coverage number the corpus-wide "
        "ablation in kinetic_ablation.py cannot report: how often severity is even "
        "*given the chance* to matter.",
        "",
        "**Read this number first.** If coverage is small, a corpus-wide tie on "
        "no-severity is expected regardless of whether severity is well-calibrated "
        "— it would be diluted below detection either way. If coverage is not "
        "small and the metrics below still tie, that is real evidence severity is "
        "low-impact even where it has a chance to matter.",
        "",
        "## Argmax disagreement rate (K0 vs. each arm)",
        "",
        "| Comparison | On severity-discriminating frames | On the full corpus |",
        "|---|---:|---:|",
    ]
    for name, (strat_d, strat_t, full_d, full_t) in disagreements.items():
        strat_pct = 100.0 * strat_d / max(1, strat_t)
        full_pct = 100.0 * full_d / max(1, full_t)
        lines.append(f"| K0 vs. `{name}` | {strat_pct:.1f}% ({strat_d}/{strat_t}) | "
                      f"{full_pct:.1f}% ({full_d}/{full_t}) |")
    lines += ["", "## Label-free metrics, restricted to severity-discriminating frames", ""]
    for metric, note in ka.METRIC_NOTES.items():
        lines += [f"### {metric}", "", f"_{note}_", "",
                  "| Variant | Value | 95% CI |", "|---|---:|---|"]
        for name in STRATIFY_VARIANTS:
            p, lo, hi = per_variant[name][metric]
            lines.append(f"| `{name}` | {p:.3f} | [{lo:.3f}, {hi:.3f}] |")
        lines.append("")
    lines += [
        "## What this does not answer",
        "",
        "Whether severity picks the *right* object even when it changes the argmax "
        "is still a value judgment — run the blinded referee "
        "(`evaluation/vlm_referee.py`) on the disagreement frames listed above, not "
        "just this script, before concluding severity is correct as calibrated.",
        "",
    ]
    (out_dir / "stratified_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--valid-streams", type=Path, default=DEFAULT_VALID_STREAMS)
    p.add_argument("--session-fraction", type=float, default=DEFAULT_SESSION_FRACTION)
    p.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    p.add_argument("--severity-ratio", type=float, default=DEFAULT_SEVERITY_RATIO,
                   help="Minimum severity ratio between two objects to count as "
                        "severity-discriminating (default 1.5x).")
    p.add_argument("--kinematic-tol", type=float, default=DEFAULT_KINEMATIC_TOL,
                   help="Maximum relative v/d difference for two objects to count "
                        "as kinematically tied (default 15%%).")
    args = p.parse_args()

    csv_paths = sorted(pth for pth in args.csv_dir.glob("*.csv") if not pth.name.endswith(".partial"))
    if not csv_paths:
        raise SystemExit(f"No session CSVs in {args.csv_dir} — run tools/stream_sanpo_perception.py first.")

    wanted = sampled_session_ids(args.valid_streams, args.session_fraction, args.sample_seed)
    csv_paths = [pth for pth in csv_paths if pth.stem in wanted]
    if not csv_paths:
        raise SystemExit(f"None of the CSVs in {args.csv_dir} are in the seeded sample.")

    sessions = {pth.stem: ka.load_session(pth) for pth in csv_paths}
    print(f"Loaded {len(sessions)} sessions")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    strat_groups_by_session = {
        sid: stratified_groups(df, args.severity_ratio, args.kinematic_tol)
        for sid, df in sessions.items()
    }
    total_frames = sum(len(ka.frame_groups(df)) for df in sessions.values())
    strat_frames = sum(len(g) for g in strat_groups_by_session.values())
    print(f"Severity-discriminating frames: {strat_frames} / {total_frames} "
          f"({100.0 * strat_frames / max(1, total_frames):.2f}%)")
    if strat_frames == 0:
        print("WARNING: zero severity-discriminating frames found — loosen "
              "--severity-ratio / --kinematic-tol, or this corpus cannot speak to "
              "severity's design intent at all.")

    per_variant: dict[str, dict] = {}
    for name, scorer in STRATIFY_VARIANTS.items():
        rows = {
            sid: ka.session_metrics(df, scorer, rng, groups=strat_groups_by_session[sid])
            for sid, df in sessions.items()
        }
        rows = {sid: m for sid, m in rows.items() if m}
        weights = [m["frames"] for m in rows.values()]
        per_variant[name] = {
            metric: ka.bootstrap_ci([m[metric] for m in rows.values()], weights, args.n_boot, args.seed)
            for metric in ka.METRIC_NOTES
        }

    baseline_scorer = STRATIFY_VARIANTS["K0  sev·v²/d"]
    disagreements = {}
    for name, scorer in STRATIFY_VARIANTS.items():
        if name == "K0  sev·v²/d":
            continue
        strat_all_groups = [g for groups in strat_groups_by_session.values() for g in groups]
        full_all_groups = [g for df in sessions.values() for g in ka.frame_groups(df)]
        strat_d, strat_t = disagreement_rate(strat_all_groups, baseline_scorer, scorer)
        full_d, full_t = disagreement_rate(full_all_groups, baseline_scorer, scorer)
        disagreements[name] = (strat_d, strat_t, full_d, full_t)

    (args.out_dir / "stratified_metrics.json").write_text(json.dumps(
        {"coverage": {"total_frames": total_frames, "severity_discriminating_frames": strat_frames},
         "per_variant": per_variant,
         "disagreement_rate": {k: {"stratified": [v[0], v[1]], "full_corpus": [v[2], v[3]]}
                                for k, v in disagreements.items()}},
        indent=2,
    ) + "\n")
    write_report(args.out_dir, len(sessions), total_frames, strat_frames,
                 per_variant, disagreements, args.severity_ratio, args.kinematic_tol)
    print(f"Report -> {args.out_dir / 'stratified_report.md'}")


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Runnable self-check: python evaluation/kinetic_ablation_stratified.py --self-check"""
    # Frame A: a bus and a person closing at *the same* v/d — a severity-only
    # tiebreak scenario, i.e. must be flagged.
    tied = pd.DataFrame([
        {"frame_idx": 0, "track_id": "bus1", "class": "bus", "confidence": 0.9,
         "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 200, "bbox_y2": 200,
         "cx_px": 100, "bearing_deg": 5.0, "distance_m": 20.0, "velocity_ms": 4.0},
        {"frame_idx": 0, "track_id": "ped1", "class": "person", "confidence": 0.9,
         "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 100, "bbox_y2": 100,
         "cx_px": 300, "bearing_deg": 2.0, "distance_m": 10.0, "velocity_ms": 2.0},
    ])
    tied = ka.prepare(tied)
    assert is_severity_discriminating(tied, DEFAULT_SEVERITY_RATIO, DEFAULT_KINEMATIC_TOL)

    # Frame B: two persons — no severity difference at all, must not be flagged
    # regardless of kinematics.
    same_class = pd.DataFrame([
        {"frame_idx": 0, "track_id": "ped1", "class": "person", "confidence": 0.9,
         "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 100, "bbox_y2": 100,
         "cx_px": 300, "bearing_deg": 2.0, "distance_m": 10.0, "velocity_ms": 2.0},
        {"frame_idx": 0, "track_id": "ped2", "class": "person", "confidence": 0.9,
         "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 100, "bbox_y2": 100,
         "cx_px": 320, "bearing_deg": 3.0, "distance_m": 10.0, "velocity_ms": 2.0},
    ])
    same_class = ka.prepare(same_class)
    assert not is_severity_discriminating(same_class, DEFAULT_SEVERITY_RATIO, DEFAULT_KINEMATIC_TOL)

    # Frame C: bus far slower-closing than the person (v/d very different) — high
    # severity ratio but NOT kinematically tied, so must not be flagged: this is
    # exactly the case where v^2 already dominates and severity has no chance to
    # matter, which is the case the corpus-wide ablation is full of.
    not_tied = pd.DataFrame([
        {"frame_idx": 0, "track_id": "bus1", "class": "bus", "confidence": 0.9,
         "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 200, "bbox_y2": 200,
         "cx_px": 100, "bearing_deg": 5.0, "distance_m": 50.0, "velocity_ms": 0.5},
        {"frame_idx": 0, "track_id": "ped1", "class": "person", "confidence": 0.9,
         "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 100, "bbox_y2": 100,
         "cx_px": 300, "bearing_deg": 2.0, "distance_m": 5.0, "velocity_ms": 3.0},
    ])
    not_tied = ka.prepare(not_tied)
    assert not is_severity_discriminating(not_tied, DEFAULT_SEVERITY_RATIO, DEFAULT_KINEMATIC_TOL)

    # On the tied frame, K0 (with severity) must pick the bus; no-severity must
    # pick whichever has the (near-identical) higher raw v/d — demonstrating the
    # arms genuinely diverge on exactly the frames this script isolates.
    g = tied
    k0_pick = g.iloc[int(np.argmax(ka.VARIANTS["K0  sev·v²/d"](g)))]["track_id"]
    assert k0_pick == "bus1"

    print("kinetic_ablation_stratified.py self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _demo()
    else:
        main()
