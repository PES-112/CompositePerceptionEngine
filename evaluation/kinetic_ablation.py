"""
kinetic_ablation.py
===================
Ablate the terms of K0 over Stage-1 CSVs and score every variant with metrics
that need no labels (docs/kinetic_score_opinion.md §5, §6).

Nothing here grades a formula against another formula's output — the two things
it compares against are the *measured* future (§6) and the formula's own
stability under perturbation. The one question these cannot answer — whether
v² and the severity weights are *right* — is exported as disagreement frames
for `evaluation/vlm_referee.py`.

Input : a directory of per-session CSVs from tools/stream_sanpo_perception.py.
Output: report.md, metrics.json, disagreements.json (blind), disagreements_key.json.

Usage
-----
    python evaluation/kinetic_ablation.py --csv-dir data/processed/ablation_30pct \\
        --out-dir evaluation/benchmarks/kinetic_score_eval/run_2026_08_19
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# stdlib-only module — importing it does not pull in ultralytics/torch.
from tools.download_sanpo_valid_streams import (  # noqa: E402
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SESSION_FRACTION,
    DEFAULT_VALID_STREAMS,
    sampled_session_ids,
)

# Load physics.py directly: importing the package pulls in ultralytics/torch,
# which this analysis does not need and reviewers may not have installed.
_spec = importlib.util.spec_from_file_location(
    "cpe_physics", PROJECT_ROOT / "src/perception_stack/physics.py"
)
physics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(physics)

D_HAZ_M = 1.5      # metres — "an encounter happened" threshold (§6)
THETA_DEG = 30.0   # bearing cone for an encounter
HORIZON_FRAMES = 20
TIE_MARGIN = 0.05
PERTURB_SIGMAS = (0.02, 0.05, 0.10)


# ── The variants under test ──────────────────────────────────────────────────
# Each maps a frame's rows to a score array. Every one of them calls the
# production kinetic_score() with different exponents, so there is no second
# implementation of the score to drift out of sync — except TTC, which is a
# different quantity by design.

def _score_k(rows: pd.DataFrame, *, gamma: float, severity: bool,
             size_exponent: float = 0.0) -> np.ndarray:
    # λ is frozen at physics.SEVERITY_LAMBDA (0.5) — see the module docstring.
    # No arm swaps it any more, so the severity law here is the production one.
    out = []
    for _, r in rows.iterrows():
        k = physics.kinetic_score(
            r["distance_m"], r["velocity_ms"], r["class"] if severity else "__none__",
            bbox_area_px=r["bbox_area_px"], gamma=gamma, size_exponent=size_exponent,
        )
        out.append(k)
    return np.asarray(out, dtype=float)


def _score_ttc(rows: pd.DataFrame) -> np.ndarray:
    """Plain time-to-hazard, ranked so that sooner = higher. Never-arriving = -inf."""
    d = rows["distance_m"].to_numpy(dtype=float)
    v = rows["velocity_ms"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(v > 0, (d - D_HAZ_M) / v, np.inf)
    return -ttc


VARIANTS: dict[str, callable] = {
    "K0  sev·v²/d":        lambda r: _score_k(r, gamma=2.0, severity=True),
    "linear  sev·v/d":     lambda r: _score_k(r, gamma=1.0, severity=True),
    "no-severity  v²/d":   lambda r: _score_k(r, gamma=2.0, severity=False),
    "no-velocity  sev/d":  lambda r: _score_k(r, gamma=0.0, severity=True),
    "size  sev·v²·s^½/d":  lambda r: _score_k(r, gamma=2.0, severity=True, size_exponent=0.5),
    "ttc  -(d-D)/v":       _score_ttc,
}
BASELINE = "K0  sev·v²/d"


# ── Per-session metric computation ───────────────────────────────────────────

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop depth-less rows and derive the columns the variants need."""
    df = df.dropna(subset=["distance_m"]).copy()
    df["velocity_ms"] = df["velocity_ms"].fillna(0.0)
    df["bbox_area_px"] = (df["bbox_x2"] - df["bbox_x1"]) * (df["bbox_y2"] - df["bbox_y1"])
    df["track_id"] = df["track_id"].astype(str)
    return df


def load_session(path: Path) -> pd.DataFrame:
    return prepare(pd.read_csv(path))


def frame_groups(df: pd.DataFrame) -> list[tuple[int, pd.DataFrame]]:
    """Frames with at least two objects — a single-object frame has no ranking to test."""
    return [(idx, g) for idx, g in df.groupby("frame_idx", sort=True) if len(g) >= 2]


def encounters(df: pd.DataFrame) -> dict[int, set[str]]:
    """
    Tracks that measurably came within D_HAZ_M inside the bearing cone within
    HORIZON_FRAMES of each frame. Uses distance and bearing only — never
    velocity, severity, or any K — so it cannot be circular with the scores.
    """
    hits: dict[int, set[str]] = {}
    close = df[(df["distance_m"] < D_HAZ_M) & (df["bearing_deg"].abs() < THETA_DEG)]
    by_track = close.groupby("track_id")["frame_idx"].apply(list).to_dict()
    for frame_idx in df["frame_idx"].unique():
        present = set()
        for tid, frames in by_track.items():
            if any(frame_idx <= f <= frame_idx + HORIZON_FRAMES for f in frames):
                present.add(tid)
        if present:
            hits[int(frame_idx)] = present
    return hits


def session_metrics(df: pd.DataFrame, scorer, rng: np.random.Generator,
                     groups: list[tuple[int, pd.DataFrame]] | None = None) -> dict:
    """
    `groups` lets a caller pass a pre-filtered subset of frame_groups(df) — e.g.
    evaluation/kinetic_ablation_stratified.py restricts to severity-discriminating
    frames only — while reusing the exact same metric math, so a stratified report
    can never silently drift out of sync with the full-corpus one.
    """
    if groups is None:
        groups = frame_groups(df)
    if not groups:
        return {}

    tops: dict[int, str] = {}
    tie_flags, taus, track_scores = [], [], {}

    for frame_idx, g in groups:
        s = scorer(g)
        if not np.isfinite(s).any():
            continue
        order = np.argsort(-s)
        tops[frame_idx] = g.iloc[order[0]]["track_id"]

        top1, top2 = s[order[0]], s[order[1]]
        tie_flags.append(bool(np.isfinite(top1) and np.isfinite(top2)
                              and top1 > 0 and (top1 - top2) / top1 < TIE_MARGIN))

        for tid, val in zip(g["track_id"], s):
            track_scores.setdefault(tid, []).append(val)

        # Rank stability: same frame, depth multiplied by measurement noise.
        for sigma in PERTURB_SIGMAS:
            noisy = g.copy()
            noisy["distance_m"] = np.maximum(
                0.1, g["distance_m"].to_numpy() * (1 + rng.normal(0, sigma, len(g)))
            )
            s2 = scorer(noisy)
            if np.isfinite(s).all() and np.isfinite(s2).all() and len(g) >= 2:
                tau = kendalltau(s, s2).statistic
                if not np.isnan(tau):
                    taus.append((sigma, tau))

    idxs = sorted(tops)
    flips = sum(1 for a, b in zip(idxs, idxs[1:]) if tops[a] != tops[b])
    flicker = flips / max(1, len(idxs) - 1)

    # Temporal smoothness, per track, normalised by that track's mean score.
    smooth = []
    for vals in track_scores.values():
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) >= 2 and np.mean(vals) > 0:
            smooth.append(float(np.mean(np.abs(np.diff(vals))) / np.mean(vals)))

    # Future self-consistency: does the pick at T survive to T+H?
    consistent = [tops[i] == tops[i + HORIZON_FRAMES]
                  for i in idxs if i + HORIZON_FRAMES in tops]

    # §6 automatic ground truth: when something really did close in, was it the pick?
    enc = encounters(df)
    enc_hits = [tops[i] in enc[i] for i in idxs if i in enc]

    out = {
        "frames": len(idxs),
        "flicker_rate": flicker,
        "tie_rate": float(np.mean(tie_flags)) if tie_flags else float("nan"),
        "smoothness": float(np.mean(smooth)) if smooth else float("nan"),
        "future_consistency": float(np.mean(consistent)) if consistent else float("nan"),
        "encounter_top1": float(np.mean(enc_hits)) if enc_hits else float("nan"),
        "encounter_frames": len(enc_hits),
    }
    for sigma in PERTURB_SIGMAS:
        vals = [t for s_, t in taus if s_ == sigma]
        out[f"rank_stability_{int(sigma * 100)}pct"] = float(np.mean(vals)) if vals else float("nan")
    return out


# ── Session-level bootstrap ──────────────────────────────────────────────────

def bootstrap_ci(per_session: list[float], weights: list[float], n_boot: int, seed: int) -> tuple:
    """
    Frames inside a session are autocorrelated, so resample whole sessions.
    Frame-count weighting keeps a 12-frame session from counting as much as a
    300-frame one.
    """
    vals = np.asarray(per_session, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(vals) & (w > 0)
    vals, w = vals[ok], w[ok]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.average(vals, weights=w))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(vals), len(vals))
        draws.append(np.average(vals[pick], weights=w[pick]))
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ── Disagreement export for the referee ──────────────────────────────────────

def collect_disagreements(sessions: dict[str, pd.DataFrame], n_per_pair: int, seed: int) -> tuple[list, list]:
    """
    Frames where the baseline and a variant pick different top objects. Everything
    else is uninformative — the formulas agree, so no referee can separate them.

    Returns (blind_payload, key). The key holds which formula picked what and is
    never handed to a referee.
    """
    found: dict[str, list] = {name: [] for name in VARIANTS if name != BASELINE}
    for sid, df in sessions.items():
        for frame_idx, g in frame_groups(df):
            base = VARIANTS[BASELINE](g)
            if not np.isfinite(base).any():
                continue
            base_pick = g.iloc[int(np.argmax(base))]["track_id"]
            for name, scorer in VARIANTS.items():
                if name == BASELINE:
                    continue
                s = scorer(g)
                if not np.isfinite(s).any():
                    continue
                pick = g.iloc[int(np.argmax(s))]["track_id"]
                if pick != base_pick:
                    found[name].append((sid, int(frame_idx), g, base_pick, pick))

    rng = random.Random(seed)
    blind, key = [], []
    for name, cases in found.items():
        sample = rng.sample(cases, min(n_per_pair, len(cases)))
        for sid, frame_idx, g, base_pick, pick in sample:
            objects = [{
                "object_id": f"O{n}",
                "track_id": r["track_id"],
                "class": r["class"],
                "distance_m": round(float(r["distance_m"]), 2),
                "closing_speed_ms": round(float(r["velocity_ms"]), 2),
                "bearing_deg": round(float(r["bearing_deg"]), 1),
                "bbox": [int(r["bbox_x1"]), int(r["bbox_y1"]), int(r["bbox_x2"]), int(r["bbox_y2"])],
            } for n, (_, r) in enumerate(g.iterrows())]
            rng.shuffle(objects)   # order must not leak either formula's ranking
            case_id = f"{sid}:{frame_idx}:{len(blind)}"
            blind.append({"case_id": case_id, "session_id": sid,
                          "frame_idx": frame_idx, "objects": objects})
            key.append({"case_id": case_id, "variant": name,
                        "baseline_pick": base_pick, "variant_pick": pick,
                        "n_objects": len(objects)})
    return blind, key


# ── Report ───────────────────────────────────────────────────────────────────

METRIC_NOTES = {
    "flicker_rate": "lower better — how often the announced top threat changes identity",
    "tie_rate": "lower better — frames where the top two are within 5%",
    "smoothness": "lower better — mean relative frame-to-frame change per track",
    "future_consistency": "higher better — pick at T still the pick at T+H",
    "encounter_top1": "higher better — pick was an object that measurably closed in (§6: eliminates, never selects)",
    "rank_stability_2pct": "higher better — Kendall τ under 2% depth noise",
    "rank_stability_5pct": "higher better — Kendall τ under 5% depth noise",
    "rank_stability_10pct": "higher better — Kendall τ under 10% depth noise",
}


def write_report(results: dict, out_dir: Path, n_sessions: int, n_frames: int) -> None:
    lines = [
        "# Kinetic Score Ablation",
        "",
        f"Sessions: {n_sessions} · scored frames (≥2 objects): {n_frames}",
        "",
        "Point estimate is frame-count-weighted across sessions; CI is a 95% "
        "percentile bootstrap resampling **whole sessions** (frames within a session "
        "are autocorrelated — frame-level CIs would be far too narrow).",
        "",
    ]
    for metric, note in METRIC_NOTES.items():
        lines += [f"## {metric}", "", f"_{note}_", "",
                  "| Variant | Value | 95% CI |", "|---|---:|---|"]
        for name in VARIANTS:
            p, lo, hi = results[name][metric]
            lines.append(f"| `{name}` | {p:.3f} | [{lo:.3f}, {hi:.3f}] |")
        lines.append("")
    lines += [
        "## What this does not answer",
        "",
        "Whether `v²` and the severity weights are *right* is a value judgment. "
        "No volume of unlabelled video settles it — see `disagreements.json` and "
        "`evaluation/vlm_referee.py`.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Ablate K0's terms with label-free metrics.")
    p.add_argument("--csv-dir", type=Path, required=True, help="Directory of per-session Stage-1 CSVs.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260819, help="Bootstrap resampling seed.")
    p.add_argument("--valid-streams", type=Path, default=DEFAULT_VALID_STREAMS,
                   help="SANPO valid-stream manifest defining the ablation sample.")
    p.add_argument("--session-fraction", type=float, default=DEFAULT_SESSION_FRACTION,
                   help="Fraction of valid streams under study. Must match the value "
                        "tools/stream_sanpo_perception.py was run with. 1.0 = no subsetting.")
    p.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED,
                   help="Seed for the session sample. Kept separate from --seed so changing "
                        "the bootstrap seed cannot silently change which sessions were evaluated.")
    p.add_argument("--disagreements-per-pair", type=int, default=60,
                   help="Frames exported per variant for the referee (§3 budget: 100–300 total).")
    args = p.parse_args()

    csv_paths = sorted(pth for pth in args.csv_dir.glob("*.csv") if not pth.name.endswith(".partial"))
    if not csv_paths:
        raise SystemExit(f"No session CSVs in {args.csv_dir} — run tools/stream_sanpo_perception.py first.")

    # Evaluate exactly the seeded sample, not whatever happens to be on disk: a
    # stale CSV from an earlier run would otherwise widen the corpus silently and
    # invalidate the session-level bootstrap CIs.
    wanted = sampled_session_ids(args.valid_streams, args.session_fraction, args.sample_seed)
    off_sample = [pth for pth in csv_paths if pth.stem not in wanted]
    csv_paths = [pth for pth in csv_paths if pth.stem in wanted]
    missing = sorted(wanted - {pth.stem for pth in csv_paths})
    print(f"Sample: {len(wanted)} of the valid streams at fraction={args.session_fraction} "
          f"seed={args.sample_seed}")
    if off_sample:
        print(f"  skipped {len(off_sample)} CSV(s) outside the sample")
    if missing:
        print(f"  WARNING: {len(missing)} sampled session(s) have no CSV yet, e.g. {missing[:3]}")
    if not csv_paths:
        raise SystemExit(
            f"None of the CSVs in {args.csv_dir} are in the seeded sample. Re-run Stage 1 with "
            f"--session-fraction {args.session_fraction} --seed {args.sample_seed}, or pass "
            f"--session-fraction 1.0 to score every session on disk."
        )

    sessions = {pth.stem: load_session(pth) for pth in csv_paths}
    sessions = {sid: df for sid, df in sessions.items() if frame_groups(df)}
    print(f"Loaded {len(sessions)} sessions with rankable frames")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    per_variant: dict[str, dict] = {}
    total_frames = 0
    for name, scorer in VARIANTS.items():
        rows = {sid: session_metrics(df, scorer, rng) for sid, df in sessions.items()}
        rows = {sid: m for sid, m in rows.items() if m}
        weights = [m["frames"] for m in rows.values()]
        total_frames = max(total_frames, sum(weights))
        per_variant[name] = {
            metric: bootstrap_ci([m[metric] for m in rows.values()], weights, args.n_boot, args.seed)
            for metric in METRIC_NOTES
        }
        print(f"  {name:22s} flicker={per_variant[name]['flicker_rate'][0]:.3f}")

    (args.out_dir / "metrics.json").write_text(json.dumps(per_variant, indent=2) + "\n")
    write_report(per_variant, args.out_dir, len(sessions), total_frames)

    blind, key = collect_disagreements(sessions, args.disagreements_per_pair, args.seed)
    (args.out_dir / "disagreements.json").write_text(json.dumps(blind, indent=2) + "\n")
    (args.out_dir / "disagreements_key.json").write_text(json.dumps(key, indent=2) + "\n")
    print(f"\n{len(blind)} disagreement frames exported -> {args.out_dir}")
    print(f"Report -> {args.out_dir / 'report.md'}")


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Runnable self-check: python evaluation/kinetic_ablation.py --self-check"""
    # Two tracks over 10 frames: a bus that never closes, a person closing fast.
    rows = []
    for f in range(10):
        rows.append({"frame_idx": f, "track_id": "bus1", "class": "bus", "confidence": 0.9,
                     "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 200, "bbox_y2": 200,
                     "cx_px": 100, "bearing_deg": 5.0, "distance_m": 20.0, "velocity_ms": 0.0})
        rows.append({"frame_idx": f, "track_id": "ped1", "class": "person", "confidence": 0.9,
                     "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 100, "bbox_y2": 100,
                     "cx_px": 300, "bearing_deg": 2.0, "distance_m": 8.0 - 0.8 * f,
                     "velocity_ms": 2.4})
    df = prepare(pd.DataFrame(rows))

    groups = frame_groups(df)
    assert len(groups) == 10

    # Every variant that can see velocity must rank the closing pedestrian above
    # the parked bus.
    g0 = groups[0][1]
    for name, scorer in VARIANTS.items():
        if name.startswith("no-velocity"):
            continue
        s = scorer(g0)
        assert g0.iloc[int(np.argmax(s))]["track_id"] == "ped1", name

    # ...and the velocity ablation must fail exactly here — a parked bus at 20 m
    # outranking a pedestrian closing at 2.4 m/s is the failure this arm exists
    # to expose. If this assertion ever passes, the ablation is not ablating.
    s = VARIANTS["no-velocity  sev/d"](g0)
    assert g0.iloc[int(np.argmax(s))]["track_id"] == "bus1"

    # A track that measurably reaches D_HAZ inside the cone is an encounter;
    # the stationary bus at 20 m never is.
    enc = encounters(df)
    assert enc and all("ped1" in tracks for tracks in enc.values())
    assert not any("bus1" in tracks for tracks in enc.values())

    m = session_metrics(df, VARIANTS[BASELINE], np.random.default_rng(0))
    assert m["flicker_rate"] == 0.0            # one object dominates throughout
    assert m["encounter_top1"] == 1.0
    assert 0.0 <= m["tie_rate"] <= 1.0

    # Bootstrap: identical sessions must give a degenerate CI around the value.
    pt, lo, hi = bootstrap_ci([0.5, 0.5, 0.5], [10, 10, 10], 200, 0)
    assert pt == lo == hi == 0.5

    print("kinetic_ablation.py self-check OK")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _demo()
    else:
        main()
