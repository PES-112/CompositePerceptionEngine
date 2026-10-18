"""
fit_kalman_noise.py
===================
Fit the GroundTracker noise model (src/threat_engine/fusion.NoiseModel) to
SANPO by maximum predictive likelihood, on the official *train* sessions only.

Two objectives (--objective):
  step   one-step-ahead (0.2 s) log-likelihood of each measurement (below)
  ahead  1 s-ahead log-likelihood: the filter's own 5-step forecast and its
         propagated covariance (+ the measurement's R) scored against where the
         object is measured 1 s later. This is the time scale the arrival
         predictors use; with accurate depth the one-step objective can explain
         frame-to-frame jitter as acceleration and make 1 s forecasts worse.

Each candidate noise model runs the tracker over every train session exactly as
the replay does (tracked Stage-1 rows from the arrival dataset, corrected
units, fixed_camera_poses.csv), and is scored by the mean one-step-ahead
log-likelihood of its measurements: a bivariate Student-t with ν = robust_nu
(Gaussian when robust_nu = 0) around the filter's prediction, with the
innovation covariance S the filter itself reports. A model that is too
confident (S too small) and one that is too vague both lose likelihood, so
this calibrates R against Q without looking at any arrival label.

It also reports, on train and validation (never test):
  NIS mean    the normalised innovation squared; 2 when S is honest (heavy
              tails push the mean up, so the median is shown too: 1.39 for an
              honest Gaussian)
  gated       share of measurements rejected by the gate
  err_1s      median distance (m) between the filter's 1 s-ahead prediction
              (5 steps, constant velocity) and the measured position then
  ahead_ll    mean 1 s-ahead log-likelihood (the `ahead` objective)

    python tools/fit_kalman_noise.py                    # fit, print, write the JSON
    python tools/fit_kalman_noise.py --no-pose          # camera frame (ego-motion ablation)
    python tools/fit_kalman_noise.py --objective ahead  # -> fit_pose_ahead.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.GRU.samples import legacy_units, load_dataset
from src.threat_engine.fusion import EgoPose, GroundTracker, Measurement, NoiseModel
from tools.replay_threat_engine import AUDIT, DEFAULT_DATASET, STEP_S, STRIDE, session_poses, session_rows

FREE = ("accel_std", "static_accel_std", "depth_base", "depth_lin", "bearing_std_deg", "init_speed_std")
AHEAD = 5   # steps: 1 s


def load_sessions(split: str, use_pose: bool) -> list[tuple[dict, list[EgoPose] | None]]:
    manifest, splits = load_dataset(DEFAULT_DATASET, splits=(split,))
    audit = {r["session_id"]: r for r in json.loads(AUDIT.read_text(encoding="utf-8"))["sessions"]}
    by = defaultdict(list)
    for s in splits[split]:
        by[s["session"]].append(s)
    ids = sorted(by)
    frames = [session_rows(by[sid], audit[sid]["fx"], audit[sid]["image_width"], legacy_units(manifest))
              for sid in ids]
    if not use_pose:
        return [(f, None) for f in frames]
    with ThreadPoolExecutor(max_workers=16) as pool:
        poses = list(pool.map(session_poses, ids))
    return list(zip(frames, poses))


def student_t_loglik(y: np.ndarray, S: np.ndarray, nu: float) -> float:
    """log density of a bivariate t (scale S, ν dof) at y; Gaussian when ν = 0."""
    _, logdet = np.linalg.slogdet(S)
    m = float(y @ np.linalg.solve(S, y))
    if nu <= 0:
        return -math.log(2 * math.pi) - 0.5 * logdet - 0.5 * m
    return (gammaln((nu + 2) / 2) - gammaln(nu / 2) - math.log(nu * math.pi) - 0.5 * logdet
            - (nu + 2) / 2 * math.log1p(m / nu))


def evaluate(noise: NoiseModel, sessions, full: bool = False, ahead: bool = False) -> dict:
    """Scores on `sessions`; 1 s-ahead forecasts are collected when full or ahead (slower)."""
    ll, nis, err, ahead_ll = [], [], [], []
    forecast = full or ahead
    for frames, poses in sessions:
        tracker = GroundTracker(noise=noise, innovations=[])
        measured: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
        forecasts: list[tuple[str, int, np.ndarray, np.ndarray]] = []
        for k in sorted(frames):
            pose = poses[min(STRIDE * k, len(poses) - 1)] if poses else EgoPose.identity()
            ms = [Measurement(r["track_id"], r["class"], r["distance_m"], r["bearing_deg"], r["confidence"], r)
                  for r in frames[k] if not str(r["track_id"]).startswith("obs_")]
            tracker.step(k * STEP_S, ms, pose)
            if forecast:
                for m in ms:
                    measured[m.track_id][k] = tracker._measure(m, pose)      # (world position, R)
                    f = tracker._filters.get(m.track_id)
                    if f is not None and f.hits >= 3 and f.measured:
                        g = replace(f, x=f.x.copy(), P=f.P.copy())
                        tracker._predict(g, AHEAD * STEP_S)
                        forecasts.append((m.track_id, k, g.x[:2], g.P[:2, :2]))
        for y, S in tracker.innovations:
            ll.append(student_t_loglik(y, S, noise.robust_nu))
            nis.append(float(y @ np.linalg.solve(S, y)))
        for tid, k, mean, cov in forecasts:
            future = measured[tid].get(k + AHEAD)
            if future is not None:
                z, R = future
                err.append(float(np.linalg.norm(mean - z)))
                ahead_ll.append(student_t_loglik(z - mean, cov + R, noise.robust_nu))
    out = {"loglik": float(np.mean(ll)), "updates": len(ll), "gated": float(np.mean(np.array(nis) > noise.gate_nis))}
    if forecast:
        out.update({"ahead_ll": float(np.mean(ahead_ll)) if ahead_ll else float("nan"), "forecasts": len(err),
                    "err_1s_median_m": float(np.median(err)) if err else float("nan")})
    if full:
        out.update({"nis_mean": float(np.mean(nis)), "nis_median": float(np.median(nis))})
    return out


def fit(sessions, start: NoiseModel, nu_grid=(0.0, 4.0), objective: str = "step") -> tuple[NoiseModel, list]:
    """Nelder-Mead on the log of the free parameters, for each ν on a grid; the best overall wins."""
    trace = []
    best = None
    for nu in nu_grid:
        base = replace(start, robust_nu=nu, gate_nis=60.0 if nu > 0 else 11.62)

        def loss(z):
            noise = replace(base, **{k: float(math.exp(v)) for k, v in zip(FREE, z)})
            scores = evaluate(noise, sessions, ahead=objective == "ahead")
            return -scores["ahead_ll" if objective == "ahead" else "loglik"]

        z0 = np.log([getattr(base, k) for k in FREE])
        res = minimize(loss, z0, method="Nelder-Mead", options={"maxfev": 100, "xatol": 0.02, "fatol": 1e-4})
        noise = replace(base, **{k: float(math.exp(v)) for k, v in zip(FREE, res.x)})
        trace.append({"robust_nu": nu, objective: -float(res.fun), "evals": int(res.nfev), "noise": asdict(noise)})
        print(f"  nu={nu:g}: {objective} {-res.fun:.4f} ({res.nfev} evals) {asdict(noise)}", flush=True)
        if best is None or -res.fun > best[1]:
            best = (noise, -float(res.fun))
    return best[0], trace


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-pose", action="store_true", help="Fit in the camera frame (no ego-motion removal).")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "evaluation/benchmarks/kalman_noise_fit")
    p.add_argument("--objective", choices=("step", "ahead"), default="step",
                   help="step: one-step log-likelihood; ahead: 1 s-ahead log-likelihood (see the docstring).")
    args = p.parse_args()
    use_pose = not args.no_pose
    started = time.time()
    train, val = load_sessions("train", use_pose), load_sessions("val", use_pose)
    print(f"{len(train)} train / {len(val)} val sessions loaded ({time.time() - started:.0f} s)", flush=True)

    declared = NoiseModel.declared()
    fitted, trace = fit(train, NoiseModel(), objective=args.objective)
    rows = {}
    for name, noise in (("declared", declared), ("fitted", fitted)):
        rows[name] = {split: evaluate(noise, data, full=True) for split, data in (("train", train), ("val", val))}
        print(name, json.dumps(rows[name]), flush=True)

    tag = ("pose" if use_pose else "nopose") + ("_ahead" if args.objective == "ahead" else "")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"fit_{tag}.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%d"), "dataset": DEFAULT_DATASET.name, "frame": tag,
        "objective": args.objective,
        "fitted": asdict(fitted), "declared": asdict(declared), "trace": trace, "scores": rows,
        "seconds": round(time.time() - started)}, indent=2), encoding="utf-8")
    print(f"-> {args.out / f'fit_{tag}.json'} ({time.time() - started:.0f} s)")


if __name__ == "__main__":
    main()
