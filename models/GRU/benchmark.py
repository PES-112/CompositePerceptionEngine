"""
benchmark.py
============
The held-out baseline ladder for arrival prediction: every learned and rule
baseline a forecasting reviewer expects, scored once on SANPO's official test
sessions, with paired session-bootstrap tests and Holm correction.

Methods
  rules (no training)
    K0, TTC, TTC_in_cone, nearest, nearest_in_cone    models/GRU/evaluate.baseline_scores
    nearest_in_corridor   in the corridor's lateral band ahead first, then by distance
    cv_last2              constant-velocity model from the last two observed steps
                          (Schöller et al., RA-L 2020), extrapolated into the zone
    cv_kalman             constant-velocity Kalman filter over the whole history
                          (src/threat_engine/fusion.GroundTracker), extrapolated
  learned without sequence modelling (fitted on the official train sessions)
    logreg_now            logistic regression on the current step's features + class
    gbt_now               gradient-boosted trees, same inputs
    gbt_history           gradient-boosted trees on the whole flattened history
  sequence models (models/GRU runs, 3 seeds each)
    e.g. gru, gru_attention, gru_hist4, gru_hist1

Geometry: the zone is the label's own (corridor: 0 < forward <= depth,
|lateral| <= width/2 + class half-width; cone: within hazard_m and ±30°), in the
camera frame, with pinhole bearings from the session's real fx. Positions come
from the history features exactly as stored (distance, legacy bearing).

Statistics: AUROC and AUPRC (average precision) per method, all objects and by
stratum (static vs dynamic classes); arrival_order, soonest_top1 and
flicker_rate per frame. Differences vs a reference method (default gru) use
1,000 paired session-bootstrap draws; two-sided p from the normal
approximation with the bootstrap SE, Holm-adjusted within each stratum over
every (method, metric) comparison.

    python -m models.GRU.benchmark \\
        --dataset dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_official_corridor_w0.75_d3_h20 \\
        --run gru=corridor_h8:gru --run gru_attention=corridor_h8:gru_attention \\
        --run gru_hist4=corridor_hist4:gru --run gru_hist1=corridor_hist1:gru
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dataset.GRU.samples import CLASS_VOCAB, FEATURES, load_dataset, truncate_history
from src.threat_engine.envelope import CLASS_HALF_WIDTH_M, DEFAULT_HALF_WIDTH_M
from models.GRU.evaluate import auroc, average_precision, baseline_scores, model_scores, per_frame, predict
from models.GRU.registry import build_model
from models.GRU.summarize import run_dirs
from src.threat_engine.fusion import GroundTracker, Measurement

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT = PROJECT_ROOT / "simulation/datasets/sanpo/session_audit.json"
STATIC_CLASSES = {"traffic light", "stop sign", "fire hydrant", "pole", "bollard", "stairs", "crosswalk",
                  "pothole", "puddle", "bench"}
N_BOOT = 1000
STEP_S = 0.2
F = {name: i for i, name in enumerate(FEATURES)}

Scores = list[np.ndarray]


# ── Geometry from the stored features ────────────────────────────────────────

class Zone:
    def __init__(self, label: dict):
        self.kind = label.get("kind", "cone")
        self.horizon = int(label["HORIZON_FRAMES"])
        self.width = float(label.get("width_m", 0.75))
        self.depth = float(label.get("depth_m", label.get("hazard_m", 3.0)))
        self.theta = math.radians(float(label.get("THETA_DEG", 30.0)))

    def inside(self, x: np.ndarray, z: np.ndarray, half: float) -> np.ndarray:
        if self.kind == "corridor":
            return (z > 0) & (z <= self.depth) & (np.abs(x) <= self.width / 2 + half)
        return (np.hypot(x, z) < self.depth) & (np.abs(np.arctan2(x, z)) < self.theta)

    def lateral_ok(self, x: np.ndarray, z: np.ndarray, half: float) -> np.ndarray:
        if self.kind == "corridor":
            return (z > 0) & (np.abs(x) <= self.width / 2 + half)
        return np.abs(np.arctan2(x, z)) < self.theta


def positions(s: dict, fx: float, width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(present [n, h], x [n, h], z [n, h]) in metres, camera frame, pinhole bearings."""
    present = s["x"][:, :, F["present"]] > 0.5
    d = s["x"][:, :, F["distance"]] * 10.0
    # The feature is legacy_bearing / 35°, and the legacy bearing mapped the image linearly onto ±35°,
    # so feature × half the image width is the pixel offset from the centre.
    dx = s["x"][:, :, F["bearing"]] * (width / 2)
    theta = np.arctan2(dx, fx)
    return present, d * np.tan(theta), d


def extrapolate(p: np.ndarray, v: np.ndarray, zone: Zone, half: float) -> float:
    """Score = minus the first step (0..horizon) at which p + v·t is in the zone; -(horizon + 1 + distance) if never."""
    t = np.arange(zone.horizon + 1)
    xs, zs = p[0] + v[0] * t, p[1] + v[1] * t
    hit = np.nonzero(zone.inside(xs, zs, half))[0]
    return -float(hit[0]) if len(hit) else -(zone.horizon + 1 + float(np.hypot(*p)))


def rule_scores(s: dict, fx: float, width: float, zone: Zone) -> dict[str, np.ndarray]:
    present, X, Z = positions(s, fx, width)
    halves = [CLASS_HALF_WIDTH_M.get(c, DEFAULT_HALF_WIDTH_M) for c in s["classes"]]
    x, z = X[:, -1], Z[:, -1]
    corridor = np.array([zone.lateral_ok(x[i:i + 1], z[i:i + 1], h)[0] for i, h in enumerate(halves)])
    out = {"nearest_in_corridor": -s["distance"] - np.where(corridor, 0.0, 1e3)}
    last2, kalman = [], []
    for i, h in enumerate(halves):
        steps = np.nonzero(present[i])[0]
        p = np.array([x[i], z[i]])
        if len(steps) >= 2:
            a, b = steps[-2], steps[-1]
            v = (np.array([X[i, b], Z[i, b]]) - np.array([X[i, a], Z[i, a]])) / (b - a)
        else:
            v = np.zeros(2)
        last2.append(extrapolate(p, v, zone, h))
        tracker = GroundTracker()
        track = None
        for k in steps:
            bearing = math.degrees(math.atan2(X[i, k], Z[i, k]))
            track = tracker.step(k * STEP_S, [Measurement("o", s["classes"][i], float(Z[i, k]), bearing)])
        if track:
            t0 = track[0]
            pk = np.array([t0.lateral_m, t0.forward_m])
            vk = np.array(t0.rel_velocity) * STEP_S
            kalman.append(extrapolate(pk, vk, zone, h))
        else:
            kalman.append(extrapolate(p, np.zeros(2), zone, h))
    out["cv_last2"] = np.array(last2)
    out["cv_kalman"] = np.array(kalman)
    return out


# ── Learned non-sequence baselines ───────────────────────────────────────────

def tabular(samples: list[dict], audit: dict, history: bool) -> np.ndarray:
    rows = []
    for s in samples:
        a = audit[s["session"]]
        present, X, Z = positions(s, a["fx"], a["image_width"])
        feats = s["x"].reshape(len(s["y"]), -1) if history else s["x"][:, -1, :]
        onehot = np.zeros((len(s["y"]), len(CLASS_VOCAB) + 1))
        onehot[np.arange(len(s["y"])), np.minimum(s["cls"], len(CLASS_VOCAB))] = 1
        rows.append(np.hstack([feats, X[:, -1:], Z[:, -1:], onehot]))
    return np.vstack(rows)


def fit_tabular(train: list[dict], test: list[dict], audit: dict, seed: int = 0) -> dict[str, Scores]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.concatenate([s["y"] for s in train])
    bounds = np.cumsum([0] + [len(s["y"]) for s in test])
    split = lambda flat: [flat[a:b] for a, b in zip(bounds[:-1], bounds[1:])]   # noqa: E731
    out = {}
    for name, history, model in (
        ("logreg_now", False, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
        ("gbt_now", False, HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)),
        ("gbt_history", True, HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)),
    ):
        model.fit(tabular(train, audit, history), y)
        out[name] = split(model.predict_proba(tabular(test, audit, history))[:, 1])
    return out


# ── Sequence models ──────────────────────────────────────────────────────────

def gru_scores(run_name: str, arch: str, manifest: dict, test: list[dict]) -> list[tuple[Scores, Scores]]:
    dirs = run_dirs(run_name).get(arch, [])
    if not dirs:
        raise SystemExit(f"no {arch} runs named {run_name}_seed*")
    out = []
    for d in dirs:
        ckpt = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        if cfg["dataset_fingerprint"] != manifest["source"]["fingerprint"] or cfg["split"] != manifest["split"] \
                or cfg["dataset_name"] != manifest["name"]:
            raise SystemExit(f"{d} was not trained on {manifest['name']}")
        model = build_model(ckpt["arch"], ckpt["model_kwargs"])
        model.load_state_dict(ckpt["state_dict"])
        samples = truncate_history(test, cfg["history_steps"]) if cfg.get("history_steps") else test
        out.append(model_scores(predict(model, samples, "cpu"), cfg["time_bins"], cfg["horizon"]))
    return out


# ── Statistics ───────────────────────────────────────────────────────────────

def object_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    return auroc(scores, labels), average_precision(scores, labels)


def bootstrap(samples: list[dict], methods: dict[str, list[tuple[Scores, Scores]]], ref: str,
              mask_of=None, seed: int = 0) -> dict[str, dict[str, list[float]]]:
    """{method: {metric: [full-data diff, draws...]}} of (method - ref), metric averaged over each side's seeds."""
    by_session = defaultdict(list)
    for i, s in enumerate(samples):
        by_session[s["session"]].append(i)
    sessions = sorted(by_session)

    def flat(arrs, idx):
        parts = [arrs[i] if mask_of is None else arrs[i][mask_of(samples[i])] for i in idx]
        return np.concatenate(parts) if parts else np.zeros(0)

    per_session = {sid: by_session[sid] for sid in sessions}
    labels = {sid: flat([s["y"] for s in samples], per_session[sid]) for sid in sessions}
    scored = {m: [{sid: flat(seed_scores[0], per_session[sid]) for sid in sessions} for seed_scores in seeds]
              for m, seeds in methods.items()}
    rng = np.random.default_rng(seed)
    draws = [np.arange(len(sessions))] + [rng.integers(0, len(sessions), len(sessions)) for _ in range(N_BOOT)]
    out = {m: {"auroc": [], "auprc": []} for m in methods if m != ref}
    for pick in draws:
        sids = [sessions[j] for j in pick]
        y = np.concatenate([labels[sid] for sid in sids])
        vals = {}
        for m, seeds in scored.items():
            ms = [object_metrics(y, np.concatenate([sc[sid] for sid in sids])) for sc in seeds]
            vals[m] = np.nanmean(ms, axis=0)
        for m in out:
            out[m]["auroc"].append(float(vals[m][0] - vals[ref][0]))
            out[m]["auprc"].append(float(vals[m][1] - vals[ref][1]))
    return out


def holm(pvals: dict) -> dict:
    keys = sorted(pvals, key=lambda k: pvals[k])
    adj, running, m = {}, 0.0, len(keys)
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return adj


def summarize_diffs(diffs: dict) -> tuple[dict, dict]:
    rows, pvals = {}, {}
    for m, by_metric in diffs.items():
        for metric, ds in by_metric.items():
            d = np.array(ds)
            boot = d[1:][np.isfinite(d[1:])]
            # Normal approximation with the bootstrap SE: a percentile p-value can't go below 1/N_BOOT,
            # which Holm over dozens of tests would never let through.
            se = float(np.std(boot)) if len(boot) else float("nan")
            p = float(math.erfc(abs(d[0]) / se / math.sqrt(2))) if se and np.isfinite(se) else float("nan")
            rows[(m, metric)] = (float(d[0]), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
            pvals[(m, metric)] = p
    return rows, pvals


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--run", action="append", default=[], help="name=run_name:arch, e.g. gru=corridor_h8:gru")
    p.add_argument("--reference", default="gru")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    manifest, splits = load_dataset(args.dataset, splits=("train", "test"))
    train, test = splits["train"], splits["test"]
    zone = Zone(manifest["label"])
    audit = {r["session_id"]: r for r in json.loads(AUDIT.read_text(encoding="utf-8"))["sessions"]}
    started = time.time()

    methods: dict[str, list[tuple[Scores, Scores]]] = {}
    for spec in args.run:
        name, rest = spec.split("=", 1)
        run_name, arch = rest.rsplit(":", 1)
        methods[name] = gru_scores(run_name, arch, manifest, test)
    for name, scores in fit_tabular(train, test, audit).items():
        methods[name] = [(scores, scores)]
    base = [baseline_scores(s) for s in test]
    rules = [rule_scores(s, audit[s["session"]]["fx"], audit[s["session"]]["image_width"], zone) for s in test]
    for name in list(base[0]) + list(rules[0]):
        sc = [b[name] for b in base] if name in base[0] else [r[name] for r in rules]
        methods[name] = [(sc, sc)]
    if args.reference not in methods:
        raise SystemExit(f"--reference {args.reference} is not among {sorted(methods)}")

    def is_static(s):
        return np.array([c in STATIC_CLASSES for c in s["classes"]])

    strata = {"all": None, "static": is_static, "dynamic": lambda s: ~is_static(s)}
    table = {}
    for name, seeds in methods.items():
        row = {}
        for stratum, mask_of in strata.items():
            ys = [s["y"] if mask_of is None else s["y"][mask_of(s)] for s in test]
            y = np.concatenate(ys)
            ms = [object_metrics(y, np.concatenate([sc if mask_of is None else sc[mask_of(s)]
                                                    for s, sc in zip(test, arrive)])) for arrive, _ in seeds]
            row[stratum] = {"auroc": float(np.nanmean([m[0] for m in ms])), "auprc": float(np.nanmean([m[1] for m in ms])),
                            "objects": int(len(y)), "positive_rate": float(y.mean()) if len(y) else float("nan")}
        frames = [per_frame(test, arrive, soon) for arrive, soon in seeds]
        for key in ("arrival_order", "soonest_top1", "flicker_rate"):
            row["all"][key] = float(np.nanmean([f[key][0] for f in frames]))
        table[name] = row

    tests = {}
    for stratum in ("all", "static", "dynamic"):
        rows, pvals = summarize_diffs(bootstrap(test, methods, args.reference, strata[stratum]))
        tests[stratum] = (rows, pvals)
    adjusted = {}
    for st, (_, pv) in tests.items():   # one Holm family per stratum
        adjusted.update({(st,) + k: v for k, v in holm(pv).items()})
    n_family = len(next(iter(tests.values()))[1])

    out = args.out_dir or PROJECT_ROOT / "evaluation/benchmarks/baseline_ladder" / manifest["name"]
    out.mkdir(parents=True, exist_ok=True)
    stats = manifest["stats"]["test"]
    order = sorted(table, key=lambda n: -table[n]["all"]["auroc"])
    lines = [f"# Baseline ladder - {manifest['name']}", "",
             f"Label: {manifest['label']['definition']}.", "",
             f"SANPO official test split: {stats['sessions']} sessions ({stats['sessions_with_encounter']} with an "
             f"arrival), {stats['objects']} objects, positive rate {stats['positive_rate']:.3f}. Learned methods are "
             f"averaged over their seeds; tabular baselines are fitted on the {manifest['stats']['train']['sessions']} "
             f"official-train sessions. Generated {time.strftime('%Y-%m-%d')} by `python -m models.GRU.benchmark` "
             f"({time.time() - started:.0f} s).", "",
             "## All objects", "",
             "| Method | AUROC | AUPRC | arrival_order | soonest_top1 | flicker ↓ | AUROC static | AUROC dynamic | AUPRC static | AUPRC dynamic |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in order:
        r = table[n]
        lines.append(f"| {n} | {r['all']['auroc']:.3f} | {r['all']['auprc']:.3f} | {r['all']['arrival_order']:.3f} | "
                     f"{r['all']['soonest_top1']:.3f} | {r['all']['flicker_rate']:.3f} | {r['static']['auroc']:.3f} | "
                     f"{r['dynamic']['auroc']:.3f} | {r['static']['auprc']:.3f} | {r['dynamic']['auprc']:.3f} |")
    s0 = table[order[0]]
    lines += ["", f"Strata: static classes {s0['static']['objects']} objects (positive rate "
                  f"{s0['static']['positive_rate']:.3f}), dynamic {s0['dynamic']['objects']} "
                  f"({s0['dynamic']['positive_rate']:.3f}).", "",
              f"## Method minus {args.reference} (95% paired session-bootstrap CI; two-sided p from the bootstrap "
              f"SE; Holm within each stratum, {n_family} tests)", "",
              "| Stratum | Method | ΔAUROC | p (Holm) | ΔAUPRC | p (Holm) |", "|---|---|---|---|---|---|"]
    for stratum, (rows, pv) in tests.items():
        for n in order:
            if n == args.reference:
                continue
            a, b = rows[(n, "auroc")], rows[(n, "auprc")]
            fmt = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"   # noqa: E731
            lines.append(f"| {stratum} | {n} | {fmt(a)} | {adjusted[(stratum, n, 'auroc')]:.3f} | {fmt(b)} | "
                         f"{adjusted[(stratum, n, 'auprc')]:.3f} |")
    lines += ["", "A difference is credible only if its Holm-adjusted p < 0.05. Static = traffic light, stop sign, "
              "fire hydrant, pole, bollard, stairs, crosswalk, pothole, puddle, bench; dynamic = the rest.", ""]
    report = "\n".join(lines)
    (out / "report.md").write_text(report, encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"table": table, "tests": {
        st: {f"{m}|{metric}": {"diff": rows[(m, metric)], "p": pv[(m, metric)], "p_holm": adjusted[(st, m, metric)]}
             for (m, metric) in rows} for st, (rows, pv) in tests.items()}}, indent=2), encoding="utf-8")
    print(report.encode("ascii", "replace").decode())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
