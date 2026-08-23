"""
topk_threat_validation.py
==========================
Per-scene K0 top-3 threat ranking, validated by three blinded local VLMs.

Where kinetic_ablation.py asks "does removing a term change K0's top-1 pick"
and vlm_referee.py referees K0 against ablation arms on the frames where they
disagree, this script asks a different question with no second formula
involved: for ordinary scenes (not disagreement frames), does an independent
VLM even agree with K0 on *which three* objects are the top threats? The VLMs
are the only check here, so the usual caveats apply harder than usual — see
docs/ablation_guide.md §4c: without human_labels.json, three VLMs agreeing
with K0 is still just a shared prior, not evidence.

Two stages, matching kinetic_ablation.py / vlm_referee.py's split:

    # 1. Build blind scenes from Stage-1 CSVs (needs the finetuned checkpoint —
    #    see tools/stream_sanpo_perception.py's --model default)
    python evaluation/topk_threat_validation.py \\
        --run-dir evaluation/benchmarks/topk_threat_eval/run_2026_08_20 \\
        --csv-dir data/processed/ablation_30pct

    # 2. Serve three VLMs (docs/ablation_guide.md §4a), then judge
    python evaluation/topk_threat_validation.py \\
        --run-dir evaluation/benchmarks/topk_threat_eval/run_2026_08_20 \\
        --frames-dir data/processed/ablation_30pct \\
        --referee qwen2.5-vl --referee internvl3 --referee gemma3

    python evaluation/topk_threat_validation.py --run-dir <dir> --human-template
    python evaluation/topk_threat_validation.py --run-dir <dir> --score-only

Output: cases.json (blind), cases_key.json (K0's top-3, held back from
referees), votes_<referee>.json, results.json, topk_report.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

import kinetic_ablation      # sibling module: session CSV loading + session bootstrap CI
import vlm_referee           # sibling module: local VLM plumbing + frame annotation

# Load physics.py directly, same as kinetic_ablation.py — avoids pulling in
# ultralytics/torch through src/perception_stack/__init__.py for an analysis
# script that only needs the scoring function.
_spec = importlib.util.spec_from_file_location(
    "cpe_physics", PROJECT_ROOT / "src/perception_stack/physics.py"
)
physics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(physics)

TOP_K = 3
MAJORITY_KEY = "vlm_majority"   # reserved pseudo-referee name — never a real --referee value

PROMPT = """You are assessing a single video frame captured by a camera worn by a blind pedestrian walking forward.

Objects detected in this frame are drawn on the image and listed below. Distances are metric depth in metres; closing speed is positive when the object is getting closer; bearing is degrees from straight ahead (negative = left).

{objects}

Question: which THREE objects are the most important threats to warn this pedestrian about right now? Rank them most urgent first.

Answer with JSON only, no other text:
{{"top3": ["<id>", "<id>", "<id>"]}}"""


# ── Scene selection + K0 scoring ─────────────────────────────────────────────

def k0_scores(g: pd.DataFrame) -> np.ndarray:
    """K0 exactly as shipped: physics.kinetic_score with its own defaults."""
    return np.array([
        physics.kinetic_score(r["distance_m"], r["velocity_ms"], r["class"])
        for _, r in g.iterrows()
    ], dtype=float)


def build_cases(csv_dir: Path, min_objects: int, scenes_per_session: int,
                 max_scenes: int | None, seed: int) -> tuple[list, list]:
    """
    Sample scenes with >= min_objects detections (so K0's top-3 is actually a
    choice among alternatives, not just "every object in the frame"), score
    each with K0, and export the blind object list plus a held-back key.
    """
    csv_paths = sorted(p for p in csv_dir.glob("*.csv") if not p.name.endswith(".partial"))
    if not csv_paths:
        raise SystemExit(f"No session CSVs in {csv_dir} — run tools/stream_sanpo_perception.py first.")

    rng = random.Random(seed)
    blind, key = [], []
    for path in csv_paths:
        sid = path.stem
        df = kinetic_ablation.load_session(path)
        groups = [(idx, g) for idx, g in df.groupby("frame_idx", sort=True) if len(g) >= min_objects]
        sample = rng.sample(groups, min(scenes_per_session, len(groups)))
        sample.sort(key=lambda t: t[0])

        for frame_idx, g in sample:
            scores = k0_scores(g)
            if not np.isfinite(scores).any():
                continue

            objects = [{
                "object_id": f"O{n}",
                "track_id": r["track_id"],
                "class": r["class"],
                "distance_m": round(float(r["distance_m"]), 2),
                "closing_speed_ms": round(float(r["velocity_ms"]), 2),
                "bearing_deg": round(float(r["bearing_deg"]), 1),
                "bbox": [int(r["bbox_x1"]), int(r["bbox_y1"]), int(r["bbox_x2"]), int(r["bbox_y2"])],
            } for n, (_, r) in enumerate(g.iterrows())]
            id_by_track = {o["track_id"]: o["object_id"] for o in objects}

            order = np.argsort(-scores)[:TOP_K]
            k0_top3 = [id_by_track[g.iloc[i]["track_id"]] for i in order]
            k0_top3_tracks = [g.iloc[i]["track_id"] for i in order]

            rng.shuffle(objects)   # display order must not leak K0's ranking

            case_id = f"{sid}:{int(frame_idx)}"
            blind.append({"case_id": case_id, "session_id": sid,
                          "frame_idx": int(frame_idx), "objects": objects})
            key.append({"case_id": case_id, "k0_top3": k0_top3,
                        "k0_top3_track_ids": k0_top3_tracks, "n_objects": len(objects)})

    if max_scenes and len(blind) > max_scenes:
        idxs = sorted(rng.sample(range(len(blind)), max_scenes))
        blind = [blind[i] for i in idxs]
        keep = {b["case_id"] for b in blind}
        key = [k_ for k_ in key if k_["case_id"] in keep]

    return blind, key


# ── Ballot parsing ────────────────────────────────────────────────────────────

def parse_vote_topk(text: str, valid_ids: set[str], k: int = TOP_K) -> list[str] | None:
    """
    Pull a ranked top-k list out of a referee reply. Any hallucinated id or a
    repeated id invalidates the whole ballot — same "None, not a guess" policy
    as vlm_referee.parse_vote — but a shorter-than-k list of valid, unique ids
    is accepted as a partial ballot rather than rejected outright.
    """
    if not text:
        return None
    body = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        picks = json.loads(body[body.index("{"): body.rindex("}") + 1]).get("top3")
    except (ValueError, AttributeError):
        return None
    if not isinstance(picks, list) or not picks:
        return None

    out: list[str] = []
    for p in picks[:k]:
        if p not in valid_ids or p in out:
            return None
        out.append(p)
    return out


# ── Scoring ──────────────────────────────────────────────────────────────────

def _jaccard(a: list[str] | None, b: list[str] | None) -> float | None:
    if not a or not b:
        return None
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def majority_ballot(ballots: list[list[str]], k: int = TOP_K) -> list[str] | None:
    """
    Combine several VLMs' top-k ballots into one "what most VLMs agreed on"
    ballot: an object only makes it in if named by a strict majority of the
    ballots present, ranked by vote count and then by how early it tended to
    be ranked. Needs at least 2 ballots — one VLM's answer isn't a majority of
    anything.
    """
    if len(ballots) < 2:
        return None
    threshold = len(ballots) // 2 + 1
    counts: dict[str, int] = defaultdict(int)
    rank_sum: dict[str, int] = defaultdict(int)
    for ballot in ballots:
        for pos, obj in enumerate(ballot):
            counts[obj] += 1
            rank_sum[obj] += pos
    qualifying = [o for o, c in counts.items() if c >= threshold]
    if not qualifying:
        return None
    qualifying.sort(key=lambda o: (-counts[o], rank_sum[o] / counts[o]))
    return qualifying[:k]


def add_majority_referee(cases: list[dict], votes: dict[str, dict[str, list[str] | None]]) -> None:
    """Derive a MAJORITY_KEY pseudo-referee from the real referees already in `votes`, in place."""
    real_referees = list(votes)
    majority: dict[str, list[str] | None] = {}
    for case in cases:
        cid = case["case_id"]
        ballots = [votes[r][cid] for r in real_referees if votes[r].get(cid)]
        mb = majority_ballot(ballots)
        if mb is not None:
            majority[cid] = mb
    votes[MAJORITY_KEY] = majority


def score(run_dir: Path, votes: dict[str, dict[str, list[str] | None]],
          n_boot: int, seed: int) -> dict:
    """
    Two label-free-adjacent metrics per referee, both session-bootstrapped
    (kinetic_ablation.bootstrap_ci) because frames in a session are
    autocorrelated:

        overlap_at_3   — mean fraction of K0's 3 picks the VLM also named
                          (order-independent; "did it flag the same objects").
        top1_agreement — did the VLM's #1 pick match K0's #1 pick.
    """
    cases = {c["case_id"]: c for c in json.loads((run_dir / "cases.json").read_text())}
    key = {k_["case_id"]: k_ for k_ in json.loads((run_dir / "cases_key.json").read_text())}

    results: dict = {}
    for referee, ballots in votes.items():
        by_session_overlap: dict[str, list] = defaultdict(list)
        by_session_top1: dict[str, list] = defaultdict(list)
        n_valid = 0
        for case_id, case in cases.items():
            ballot = ballots.get(case_id)
            if not ballot:
                continue
            n_valid += 1
            k0_top3 = key[case_id]["k0_top3"]
            overlap = len(set(ballot) & set(k0_top3)) / len(k0_top3)
            top1 = 1.0 if ballot[0] == k0_top3[0] else 0.0
            by_session_overlap[case["session_id"]].append(overlap)
            by_session_top1[case["session_id"]].append(top1)

        sessions = sorted(by_session_overlap)
        weights = [len(by_session_overlap[s]) for s in sessions]
        results[referee] = {
            "n_cases": len(cases),
            "n_valid": n_valid,
            "overlap_at_3": kinetic_ablation.bootstrap_ci(
                [float(np.mean(by_session_overlap[s])) for s in sessions], weights, n_boot, seed),
            "top1_agreement": kinetic_ablation.bootstrap_ci(
                [float(np.mean(by_session_top1[s])) for s in sessions], weights, n_boot, seed),
        }

    # Referee reliability: mean Jaccard of the top-3 sets, over cases both answered.
    # MAJORITY_KEY is derived from these referees, so comparing it against them
    # here would just be circular — excluded from this pairwise table only.
    names = sorted(v for v in votes if v != MAJORITY_KEY)
    results["_referee_agreement"] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            js = [j for cid in cases
                  if (j := _jaccard(votes[a].get(cid), votes[b].get(cid))) is not None]
            results["_referee_agreement"][f"{a}|{b}"] = float(np.mean(js)) if js else float("nan")

    human_path = run_dir / "human_labels.json"
    if human_path.exists():
        human = json.loads(human_path.read_text())
        results["_human_agreement"] = {}
        for name in sorted(votes):   # MAJORITY_KEY included here — human vs. consensus is meaningful
            js = [j for cid in cases
                  if (j := _jaccard(votes[name].get(cid), human.get(cid))) is not None]
            results["_human_agreement"][name] = float(np.mean(js)) if js else float("nan")
    else:
        results["_human_agreement"] = "MISSING — run --human-template, label cases, save as human_labels.json"
    return results


def write_report(run_dir: Path, results: dict) -> None:
    lines = [
        "# Top-3 Threat Validation — K0 vs Blinded VLMs", "",
        f"K0's top-{TOP_K} most-urgent objects per scene, checked against each VLM's "
        "independently-chosen top-3 — no scores, formula name, or K0's picks are shown to the "
        "referee. `overlap_at_3` is the fraction of K0's 3 picks the VLM also named (any order); "
        "`top1_agreement` is whether the VLM's #1 pick matches K0's #1 pick. Both are a 95% "
        "session-level bootstrap CI — see docs/ablation_guide.md §3.", "",
        "Until `_human_agreement` is populated, these numbers show whether the VLMs agree with "
        "K0, not whether K0 is right: three VLMs agreeing can be one shared training prior three "
        "times over (docs/ablation_guide.md §4c).", "",
        f"`{MAJORITY_KEY}` is not a fourth VLM — it's the objects a strict majority (≥2 of the "
        "VLMs that answered) named in their top-3, combined into one consensus ballot per scene. "
        "It answers 'how often does K0 match what most VLMs agreed on', which is a steadier "
        "signal than any single model's row above it. Excluded from the referee-agreement table "
        "below since it's derived from those referees, not independent of them.", "",
        "| Referee | overlap_at_3 | 95% CI | top1_agreement | 95% CI | valid votes |",
        "|---|---:|---|---:|---|---:|",
    ]
    for referee, r in results.items():
        if referee.startswith("_"):
            continue
        op, olo, ohi = r["overlap_at_3"]
        tp, tlo, thi = r["top1_agreement"]
        lines.append(f"| `{referee}` | {op:.2f} | [{olo:.2f}, {ohi:.2f}] | "
                     f"{tp:.2f} | [{tlo:.2f}, {thi:.2f}] | {r['n_valid']}/{r['n_cases']} |")
    lines += ["", "## Referee agreement (mean Jaccard of top-3 sets)", ""]
    lines += [f"- `{pair}`: {j:.2f}" for pair, j in results["_referee_agreement"].items()]
    lines += ["", "## Agreement with human labels (mean Jaccard)", ""]
    human = results["_human_agreement"]
    if isinstance(human, dict):
        lines += [f"- `{name}`: {j:.2f}" for name, j in human.items()]
    else:
        lines.append(human)
    lines.append("")
    (run_dir / "topk_report.md").write_text("\n".join(lines), encoding="utf-8")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="K0 top-3 threat ranking per scene, validated by blinded local VLMs.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--csv-dir", type=Path,
                   help="Build scenes from Stage-1 CSVs and exit — run this before --referee.")
    p.add_argument("--frames-dir", type=Path, help="Directory holding <session>.frames.json.")
    p.add_argument("--min-objects", type=int, default=5,
                   help="Scenes need at least this many detections, so K0's top-3 is a real "
                        "subset choice rather than just 'every object in the frame'.")
    p.add_argument("--scenes-per-session", type=int, default=8)
    p.add_argument("--max-scenes", type=int, default=None, help="Global cap (referee budget).")
    p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--referee", action="append", default=None,
                   help=f"Referee to run; repeatable. Known: {', '.join(sorted(vlm_referee.LOCAL_REFEREES))}.")
    p.add_argument("--model", action="append", default=[], metavar="NAME=ID")
    p.add_argument("--endpoint", action="append", default=[], metavar="NAME=URL")
    p.add_argument("--limit", type=int, default=None, help="Only judge the first N scenes (smoke test).")
    p.add_argument("--human-template", action="store_true")
    p.add_argument("--score-only", action="store_true")
    args = p.parse_args()

    if args.csv_dir:
        blind, key = build_cases(args.csv_dir, args.min_objects, args.scenes_per_session,
                                  args.max_scenes, args.seed)
        args.run_dir.mkdir(parents=True, exist_ok=True)
        (args.run_dir / "cases.json").write_text(json.dumps(blind, indent=2) + "\n")
        (args.run_dir / "cases_key.json").write_text(json.dumps(key, indent=2) + "\n")
        print(f"{len(blind)} scenes -> {args.run_dir}")
        return

    cases_path = args.run_dir / "cases.json"
    if not cases_path.exists():
        raise SystemExit(f"No cases.json in {args.run_dir} — run with --csv-dir first.")
    all_cases = json.loads(cases_path.read_text())
    blind = all_cases[: args.limit] if args.limit else all_cases

    if args.human_template:
        template = {c["case_id"]: [] for c in blind}
        out = args.run_dir / "human_labels_template.json"
        out.write_text(json.dumps(template, indent=2) + "\n")
        print(f"{len(template)} blank cases -> {out}\n"
              f"Fill in up to {TOP_K} object_ids each, most urgent first, save as human_labels.json.")
        return

    endpoints = {k: v[0] for k, v in vlm_referee.LOCAL_REFEREES.items()}
    models = {k: v[1] for k, v in vlm_referee.LOCAL_REFEREES.items()}
    for target, overrides in ((endpoints, args.endpoint), (models, args.model)):
        for override in overrides:
            name, _, value = override.partition("=")
            target[name] = value
    for referee in args.referee or sorted(vlm_referee.LOCAL_REFEREES):
        if referee not in endpoints or referee not in models:
            raise SystemExit(f"Referee {referee!r} needs both --endpoint and --model")

    votes: dict[str, dict[str, list[str] | None]] = {}
    if not args.score_only:
        if not args.frames_dir:
            raise SystemExit("--frames-dir is required unless --score-only")
        for referee in args.referee or sorted(vlm_referee.LOCAL_REFEREES):
            votes_path = args.run_dir / f"votes_{referee}.json"
            done = json.loads(votes_path.read_text()) if votes_path.exists() else {}
            for i, case in enumerate(blind, 1):
                if case["case_id"] in done:
                    continue
                objects = case["objects"]
                png = vlm_referee.annotated_frame_png(
                    vlm_referee.frame_url(args.frames_dir, case["session_id"], case["frame_idx"]),
                    objects,
                    args.run_dir / "frames" / f"{case['session_id']}_{case['frame_idx']}.jpg",
                )
                prompt = PROMPT.format(objects=vlm_referee.object_table(objects))
                try:
                    reply = vlm_referee.ask_local(endpoints[referee], models[referee], prompt, png)
                    done[case["case_id"]] = parse_vote_topk(reply, {o["object_id"] for o in objects})
                except Exception as exc:      # one bad case must not lose the whole ballot
                    print(f"  {referee} {case['case_id']}: {type(exc).__name__}: {exc}", flush=True)
                    done[case["case_id"]] = None
                votes_path.write_text(json.dumps(done, indent=2) + "\n")
                if i % 20 == 0:
                    print(f"  {referee}: {i}/{len(blind)}", flush=True)
            votes[referee] = done
            print(f"{referee}: {sum(v is not None for v in done.values())}/{len(done)} valid votes")

    for votes_path in args.run_dir.glob("votes_*.json"):
        votes.setdefault(votes_path.stem.removeprefix("votes_"), json.loads(votes_path.read_text()))
    if not votes:
        raise SystemExit("No votes found — run without --score-only first.")

    # Majority ballots are computed over every case with real votes on disk, not
    # just the (possibly --limit-truncated) set judged this invocation — score()
    # below reads the full cases.json too, so this must match or the majority
    # row would silently under-cover relative to the real referees' rows.
    add_majority_referee(all_cases, votes)
    results = score(args.run_dir, votes, args.n_boot, args.seed)
    (args.run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    write_report(args.run_dir, results)
    print(f"\nReport -> {args.run_dir / 'topk_report.md'}")


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Runnable self-check: python evaluation/topk_threat_validation.py --self-check"""
    import tempfile

    # ── K0 top-3 ordering on a synthetic 5-object frame ──
    rows = [
        {"frame_idx": 0, "track_id": "ped_close", "class": "person", "confidence": 0.9,
         "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 100, "bbox_y2": 100,
         "cx_px": 300, "bearing_deg": 2.0, "distance_m": 3.0, "velocity_ms": 2.0},
        {"frame_idx": 0, "track_id": "bike_close", "class": "bicycle", "confidence": 0.9,
         "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 80, "bbox_y2": 80,
         "cx_px": 250, "bearing_deg": -5.0, "distance_m": 5.0, "velocity_ms": 1.5},
        {"frame_idx": 0, "track_id": "car_mid", "class": "car", "confidence": 0.9,
         "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 150, "bbox_y2": 150,
         "cx_px": 400, "bearing_deg": 10.0, "distance_m": 15.0, "velocity_ms": 1.0},
        {"frame_idx": 0, "track_id": "bus_parked", "class": "bus", "confidence": 0.9,
         "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 200, "bbox_y2": 200,
         "cx_px": 100, "bearing_deg": -20.0, "distance_m": 20.0, "velocity_ms": 0.0},
        {"frame_idx": 0, "track_id": "pole_far", "class": "pole", "confidence": 0.9,
         "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 20, "bbox_y2": 40,
         "cx_px": 500, "bearing_deg": 25.0, "distance_m": 12.0, "velocity_ms": 0.0},
    ]
    df = kinetic_ablation.prepare(pd.DataFrame(rows))
    g = df[df["frame_idx"] == 0]
    scores = k0_scores(g)
    order = np.argsort(-scores)
    ranked = [g.iloc[i]["track_id"] for i in order]
    # K = severity(class) * v^2 / d — car's mass-derived severity (~4.6x a person)
    # outranks the closer, faster bicycle (~0.56x) despite the bicycle's kinematics.
    assert ranked[:3] == ["ped_close", "car_mid", "bike_close"], ranked

    # ── Ballot parsing ──
    ids = {"O0", "O1", "O2", "O3", "O4"}
    assert parse_vote_topk('{"top3": ["O0", "O1", "O2"]}', ids) == ["O0", "O1", "O2"]
    assert parse_vote_topk('```json\n{"top3": ["O2", "O0"]}\n```', ids) == ["O2", "O0"]   # partial ok
    assert parse_vote_topk('{"top3": ["O0", "O9", "O2"]}', ids) is None                   # hallucinated
    assert parse_vote_topk('{"top3": ["O0", "O0", "O2"]}', ids) is None                   # duplicate
    assert parse_vote_topk('{"top3": []}', ids) is None
    assert parse_vote_topk("no json here", ids) is None
    assert parse_vote_topk("", ids) is None

    # ── build_cases: min-objects filter + deterministic per-session sampling ──
    with tempfile.TemporaryDirectory() as tmp:
        csv_dir = Path(tmp)
        big_rows = []
        for f in range(3):
            for r in rows:                                 # 5 objects — passes min_objects=5
                big_rows.append({**r, "frame_idx": f})
            big_rows.append({                                # a 1-object frame — must be dropped
                "frame_idx": 10 + f, "track_id": "solo", "class": "person", "confidence": 0.9,
                "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 50, "bbox_y2": 50,
                "cx_px": 320, "bearing_deg": 0.0, "distance_m": 4.0, "velocity_ms": 1.0,
            })
        pd.DataFrame(big_rows).to_csv(csv_dir / "sess_a.csv", index=False)

        blind, key = build_cases(csv_dir, min_objects=5, scenes_per_session=8,
                                  max_scenes=None, seed=20260819)
        assert len(blind) == 3, len(blind)                  # only the 3 five-object frames
        assert all(c["n_objects"] == 5 for c in
                   [k_ for k_ in key]), key
        k0_top3 = key[0]["k0_top3"]
        assert len(k0_top3) == 3 and len(set(k0_top3)) == 3
        obj_by_id = {o["object_id"]: o for o in blind[0]["objects"]}
        assert {obj_by_id[i]["track_id"] for i in k0_top3} == {"ped_close", "car_mid", "bike_close"}

        blind2, _ = build_cases(csv_dir, min_objects=5, scenes_per_session=8,
                                 max_scenes=None, seed=20260819)
        assert [c["case_id"] for c in blind] == [c["case_id"] for c in blind2]   # seed reproducible

        # ── scoring: one referee agrees fully, one partially, one hallucinated ──
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        (run_dir / "cases.json").write_text(json.dumps(blind))
        (run_dir / "cases_key.json").write_text(json.dumps(key))
        top3 = key[0]["k0_top3"]
        votes = {
            "perfect": {blind[0]["case_id"]: list(top3)},
            "partial": {blind[0]["case_id"]: [top3[0], "O_wrong", top3[1]]
                        if "O_wrong" in {o["object_id"] for o in blind[0]["objects"]} else [top3[0]]},
        }
        # build a clean partial ballot: K0's #1 plus one non-K0 object
        other = next(o["object_id"] for o in blind[0]["objects"] if o["object_id"] not in top3)
        votes["partial"] = {blind[0]["case_id"]: [top3[0], other]}
        results = score(run_dir, votes, n_boot=100, seed=0)
        assert results["perfect"]["overlap_at_3"][0] == 1.0
        assert results["perfect"]["top1_agreement"][0] == 1.0
        assert abs(results["partial"]["overlap_at_3"][0] - (1 / 3)) < 1e-9
        assert results["partial"]["top1_agreement"][0] == 1.0
        assert isinstance(results["_human_agreement"], str)   # no human labels yet -> flagged

        # ── majority-vote pseudo-referee ──
        assert majority_ballot([["O0", "O1", "O2"]]) is None        # one ballot is never a majority
        assert majority_ballot([["O0", "O1", "O2"], ["O0", "D", "O1"]]) == ["O0", "O1"]
        assert majority_ballot([["A", "B", "C"], ["A", "D", "B"], ["E", "A", "F"]]) == ["A", "B"]

        other = next(o["object_id"] for o in blind[0]["objects"] if o["object_id"] not in top3)
        cid = blind[0]["case_id"]
        maj_votes = {
            "ref_a": {cid: [top3[0], top3[1], top3[2]]},
            "ref_b": {cid: [top3[0], top3[2], top3[1]]},
            "ref_c": {cid: [other, top3[0], top3[1]]},
        }
        add_majority_referee(blind, maj_votes)
        assert maj_votes[MAJORITY_KEY][cid] == top3, maj_votes[MAJORITY_KEY]

        maj_results = score(run_dir, maj_votes, n_boot=100, seed=0)
        assert maj_results[MAJORITY_KEY]["overlap_at_3"][0] == 1.0
        assert maj_results[MAJORITY_KEY]["top1_agreement"][0] == 1.0
        assert not any(MAJORITY_KEY in pair for pair in maj_results["_referee_agreement"])

    print("topk_threat_validation.py self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _demo()
    else:
        main()
