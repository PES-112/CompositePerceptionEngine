"""
vlm_referee.py
==============
Blinded multi-VLM referee on announcement order: on frames where engines pick
different top objects, which object should the walker hear about? Protocol v2
(2026-10-17, `reports/CPE v2 next steps.md` move 9; docs/methodology.md §4.7).

Cases. `<run-dir>/cases.json` (blind: session, frame, objects with track_id,
class, distance_m, closing_speed_ms, bearing_deg, bbox) and `cases_key.json`
([{case_id, picks: {engine: track_id}}]). The v1 export of
evaluation/kinetic_ablation.py (`disagreements.json` + `disagreements_key.json`,
baseline_pick / variant_pick) is read too.

What changed from v1, and why (each from the judge literature in
research_notes/CPE v2 next steps/evaluation_methodology.md):
  1. Forced choice over NUMBERED boxes. Objects are numbered 1..k on the image
     and in the list, in a seeded random order. The referee answers a number.
  2. Order swap. Every case is asked twice, the second time with the order
     (and so the numbers) reversed. Only order-consistent picks count; the
     consistency rate is reported (position bias is the commonest judge
     failure, GPT-4 was consistent on 65% of swapped pairs).
  3. Text-only control. The same list without the image (`--condition text`).
     If a referee agrees with itself across conditions as often as with
     humans, it is reading the distance column, not the scene.
  4. Human ceiling. Two people label the same >= 50 cases
     (human_labels_<name>.json); human-human agreement bounds what any
     referee can reach.
  5. Size. --sample draws n cases (default 250) stratified by session; at
     p_o ~ 0.8 with ~3 candidates, 100 cases give a 95% CI of about +/-0.12,
     250 about +/-0.08.
  6. Coefficients. Raw agreement against the per-frame chance level mean(1/k);
     Gwet's AC1 (robust where one answer dominates, unlike Cohen's kappa, whose
     chance term explodes at high prevalence - the "kappa paradox"); and
     Krippendorff's alpha (nominal) across the referee panel, which handles
     missing ballots. For both, a pick is coded by its distance rank in the
     frame (1 = nearest), so categories mean the same thing across frames.
     Cohen's kappa is kept only for binary reductions ("picked engine E's
     object?"). Every coefficient gets a 95% CI from 1,000 session-cluster
     bootstrap draws.
  7. Pre-registration. The first run writes protocol.json (sample, seed,
     conditions, coefficients, gate) and later runs refuse to change it. The
     gate: a referee's results stand in for human labels only if the lower 95%
     bound of its AC1 against every human is > 0.4.

Everything runs locally: each referee is an OpenAI-compatible
/chat/completions server (vLLM, Ollama, llama.cpp, LM Studio, SGLang).

    python evaluation/vlm_referee.py --run-dir RUN --frames-dir data/processed/<frames> --referee qwen2.5-vl
    python evaluation/vlm_referee.py --run-dir RUN --frames-dir ... --referee qwen2.5-vl --condition text
    python evaluation/vlm_referee.py --run-dir RUN --frames-dir ... --human-template   # numbered images + blank sheet
    python evaluation/vlm_referee.py --run-dir RUN --score-only
    python evaluation/vlm_referee.py --self-check
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GCS_BASE = "https://storage.googleapis.com/gresearch/"

LOCAL_REFEREES: dict[str, tuple[str, str]] = {
    "qwen2.5-vl": ("http://localhost:8001/v1", "Qwen/Qwen2.5-VL-7B-Instruct"),
    "internvl3":  ("http://localhost:8002/v1", "OpenGVLab/InternVL3-8B"),
    "gemma3":     ("http://localhost:8003/v1", "google/gemma-3-12b-it"),
}
CONDITIONS = ("image", "text")
ORDERS = ("fwd", "rev")
DEFAULT_SAMPLE = 250
N_BOOT = 1000
GATE_AC1_LOWER = 0.4
MIN_HUMAN_OVERLAP = 50

PROMPT = """You are helping a blind pedestrian who is walking forward with a head-mounted camera.

{where} numbered 1 to {k}. Distances are metric depth in metres; closing speed is positive when the object is getting closer; bearing is degrees from straight ahead (negative = left).

{objects}

Question: which ONE numbered object should the pedestrian be warned about first, right now?

Answer with JSON only, no other text:
{{"number": <1 to {k}>, "reason": "<one short sentence>"}}"""
WHERE = {"image": "Objects detected in this frame are drawn on the image as numbered boxes and listed below,",
         "text": "Objects detected in this frame are listed below (no image is available),"}


# ── Cases ────────────────────────────────────────────────────────────────────

def load_cases(run_dir: Path) -> tuple[list[dict], dict[str, dict[str, str]]]:
    """(blind cases, {case_id: {engine: picked track_id}}), from the v2 or the v1 export."""
    if (run_dir / "cases.json").exists():
        cases = json.loads((run_dir / "cases.json").read_text(encoding="utf-8"))
        key = {e["case_id"]: e["picks"] for e in json.loads((run_dir / "cases_key.json").read_text(encoding="utf-8"))}
        return cases, key
    cases = json.loads((run_dir / "disagreements.json").read_text(encoding="utf-8"))
    key: dict[str, dict[str, str]] = defaultdict(dict)
    for e in json.loads((run_dir / "disagreements_key.json").read_text(encoding="utf-8")):
        key[e["case_id"]]["K0"] = e["baseline_pick"]
        key[e["case_id"]][e["variant"]] = e["variant_pick"]
    return cases, dict(key)


def sample_cases(cases: list[dict], n: int, seed: int) -> list[dict]:
    """n cases, dealt round-robin across sessions (shuffled within each), so no session dominates."""
    rng = random.Random(seed)
    by: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by[c["session_id"]].append(c)
    pools = [rng.sample(v, len(v)) for _, v in sorted(by.items())]
    rng.shuffle(pools)
    out = []
    while len(out) < n and any(pools):
        for pool in pools:
            if pool and len(out) < n:
                out.append(pool.pop())
    return sorted(out, key=lambda c: c["case_id"])


def order(case: dict, which: str, seed: int) -> list[dict]:
    """The case's objects in presentation order: a seeded shuffle ("fwd") or its reverse ("rev")."""
    objs = sorted(case["objects"], key=lambda o: str(o["track_id"]))
    random.Random(f"{seed}:{case['case_id']}").shuffle(objs)
    return objs if which == "fwd" else objs[::-1]


def distance_rank(case: dict) -> dict[str, int]:
    """track_id -> 1 for the nearest object, 2 for the next, ... (the coding for AC1 and alpha)."""
    ranked = sorted(case["objects"], key=lambda o: (float(o["distance_m"]), str(o["track_id"])))
    return {o["track_id"]: i + 1 for i, o in enumerate(ranked)}


def numbered_table(objs: list[dict]) -> str:
    return "\n".join(f"{i}. {o['class']}, {o['distance_m']} m away, closing at {o['closing_speed_ms']} m/s, "
                     f"bearing {o['bearing_deg']}°" for i, o in enumerate(objs, 1))


def prompt_for(objs: list[dict], condition: str) -> str:
    return PROMPT.format(where=WHERE[condition], k=len(objs), objects=numbered_table(objs))


# ── Frames ───────────────────────────────────────────────────────────────────

def frame_url(frames_dir: Path, session_id: str, frame_idx: int) -> str:
    manifest = json.loads((frames_dir / f"{session_id}.frames.json").read_text(encoding="utf-8"))
    return GCS_BASE + urllib.parse.quote(manifest["rgb_objects"][frame_idx])


def numbered_frame_png(url: str, objs: list[dict], cache: Path) -> bytes:
    """The frame with one box per object, labelled 1..k in presentation order, in a large filled tag."""
    from PIL import Image, ImageDraw, ImageFont

    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            cache.write_bytes(response.read())
    image = Image.open(io.BytesIO(cache.read_bytes())).convert("RGB")
    draw = ImageDraw.Draw(image)
    size = max(18, image.height // 30)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    for i, o in enumerate(objs, 1):
        x1, y1, x2, y2 = o["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        tag = [x1, max(0, y1 - size - 6), x1 + size * (len(str(i)) * 0.7 + 0.6), max(size + 6, y1)]
        draw.rectangle(tag, fill=(255, 0, 0))
        draw.text((tag[0] + 4, tag[1] + 2), str(i), fill=(255, 255, 255), font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def object_table(objects: list[dict]) -> str:
    """Legacy (protocol v1) list keyed by object_id; still used by evaluation/topk_threat_validation.py."""
    return "\n".join(f"- {o['object_id']}: {o['class']}, {o['distance_m']} m away, "
                     f"closing at {o['closing_speed_ms']} m/s, bearing {o['bearing_deg']}°" for o in objects)


def annotated_frame_png(url: str, objects: list[dict], cache: Path) -> bytes:
    """Legacy (protocol v1) boxes labelled with object_id; still used by evaluation/topk_threat_validation.py."""
    from PIL import Image, ImageDraw

    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            cache.write_bytes(response.read())
    image = Image.open(io.BytesIO(cache.read_bytes())).convert("RGB")
    draw = ImageDraw.Draw(image)
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1 + 4, max(0, y1 - 12)), obj["object_id"], fill=(255, 255, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ── Referees ─────────────────────────────────────────────────────────────────

def ask_local(endpoint: str, model: str, prompt: str, png: bytes | None, timeout: int = 300) -> str:
    """One OpenAI-compatible /chat/completions call; temperature 0 so a re-run gives the same ballot."""
    content = [{"type": "text", "text": prompt}]
    if png is not None:
        content.insert(0, {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.standard_b64encode(png).decode()}})
    body = json.dumps({"model": model, "max_tokens": 512, "temperature": 0.0,
                       "messages": [{"role": "user", "content": content}]}).encode()
    request = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def parse_number(text: str, k: int) -> int | None:
    """The 1..k number in a referee reply, tolerating fenced JSON; None if absent or out of range."""
    if not text:
        return None
    body = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        pick = json.loads(body[body.index("{"): body.rindex("}") + 1]).get("number")
        pick = int(pick)
    except (ValueError, AttributeError, TypeError):
        return None
    return pick if 1 <= pick <= k else None


def consistent(ballot: dict | None) -> str | None:
    """The track picked in both orders, or None (missing, unparsed or order-inconsistent)."""
    if not ballot or ballot.get("fwd") is None or ballot.get("fwd") != ballot.get("rev"):
        return None
    return ballot["fwd"]


# ── Agreement coefficients ───────────────────────────────────────────────────

def raw_agreement(a: list, b: list) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    return sum(x == y for x, y in pairs) / len(pairs) if pairs else float("nan")


def gwet_ac1(a: list, b: list) -> float:
    """Gwet's AC1 for two raters, nominal categories; items either rater skipped are dropped."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return float("nan")
    cats = sorted({c for p in pairs for c in p})
    q = len(cats)
    p_o = sum(x == y for x, y in pairs) / len(pairs)
    if q < 2:
        return 1.0 if p_o == 1.0 else float("nan")
    pi = {c: (sum(x == c for x, _ in pairs) + sum(y == c for _, y in pairs)) / (2 * len(pairs)) for c in cats}
    p_e = sum(v * (1 - v) for v in pi.values()) / (q - 1)
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else float("nan")


def krippendorff_alpha(ratings: list[list]) -> float:
    """Krippendorff's alpha, nominal: ratings[unit] = one value per rater (None = missing)."""
    units = [[v for v in u if v is not None] for u in ratings]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")
    coinc: dict[tuple, float] = defaultdict(float)
    for u in units:
        m = len(u)
        for i, x in enumerate(u):
            for j, y in enumerate(u):
                if i != j:
                    coinc[(x, y)] += 1.0 / (m - 1)
    n_c: dict = defaultdict(float)
    for (x, _), w in coinc.items():
        n_c[x] += w
    n = sum(n_c.values())
    d_o = sum(w for (x, y), w in coinc.items() if x != y)
    d_e = sum(n_c[x] * n_c[y] for x in n_c for y in n_c if x != y) / (n - 1)
    return 1.0 - d_o / d_e if d_e > 0 else float("nan")


def cohens_kappa(a: list, b: list) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return float("nan")
    n = len(pairs)
    p_o = sum(x == y for x, y in pairs) / n
    labels = {v for p in pairs for v in p}
    p_e = sum((sum(x == v for x, _ in pairs) / n) * (sum(y == v for _, y in pairs) / n) for v in labels)
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else float("nan")


def bootstrap_ci(fn, sessions: list[str], seed: int = 0) -> tuple[float, float, float]:
    """(estimate, 2.5%, 97.5%) of fn(list of case indices), resampling whole sessions."""
    by: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sessions):
        by[s].append(i)
    ids = sorted(by)
    full = fn(list(range(len(sessions))))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(N_BOOT):
        pick = [i for j in rng.integers(0, len(ids), len(ids)) for i in by[ids[j]]]
        draws.append(fn(pick))
    draws = np.array(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return full, float("nan"), float("nan")
    return full, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ── Scoring ──────────────────────────────────────────────────────────────────

def score(cases: list[dict], key: dict[str, dict[str, str]], votes: dict[str, dict[str, dict]],
          humans: dict[str, dict[str, str]]) -> dict:
    """
    votes[f"{referee}|{condition}"][case_id] = {"fwd": track, "rev": track}
    humans[name][case_id] = track
    """
    sessions = [c["session_id"] for c in cases]
    ids = [c["case_id"] for c in cases]
    rank = {c["case_id"]: distance_rank(c) for c in cases}
    chance = float(np.mean([1 / len(c["objects"]) for c in cases])) if cases else float("nan")
    picks = {name: [consistent(v.get(cid)) for cid in ids] for name, v in votes.items()}
    raters = {**picks, **{f"human:{h}": [lab.get(cid) for cid in ids] for h, lab in humans.items()}}
    coded = {r: [rank[cid].get(t) if t is not None else None for cid, t in zip(ids, v)] for r, v in raters.items()}
    engines = sorted({e for p in key.values() for e in p})

    def sub(xs, idx):
        return [xs[i] for i in idx]

    out: dict = {"n_cases": len(cases), "n_sessions": len(set(sessions)), "chance_mean_1_over_k": chance,
                 "referees": {}, "pairs": {}}
    for name, v in votes.items():
        answered = [cid for cid in ids if v.get(cid, {}).get("fwd") is not None and v.get(cid, {}).get("rev") is not None]
        same = [cid for cid in answered if v[cid]["fwd"] == v[cid]["rev"]]
        win = {}
        for e in engines:
            def rate(idx, e=e):
                got = [(picks[name][i], key.get(ids[i], {}).get(e)) for i in idx]
                got = [(p, k) for p, k in got if p is not None and k is not None]
                return sum(p == k for p, k in got) / len(got) if got else float("nan")
            win[e] = bootstrap_ci(rate, sessions)
        out["referees"][name] = {"answered_both_orders": len(answered),
                                 "order_consistency": len(same) / len(answered) if answered else float("nan"),
                                 "consistent_picks": len(same), "engine_match_rate": win}

    names = sorted(raters)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            row = {"n": sum(x is not None and y is not None for x, y in zip(raters[a], raters[b]))}
            row["raw"] = bootstrap_ci(lambda idx: raw_agreement(sub(raters[a], idx), sub(raters[b], idx)), sessions)
            row["ac1"] = bootstrap_ci(lambda idx: gwet_ac1(sub(coded[a], idx), sub(coded[b], idx)), sessions)
            for e in engines:   # binary reduction: did each rater pick engine e's object?
                eb = [key.get(cid, {}).get(e) for cid in ids]
                ra = [None if x is None or k is None else x == k for x, k in zip(raters[a], eb)]
                rb = [None if y is None or k is None else y == k for y, k in zip(raters[b], eb)]
                row[f"kappa_picked_{e}"] = cohens_kappa(ra, rb)
            out["pairs"][f"{a} ~ {b}"] = row

    vlm = [n for n in names if not n.startswith("human:") and n.endswith("|image")]
    if len(vlm) >= 2:
        out["panel_alpha_image"] = bootstrap_ci(
            lambda idx: krippendorff_alpha([[coded[r][i] for r in vlm] for i in idx]), sessions)
    hum = [n for n in names if n.startswith("human:")]
    if len(hum) >= 2:
        out["human_ceiling"] = {f"{a} ~ {b}": out["pairs"][f"{a} ~ {b}"] for i, a in enumerate(hum) for b in hum[i + 1:]}
    out["gate"] = {}
    for name in votes:
        lows = [out["pairs"][" ~ ".join(sorted([name, h]))]["ac1"][1] for h in hum]
        out["gate"][name] = (bool(lows) and all(np.isfinite(lows)) and min(lows) > GATE_AC1_LOWER,
                             min(lows) if lows else float("nan"))
    return out


def fmt(v) -> str:
    if isinstance(v, tuple):
        return "n/a" if not np.isfinite(v[0]) else f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.2f}"


def write_report(run_dir: Path, protocol: dict, results: dict) -> None:
    lines = ["# Blinded referee on announcement order (protocol v2)", "",
             f"{results['n_cases']} cases from {results['n_sessions']} sessions (sample seed {protocol['seed']}). "
             f"Chance level for picking one object at random: {results['chance_mean_1_over_k']:.2f}. "
             "A referee's pick counts only if it is the same object in both presentation orders. "
             "95% CIs from 1,000 session-cluster bootstrap draws.", "",
             "## Referees", "",
             "| Referee | condition | answered both orders | order-consistent | " +
             " | ".join(f"picked {e}" for e in protocol["engines"]) + " |",
             "|---|---|---:|---:|" + "---|" * len(protocol["engines"])]
    for name, r in sorted(results["referees"].items()):
        ref, cond = name.split("|")
        lines.append(f"| {ref} | {cond} | {r['answered_both_orders']} | {fmt(r['order_consistency'])} | " +
                     " | ".join(fmt(r["engine_match_rate"].get(e, (float('nan'),))) for e in protocol["engines"]) + " |")
    lines += ["", "## Agreement (raw, and Gwet's AC1 on distance-rank coding)", "",
              "| Pair | n | raw agreement | AC1 |", "|---|---:|---|---|"]
    for pair, r in results["pairs"].items():
        lines.append(f"| {pair} | {r['n']} | {fmt(r['raw'])} | {fmt(r['ac1'])} |")
    if "panel_alpha_image" in results:
        lines += ["", f"Krippendorff's α across the image-condition referees: {fmt(results['panel_alpha_image'])}."]
    lines += ["", "## Gate", "",
              f"A referee stands in for human labels only if the lower 95% bound of its AC1 against every human "
              f"is > {GATE_AC1_LOWER}.", ""]
    for name, (ok, low) in sorted(results["gate"].items()):
        lines.append(f"- `{name}`: {'PASS' if ok else 'FAIL'} (lowest lower bound {fmt(low)})")
    if not results["gate"] or not any(n.startswith("human:") for p in results["pairs"] for n in p.split(" ~ ")):
        lines.append("- No human labels yet: run --human-template; until then no referee result is evidence.")
    (run_dir / "referee_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Entry point ──────────────────────────────────────────────────────────────

def protocol_for(run_dir: Path, args, cases: list[dict], key: dict) -> dict:
    """Write protocol.json on the first run; afterwards refuse to change it."""
    proto = {"version": 2, "sample": args.sample, "seed": args.seed,
             "conditions": list(CONDITIONS), "orders": list(ORDERS),
             "coefficients": ["raw agreement vs mean(1/k)", "Gwet AC1 (distance-rank coding)",
                              "Krippendorff alpha nominal (panel)", "Cohen kappa (binary reductions only)"],
             "gate": f"AC1 lower 95% bound vs every human > {GATE_AC1_LOWER}",
             "min_human_overlap": MIN_HUMAN_OVERLAP,
             "engines": sorted({e for p in key.values() for e in p}),
             "case_ids": [c["case_id"] for c in cases]}
    path = run_dir / "protocol.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in ("sample", "seed", "case_ids")} != {k: proto[k] for k in ("sample", "seed", "case_ids")}:
            raise SystemExit(f"{path} is pre-registered with a different sample; use a new --run-dir")
        return old
    path.write_text(json.dumps(proto, indent=2) + "\n", encoding="utf-8")
    return proto


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--frames-dir", type=Path, help="Directory holding <session>.frames.json.")
    p.add_argument("--referee", action="append", default=None)
    p.add_argument("--condition", choices=CONDITIONS, default="image")
    p.add_argument("--model", action="append", default=[], metavar="NAME=ID")
    p.add_argument("--endpoint", action="append", default=[], metavar="NAME=URL")
    p.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Only the first N sampled cases (smoke test; not scored).")
    p.add_argument("--human-template", action="store_true")
    p.add_argument("--score-only", action="store_true")
    args = p.parse_args()

    all_cases, key = load_cases(args.run_dir)
    cases = sample_cases(all_cases, args.sample, args.seed)
    protocol = protocol_for(args.run_dir, args, cases, key)
    if args.limit:
        cases = cases[: args.limit]

    if args.human_template:
        if not args.frames_dir:
            raise SystemExit("--frames-dir is required to draw the numbered images")
        out_dir = args.run_dir / "human"
        for c in cases:
            objs = order(c, "fwd", args.seed)
            png = numbered_frame_png(frame_url(args.frames_dir, c["session_id"], c["frame_idx"]), objs,
                                     args.run_dir / "frames" / f"{c['session_id']}_{c['frame_idx']}.jpg")
            (out_dir / "images").mkdir(parents=True, exist_ok=True)
            (out_dir / "images" / f"{c['case_id'].replace(':', '_')}.png").write_bytes(png)
        sheet = {c["case_id"]: {"image": f"images/{c['case_id'].replace(':', '_')}.png",
                                "objects": numbered_table(order(c, "fwd", args.seed)), "number": None} for c in cases}
        (out_dir / "labels_template.json").write_text(json.dumps(sheet, indent=2) + "\n", encoding="utf-8")
        print(f"{len(sheet)} cases -> {out_dir}. Each labeller fills in 'number' independently and saves "
              f"human_labels_<name>.json in the run dir; at least {MIN_HUMAN_OVERLAP} cases must be labelled by two.")
        return

    endpoints = {k: v[0] for k, v in LOCAL_REFEREES.items()}
    models = {k: v[1] for k, v in LOCAL_REFEREES.items()}
    for target, overrides in ((endpoints, args.endpoint), (models, args.model)):
        for override in overrides:
            name, _, value = override.partition("=")
            target[name] = value

    if not args.score_only:
        if not args.frames_dir and args.condition == "image":
            raise SystemExit("--frames-dir is required for the image condition")
        for referee in args.referee or sorted(LOCAL_REFEREES):
            if referee not in endpoints or referee not in models:
                raise SystemExit(f"Referee {referee!r} needs both --endpoint and --model")
            path = args.run_dir / f"votes_{referee}_{args.condition}.json"
            done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            for i, c in enumerate(cases, 1):
                ballot = done.setdefault(c["case_id"], {})
                for o in ORDERS:
                    if o in ballot:
                        continue
                    objs = order(c, o, args.seed)
                    png = None
                    if args.condition == "image":
                        png = numbered_frame_png(frame_url(args.frames_dir, c["session_id"], c["frame_idx"]), objs,
                                                 args.run_dir / "frames" / f"{c['session_id']}_{c['frame_idx']}.jpg")
                    try:
                        n = parse_number(ask_local(endpoints[referee], models[referee],
                                                   prompt_for(objs, args.condition), png), len(objs))
                    except Exception as exc:      # one bad case must not lose the whole ballot
                        print(f"  {referee} {c['case_id']} {o}: {type(exc).__name__}: {exc}", flush=True)
                        n = None
                    ballot[o] = objs[n - 1]["track_id"] if n else None
                    path.write_text(json.dumps(done, indent=2) + "\n", encoding="utf-8")
                if i % 20 == 0:
                    print(f"  {referee} ({args.condition}): {i}/{len(cases)}", flush=True)
            ok = sum(consistent(b) is not None for b in done.values())
            print(f"{referee} ({args.condition}): {ok}/{len(done)} order-consistent ballots")

    if args.limit:
        return
    votes = {}
    for path in args.run_dir.glob("votes_*_*.json"):
        stem = path.stem.removeprefix("votes_")
        ref, _, cond = stem.rpartition("_")
        if cond in CONDITIONS:
            votes[f"{ref}|{cond}"] = json.loads(path.read_text(encoding="utf-8"))
    humans = {}
    by_id = {c["case_id"]: c for c in cases}
    for path in args.run_dir.glob("human_labels_*.json"):
        sheet = json.loads(path.read_text(encoding="utf-8"))
        humans[path.stem.removeprefix("human_labels_")] = {
            cid: order(by_id[cid], "fwd", protocol["seed"])[int(v["number"]) - 1]["track_id"]
            for cid, v in sheet.items() if cid in by_id and v.get("number")}
    if not votes and not humans:
        raise SystemExit("No votes or human labels found.")
    results = score(cases, key, votes, humans)
    (args.run_dir / "referee_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n",
                                                       encoding="utf-8")
    write_report(args.run_dir, protocol, results)
    print(f"Report -> {args.run_dir / 'referee_report.md'}")


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Runnable self-check: python evaluation/vlm_referee.py --self-check"""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            seen["path"] = self.path
            payload = json.dumps({"choices": [{"message": {"content": '{"number": 2}'}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    for png in (b"fake-png", None):
        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.handle_request, daemon=True).start()
        reply = ask_local(f"http://127.0.0.1:{server.server_port}/v1", "m", "pick", png)
        server.server_close()
        assert reply == '{"number": 2}' and seen["path"] == "/v1/chat/completions"
        parts = seen["messages"][0]["content"]
        assert len(parts) == (2 if png else 1) and parts[-1]["text"] == "pick"   # text-only: no image part

    assert parse_number('{"number": 2, "reason": "closest"}', 3) == 2
    assert parse_number('```json\n{"number": "1"}\n```', 3) == 1
    assert parse_number('{"number": 4}', 3) is None and parse_number("two", 3) is None

    case = {"case_id": "s:1", "session_id": "s", "frame_idx": 1, "objects": [
        {"track_id": t, "class": "person", "distance_m": d, "closing_speed_ms": 0.5, "bearing_deg": 0, "bbox": [0, 0, 1, 1]}
        for t, d in (("a", 4.0), ("b", 2.0), ("c", 6.0))]}
    f, r = order(case, "fwd", 0), order(case, "rev", 0)
    assert [o["track_id"] for o in f] == [o["track_id"] for o in r][::-1] and f == order(case, "fwd", 0)
    assert distance_rank(case) == {"b": 1, "a": 2, "c": 3}
    assert "3. " in prompt_for(f, "text") and "no image" in prompt_for(f, "text")
    assert consistent({"fwd": "a", "rev": "a"}) == "a" and consistent({"fwd": "a", "rev": "b"}) is None

    # Coefficients against known values.
    assert gwet_ac1([1, 2, 1, 2], [1, 2, 1, 2]) == 1.0
    a, b = [1] * 18 + [2, 1], [1] * 18 + [1, 2]          # high prevalence: kappa collapses, AC1 doesn't
    assert cohens_kappa(a, b) < 0 and gwet_ac1(a, b) > 0.85
    assert abs(krippendorff_alpha([[1, 1], [2, 2], [1, 1]]) - 1.0) < 1e-12
    # Krippendorff (2011) nominal example, 4 coders x 12 units with missing data: alpha = 0.743
    kd = [[1, 1, None, 1], [2, 2, 3, 2], [3, 3, 3, 3], [3, 3, 3, 3], [2, 2, 2, 2], [1, 2, 3, 4], [4, 4, 4, 4],
          [1, 1, 2, 1], [2, 2, 2, 2], [None, 5, 5, 5], [None, None, 1, 1], [None, None, 3, None]]
    assert abs(krippendorff_alpha(kd) - 0.743) < 1e-3, krippendorff_alpha(kd)

    # Scoring end to end: one referee consistent on 2 of 3 cases, two humans.
    cases = [{**case, "case_id": f"s{i}:1", "session_id": f"s{i}"} for i in range(3)]
    key = {c["case_id"]: {"v2": "b", "K0": "c"} for c in cases}
    votes = {"q|image": {"s0:1": {"fwd": "b", "rev": "b"}, "s1:1": {"fwd": "b", "rev": "a"},
                         "s2:1": {"fwd": "c", "rev": "c"}}}
    humans = {"x": {"s0:1": "b", "s1:1": "b", "s2:1": "c"}, "y": {"s0:1": "b", "s1:1": "a", "s2:1": "c"}}
    res = score(cases, key, votes, humans)
    q = res["referees"]["q|image"]
    assert q["order_consistency"] == 2 / 3 and q["engine_match_rate"]["v2"][0] == 0.5
    assert res["pairs"]["human:x ~ q|image"]["raw"][0] == 1.0
    assert "human:x ~ human:y" in res["human_ceiling"]
    print("vlm_referee.py self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _demo()
    else:
        main()
