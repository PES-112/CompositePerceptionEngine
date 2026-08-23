"""
vlm_referee.py
==============
Blinded multi-VLM referee over the disagreement frames exported by
`evaluation/kinetic_ablation.py`.

The referees are **not** given a formula, a score, or a variant name. They see
the RGB frame with neutrally-labelled boxes plus a neutral object list, and
answer one question: *which object is the top threat?* The formula whose argmax
matches wins that frame. That makes the VLM an oracle rather than a contestant —
see docs/kinetic_score_opinion.md §3–§4.

Three referees from three model *families*, because agreement between models
that share training data is a shared prior, not truth. Even so, three agreeing
VLMs are not a ground truth: run `--human-template`, label 100–150 of the same
cases by hand, and report Cohen's κ between each VLM and the human. If κ is
poor, the VLM numbers do not belong in the paper.

Everything runs **locally** — no API keys, no data leaving the box. Each referee
is a local server speaking the OpenAI-compatible `/chat/completions` shape,
which vLLM, Ollama, llama.cpp's `llama-server`, LM Studio and SGLang all speak.
Serve one model per port (see `docs/ablation_guide.md`), or point all three
`--endpoint` flags at one port and swap models if you only have one GPU.

Usage
-----
    python evaluation/vlm_referee.py \\
        --run-dir evaluation/benchmarks/kinetic_score_eval/run_2026_08_19 \\
        --frames-dir data/processed/ablation_30pct \\
        --referee claude --referee gemini --referee gpt

    python evaluation/vlm_referee.py --run-dir <dir> --human-template   # blank sheet to label
    python evaluation/vlm_referee.py --run-dir <dir> --score-only       # aggregate existing votes

Override a referee's server or weights without editing this file:

    --endpoint qwen2.5-vl=http://localhost:8010/v1  --model qwen2.5-vl=<hf-id>
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GCS_BASE = "https://storage.googleapis.com/gresearch/"

# Three local VLMs from three different pretraining lineages — agreement between
# two checkpoints of the same family would measure a shared prior, not truth.
# One server per port; override with --endpoint/--model rather than editing.
LOCAL_REFEREES: dict[str, tuple[str, str]] = {
    "qwen2.5-vl": ("http://localhost:8001/v1", "Qwen/Qwen2.5-VL-7B-Instruct"),
    "internvl3":  ("http://localhost:8002/v1", "OpenGVLab/InternVL3-8B"),
    "gemma3":     ("http://localhost:8003/v1", "google/gemma-3-12b-it"),
}

PROMPT = """You are assessing a single video frame captured by a camera worn by a blind pedestrian walking forward.

Objects detected in this frame are drawn on the image and listed below. Distances are metric depth in metres; closing speed is positive when the object is getting closer; bearing is degrees from straight ahead (negative = left).

{objects}

Question: which ONE object is the most important threat to warn this pedestrian about right now?

Answer with JSON only, no other text:
{{"object_id": "<one of the ids above>", "reason": "<one short sentence>"}}"""


# ── Frame fetch and annotation ───────────────────────────────────────────────

def frame_url(frames_dir: Path, session_id: str, frame_idx: int) -> str:
    manifest = json.loads((frames_dir / f"{session_id}.frames.json").read_text())
    return GCS_BASE + urllib.parse.quote(manifest["rgb_objects"][frame_idx])


def annotated_frame_png(url: str, objects: list[dict], cache: Path) -> bytes:
    """
    Fetch the frame and draw one labelled box per object.

    The label is the neutral object_id, in list order, so nothing about either
    formula's ranking reaches the referee through the image.
    """
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


def object_table(objects: list[dict]) -> str:
    return "\n".join(
        f"- {o['object_id']}: {o['class']}, {o['distance_m']} m away, "
        f"closing at {o['closing_speed_ms']} m/s, bearing {o['bearing_deg']}°"
        for o in objects
    )


# ── Referees ─────────────────────────────────────────────────────────────────

def ask_local(endpoint: str, model: str, prompt: str, png: bytes, timeout: int = 300) -> str:
    """
    One OpenAI-compatible /chat/completions call against a local server.

    urllib rather than the openai SDK: this is a single unauthenticated POST to
    localhost and the SDK would be a dependency for nothing. temperature=0 so a
    re-run of the same case gives the same ballot.
    """
    body = json.dumps({
        "model": model,
        "max_tokens": 512,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.standard_b64encode(png).decode()}},
            {"type": "text", "text": prompt},
        ]}],
    }).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 # Some servers demand the header even though they ignore it.
                 "Authorization": "Bearer local"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def parse_vote(text: str, valid_ids: set[str]) -> str | None:
    """Pull the object_id out of a referee reply, tolerating fenced JSON."""
    if not text:
        return None
    body = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        pick = json.loads(body[body.index("{"): body.rindex("}") + 1]).get("object_id")
    except (ValueError, AttributeError):
        return None
    return pick if pick in valid_ids else None


# ── Scoring ──────────────────────────────────────────────────────────────────

def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Agreement between two raters, corrected for agreement by chance."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return float("nan")
    n = len(pairs)
    observed = sum(x == y for x, y in pairs) / n
    labels = {lbl for pair in pairs for lbl in pair}
    expected = sum(
        (sum(x == lbl for x, _ in pairs) / n) * (sum(y == lbl for _, y in pairs) / n)
        for lbl in labels
    )
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def score(run_dir: Path, votes: dict[str, dict[str, str]]) -> dict:
    """
    Win rate per variant against the baseline, on the frames where they differ.

    A frame counts for the baseline if the referee picked the baseline's object,
    for the variant if it picked the variant's, and for neither if it picked a
    third object — that third case is reported, not silently dropped, because a
    high rate of it means *both* formulas were wrong there.
    """
    blind = {c["case_id"]: c for c in json.loads((run_dir / "disagreements.json").read_text())}
    key = json.loads((run_dir / "disagreements_key.json").read_text())

    track_of = {
        c["case_id"]: {o["track_id"]: o["object_id"] for o in c["objects"]}
        for c in blind.values()
    }

    results: dict = {}
    for referee, ballots in votes.items():
        tally: dict[str, dict] = defaultdict(lambda: {"baseline": 0, "variant": 0, "neither": 0})
        by_session: dict[str, list] = defaultdict(list)
        for entry in key:
            pick = ballots.get(entry["case_id"])
            if pick is None:
                continue
            ids = track_of[entry["case_id"]]
            bucket = ("baseline" if pick == ids.get(entry["baseline_pick"])
                      else "variant" if pick == ids.get(entry["variant_pick"])
                      else "neither")
            tally[entry["variant"]][bucket] += 1
            by_session[entry["variant"]].append(
                (blind[entry["case_id"]]["session_id"], bucket)
            )

        results[referee] = {}
        for variant, counts in tally.items():
            decided = counts["baseline"] + counts["variant"]
            results[referee][variant] = {
                **counts,
                "baseline_win_rate": counts["baseline"] / decided if decided else float("nan"),
                "n_sessions": len({sid for sid, _ in by_session[variant]}),
            }

    # Referee reliability: pairwise κ over the cases every pair actually answered.
    names = sorted(votes)
    case_ids = [c["case_id"] for c in blind.values()]
    results["_referee_agreement"] = {
        f"{a}|{b}": cohens_kappa([votes[a].get(cid) for cid in case_ids],
                                 [votes[b].get(cid) for cid in case_ids])
        for i, a in enumerate(names) for b in names[i + 1:]
    }

    human_path = run_dir / "human_labels.json"
    if human_path.exists():
        human = json.loads(human_path.read_text())
        results["_human_kappa"] = {
            name: cohens_kappa([votes[name].get(cid) for cid in case_ids],
                               [human.get(cid) for cid in case_ids])
            for name in names
        }
    else:
        results["_human_kappa"] = "MISSING — run --human-template and label 100–150 cases"
    return results


def write_report(run_dir: Path, results: dict) -> None:
    lines = ["# Blinded Referee — Kinetic Score Disagreements", "",
             "Win rate is over frames where the baseline (K0) and the variant picked "
             "different top objects; `neither` means the referee chose a third object, "
             "i.e. both formulas were wrong there.", ""]
    for referee, table in results.items():
        if referee.startswith("_"):
            continue
        lines += [f"## Referee: `{referee}`", "",
                  "| Variant | K0 wins | variant wins | neither | K0 win rate | sessions |",
                  "|---|---:|---:|---:|---:|---:|"]
        for variant, r in table.items():
            lines.append(f"| `{variant}` | {r['baseline']} | {r['variant']} | {r['neither']} | "
                         f"{r['baseline_win_rate']:.2f} | {r['n_sessions']} |")
        lines.append("")
    lines += ["## Referee agreement (Cohen's κ)", ""]
    lines += [f"- `{pair}`: {k:.2f}" for pair, k in results["_referee_agreement"].items()]
    lines += ["", "## Agreement with human labels (Cohen's κ)", ""]
    human_kappa = results["_human_kappa"]
    if isinstance(human_kappa, dict):
        lines += [f"- `{name}`: {k:.2f}" for name, k in human_kappa.items()]
    else:
        lines.append(human_kappa)
    lines.append("")
    (run_dir / "referee_report.md").write_text("\n".join(lines), encoding="utf-8")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Blinded multi-VLM referee over disagreement frames.")
    p.add_argument("--run-dir", type=Path, required=True, help="Output dir of kinetic_ablation.py.")
    p.add_argument("--frames-dir", type=Path, help="Directory holding <session>.frames.json.")
    p.add_argument("--referee", action="append", default=None,
                   help=f"Referee to run; repeatable. Known: {', '.join(sorted(LOCAL_REFEREES))}.")
    p.add_argument("--model", action="append", default=[], metavar="NAME=ID",
                   help="Override a referee's weights, e.g. --model gemma3=google/gemma-3-27b-it.")
    p.add_argument("--endpoint", action="append", default=[], metavar="NAME=URL",
                   help="Override a referee's server, e.g. --endpoint qwen2.5-vl=http://gpu02:8001/v1.")
    p.add_argument("--limit", type=int, default=None, help="Only judge the first N cases (smoke test).")
    p.add_argument("--human-template", action="store_true",
                   help="Write a blank human_labels.json to fill in, then exit.")
    p.add_argument("--score-only", action="store_true", help="Aggregate existing votes, no API calls.")
    args = p.parse_args()

    blind = json.loads((args.run_dir / "disagreements.json").read_text())
    if args.limit:
        blind = blind[: args.limit]

    if args.human_template:
        template = {c["case_id"]: "" for c in blind}
        out = args.run_dir / "human_labels_template.json"
        out.write_text(json.dumps(template, indent=2) + "\n")
        print(f"{len(template)} blank cases -> {out}\nFill in object_ids, save as human_labels.json.")
        return

    endpoints = {k: v[0] for k, v in LOCAL_REFEREES.items()}
    models = {k: v[1] for k, v in LOCAL_REFEREES.items()}
    for target, overrides in ((endpoints, args.endpoint), (models, args.model)):
        for override in overrides:
            name, _, value = override.partition("=")
            target[name] = value

    for referee in args.referee or sorted(LOCAL_REFEREES):
        if referee not in endpoints or referee not in models:
            raise SystemExit(f"Referee {referee!r} needs both --endpoint and --model")

    votes: dict[str, dict[str, str]] = {}
    if not args.score_only:
        if not args.frames_dir:
            raise SystemExit("--frames-dir is required unless --score-only")
        for referee in args.referee or sorted(LOCAL_REFEREES):
            votes_path = args.run_dir / f"votes_{referee}.json"
            done = json.loads(votes_path.read_text()) if votes_path.exists() else {}
            for i, case in enumerate(blind, 1):
                if case["case_id"] in done:
                    continue
                objects = case["objects"]
                png = annotated_frame_png(
                    frame_url(args.frames_dir, case["session_id"], case["frame_idx"]),
                    objects,
                    args.run_dir / "frames" / f"{case['session_id']}_{case['frame_idx']}.jpg",
                )
                prompt = PROMPT.format(objects=object_table(objects))
                try:
                    reply = ask_local(endpoints[referee], models[referee], prompt, png)
                    done[case["case_id"]] = parse_vote(reply, {o["object_id"] for o in objects})
                except Exception as exc:      # one bad case must not lose the whole ballot
                    print(f"  {referee} {case['case_id']}: {type(exc).__name__}: {exc}", flush=True)
                    done[case["case_id"]] = None
                # Written every case: API runs are slow and interruptible.
                votes_path.write_text(json.dumps(done, indent=2) + "\n")
                if i % 20 == 0:
                    print(f"  {referee}: {i}/{len(blind)}", flush=True)
            votes[referee] = done
            print(f"{referee}: {sum(v is not None for v in done.values())}/{len(done)} valid votes")

    for votes_path in args.run_dir.glob("votes_*.json"):
        votes.setdefault(votes_path.stem.removeprefix("votes_"), json.loads(votes_path.read_text()))
    if not votes:
        raise SystemExit("No votes found — run without --score-only first.")

    results = score(args.run_dir, votes)
    (args.run_dir / "referee_results.json").write_text(json.dumps(results, indent=2) + "\n")
    write_report(args.run_dir, results)
    print(f"\nReport -> {args.run_dir / 'referee_report.md'}")


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Runnable self-check: python evaluation/vlm_referee.py --self-check"""
    import tempfile

    # ask_local against a throwaway server: the payload shape is the only thing
    # that can silently break when a serving stack is swapped out.
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            seen["path"] = self.path
            payload = json.dumps({"choices": [{"message": {"content": '{"object_id": "O1"}'}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):    # keep the self-check output clean
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    reply = ask_local(f"http://127.0.0.1:{server.server_port}/v1",
                      "test-model", "pick one", b"fake-png")
    server.server_close()
    assert reply == '{"object_id": "O1"}'
    assert seen["path"] == "/v1/chat/completions", seen["path"]
    assert seen["model"] == "test-model" and seen["temperature"] == 0.0
    parts = seen["messages"][0]["content"]
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[1]["text"] == "pick one"

    ids = {"O0", "O1"}
    assert parse_vote('{"object_id": "O1", "reason": "closest"}', ids) == "O1"
    assert parse_vote('```json\n{"object_id": "O0"}\n```', ids) == "O0"
    assert parse_vote("The answer is O0.", ids) is None      # unparseable, not a guess
    assert parse_vote('{"object_id": "O9"}', ids) is None    # hallucinated id rejected
    assert parse_vote("", ids) is None

    assert cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    assert cohens_kappa(["a", "a", "b"], ["b", "b", "a"]) < 0
    assert cohens_kappa([None, None], ["a", "b"]) != cohens_kappa([None, None], ["a", "b"])  # nan: nothing to compare

    # Scoring: referee agrees with the baseline on one case, the variant on the
    # next, and picks a third object on the third.
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        objs = lambda: [{"object_id": f"O{i}", "track_id": f"t{i}", "class": "person",
                         "distance_m": 3.0, "closing_speed_ms": 1.0, "bearing_deg": 0.0,
                         "bbox": [0, 0, 10, 10]} for i in range(3)]
        (run / "disagreements.json").write_text(json.dumps([
            {"case_id": f"s:{i}:{i}", "session_id": "s", "frame_idx": i, "objects": objs()}
            for i in range(3)
        ]))
        (run / "disagreements_key.json").write_text(json.dumps([
            {"case_id": f"s:{i}:{i}", "variant": "linear", "baseline_pick": "t0",
             "variant_pick": "t1", "n_objects": 3} for i in range(3)
        ]))
        results = score(run, {"claude": {"s:0:0": "O0", "s:1:1": "O1", "s:2:2": "O2"}})
        row = results["claude"]["linear"]
        assert (row["baseline"], row["variant"], row["neither"]) == (1, 1, 1), row
        assert row["baseline_win_rate"] == 0.5
        assert isinstance(results["_human_kappa"], str)   # no human labels yet → flagged

    print("vlm_referee.py self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _demo()
    else:
        main()
