"""
run_full_pipeline_demo.py
==========================
Runs a synthetic multi-frame scene through a runtime chain and prints what
would actually reach the user.

Default: the v2 threat engine (docs/architecture.md §5) — Stage-1 rows ->
Kalman fusion -> safety envelope + fused priority -> announcement manager ->
Narration/Translation/TTS — on a 5 Hz scene: a car closing in from the left
while the walker approaches a pothole.

--v1: the v1 chain kept as the replay baseline — Perception row -> Threat
Prioritizer -> (stub) Cognitive Layer -> Physics Verification -> Narration.
The rest of this docstring describes that v1 scenario.

This is the milestone docs/pending_work.md §5 calls "the single most valuable
integration milestone remaining before a demo": proving the runtime contracts
actually compose, end-to-end, in one process. It needs no SANPO data, no GPU,
and no network — every object below is a synthetic PerceivedObject built the
same way tests/test_threat_prioritizer.py and tests/test_reflex_layer.py build
theirs, just strung across frames instead of one frame at a time.

Scenario: a car approaches from the left at a constant 1.5 m/s (slow —
chosen deliberately so the routing genuinely walks through every lane
instead of entering at "reflex" immediately; a car's severity weight alone
puts it at K>=LOW_K_THRESHOLD from a fairly long distance, verified with
src/perception_stack/physics.kinetic_score() directly before picking these
numbers — see the comment above build_scenario()) over 6 frames until it
crosses the TTC override threshold, while a pothole sits at near-static
hazard range the entire time — exercising ignore -> cognitive -> reflex
escalation for one object and a simultaneous reflex+cognitive frame
(v1 architecture.md §2.3 (git a6e37a7): "Both tracks can fire simultaneously for different
objects").

Usage
-----
    python tools/run_full_pipeline_demo.py             # v2 threat engine
    python tools/run_full_pipeline_demo.py --v1        # v1 baseline chain
    python tools/run_full_pipeline_demo.py --lang hi   # exercise the phrase-table path
    python tools/run_full_pipeline_demo.py --piper-voice models/tts/piper/en_US-lessac-low.onnx
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.narration.pipeline import NarrationPipeline  # noqa: E402
from src.narration.translation import PhraseTableTranslator  # noqa: E402
from src.narration.tts import CachedClipTTS, PiperTTS, TextToSpeech  # noqa: E402
from src.pipeline import process_frames  # noqa: E402
from src.pipeline.orchestrator import V2Pipeline  # noqa: E402
from src.threat_prioritizer import build_threat_frame  # noqa: E402

# Pre-translated phrases, to exercise PhraseTableTranslator without needing
# IndicTrans2 model weights — see src/narration/translation.py. Note the
# override phrase below is intentionally never used: the critical lane always
# bypasses the translator by construction (src/narration/pipeline.py), so
# having a table entry for it demonstrates that bypass rather than a miss.
DEMO_PHRASE_TABLE = {
    ("Pothole close, ahead.", "hi"): "गड्ढा पास है, आगे।",
    ("Car left, 10 meters.", "hi"): "गाड़ी बाईं ओर, 10 मीटर।",
    ("Stop! Car very close left.", "hi"): "रुको! गाड़ी बिल्कुल पास बाईं ओर है।",
}


def _car_row(frame_idx: int, distance_m: float, velocity_ms: float) -> dict:
    return {
        "frame_idx": frame_idx, "source": "synthetic_demo", "track_id": "car1", "class": "car",
        "confidence": 0.92, "bbox_x1": 400, "bbox_y1": 300, "bbox_x2": 520, "bbox_y2": 420,
        "cx_px": 460, "bearing_deg": -18.0, "distance_m": distance_m, "velocity_ms": velocity_ms,
    }


def _pothole_row(frame_idx: int) -> dict:
    return {
        "frame_idx": frame_idx, "source": "synthetic_demo", "track_id": "pothole1", "class": "pothole",
        "confidence": 0.88, "bbox_x1": 200, "bbox_y1": 600, "bbox_x2": 260, "bbox_y2": 660,
        "cx_px": 230, "bearing_deg": 4.0, "distance_m": 1.3, "velocity_ms": 0.0,
    }


def build_scenario() -> list[list[dict]]:
    """
    6 frames, car closing at a constant 1.5 m/s. Distances chosen by actually
    running physics.kinetic_score("car", ...) first (not guessed) so the
    frame-by-frame routing genuinely walks ignore -> cognitive -> reflex:

        d=25.0  K=0.42  ttc=16.7s -> ignore     (below LOW_K_THRESHOLD=0.5)
        d=15.0  K=0.69  ttc=10.0s -> cognitive
        d=10.0  K=1.04  ttc= 6.7s -> cognitive
        d= 6.0  K=1.74  ttc= 4.0s -> cognitive
        d= 3.0  K=3.47  ttc= 2.0s -> cognitive
        d= 1.2  K=8.68  ttc= 0.8s -> reflex (TTC gate; K alone would also cross
                                     HIGH_K_THRESHOLD=5.0 here independently)

    Pothole present throughout at cognitive-route range, so the last frame
    exercises a genuine simultaneous reflex+cognitive frame.
    """
    car_distances = [25.0, 15.0, 10.0, 6.0, 3.0, 1.2]
    frames = []
    for i, d in enumerate(car_distances, start=1):
        frames.append([_car_row(i, d, velocity_ms=1.5), _pothole_row(i)])
    return frames


STEP_S = 0.2   # SANPO processing rate: 15 FPS, stride 3


def build_v2_scenario() -> list[list[dict]]:
    """
    25 frames at 5 Hz. A car approaches from about 20° left at 3 m/s, starting
    14 m out; the walker closes on a pothole straight ahead at 1.4 m/s from
    8 m. Rows carry no pose, so motion is camera-relative (no pose source).
    """
    frames = []
    for k in range(25):
        rows = [_car_row(k, 14.0 - 3.0 * STEP_S * k, velocity_ms=0.0)]
        rows[0]["bearing_deg"] = -20.0
        rows.append({**_pothole_row(k), "distance_m": max(0.3, 8.0 - 1.4 * STEP_S * k), "bearing_deg": 2.0})
        frames.append(rows)
    return frames


def run_v2(pipeline: NarrationPipeline, lang: str) -> None:
    v2 = V2Pipeline(pipeline, target_lang=lang)
    print(f"{'t (s)':>5} | {'Kind':<8} | {'Object':<9} | {'Lane':<9} | Narration")
    print("-" * 80)
    spoken = 0
    for k, rows in enumerate(build_v2_scenario()):
        result = v2.step(k, k * STEP_S, rows)
        a = result.frame.announcement
        if a is None:
            continue
        spoken += 1
        print(f"{k * STEP_S:>5.1f} | {a.kind:<8} | {a.track_id:<9} | {result.narration.lane:<9} | "
              f"{result.narration.translated_text}")
    print(f"\n{spoken} announcements over {25 * STEP_S:.0f} s. Every row went through: Kalman fusion -> "
          "safety envelope + fused priority -> announcement manager -> narration, on synthetic data only.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", default="en", help="Target language for the narration lane (e.g. 'hi').")
    p.add_argument("--piper-voice", type=Path,
                   default=PROJECT_ROOT / "models/tts/piper/en_US-lessac-low.onnx")
    p.add_argument("--v1", action="store_true", help="Run the v1 baseline chain instead of the v2 engine.")
    args = p.parse_args()

    translator = PhraseTableTranslator(dict(DEMO_PHRASE_TABLE))

    narration_tts: TextToSpeech
    if args.piper_voice.exists() and importlib.util.find_spec("piper") is not None:
        narration_tts = PiperTTS(args.piper_voice)
        print(f"Using PiperTTS ({args.piper_voice.name}) for the narration lane.\n")
    else:
        narration_tts = CachedClipTTS(fallback=b"(no cached clip, no Piper voice found)")  # placeholder audio
        print(f"No Piper voice or piper-tts package — narration lane will use "
              f"CachedClipTTS fallback only. See docs/architecture.md §6.6.\n")

    pipeline = NarrationPipeline(
        translator=translator,
        tts=narration_tts,
        critical_tts=CachedClipTTS(fallback=b"(critical beep - no clip cached)"),
    )

    if not args.v1:
        run_v2(pipeline, args.lang)
        return

    threat_frames = [build_threat_frame(rows) for rows in build_scenario()]
    results = process_frames(threat_frames, pipeline, target_lang=args.lang)

    print(f"{'Frame':>5} | {'Routes':<22} | {'Semantic pick':<14} | {'Alert':<6} | {'Lane':<9} | Narration")
    print("-" * 100)
    for frame, result in zip(threat_frames, results):
        routes = ", ".join(f"{e.track_id}:{e.route}" for e in frame.events)
        semantic = result.semantic_eval.primary_threat_id if result.semantic_eval else "-"
        alert = "yes" if result.has_alert else "no"
        lane = result.narration.lane if result.narration else "-"
        text = result.narration.translated_text if result.narration else "-"
        print(f"{frame.frame_idx:>5} | {routes:<22} | {semantic:<14} | {alert:<6} | {lane:<9} | {text}")

    n_alerts = sum(1 for r in results if r.has_alert)
    n_override = sum(1 for r in results if r.narrator_event and r.narrator_event.is_override)
    print(f"\n{n_alerts}/{len(results)} frames produced an alert; {n_override} were override (critical) events.")
    print("Every row above went through: Threat Prioritizer -> stub Cognitive Layer "
          "-> Physics Verification -> Narration/Translation/TTS, in one process, "
          "on synthetic data only.")
    if args.lang != "en":
        critical_rows = [r for r in results if r.narration and r.narration.lane == "critical"]
        if critical_rows and critical_rows[0].narration.translated_text == critical_rows[0].narration.text:
            print(f"Note: the override frame stayed in English even with --lang {args.lang} — that's "
                  f"the critical lane's translation bypass (architecture.md §6.6) working as designed, "
                  f"not a missing phrase-table entry (one exists for it and was intentionally not used).")


if __name__ == "__main__":
    main()
