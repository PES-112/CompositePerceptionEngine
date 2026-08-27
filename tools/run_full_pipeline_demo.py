"""
run_full_pipeline_demo.py
==========================
Runs a synthetic multi-frame scene through every runtime component built so
far — Perception row -> Threat Prioritizer -> (stub) Cognitive Layer ->
Physics Verification -> Narration/Translation/TTS — and prints what would
actually reach the user each frame.

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
(architecture.md §2.3: "Both tracks can fire simultaneously for different
objects").

Usage
-----
    python tools/run_full_pipeline_demo.py
    python tools/run_full_pipeline_demo.py --lang hi   # exercise the phrase-table path
    python tools/run_full_pipeline_demo.py --piper-voice models/tts/piper/en_US-lessac-low.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.narration.pipeline import NarrationPipeline  # noqa: E402
from src.narration.translation import PhraseTableTranslator  # noqa: E402
from src.narration.tts import CachedClipTTS, PiperTTS, TextToSpeech  # noqa: E402
from src.pipeline import process_frames  # noqa: E402
from src.threat_prioritizer import build_threat_frame  # noqa: E402

# Pre-translated phrases, to exercise PhraseTableTranslator without needing
# IndicTrans2 model weights — see src/narration/translation.py. Note the
# override phrase below is intentionally never used: the critical lane always
# bypasses the translator by construction (src/narration/pipeline.py), so
# having a table entry for it demonstrates that bypass rather than a miss.
DEMO_PHRASE_TABLE = {
    ("Pothole close, ahead.", "hi"): "गड्ढा पास है, आगे।",
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", default="en", help="Target language for the narration lane (e.g. 'hi').")
    p.add_argument("--piper-voice", type=Path,
                   default=PROJECT_ROOT / "models/tts/piper/en_US-lessac-low.onnx")
    args = p.parse_args()

    translator = PhraseTableTranslator(dict(DEMO_PHRASE_TABLE))

    narration_tts: TextToSpeech
    if args.piper_voice.exists():
        narration_tts = PiperTTS(args.piper_voice)
        print(f"Using PiperTTS ({args.piper_voice.name}) for the narration lane.\n")
    else:
        narration_tts = CachedClipTTS(fallback=b"(no cached clip, no Piper voice found)")  # placeholder audio
        print(f"No Piper voice at {args.piper_voice} — narration lane will use "
              f"CachedClipTTS fallback only. See docs/architecture.md §11.5.\n")

    pipeline = NarrationPipeline(
        translator=translator,
        tts=narration_tts,
        critical_tts=CachedClipTTS(fallback=b"(critical beep - no clip cached)"),
    )

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
                  f"the critical lane's translation bypass (architecture.md §11.6) working as designed, "
                  f"not a missing phrase-table entry (one exists for it and was intentionally not used).")


if __name__ == "__main__":
    main()
