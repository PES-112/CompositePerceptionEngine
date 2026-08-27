"""
benchmark_narration_latency.py
===============================
Standalone latency benchmark for the narration/translation/TTS layer
(src/narration/) — needs no SANPO data, no GPU, just a downloaded Piper voice.
Confirms the orchestration-lane rule (architecture.md §11.6) holds in practice:
the critical lane must stay near-instant and dependency-free; the narration
lane may be slower since it never gates a reflex alert.

Note: no formal TTS latency budget is documented anywhere in docs/ today —
hardware_targets.md defines the 50ms reflex and 500ms cognitive budgets but
nothing for narration TTS specifically. This script reports raw numbers and
flags that gap rather than inventing a budget to grade against.

Usage
-----
    python tools/benchmark_narration_latency.py \\
        --piper-voice models/tts/piper/en_US-lessac-low.onnx \\
        --out-dir evaluation/benchmarks/narration_latency
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.narration.templates import narrate  # noqa: E402
from src.narration.tts import CachedClipTTS, PiperTTS  # noqa: E402
from src.physics_verification.physics_verification import NarratorEvent  # noqa: E402

SAMPLE_EVENTS = {
    "override": NarratorEvent(
        track_id="t1", object_class="car", distance_m=1.0, closing_velocity_ms=5.0,
        bearing_deg=2.0, reason="OVERRIDE", is_override=True,
    ),
    "fast_closing": NarratorEvent(
        track_id="t2", object_class="motorcycle", distance_m=6.0, closing_velocity_ms=8.0,
        bearing_deg=-30.0, reason="fast closing", is_override=False,
    ),
    "near_static": NarratorEvent(
        track_id="t3", object_class="person", distance_m=2.0, closing_velocity_ms=0.3,
        bearing_deg=10.0, reason="near static", is_override=False,
    ),
    "context": NarratorEvent(
        track_id="t4", object_class="bus", distance_m=25.0, closing_velocity_ms=1.0,
        bearing_deg=50.0, reason="context", is_override=False,
    ),
}


def _timed_ms(fn) -> tuple[float, object]:
    t0 = time.perf_counter()
    result = fn()
    return (time.perf_counter() - t0) * 1000.0, result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--piper-voice", type=Path,
                   default=PROJECT_ROOT / "models/tts/piper/en_US-lessac-low.onnx")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "evaluation/benchmarks/narration_latency")
    args = p.parse_args()

    rows = []

    # 1. Template generation — should be microseconds; this is the number the
    # critical lane actually depends on.
    for name, event in SAMPLE_EVENTS.items():
        ms, text = _timed_ms(lambda e=event: narrate(e))
        rows.append({"stage": "template", "case": name, "ms": ms, "output": text})

    # 2. Critical lane audio — CachedClipTTS only, must stay near-instant and
    # never touch a model, by construction (src/narration/pipeline.py).
    critical_tts = CachedClipTTS(fallback=b"BEEP")
    ms, _ = _timed_ms(lambda: critical_tts.synthesize("Stop! Car very close ahead."))
    rows.append({"stage": "critical_tts_cache_miss", "case": "override", "ms": ms,
                 "output": "(fallback beep — no clip cached)"})

    # 3. Narration lane audio — real Piper synthesis, if a voice is available.
    piper_available = args.piper_voice.exists()
    if piper_available:
        tts = PiperTTS(args.piper_voice)
        ms, _ = _timed_ms(lambda: tts._load())  # one-time model load, not per-utterance
        rows.append({"stage": "piper_load", "case": "n/a", "ms": ms, "output": None})
        for name, event in SAMPLE_EVENTS.items():
            text = narrate(event)
            ms, audio = _timed_ms(lambda t=text: tts.synthesize(t))
            rows.append({"stage": "piper_synthesize", "case": name, "ms": ms,
                         "output": f"{len(audio)} bytes"})
    else:
        print(f"Piper voice not found at {args.piper_voice} — skipping synthesis rows. "
              f"Download with: python -m piper.download_voices "
              f"--download-dir {args.piper_voice.parent} {args.piper_voice.stem}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "latency.json").write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        "# Narration Layer Latency Benchmark",
        "",
        "No SANPO data or GPU required. Measures src/narration/ in isolation.",
        "",
        "**Documentation gap found while writing this benchmark:** no formal TTS latency "
        "budget is defined anywhere in docs/ — hardware_targets.md has the 50ms reflex and "
        "500ms cognitive budgets, nothing for narration TTS. Recommend adding one once real "
        "numbers exist (see raw rows below for a starting point).",
        "",
        "| Stage | Case | Latency (ms) | Output |",
        "|---|---|---:|---|",
    ]
    for r in rows:
        lines.append(f"| {r['stage']} | {r['case']} | {r['ms']:.2f} | {r['output']} |")
    lines += [
        "",
        "## Reading this",
        "",
        "- `template` rows: the critical lane's actual dependency — must stay near-zero, and "
        "does (pure Python, no I/O).",
        "- `critical_tts_cache_miss` row: confirms a missing cached clip still returns "
        "instantly (a fallback beep), never blocks waiting on a model.",
        "- `piper_load` row: one-time cost at process startup, not per-alert — acceptable even "
        "if large, as long as it happens before the first real event.",
        "- `piper_synthesize` rows: real per-utterance cost for the narration (non-critical) "
        "lane only. Compare against whatever budget gets documented for this lane.",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Report -> {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
