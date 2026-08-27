"""
pipeline.py
===========
Wires template narration + translation + TTS together, enforcing the
orchestration-lane rule from docs/architecture.md §11.6 in code rather than
only in documentation: an override event must never route through translation
or a heavy TTS model — only the deterministic template plus a cached/beep clip.
Everything else may use the fuller translation/TTS stack, best-effort.

This is deliberately the thin layer on top of templates.py / translation.py /
tts.py — each of those stays independently testable, and this module only
enforces sequencing and the critical/non-critical split.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.narration.templates import narrate
from src.narration.translation import EnglishFallbackTranslator, Translator
from src.narration.tts import CachedClipTTS, TextToSpeech
from src.physics_verification.physics_verification import NarratorEvent


@dataclass
class NarrationOutput:
    text: str                 # English template output
    translated_text: str      # == text if target_lang == "en" or on the critical lane
    audio: bytes
    lane: str                 # "critical" | "narration"


class NarrationPipeline:
    """
    critical_tts should be a fast, dependency-light backend (CachedClipTTS by
    default) — it is the one thing an override event is allowed to call.
    tts/translator may be heavier (PiperTTS, IndicTrans2Translator) since they
    only run on the best-effort narration lane.
    """

    def __init__(
        self,
        translator: Translator | None = None,
        tts: TextToSpeech | None = None,
        critical_tts: TextToSpeech | None = None,
    ):
        self.translator = translator or EnglishFallbackTranslator()
        self.tts = tts or CachedClipTTS()
        self.critical_tts = critical_tts or CachedClipTTS()

    def process(self, event: NarratorEvent, target_lang: str = "en") -> NarrationOutput:
        text = narrate(event)

        if event.is_override:
            # Critical lane: English only, cached/beep audio only. No translator
            # or model-backed TTS call — structurally, not just by convention.
            audio = self.critical_tts.synthesize(text)
            return NarrationOutput(text=text, translated_text=text, audio=audio, lane="critical")

        translated = text if target_lang == "en" else self.translator.translate(text, target_lang)
        audio = self.tts.synthesize(translated)
        return NarrationOutput(text=text, translated_text=translated, audio=audio, lane="narration")


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Runnable self-check: python -m src.narration.pipeline --self-check"""
    from src.narration.translation import PhraseTableTranslator

    table = PhraseTableTranslator()
    table.add("Car fast from your left.", "hi", "TRANSLATED")
    tts = CachedClipTTS(fallback=b"BEEP")
    tts.add("TRANSLATED", b"NARRATION_AUDIO")
    critical_tts = CachedClipTTS(fallback=b"CRITICAL_BEEP")

    pipeline = NarrationPipeline(translator=table, tts=tts, critical_tts=critical_tts)

    fast = NarratorEvent(track_id="t1", object_class="car", distance_m=6.0,
                          closing_velocity_ms=8.0, bearing_deg=-30.0,
                          reason="fast closing", is_override=False)
    out = pipeline.process(fast, target_lang="hi")
    assert out.lane == "narration"
    assert out.text == "Car fast from your left."
    assert out.translated_text == "TRANSLATED"
    assert out.audio == b"NARRATION_AUDIO"

    override = NarratorEvent(track_id="t2", object_class="car", distance_m=1.0,
                              closing_velocity_ms=4.0, bearing_deg=0.0,
                              reason="OVERRIDE", is_override=True)
    # Even with target_lang="hi", an override must stay English + critical audio
    # — this is the property the whole module exists to guarantee.
    out = pipeline.process(override, target_lang="hi")
    assert out.lane == "critical"
    assert out.translated_text == out.text          # no translation call happened
    assert out.audio == b"CRITICAL_BEEP"             # not narration tts's cache

    print("pipeline.py self-check OK")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _demo()
    else:
        print(__doc__)
