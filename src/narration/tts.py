"""
tts.py
======
Text-to-speech layer per docs/architecture.md §11.5's two-tier strategy:
critical path = cached clips / beep-haptic (must not block), non-critical
narration = a real local model, benchmarked before it's allowed near the
real-time path.

Status (2026-08-27): `CachedClipTTS` is real and dependency-free. `PiperTTS` is
real and has been smoke-tested end-to-end in this environment — see
docs/pending_work.md / CHANGELOG for the measured numbers
(tools/benchmark_narration_latency.py reproduces them). FastSpeech2 and the
AI4Bharat Indic voices from architecture.md §11.5 are documented but not
implemented here — they are heavier and explicitly gated on latency testing
first; add them the same way PiperTTS is added, once that testing happens.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class TextToSpeech(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Returns WAV-encoded audio bytes."""
        raise NotImplementedError


class CachedClipTTS(TextToSpeech):
    """
    Critical-path implementation: looks up pre-recorded audio for a known
    phrase. Never calls a model, never blocks. A miss returns a generic
    fallback clip (e.g. an alert beep) rather than raising or returning
    silence — the critical path must always produce *some* audio.
    """

    def __init__(self, clips: dict[str, bytes] | None = None, fallback: bytes = b""):
        self._clips = clips or {}
        self._fallback = fallback

    def add(self, text: str, wav_bytes: bytes) -> None:
        self._clips[text] = wav_bytes

    def synthesize(self, text: str) -> bytes:
        return self._clips.get(text, self._fallback)


class PiperTTS(TextToSpeech):
    """
    Local ONNX neural TTS (rhasspy/piper, VITS architecture — see
    docs/related_work.md §9). Real, tested backend for non-critical narration.

    Voice files: download with
        python -m piper.download_voices --download-dir models/tts/piper <voice-name>
    e.g. `en_US-lessac-low` — see models/tts/piper/ for what's already fetched.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._voice = None

    def _load(self):
        if self._voice is not None:
            return self._voice
        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise ImportError(
                "PiperTTS needs the piper-tts package: pip install piper-tts"
            ) from exc
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Piper voice not found at {self.model_path}. Download with: "
                f"python -m piper.download_voices --download-dir "
                f"{self.model_path.parent} {self.model_path.stem}"
            )
        self._voice = PiperVoice.load(str(self.model_path))
        return self._voice

    def synthesize(self, text: str) -> bytes:
        import io
        import wave

        voice = self._load()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            voice.synthesize_wav(text, wf)
        return buf.getvalue()


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Runnable self-check: python src/narration/tts.py --self-check
    Exercises CachedClipTTS only (dependency-free). PiperTTS is exercised by
    tools/benchmark_narration_latency.py, which needs a downloaded voice."""
    cache = CachedClipTTS(fallback=b"BEEP")
    cache.add("Stop! Car very close ahead.", b"FAKE_WAV_BYTES")
    assert cache.synthesize("Stop! Car very close ahead.") == b"FAKE_WAV_BYTES"
    assert cache.synthesize("Never cached this one.") == b"BEEP"   # never empty

    print("tts.py self-check OK")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _demo()
    else:
        print(__doc__)
