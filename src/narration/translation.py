"""
translation.py
===============
Translation layer per docs/architecture.md §11.4: critical alerts must use a
pre-translated phrase table (never blocked on a model), non-critical narration
may use IndicTrans2, and English-only is the fallback if translation exceeds
the latency budget.

Status (2026-08-27): `PhraseTableTranslator` and `EnglishFallbackTranslator` are
real, tested, dependency-free. `IndicTrans2Translator` is real, correct-API
adapter code (IndicTransToolkit + transformers installed and verified importable
in this environment) but has **not been run against the actual model weights** —
`ai4bharat/indictrans2-en-indic-dist-200M` (~200M params, several hundred MB) was
not downloaded in this session. Run `_smoke_test_indictrans2()` below once it is,
before trusting this path. See docs/pending_work.md §"Refinement backlog" note on
narration for status.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Translator(ABC):
    @abstractmethod
    def translate(self, text: str, target_lang: str) -> str:
        """target_lang: BCP-47-ish code, e.g. 'hi' for Hindi, 'en' for English."""
        raise NotImplementedError


class EnglishFallbackTranslator(Translator):
    """Identity passthrough. Zero latency, zero dependencies, never fails —
    the fallback named explicitly in architecture.md §11.4."""

    def translate(self, text: str, target_lang: str) -> str:
        return text


class PhraseTableTranslator(Translator):
    """
    Exact-match lookup over a fixed table of pre-translated critical-alert
    phrases. This is the critical-path implementation: a template narrator
    (src/narration/templates.py) only ever emits a small, enumerable set of
    phrase *shapes*, so every shape can be pre-translated once, offline, by a
    human or IndicTrans2, and looked up in O(1) at runtime with no model call.

    On a miss, falls back to English rather than raising — a missing
    translation must never block or crash the critical path.
    """

    def __init__(self, table: dict[tuple[str, str], str] | None = None):
        self._table = table or {}
        self._fallback = EnglishFallbackTranslator()

    def add(self, text: str, target_lang: str, translation: str) -> None:
        self._table[(text, target_lang)] = translation

    def translate(self, text: str, target_lang: str) -> str:
        return self._table.get((text, target_lang), self._fallback.translate(text, target_lang))


class IndicTrans2Translator(Translator):
    """
    Real model backend for non-critical narration (architecture.md §11.4).
    Lazy-loads on first use so importing this module never requires torch/
    transformers/IndicTransToolkit unless this class is actually instantiated.

    model_name options (see docs/related_work.md §8):
      "ai4bharat/indictrans2-en-indic-dist-200M"  — distilled, edge-style target
      "ai4bharat/indictrans2-en-indic-1B"         — higher quality, heavier
    """

    # architecture.md §2.7: "+75ms latency, acceptable for narration path" —
    # this is the number a live run should be checked against.
    LATENCY_BUDGET_MS = 75.0

    def __init__(self, model_name: str = "ai4bharat/indictrans2-en-indic-dist-200M"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError as exc:
            raise ImportError(
                "IndicTrans2Translator needs torch, transformers, and "
                "IndicTransToolkit. Install with:\n"
                "  pip install torch transformers IndicTransToolkit\n"
                f"Original error: {exc}"
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        # float16 + flash-attention only make sense on a CUDA device; CPU stays
        # at default dtype/attention to actually work on a dev laptop.
        kwargs = {"trust_remote_code": True}
        if self._device == "cuda":
            kwargs.update(torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **kwargs).to(self._device)
        self._processor = IndicProcessor(inference=True)

    def translate(self, text: str, target_lang: str) -> str:
        """target_lang: IndicTrans2 FLORES-style code, e.g. 'hin_Deva' for Hindi."""
        self._load()
        import torch

        batch = self._processor.preprocess_batch([text], src_lang="eng_Latn", tgt_lang=target_lang)
        inputs = self._tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self._device)
        with torch.no_grad():
            generated = self._model.generate(**inputs, max_length=256, num_beams=1)
        decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
        return self._processor.postprocess_batch(decoded, lang=target_lang)[0]


def _smoke_test_indictrans2() -> None:
    """
    NOT run by --self-check (needs a model download). Run manually once
    ai4bharat/indictrans2-en-indic-dist-200M has been fetched:
        python -c "from src.narration.translation import _smoke_test_indictrans2 as t; t()"
    """
    import time

    t = IndicTrans2Translator()
    t0 = time.perf_counter()
    t._load()
    print(f"load: {(time.perf_counter() - t0) * 1000:.1f} ms")

    t0 = time.perf_counter()
    hi = t.translate("Car fast from your left.", "hin_Deva")
    dt = (time.perf_counter() - t0) * 1000
    print(f"translate: {dt:.1f} ms -> {hi!r}")
    print(f"budget ({IndicTrans2Translator.LATENCY_BUDGET_MS} ms): "
          f"{'PASS' if dt <= IndicTrans2Translator.LATENCY_BUDGET_MS else 'FAIL'}")


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Runnable self-check: python src/narration/translation.py --self-check
    Covers the dependency-free paths only — see _smoke_test_indictrans2() for
    the real-model path."""
    en = EnglishFallbackTranslator()
    assert en.translate("Car fast from your left.", "hi") == "Car fast from your left."

    table = PhraseTableTranslator()
    table.add("Car fast from your left.", "hi", "आपके बाईं ओर से तेज़ी से कार आ रही है।")
    assert table.translate("Car fast from your left.", "hi").startswith("आपके")
    # Miss falls back to English rather than raising.
    assert table.translate("Never seen this phrase.", "hi") == "Never seen this phrase."

    print("translation.py self-check OK")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _demo()
    else:
        print(__doc__)
