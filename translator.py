from __future__ import annotations

import copy
import dataclasses
import json
import os
import time
from typing import Callable, List, Optional

from loguru import logger

import config
from transcription_engine import Segment, WordToken


_TO_GOOGLE: dict[str, str] = {
    "zh":  "zh-CN",
    "zht": "zh-TW",
    "he":  "iw",
    "jv":  "jw",
}

_TO_DEEPL: dict[str, str] = {
    "en": "EN-US", "zh": "ZH",  "ja": "JA",  "ko": "KO",
    "de": "DE",    "fr": "FR",  "es": "ES",  "it": "IT",
    "pt": "PT-BR", "ru": "RU",  "nl": "NL",  "pl": "PL",
    "ar": "AR",    "tr": "TR",  "sv": "SV",  "da": "DA",
    "fi": "FI",    "el": "EL",  "ro": "RO",  "cs": "CS",
    "hu": "HU",    "bg": "BG",  "uk": "UK",  "lt": "LT",
    "lv": "LV",    "et": "ET",  "sk": "SK",  "sl": "SL",
    "id": "ID",    "nb": "NB",
}


def _google_code(lang: str) -> str:
    lang = lang.lower().strip()
    return _TO_GOOGLE.get(lang, lang)


def _deepl_code(lang: str) -> str:
    lang = lang.lower().strip()
    return _TO_DEEPL.get(lang, lang.upper())


# ── Translator ────────────────────────────────────────────────────

class Translator:
    """
    Translates a list of :class:`~transcription_engine.Segment` objects to
    a target language using the configured backend.

    After translation, ``seg.words`` is set to ``[]`` for each segment
    because word-level timestamps are no longer valid for the translated text.

    Args:
        target_lang:    ISO-639-1 target language code (e.g. ``"zh"``, ``"en"``).
        source_lang:    ISO-639-1 source code, or ``"auto"`` to auto-detect.
        backend:        Translation backend: ``"google"`` | ``"deepl"`` | ``"argos"``.
        deepl_api_key:  DeepL API key (only for ``backend="deepl"``).
                        Falls back to the ``DEEPL_API_KEY`` environment variable.
        batch_delay:    Seconds to sleep between successive API calls
                        (prevents rate-limiting for online backends).
    """

    def __init__(
        self,
        target_lang:   str,
        source_lang:   str   = config.TRANSLATION_SOURCE_LANG,
        backend:       str   = config.TRANSLATION_BACKEND,
        deepl_api_key: str   = config.DEEPL_API_KEY,
        batch_delay:   float = config.TRANSLATION_BATCH_DELAY,
    ) -> None:
        self.target_lang   = target_lang.lower().strip()
        self.source_lang   = source_lang.lower().strip()
        self.backend       = backend.lower().strip()
        self.deepl_api_key = deepl_api_key or os.environ.get("DEEPL_API_KEY", "")
        self.batch_delay   = batch_delay

        self._translate_fn: Callable[[str], str] = self._build_fn()

    def translate(self, text: str) -> str:
        if not text.strip():
            return text
        try:
            result = self._translate_fn(text)
            return result if result else text
        except Exception as exc:
            logger.warning(f"[Translator] Translation failed: {exc!r}")
            return text

    def translate_segments(
        self,
        segments: List[Segment],
        source_lang_hint: Optional[str] = None,
    ) -> List[Segment]:
        if not segments:
            return []


        if self.backend == "argos" and self.source_lang == "auto":
            resolved = (
                source_lang_hint
                or (segments[0].language if segments[0].language else "en")
            )
            logger.info(f"[Translator][Argos] Resolved source language: {resolved}")
            self._translate_fn = self._build_argos_fn(src=resolved, tgt=self.target_lang)

        out   = copy.deepcopy(segments)
        total = len(out)

        logger.info(
            f"[Translator] Translating {total} segment(s) → '{self.target_lang}'  "
            f"(backend={self.backend})"
        )

        for i, seg in enumerate(out):
            if seg.text.strip():
                original  = seg.text
                seg.text  = self.translate(seg.text)
                seg.words = []  # word timestamps no longer valid after translation
                logger.debug(f"  [{i+1}/{total}] {original!r} → {seg.text!r}")

            if self.backend in ("google", "deepl") and i < total - 1:
                time.sleep(self.batch_delay)

        logger.success(
            f"[Translator] ✓ {total} segment(s) translated → '{self.target_lang}'"
        )
        return out

    def _build_fn(self) -> Callable[[str], str]:
        if self.backend == "google":
            return self._build_google_fn()
        elif self.backend == "deepl":
            return self._build_deepl_fn()
        elif self.backend == "argos":
            if self.source_lang != "auto":
                return self._build_argos_fn(src=self.source_lang, tgt=self.target_lang)
            return lambda text: text
        else:
            raise ValueError(
                f"Unknown translation backend: {self.backend!r}. "
                "Valid choices: google | deepl | argos"
            )

    def _build_google_fn(self) -> Callable[[str], str]:
        try:
            from deep_translator import GoogleTranslator  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "deep-translator is not installed.\n"
                "Run: pip install deep-translator"
            )
        src = "auto" if self.source_lang == "auto" else _google_code(self.source_lang)
        tgt = _google_code(self.target_lang)
        logger.info(f"[Translator] Backend=Google  {src} → {tgt}")
        return GoogleTranslator(source=src, target=tgt).translate

    def _build_deepl_fn(self) -> Callable[[str], str]:
        if not self.deepl_api_key:
            raise ValueError(
                "DeepL backend requires an API key.\n"
                "Set the DEEPL_API_KEY environment variable or pass "
                "--deepl-key <KEY> on the command line."
            )
        try:
            from deep_translator import DeepLTranslator  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "deep-translator is not installed.\n"
                "Run: pip install deep-translator"
            )
        src = "auto" if self.source_lang == "auto" else _deepl_code(self.source_lang)
        tgt = _deepl_code(self.target_lang)
        logger.info(f"[Translator] Backend=DeepL  {src} → {tgt}")
        return DeepLTranslator(api_key=self.deepl_api_key, source=src, target=tgt).translate

    def _build_argos_fn(self, src: str, tgt: str) -> Callable[[str], str]:
        try:
            import argostranslate.translate  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "argostranslate is not installed.\n"
                "Run: pip install argostranslate"
            )
        logger.info(f"[Translator] Backend=Argos  {src} → {tgt}")
        langs    = argostranslate.translate.get_installed_languages()
        src_lang = next((l for l in langs if l.code == src), None)
        tgt_lang = next((l for l in langs if l.code == tgt), None)
        missing  = [c for c, l in [(src, src_lang), (tgt, tgt_lang)] if l is None]
        if missing:
            raise RuntimeError(
                f"Argostranslate language package(s) not installed: {missing}.\n"
                f"Run: python translator.py --install-argos {src} {tgt}"
            )
        translation = src_lang.get_translation(tgt_lang)  # type: ignore[union-attr]
        if translation is None:
            raise RuntimeError(
                f"No argostranslate translation model found for {src} → {tgt}.\n"
                f"Run: python translator.py --install-argos {src} {tgt}"
            )
        return translation.translate

    @staticmethod
    def install_argos_package(src: str, tgt: str) -> None:
        try:
            import argostranslate.package    # type: ignore[import]
            import argostranslate.translate  # type: ignore[import]
        except ImportError:
            raise ImportError("Run: pip install argostranslate")

        logger.info("[Argos] Updating package index …")
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()

        pkg = next(
            (p for p in available if p.from_code == src and p.to_code == tgt),
            None,
        )
        if pkg is None:
            raise ValueError(
                f"No argostranslate package found for '{src}' → '{tgt}'.\n"
                "Browse supported pairs at https://www.argosopentech.com/"
            )

        path = pkg.download()
        argostranslate.package.install_from_path(path)
        logger.success(f"[Argos] Package installed: {src} → {tgt}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Translate transcription segments to a target language",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("json_file", nargs="?",
                        help="Input segments JSON file (required unless --install-argos)")
    parser.add_argument("--target",   required=False,
                        help="Target language ISO-639-1 code (e.g. zh, en, ja)")
    parser.add_argument("--source",   default="auto",
                        help="Source language code or 'auto'")
    parser.add_argument("--backend",  default=config.TRANSLATION_BACKEND,
                        choices=["google", "deepl", "argos"])
    parser.add_argument("--deepl-key", default="",
                        help="DeepL API key (or set DEEPL_API_KEY env var)")
    parser.add_argument("--out",      default=None,
                        help="Output JSON file path (default: overwrite input)")
    parser.add_argument("--install-argos", nargs=2, metavar=("SRC", "TGT"),
                        help="Download and install an argostranslate language pair")
    args = parser.parse_args()

    if args.install_argos:
        Translator.install_argos_package(*args.install_argos)
    else:
        if not args.json_file or not args.target:
            parser.error("json_file and --target are required unless --install-argos is used.")

        with open(args.json_file, encoding="utf-8") as f:
            raw = json.load(f)

        segs = [
            Segment(
                text=s["text"], start=s["start"], end=s["end"],
                language=s.get("language", ""),
                words=[WordToken(**w) for w in s.get("words", [])],
            )
            for s in raw
        ]

        tr      = Translator(target_lang=args.target, source_lang=args.source,
                             backend=args.backend, deepl_api_key=args.deepl_key)
        results = tr.translate_segments(segs)

        out_path = args.out or args.json_file
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([dataclasses.asdict(s) for s in results],
                      f, ensure_ascii=False, indent=2)
        print(f"Translated segments saved → {out_path}  ({len(results)} segments)")
