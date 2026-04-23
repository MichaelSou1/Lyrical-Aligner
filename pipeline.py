from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Optional

from loguru import logger

import config
from vocal_extractor      import VocalExtractor
from transcription_engine import TranscriptionEngine
from lrc_generator        import LrcGenerator
from postprocessor        import PostProcessor
from translator           import Translator


class LyricalAlignerPipeline:

    def __init__(
        self,
        language:            Optional[str] = config.WHISPER_LANGUAGE,
        whisper_model:       str           = config.WHISPER_MODEL_SIZE,
        demucs_model:        str           = config.DEMUCS_MODEL,
        device:              str           = config.WHISPER_DEVICE,
        skip_separation:     bool          = config.SKIP_SEPARATION,
        target_lang:         Optional[str] = config.TRANSLATION_TARGET_LANG,
        translation_backend: str           = config.TRANSLATION_BACKEND,
    ) -> None:
        self.skip_separation = skip_separation
        self.target_lang     = target_lang.lower().strip() if target_lang else None

        logger.info("Initialising pipeline components …")

        self.extractor    = VocalExtractor(model=demucs_model, device=device)
        self.engine       = TranscriptionEngine(
            model_size=whisper_model,
            device=device,
            language=language,
        )
        self.postprocessor = PostProcessor()
        self.translator    = (
            Translator(target_lang=target_lang, backend=translation_backend)
            if target_lang else None
        )
        self.generator     = LrcGenerator()

    def run(
        self,
        audio_path:         str | Path,
        output_dir:         Optional[Path | str] = None,
        title:              str  = config.LRC_TITLE,
        artist:             str  = config.LRC_ARTIST,
        album:              str  = config.LRC_ALBUM,
        by_word:            bool = config.LRC_BY_WORD,
        save_intermediates: bool = config.SAVE_INTERMEDIATES,
    ) -> Path:
        audio_path = Path(audio_path).resolve()
        output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = audio_path.stem

        logger.info(f"{'═' * 55}")
        logger.info(f"  Lyrical-Aligner  ·  {stem}")
        logger.info(f"{'═' * 55}")

        # step 1: vocal separation
        if self.skip_separation:
            vocal_path = audio_path
            logger.info("[Step 1/5] Vocal separation SKIPPED (SKIP_SEPARATION = True in config.py)")
        else:
            logger.info("[Step 1/5] Vocal separation …")
            vocal_path = self.extractor.extract(audio_path)

        # step 2: transcription
        logger.info("[Step 2/5] Transcription …")
        segments = self.engine.transcribe(vocal_path)

        # step 3: post-processing
        logger.info("[Step 3/5] Post-processing …")
        segments = self.postprocessor.process(segments)

        # step 4: translation
        detected_lang = segments[0].language if segments else ""
        if self.translator:
            if detected_lang and detected_lang.lower() == self.target_lang:
                logger.info(
                    f"[Step 4/5] Translation SKIPPED "
                    f"(detected language '{detected_lang}' already matches target)"
                )
            else:
                logger.info(
                    f"[Step 4/5] Translating '{detected_lang}' → '{self.target_lang}' …"
                )
                segments = self.translator.translate_segments(
                    segments, source_lang_hint=detected_lang
                )
        else:
            logger.info("[Step 4/5] Translation SKIPPED (TRANSLATION_TARGET_LANG = None in config.py)")

        # step 5a: save intermediate JSON
        if save_intermediates:
            json_path = output_dir / f"{stem}_segments.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    [dataclasses.asdict(s) for s in segments],
                    f, ensure_ascii=False, indent=2,
                )
            logger.info(f"  Segments JSON → {json_path}")

        # step 5b: generate LRC
        logger.info("[Step 5/5] LRC generation …")
        self.generator.title  = title  or stem
        self.generator.artist = artist
        self.generator.album  = album
        lrc_path = output_dir / f"{stem}.lrc"
        self.generator.generate(segments, lrc_path, by_word=by_word)

        logger.info(f"{'═' * 55}")
        logger.success(f"  Done!  →  {lrc_path}")
        logger.info(f"{'═' * 55}")
        return lrc_path



def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Lyrical-Aligner: Audio → LRC pipeline\n"
            "All settings are configured in config.py — edit that file first."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "audio",
        nargs="+",
        help="One or more input audio files (mp3 / wav / flac / …)",
    )
    return p


def main() -> None:
    args  = _build_parser().parse_args()
    pipe  = LyricalAlignerPipeline()

    for audio_file in args.audio:
        pipe.run(audio_path=audio_file)


if __name__ == "__main__":
    main()
