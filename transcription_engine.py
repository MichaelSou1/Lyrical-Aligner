from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import List, Optional

from loguru import logger

import config


@dataclasses.dataclass
class WordToken:
    word:        str
    start:       float
    end:         float
    probability: float


@dataclasses.dataclass
class Segment:
    text:     str
    start:    float
    end:      float
    words:    List[WordToken]
    language: str = ""


# ── Engine ────────────────────────────────────────────────────────

class TranscriptionEngine:
    """
    Thin, re-usable wrapper around :class:`faster_whisper.WhisperModel`.

    Exposes a single ``transcribe(audio_path)`` method that returns a
    typed list of :class:`Segment` objects with full word-level timing.

    Args:
        model:        A pre-loaded ``WhisperModel`` instance.  If *None*,
                      a new model is created from the remaining kwargs.
        model_size:   Model name or path (used when *model* is None).
        device:       ``"cuda"`` or ``"cpu"``.
        compute_type: CTranslate2 quantisation level.
        language:     ISO-639 language code, or ``None`` for auto-detect.
        beam_size:    Beam-search width (higher ⟹ slower but more accurate).
        vad_filter:   Enable Silero-VAD silence filtering.
        vad_threshold: VAD speech probability threshold (0.0–1.0).
    """

    def __init__(
        self,
        model=None,
        model_size:   str   = config.WHISPER_MODEL_SIZE,
        device:       str   = config.WHISPER_DEVICE,
        compute_type: str   = config.WHISPER_COMPUTE_TYPE,
        language:     Optional[str]   = config.WHISPER_LANGUAGE,
        beam_size:    int   = config.WHISPER_BEAM_SIZE,
        vad_filter:   bool  = config.WHISPER_VAD_FILTER,
        vad_threshold: float = config.WHISPER_VAD_THRESHOLD,
    ) -> None:
        self.language      = language
        self.beam_size     = beam_size
        self.vad_filter    = vad_filter
        self.vad_threshold = vad_threshold

        if model is not None:
            self._model = model
        else:
            from download_models import load_whisper_model
            self._model = load_whisper_model(
                model_size_or_path=model_size,
                device=device,
                compute_type=compute_type,
            )

    def transcribe(self, audio_path: str | Path) -> List[Segment]:
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"[TranscriptionEngine] ▶ {audio_path.name}")
        logger.info(
            f"  model_size=large-v3  beam={self.beam_size}  "
            f"vad={self.vad_filter}  lang={self.language or 'auto'}"
        )

        raw_segments, info = self._model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=self.vad_filter,
            vad_parameters={"threshold": self.vad_threshold},
            condition_on_previous_text=True,
            suppress_blank=True,
        )

        detected_lang = info.language
        lang_prob     = info.language_probability
        logger.info(
            f"[TranscriptionEngine] Detected language: {detected_lang} "
            f"(confidence {lang_prob:.1%})"
        )

        segments: List[Segment] = []
        for raw in raw_segments:
            words = [
                WordToken(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    probability=w.probability,
                )
                for w in (raw.words or [])
            ]
            seg = Segment(
                text=raw.text.strip(),
                start=raw.start,
                end=raw.end,
                words=words,
                language=detected_lang,
            )
            segments.append(seg)
            logger.debug(f"  [{seg.start:7.3f}s → {seg.end:7.3f}s] {seg.text}")

        logger.success(
            f"[TranscriptionEngine] ✓ {len(segments)} segments  "
            f"lang={detected_lang}"
        )
        return segments

    @staticmethod
    def to_json(segments: List[Segment], path: str | Path) -> Path:
        """Serialise segments to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [dataclasses.asdict(s) for s in segments],
                f, ensure_ascii=False, indent=2,
            )
        logger.info(f"[TranscriptionEngine] Segments saved → {path}")
        return path

    @staticmethod
    def from_json(path: str | Path) -> List[Segment]:
        """Load segments from a JSON file."""
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return [
            Segment(
                text=s["text"],
                start=s["start"],
                end=s["end"],
                language=s.get("language", ""),
                words=[WordToken(**w) for w in s.get("words", [])],
            )
            for s in raw
        ]



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe audio with faster-whisper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio",           help="Input audio file path")
    parser.add_argument("--language",      default=None,
                        help="Force language (e.g. zh, en, ja). Default: auto")
    parser.add_argument("--model",         default=config.WHISPER_MODEL_SIZE)
    parser.add_argument("--device",        default=config.WHISPER_DEVICE,
                        choices=["cuda", "cpu"])
    parser.add_argument("--compute-type",  default=config.WHISPER_COMPUTE_TYPE)
    parser.add_argument("--out-json",      default=None,
                        help="Optional: save segments to this JSON file")
    args = parser.parse_args()

    engine = TranscriptionEngine(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
    )
    segs = engine.transcribe(args.audio)

    for s in segs:
        print(f"[{s.start:7.3f} → {s.end:7.3f}]  {s.text}")

    if args.out_json:
        TranscriptionEngine.to_json(segs, args.out_json)
        print(f"\nSegments saved → {args.out_json}")
