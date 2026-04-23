"""
Forced alignment module — refines Whisper word timestamps using CTC alignment.

Uses torchaudio MMS_FA (Massively Multilingual Speech forced aligner),
which supports 1162 languages and runs locally with no API calls.

The aligner takes Whisper's segment boundaries as anchors and re-estimates
each word's start/end time by aligning the transcript against the audio's
acoustic emissions, reducing timestamp error from ~±0.2 s to ~±0.05 s.

For CJK scripts (Chinese / Japanese / Korean) and any segment where
tokenisation fails, the original Whisper timestamps are kept unchanged.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio
from loguru import logger

import config
from transcription_engine import Segment, WordToken


# ── text normalisation ────────────────────────────────────────────────────────

_CJK_RANGES = (
    (0x4E00, 0x9FFF),   # CJK Unified
    (0x3040, 0x30FF),   # Hiragana / Katakana
    (0xAC00, 0xD7AF),   # Hangul
    (0x3400, 0x4DBF),   # CJK Extension A
)


def _is_cjk(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in _CJK_RANGES):
            return True
    return False


def _normalize(text: str) -> str:
    """
    Strip diacritics and reduce to the ASCII character set accepted by MMS_FA:
    [a-z] plus apostrophe. Used only for tokenisation; original words are kept.
    """
    nfd = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    ascii_only = re.sub(r"[^a-z']", " ", stripped)
    return re.sub(r"\s+", " ", ascii_only).strip()


# ── aligner ───────────────────────────────────────────────────────────────────

class ForcedAligner:
    """
    Wraps torchaudio MMS_FA to refine word-level timestamps after Whisper
    transcription. Loads the acoustic model once and reuses it across calls.

    Args:
        device: ``"cuda"`` or ``"cpu"``.
    """

    _SAMPLE_RATE = 16_000   # MMS_FA requirement

    def __init__(self, device: str = config.WHISPER_DEVICE) -> None:
        self.device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        logger.info("[ForcedAligner] Loading MMS_FA acoustic model …")
        bundle = torchaudio.pipelines.MMS_FA
        self._model     = bundle.get_model().to(self.device)
        self._tokenizer = bundle.get_tokenizer()
        self._aligner   = bundle.get_aligner()
        logger.success(f"[ForcedAligner] Model ready  device={self.device}")

    # ── public API ────────────────────────────────────────────────────────────

    def align(self, audio_path: str | Path, segments: List[Segment]) -> List[Segment]:
        """
        Refine word timestamps for every segment in *segments*.

        Segments whose text is CJK, has no word list, or fails tokenisation
        are returned unchanged (Whisper timestamps preserved).

        Args:
            audio_path: Path to the vocals WAV (output of VocalExtractor).
            segments:   Whisper segments with preliminary word timestamps.

        Returns:
            New list of Segment objects with refined ``start``/``end`` and
            per-word ``start``/``end`` fields.
        """
        audio_path = Path(audio_path)
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self._SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, self._SAMPLE_RATE)
        waveform = waveform.to(self.device)

        refined: List[Segment] = []
        n_ok = n_skip = 0

        for seg in segments:
            result = self._align_segment(waveform, seg)
            if result is not None:
                refined.append(result)
                n_ok += 1
            else:
                refined.append(seg)
                n_skip += 1

        logger.success(
            f"[ForcedAligner] ✓ {n_ok} segments aligned, {n_skip} kept as-is"
        )
        return refined

    # ── internals ─────────────────────────────────────────────────────────────

    def _align_segment(self, full_waveform: torch.Tensor, seg: Segment) -> Optional[Segment]:
        """
        Align one segment. Returns None if the segment should be skipped.
        full_waveform: (1, N) at self._SAMPLE_RATE.
        """
        if not seg.words:
            return None
        if _is_cjk(seg.text):
            return None

        # Extract the audio chunk for this segment
        start_sample = int(seg.start * self._SAMPLE_RATE)
        end_sample   = int(seg.end   * self._SAMPLE_RATE)
        chunk = full_waveform[:, start_sample:end_sample]
        if chunk.shape[1] < 400:   # too short for the model (~25 ms)
            return None

        # Build normalised word list for tokenisation
        norm_words = [_normalize(w.word) for w in seg.words]
        norm_words = [w for w in norm_words if w]  # drop empty after normalisation
        if not norm_words:
            return None

        try:
            with torch.inference_mode():
                emission, _ = self._model(chunk)   # (1, T, C)

            tokens = self._tokenizer(norm_words)
            token_spans = self._aligner(emission[0], tokens)

        except Exception as exc:
            logger.debug(f"[ForcedAligner] Skipped segment '{seg.text[:30]}': {exc}")
            return None

        return self._build_refined_segment(seg, token_spans, emission.shape[1])

    def _build_refined_segment(
        self,
        seg: Segment,
        token_spans,
        num_frames: int,
    ) -> Segment:
        """Convert frame-level spans back to absolute timestamps."""
        audio_duration = seg.end - seg.start
        frames_to_sec  = audio_duration / num_frames

        # Map each original word to a span (some words may have been dropped
        # by normalisation; fall back to original timestamp for those)
        original_words = [w for w in seg.words if _normalize(w.word)]
        span_count     = min(len(original_words), len(token_spans))

        refined_words: List[WordToken] = []
        for i, word in enumerate(seg.words):
            if i < span_count:
                span  = token_spans[i]
                w_start = round(seg.start + span[0].start * frames_to_sec, 3)
                w_end   = round(seg.start + span[-1].end  * frames_to_sec, 3)
                refined_words.append(WordToken(
                    word=word.word,
                    start=w_start,
                    end=w_end,
                    probability=word.probability,
                ))
            else:
                refined_words.append(word)

        seg_start = refined_words[0].start  if refined_words else seg.start
        seg_end   = refined_words[-1].end   if refined_words else seg.end

        return Segment(
            text=seg.text,
            start=seg_start,
            end=seg_end,
            words=refined_words,
            language=seg.language,
        )


# ── standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import dataclasses

    parser = argparse.ArgumentParser(
        description="Refine Whisper word timestamps with CTC forced alignment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio",      help="Vocals WAV file")
    parser.add_argument("segments",   help="Segments JSON (from transcription_engine)")
    parser.add_argument("--device",   default=config.WHISPER_DEVICE, choices=["cuda", "cpu"])
    parser.add_argument("--out-json", default=None, help="Save refined segments JSON here")
    args = parser.parse_args()

    from transcription_engine import TranscriptionEngine
    segs = TranscriptionEngine.from_json(args.segments)

    aligner  = ForcedAligner(device=args.device)
    refined  = aligner.align(args.audio, segs)

    for s in refined:
        print(f"[{s.start:.3f} → {s.end:.3f}]  {s.text}")
        for w in s.words:
            print(f"    {w.start:.3f}–{w.end:.3f}  {w.word.strip()}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump([dataclasses.asdict(s) for s in refined], f, ensure_ascii=False, indent=2)
        print(f"\nRefined segments saved → {args.out_json}")
