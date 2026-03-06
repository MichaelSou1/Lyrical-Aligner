"""
postprocessor.py
────────────────
Rule-based post-processing pipeline for ASR transcription errors.

Correction passes (applied in order)
──────────────────────────────────────
  1. Short-segment filter   — discard segments below minimum duration
  2. Hallucination removal  — drop segments matching known Whisper artefact phrases
  3. Text cleanup           — strip leading/trailing junk punctuation; normalise spaces
  4. Repetition removal     — collapse consecutive duplicate words (a common Whisper bug)
  5. Segment merging        — fuse adjacent segments whose gap < threshold
  6. Long-line splitting    — break over-long segments at word boundaries

All passes operate on deep-copies so the original list is never mutated.

Usage (CLI)
───────────
    python postprocessor.py segments.json --out cleaned.json
"""

from __future__ import annotations

import dataclasses
import json
import re
from copy import deepcopy
from typing import List

from loguru import logger

import config
from transcription_engine import Segment, WordToken


# regex patterns
_RE_REPEAT_WORD = re.compile(r'\b(\w{2,})\s+\1\b', re.IGNORECASE | re.UNICODE)
_RE_LEADING_JUNK = re.compile(r'^[\s,.\-\u2013\u2014!?;:]+')
_RE_TRAILING_JUNK = re.compile(r'[\s,\-\u2013\u2014;:]+$')
_RE_MULTI_SPACE = re.compile(r'\s{2,}')

# known Whisper hallucination phrases
_RE_HALLUCINATION = re.compile(
    r'('
    r'thanks\s+for\s+watching'
    r'|please\s+(like|subscribe|comment)'
    r'|subtitles\s+by'
    r'|字幕[提供制作]'
    r'|翻譯'
    r'|amara\.org'
    r'|www\.'
    r')',
    re.IGNORECASE | re.UNICODE,
)


# ── PostProcessor ─────────────────────────────────────────────────

class PostProcessor:
    """
    Applies a configurable chain of rule-based corrections to a list of
    :class:`~transcription_engine.Segment` objects.

    All public methods return a **new** list; the input is never modified.

    Args:
        merge_gap:            Max silence gap (seconds) between segments
                              that will be merged into one line.
        min_duration:         Discard segments shorter than this (seconds).
        max_chars:            Soft character cap per LRC line; segments
                              exceeding this are split at word boundaries.
        fix_repetitions:      Toggle consecutive duplicate-word removal.
        fix_hallucinations:   Toggle hallucination-phrase removal.
    """

    def __init__(
        self,
        merge_gap:          float = config.PP_MERGE_GAP_THRESHOLD,
        min_duration:       float = config.PP_MIN_SEGMENT_DURATION,
        max_chars:          int   = config.PP_MAX_CHARS_PER_LINE,
        fix_repetitions:    bool  = True,
        fix_hallucinations: bool  = True,
    ) -> None:
        self.merge_gap          = merge_gap
        self.min_duration       = min_duration
        self.max_chars          = max_chars
        self.fix_repetitions    = fix_repetitions
        self.fix_hallucinations = fix_hallucinations

    def process(self, segments: List[Segment]) -> List[Segment]:
        segs = deepcopy(segments)
        n_in = len(segs)

        segs = self._filter_short(segs)
        segs = self._remove_hallucinations(segs)
        segs = self._clean_text(segs)
        segs = self._remove_repetitions(segs)
        segs = self._merge_segments(segs)
        segs = self._split_long_lines(segs)

        logger.info(
            f"[PostProcessor] {n_in} → {len(segs)} segments  "
            f"(merge_gap={self.merge_gap}s  max_chars={self.max_chars})"
        )
        return segs

    def _filter_short(self, segs: List[Segment]) -> List[Segment]:
        out = [
            s for s in segs
            if (s.end - s.start) >= self.min_duration and s.text.strip()
        ]
        removed = len(segs) - len(out)
        if removed:
            logger.debug(f"  [filter_short] Removed {removed} short/empty segment(s)")
        return out

    def _remove_hallucinations(self, segs: List[Segment]) -> List[Segment]:
        if not self.fix_hallucinations:
            return segs
        out = [s for s in segs if not _RE_HALLUCINATION.search(s.text)]
        removed = len(segs) - len(out)
        if removed:
            logger.debug(f"  [hallucinations] Removed {removed} hallucinated segment(s)")
        return out

    def _clean_text(self, segs: List[Segment]) -> List[Segment]:
        for seg in segs:
            t = seg.text
            t = _RE_LEADING_JUNK.sub("", t)
            t = _RE_TRAILING_JUNK.sub("", t)
            t = _RE_MULTI_SPACE.sub(" ", t)
            seg.text = t.strip()
        return segs

    def _remove_repetitions(self, segs: List[Segment]) -> List[Segment]:
        if not self.fix_repetitions:
            return segs
        for seg in segs:
            prev = None
            while seg.text != prev:
                prev = seg.text
                seg.text = _RE_REPEAT_WORD.sub(r'\1', seg.text)
        return segs

    def _merge_segments(self, segs: List[Segment]) -> List[Segment]:
        if not segs:
            return segs

        merged: List[Segment] = [deepcopy(segs[0])]

        for curr in segs[1:]:
            prev = merged[-1]
            gap  = curr.start - prev.end
            combined_len = len(prev.text) + len(curr.text) + 1

            if gap <= self.merge_gap and combined_len <= self.max_chars:
                prev.text  = f"{prev.text} {curr.text}".strip()
                prev.end   = curr.end
                prev.words.extend(curr.words)
            else:
                merged.append(deepcopy(curr))

        logger.debug(f"  [merge] {len(segs)} → {len(merged)} segment(s)")
        return merged

    def _split_long_lines(self, segs: List[Segment]) -> List[Segment]:
        result: List[Segment] = []
        for seg in segs:
            if len(seg.text) <= self.max_chars or not seg.words:
                result.append(seg)
            else:
                result.extend(self._split_by_words(seg))
        return result

    def _split_by_words(self, seg: Segment) -> List[Segment]:
        chunks:        List[List[WordToken]] = []
        current_chunk: List[WordToken]       = []
        current_len                          = 0

        for word in seg.words:
            word_len = len(word.word) + 1

            if current_len + word_len > self.max_chars and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [word]
                current_len   = word_len
            else:
                current_chunk.append(word)
                current_len += word_len

        if current_chunk:
            chunks.append(current_chunk)

        result: List[Segment] = []
        for chunk in chunks:
            text = "".join(w.word for w in chunk).strip()
            result.append(Segment(
                text=text,
                start=chunk[0].start,
                end=chunk[-1].end,
                words=chunk,
                language=seg.language,
            ))
        return result



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Post-process transcription segments (rule-based)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("json_file",     help="Input segments JSON file")
    parser.add_argument("--out",         required=True, help="Output JSON file")
    parser.add_argument("--merge-gap",   default=config.PP_MERGE_GAP_THRESHOLD, type=float)
    parser.add_argument("--max-chars",   default=config.PP_MAX_CHARS_PER_LINE,  type=int)
    parser.add_argument("--min-dur",     default=config.PP_MIN_SEGMENT_DURATION, type=float)
    args = parser.parse_args()

    with open(args.json_file, encoding="utf-8") as f:
        raw = json.load(f)

    segs = [
        Segment(
            text=s["text"],
            start=s["start"],
            end=s["end"],
            language=s.get("language", ""),
            words=[WordToken(**w) for w in s.get("words", [])],
        )
        for s in raw
    ]

    pp = PostProcessor(
        merge_gap=args.merge_gap,
        min_duration=args.min_dur,
        max_chars=args.max_chars,
    )
    cleaned = pp.process(segs)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            [dataclasses.asdict(s) for s in cleaned],
            f, ensure_ascii=False, indent=2,
        )
    print(f"Post-processed segments saved → {args.out}  ({len(cleaned)} segments)")
