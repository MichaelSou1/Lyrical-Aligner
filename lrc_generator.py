"""
lrc_generator.py
────────────────
Converts a list of ``Segment`` objects into a standard LRC file with
millisecond-accurate timestamps.

LRC format reference
────────────────────
Standard line-level:
    [mm:ss.xx]lyric line text

Extended word-level (A2 / Enhanced LRC):
    [mm:ss.xx]<mm:ss.xx>word1 <mm:ss.xx>word2 ...

Both modes are supported.  The ``offset`` tag written in the header
follows the LRC standard where a **positive** value delays the lyrics.

Usage (CLI)
───────────
    python lrc_generator.py segments.json --out song.lrc
    python lrc_generator.py segments.json --out song.lrc --by-word
    python lrc_generator.py segments.json --out song.lrc --title "My Song" --artist "Artist"
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from loguru import logger

import config
from transcription_engine import Segment, WordToken


# ── Timestamp helper ──────────────────────────────────────────────

def _fmt_ts(seconds: float, offset_ms: int = 0) -> str:
    """
    Convert *seconds* (float) to an LRC timestamp string ``[mm:ss.xx]``.

    Args:
        seconds:   Time position in seconds.
        offset_ms: Global offset added to every timestamp (milliseconds).
                   Positive values push lyrics later.

    Returns:
        Formatted string, e.g. ``"[01:23.45]"``.
    """
    total_ms = max(0, int(round(seconds * 1000)) + offset_ms)
    minutes  = total_ms // 60_000
    rem_ms   = total_ms % 60_000
    secs     = rem_ms // 1000
    centis   = (rem_ms % 1000) // 10    # centiseconds (standard LRC precision)
    return f"[{minutes:02d}:{secs:02d}.{centis:02d}]"


def _fmt_inline_ts(seconds: float, offset_ms: int = 0) -> str:
    """
    Like ``_fmt_ts`` but without outer brackets — used inside word tags.

    Returns: e.g. ``"01:23.45"``
    """
    return _fmt_ts(seconds, offset_ms)[1:-1]   # strip leading '[' and trailing ']'


# ── Generator ─────────────────────────────────────────────────────

class LrcGenerator:
    """
    Builds an LRC file from a sequence of :class:`~transcription_engine.Segment`
    objects produced by :class:`~transcription_engine.TranscriptionEngine`
    (after optional post-processing).

    Args:
        title:     Song title written to the ``[ti:]`` tag.
        artist:    Artist name written to the ``[ar:]`` tag.
        album:     Album name written to the ``[al:]`` tag.
        offset_ms: Global time offset in milliseconds (written to ``[offset:]``).
    """

    def __init__(
        self,
        title:     str = config.LRC_TITLE,
        artist:    str = config.LRC_ARTIST,
        album:     str = config.LRC_ALBUM,
        offset_ms: int = config.LRC_OFFSET,
    ) -> None:
        self.title     = title
        self.artist    = artist
        self.album     = album
        self.offset_ms = offset_ms

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def generate(
        self,
        segments:    List[Segment],
        output_path: str | Path,
        by_word:     bool = False,
    ) -> Path:
        """
        Write an ``.lrc`` file to *output_path*.

        Args:
            segments:    Ordered list of :class:`Segment` objects.
            output_path: Destination file path (created if necessary).
            by_word:     If ``True``, emit word-level inline timestamps using
                         the Enhanced LRC (A2) extension.

        Returns:
            Resolved :class:`~pathlib.Path` of the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        content = self.to_string(segments, by_word=by_word)
        output_path.write_text(content, encoding="utf-8")

        line_count = content.count("\n")
        logger.success(
            f"[LrcGenerator] LRC written → {output_path}  "
            f"({line_count} lines, by_word={by_word})"
        )
        return output_path

    def to_string(self, segments: List[Segment], by_word: bool = False) -> str:
        """
        Render the LRC content as a string without writing to disk.

        Useful for in-memory processing or preview.
        """
        lines: List[str] = self._build_header()

        if by_word:
            lines.extend(self._build_word_lines(segments))
        else:
            lines.extend(self._build_segment_lines(segments))

        return "\n".join(lines) + "\n"

    # ──────────────────────────────────────────────────────────────
    # Line builders
    # ──────────────────────────────────────────────────────────────

    def _build_header(self) -> List[str]:
        """Return the LRC metadata header lines."""
        return [
            f"[ti:{self.title}]",
            f"[ar:{self.artist}]",
            f"[al:{self.album}]",
            f"[offset:{self.offset_ms}]",
            "[by:Lyrical-Aligner]",
            "",   # blank line separator between header and lyrics
        ]

    def _build_segment_lines(self, segments: List[Segment]) -> List[str]:
        """
        One ``[mm:ss.xx]text`` line per segment (standard LRC).

        Empty / whitespace-only segments are silently skipped.
        """
        lines: List[str] = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            ts = _fmt_ts(seg.start, self.offset_ms)
            lines.append(f"{ts}{text}")
        return lines

    def _build_word_lines(self, segments: List[Segment]) -> List[str]:
        """
        Word-level timestamps using the Enhanced LRC (A2) extension.

        Format per line::

            [mm:ss.xx]<mm:ss.xx>word1 <mm:ss.xx>word2 <mm:ss.xx>word3

        Falls back to segment-level if a segment has no word tokens.
        """
        lines: List[str] = []
        for seg in segments:
            if not seg.text.strip():
                continue

            if not seg.words:
                # Graceful fallback
                ts = _fmt_ts(seg.start, self.offset_ms)
                lines.append(f"{ts}{seg.text.strip()}")
                continue

            line_ts    = _fmt_ts(seg.start, self.offset_ms)
            word_parts = []
            for w in seg.words:
                inline = _fmt_inline_ts(w.start, self.offset_ms)
                word_parts.append(f"<{inline}>{w.word}")

            lines.append(f"{line_ts}{''.join(word_parts)}")

        return lines


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import dataclasses
    import json

    parser = argparse.ArgumentParser(
        description="Generate LRC from a transcription JSON file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("json_file",  help="Path to segments JSON")
    parser.add_argument("--out",      required=True, help="Output .lrc file path")
    parser.add_argument("--title",    default="")
    parser.add_argument("--artist",   default="")
    parser.add_argument("--album",    default="")
    parser.add_argument("--offset",   default=0, type=int,
                        help="Global offset in milliseconds")
    parser.add_argument("--by-word",  action="store_true",
                        help="Use word-level Enhanced LRC format")
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

    gen = LrcGenerator(
        title=args.title,
        artist=args.artist,
        album=args.album,
        offset_ms=args.offset,
    )
    out = gen.generate(segs, args.out, by_word=args.by_word)
    print(f"LRC saved → {out}")
