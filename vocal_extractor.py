"""
vocal_extractor.py
──────────────────
Extracts the vocal stem from a mixed music track using Facebook's Demucs.

Supported Demucs models
───────────────────────
  htdemucs     — 4-stem hybrid Transformer (drums / bass / other / vocals)
  htdemucs_ft  — Fine-tuned htdemucs; higher quality, slower runtime
  htdemucs_6s  — 6-stem variant (adds guitar + piano)
  mdx_extra    — MDX-Net-based model, strong on pop/rock

Demucs is invoked via ``subprocess`` so that its large memory footprint is
released before the Whisper model is loaded into GPU VRAM.

Output
──────
A 44 100 Hz stereo WAV file containing *only* the vocal track, written to
``<out_dir>/<stem>_vocals.wav``.

Usage (CLI)
───────────
    python vocal_extractor.py song.mp3
    python vocal_extractor.py song.mp3 --model htdemucs_ft --device cpu
    python vocal_extractor.py song.mp3 --out-dir ./my_output
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger

import config


class VocalExtractor:
    """
    Wraps Demucs to isolate the vocal stem from a music track.

    Args:
        model:    Demucs model name (default: ``config.DEMUCS_MODEL``).
        device:   ``"cuda"`` or ``"cpu"``.
        segment:  Chunk size in seconds (reduce if VRAM is exhausted).
        overlap:  Overlap fraction between consecutive chunks.
        shifts:   Number of random-shift augmentations (≥2 improves quality
                  at the cost of extra compute).
        out_dir:  Directory for output WAV files.  Defaults to
                  ``config.TEMP_DIR/demucs``.
    """

    def __init__(
        self,
        model: str = config.DEMUCS_MODEL,
        device: str = config.DEMUCS_DEVICE,
        segment: float = config.DEMUCS_SEGMENT,
        overlap: float = config.DEMUCS_OVERLAP,
        shifts: int = config.DEMUCS_SHIFTS,
        out_dir: Optional[Path | str] = None,
    ) -> None:
        self.model   = model
        self.device  = device
        self.segment = segment
        self.overlap = overlap
        self.shifts  = shifts
        self.out_dir = Path(out_dir) if out_dir else config.TEMP_DIR / "demucs"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def extract(self, audio_path: str | Path) -> Path:
        """
        Separate vocals from *audio_path* and return the vocal WAV path.

        The method runs Demucs in a temporary directory and copies the
        resulting ``vocals.wav`` to ``self.out_dir``.

        Args:
            audio_path: Path to the input audio file (mp3 / wav / flac / …).

        Returns:
            Absolute :class:`~pathlib.Path` to the extracted vocal WAV.

        Raises:
            FileNotFoundError: If *audio_path* does not exist.
            RuntimeError:      If Demucs exits with a non-zero return code.
        """
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"[VocalExtractor] ▶ {audio_path.name}")
        logger.info(
            f"  model={self.model}  device={self.device}  "
            f"segment={self.segment}s  shifts={self.shifts}"
        )

        dest = self.out_dir / f"{audio_path.stem}_vocals.wav"

        with tempfile.TemporaryDirectory(prefix="lyrical_demucs_") as tmp:
            tmp_path = Path(tmp)
            self._run_demucs(audio_path, tmp_path)
            vocal_wav = self._find_vocals(tmp_path)
            shutil.copy2(vocal_wav, dest)

        logger.success(f"[VocalExtractor] Vocals → {dest}")
        return dest

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _run_demucs(self, audio_path: Path, tmp_dir: Path) -> None:
        """
        Invoke Demucs via ``python -m demucs`` in a subprocess.

        Using ``--two-stems vocals`` makes Demucs output only
        ``vocals.wav`` + ``no_vocals.wav``, which is ~2× faster than
        full 4-stem separation.
        """
        cmd = [
            "python", "-m", "demucs",
            "--name",       self.model,
            "--device",     self.device,
            "--segment",    str(self.segment),
            "--overlap",    str(self.overlap),
            "--shifts",     str(self.shifts),
            "--two-stems",  "vocals",   # vocals vs. accompaniment only
            "--out",        str(tmp_dir),
            str(audio_path),
        ]

        logger.debug(f"  CMD: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

        if result.returncode != 0:
            logger.error(result.stderr[-3000:])   # last 3 kB of stderr
            raise RuntimeError(
                f"Demucs exited with code {result.returncode}.\n"
                f"Stderr (last 3 kB):\n{result.stderr[-3000:]}"
            )

    def _find_vocals(self, tmp_dir: Path) -> Path:
        """
        Locate the ``vocals.wav`` produced by Demucs.

        Demucs writes:
          ``<out>/<model_name>/<track_stem>/vocals.wav``

        Falls back to any ``*.wav`` whose name contains ``"vocals"``.
        """
        candidates = list(tmp_dir.rglob("vocals.wav"))

        if not candidates:
            # Broader search
            candidates = [p for p in tmp_dir.rglob("*.wav") if "vocal" in p.stem.lower()]

        if not candidates:
            raise FileNotFoundError(
                f"Demucs did not produce a vocals WAV under {tmp_dir}. "
                "Check the stderr output above."
            )

        if len(candidates) > 1:
            logger.debug(f"  Multiple vocal files found; using first: {candidates[0]}")

        return candidates[0]


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract vocals with Demucs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio",               help="Input audio file")
    parser.add_argument("--model",   default=config.DEMUCS_MODEL)
    parser.add_argument("--device",  default=config.DEMUCS_DEVICE, choices=["cuda", "cpu"])
    parser.add_argument("--segment", default=config.DEMUCS_SEGMENT, type=float)
    parser.add_argument("--shifts",  default=config.DEMUCS_SHIFTS,  type=int)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    extractor = VocalExtractor(
        model=args.model,
        device=args.device,
        segment=args.segment,
        shifts=args.shifts,
        out_dir=args.out_dir,
    )
    result_path = extractor.extract(args.audio)
    print(f"\nVocal track saved → {result_path}")
