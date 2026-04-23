from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger

import config


class VocalExtractor:
    """Extracts vocals from a music track using Demucs."""

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

    def extract(self, audio_path: str | Path) -> Path:
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"[VocalExtractor] ▶ {audio_path.name}")
        logger.info(
            f"  model={self.model}  device={self.device}  "
            f"segment={self.segment}s  shifts={self.shifts}"
        )

        stem = audio_path.stem
        dest_vocals = self.out_dir / f"{stem}_vocals.wav"
        dest_accompaniment = self.out_dir / f"{stem}_accompaniment.wav"

        with tempfile.TemporaryDirectory(prefix="lyrical_demucs_") as tmp:
            tmp_path = Path(tmp)
            if audio_path.suffix.lower() != ".wav":
                audio_path = self._to_wav(audio_path, tmp_path)
            self._run_demucs(audio_path, tmp_path)
            vocal_wav = self._find_stems(tmp_path, "vocals")
            no_vocal_wav = self._find_stems(tmp_path, "no_vocals")
            shutil.copy2(vocal_wav, dest_vocals)
            shutil.copy2(no_vocal_wav, dest_accompaniment)

        logger.success(f"[VocalExtractor] Vocals        → {dest_vocals}")
        logger.success(f"[VocalExtractor] Accompaniment → {dest_accompaniment}")
        return dest_vocals

    def _to_wav(self, audio_path: Path, tmp_dir: Path) -> Path:
        import av as _av
        wav_path = tmp_dir / f"{audio_path.stem}.wav"
        logger.info(f"[VocalExtractor] Converting {audio_path.suffix} → WAV via PyAV")
        with _av.open(str(audio_path)) as src:
            with _av.open(str(wav_path), "w", format="wav") as dst:
                out_stream = dst.add_stream("pcm_s16le", rate=44100, layout="stereo")
                resampler = _av.AudioResampler(format="s16", layout="stereo", rate=44100)
                for frame in src.decode(audio=0):
                    for rf in resampler.resample(frame):
                        rf.pts = None
                        dst.mux(out_stream.encode(rf))
                for pkt in out_stream.encode(None):
                    dst.mux(pkt)
        return wav_path

    def _run_demucs(self, audio_path: Path, tmp_dir: Path) -> None:
        cmd = [
            sys.executable, "-m", "demucs",
            "--name",       self.model,
            "--device",     self.device,
            "--segment",    str(int(self.segment)),
            "--overlap",    str(self.overlap),
            "--shifts",     str(self.shifts),
            "--two-stems",  "vocals",
            "--out",        str(tmp_dir),
            str(audio_path),
        ]

        env = os.environ.copy()
        conda_bin = Path(sys.executable).parent / "Library" / "bin"
        if conda_bin.exists():
            env["PATH"] = str(conda_bin) + os.pathsep + env.get("PATH", "")

        logger.debug(f"  CMD: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, text=True, encoding="utf-8", env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

        if result.returncode != 0:
            logger.error(result.stdout[-3000:])
            raise RuntimeError(
                f"Demucs exited with code {result.returncode}.\n"
                f"Output (last 3 kB):\n{result.stdout[-3000:]}"
            )

    def _find_stems(self, tmp_dir: Path, stem_name: str) -> Path:
        candidates = list(tmp_dir.rglob(f"{stem_name}.wav"))
        if not candidates:
            raise FileNotFoundError(
                f"Demucs did not produce '{stem_name}.wav' under {tmp_dir}."
            )
        return candidates[0]



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
