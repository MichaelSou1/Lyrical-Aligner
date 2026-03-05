"""
download_models.py
──────────────────
Handles downloading / caching of faster-whisper models and verifies
that Demucs weights are available via torch.hub.

Usage
─────
    # Download default model (large-v3) from HuggingFace Hub
    python download_models.py

    # Specify a different model size
    python download_models.py --model medium

    # Load from a local directory (no network required)
    python download_models.py --local D:/models/faster-whisper-large-v3

    # Skip Demucs check (faster when only Whisper is needed)
    python download_models.py --skip-demucs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

import config


# ── Whisper loader ────────────────────────────────────────────────

def load_whisper_model(
    model_size_or_path: str = config.WHISPER_MODEL_SIZE,
    device: str = config.WHISPER_DEVICE,
    compute_type: str = config.WHISPER_COMPUTE_TYPE,
):
    """
    Load a faster-whisper WhisperModel from HuggingFace Hub or a local path.

    Args:
        model_size_or_path: Model name (e.g. ``"large-v3"``) **or** an absolute
                            path to a locally stored model directory.
        device:             ``"cuda"`` or ``"cpu"``.
        compute_type:       ``"float16"`` (GPU) / ``"int8"`` (CPU) /
                            ``"int8_float16"`` / ``"float32"``.

    Returns:
        A loaded :class:`faster_whisper.WhisperModel` instance.

    Raises:
        RuntimeError: If CUDA is requested but not available and no fallback
                      is configured.
    """
    from faster_whisper import WhisperModel

    # Prefer a local path if explicitly configured in config.py
    source = config.WHISPER_LOCAL_MODEL_PATH or model_size_or_path

    # Resolve: is it a local directory?
    if Path(source).is_dir():
        logger.info(f"[Whisper] Loading model from local path → {source}")
    else:
        logger.info(f"[Whisper] Downloading / loading model from Hub → {source}")
        logger.info(f"          Cache directory: {config.MODELS_DIR / 'whisper'}")

    # CUDA availability guard
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning(
                    "CUDA requested but torch.cuda.is_available() = False. "
                    "Falling back to CPU with int8 compute type."
                )
                device = "cpu"
                compute_type = "int8"
        except ImportError:
            logger.warning("torch not importable; defaulting to CPU.")
            device = "cpu"
            compute_type = "int8"

    model = WhisperModel(
        source,
        device=device,
        compute_type=compute_type,
        download_root=str(config.MODELS_DIR / "whisper"),
    )

    logger.success(
        f"[Whisper] Model ready  ·  source={source}  device={device}  "
        f"compute_type={compute_type}"
    )
    return model


# ── Demucs verifier ───────────────────────────────────────────────

def verify_demucs_model(model_name: str = config.DEMUCS_MODEL) -> None:
    """
    Trigger a Demucs weight download if the model is not already cached.

    Demucs manages its own cache via ``torch.hub``; calling
    ``demucs.pretrained.get_model`` the first time downloads the weights.

    Args:
        model_name: Demucs model identifier, e.g. ``"htdemucs"``.
    """
    try:
        import demucs.pretrained as dp  # type: ignore[import]
        logger.info(f"[Demucs] Verifying model weights → {model_name}")
        dp.get_model(model_name)
        logger.success(f"[Demucs] Model [{model_name}] is ready.")
    except Exception as exc:
        logger.error(f"[Demucs] Failed to load [{model_name}]: {exc}")
        raise


# ── CLI ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Lyrical-Aligner — Model downloader / verifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default=config.WHISPER_MODEL_SIZE,
        help="Whisper model size (tiny/base/small/medium/large-v2/large-v3) "
             "or HuggingFace model ID",
    )
    p.add_argument(
        "--local",
        default=None,
        metavar="PATH",
        help="Absolute path to a locally stored faster-whisper model directory",
    )
    p.add_argument(
        "--device",
        default=config.WHISPER_DEVICE,
        choices=["cuda", "cpu"],
    )
    p.add_argument(
        "--compute-type",
        default=config.WHISPER_COMPUTE_TYPE,
        choices=["float16", "int8_float16", "int8", "float32"],
    )
    p.add_argument(
        "--demucs-model",
        default=config.DEMUCS_MODEL,
        help="Demucs model name to verify / pre-download",
    )
    p.add_argument(
        "--skip-demucs",
        action="store_true",
        help="Skip Demucs model verification",
    )
    p.add_argument(
        "--skip-whisper",
        action="store_true",
        help="Skip Whisper model download / verification",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # Override local path if provided via CLI
    if args.local:
        config.WHISPER_LOCAL_MODEL_PATH = args.local

    if not args.skip_whisper:
        load_whisper_model(
            model_size_or_path=args.local or args.model,
            device=args.device,
            compute_type=args.compute_type,
        )

    if not args.skip_demucs:
        verify_demucs_model(model_name=args.demucs_model)

    logger.success("All requested models are ready.")


if __name__ == "__main__":
    main()
