"""
config.py
─────────
Central configuration for the Lyrical-Aligner project.
All tuneable parameters live here; individual modules import this file
instead of hard-coding values.
"""

from pathlib import Path

# ── Project layout ────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).parent.resolve()
INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output"
MODELS_DIR = ROOT_DIR / "models"
TEMP_DIR   = ROOT_DIR / "temp"

# Ensure runtime directories exist
for _d in [INPUT_DIR, OUTPUT_DIR, MODELS_DIR, TEMP_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ── Demucs (Vocal Separation) ─────────────────────────────────────
# Available models: htdemucs | htdemucs_ft | htdemucs_6s | mdx_extra
DEMUCS_MODEL   = "htdemucs"   # htdemucs_ft for higher quality (slower)
DEMUCS_DEVICE  = "cuda"       # "cuda" or "cpu"
DEMUCS_SEGMENT = 7.8          # Chunk length in seconds; reduce to 4–5 if OOM
DEMUCS_OVERLAP = 0.25         # Overlap fraction between consecutive chunks
DEMUCS_SHIFTS  = 1            # Random-shift augmentations (≥2 improves quality)


# ── faster-whisper (ASR) ──────────────────────────────────────────
# Model sizes:  tiny | base | small | medium | large-v2 | large-v3
WHISPER_MODEL_SIZE    = "large-v3"
WHISPER_DEVICE        = "cuda"         # "cuda" or "cpu"
WHISPER_COMPUTE_TYPE  = "float16"      # GPU: "float16" / CPU: "int8"

# Set to an absolute local directory path to load model from disk instead of HF Hub.
# Example: r"D:\models\faster-whisper-large-v3"
WHISPER_LOCAL_MODEL_PATH: str | None = None

# Transcription hyper-parameters
WHISPER_LANGUAGE        = None   # None = auto-detect; force with "zh", "en", "ja", …
WHISPER_BEAM_SIZE       = 5
WHISPER_WORD_TIMESTAMPS = True
WHISPER_VAD_FILTER      = True   # Silero-VAD pre-filter — reduces hallucinations
WHISPER_VAD_THRESHOLD   = 0.5    # Probability threshold for speech frames


# ── LRC Output ────────────────────────────────────────────────────
LRC_TITLE  = ""   # Written to [ti:] tag; auto-filled from filename if empty
LRC_ARTIST = ""   # Written to [ar:] tag
LRC_ALBUM  = ""   # Written to [al:] tag
LRC_OFFSET = 0    # Global time offset in milliseconds (written to [offset:] tag)


# ── Post-Processing ───────────────────────────────────────────────
PP_MIN_SEGMENT_DURATION = 0.3   # Discard segments shorter than this (seconds)
PP_MAX_CHARS_PER_LINE   = 50    # Soft character cap per LRC line before splitting
PP_MERGE_GAP_THRESHOLD  = 0.30  # Merge adjacent segments with gap ≤ this (seconds)


# ── Translation (optional) ────────────────────────────────────────
# Set TRANSLATION_TARGET_LANG to an ISO-639-1 code (e.g. "zh", "en", "ja")
# to translate lyrics to that language before generating the LRC file.
# Leave as None to disable translation (default).
TRANSLATION_TARGET_LANG: str | None = None

# Backend choices: "google" | "deepl" | "argos"
#   google — free, no API key, requires internet (via deep-translator)
#   deepl  — higher quality, requires DEEPL_API_KEY env var
#   argos  — fully offline; run `python translator.py --install-argos <src> <tgt>`
TRANSLATION_BACKEND     = "google"

# Source language for translation ("auto" = detect from ASR result)
TRANSLATION_SOURCE_LANG = "auto"

# Seconds to sleep between successive API calls (rate-limit safety)
TRANSLATION_BATCH_DELAY = 0.5

# DeepL API key — only needed when TRANSLATION_BACKEND = "deepl".
# Recommended: set the DEEPL_API_KEY environment variable instead.
DEEPL_API_KEY: str = ""


# ── Pipeline behaviour ────────────────────────────────────────────
# These settings control the end-to-end pipeline run by pipeline.py.
# Edit them here instead of passing command-line flags.

# Set to True if the input audio is already a vocals-only file (skips Demucs)
SKIP_SEPARATION   = False

# Set to True to use Enhanced LRC (A2) format with word-level inline timestamps.
# NOTE: word-level timestamps are cleared when translation is active.
LRC_BY_WORD       = False

# Set to True to save a <stem>_segments.json alongside the .lrc for debugging
SAVE_INTERMEDIATES = True
