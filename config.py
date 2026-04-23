from pathlib import Path

ROOT_DIR  = Path(__file__).parent.resolve()
INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output"
MODELS_DIR = ROOT_DIR / "models"
TEMP_DIR   = ROOT_DIR / "temp"

for _d in [INPUT_DIR, OUTPUT_DIR, MODELS_DIR, TEMP_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# Demucs
DEMUCS_MODEL   = "htdemucs"
DEMUCS_DEVICE  = "cuda"
DEMUCS_SEGMENT = 7.8
DEMUCS_OVERLAP = 0.25
DEMUCS_SHIFTS  = 1


# Whisper
WHISPER_MODEL_SIZE    = "large-v3"
WHISPER_DEVICE        = "cuda"
WHISPER_COMPUTE_TYPE  = "float16"
WHISPER_LOCAL_MODEL_PATH: str | None = None  # set to local path to skip download
WHISPER_LANGUAGE        = "fr"  # None = auto-detect
# 常见语言缩写: zh=中文 en=英文 ja=日文 ko=韩文 fr=法文 de=德文
#               es=西班牙文 it=意大利文 ru=俄文 pt=葡萄牙文 ar=阿拉伯文
WHISPER_BEAM_SIZE       = 5
WHISPER_WORD_TIMESTAMPS = True
WHISPER_VAD_FILTER      = True
WHISPER_VAD_THRESHOLD   = 0.5


# LRC output
LRC_TITLE  = ""
LRC_ARTIST = ""
LRC_ALBUM  = ""
LRC_OFFSET = 0


# Post-processing
PP_MIN_SEGMENT_DURATION = 0.3
PP_MAX_CHARS_PER_LINE   = 50
PP_MERGE_GAP_THRESHOLD  = 0.30


# Translation (set TRANSLATION_TARGET_LANG to enable, e.g. "zh", "en")
TRANSLATION_TARGET_LANG: str | None = "zh"
TRANSLATION_BACKEND     = "google"  # "google" | "deepl" | "argos"
TRANSLATION_SOURCE_LANG = "fr" # set to None for auto-detection
TRANSLATION_BATCH_DELAY = 0.5
DEEPL_API_KEY: str = ""


# Pipeline
SKIP_SEPARATION   = False  # True if input is already vocals-only
LRC_BY_WORD       = False  # True for word-level Enhanced LRC
SAVE_INTERMEDIATES = True
