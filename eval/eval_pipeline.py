"""
Evaluation pipeline for Lyrical-Aligner.

Runs any combination of the four evaluation stages depending on what
ground-truth data is available in the testset JSON.

Usage
-----
    # Full eval (all stages with GT)
    python eval/eval_pipeline.py --testset eval/testset_example.json

    # Skip vocal separation (use pre-existing vocals in testset vocals_path field)
    python eval/eval_pipeline.py --testset eval/testset_example.json --skip-separation

    # Save JSON report
    python eval/eval_pipeline.py --testset eval/testset_example.json --output eval/report.json

Testset format
--------------
See eval/testset_example.json for the full schema.
Each entry may omit fields it cannot provide — those stages are skipped gracefully.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from vocal_extractor      import VocalExtractor
from transcription_engine import TranscriptionEngine, Segment
from postprocessor        import PostProcessor
from translator           import Translator
from metrics import wer, cer, timestamp_mae, timestamp_accuracy, sdr, bleu, chrf


# ── audio helpers ─────────────────────────────────────────────────────────────

def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


def _resample_if_needed(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=src_sr, target_sr=tgt_sr)


# ── per-stage evaluators ──────────────────────────────────────────────────────

def eval_separation(
    audio_path: Path,
    vocals_ref_path: Path,
    extractor: VocalExtractor,
) -> dict:
    logger.info("[Eval/Separation] Running Demucs ...")
    vocals_path = extractor.extract(audio_path)

    ref, sr_ref = _load_mono(vocals_ref_path)
    est, sr_est = _load_mono(vocals_path)
    est = _resample_if_needed(est, sr_est, sr_ref)

    score = sdr(ref, est)
    logger.info(f"[Eval/Separation] SDR = {score:.2f} dB")
    return {
        "sdr_db": round(score, 3),
        "vocals_path": str(vocals_path),
    }


def eval_transcription(
    vocals_path: Path,
    segments_ref: List[dict],
    engine: TranscriptionEngine,
    postprocessor: PostProcessor,
) -> dict:
    logger.info("[Eval/Transcription] Running Whisper ...")
    segments = postprocessor.process(engine.transcribe(vocals_path))

    n_ref = len(segments_ref)
    pred_text = " ".join(s.text for s in segments[:n_ref])
    ref_text  = " ".join(r["text"] for r in segments_ref)

    pred_dicts = [{"start": s.start, "end": s.end} for s in segments]

    result = {
        "wer":                   round(wer(ref_text, pred_text), 4),
        "cer":                   round(cer(ref_text, pred_text), 4),
        "timestamp_mae_start_s": round(timestamp_mae(pred_dicts, segments_ref, "start"), 4),
        "timestamp_mae_end_s":   round(timestamp_mae(pred_dicts, segments_ref, "end"), 4),
        "timestamp_acc_0.3s":    round(timestamp_accuracy(pred_dicts, segments_ref, 0.3, "start"), 4),
        "n_pred_segments":       len(segments),
        "n_ref_segments":        len(segments_ref),
    }

    logger.info(
        f"[Eval/Transcription] WER={result['wer']:.3f}  CER={result['cer']:.3f}  "
        f"MAE={result['timestamp_mae_start_s']:.3f}s  "
        f"Acc@0.3s={result['timestamp_acc_0.3s']:.1%}"
    )
    return result, segments   # return segments for optional translation stage


def eval_translation(
    pred_segments: List[Segment],
    translation_ref: List[dict],
    target_lang: str,
    source_lang: str,
) -> dict:
    logger.info(f"[Eval/Translation] Translating {source_lang} → {target_lang} ...")
    translator = Translator(
        target_lang=target_lang,
        backend=config.TRANSLATION_BACKEND,
    )
    translated = translator.translate_segments(pred_segments, source_lang_hint=source_lang)

    n = min(len(translated), len(translation_ref))
    pred_text = " ".join(translated[i].text for i in range(n))
    ref_text  = " ".join(translation_ref[i]["text"] for i in range(n))

    result = {
        "bleu": bleu(ref_text, pred_text),
        "chrf": chrf(ref_text, pred_text, n=2),
    }
    logger.info(f"[Eval/Translation] BLEU={result['bleu']:.3f}  chrF={result['chrf']:.3f}")
    return result


# ── per-item runner ───────────────────────────────────────────────────────────

def run_item(
    item: dict,
    extractor: VocalExtractor,
    engine: TranscriptionEngine,
    postprocessor: PostProcessor,
    skip_separation: bool,
) -> dict:
    name        = item["name"]
    audio_path  = Path(item["audio"]).resolve()
    source_lang = item.get("language", "")
    target_lang = item.get("target_lang")

    vocals_ref_path  = Path(item["vocals_ref"]).resolve()  if item.get("vocals_ref")  else None
    vocals_path_hint = Path(item["vocals_path"]).resolve() if item.get("vocals_path") else None
    segments_ref     = item.get("segments_ref", [])
    translation_ref  = item.get("translation_ref", [])

    result = {"name": name}
    vocals_path: Optional[Path] = vocals_path_hint

    # ── Stage 1: vocal separation ─────────────────────────────────────────────
    if vocals_ref_path and not skip_separation:
        sep = eval_separation(audio_path, vocals_ref_path, extractor)
        result["separation"] = sep
        vocals_path = Path(sep["vocals_path"])
    elif vocals_path is None:
        # No ref and no pre-computed path: just run extraction silently
        logger.info("[Eval] No vocals_ref — running extraction without SDR measurement")
        vocals_path = extractor.extract(audio_path)

    # ── Stage 2+3: transcription + post-processing ────────────────────────────
    if segments_ref:
        trans_metrics, pred_segments = eval_transcription(
            vocals_path, segments_ref, engine, postprocessor
        )
        result["transcription"] = trans_metrics
    else:
        logger.info("[Eval] No segments_ref — skipping transcription metrics")
        pred_segments = postprocessor.process(engine.transcribe(vocals_path))

    # ── Stage 4: translation ──────────────────────────────────────────────────
    if translation_ref and target_lang:
        result["translation"] = eval_translation(
            pred_segments, translation_ref, target_lang, source_lang
        )
    elif target_lang and not translation_ref:
        logger.info("[Eval] No translation_ref — skipping translation metrics")

    return result


# ── summary ───────────────────────────────────────────────────────────────────

def _mean(values: list) -> Optional[float]:
    finite = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]
    return round(float(np.mean(finite)), 4) if finite else None


def print_summary(results: list[dict]) -> None:
    sep_sdrs   = [r["separation"]["sdr_db"]                      for r in results if "separation"    in r]
    wers       = [r["transcription"]["wer"]                      for r in results if "transcription" in r]
    cers       = [r["transcription"]["cer"]                      for r in results if "transcription" in r]
    mae_starts = [r["transcription"]["timestamp_mae_start_s"]    for r in results if "transcription" in r]
    accs       = [r["transcription"]["timestamp_acc_0.3s"]       for r in results if "transcription" in r]
    bleus      = [r["translation"]["bleu"]                       for r in results if "translation"   in r]
    chrfs      = [r["translation"]["chrf"]                       for r in results if "translation"   in r]

    w = 52
    print("\n" + "═" * w)
    print(f"{'EVALUATION SUMMARY':^{w}}")
    print("═" * w)
    print(f"  Songs evaluated : {len(results)}")

    if sep_sdrs:
        print(f"\n  [Vocal Separation]")
        print(f"    SDR            : {_mean(sep_sdrs):>7.2f} dB   (higher is better, >10 dB = good)")

    if wers:
        print(f"\n  [Transcription]")
        print(f"    WER            : {_mean(wers):>7.1%}   (lower is better)")
        print(f"    CER            : {_mean(cers):>7.1%}   (lower is better, use for CJK)")
        print(f"    Timestamp MAE  : {_mean(mae_starts):>7.3f} s   (lower is better, ≤0.3 s = good)")
        print(f"    Acc @ ±0.3 s   : {_mean(accs):>7.1%}   (higher is better)")

    if bleus:
        print(f"\n  [Translation]")
        print(f"    BLEU           : {_mean(bleus):>7.3f}    (0–1, higher is better)")
        print(f"    chrF           : {_mean(chrfs):>7.3f}    (0–1, higher is better)")

    print("═" * w + "\n")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lyrical-Aligner evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--testset",          required=True,  help="Path to testset JSON")
    parser.add_argument("--output",           default=None,   help="Save JSON report to this path")
    parser.add_argument("--skip-separation",  action="store_true",
                        help="Skip Demucs; use vocals_path field from testset instead")
    args = parser.parse_args()

    testset_path = Path(args.testset)
    with open(testset_path, encoding="utf-8") as f:
        testset = json.load(f)

    logger.info(f"Loaded testset: {len(testset)} item(s) from {testset_path}")

    extractor    = VocalExtractor()
    engine       = TranscriptionEngine(
        model_size=config.WHISPER_MODEL_SIZE,
        device=config.WHISPER_DEVICE,
        language=config.WHISPER_LANGUAGE,
    )
    postprocessor = PostProcessor()

    results = []
    for item in testset:
        logger.info(f"\n{'─' * 52}\n  Item: {item['name']}\n{'─' * 52}")
        result = run_item(item, extractor, engine, postprocessor, args.skip_separation)
        results.append(result)
        logger.success(f"  Finished: {item['name']}")

    print_summary(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.success(f"Report saved → {out_path}")


if __name__ == "__main__":
    main()
