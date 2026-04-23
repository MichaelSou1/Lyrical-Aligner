"""
Evaluation metrics for Lyrical-Aligner.

  wer / cer      — transcription accuracy
  timestamp_mae  — timestamp precision (seconds)
  sdr            — vocal separation quality (dB)
  bleu / chrf    — translation quality
"""
from __future__ import annotations

import math
from collections import Counter
from typing import List

import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """
    Word-level tokenizer that inserts spaces around each CJK character
    so BLEU works for Chinese / Japanese / Korean without a dedicated segmenter.
    Latin text is left as-is (split on whitespace).
    """
    tokens: List[str] = []
    for char in text:
        cp = ord(char)
        if (0x4E00 <= cp <= 0x9FFF      # CJK Unified
                or 0x3040 <= cp <= 0x30FF  # Hiragana / Katakana
                or 0xAC00 <= cp <= 0xD7AF  # Hangul
                or 0x3400 <= cp <= 0x4DBF):
            tokens.append(char)
        else:
            tokens.extend(char.split())  # handles spaces, punctuation clusters
    return [t for t in tokens if t.strip()]


def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if a[i - 1] == b[j - 1] else 1 + min(prev[j - 1], prev[j], dp[j - 1])
    return dp[n]


# ── transcription ─────────────────────────────────────────────────────────────

def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate. Lower is better (0 = perfect)."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate. Lower is better. Preferred for CJK languages."""
    ref = list(reference.replace(" ", ""))
    hyp = list(hypothesis.replace(" ", ""))
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


# ── timestamp alignment ───────────────────────────────────────────────────────

def timestamp_mae(
    pred: List[dict],
    ref: List[dict],
    field: str = "start",
) -> float:
    """
    Mean Absolute Error between predicted and reference timestamps (seconds).
    Pairs are matched by position up to min(len(pred), len(ref)).
    Industry threshold: ≤ 0.3 s is considered acceptable.
    """
    n = min(len(pred), len(ref))
    if n == 0:
        return float("nan")
    errors = [abs(pred[i][field] - ref[i][field]) for i in range(n)]
    return float(np.mean(errors))


def timestamp_accuracy(
    pred: List[dict],
    ref: List[dict],
    threshold: float = 0.3,
    field: str = "start",
) -> float:
    """Fraction of timestamps within ±threshold seconds of reference."""
    n = min(len(pred), len(ref))
    if n == 0:
        return float("nan")
    hits = sum(abs(pred[i][field] - ref[i][field]) <= threshold for i in range(n))
    return hits / n


# ── source separation ─────────────────────────────────────────────────────────

def sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Signal-to-Distortion Ratio (dB). Higher is better.
    Projects estimate onto reference to isolate the target component,
    then measures signal vs. distortion energy.

    Both arrays should be mono float32. Length mismatch is handled by truncation.
    Typical good values: > 10 dB.
    """
    min_len = min(len(reference), len(estimate))
    ref = reference[:min_len].astype(np.float64)
    est = estimate[:min_len].astype(np.float64)

    alpha = np.dot(ref, est) / (np.dot(ref, ref) + 1e-10)
    target = alpha * ref
    noise  = est - target

    return float(10 * np.log10((np.sum(target ** 2) + 1e-10) / (np.sum(noise ** 2) + 1e-10)))


# ── translation ───────────────────────────────────────────────────────────────

def _ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    Sentence-level BLEU-4 with brevity penalty (single reference).
    Range: 0–1. Higher is better.
    Automatically uses character-level tokenization for CJK text.
    Note: song lyric translations are often paraphrases; use chrF as complement.
    """
    ref_tok = _tokenize(reference)
    hyp_tok = _tokenize(hypothesis)
    if not hyp_tok or not ref_tok:
        return 0.0

    log_score = 0.0
    for n in range(1, max_n + 1):
        ref_ng = _ngrams(ref_tok, n)
        hyp_ng = _ngrams(hyp_tok, n)
        clipped = sum(min(cnt, ref_ng[gram]) for gram, cnt in hyp_ng.items())
        total   = max(len(hyp_tok) - n + 1, 0)
        if total == 0 or clipped == 0:
            return 0.0
        log_score += math.log(clipped / total)

    bp = min(1.0, math.exp(1 - len(ref_tok) / max(len(hyp_tok), 1)))
    return round(bp * math.exp(log_score / max_n), 4)


def chrf(reference: str, hypothesis: str, n: int = 6, beta: float = 1.0) -> float:
    """
    chrF (character n-gram F-score). Range: 0–1. Higher is better.
    More robust than BLEU for morphologically rich or CJK languages.
    """
    def char_ngrams(text: str, n: int) -> Counter:
        s = text.replace(" ", "")
        return Counter(s[i: i + n] for i in range(len(s) - n + 1))

    ref_ng = char_ngrams(reference, n)
    hyp_ng = char_ngrams(hypothesis, n)
    if not ref_ng or not hyp_ng:
        return 0.0

    matches   = sum(min(ref_ng[g], hyp_ng[g]) for g in hyp_ng)
    precision = matches / sum(hyp_ng.values())
    recall    = matches / sum(ref_ng.values())
    if precision + recall == 0:
        return 0.0

    return round((1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall), 4)
