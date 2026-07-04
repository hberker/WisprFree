"""Word error rate — the metric that tells the user whether personal
fine-tuning actually helped their voice. Kept dependency-free."""

from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[^\w\s']", flags=re.UNICODE)


def normalize(text: str) -> list[str]:
    return _PUNCT_RE.sub(" ", text.lower()).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words / reference length."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    # standard DP edit distance
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current[j] = min(
                previous[j] + 1,        # deletion
                current[j - 1] + 1,     # insertion
                previous[j - 1] + cost, # substitution
            )
        previous = current
    return previous[-1] / len(ref)


def corpus_wer(pairs: list[tuple[str, str]]) -> float:
    """WER over a list of (reference, hypothesis), weighted by ref length."""
    total_errors = 0.0
    total_words = 0
    for reference, hypothesis in pairs:
        ref_len = len(normalize(reference))
        total_errors += word_error_rate(reference, hypothesis) * max(ref_len, 1)
        total_words += max(ref_len, 1)
    return total_errors / total_words if total_words else 0.0
