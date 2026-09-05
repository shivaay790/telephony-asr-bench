"""Error rates, and the text normalisation that decides them.

Most of the disagreement between two people's WER numbers comes from
normalisation, not from the recogniser. A provider that writes "twenty five
dollars" loses to one that writes "$25" unless both are normalised first, and
a provider that omits punctuation wins unfairly if punctuation is not stripped.

Normalisation here is explicit, ordered, and reported alongside the score so a
number can be reproduced or argued with.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

_PUNCT = re.compile(r"[^\w\s']", flags=re.UNICODE)
_WS = re.compile(r"\s+")

CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
    "'m": " am", "let's": "let us", "it's": "it is",
}

_ONES = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


@dataclass(frozen=True)
class NormalizerConfig:
    lowercase: bool = True
    strip_punctuation: bool = True
    expand_contractions: bool = True
    spell_out_numbers: bool = True
    collapse_whitespace: bool = True
    strip_filler: bool = False   # "uh", "um" — off by default; they are real speech

    def describe(self) -> str:
        on = [k for k, v in self.__dict__.items() if v]
        return ", ".join(on) if on else "none"


FILLERS = {"uh", "um", "erm", "mm", "hmm", "uhh", "ah"}


_DIGITS = {k: v for k, v in _ONES.items() if v < 10}


def _words_to_number(tokens: Sequence[str]) -> str | None:
    """Convert a run of number words to digits.

    Two different things get spoken as number words and they must not be
    conflated:

      "twenty five"            -> 25    a quantity, summed
      "four eight two"         -> 482   a digit string, concatenated

    Claim numbers, member IDs, phone numbers and dates of birth are all read
    out as digit strings, and they are exactly the fields a voice agent has to
    get exactly right. Summing them ("one two three" -> 6) silently corrupts
    the reference and every WER computed against it.
    """
    if not tokens:
        return None

    # a run of two or more bare single digits is a digit string
    if len(tokens) >= 2 and all(t in _DIGITS for t in tokens):
        return "".join(str(_DIGITS[t]) for t in tokens)

    total, current, seen = 0, 0, False
    for tok in tokens:
        if tok in _ONES:
            current += _ONES[tok]; seen = True
        elif tok in _TENS:
            current += _TENS[tok]; seen = True
        elif tok == "hundred" and seen:
            current = max(current, 1) * 100
        elif tok == "thousand" and seen:
            total += max(current, 1) * 1000; current = 0
        else:
            return None
    return str(total + current) if seen else None


def normalize(text: str, cfg: NormalizerConfig = NormalizerConfig()) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    if cfg.lowercase:
        t = t.lower()
    if cfg.expand_contractions:
        for src, dst in CONTRACTIONS.items():
            t = t.replace(src, dst)
    if cfg.strip_punctuation:
        t = _PUNCT.sub(" ", t)
    if cfg.collapse_whitespace:
        t = _WS.sub(" ", t).strip()

    tokens = t.split()
    if cfg.strip_filler:
        tokens = [w for w in tokens if w not in FILLERS]

    if cfg.spell_out_numbers and tokens:
        out, buf = [], []
        for tok in tokens:
            if tok in _ONES or tok in _TENS or tok in ("hundred", "thousand"):
                buf.append(tok)
                continue
            if buf:
                num = _words_to_number(buf)
                out.extend([num] if num else buf)
                buf = []
            out.append(tok)
        if buf:
            num = _words_to_number(buf)
            out.extend([num] if num else buf)
        tokens = out

    return " ".join(tokens)


# --------------------------------------------------------------------------
# edit distance
# --------------------------------------------------------------------------

@dataclass
class ErrorCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_length: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """Error rate. Can exceed 1.0 when the hypothesis hallucinates —
        which is worth seeing rather than clamping away, because a model
        that invents 40 words on a silent segment is a production incident."""
        if self.reference_length == 0:
            return 0.0 if self.errors == 0 else float("inf")
        return self.errors / self.reference_length

    def merged(self, other: "ErrorCounts") -> "ErrorCounts":
        return ErrorCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.reference_length + other.reference_length,
        )


def _levenshtein(ref: Sequence, hyp: Sequence) -> ErrorCounts:
    """Standard DP alignment, two rows plus a backtrace-free operation count.

    Operation counts are tracked alongside the cost so the report can say
    *how* a provider fails: deletions mean it is dropping audio, insertions
    mean it is hallucinating, substitutions mean it mishears.
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return ErrorCounts(0, 0, m, 0)
    if m == 0:
        return ErrorCounts(0, n, 0, n)

    # each cell: (cost, sub, del, ins)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [(0, 0, 0, 0)] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost, s, d, ins = prev[j - 1]
                cur[j] = (cost, s, d, ins)
                continue
            sub_c, sub_s, sub_d, sub_i = prev[j - 1]
            del_c, del_s, del_d, del_i = prev[j]
            ins_c, ins_s, ins_d, ins_i = cur[j - 1]
            best = min((sub_c + 1, sub_s + 1, sub_d, sub_i),
                       (del_c + 1, del_s, del_d + 1, del_i),
                       (ins_c + 1, ins_s, ins_d, ins_i + 1),
                       key=lambda t: t[0])
            cur[j] = best
        prev = cur
    cost, s, d, ins = prev[m]
    return ErrorCounts(s, d, ins, n)


def word_errors(reference: str, hypothesis: str,
                cfg: NormalizerConfig = NormalizerConfig()) -> ErrorCounts:
    ref = normalize(reference, cfg).split()
    hyp = normalize(hypothesis, cfg).split()
    return _levenshtein(ref, hyp)


def char_errors(reference: str, hypothesis: str,
                cfg: NormalizerConfig = NormalizerConfig()) -> ErrorCounts:
    ref = list(normalize(reference, cfg).replace(" ", ""))
    hyp = list(normalize(hypothesis, cfg).replace(" ", ""))
    return _levenshtein(ref, hyp)


def corpus_wer(pairs: Iterable[tuple[str, str]],
               cfg: NormalizerConfig = NormalizerConfig()) -> ErrorCounts:
    """Corpus WER: total errors over total reference words.

    Not the mean of per-utterance WERs, which over-weights short utterances
    and is the second most common way these numbers get quietly inflated.
    """
    total = ErrorCounts()
    for ref, hyp in pairs:
        total = total.merged(word_errors(ref, hyp, cfg))
    return total


def bootstrap_ci(pairs: Sequence[tuple[str, str]], n_resamples: int = 1000,
                 confidence: float = 0.95, seed: int = 0,
                 cfg: NormalizerConfig = NormalizerConfig()) -> tuple[float, float]:
    """Bootstrap confidence interval over utterances.

    A WER quoted without an interval is not comparable to another WER. On the
    small sets people actually run, the interval is frequently wider than the
    gap between two providers — which is the finding, not a footnote.
    """
    import random
    rng = random.Random(seed)
    per_utt = [word_errors(r, h, cfg) for r, h in pairs]
    if not per_utt:
        return (0.0, 0.0)
    rates = []
    n = len(per_utt)
    for _ in range(n_resamples):
        total = ErrorCounts()
        for _ in range(n):
            total = total.merged(per_utt[rng.randrange(n)])
        rates.append(total.rate)
    rates.sort()
    lo = rates[int((1 - confidence) / 2 * n_resamples)]
    hi = rates[min(n_resamples - 1, int((1 + confidence) / 2 * n_resamples))]
    return (lo, hi)
