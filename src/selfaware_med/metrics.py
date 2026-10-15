from __future__ import annotations

from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(prediction: str | None, reference: str | None) -> float:
    if not prediction or not reference:
        return 0.0
    scores = _scorer.score(reference, prediction)
    return float(scores["rougeL"].fmeasure)


def mc_accuracy(prediction: str | None, item) -> float:
    if not prediction:
        return 0.0

    text = prediction.strip()
    if not text:
        return 0.0

    upper = text.upper()
    first = upper[0]
    if first in item.options:
        return 1.0 if first == item.answer_idx else 0.0

    lowered = text.lower()
    for k, v in item.options.items():
        if v.lower() in lowered:
            return 1.0 if k == item.answer_idx else 0.0

    return 0.0


def starts_with_option_letter(prediction: str | None, item) -> bool:
    if not prediction:
        return False
    text = prediction.strip()
    if not text:
        return False
    return text.upper()[0] in item.options
