from __future__ import annotations

from rouge_score import rouge_scorer


_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(prediction: str, reference: str) -> float:
    if prediction is None or reference is None:
        return 0.0
    scores = _scorer.score(reference, prediction)
    return float(scores["rougeL"].fmeasure)
