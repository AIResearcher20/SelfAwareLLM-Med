# Data

## Source

MedQA, 4-option English questions.

Original dataset: Jin et al., 2020.

## Files

- sample_1000.jsonl: 1000 questions from the test split.
- SHA-256: a0a286159a1f372057a53b7d790381e46dbd52a1620d0adbe29c9530dca6accb

## Format

Each line:

{
  "qid": "...",
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer_idx": "A",
  "answer_text": "..."
}
