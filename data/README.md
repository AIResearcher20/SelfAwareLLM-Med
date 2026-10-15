# Data

## Source

MedQA, 4-option English questions from the USMLE.

Original dataset: Jin et al., 2020.

Mirror used: awinml/medqa on Hugging Face, config "questions", split "test".

## Files

- sample_1000.jsonl: 1000 unique questions from the test split.
- All question_ids are unique.
- Splits: test only.
- meta_info covers both step1 and step2&3.
- SHA-256: a0a286159a1f372057a53b7d790381e46dbd52a1620d0adbe29c9530dca6accb

## Format

Each line:

{
  "question_id": "medqa_test_0000",
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "...",
  "answer_idx": "B",
  "meta_info": "step1",
  "source": "awinml/medqa",
  "split": "test",
  "source_row": 0
}

The loader accepts both "question_id" and "qid", and both "answer" and "answer_text".

## Regenerating

from selfaware_med.data import load_hf, write_jsonl

items = load_hf(
    dataset_name="awinml/medqa",
    config="questions",
    split="test",
    limit=1000,
    seed=13,
)
write_jsonl(items, "data/sample_1000.jsonl")
