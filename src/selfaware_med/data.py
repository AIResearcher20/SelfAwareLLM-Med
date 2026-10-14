from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MedQAItem:
    qid: str
    question: str
    options: dict[str, str]
    answer_idx: str
    answer_text: str


def _item_from_hf(row: dict, idx: int) -> MedQAItem:
    options = row.get("options") or {}
    if isinstance(options, str):
        options = json.loads(options)

    answer_idx = row.get("answer_idx") or row.get("answer") or ""
    answer_text = options.get(answer_idx, "")

    return MedQAItem(
        qid=row.get("id") or f"hf_{idx}",
        question=row["question"],
        options={k: v for k, v in options.items()},
        answer_idx=answer_idx,
        answer_text=answer_text,
    )


def load_hf(
    dataset_name: str = "GBaker/MedQA-USMLE-4-options",
    split: str = "test",
    limit: int | None = None,
    seed: int = 13,
) -> list[MedQAItem]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    items = [_item_from_hf(dict(row), i) for i, row in enumerate(ds)]

    if limit is not None and limit < len(items):
        rng = random.Random(seed)
        idxs = sorted(rng.sample(range(len(items)), limit))
        items = [items[i] for i in idxs]

    return items


def load_jsonl(path: str | Path) -> list[MedQAItem]:
    path = Path(path)
    items: list[MedQAItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(
                MedQAItem(
                    qid=row["qid"],
                    question=row["question"],
                    options=row["options"],
                    answer_idx=row["answer_idx"],
                    answer_text=row["answer_text"],
                )
            )
    return items


def filter_by_ids(items: list[MedQAItem], ids_path: str | Path) -> list[MedQAItem]:
    path = Path(ids_path)
    wanted = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {it.qid: it for it in items}
    missing = [qid for qid in wanted if qid not in by_id]
    if missing:
        raise KeyError(f"{len(missing)} sample_ids not found in dataset, first: {missing[0]}")
    return [by_id[qid] for qid in wanted]


def write_jsonl(items: list[MedQAItem], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(
                json.dumps(
                    {
                        "qid": it.qid,
                        "question": it.question,
                        "options": it.options,
                        "answer_idx": it.answer_idx,
                        "answer_text": it.answer_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def format_prompt(item: MedQAItem) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(item.options.items()))
    return (
        "You are answering a USMLE-style medical question. "
        "Give a short final answer.\n\n"
        f"Question: {item.question}\n\n"
        f"Options:\n{opts}\n\n"
        "Answer:"
    )


def reference_answer(item: MedQAItem) -> str:
    return item.answer_text
