from __future__ import annotations

import json
from pathlib import Path

import pytest

from selfaware_med.data import (
    MedQAItem,
    filter_by_ids,
    format_prompt,
    load_jsonl,
    reference_answer,
    write_jsonl,
)


def _make_item(qid: str = "q1") -> MedQAItem:
    return MedQAItem(
        qid=qid,
        question="What is the first-line treatment for anaphylaxis?",
        options={
            "A": "Epinephrine",
            "B": "Diphenhydramine",
            "C": "Corticosteroids",
            "D": "Albuterol",
        },
        answer_idx="A",
        answer_text="Epinephrine",
    )


def test_write_and_load_roundtrip(tmp_path: Path):
    items = [_make_item("q1"), _make_item("q2")]
    path = tmp_path / "sample.jsonl"
    write_jsonl(items, path)
    loaded = load_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0].qid == "q1"
    assert loaded[0].answer_text == "Epinephrine"
    assert loaded[1].options["B"] == "Diphenhydramine"


def test_load_jsonl_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "sample.jsonl"
    path.write_text(
        json.dumps(
            {
                "qid": "q1",
                "question": "q",
                "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "answer_idx": "A",
                "answer_text": "a",
            }
        )
        + "\n\n",
        encoding="utf-8",
    )
    loaded = load_jsonl(path)
    assert len(loaded) == 1


def test_filter_by_ids_returns_requested_order(tmp_path: Path):
    items = [_make_item("q1"), _make_item("q2"), _make_item("q3")]
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text("q3\nq1\n", encoding="utf-8")
    filtered = filter_by_ids(items, ids_path)
    assert [it.qid for it in filtered] == ["q3", "q1"]


def test_filter_by_ids_missing_raises(tmp_path: Path):
    items = [_make_item("q1")]
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text("q1\nq999\n", encoding="utf-8")
    with pytest.raises(KeyError):
        filter_by_ids(items, ids_path)


def test_format_prompt_contains_question_and_options():
    item = _make_item()
    prompt = format_prompt(item)
    assert item.question in prompt
    assert "A. Epinephrine" in prompt
    assert "B. Diphenhydramine" in prompt
    assert prompt.strip().endswith("Answer:")


def test_reference_answer_returns_full_text():
    item = _make_item()
    assert reference_answer(item) == "Epinephrine"
