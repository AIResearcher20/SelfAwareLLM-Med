from __future__ import annotations

import pytest

from selfaware_med.metrics import rouge_l


def test_rouge_l_identical_text():
    assert rouge_l("the patient has pneumonia", "the patient has pneumonia") == pytest.approx(1.0)


def test_rouge_l_completely_different():
    score = rouge_l("pneumonia", "myocardial infarction")
    assert score < 0.3


def test_rouge_l_partial_overlap():
    score = rouge_l("the patient has pneumonia", "patient has pneumonia today")
    assert 0.5 < score < 1.0


def test_rouge_l_empty_prediction():
    assert rouge_l("", "pneumonia") == pytest.approx(0.0)


def test_rouge_l_none_prediction():
    assert rouge_l(None, "pneumonia") == pytest.approx(0.0)
