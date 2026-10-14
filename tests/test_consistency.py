from __future__ import annotations

import numpy as np
import pytest

from selfaware_med.consistency import cosine, mean_pairwise


def test_cosine_identical_vectors():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine(a, b) == pytest.approx(0.0)


def test_cosine_opposite_vectors():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    assert cosine(a, b) == pytest.approx(-1.0)


def test_cosine_zero_norm_raises():
    a = np.array([0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError):
        cosine(a, b)


def test_mean_pairwise_insufficient_returns_none():
    assert mean_pairwise([]) is None
    assert mean_pairwise([np.array([1.0, 0.0], dtype=np.float32)]) is None


def test_mean_pairwise_two_identical():
    v = np.array([1.0, 0.0], dtype=np.float32)
    assert mean_pairwise([v, v]) == pytest.approx(1.0)


def test_mean_pairwise_two_orthogonal():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert mean_pairwise([a, b]) == pytest.approx(0.0)


def test_mean_pairwise_averages_all_pairs():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    c = np.array([-1.0, 0.0], dtype=np.float32)
    # pairs: (a,b)=0, (a,c)=-1, (b,c)=0 => mean=-1/3
    assert mean_pairwise([a, b, c]) == pytest.approx(-1.0 / 3.0)
