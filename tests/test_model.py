from __future__ import annotations

import numpy as np

from selfaware_med.model import FakeModelClient


def test_fake_generate_deterministic_per_seed():
    c = FakeModelClient()
    a = c.generate("prompt", temperature=1.0, seed=1)
    b = c.generate("prompt", temperature=1.0, seed=1)
    assert a.ok and b.ok
    assert a.text == b.text


def test_fake_generate_differs_across_seeds():
    c = FakeModelClient()
    a = c.generate("prompt", temperature=1.0, seed=1)
    b = c.generate("prompt", temperature=1.0, seed=2)
    assert a.text != b.text


def test_fake_embed_returns_unit_vector():
    c = FakeModelClient(dim=16)
    e = c.embed("hello")
    assert e.ok
    assert e.vector.shape == (16,)
    assert np.linalg.norm(e.vector) == 1.0 or abs(np.linalg.norm(e.vector) - 1.0) < 1e-5


def test_fake_embed_same_text_same_vector():
    c = FakeModelClient(dim=8)
    a = c.embed("hello")
    b = c.embed("hello")
    assert np.allclose(a.vector, b.vector)


def test_fake_embed_different_text_different_vector():
    c = FakeModelClient(dim=8)
    a = c.embed("hello")
    b = c.embed("world")
    assert not np.allclose(a.vector, b.vector)


def test_fake_forced_failure_generation():
    c = FakeModelClient(fail_every=2)
    c.generate("p", seed=1)
    r = c.generate("p", seed=2)
    assert not r.ok
    assert r.text is None
    assert r.error == "fake_forced_failure"


def test_fake_stats_accumulate():
    c = FakeModelClient(fail_every=3)
    for i in range(3):
        c.generate("p", seed=i)
        c.embed(f"e{i}")
    s = c.stats()
    assert s.generation_calls == 3
    assert s.generation_errors == 1
    assert s.embedding_calls == 3
