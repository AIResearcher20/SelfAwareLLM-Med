from __future__ import annotations

from selfaware_med.data import MedQAItem
from selfaware_med.model import FakeModelClient
from selfaware_med.samplers import AUASSampler, AdaptiveSampler, FixedSampler


def _item() -> MedQAItem:
    return MedQAItem(
        qid="q1",
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


def test_fixed_sampler_runs_n_consistency_plus_one_evaluation():
    client = FakeModelClient()
    sampler = FixedSampler(n_samples=5, base_seed=7)
    out = sampler.run(_item(), client)

    assert out.method == "fixed"
    assert out.stop_reason == "fixed"
    assert out.n_consistency_samples == 5
    # 5 consistency + 1 evaluation
    assert out.generation_calls == 6
    assert out.embedding_calls == 5
    assert out.consistency is not None
    assert out.error is None


def test_fixed_sampler_with_failures_marks_error_but_keeps_running():
    client = FakeModelClient(fail_every=2)
    sampler = FixedSampler(n_samples=5, base_seed=7)
    out = sampler.run(_item(), client)

    assert out.failed_responses > 0
    # Should still have some valid samples or an explicit error, never a silent zero
    assert out.error in {"insufficient_valid_responses", None}


def test_adaptive_sampler_stops_at_threshold_when_consistent():
    client = FakeModelClient()
    sampler = AdaptiveSampler(
        threshold=0.0,  # any consistency passes immediately at min_samples
        min_samples=3,
        max_samples=10,
        base_seed=7,
    )
    out = sampler.run(_item(), client)

    assert out.method == "adaptive"
    assert out.stop_reason == "threshold_reached"
    assert out.n_consistency_samples >= 3
    assert out.n_consistency_samples <= 10


def test_adaptive_sampler_runs_to_max_when_threshold_impossible():
    client = FakeModelClient()
    sampler = AdaptiveSampler(
        threshold=1.1,  # impossible
        min_samples=3,
        max_samples=10,
        base_seed=7,
    )
    out = sampler.run(_item(), client)

    assert out.stop_reason == "max_samples"
    assert out.n_consistency_samples == 10


def test_adaptive_and_auas_use_equivalent_thresholds():
    a = AdaptiveSampler(threshold=0.85, min_samples=2, max_samples=5, base_seed=1)
    u = AUASSampler(uncertainty_threshold=0.15, min_samples=2, max_samples=5, base_seed=1)
    assert abs(a.threshold - (1.0 - u.uncertainty_threshold)) < 1e-9
    assert a.max_samples == u.max_samples


def test_auas_sampler_returns_outcome():
    client = FakeModelClient()
    sampler = AUASSampler(
        uncertainty_threshold=0.15, min_samples=2, max_samples=10, base_seed=3
    )
    out = sampler.run(_item(), client)
    assert out.method == "auas"
    assert out.stop_reason in {"threshold_reached", "max_samples"}
    assert out.n_consistency_samples >= 2


def test_outcome_generation_calls_account_for_evaluation():
    client = FakeModelClient()
    sampler = FixedSampler(n_samples=3, base_seed=5)
    out = sampler.run(_item(), client)
    # 3 consistency + 1 evaluation response
    assert out.generation_calls == 4
    assert out.embedding_calls == 3
