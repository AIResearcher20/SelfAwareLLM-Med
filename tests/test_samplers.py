from __future__ import annotations

from selfaware_med.data import MedQAItem
from selfaware_med.model import FakeModelClient
from selfaware_med.samplers import AdaptiveSampler, AUASSampler, FixedSampler


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
    assert out.generation_calls == 6
    assert out.embedding_calls == 5
    assert out.consistency is not None
    assert out.error is None


def test_fixed_sampler_with_failures_marks_error_but_keeps_running():
    client = FakeModelClient(fail_every=2)
    sampler = FixedSampler(n_samples=5, base_seed=7)
    out = sampler.run(_item(), client)

    assert out.failed_responses > 0
    assert out.error is not None


def test_adaptive_sampler_stops_at_threshold_when_consistent():
    client = FakeModelClient()
    sampler = AdaptiveSampler(
        threshold=0.0,
        min_samples=3,
        max_samples=10,
        base_seed=7,
    )
    out = sampler.run(_item(), client)

    assert out.method == "adaptive"
    assert out.stop_reason == "threshold_reached"
    assert 3 <= out.n_consistency_samples <= 10


def test_adaptive_sampler_runs_to_max_when_threshold_impossible():
    client = FakeModelClient()
    sampler = AdaptiveSampler(
        threshold=1.1,
        min_samples=3,
        max_samples=10,
        base_seed=7,
    )
    out = sampler.run(_item(), client)

    assert out.stop_reason == "max_samples"
    assert out.n_consistency_samples == 10


def test_adaptive_and_auas_use_equivalent_thresholds():
    a = AdaptiveSampler(threshold=0.85, min_samples=2, max_samples=5, base_seed=1)
    u = AUASSampler(
        uncertainty_threshold=0.15, min_samples=2, max_samples=5, base_seed=1
    )
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
    assert out.generation_calls == 4
    assert out.embedding_calls == 3


def test_outcome_includes_accuracy_and_responses():
    client = FakeModelClient()
    sampler = FixedSampler(n_samples=3, base_seed=5)
    out = sampler.run(_item(), client)

    assert out.accuracy is not None
    assert out.accuracy in {0.0, 1.0}
    assert out.evaluation_response is not None
    assert len(out.consistency_responses) == 3


def test_adaptive_defaults_to_new_min_max():
    a = AdaptiveSampler()
    u = AUASSampler()
    assert a.min_samples == 4
    assert a.max_samples == 5
    assert u.min_samples == 4
    assert u.max_samples == 5
