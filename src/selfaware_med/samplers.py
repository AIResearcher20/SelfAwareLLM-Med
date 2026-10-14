from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .consistency import mean_pairwise
from .data import MedQAItem, format_prompt, reference_answer
from .metrics import rouge_l
from .model import ModelClient


@dataclass
class SampleOutcome:
    qid: str
    method: str
    consistency: float | None
    rouge_l: float | None
    n_consistency_samples: int
    stop_reason: str
    valid_responses: int
    failed_responses: int
    generation_calls: int
    embedding_calls: int
    latency_s: float
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "qid": self.qid,
            "method": self.method,
            "consistency": self.consistency,
            "rouge_l": self.rouge_l,
            "n_consistency_samples": self.n_consistency_samples,
            "stop_reason": self.stop_reason,
            "valid_responses": self.valid_responses,
            "failed_responses": self.failed_responses,
            "generation_calls": self.generation_calls,
            "embedding_calls": self.embedding_calls,
            "latency_s": round(self.latency_s, 3),
            "error": self.error,
        }


class Sampler(Protocol):
    name: str

    def run(self, item: MedQAItem, client: ModelClient) -> SampleOutcome: ...


class _BaseSampler:
    name = "base"

    def __init__(
        self,
        consistency_temperature: float = 1.0,
        evaluation_temperature: float = 0.7,
        base_seed: int | None = None,
    ) -> None:
        self.consistency_temperature = consistency_temperature
        self.evaluation_temperature = evaluation_temperature
        self.base_seed = base_seed

    def _seed_for(self, item: MedQAItem, k: int) -> int | None:
        if self.base_seed is None:
            return None
        h = abs(hash((item.qid, self.name, k))) % (2**31)
        return (self.base_seed + h) % (2**31)

    def _generate_batch(
        self,
        prompt: str,
        client: ModelClient,
        item: MedQAItem,
        start_idx: int,
        count: int,
        temperature: float,
    ) -> tuple[list[str], list[np.ndarray], int, int, float]:
        texts: list[str] = []
        vectors: list[np.ndarray] = []
        gen_calls = 0
        failed = 0
        latency = 0.0

        for k in range(start_idx, start_idx + count):
            seed = self._seed_for(item, k)
            g = client.generate(prompt, temperature=temperature, seed=seed)
            gen_calls += 1
            latency += g.latency_s

            if not g.ok:
                failed += 1
                continue

            e = client.embed(g.text)
            latency += e.latency_s

            if not e.ok:
                failed += 1
                continue

            texts.append(g.text)
            vectors.append(e.vector)

        return texts, vectors, gen_calls, failed, latency

    def _evaluate(
        self, item: MedQAItem, client: ModelClient
    ) -> tuple[float | None, int, int, float]:
        prompt = format_prompt(item)
        seed = self._seed_for(item, 10_000)
        g = client.generate(
            prompt, temperature=self.evaluation_temperature, seed=seed
        )
        gen_calls = 1
        latency = g.latency_s

        if not g.ok:
            return None, gen_calls, 1, latency

        ref = reference_answer(item)
        return rouge_l(g.text, ref), gen_calls, 0, latency


class FixedSampler(_BaseSampler):
    name = "fixed"

    def __init__(
        self,
        n_samples: int = 5,
        consistency_temperature: float = 1.0,
        evaluation_temperature: float = 0.7,
        base_seed: int | None = None,
    ) -> None:
        super().__init__(
            consistency_temperature, evaluation_temperature, base_seed
        )
        self.n_samples = n_samples

    def run(self, item: MedQAItem, client: ModelClient) -> SampleOutcome:
        prompt = format_prompt(item)

        texts, vectors, gen_calls, failed, latency = self._generate_batch(
            prompt, client, item, 0, self.n_samples, self.consistency_temperature
        )

        consistency = mean_pairwise(vectors)

        eval_rouge, eval_gen_calls, eval_failed, eval_latency = self._evaluate(
            item, client
        )
        gen_calls += eval_gen_calls
        failed += eval_failed
        latency += eval_latency

        error = None
        if consistency is None:
            error = "insufficient_valid_responses"
        elif eval_rouge is None:
            error = "evaluation_generation_failed"

        return SampleOutcome(
            qid=item.qid,
            method=self.name,
            consistency=consistency,
            rouge_l=eval_rouge,
            n_consistency_samples=len(vectors),
            stop_reason="fixed",
            valid_responses=len(vectors),
            failed_responses=failed,
            generation_calls=gen_calls,
            embedding_calls=len(vectors),
            latency_s=latency,
            error=error,
        )


class AdaptiveSampler(_BaseSampler):
    name = "adaptive"

    def __init__(
        self,
        threshold: float = 0.85,
        min_samples: int = 3,
        max_samples: int = 10,
        consistency_temperature: float = 1.0,
        evaluation_temperature: float = 0.7,
        base_seed: int | None = None,
    ) -> None:
        super().__init__(
            consistency_temperature, evaluation_temperature, base_seed
        )
        self.threshold = threshold
        self.min_samples = min_samples
        self.max_samples = max_samples

    def run(self, item: MedQAItem, client: ModelClient) -> SampleOutcome:
        prompt = format_prompt(item)

        vectors: list[np.ndarray] = []
        gen_calls = 0
        failed = 0
        latency = 0.0
        consistency: float | None = None
        stop_reason = "max_samples"

        for k in range(self.max_samples):
            _, new_vecs, g, f, lat = self._generate_batch(
                prompt, client, item, k, 1, self.consistency_temperature
            )
            gen_calls += g
            failed += f
            latency += lat
            vectors.extend(new_vecs)

            if len(vectors) < self.min_samples:
                continue

            current = mean_pairwise(vectors)
            if current is None:
                continue
            consistency = current

            if current >= self.threshold:
                stop_reason = "threshold_reached"
                break

        if consistency is None and len(vectors) >= 2:
            consistency = mean_pairwise(vectors)

        eval_rouge, eval_gen_calls, eval_failed, eval_latency = self._evaluate(
            item, client
        )
        gen_calls += eval_gen_calls
        failed += eval_failed
        latency += eval_latency

        error = None
        if consistency is None:
            error = "insufficient_valid_responses"
        elif eval_rouge is None:
            error = "evaluation_generation_failed"

        return SampleOutcome(
            qid=item.qid,
            method=self.name,
            consistency=consistency,
            rouge_l=eval_rouge,
            n_consistency_samples=len(vectors),
            stop_reason=stop_reason,
            valid_responses=len(vectors),
            failed_responses=failed,
            generation_calls=gen_calls,
            embedding_calls=len(vectors),
            latency_s=latency,
            error=error,
        )


class AUASSampler(AdaptiveSampler):
    name = "auas"

    def __init__(
        self,
        uncertainty_threshold: float = 0.15,
        min_samples: int = 2,
        max_samples: int = 10,
        consistency_temperature: float = 1.0,
        evaluation_temperature: float = 0.7,
        base_seed: int | None = None,
    ) -> None:
        super().__init__(
            threshold=1.0 - uncertainty_threshold,
            min_samples=min_samples,
            max_samples=max_samples,
            consistency_temperature=consistency_temperature,
            evaluation_temperature=evaluation_temperature,
            base_seed=base_seed,
        )
        self.uncertainty_threshold = uncertainty_threshold
