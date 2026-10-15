from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .data import MedQAItem
from .model import ModelClient
from .samplers import (
    AdaptiveSampler,
    AUASSampler,
    FixedSampler,
    SampleOutcome,
    Sampler,
)


def build_samplers(config: dict) -> list[Sampler]:
    seed = config.get("seed")
    ct = config.get("consistency_temperature", 1.0)
    et = config.get("evaluation_temperature", 0.7)

    fixed = FixedSampler(
        n_samples=config.get("fixed", {}).get("n_samples", 5),
        consistency_temperature=ct,
        evaluation_temperature=et,
        base_seed=seed,
    )
    adaptive = AdaptiveSampler(
        threshold=config.get("adaptive", {}).get("threshold", 0.85),
        min_samples=config.get("adaptive", {}).get("min_samples", 4),
        max_samples=config.get("adaptive", {}).get("max_samples", 5),
        consistency_temperature=ct,
        evaluation_temperature=et,
        base_seed=seed,
    )
    auas = AUASSampler(
        uncertainty_threshold=config.get("auas", {}).get("uncertainty_threshold", 0.15),
        min_samples=config.get("auas", {}).get("min_samples", 4),
        max_samples=config.get("auas", {}).get("max_samples", 5),
        consistency_temperature=ct,
        evaluation_temperature=et,
        base_seed=seed,
    )
    return [fixed, adaptive, auas]


@dataclass
class RunRecord:
    qid: str
    method: str
    consistency: float | None
    rouge_l: float | None
    accuracy: float | None
    n_consistency_samples: int
    stop_reason: str
    valid_responses: int
    failed_responses: int
    generation_calls: int
    embedding_calls: int
    latency_s: float
    error: str | None
    evaluation_starts_with_letter: bool

    @classmethod
    def from_outcome(cls, o: SampleOutcome) -> RunRecord:
        d = o.as_dict()
        d["evaluation_starts_with_letter"] = o.evaluation_starts_with_letter
        return cls(**d)


def save_responses_csv(
    outcomes: list[SampleOutcome], out_dir: str | Path
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    by_method: dict[str, list[SampleOutcome]] = {}
    for o in outcomes:
        by_method.setdefault(o.method, []).append(o)

    for method, outs in by_method.items():
        path = out_dir / f"responses_{method}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "qid",
                    "evaluation_response",
                    "evaluation_starts_with_letter",
                    "consistency_responses",
                ]
            )
            for o in outs:
                writer.writerow(
                    [
                        o.qid,
                        o.evaluation_response or "",
                        int(o.evaluation_starts_with_letter),
                        " ||| ".join(o.consistency_responses),
                    ]
                )
        paths[method] = path

    return paths


def run_experiment(
    items: Iterable[MedQAItem],
    client: ModelClient,
    config: dict,
) -> tuple[list[RunRecord], list[SampleOutcome]]:
    samplers = build_samplers(config)
    records: list[RunRecord] = []
    raw_outcomes: list[SampleOutcome] = []

    items_list = list(items)
    total = len(items_list) * len(samplers)

    with tqdm(total=total, desc="runs") as bar:
        for item in items_list:
            for sampler in samplers:
                try:
                    outcome = sampler.run(item, client)
                except Exception as e:
                    outcome = SampleOutcome(
                        qid=item.qid,
                        method=sampler.name,
                        consistency=None,
                        rouge_l=None,
                        accuracy=None,
                        n_consistency_samples=0,
                        stop_reason="exception",
                        valid_responses=0,
                        failed_responses=0,
                        generation_calls=0,
                        embedding_calls=0,
                        latency_s=0.0,
                        error=f"{type(e).__name__}: {e}",
                    )
                records.append(RunRecord.from_outcome(outcome))
                raw_outcomes.append(outcome)
                bar.update(1)

    return records, raw_outcomes
