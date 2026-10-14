from __future__ import annotations

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        raise ValueError("zero-norm vector in cosine")
    return float(np.dot(a, b) / (na * nb))


def mean_pairwise(vectors: list[np.ndarray]) -> float | None:
    n = len(vectors)
    if n < 2:
        return None

    sims: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine(vectors[i], vectors[j]))

    if not sims:
        return None
    return float(np.mean(sims))
