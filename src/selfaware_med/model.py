from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import requests


@dataclass
class GenerationResult:
    text: str | None
    error: str | None
    latency_s: float
    model: str
    temperature: float
    seed: int | None

    @property
    def ok(self) -> bool:
        return self.error is None and self.text is not None and self.text.strip() != ""


@dataclass
class EmbeddingResult:
    vector: np.ndarray | None
    error: str | None
    latency_s: float
    model: str

    @property
    def ok(self) -> bool:
        return self.error is None and self.vector is not None


@dataclass
class CallStats:
    generation_calls: int = 0
    embedding_calls: int = 0
    generation_errors: int = 0
    embedding_errors: int = 0
    total_latency_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "generation_calls": self.generation_calls,
            "embedding_calls": self.embedding_calls,
            "generation_errors": self.generation_errors,
            "embedding_errors": self.embedding_errors,
            "total_latency_s": round(self.total_latency_s, 3),
        }


class ModelClient(Protocol):
    name: str

    def generate(
        self, prompt: str, *, temperature: float = 1.0, seed: int | None = None
    ) -> GenerationResult: ...

    def embed(self, text: str) -> EmbeddingResult: ...

    def stats(self) -> CallStats: ...


class EmbeddingCache:
    def __init__(self) -> None:
        self._store: dict[tuple[str, str], np.ndarray] = {}

    @staticmethod
    def _key(model: str, text: str) -> tuple[str, str]:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return (model, h)

    def get(self, model: str, text: str) -> np.ndarray | None:
        return self._store.get(self._key(model, text))

    def put(self, model: str, text: str, vector: np.ndarray) -> None:
        self._store[self._key(model, text)] = vector


class OllamaModel:
    def __init__(
        self,
        gen_model: str = "llama3.2:3b",
        embed_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 120.0,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self.name = f"ollama:{gen_model}+{embed_model}"
        self.gen_model = gen_model
        self.embed_model = embed_model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.cache = cache
        self._stats = CallStats()
        self._embed_endpoint = self._detect_embed_endpoint()

    def _detect_embed_endpoint(self) -> str:
        try:
            r = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.embed_model, "input": "ping"},
                timeout=10,
            )
            if r.status_code == 200:
                return "/api/embed"
        except requests.RequestException:
            pass
        return "/api/embeddings"

    def generate(
        self, prompt: str, *, temperature: float = 1.0, seed: int | None = None
    ) -> GenerationResult:
        t0 = time.time()
        options: dict = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed

        payload = {
            "model": self.gen_model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        try:
            r = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=self.timeout_s
            )
        except requests.RequestException as e:
            self._stats.generation_calls += 1
            self._stats.generation_errors += 1
            self._stats.total_latency_s += time.time() - t0
            return GenerationResult(
                text=None,
                error=f"request_failed: {e}",
                latency_s=time.time() - t0,
                model=self.gen_model,
                temperature=temperature,
                seed=seed,
            )

        latency = time.time() - t0
        self._stats.generation_calls += 1
        self._stats.total_latency_s += latency

        if r.status_code != 200:
            self._stats.generation_errors += 1
            return GenerationResult(
                text=None,
                error=f"http_{r.status_code}: {r.text[:200]}",
                latency_s=latency,
                model=self.gen_model,
                temperature=temperature,
                seed=seed,
            )

        try:
            text = r.json().get("response")
        except json.JSONDecodeError as e:
            self._stats.generation_errors += 1
            return GenerationResult(
                text=None,
                error=f"bad_json: {e}",
                latency_s=latency,
                model=self.gen_model,
                temperature=temperature,
                seed=seed,
            )

        if text is None or text.strip() == "":
            self._stats.generation_errors += 1
            return GenerationResult(
                text=None,
                error="empty_response",
                latency_s=latency,
                model=self.gen_model,
                temperature=temperature,
                seed=seed,
            )

        return GenerationResult(
            text=text,
            error=None,
            latency_s=latency,
            model=self.gen_model,
            temperature=temperature,
            seed=seed,
        )

    def embed(self, text: str) -> EmbeddingResult:
        t0 = time.time()

        if self.cache is not None:
            cached = self.cache.get(self.embed_model, text)
            if cached is not None:
                return EmbeddingResult(
                    vector=cached, error=None, latency_s=0.0, model=self.embed_model
                )

        if self._embed_endpoint == "/api/embed":
            payload = {"model": self.embed_model, "input": text}
            key = "embeddings"
        else:
            payload = {"model": self.embed_model, "prompt": text}
            key = "embedding"

        try:
            r = requests.post(
                f"{self.base_url}{self._embed_endpoint}",
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.RequestException as e:
            self._stats.embedding_calls += 1
            self._stats.embedding_errors += 1
            self._stats.total_latency_s += time.time() - t0
            return EmbeddingResult(
                vector=None,
                error=f"request_failed: {e}",
                latency_s=time.time() - t0,
                model=self.embed_model,
            )

        latency = time.time() - t0
        self._stats.embedding_calls += 1
        self._stats.total_latency_s += latency

        if r.status_code != 200:
            self._stats.embedding_errors += 1
            return EmbeddingResult(
                vector=None,
                error=f"http_{r.status_code}: {r.text[:200]}",
                latency_s=latency,
                model=self.embed_model,
            )

        try:
            data = r.json()
            vec = data[key]
            if isinstance(vec, list) and vec and isinstance(vec[0], list):
                vec = vec[0]
            vector = np.asarray(vec, dtype=np.float32)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            self._stats.embedding_errors += 1
            return EmbeddingResult(
                vector=None,
                error=f"bad_embedding_payload: {e}",
                latency_s=latency,
                model=self.embed_model,
            )

        if self.cache is not None:
            self.cache.put(self.embed_model, text, vector)

        return EmbeddingResult(
            vector=vector, error=None, latency_s=latency, model=self.embed_model
        )

    def stats(self) -> CallStats:
        return self._stats


class FakeModelClient:
    def __init__(self, dim: int = 32, fail_every: int | None = None) -> None:
        self.name = "fake"
        self.dim = dim
        self.fail_every = fail_every
        self._stats = CallStats()
        self._gen_counter = 0
        self._emb_counter = 0

    def _hash_to_vec(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        v = rng.standard_normal(self.dim).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def generate(
        self, prompt: str, *, temperature: float = 1.0, seed: int | None = None
    ) -> GenerationResult:
        self._gen_counter += 1
        self._stats.generation_calls += 1

        if self.fail_every and self._gen_counter % self.fail_every == 0:
            self._stats.generation_errors += 1
            return GenerationResult(
                text=None,
                error="fake_forced_failure",
                latency_s=0.0,
                model="fake",
                temperature=temperature,
                seed=seed,
            )

        key = f"{prompt}|{seed}"
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        return GenerationResult(
            text=f"fake_answer::{h}",
            error=None,
            latency_s=0.0,
            model="fake",
            temperature=temperature,
            seed=seed,
        )

    def embed(self, text: str) -> EmbeddingResult:
        self._emb_counter += 1
        self._stats.embedding_calls += 1
        return EmbeddingResult(
            vector=self._hash_to_vec(text), error=None, latency_s=0.0, model="fake"
        )

    def stats(self) -> CallStats:
        return self._stats
