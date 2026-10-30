"""One HTTP client for every decision backend that speaks Jev's wire protocol.

Laya (local `laya-serve`) and Jev (`api.typesafe.ai`) accept the same request,
`POST /v1/systemone` with `{model, state, questions}`, so a backend is just a
base URL, a key and a model name.

Confidence is recomputed here from `probabilities` with Jev's formula,
(n·p_max − 1)/(n − 1), because Laya's own `confidence` field uses a different one
(1 − normalised entropy) and the two must be comparable.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import httpx

LAYA_URL = "http://127.0.0.1:8000"
JEV_URL = "https://api.typesafe.ai"


@dataclass(frozen=True)
class Backend:
    name: str
    base_url: str
    model: str
    api_key: str = "local"  # laya-serve accepts any bearer when LAYA_API_KEY is unset
    usd_per_mtok_in: float = 0.0


def laya(model: str = "english", name: str = "laya-base") -> Backend:
    return Backend(name, os.environ.get("LAYA_BASE_URL", LAYA_URL), model)


def jev(model: str = "jev-1.14.0") -> Backend | None:
    """None when no key is set, so callers can skip Jev."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        return None
    return Backend("jev", os.environ.get("TYPESAFE_BASE_URL", JEV_URL), model, key, 0.042)


@dataclass
class Decision:
    backend: str
    model: str
    answers: dict
    latency_s: float
    input_tokens: int
    raw: dict = field(repr=False)

    def confidence(self, question: str) -> float | None:
        """Jev-formula confidence from probabilities; noul has none (only P(true))."""
        probs = self.answers[question].get("probabilities")
        if not probs:
            return None
        n = len(probs)
        return (n * max(probs.values()) - 1) / (n - 1) if n > 1 else 1.0


class DecisionClient:
    def __init__(self, backend: Backend, timeout: float = 30.0):
        self.backend = backend
        self._http = httpx.Client(
            base_url=backend.base_url,
            headers={"Authorization": f"Bearer {backend.api_key}"},
            timeout=timeout,
        )

    def decide(self, state, questions: dict) -> Decision:
        body = {"model": self.backend.model, "state": state, "questions": questions}
        start = time.perf_counter()
        resp = self._http.post("/v1/systemone", json=body)
        latency = time.perf_counter() - start
        resp.raise_for_status()
        data = resp.json()
        return Decision(
            backend=self.backend.name,
            model=data.get("model", self.backend.model),
            answers=data["answers"],
            latency_s=latency,
            input_tokens=data.get("usage", {}).get("input_tokens", 0),
            raw=data,
        )

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
