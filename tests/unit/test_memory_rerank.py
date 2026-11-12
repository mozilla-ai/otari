"""The memory rerank call: one score per document in the order given, and what it is metered as.

The provider call is captured at the any-llm boundary and never made.
"""

import asyncio
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from any_llm.types.rerank import RerankResponse, RerankResult, RerankUsage

from gateway.core.config import GatewayConfig
from gateway.services.memory._models import RERANK_ENDPOINT, MemoryModelError, MemoryModels

MODEL = "cohere:rerank-v3.5"


def _models(**config: Any) -> MemoryModels:
    models = MemoryModels(GatewayConfig(memory_rerank_model=MODEL, **config))

    async def resolve(model: str) -> SimpleNamespace:
        return SimpleNamespace(model="rerank-v3.5", provider="cohere", kwargs={"api_key": "k"})

    models._resolve = resolve  # type: ignore[method-assign]
    return models


@pytest.mark.asyncio
async def test_scores_come_back_in_the_order_the_documents_were_given() -> None:
    sent: dict[str, Any] = {}

    async def fake(model: str, query: str, documents: list[str], **kwargs: Any) -> RerankResponse:
        sent.update(model=model, query=query, documents=documents, **kwargs)
        # Providers answer best first, not in the order they were sent.
        results = [RerankResult(index=2, relevance_score=0.9), RerankResult(index=0, relevance_score=0.2)]
        return RerankResponse(results=results, usage=RerankUsage(total_tokens=41))

    with patch("gateway.services.memory._models.arerank", new=fake):
        scores, use = await _models().rerank("which branch?", ["a", "b", "c"])

    assert scores == [0.2, 0.0, 0.9], "a document the provider left out scores nothing"
    assert sent == {
        "model": "rerank-v3.5",
        "query": "which branch?",
        "documents": ["a", "b", "c"],
        "provider": "cohere",
        "api_key": "k",
    }
    assert (use.endpoint, use.model, use.provider, use.succeeded) == (RERANK_ENDPOINT, MODEL, "cohere", True)
    assert (use.prompt_tokens, use.completion_tokens) == (41, None)


@pytest.mark.asyncio
async def test_a_failed_call_or_an_answer_about_documents_never_sent_is_an_error() -> None:
    async def failing(*args: Any, **kwargs: Any) -> RerankResponse:
        raise RuntimeError("provider down")

    async def stray(*args: Any, **kwargs: Any) -> RerankResponse:
        return RerankResponse(results=[RerankResult(index=5, relevance_score=0.9)])

    with patch("gateway.services.memory._models.arerank", new=failing), pytest.raises(MemoryModelError) as failed:
        await _models().rerank("q", ["a"])
    assert failed.value.use is not None and not failed.value.use.succeeded

    with patch("gateway.services.memory._models.arerank", new=stray), pytest.raises(MemoryModelError) as strayed:
        await _models().rerank("q", ["a"])
    assert strayed.value.use is not None and strayed.value.use.succeeded

    with pytest.raises(MemoryModelError):
        await MemoryModels(GatewayConfig()).rerank("q", ["a"])
    assert not MemoryModels(GatewayConfig()).can_rerank and _models().can_rerank


@pytest.mark.asyncio
async def test_a_check_slower_than_its_time_is_cut_short_and_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def slow(*args: Any, **kwargs: Any) -> RerankResponse:
        await asyncio.sleep(5)
        return RerankResponse(results=[])

    monkeypatch.setattr("gateway.services.memory._models.RERANK_TIMEOUT_S", 0.05)
    started = time.monotonic()
    with patch("gateway.services.memory._models.arerank", new=slow), pytest.raises(MemoryModelError) as cut:
        await _models().rerank("q", ["a"])

    assert time.monotonic() - started < 1
    assert cut.value.use is not None and not cut.value.use.succeeded and cut.value.use.endpoint == RERANK_ENDPOINT
