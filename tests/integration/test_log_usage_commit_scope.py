"""Tests for usage logging via the log writer abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest
from any_llm.types.completion import CompletionUsage
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes.chat import log_usage
from gateway.core.usage import GatewayUsage
from gateway.models.entities import ModelPricing, UsageLog


@dataclass
class StubLogWriter:
    logs: list[UsageLog]

    def __init__(self) -> None:
        self.logs = []

    async def put(self, log: UsageLog) -> None:
        self.logs.append(log)

    async def start(self) -> None:  # pragma: no cover - not used
        return None

    async def stop(self) -> None:  # pragma: no cover - not used
        return None


@pytest.mark.asyncio
async def test_log_usage_records_usage_data(async_db: AsyncSession) -> None:
    pricing = ModelPricing(model_key="openai:gpt-4o", input_price_per_million=2.0, output_price_per_million=4.0)
    async_db.add(pricing)
    await async_db.commit()

    usage = CompletionUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    writer = StubLogWriter()

    await log_usage(
        db=async_db,
        log_writer=writer,
        api_key_id=None,
        model="gpt-4o",
        provider="openai",
        endpoint="/v1/chat/completions",
        usage_override=usage,
    )

    assert len(writer.logs) == 1
    log = writer.logs[0]
    assert log.prompt_tokens == 1000
    assert log.completion_tokens == 500
    assert log.cost == (Decimal(1000) * Decimal("2") + Decimal(500) * Decimal("4")) / Decimal(1_000_000)
    assert log.status == "success"


@pytest.mark.asyncio
async def test_log_usage_records_cache_tokens(async_db: AsyncSession) -> None:
    usage = GatewayUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        cache_read_tokens=200,
        cache_write_tokens=50,
    )
    writer = StubLogWriter()

    await log_usage(
        db=async_db,
        log_writer=writer,
        api_key_id=None,
        model="gpt-4o",
        provider="openai",
        endpoint="/v1/chat/completions",
        usage_override=usage,
    )

    assert len(writer.logs) == 1
    log = writer.logs[0]
    assert log.cache_read_tokens == 200
    assert log.cache_write_tokens == 50


@pytest.mark.asyncio
async def test_log_usage_records_the_cached_token_convention(async_db: AsyncSession) -> None:
    """Settlement stores which convention the provider reported under, not just the counts.

    The two shapes are indistinguishable from the numbers alone, so a row that
    is repriced later has to read the convention rather than infer it
    (mozilla-ai/otari#690).
    """
    writer = StubLogWriter()
    await log_usage(
        db=async_db,
        log_writer=writer,
        api_key_id=None,
        model="claude-sonnet-4-5",
        provider="anthropic",
        endpoint="/v1/messages",
        usage_override=GatewayUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            cache_read_tokens=200,
            cache_tokens_in_prompt=False,
        ),
    )
    # A plain CompletionUsage is OpenAI-shaped, where the cached slice is inside
    # the prompt, so the same call records the other convention.
    await log_usage(
        db=async_db,
        log_writer=writer,
        api_key_id=None,
        model="gpt-4o",
        provider="openai",
        endpoint="/v1/chat/completions",
        usage_override=CompletionUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
    )

    assert [log.cache_tokens_in_prompt for log in writer.logs] == [False, True]


@pytest.mark.asyncio
async def test_log_usage_records_error(async_db: AsyncSession) -> None:
    writer = StubLogWriter()

    await log_usage(
        db=async_db,
        log_writer=writer,
        api_key_id=None,
        model="gpt-4o",
        provider="openai",
        endpoint="/v1/chat/completions",
        error="Provider timeout",
    )

    assert len(writer.logs) == 1
    log = writer.logs[0]
    assert log.status == "error"
    assert log.error_message == "Provider timeout"
