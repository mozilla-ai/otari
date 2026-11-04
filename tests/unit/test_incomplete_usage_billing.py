"""An abandoned stream's estimated row is billed on tokens alone, never a per-request price."""

import uuid
from decimal import Decimal
from typing import Any

import pytest
from any_llm.types.completion import CompletionUsage

from gateway.api.routes import _pipeline as pipeline
from gateway.models.pricing import ModelPricing
from gateway.services.pricing_service import ResolvedPricing


class _Prices:
    def __init__(self, pricing: ModelPricing) -> None:
        self._resolved = ResolvedPricing(pricing, "defaults")

    async def resolve(self, *_: Any, **__: Any) -> ResolvedPricing:
        return self._resolved


class _Writer:
    def __init__(self) -> None:
        self.rows: list[Any] = []

    async def put(self, row: Any) -> None:
        self.rows.append(row)


async def _record(pricing: ModelPricing, *, incomplete: bool) -> tuple[Any, Any]:
    writer = _Writer()
    logged = await pipeline.record_usage(
        db=None,  # type: ignore[arg-type]
        log_writer=writer,  # type: ignore[arg-type]
        api_key_id=None,
        model="m",
        provider="p",
        endpoint="/v1/chat/completions",
        user_id=None,
        usage_override=CompletionUsage(prompt_tokens=1_000_000, completion_tokens=0, total_tokens=1_000_000),
        error=None,
        incomplete=incomplete,
        workspace_id=uuid.uuid4(),
        prices=_Prices(pricing),  # type: ignore[arg-type]
    )
    return logged, writer.rows[0]


@pytest.mark.asyncio
async def test_an_incomplete_row_is_not_charged_a_per_request_price() -> None:
    per_request = ModelPricing(
        model_key="p:m", input_price_per_million=Decimal(5000), output_price_per_million=Decimal(0), unit="request"
    )

    done, _ = await _record(per_request, incomplete=False)
    abandoned, row = await _record(per_request, incomplete=True)

    assert done.cost
    assert abandoned.cost is None
    assert row.status == "incomplete"


@pytest.mark.asyncio
async def test_an_incomplete_row_is_still_billed_its_estimated_tokens() -> None:
    per_token = ModelPricing(model_key="p:m", input_price_per_million=Decimal(3), output_price_per_million=Decimal(6))

    abandoned, row = await _record(per_token, incomplete=True)

    assert abandoned.cost == Decimal(3)
    assert row.prompt_tokens == 1_000_000


@pytest.mark.asyncio
async def test_an_incomplete_row_counts_as_a_failed_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[bool] = []
    monkeypatch.setattr(pipeline, "record_attempt", lambda *_, failed, **__: attempts.append(failed))
    per_token = ModelPricing(model_key="p:m", input_price_per_million=Decimal(3), output_price_per_million=Decimal(6))

    await _record(per_token, incomplete=False)
    await _record(per_token, incomplete=True)

    assert attempts == [False, True]
