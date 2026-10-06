"""A request prices each model it bills once: admission, a top-up and settlement share it."""

import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from gateway.services import pricing_service
from gateway.services.pricing_service import RequestPrices


@pytest.mark.asyncio
async def test_a_model_is_resolved_once_per_request(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved: Any = SimpleNamespace(pricing=object())
    lookup = AsyncMock(return_value=resolved)
    monkeypatch.setattr(pricing_service, "resolve_model_pricing", lookup)
    organization = uuid.uuid4()
    prices = RequestPrices(organization)
    db: Any = object()

    assert await prices.resolve(db, "openai", "gpt-5") is resolved
    assert await prices.resolve(db, "openai", "gpt-5") is resolved

    lookup.assert_awaited_once_with(db, "openai", "gpt-5", organization_id=organization)


@pytest.mark.asyncio
async def test_each_model_is_resolved_on_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    by_model: dict[tuple[str, str], Any] = {
        ("openai", "gpt-5"): SimpleNamespace(pricing=object()),
        ("together", "llama"): SimpleNamespace(pricing=object()),
    }
    lookup = AsyncMock(side_effect=lambda _db, provider, model, **_: by_model[(provider, model)])
    monkeypatch.setattr(pricing_service, "resolve_model_pricing", lookup)
    prices = RequestPrices(None)

    assert await prices.resolve(None, "openai", "gpt-5") is by_model[("openai", "gpt-5")]  # type: ignore[arg-type]
    assert await prices.resolve(None, "together", "llama") is by_model[("together", "llama")]  # type: ignore[arg-type]
    assert lookup.await_count == 2


@pytest.mark.asyncio
async def test_an_unpriced_model_is_remembered_as_unpriced(monkeypatch: pytest.MonkeyPatch) -> None:
    lookup = AsyncMock(return_value=None)
    monkeypatch.setattr(pricing_service, "resolve_model_pricing", lookup)
    prices = RequestPrices(None)

    assert await prices.resolve(None, "local", "free") is None  # type: ignore[arg-type]
    assert await prices.resolve(None, "local", "free") is None  # type: ignore[arg-type]
    assert lookup.await_count == 1
