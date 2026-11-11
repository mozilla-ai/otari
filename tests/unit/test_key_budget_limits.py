"""How the calling key's budget read orders its caps and reads each one."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.models.budgets import SCOPE_API_TOKEN, SCOPE_WORKSPACE
from gateway.repositories.budgets import CeilingUsage, LimitUsage
from gateway.services.budgets._key_limits import _KeyLimits, _public


def _usage(
    *,
    max_budget: str | None = None,
    spent: str = "0",
    reserved: str = "0",
    token_limit: int | None = None,
    tokens: int = 0,
    reserved_tokens: int = 0,
    request_limit: int | None = None,
    requests: int = 0,
    reserved_requests: int = 0,
    period_end: datetime | None = None,
) -> CeilingUsage:
    return CeilingUsage(
        max_budget=None if max_budget is None else Decimal(max_budget),
        current_spend=Decimal(spent),
        reserved_spend=Decimal(reserved),
        token_limit=token_limit,
        current_tokens=tokens,
        reserved_tokens=reserved_tokens,
        request_limit=request_limit,
        current_requests=requests,
        reserved_requests=reserved_requests,
        period_end=period_end,
    )


def _limit(
    usage: CeilingUsage,
    *,
    scope: str = SCOPE_API_TOKEN,
    provider: str | None = None,
    alignment: str | None = None,
) -> LimitUsage:
    return LimitUsage(
        scope=scope, provider_key_id=provider, budget_duration_sec=None, reset_alignment=alignment, usage=usage
    )


def _key_limits(ceilings: list[LimitUsage]) -> _KeyLimits:
    repositories = MagicMock()
    repositories.ceilings.usage_in_scopes = AsyncMock(return_value=ceilings)
    organizations = MagicMock()
    organizations.get_organization_id_for_workspace = AsyncMock(return_value=None)
    users = MagicMock()
    users.budget_use = AsyncMock(return_value=None)
    return _KeyLimits(repositories, organizations, users)


@pytest.mark.asyncio
async def test_a_provider_cap_reads_before_its_scopes_other_cap_and_never_sets_the_keys_dollars() -> None:
    workspace = _limit(_usage(max_budget="5"), scope=SCOPE_WORKSPACE)
    every_provider = _limit(_usage(max_budget="3"))
    one_provider = _limit(_usage(max_budget="1"), provider="openai")
    api_key = SimpleNamespace(id="key-1", user_id=None, workspace_id=uuid.uuid4())

    body = await _key_limits([workspace, every_provider, one_provider]).read(api_key)  # type: ignore[arg-type]

    assert [(limit.scope, limit.provider) for limit in body.limits] == [
        (SCOPE_API_TOKEN, "openai"),
        (SCOPE_API_TOKEN, None),
        (SCOPE_WORKSPACE, None),
    ]
    # The openai cap leaves less, but a request on another provider never meets it.
    assert body.remaining_usd == 3.0


@pytest.mark.asyncio
async def test_a_key_with_no_dollar_cap_has_no_dollar_total() -> None:
    api_key = SimpleNamespace(id="key-1", user_id=None, workspace_id=uuid.uuid4())

    body = await _key_limits([_limit(_usage(token_limit=100))]).read(api_key)  # type: ignore[arg-type]

    assert body.remaining_usd is None
    assert body.limits[0].remaining_tokens == 100


def test_an_ended_window_with_an_unknown_alignment_keeps_its_recorded_use() -> None:
    ended = datetime.now(UTC) - timedelta(hours=1)
    limit = _limit(_usage(max_budget="1", spent="0.4", reserved="0.1", period_end=ended), alignment="fortnight")

    public = _public(limit, datetime.now(UTC))

    assert public.resets_at == ended
    assert (public.used_usd, public.remaining_usd) == (0.5, 0.5)


def test_use_past_a_cap_reads_as_nothing_left_on_every_axis() -> None:
    limit = _limit(
        _usage(
            max_budget="1",
            spent="0.9",
            reserved="0.3",
            token_limit=10,
            tokens=8,
            reserved_tokens=5,
            request_limit=2,
            requests=2,
            reserved_requests=1,
        )
    )

    public = _public(limit, datetime.now(UTC))

    assert (public.used_usd, public.remaining_usd) == (1.2, 0.0)
    assert (public.used_tokens, public.remaining_tokens) == (13, 0)
    assert (public.used_requests, public.remaining_requests) == (3, 0)
