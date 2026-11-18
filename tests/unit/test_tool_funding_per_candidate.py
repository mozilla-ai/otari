"""The gateway tools' funding gate decides each candidate's tools on their own.

A fallback whose tools the organization cannot fund is skipped, not the
request: the head may run the same tool natively and need nothing funded.
"""

import uuid
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from gateway.api.routes import _pipeline as pipeline
from gateway.ports.billing_port import BillingPort, InsufficientFundsError
from gateway.services.tools import PricedTool

HEAD = frozenset({"head"})
FALLBACK = frozenset({"fallback"})


class _Billing:
    def __init__(self, unfundable: frozenset[str]) -> None:
        self.unfundable = unfundable
        self.asked: list[frozenset[str]] = []

    async def apply_due_credit(self, *, organization_id: uuid.UUID) -> None:
        return None

    async def require_funds_on_deposit(
        self, *, organization_id: Any, billed: Any, tools: frozenset[str], actor_user_id: Any
    ) -> None:
        self.asked.append(tools)
        if tools & self.unfundable:
            raise InsufficientFundsError(Decimal(0))


def _answer(*paid: str) -> Any:
    return SimpleNamespace(code=SimpleNamespace(use_sandbox=False), web_access=paid)


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(
        pipeline,
        "admitted_tools",
        lambda *, code_execution, web: [PricedTool(name, None, (), deployment_paid=True) for name in web],
    )
    monkeypatch.setattr(pipeline, "release_session", AsyncMock(return_value=True))
    refused = AsyncMock()
    monkeypatch.setattr(pipeline, "_refuse_tool_funding", refused)
    adapter = SimpleNamespace(error=lambda code, detail, kind, headers=None: HTTPException(code, detail, headers))
    ctx = SimpleNamespace(
        organization_id=uuid.uuid4(), api_key_id="key", user_id=None, request_id="req", db=object(), tool_funding=None
    )
    return adapter, ctx, refused


@pytest.mark.asyncio
async def test_a_fallback_whose_tools_cannot_be_funded_is_skipped_and_the_head_served(gate: Any) -> None:
    adapter, ctx, refused = gate
    billing = _Billing(unfundable=frozenset({"otari:web_search"}))

    unfunded = await pipeline._admit_tool_funding(
        adapter, ctx, cast(BillingPort, billing), {HEAD: _answer(), FALLBACK: _answer("web_search")}
    )

    assert set(unfunded) == {FALLBACK}
    assert unfunded[FALLBACK].error.status_code == 402
    refused.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_request_is_refused_when_no_candidate_can_fund_its_tools(gate: Any) -> None:
    adapter, ctx, refused = gate
    billing = _Billing(unfundable=frozenset({"otari:web_search"}))

    with pytest.raises(HTTPException) as raised:
        await pipeline._admit_tool_funding(
            adapter, ctx, cast(BillingPort, billing), {HEAD: _answer("web_search"), FALLBACK: _answer("web_search")}
        )

    assert raised.value.status_code == 402
    assert "The gateway tool 'otari_web_search' runs on this deployment's account" in raised.value.detail
    assert billing.asked == [frozenset({"otari:web_search"})]
    refused.assert_awaited_once()


@pytest.mark.asyncio
async def test_funded_tools_admit_every_candidate(gate: Any) -> None:
    adapter, ctx, refused = gate
    billing = _Billing(unfundable=frozenset())

    unfunded = await pipeline._admit_tool_funding(
        adapter, ctx, cast(BillingPort, billing), {HEAD: _answer("web_search"), FALLBACK: _answer("web_search")}
    )

    assert unfunded == {}
    assert ctx.tool_funding is not None
