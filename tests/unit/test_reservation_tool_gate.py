"""The reservation tool gate holds each gateway-run call's price before it runs, and a search's pages as one hold.

The rate lookup and the hold are stubbed, so what is pinned is the wiring: one
rate is resolved per label, and the hold is asked for as many calls as the tally
asked for, at that rate.
"""

import uuid
from collections.abc import Sequence
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

import gateway.api.routes._pipeline as pipeline
from gateway.core.config import GatewayConfig
from gateway.services.budgets import ReservationHandle
from gateway.services.pricing import ToolCharges
from gateway.types.tool_charges import LabelCalls

_UNIT = Decimal("0.0000004")
_ORGANIZATION = uuid.UUID(int=7)


def _call(calls: int = 1) -> LabelCalls:
    return LabelCalls(
        tool="web_fetch",
        label="fake:fake-fetch",
        rate_keys=("fake:fake-fetch",),
        calls=calls,
        unreported_calls=calls,
        reported_cost=Decimal(0),
        deployment_paid=True,
    )


def _ctx() -> Any:
    """A context carrying only what the gate reads, with a reservation that counts."""
    return SimpleNamespace(
        db=object(),
        organization_id=_ORGANIZATION,
        reservation=ReservationHandle(
            user_id="user-1",
            estimate=Decimal(1),
            reserved=True,
            strategy="for_update",
            scope_ceilings=(),
            reservation_id="res-7",
        ),
        config=GatewayConfig(),
        tool_hold=Decimal(0),
    )


@pytest.fixture
def priced(monkeypatch: pytest.MonkeyPatch) -> list[LabelCalls]:
    """Price every call at ``_UNIT``, below what a cost rounds to, and keep each call the rate was asked for.

    The total is rounded to nothing, as settlement rounds it.
    """
    asked: list[LabelCalls] = []

    async def fake_price(_db: Any, calls: Sequence[LabelCalls], *, organization_id: uuid.UUID) -> ToolCharges:
        assert organization_id == _ORGANIZATION, "priced for the request's organization"
        asked.extend(calls)
        return ToolCharges(Decimal(0), [], {call.tool: _UNIT for call in calls}, [], Decimal(0))

    async def fake_release(_db: Any) -> None:
        pass

    monkeypatch.setattr(pipeline, "price_tool_calls", fake_price)
    monkeypatch.setattr(pipeline, "release_session", fake_release)
    return asked


@pytest.fixture
def holds(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Decimal, int]]:
    """Let every hold through, as the budgets domain holds it, and keep each unit amount and count asked for."""
    held: list[tuple[Decimal, int]] = []

    async def fake_hold(_db: Any, _handle: Any, amount: Decimal, *, strategy: str, calls: int) -> Decimal:
        held.append((amount, calls))
        return amount * calls

    monkeypatch.setattr(pipeline, "hold_tool_call", fake_hold)
    return held


@pytest.mark.asyncio
async def test_the_hold_is_asked_for_the_calls_the_tally_asked_for_at_the_unrounded_rate(
    priced: list[LabelCalls], holds: list[tuple[Decimal, int]]
) -> None:
    """Twenty pages at a rate below a micro-dollar are held at that rate, which rounding one call would lose."""
    ctx = _ctx()
    pages = _call(calls=20)

    assert await pipeline.ReservationToolGate(ctx).admit(pages)

    assert holds == [(_UNIT, 20)]
    assert ctx.tool_hold == Decimal("0.000008"), "what the budgets domain held"
    assert priced == [pages]


@pytest.mark.asyncio
async def test_a_labels_rate_is_resolved_once(priced: list[LabelCalls], holds: list[tuple[Decimal, int]]) -> None:
    gate = pipeline.ReservationToolGate(_ctx())

    assert await gate.admit(_call(calls=2))
    assert await gate.admit(_call())

    assert holds == [(_UNIT, 2), (_UNIT, 1)]
    assert len(priced) == 1


@pytest.mark.asyncio
async def test_a_hold_that_does_not_fit_refuses_the_calls_and_holds_nothing(
    priced: list[LabelCalls], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def refuse(_db: Any, _handle: Any, _amount: Decimal, *, strategy: str, calls: int) -> None:
        return None

    monkeypatch.setattr(pipeline, "hold_tool_call", refuse)
    ctx = _ctx()

    assert not await pipeline.ReservationToolGate(ctx).admit(_call(calls=3))

    assert ctx.tool_hold == Decimal(0)


@pytest.mark.asyncio
async def test_a_reservation_that_holds_nothing_admits_every_call(
    priced: list[LabelCalls], holds: list[tuple[Decimal, int]]
) -> None:
    ctx = _ctx()
    ctx.reservation.reservation_id = None

    assert await pipeline.ReservationToolGate(ctx).admit(_call(calls=20))

    assert holds == []
    assert priced == []
