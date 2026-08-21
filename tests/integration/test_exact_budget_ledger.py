"""The spend ledger equals the settled rows that fed it, exactly.

``usage_logs.cost`` has been exact since mozilla-ai/otari#661, and the counters
the budget gate enforces against are exact as of mozilla-ai/otari#691. What that
buys is the invariant asserted here: over any number of requests,
``users.spend`` is the sum of the rows attributed to the user, with no residue.

Every amount here is compared exactly, and the first test also carries the
float-era addition alongside its assertion. That guard is the point: the drift a
float counter carried was far below a cent, so a suite that compared
approximately would keep passing on a counter that had quietly gone back to
binary floating point.
"""

from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER
from gateway.models.entities import Budget, ScopedBudget, UsageLog, User
from gateway.services.budget_service import ReservationHandle, reconcile_reservation, reserve_budget
from gateway.services.scoped_budget_service import ApplicableBudget, reserve, settle

from .conftest import MODEL_NAME

# $1.00 per million in, $1.00 per million out, so a token count is a cost in
# micro-dollars and each request's amount is legible in the numbers below.
_RATE = 1.0

# Token counts chosen so the settled amounts are the textbook binary-floating-point
# offenders: 0.1 + 0.2 + 0.3 is 0.6000000000000001 as doubles, and the long tail
# adds the sub-cent amounts a real deployment's cheap calls produce.
_REQUESTS = (
    (100_000, 0),
    (200_000, 0),
    (0, 300_000),
    (7, 0),
    (1_234, 5_678),
    (0, 9),
    (33_333, 66_667),
)


def _completion(prompt_tokens: int, completion_tokens: int) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-ledger",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _chat(client: TestClient, headers: dict[str, str], usage: ChatCompletion) -> Any:
    async def _acompletion(**_kwargs: Any) -> ChatCompletion:
        return usage

    with patch("gateway.api.routes.chat.acompletion") as mock:
        mock.side_effect = _acompletion
        return client.post(
            "/v1/chat/completions",
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}]},
            headers=headers,
        )


def test_spend_equals_the_exact_sum_of_the_rows_that_produced_it(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """The reproduction from mozilla-ai/otari#691, as an assertion.

    Seven completions through the live gateway, each settling to an exact
    micro-dollar amount. The counter they accumulate into is the one a 403 is
    decided against, so it has to be their sum and not an approximation of it.
    """
    assert (
        client.post(
            "/v1/pricing",
            json={
                "model_key": MODEL_NAME,
                "input_price_per_million": _RATE,
                "output_price_per_million": _RATE,
            },
            headers=master_key_header,
        ).status_code
        == 200
    )
    assert client.post("/v1/users", json={"user_id": "ledger-user"}, headers=master_key_header).status_code == 200
    key = client.post("/v1/keys", json={"key_name": "ledger", "user_id": "ledger-user"}, headers=master_key_header)
    assert key.status_code == 200, key.text
    headers = {API_KEY_HEADER: f"Bearer {key.json()['key']}"}

    for prompt_tokens, completion_tokens in _REQUESTS:
        response = _chat(client, headers, _completion(prompt_tokens, completion_tokens))
        assert response.status_code == 200, response.text

    costs = [
        cost
        for cost in db_session.execute(select(UsageLog.cost).where(UsageLog.user_id == "ledger-user")).scalars()
        if cost is not None
    ]
    assert len(costs) == len(_REQUESTS)

    user = db_session.execute(select(User).where(User.user_id == "ledger-user")).scalar_one()
    assert user.spend == sum(costs, Decimal(0))
    # Nothing is still held: every reservation was reconciled, and the amount
    # released was the amount taken.
    assert user.reserved == Decimal(0)

    # The same additions as doubles, which is what the counter used to be. If
    # this ever stops differing, the fixture has drifted to amounts that do not
    # exercise the problem and the assertion above has stopped meaning anything.
    float_total = 0.0
    for cost in costs:
        float_total += float(cost)
    assert Decimal(str(float_total)) != user.spend


@pytest.mark.asyncio
async def test_many_reconciles_leave_no_residue(async_db: AsyncSession) -> None:
    """A thousand settlements, and the counter is still the sum of them.

    The per-request error a float counter carried was far below a cent; what made
    it matter was that it accumulated for a whole budget period. So the shape of
    the test is the shape of the bug: settle many times, then compare.
    """
    async_db.add(User(user_id="residue-user"))
    await async_db.commit()

    amount = Decimal("0.000001")
    handle = ReservationHandle(user_id="residue-user", estimate=Decimal(0), reserved=False, strategy="disabled")
    for _ in range(1_000):
        await reconcile_reservation(async_db, handle, amount)

    async_db.expire_all()
    user = (await async_db.execute(select(User).where(User.user_id == "residue-user"))).scalar_one()
    assert user.spend == Decimal("0.001000")


@pytest.mark.asyncio
async def test_a_hold_is_released_at_the_amount_it_was_taken(async_db: AsyncSession) -> None:
    """Reserve and settle a repeating fraction; the hold comes back to zero.

    The clamp in ``_release_reserved`` is a floor, not a correction: if the
    release were a hair larger than the hold it would silently zero the counter
    and hide the mismatch, and if it were a hair smaller the remainder would
    outlive the request and permanently shrink the user's budget.
    """
    async_db.add(Budget(budget_id="hold-budget", max_budget=Decimal("10")))
    async_db.add(User(user_id="hold-user", budget_id="hold-budget"))
    await async_db.commit()

    for _ in range(50):
        handle = await reserve_budget(async_db, "hold-user", Decimal("0.033333"))
        assert handle.reserved
        await reconcile_reservation(async_db, handle, Decimal("0.033333"))

    async_db.expire_all()
    user = (await async_db.execute(select(User).where(User.user_id == "hold-user"))).scalar_one()
    assert user.reserved == Decimal(0)
    assert user.spend == Decimal("1.666650")


@pytest.mark.asyncio
async def test_a_scoped_ceiling_accumulates_exactly(async_db: AsyncSession) -> None:
    """The second enforcement mechanism keeps the same invariant.

    ``scoped_budgets`` has its own counters and its own conditional UPDATE (and
    since ``f3a5c7e9d1b4`` reads its cap off the budget it names, through a
    correlated subquery), so exactness on ``users.spend`` says nothing about it.
    """
    async_db.add(Budget(budget_id="ceiling-budget", max_budget=Decimal("5")))
    ceiling = ScopedBudget(scope_type="workspace", scope_id="ws-exact", budget_id="ceiling-budget")
    async_db.add(ceiling)
    await async_db.commit()
    ceiling_id = ceiling.id

    applicable = (ApplicableBudget(budget_id=ceiling_id, scope_type="workspace", provider_key_id=None),)
    amount = Decimal("0.011111")
    for _ in range(90):
        assert await reserve(async_db, applicable, amount) is None
        await settle(async_db, [ceiling_id], actual_cost=amount, held=amount)

    async_db.expire_all()
    row = (await async_db.execute(select(ScopedBudget).where(ScopedBudget.id == ceiling_id))).scalar_one()
    assert row.current_spend == Decimal("0.999990")
    assert row.reserved_spend == Decimal(0)
