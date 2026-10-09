"""A rerank model priced ``unit: requests`` is charged per search unit the provider bills.

Rerank providers that bill by query batch report ``billed_units.search_units``
and no tokens, so a token rate never prices them. These pin that a per-request
rate bills the reported units, bills one unit when none is reported, reserves
one unit up front, and leaves a token-priced model on the token path.
"""

import time
from collections.abc import Callable
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest
from any_llm.types.rerank import RerankMeta, RerankResponse, RerankResult, RerankUsage
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT
from gateway.models.usage import UsageLog
from gateway.services.budgets import reserve_budget

MODEL = "cohere:rerank-v3.5"
USER_ID = "rerank-unit-user"
# $0.002 per search unit, stored as USD per million requests.
PER_UNIT_RATE = 2000.0
UNIT_COST = Decimal("0.002")
# $3 per million input tokens, for the token-priced case.
TOKEN_RATE = 3.0


def _price(client: TestClient, headers: dict[str, str], *, unit: str, rate: float) -> None:
    response = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": MODEL, "input_price_per_million": rate, "output_price_per_million": 0.0, "unit": unit},
        headers=headers,
    )
    assert response.status_code == 200, response.text


def _user_key(client: TestClient, headers: dict[str, str]) -> dict[str, str]:
    budget = client.post(f"{API_ROOT}/budgets", json={"max_budget": 1.0}, headers=headers)
    assert budget.status_code == 200, budget.text
    user = client.post(
        f"{API_ROOT}/users", json={"user_id": USER_ID, "budget_id": budget.json()["budget_id"]}, headers=headers
    )
    assert user.status_code == 200, user.text
    key = client.post(f"{API_ROOT}/keys", json={"key_name": "rerank-key", "user_id": USER_ID}, headers=headers)
    assert key.status_code == 200, key.text
    return {API_KEY_HEADER: f"Bearer {key.json()['key']}"}


def _response(*, search_units: float | None, tokens: int | None = None) -> RerankResponse:
    return RerankResponse(
        id="rerank-1",
        model="rerank-v3.5",
        results=[RerankResult(index=0, relevance_score=0.9), RerankResult(index=1, relevance_score=0.1)],
        meta=RerankMeta(billed_units={"search_units": search_units} if search_units is not None else None),
        usage=RerankUsage(total_tokens=tokens) if tokens is not None else None,
    )


def _rerank(client: TestClient, headers: dict[str, str], result: RerankResponse) -> tuple[Any, list[Decimal]]:
    """Send one rerank request and return the response and every amount reserved for it."""

    async def _arerank(**_kwargs: Any) -> RerankResponse:
        return result

    reserved: list[Decimal] = []

    async def _capture(*args: Any, **kwargs: Any) -> Any:
        reserved.append(Decimal(str(args[2])))
        return await reserve_budget(*args, **kwargs)

    with (
        patch("gateway.api.routes.rerank.arerank", side_effect=_arerank),
        patch("gateway.api.routes._passthrough.reserve_budget", side_effect=_capture),
    ):
        response = client.post(
            f"{API_ROOT}/rerank",
            json={"model": MODEL, "query": "capital of France", "documents": ["Paris.", "Berlin."]},
            headers=headers,
        )
    return response, reserved


def _row(make_session: Callable[[], Session], *, timeout: float = 3.0) -> UsageLog:
    """The request's usage row, polled because a background log writer commits it."""
    deadline = time.monotonic() + timeout
    while True:
        with make_session() as db:
            row = db.execute(select(UsageLog).where(UsageLog.user_id == USER_ID)).scalar_one_or_none()
            if row is not None:
                db.expunge(row)
                return row
        assert time.monotonic() < deadline, "the usage row was never written"
        time.sleep(0.1)


def _user(client: TestClient, headers: dict[str, str]) -> dict[str, Any]:
    response = client.get(f"{API_ROOT}/users/{USER_ID}", headers=headers)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def test_reported_search_units_are_billed_at_the_per_request_rate(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _price(client, master_key_header, unit="requests", rate=PER_UNIT_RATE)
    headers = _user_key(client, master_key_header)

    response, reserved = _rerank(client, headers, _response(search_units=2.0))

    assert response.status_code == 200, response.text
    assert response.json()["meta"]["billed_units"] == {"search_units": 2.0}
    # One unit is held up front; the provider's count settles the rest.
    assert reserved == [UNIT_COST]
    row = _row(db_session_factory)
    assert row.status == "success"
    assert row.cost == Decimal("0.004")
    assert row.billing_meters == {"search_units": 2}
    assert row.pricing_breakdown == [{"meter": "search_units", "units": 2, "unit_rate": 0.002, "cost": 0.004}]
    assert (row.prompt_tokens, row.total_tokens) == (None, None)
    user = _user(client, master_key_header)
    assert user["spend"] == pytest.approx(0.004)
    assert user["reserved"] == pytest.approx(0.0)


def test_a_provider_reporting_no_units_is_billed_one(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _price(client, master_key_header, unit="requests", rate=PER_UNIT_RATE)
    headers = _user_key(client, master_key_header)

    response, _ = _rerank(client, headers, _response(search_units=None))

    assert response.status_code == 200, response.text
    row = _row(db_session_factory)
    assert row.cost == UNIT_COST
    assert row.billing_meters == {"search_units": 1}


def test_a_token_priced_model_still_bills_reported_tokens(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _price(client, master_key_header, unit="tokens", rate=TOKEN_RATE)
    headers = _user_key(client, master_key_header)

    response, _ = _rerank(client, headers, _response(search_units=1.0, tokens=1000))

    assert response.status_code == 200, response.text
    row = _row(db_session_factory)
    assert row.cost == Decimal("0.003")
    assert row.prompt_tokens == 1000
    assert row.billing_meters == {"total_input_tokens": 1000}


def test_a_token_priced_model_with_only_search_units_stays_unpriced(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """A token rate cannot price a count of searches, so the row carries no cost."""
    _price(client, master_key_header, unit="tokens", rate=TOKEN_RATE)
    headers = _user_key(client, master_key_header)

    response, _ = _rerank(client, headers, _response(search_units=1.0))

    assert response.status_code == 200, response.text
    row = _row(db_session_factory)
    assert row.status == "success"
    assert row.cost is None
    assert row.billing_meters is None
