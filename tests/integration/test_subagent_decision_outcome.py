"""``PUT /api/v1/routing/decisions/{id}/outcome``: what a routed subagent consumed, priced twice.

A report is priced at the models it names and at the decision's baseline, both
from this deployment's rates. These pin the arithmetic, that a report replaces
the one before it, who may report on a decision, and what is stored where a
side has no price.
"""

import uuid
from collections.abc import Iterator
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT
from gateway.models.api_keys import APIKey
from gateway.models.routing import SubagentDecision
from gateway.services.routing.recommend import RECOMMENDATION_QUESTION
from gateway.services.secret_box import generate_secret_key

from .conftest import ALL_KEY_SCOPES

RECOMMEND = f"{API_ROOT}/routing/recommend"
ROUTERS = f"{API_ROOT}/routing/recommenders"
OPUS = "claude-opus-outcome-test"
SONNET = "claude-sonnet-outcome-test"
HAIKU = "claude-haiku-outcome-test"
ANSWER: dict[str, Any] = {
    "model": "decider-1.0",
    "answers": {RECOMMENDATION_QUESTION: {"type": "choice", "choice": "haiku"}},
    "usage": {"input_tokens": 800, "output_tokens": 0},
}
LINE: dict[str, Any] = {
    "model": HAIKU,
    "input_tokens": 1000,
    "output_tokens": 2000,
    "cache_read_tokens": 10_000,
    "cache_write_tokens": 4000,
}


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


def _price(client: TestClient, headers: dict[str, str], model: str, *, input_price: float) -> None:
    """Price ``model`` with output at 5x input, cache reads at a tenth and cache writes at 1.25x."""
    response = client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": f"anthropic:{model}",
            "input_price_per_million": input_price,
            "output_price_per_million": input_price * 5,
            "cache_read_price_per_million": input_price / 10,
            "cache_write_price_per_million": input_price * 1.25,
        },
        headers=headers,
    )
    assert response.status_code == 200, response.text


@pytest.fixture
def priced(client: TestClient, master_key_header: dict[str, str]) -> None:
    _price(client, master_key_header, OPUS, input_price=10)
    _price(client, master_key_header, HAIKU, input_price=1)


def _decide(client: TestClient, master_key_header: dict[str, str], headers: dict[str, str], **spawn: Any) -> str:
    """Store a router if none is, ask it once, and return the decision's id."""
    if not client.get(ROUTERS, headers=master_key_header).json()["data"]:
        stored = client.post(
            ROUTERS,
            json={"provider": "openrouter", "model": "stored-decider", "api_key": "sk-or-1234"},
            headers=master_key_header,
        )
        assert stored.status_code == 201, stored.text
    body: dict[str, Any] = {
        "harness": "claude-code",
        "session_id": "bf05abe2-5ff2-4eb0-8459-ab578d5c9468",
        "tool_use_id": "toolu_01",
        "agent_type": "Explore",
        "prompt": "List the files.",
        "parent_model": OPUS,
    }
    body.update(spawn)
    with patch("gateway.adapters.agent_model_recommender_adapter.request_decision", AsyncMock(return_value=ANSWER)):
        response = client.post(RECOMMEND, json=body, headers=headers)
    assert response.status_code == 200, response.text
    decision_id: str = response.json()["decision_id"]
    return decision_id


def _outcome(decision_id: str) -> str:
    return f"{API_ROOT}/routing/decisions/{decision_id}/outcome"


def _stored(db: Session, decision_id: str) -> SubagentDecision:
    db.expire_all()
    return db.execute(select(SubagentDecision).where(SubagentDecision.id == uuid.UUID(decision_id))).scalar_one()


def test_a_report_is_priced_at_the_models_it_names_and_at_the_baseline(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    priced: None,
    test_db: Session,
) -> None:
    decision_id = _decide(client, master_key_header, api_key_header)

    response = client.put(_outcome(decision_id), json={"usage": [LINE]}, headers=api_key_header)

    assert response.status_code == 204, response.text
    decision = _stored(test_db, decision_id)
    assert (decision.outcome_kind, decision.ran_on, decision.unpriced_reason) == ("estimated", HAIKU, None)
    assert decision.outcome_at is not None
    assert decision.baseline_model_key == f"anthropic:{OPUS}"
    # 1000 in at $1, 2000 out at $5, 10,000 cache read at $0.10, 4000 cache write at $1.25, per million.
    assert decision.actual_cost == Decimal("0.017")
    # The same tokens at ten times the rates.
    assert decision.baseline_cost == Decimal("0.17")
    assert (decision.input_tokens, decision.output_tokens) == (1000, 2000)
    assert (decision.cache_read_tokens, decision.cache_write_tokens) == (10_000, 4000)
    assert decision.usage == [LINE]


def test_a_report_replaces_the_one_before_it(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    priced: None,
    test_db: Session,
) -> None:
    decision_id = _decide(client, master_key_header, api_key_header)
    first = {"model": HAIKU, "input_tokens": 100, "output_tokens": 100}
    # The subagent went on, partly on the parent's model.
    second = [{**LINE, "output_tokens": 1000}, {"model": OPUS, "input_tokens": 0, "output_tokens": 3000}]

    assert client.put(_outcome(decision_id), json={"usage": [first]}, headers=api_key_header).status_code == 204
    assert client.put(_outcome(decision_id), json={"usage": second}, headers=api_key_header).status_code == 204

    decision = _stored(test_db, decision_id)
    assert (decision.output_tokens, decision.ran_on) == (4000, OPUS)
    # Haiku: 1000 in, 1000 out, the cache counts. Opus: 3000 out at $50.
    assert decision.actual_cost == Decimal("0.012") + Decimal("0.15")
    assert decision.baseline_cost == Decimal("0.12") + Decimal("0.15")


@pytest.mark.parametrize(
    ("spawn", "ran_on", "key", "reason"),
    [
        # An alias the parent is of, read as the parent's model.
        ({"requested_model": "opus"}, HAIKU, f"anthropic:{OPUS}", None),
        # An alias nothing names a model for.
        ({"requested_model": "opus", "parent_model": SONNET}, HAIKU, None, "baseline_unresolved"),
        ({"parent_model": SONNET}, HAIKU, f"anthropic:{SONNET}", "baseline_unpriced"),
        ({}, SONNET, f"anthropic:{OPUS}", "actual_unpriced"),
    ],
)
def test_a_side_with_no_price_leaves_both_costs_empty_and_says_why(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    priced: None,
    test_db: Session,
    spawn: dict[str, Any],
    ran_on: str,
    key: str | None,
    reason: str | None,
) -> None:
    decision_id = _decide(client, master_key_header, api_key_header, **spawn)

    response = client.put(_outcome(decision_id), json={"usage": [{**LINE, "model": ran_on}]}, headers=api_key_header)

    assert response.status_code == 204, response.text
    decision = _stored(test_db, decision_id)
    assert (decision.outcome_kind, decision.baseline_model_key, decision.unpriced_reason) == ("estimated", key, reason)
    assert (decision.baseline_cost is None, decision.actual_cost is None) == (reason is not None, reason is not None)
    assert decision.output_tokens == 2000


def test_a_per_request_rate_prices_no_tokens(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    priced: None,
    test_db: Session,
) -> None:
    flat = client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": f"anthropic:{SONNET}",
            "input_price_per_million": 10_000,
            "output_price_per_million": 0,
            "unit": "request",
        },
        headers=master_key_header,
    )
    assert flat.status_code == 200, flat.text
    decision_id = _decide(client, master_key_header, api_key_header, parent_model=SONNET)

    assert client.put(_outcome(decision_id), json={"usage": [LINE]}, headers=api_key_header).status_code == 204

    decision = _stored(test_db, decision_id)
    assert (decision.unpriced_reason, decision.baseline_cost, decision.actual_cost) == ("per_request_rate", None, None)


def test_another_callers_decision_is_not_found(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    priced: None,
    test_db: Session,
) -> None:
    decision_id = _decide(client, master_key_header, api_key_header)
    assert client.post(f"{API_ROOT}/users", json={"user_id": "someone-else"}, headers=master_key_header).is_success
    other = client.post(
        f"{API_ROOT}/keys",
        json={"key_name": "other", "scopes": ALL_KEY_SCOPES, "user_id": "someone-else"},
        headers=master_key_header,
    ).json()

    response = client.put(
        _outcome(decision_id), json={"usage": [LINE]}, headers={API_KEY_HEADER: f"Bearer {other['key']}"}
    )

    assert response.status_code == 404, response.text
    assert response.json() == {"detail": "No subagent decision has this id"}
    assert _stored(test_db, decision_id).outcome_kind == "none"


def _userless_key(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> tuple[str, dict[str, str]]:
    """Create a key, take its user away, and return its id and header."""
    created = client.post(
        f"{API_ROOT}/keys", json={"key_name": "userless", "scopes": ALL_KEY_SCOPES}, headers=master_key_header
    ).json()
    key = test_db.execute(select(APIKey).where(APIKey.id == created["id"])).scalar_one()
    key.user_id = None
    test_db.commit()
    return created["id"], {API_KEY_HEADER: f"Bearer {created['key']}"}


def test_a_key_without_a_user_reaches_only_its_own_decisions(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    priced: None,
    test_db: Session,
) -> None:
    decision_id = _decide(client, master_key_header, api_key_header)
    key_id, userless = _userless_key(client, master_key_header, test_db)

    refused = client.put(_outcome(decision_id), json={"usage": [LINE]}, headers=userless)

    assert refused.status_code == 404, refused.text
    assert _stored(test_db, decision_id).outcome_kind == "none"

    # A key with no user is refused a recommendation, so its own decision is written directly.
    other = _stored(test_db, decision_id)
    own_id = uuid.uuid4()
    test_db.add(
        SubagentDecision(
            id=own_id,
            workspace_id=other.workspace_id,
            user_id=None,
            api_key_id=key_id,
            harness=other.harness,
            agent_type=other.agent_type,
            parent_model=other.parent_model,
            baseline_model=other.baseline_model,
            baseline_source=other.baseline_source,
            recommended_model=other.recommended_model,
        )
    )
    test_db.commit()

    assert client.put(_outcome(str(own_id)), json={"usage": [LINE]}, headers=userless).status_code == 204
    assert _stored(test_db, str(own_id)).outcome_kind == "estimated"


def test_a_line_with_no_tokens_does_not_leave_a_report_unpriced(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    priced: None,
    test_db: Session,
) -> None:
    decision_id = _decide(client, master_key_header, api_key_header)
    placeholder = {"model": "<synthetic>", "input_tokens": 0, "output_tokens": 0}

    response = client.put(_outcome(decision_id), json={"usage": [placeholder, LINE]}, headers=api_key_header)

    assert response.status_code == 204, response.text
    decision = _stored(test_db, decision_id)
    assert (decision.unpriced_reason, decision.ran_on) == (None, HAIKU)
    assert (decision.actual_cost, decision.baseline_cost) == (Decimal("0.017"), Decimal("0.17"))
    assert len(decision.usage or []) == 2


def test_a_report_of_only_empty_lines_prices_to_zero(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    priced: None,
    test_db: Session,
) -> None:
    decision_id = _decide(client, master_key_header, api_key_header)

    response = client.put(
        _outcome(decision_id),
        json={"usage": [{"model": "<synthetic>", "input_tokens": 0, "output_tokens": 0}]},
        headers=api_key_header,
    )

    assert response.status_code == 204, response.text
    decision = _stored(test_db, decision_id)
    assert decision.unpriced_reason is None
    assert (decision.actual_cost, decision.baseline_cost) == (Decimal(0), Decimal(0))


def test_an_unknown_decision_is_not_found(client: TestClient, api_key_header: dict[str, str]) -> None:
    response = client.put(_outcome(str(uuid.uuid4())), json={"usage": [LINE]}, headers=api_key_header)

    assert response.status_code == 404, response.text


def test_the_master_key_reports_on_any_decision(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    priced: None,
    test_db: Session,
) -> None:
    decision_id = _decide(client, master_key_header, api_key_header)

    assert client.put(_outcome(decision_id), json={"usage": [LINE]}, headers=master_key_header).status_code == 204

    assert _stored(test_db, decision_id).outcome_kind == "estimated"


def test_metered_usage_is_not_replaced_by_a_report(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    priced: None,
    test_db: Session,
) -> None:
    decision_id = _decide(client, master_key_header, api_key_header)
    measured = _stored(test_db, decision_id)
    measured.outcome_kind = "measured"
    measured.actual_cost = Decimal("0.5")
    test_db.commit()

    assert client.put(_outcome(decision_id), json={"usage": [LINE]}, headers=api_key_header).status_code == 204

    decision = _stored(test_db, decision_id)
    assert (decision.outcome_kind, decision.actual_cost, decision.usage) == ("measured", Decimal("0.5"), None)


@pytest.mark.parametrize(
    "body",
    [
        {"usage": []},
        {"usage": [{**LINE, "input_tokens": -1}]},
        {"usage": [{**LINE, "prompt": "the task"}]},
        {"usage": [LINE] * 17},
        {"usage": [LINE], "description": "the task"},
    ],
    ids=["no-lines", "negative-count", "unknown-line-field", "too-many-lines", "unknown-field"],
)
def test_a_malformed_report_is_unprocessable(
    client: TestClient, api_key_header: dict[str, str], body: dict[str, Any]
) -> None:
    response = client.put(_outcome(str(uuid.uuid4())), json=body, headers=api_key_header)

    assert response.status_code == 422, response.text


def test_a_report_needs_a_credential(client: TestClient) -> None:
    response = client.put(_outcome(str(uuid.uuid4())), json={"usage": [LINE]})

    assert response.status_code == 401, response.text


def test_a_key_without_the_agents_scope_is_refused(client: TestClient, master_key_header: dict[str, str]) -> None:
    key = client.post(
        f"{API_ROOT}/keys", json={"key_name": "inference-only", "scopes": ["inference:run"]}, headers=master_key_header
    ).json()

    response = client.put(
        _outcome(str(uuid.uuid4())), json={"usage": [LINE]}, headers={API_KEY_HEADER: f"Bearer {key['key']}"}
    )

    assert response.status_code == 403, response.text
    assert "'agents:write'" in response.text
