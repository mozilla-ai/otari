"""A key flagged exclude_from_budget logs cost but never touches users.spend.

Proves the load-bearing invariant: the spend write is gated on the reservation
handle, not merely skipped at reserve time, so an excluded key is billed to the
usage log yet neither consumes budget nor is gated by it.
"""

from typing import Any
from unittest.mock import patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER
from gateway.models.entities import UsageLog, User

from .conftest import MODEL_NAME


def _mock_completion() -> ChatCompletion:
    # 1M input @ $2.5 + 0.5M output @ $10 = $7.50, far above any tiny budget below.
    return ChatCompletion(
        id="chatcmpl-x",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[
            Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")
        ],
        usage=CompletionUsage(prompt_tokens=1_000_000, completion_tokens=500_000, total_tokens=1_500_000),
    )


def _seed(client: TestClient, master_key_header: dict[str, str]) -> None:
    client.post("/v1/pricing", json={
        "model_key": MODEL_NAME, "input_price_per_million": 2.5, "output_price_per_million": 10.0,
    }, headers=master_key_header)


def _make_key(
    client: TestClient, master_key_header: dict[str, str], user_id: str, *, exclude: bool
) -> dict[str, str]:
    resp = client.post(
        "/v1/keys",
        json={"key_name": user_id, "user_id": user_id, "exclude_from_budget": exclude},
        headers=master_key_header,
    )
    assert resp.status_code == 200, resp.text
    return {API_KEY_HEADER: f"Bearer {resp.json()['key']}"}


async def _mock_acompletion(**_kwargs: Any) -> ChatCompletion:
    return _mock_completion()


def _chat(client: TestClient, headers: dict[str, str]) -> Any:
    with patch("gateway.api.routes.chat.acompletion") as mock:
        mock.side_effect = _mock_acompletion
        return client.post(
            "/v1/chat/completions",
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}]},
            headers=headers,
        )


def test_excluded_key_logged_but_not_billed_or_gated(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """Excluded key: request succeeds even for a maxed-out budget, spend stays 0."""
    _seed(client, master_key_header)
    # A tiny budget that a normal $7.50 request could never pass.
    budget = client.post(
        "/v1/budgets", json={"max_budget": 0.01, "budget_duration_sec": 86400}, headers=master_key_header
    ).json()
    client.post(
        "/v1/users",
        json={"user_id": "exempt-user", "budget_id": budget["budget_id"]},
        headers=master_key_header,
    )
    headers = _make_key(client, master_key_header, "exempt-user", exclude=True)

    # Two requests: both succeed despite the tiny budget (never reserved/gated).
    assert _chat(client, headers).status_code == 200
    assert _chat(client, headers).status_code == 200

    user = db_session.query(User).filter(User.user_id == "exempt-user").one()
    assert float(user.spend) == pytest.approx(0.0)
    assert float(user.reserved) == pytest.approx(0.0)

    rows = db_session.query(UsageLog).filter(UsageLog.user_id == "exempt-user").all()
    assert len(rows) == 2
    assert all(r.counts_toward_budget is False for r in rows)
    # Cost is still computed and recorded, just not billed to spend.
    assert all(r.cost == pytest.approx(7.5) for r in rows)


def test_normal_key_still_bills_spend(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """Regression: a normal (non-exempt) key still reconciles cost into spend."""
    _seed(client, master_key_header)
    client.post("/v1/users", json={"user_id": "billed-user"}, headers=master_key_header)
    headers = _make_key(client, master_key_header, "billed-user", exclude=False)

    assert _chat(client, headers).status_code == 200

    user = db_session.query(User).filter(User.user_id == "billed-user").one()
    assert float(user.spend) == pytest.approx(7.5)
    row = db_session.query(UsageLog).filter(UsageLog.user_id == "billed-user").one()
    assert row.counts_toward_budget is True
