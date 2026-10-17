"""``mock_response``: a load test's chat completion answered by Otari without calling the provider.

Allowed only on a key with ``allow_mock_response``. The request is still
admitted against budgets, priced and settled, so a load test exercises the
gateway as production traffic does, without paying for or waiting on a model.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT
from gateway.models.usage import UsageLog

from .conftest import MODEL_NAME

MOCK = "This answer never reached a provider."


def _price(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": MODEL_NAME, "input_price_per_million": 1.0, "output_price_per_million": 2.0},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text


def _key(client: TestClient, master_key_header: dict[str, str], **settings: Any) -> dict[str, str]:
    created = client.post(
        f"{API_ROOT}/keys", json={"key_name": "loadtest", "user_id": "loadtest", **settings}, headers=master_key_header
    )
    assert created.status_code == 200, created.text
    return {API_KEY_HEADER: f"Bearer {created.json()['key']}"}


def _chat(client: TestClient, headers: dict[str, str], **extra: Any) -> Any:
    body = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "Load test prompt"}], **extra}
    # Any call to the provider fails the test: a mocked request must never reach one.
    provider = AsyncMock(side_effect=AssertionError("the provider was called"))
    with patch("gateway.api.routes.chat.acompletion", new=provider):
        response = client.post(f"{API_ROOT}/chat/completions", json=body, headers=headers)
        response.read()
    return response


def test_a_mocked_completion_answers_without_the_provider_and_is_billed(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _price(client, master_key_header)
    headers = _key(client, master_key_header, allow_mock_response=True)

    response = _chat(client, headers, mock_response=MOCK)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"] == MOCK
    assert body["usage"]["completion_tokens"] == (len(MOCK) + 3) // 4
    assert body["usage"]["cost_usd"] > 0
    row = db_session.query(UsageLog).one()
    assert row.status == "success"
    assert row.cost is not None and row.cost > 0


def test_a_mocked_stream_carries_the_text_and_its_usage(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _price(client, master_key_header)
    headers = _key(client, master_key_header, allow_mock_response=True)

    response = _chat(client, headers, mock_response=MOCK, stream=True)

    assert response.status_code == 200, response.text
    assert MOCK in response.text
    assert response.text.rstrip().endswith("data: [DONE]")
    assert db_session.query(UsageLog).one().status == "success"


@pytest.mark.parametrize("settings", [{}, {"allow_mock_response": False}])
def test_a_key_without_the_flag_is_refused_before_anything_is_billed(
    client: TestClient, master_key_header: dict[str, str], db_session: Session, settings: dict[str, Any]
) -> None:
    _price(client, master_key_header)
    headers = _key(client, master_key_header, **settings)

    response = _chat(client, headers, mock_response=MOCK)

    assert response.status_code == 403
    assert "allow_mock_response" in response.text
    assert db_session.query(UsageLog).count() == 0


def test_the_master_key_cannot_mock(client: TestClient, master_key_header: dict[str, str]) -> None:
    _price(client, master_key_header)

    response = _chat(client, master_key_header, mock_response=MOCK, user="loadtest")

    assert response.status_code == 403


def test_the_flag_is_read_and_changed_through_the_keys_api(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    created = client.post(
        f"{API_ROOT}/keys", json={"key_name": "lt", "user_id": "lt", "allow_mock_response": True}, headers=master_key_header
    ).json()
    turned_off = client.patch(
        f"{API_ROOT}/keys/{created['id']}", json={"allow_mock_response": False}, headers=master_key_header
    )

    assert created["allow_mock_response"] is True
    assert turned_off.json()["allow_mock_response"] is False
