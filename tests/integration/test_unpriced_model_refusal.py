"""What a request for an unpriced model is told on a gateway configured the README way.

``default_pricing: true`` with ``require_pricing`` at its default: a model the
bundled price list misses is refused with a 402. Two causes hide behind that
miss, and the refusal names the one that applies: a provider the gateway holds
no credential for answers ``provider_not_configured``, because pricing the model
would not make it servable, and a misspelled model on a configured provider
gets a 402 that points at the typo rather than at the fallback already on.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.error_codes import ERROR_CODE_HEADER, PRICING_REQUIRED, PROVIDER_NOT_CONFIGURED
from gateway.models.usage import UsageLog

from .conftest import build_test_client

_MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
_USER = "unpriced-user"


@pytest.fixture(autouse=True)
def _no_provider_keys_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The unconfigured providers below must stay unconfigured whatever the shell holds."""
    for name in ("COHERE_API_KEY", "CO_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def readme_client(postgres_url: str) -> Generator[TestClient]:
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=True,
        default_pricing=True,
        model_discovery=False,
        providers={"openai": {"api_key": "sk-openai"}},
    )
    for client in build_test_client(config):
        assert client.post(f"{API_ROOT}/users", json={"user_id": _USER}, headers=_MASTER).status_code == 200
        yield client


def _assert_typo_402(response: Any, model: str) -> str:
    assert response.status_code == 402, response.text
    assert response.headers[ERROR_CODE_HEADER] == PRICING_REQUIRED
    assert response.json()["code"] == PRICING_REQUIRED
    detail = str(response.json()["detail"])
    assert f"'{model}'" in detail
    assert "Default pricing is on" in detail
    assert "typo" in detail
    assert f"GET {API_ROOT}/models" in detail
    assert "enable the default-pricing fallback" not in detail
    return detail


def _assert_provider_not_configured(response: Any, provider: str) -> None:
    assert response.status_code == 424, response.text
    assert response.headers[ERROR_CODE_HEADER] == PROVIDER_NOT_CONFIGURED
    body = response.json()
    assert body["code"] == PROVIDER_NOT_CONFIGURED
    assert f"'{provider}'" in body["detail"]
    assert "pricing" not in body["detail"]


def _chat(client: TestClient, model: str) -> Any:
    return client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}], "user": _USER},
        headers=_MASTER,
    )


def test_chat_typo_on_a_configured_provider_points_at_the_name(readme_client: TestClient, db_session: Session) -> None:
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock()) as dispatch:
        response = _chat(readme_client, "openai:gpt5-mini")
    detail = _assert_typo_402(response, "openai:gpt5-mini")
    dispatch.assert_not_awaited()
    row = db_session.query(UsageLog).filter(UsageLog.user_id == _USER).one()
    assert (row.status_code, row.error_message) == (402, detail)


def test_chat_on_an_unconfigured_provider_names_the_provider(readme_client: TestClient, db_session: Session) -> None:
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock()) as dispatch:
        response = _chat(readme_client, "cohere:command-typo")
    _assert_provider_not_configured(response, "cohere")
    dispatch.assert_not_awaited()
    # The hold taken at admission is released, and the refusal is recorded as what it is.
    user = readme_client.get(f"{API_ROOT}/users/{_USER}", headers=_MASTER).json()
    assert float(user["reserved"]) == 0.0
    row = db_session.query(UsageLog).filter(UsageLog.user_id == _USER).one()
    assert row.status_code == 424


def test_embeddings_typo_on_a_configured_provider_points_at_the_name(readme_client: TestClient) -> None:
    response = readme_client.post(
        f"{API_ROOT}/embeddings",
        json={"model": "openai:text-embeding-3-small", "input": "hello", "user": _USER},
        headers=_MASTER,
    )
    _assert_typo_402(response, "openai:text-embeding-3-small")


def test_rerank_on_an_unconfigured_provider_names_the_provider(readme_client: TestClient) -> None:
    response = readme_client.post(
        f"{API_ROOT}/rerank",
        json={"model": "cohere:rerank-v3.5", "query": "q", "documents": ["a", "b"], "user": _USER},
        headers=_MASTER,
    )
    _assert_provider_not_configured(response, "cohere")
