"""A hosted provider's own ``supported_countries``, checked when a request is served on its credential."""

from collections.abc import Generator, Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.model import Model
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT
from gateway.core.error_codes import COUNTRY_NOT_SUPPORTED, ERROR_CODE_HEADER
from gateway.models.routing import RoutingConfig
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client, shared_test_config
from .test_batches_endpoint import _SupportsBatch
from .test_routing_policies import _awaited_model, _completion

HEADERS = {API_KEY_HEADER: "Bearer test-master-key"}
_COUNTRY_HEADER = "CF-IPCountry"


@pytest.fixture(autouse=True)
def _environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A shell key would credential the ladder and skip the hosted rung."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        names = ("gpt-4o", "text-embedding-3-small")
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id=name, object="model", created=0, owned_by=impl_name) for name in names]
        )

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _stub)
    yield


@pytest.fixture
def client(postgres_url: str) -> Generator[TestClient]:
    config = shared_test_config(postgres_url, {}).model_copy(
        update={
            "client_country_header": _COUNTRY_HEADER,
            "routing": RoutingConfig.model_validate(
                {"policies": {"fast": {"select": [{"default": "openai:gpt-4o"}], "on_failure": ["openai-eu:gpt-4o"]}}}
            ),
        }
    )
    for client in build_test_client(config):
        assert client.post(f"{API_ROOT}/users", json={"user_id": "u1"}, headers=HEADERS).status_code == 200
        # Priced first: a hosted provider switches on only the models something prices.
        for model_key in ("openai:gpt-4o", "openai:text-embedding-3-small", "openai-eu:gpt-4o"):
            response = client.post(
                f"{API_ROOT}/pricing",
                json={"model_key": model_key, "input_price_per_million": 2.5, "output_price_per_million": 10},
                headers=HEADERS,
            )
            assert response.status_code == 200, response.text
        _hosted(client, "openai", supported_countries=["US"])
        _hosted(client, "openai-eu", provider="openai")
        yield client


def _hosted(client: TestClient, name: str, *, provider: str | None = None, **fields: Any) -> None:
    response = client.post(
        f"{API_ROOT}/hosted-providers",
        json={"provider": provider or name, "name": name, "api_key": f"sk-{name}", **fields},
        headers=HEADERS,
    )
    assert response.status_code == 201, response.text


def _chat(client: TestClient, model: str, country: str) -> Any:
    return client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}], "user": "u1"},
        headers={**HEADERS, _COUNTRY_HEADER: country},
    )


def test_a_listed_country_is_served_on_the_hosted_credential(client: TestClient) -> None:
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("gpt-4o"))) as mock:
        resp = _chat(client, "openai:gpt-4o", "US")

    assert resp.status_code == 200, resp.text
    assert mock.await_args is not None
    assert mock.await_args.kwargs["api_key"] == "sk-openai"


def test_another_country_is_refused_on_the_hosted_credential(client: TestClient) -> None:
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock()) as mock:
        resp = _chat(client, "openai:gpt-4o", "FR")

    assert resp.status_code == 403, resp.text
    assert resp.headers[ERROR_CODE_HEADER] == COUNTRY_NOT_SUPPORTED
    mock.assert_not_awaited()
    rows = client.get(f"{API_ROOT}/usage", params={"status": "error"}, headers=HEADERS).json()
    assert [(row["provider"], row["status_code"]) for row in rows] == [("openai", 403)]


def test_a_policy_falls_over_to_a_hosted_provider_that_serves_the_country(client: TestClient) -> None:
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("gpt-4o"))) as mock:
        resp = _chat(client, "fast", "FR")

    assert resp.status_code == 200, resp.text
    assert _awaited_model(mock) == "openai:gpt-4o"
    assert mock.await_args is not None
    assert mock.await_args.kwargs["api_key"] == "sk-openai-eu"


@pytest.mark.parametrize(("country", "status"), [("US", 200), ("FR", 403)])
def test_an_attachment_is_uploaded_on_the_hosted_credential_only_for_a_listed_country(
    client: TestClient, country: str, status: int
) -> None:
    credentials: list[Any] = []

    async def _normalize(messages: Any, **kwargs: Any) -> tuple[Any, SimpleNamespace]:
        # What moving an attachment past the provider's inline limit asks for first.
        credentials.append(await kwargs["hosted_credential"]())
        return messages, SimpleNamespace(sandbox_inputs=[], vision_usage=lambda: None)

    with (
        patch("gateway.api.routes.chat.normalize_request_messages", new=_normalize),
        patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("gpt-4o"))),
    ):
        resp = _chat(client, "openai:gpt-4o", country)

    assert resp.status_code == status, resp.text
    if status == 403:
        assert resp.headers[ERROR_CODE_HEADER] == COUNTRY_NOT_SUPPORTED
        assert credentials == []
    else:
        assert [credential.api_key for credential in credentials] == ["sk-openai"]


def test_a_passthrough_call_is_refused_on_the_hosted_credential(client: TestClient) -> None:
    with patch("gateway.api.routes.embeddings.aembedding", new=AsyncMock()) as mock:
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hi", "user": "u1"},
            headers={**HEADERS, _COUNTRY_HEADER: "FR"},
        )

    assert resp.status_code == 403, resp.text
    assert resp.headers[ERROR_CODE_HEADER] == COUNTRY_NOT_SUPPORTED
    mock.assert_not_awaited()


def test_a_batch_is_refused_on_the_hosted_credential(client: TestClient) -> None:
    with (
        patch("gateway.api.routes.batches.acreate_batch", new=AsyncMock()) as mock,
        patch("gateway.api.routes.batches.AnyLLM.get_provider_class", return_value=_SupportsBatch),
    ):
        resp = client.post(
            f"{API_ROOT}/batches",
            json={
                "model": "openai:gpt-4o",
                "user": "u1",
                "requests": [{"custom_id": "req-1", "body": {"messages": [{"role": "user", "content": "hi"}]}}],
            },
            headers={**HEADERS, _COUNTRY_HEADER: "FR"},
        )

    assert resp.status_code == 403, resp.text
    assert resp.headers[ERROR_CODE_HEADER] == COUNTRY_NOT_SUPPORTED
    mock.assert_not_awaited()
