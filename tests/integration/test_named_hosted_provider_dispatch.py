"""A hosted provider named otherwise than its implementation serves ``name:model``, end to end.

``openai-eu`` is an OpenAI endpoint of the deployment's own: a request for
``openai-eu:gpt-4o`` is dispatched to its endpoint on its key, and settles and
lists under its name, while ``openai:gpt-4o`` stays the plain OpenAI one. Model
discovery is stubbed; the provider call is captured at the any-llm boundary.
"""

from collections.abc import Callable, Iterator
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from any_llm.types.model import Model
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.settings.providers import parse_hosted_providers
from gateway.models.usage import UsageLog
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.providers import hosted_implementation
from gateway.services.secret_box import generate_secret_key

from .conftest import shared_test_config

EU_KEY = "sk-live-eu-1234"
US_KEY = "sk-live-us-5678"


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    """The shared configuration declaring no provider, so nothing configured shadows a hosted one."""
    return shared_test_config(postgres_url, {})


@pytest.fixture(autouse=True)
def _environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture(autouse=True)
def _discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id="gpt-4o", object="model", created=0, owned_by=impl_name)]
        )

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _stub)


def _setup(client: TestClient, headers: dict[str, str]) -> None:
    assert client.post(f"{API_ROOT}/users", json={"user_id": "u1", "alias": "u1"}, headers=headers).status_code == 200
    for model_key in ("openai:gpt-4o", "openai-eu:gpt-4o"):
        priced = client.post(
            f"{API_ROOT}/pricing",
            json={"model_key": model_key, "input_price_per_million": 2.5, "output_price_per_million": 10},
            headers=headers,
        )
        assert priced.status_code == 200, priced.text
    for body in (
        {"provider": "openai", "api_key": US_KEY},
        {"provider": "openai", "name": "openai-eu", "api_key": EU_KEY, "api_base": "https://eu.test/v1"},
    ):
        created = client.post(f"{API_ROOT}/hosted-providers", json=body, headers=headers)
        assert created.status_code == 201, created.text


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-eu",
        object="chat.completion",
        created=0,
        model="gpt-4o",
        choices=[Choice(index=0, finish_reason="stop", message=ChatCompletionMessage(role="assistant", content="Hi"))],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _chat(client: TestClient, headers: dict[str, str], model: str) -> tuple[dict[str, object], int]:
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> ChatCompletion:
        captured.update(kwargs)
        return _completion()

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "Hi"}], "user": "u1"},
            headers=headers,
        )
    return captured, response.status_code


def test_a_named_hosted_provider_is_served_on_its_own_endpoint_and_key(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _setup(client, master_key_header)

    eu, eu_status = _chat(client, master_key_header, "openai-eu:gpt-4o")
    us, us_status = _chat(client, master_key_header, "openai:gpt-4o")

    assert (eu_status, us_status) == (200, 200)
    assert (eu["api_key"], eu["api_base"]) == (EU_KEY, "https://eu.test/v1")
    # any-llm is called as the implementation, so it can build the client.
    assert eu["model"] == "openai:gpt-4o"
    assert us["api_key"] == US_KEY
    assert "api_base" not in us
    session = db_session_factory()
    try:
        rows = session.execute(select(UsageLog.provider, UsageLog.deployment_paid, UsageLog.cost)).all()
    finally:
        session.close()
    providers = {provider: (paid, cost) for provider, paid, cost in rows}
    assert providers["openai-eu"][0] is True
    assert providers["openai-eu"][1] is not None


def test_a_named_hosted_provider_is_listed_under_its_name(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str]
) -> None:
    _setup(client, master_key_header)

    listed = client.get(f"{API_ROOT}/models", headers=api_key_header)

    assert listed.status_code == 200, listed.text
    ids = {model["id"] for model in listed.json()["data"]}
    assert "openai/gpt-4o" in ids
    detail = client.get(f"{API_ROOT}/catalog/models/openai/gpt-4o", headers=api_key_header)
    assert detail.status_code == 200, detail.text
    assert {"openai-eu:gpt-4o", "openai:gpt-4o"} <= {offering["selector"] for offering in detail.json()["offerings"]}


def test_a_name_nothing_defines_is_still_unknown(client: TestClient, master_key_header: dict[str, str]) -> None:
    _setup(client, master_key_header)

    _, status = _chat(client, master_key_header, "openai-us:gpt-4o")

    assert status == 400


def test_deleting_a_row_that_overrides_a_configured_name_restores_the_configured_implementation(
    test_config: GatewayConfig, client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The worker that deleted the row maps the name back to what configuration serves it as."""
    entries = {"openai-eu": {"provider": "openai", "api_key": EU_KEY}}
    with patch.object(type(test_config), "hosted_provider_entries", lambda _self: parse_hosted_providers(entries)):
        created = client.post(
            f"{API_ROOT}/hosted-providers",
            json={"provider": "mistral", "name": "openai-eu", "api_key": US_KEY},
            headers=master_key_header,
        )
        assert created.status_code == 201, created.text
        assert hosted_implementation("openai-eu") == "mistral"

        deleted = client.delete(f"{API_ROOT}/hosted-providers/openai-eu", headers=master_key_header)

        assert deleted.status_code == 200, deleted.text
        assert hosted_implementation("openai-eu") == "openai"
