"""The core ``ModelProviderPort`` adapter serves a request on a hosted provider, end to end.

The store and the service are pinned in `test_hosted_providers_service.py`; the
operator API in `test_hosted_providers_api.py`. This file pins the half that
makes them worth having: a request that brings no key of its own, for a
provider nobody configured in ``config.yml``, is dispatched on the credential
the operator stored, with the extras it stored, and a model the operator
switched off or never offered is not. Model discovery is stubbed; the provider call is captured
at the any-llm boundary and never made.
"""

from collections.abc import Callable, Iterator
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from any_llm.types.model import Model
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.models.usage import UsageLog
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client, shared_test_config


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    """The shared configuration declaring no provider, so nothing configured shadows a hosted one."""
    return shared_test_config(postgres_url, {})


KEY = "sk-live-hosted-1234"


class _MockCompletionError(Exception):
    """Raised to short-circuit the mocked provider call after capturing its kwargs."""


@pytest.fixture(autouse=True)
def _no_ambient_provider_keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A developer's own shell key would credential the ladder and skip the port."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture(autouse=True)
def _discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name,
            models=[
                Model(id=name, object="model", created=0, owned_by=impl_name) for name in ("gpt-4o", "gpt-4o-mini")
            ],
        )

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _stub)


def _create_user(client: TestClient, headers: dict[str, str]) -> None:
    assert client.post(f"{API_ROOT}/users", json={"user_id": "u1", "alias": "u1"}, headers=headers).status_code == 200


def _price(client: TestClient, headers: dict[str, str], model_key: str) -> None:
    """Put a model on the deployment price list, which is what switches an offered model on.

    The test deployment consults no community dataset, so a model nothing
    prices arrives switched off, as the offer rule says it must.
    """
    response = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": model_key, "input_price_per_million": 2.5, "output_price_per_million": 10},
        headers=headers,
    )
    assert response.status_code == 200, response.text


def _hosted_openai(client: TestClient, headers: dict[str, str]) -> None:
    response = client.post(
        f"{API_ROOT}/hosted-providers",
        json={
            "provider": "openai",
            "api_key": KEY,
            "api_base": "https://hosted.test/v1",
            "client_args": {"timeout": 9},
        },
        headers=headers,
    )
    assert response.status_code == 201, response.text


def _post_chat_capture(client: TestClient, headers: dict[str, str], model: str) -> tuple[dict[str, object], int]:
    """POST a completion with the provider call mocked; return its kwargs and the status."""
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> None:
        captured.update(kwargs)
        raise _MockCompletionError

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "Hi"}], "user": "u1"},
            headers=headers,
        )
    return captured, response.status_code


def test_a_hosted_provider_serves_a_candidate_nothing_else_credentials(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _create_user(client, master_key_header)
    _price(client, master_key_header, "openai:gpt-4o")
    _hosted_openai(client, master_key_header)

    captured, _ = _post_chat_capture(client, master_key_header, "openai:gpt-4o")

    assert captured, "the provider call was never made"
    assert captured["api_key"] == KEY
    assert captured["api_base"] == "https://hosted.test/v1"
    assert captured["client_args"] == {"timeout": 9}
    # Served under the name asked for, so the pricing key does not move.
    assert captured["model"] == "openai:gpt-4o"


def test_a_request_a_hosted_provider_served_is_recorded_as_deployment_paid(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """The deployment settles the upstream bill for it, so its usage row says so."""
    _create_user(client, master_key_header)
    _price(client, master_key_header, "openai:gpt-4o")
    _hosted_openai(client, master_key_header)
    completion = ChatCompletion(
        id="chatcmpl-hosted",
        object="chat.completion",
        created=0,
        model="gpt-4o",
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=completion)):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "openai:gpt-4o", "messages": [{"role": "user", "content": "Hi"}], "user": "u1"},
            headers=master_key_header,
        )

    assert response.status_code == 200, response.text
    with db_session_factory() as db:
        flags = list(db.scalars(select(UsageLog.deployment_paid).where(UsageLog.user_id == "u1")))
    assert flags == [True]


def test_with_no_hosted_provider_an_undeclared_candidate_is_refused(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """A deployment that configured nothing serves nothing, rather than sending the call out uncredentialed."""
    _create_user(client, master_key_header)

    captured, status_code = _post_chat_capture(client, master_key_header, "openai:gpt-4o")

    assert status_code == 424
    assert not captured, "an undeclared provider must never be called"


def test_neither_a_model_switched_off_nor_one_the_roster_never_listed_is_served(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _create_user(client, master_key_header)
    _hosted_openai(client, master_key_header)
    models = client.get(f"{API_ROOT}/hosted-providers/openai/models", headers=master_key_header).json()["data"]
    [mini] = [row for row in models if row["model"] == "gpt-4o-mini"]
    assert (
        client.patch(
            f"{API_ROOT}/hosted-providers/openai/models/{mini['id']}",
            json={"enabled": False},
            headers=master_key_header,
        ).status_code
        == 200
    )

    refused, _ = _post_chat_capture(client, master_key_header, "openai:gpt-4o-mini")
    unlisted, _ = _post_chat_capture(client, master_key_header, "openai:o3")

    assert "api_key" not in refused
    assert "api_key" not in unlisted


def test_a_provider_switched_off_stops_serving(client: TestClient, master_key_header: dict[str, str]) -> None:
    _create_user(client, master_key_header)
    _hosted_openai(client, master_key_header)
    assert (
        client.patch(
            f"{API_ROOT}/hosted-providers/openai", json={"enabled": False}, headers=master_key_header
        ).status_code
        == 200
    )

    captured, _ = _post_chat_capture(client, master_key_header, "openai:gpt-4o")

    assert "api_key" not in captured


def test_the_catalog_lists_what_the_hosted_provider_advertises(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The port's other half: the roster the operator switched on reaches the Models page."""
    _price(client, master_key_header, "openai:gpt-4o")
    _price(client, master_key_header, "openai:gpt-4o-mini")
    _hosted_openai(client, master_key_header)
    models = client.get(f"{API_ROOT}/hosted-providers/openai/models", headers=master_key_header).json()["data"]
    [mini] = [row for row in models if row["model"] == "gpt-4o-mini"]
    client.patch(
        f"{API_ROOT}/hosted-providers/openai/models/{mini['id']}", json={"enabled": False}, headers=master_key_header
    )

    listed = client.get(f"{API_ROOT}/models", headers=master_key_header)

    assert listed.status_code == 200, listed.text
    ids = {model["id"] for model in listed.json()["data"]}
    assert "openai:gpt-4o" in ids
    assert "openai:gpt-4o-mini" not in ids


def test_a_model_not_switched_on_is_refused_as_not_offered_rather_than_unconfigured(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The refusal neither points at an environment variable nor names anything of the deployment's."""
    _create_user(client, master_key_header)
    _hosted_openai(client, master_key_header)
    models = client.get(f"{API_ROOT}/hosted-providers/openai/models", headers=master_key_header).json()["data"]
    [mini] = [row for row in models if row["model"] == "gpt-4o-mini"]
    client.patch(
        f"{API_ROOT}/hosted-providers/openai/models/{mini['id']}", json={"enabled": False}, headers=master_key_header
    )

    # Switched off, and never on the roster at all.
    for model in ("openai:gpt-4o-mini", "openai:o3"):
        with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_MockCompletionError)) as called:
            response = client.post(
                f"{API_ROOT}/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": "Hi"}], "user": "u1"},
                headers=master_key_header,
            )

        assert response.status_code == 424, response.text
        assert response.headers["Otari-Error-Code"] == "provider_not_configured"
        assert response.json()["detail"] == f"Model '{model}' is not offered on this deployment."
        assert "OPENAI_API_KEY" not in response.text
        assert KEY not in response.text
        called.assert_not_called()


@pytest.fixture
def configured_openai_client(postgres_url: str) -> Iterator[TestClient]:
    """A deployment whose ``config.yml`` defines an ``openai`` instance, beside a hosted openai."""
    config = GatewayConfig(
        import_configured_providers=False,
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        providers={"openai": {"api_key": "sk-config-instance"}},
    )
    yield from build_test_client(config)


def test_a_configured_instance_of_the_same_name_never_reaches_the_hosted_key(
    configured_openai_client: TestClient,
) -> None:
    """Rates kept under the instance name serve on the instance's own key, so they cannot open the hosted one."""
    headers = {API_KEY_HEADER: "Bearer test-master-key"}
    client = configured_openai_client
    _create_user(client, headers)
    _hosted_openai(client, headers)
    _price(client, headers, "openai:o3")

    for model in ("openai:gpt-4o", "openai:o3", "openai:not-on-the-roster"):
        captured, _ = _post_chat_capture(client, headers, model)
        assert captured["api_key"] == "sk-config-instance", model
