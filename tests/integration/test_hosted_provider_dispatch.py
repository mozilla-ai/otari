"""The core ``ModelProviderPort`` adapter serves a request on a hosted provider, end to end.

The store and the service are pinned in `test_hosted_providers_service.py`; the
operator API in `test_hosted_providers_api.py`. This file pins the half that
makes them worth having: a request that brings no key of its own, for a
provider nobody configured in ``config.yml``, is dispatched on the credential
the operator stored, with the extras it stored, and a model the operator
switched off is not. Model discovery is stubbed; the provider call is captured
at the any-llm boundary and never made.
"""

from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.model import Model
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import generate_secret_key

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


def test_with_no_hosted_provider_the_candidate_goes_out_uncredentialed(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The acceptance case for a deployment that configured nothing: nothing changed."""
    _create_user(client, master_key_header)

    captured, _ = _post_chat_capture(client, master_key_header, "openai:gpt-4o")

    assert captured, "the provider call was never made"
    assert "api_key" not in captured
    assert captured["model"] == "openai:gpt-4o"


def test_a_model_switched_off_is_not_served_and_an_unlisted_one_is(
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
    assert unlisted["api_key"] == KEY


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
