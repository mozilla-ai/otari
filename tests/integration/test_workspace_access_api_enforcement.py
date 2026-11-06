"""A workspace's narrowing holds at the API, not only on the dashboard.

An API key in a workspace calls the gateway the way an application does. What
the workspace's admins switched off or narrowed on its Providers tab (its
organization's own keys and the deployment's hosted providers) is refused, and
the provider is never called. The provider call is captured at the any-llm
boundary.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.model import Model
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.providers import reset_workspace_hosted_access_cache
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy.org_provider_key_service import reset_org_provider_cache

from .conftest import shared_test_config

ORG_KEY = "sk-org-own-5678"
HOSTED_KEY = "sk-hosted-1234"


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    """No configured provider, so only the organization's keys and the hosted providers can serve."""
    return shared_test_config(postgres_url, {})


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    reset_workspace_hosted_access_cache()
    reset_org_provider_cache()
    yield
    reset_workspace_hosted_access_cache()
    reset_org_provider_cache()


@pytest.fixture(autouse=True)
def _discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name,
            models=[Model(id="gpt-4o", object="model", created=0, owned_by=impl_name)],
        )

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _stub)


def _ok(response: Any) -> Any:
    assert response.status_code in (200, 201), response.text
    return response.json()


def _workspace_and_key(client: TestClient, master: dict[str, str]) -> tuple[str, dict[str, str]]:
    """An API key in the deployment's default workspace, and that workspace's id."""
    key = _ok(client.post(f"{API_ROOT}/keys", json={"key_name": "app"}, headers=master))
    [workspace] = _ok(client.get(f"{API_ROOT}/workspaces", headers=master))["data"]
    return workspace["id"], {API_KEY_HEADER: f"Bearer {key['key']}"}


def _price(client: TestClient, master: dict[str, str], *model_keys: str) -> None:
    for model_key in model_keys:
        _ok(
            client.post(
                f"{API_ROOT}/pricing",
                json={"model_key": model_key, "input_price_per_million": 1, "output_price_per_million": 2},
                headers=master,
            )
        )


def _org_key(client: TestClient, master: dict[str, str]) -> str:
    created = _ok(
        client.post(
            f"{API_ROOT}/organizations/me/provider-keys",
            json={"provider": "anthropic", "name": "team", "api_key": ORG_KEY},
            headers=master,
        )
    )
    return str(created["id"])


def _chat(client: TestClient, headers: dict[str, str], model: str) -> tuple[int, dict[str, object]]:
    captured: dict[str, object] = {}

    async def fake(**kwargs: object) -> None:
        captured.update(kwargs)
        raise RuntimeError("captured")

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=fake)):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "Hi"}]},
            headers=headers,
        )
    return response.status_code, captured


def test_a_model_left_out_of_the_workspace_list_on_its_own_key_is_refused(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _price(client, master_key_header, "anthropic:claude-sonnet", "anthropic:claude-opus")
    key_id = _org_key(client, master_key_header)
    workspace_id, headers = _workspace_and_key(client, master_key_header)
    _ok(
        client.post(
            f"{API_ROOT}/workspaces/{workspace_id}/provider-keys/{key_id}/models",
            json={"model": "claude-sonnet"},
            headers=master_key_header,
        )
    )

    refused_status, refused = _chat(client, headers, "anthropic:claude-opus")
    _, served = _chat(client, headers, "anthropic:claude-sonnet")

    assert refused_status == 403
    assert not refused, "a model the workspace left out must never reach the provider"
    assert served.get("api_key") == ORG_KEY


def test_a_key_switched_off_for_the_workspace_serves_nothing(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _price(client, master_key_header, "anthropic:claude-sonnet")
    key_id = _org_key(client, master_key_header)
    workspace_id, headers = _workspace_and_key(client, master_key_header)
    _ok(
        client.patch(
            f"{API_ROOT}/workspaces/{workspace_id}/provider-keys/{key_id}",
            json={"disabled": True},
            headers=master_key_header,
        )
    )

    status_code, captured = _chat(client, headers, "anthropic:claude-sonnet")

    assert status_code >= 400
    assert captured.get("api_key") != ORG_KEY, "a key the workspace switched off must not be used"


def _hosted_openai(client: TestClient, master: dict[str, str]) -> None:
    _price(client, master, "openai:gpt-4o", "openai:text-embedding-3-small")
    _ok(
        client.post(
            f"{API_ROOT}/hosted-providers",
            json={"provider": "openai", "api_key": HOSTED_KEY},
            headers=master,
        )
    )
    _ok(
        client.post(
            f"{API_ROOT}/hosted-providers/openai/models",
            json={"model": "text-embedding-3-small"},
            headers=master,
        )
    )


def test_a_hosted_provider_switched_off_refuses_a_passthrough_call(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Embeddings take the pass-through path rather than the completion pipeline; the rule holds there too."""
    _hosted_openai(client, master_key_header)
    workspace_id, headers = _workspace_and_key(client, master_key_header)
    _ok(
        client.put(
            f"{API_ROOT}/workspaces/{workspace_id}/hosted-providers/openai",
            json={"enabled": False},
            headers=master_key_header,
        )
    )

    with patch("gateway.api.routes.embeddings.aembedding", new=AsyncMock()) as mock:
        response = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hi"},
            headers=headers,
        )

    assert response.status_code == 403, response.text
    mock.assert_not_awaited()


def test_a_hosted_model_left_out_of_the_workspace_list_refuses_a_passthrough_call(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _hosted_openai(client, master_key_header)
    workspace_id, headers = _workspace_and_key(client, master_key_header)
    _ok(
        client.put(
            f"{API_ROOT}/workspaces/{workspace_id}/hosted-providers/openai",
            json={"enabled": True, "models": ["gpt-4o"]},
            headers=master_key_header,
        )
    )

    with patch("gateway.api.routes.embeddings.aembedding", new=AsyncMock()) as mock:
        response = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hi"},
            headers=headers,
        )

    assert response.status_code == 403, response.text
    mock.assert_not_awaited()


def test_the_api_keys_model_list_leaves_out_what_the_workspace_refuses(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _hosted_openai(client, master_key_header)
    workspace_id, headers = _workspace_and_key(client, master_key_header)
    _ok(
        client.put(
            f"{API_ROOT}/workspaces/{workspace_id}/hosted-providers/openai",
            json={"enabled": True, "models": ["gpt-4o"]},
            headers=master_key_header,
        )
    )

    listed = {model["id"] for model in _ok(client.get(f"{API_ROOT}/models", headers=headers))["data"]}

    assert "openai/gpt-4o" in listed
    assert not any("text-embedding-3-small" in model for model in listed)


def _require_zero_data_retention(client: TestClient, master: dict[str, str], workspace_id: str) -> None:
    policy = _ok(
        client.put(
            f"{API_ROOT}/workspaces/{workspace_id}/provider-policy",
            json={"require_zero_data_retention": True},
            headers=master,
        )
    )
    assert policy == {"require_zero_data_retention": True}


def test_requiring_zero_data_retention_refuses_a_hosted_provider_until_it_is_declared(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _hosted_openai(client, master_key_header)
    workspace_id, headers = _workspace_and_key(client, master_key_header)
    _require_zero_data_retention(client, master_key_header, workspace_id)

    refused_status, refused = _chat(client, headers, "openai:gpt-4o")
    _ok(
        client.patch(
            f"{API_ROOT}/hosted-providers/openai", json={"zero_data_retention": True}, headers=master_key_header
        )
    )
    _, served = _chat(client, headers, "openai:gpt-4o")

    assert refused_status == 403
    assert not refused, "a provider not declared zero data retention must never be called"
    assert served.get("api_key") == HOSTED_KEY


def test_requiring_zero_data_retention_refuses_an_organization_key_until_it_is_declared(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _price(client, master_key_header, "anthropic:claude-sonnet")
    key_id = _org_key(client, master_key_header)
    workspace_id, headers = _workspace_and_key(client, master_key_header)
    _require_zero_data_retention(client, master_key_header, workspace_id)

    refused_status, refused = _chat(client, headers, "anthropic:claude-sonnet")
    _ok(
        client.patch(
            f"{API_ROOT}/organizations/me/provider-keys/{key_id}",
            json={"zero_data_retention": True},
            headers=master_key_header,
        )
    )
    _, served = _chat(client, headers, "anthropic:claude-sonnet")

    assert refused_status == 403
    assert not refused
    assert served.get("api_key") == ORG_KEY


def test_the_policy_reads_back_and_starts_off(client: TestClient, master_key_header: dict[str, str]) -> None:
    workspace_id, _ = _workspace_and_key(client, master_key_header)

    policy = _ok(client.get(f"{API_ROOT}/workspaces/{workspace_id}/provider-policy", headers=master_key_header))

    assert policy == {"require_zero_data_retention": False}


def test_an_alias_to_a_hosted_model_the_workspace_refuses_is_not_listed(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The workspace's rule reaches an alias through its target, as dispatch reaches it."""
    _hosted_openai(client, master_key_header)
    workspace_id, headers = _workspace_and_key(client, master_key_header)
    _ok(client.post(f"{API_ROOT}/aliases", json={"name": "fast", "target": "openai:gpt-4o"}, headers=master_key_header))
    url = f"{API_ROOT}/workspaces/{workspace_id}/hosted-providers/openai"

    def listed() -> set[str]:
        return {model["id"] for model in _ok(client.get(f"{API_ROOT}/models", headers=headers))["data"]}

    assert "fast" in listed()
    _ok(client.put(url, json={"enabled": False}, headers=master_key_header))
    assert "fast" not in listed()
    _ok(client.delete(url, headers=master_key_header))
    assert "fast" in listed()
