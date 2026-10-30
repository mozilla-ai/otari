"""Automatic routing serves a catalog ID on the caller's organization's own key first.

Through the routes: the deployment serves GLM-5.3 on its own ``fireworks``
instance, priced far below the default rate of the organization's own Nebius
key, which offers the same model. The selector index is rebuilt from the database the
way the refresher does it, so the organization's view, the ranking and the
dispatch are checked as one path.

``openai/gpt-4o`` on an organization's own OpenAI key is the other common shape,
covered at the end: there the catalog ID is also the key's legacy
``instance/model`` spelling, so the request is treated as pinned and never
ranked, and the organization's view alone puts its key first.
"""

import asyncio
from collections.abc import Awaitable, Callable, Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from any_llm.types.model import Model
from fastapi import status
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.services import catalog as selectors
from gateway.services import model_catalog_service as mcs
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy.org_provider_key_service import reset_org_provider_cache

from .conftest import build_test_client

_MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
_KEYS = f"{API_ROOT}/organizations/me/provider-keys"
_OWN_KEY = "sk-org-own-1234"
_DEPLOYMENT_KEY = "sk-deployment"
_DEPLOYMENT_RATE = 0.01

_GLM = "z-ai/glm-5.3"
_NEBIUS_MODEL = "zai-org/GLM-5.3"
_NEBIUS = f"nebius:{_NEBIUS_MODEL}"
_FIREWORKS = "fireworks:accounts/fireworks/models/glm-5p3"

_GPT = "openai/gpt-4o"
_OPENAI = "openai:gpt-4o"
_FLEET = "fleet:gpt-4o"

# Enough of models.dev to fold each pair of spellings into one model.
_MODELS_DEV: dict[str, Any] = {
    "nebius": {
        "id": "nebius",
        "name": "Nebius",
        "models": {
            _NEBIUS_MODEL: {
                "id": _NEBIUS_MODEL,
                "name": "GLM-5.3",
                "family": "glm",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 200000, "output": 128000},
            }
        },
    },
    "fireworks-ai": {
        "id": "fireworks-ai",
        "name": "Fireworks",
        "models": {
            "accounts/fireworks/models/glm-5p3": {
                "id": "accounts/fireworks/models/glm-5p3",
                "name": "GLM 5.3",
                "family": "glm",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 131072, "output": 16384},
            }
        },
    },
    "openai": {
        "id": "openai",
        "name": "OpenAI",
        "models": {
            "gpt-4o": {
                "id": "gpt-4o",
                "name": "GPT-4o",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 128000, "output": 16384},
            }
        },
    },
}

Dispatch = Callable[..., Awaitable[ChatCompletion]]


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    for name in ("OPENAI_API_KEY", "NEBIUS_API_KEY", "FIREWORKS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    reset_org_provider_cache()
    mcs.clear_catalog_cache()
    yield
    reset_org_provider_cache()
    mcs.clear_catalog_cache()
    selectors.reset_selector_index()


def _client(postgres_url: str, monkeypatch: pytest.MonkeyPatch, providers: dict[str, Any]) -> Generator[TestClient]:
    """A gateway whose index is rebuilt only when a test asks, never by the scheduled refresher."""

    async def _idle(*_args: Any, **_kwargs: Any) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr("gateway.main.run_selector_index_refresher", _idle)
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        import_configured_providers=False,
        providers=providers,
    )
    yield from build_test_client(config)


def _discover(monkeypatch: pytest.MonkeyPatch, model: str) -> None:
    """The organization's key lists ``model`` and nothing else."""

    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id=model, object="model", created=0, owned_by=impl_name)], error=None
        )

    monkeypatch.setattr("gateway.services.providers._org_provider_model_service.test_provider_credentials", _stub)


def _price(client: TestClient, model_key: str, rate: float) -> None:
    response = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": model_key, "input_price_per_million": rate, "output_price_per_million": rate},
        headers=_MASTER,
    )
    assert response.status_code in (status.HTTP_200_OK, status.HTTP_201_CREATED), response.text


def _offer_own_key(client: TestClient, provider: str, model: str) -> None:
    """Store a key for the operator's organization, offering ``model`` and serving it at its default rate."""
    created = client.post(_KEYS, json={"provider": provider, "name": "primary", "api_key": _OWN_KEY}, headers=_MASTER)
    assert created.status_code == status.HTTP_201_CREATED, created.text
    key_id = created.json()["id"]
    assert client.post(f"{_KEYS}/{key_id}/default", headers=_MASTER).status_code == status.HTTP_200_OK
    listed = client.get(f"{_KEYS}/{key_id}/models", headers=_MASTER)
    assert listed.status_code == status.HTTP_200_OK, listed.text
    [offered] = listed.json()["data"]
    assert (offered["model"], offered["enabled"], offered["price_source"]) == (model, True, "defaults")
    # Dearer than the deployment's offering, so cost alone would rank the key last.
    assert offered["input_price_per_million"] > _DEPLOYMENT_RATE


def _refresh_index(client: TestClient, catalog_id: str, deployment: str, own: str) -> None:
    """Rebuild the index and check it holds the deployment's offering, and the organization's key in its view."""
    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=_MODELS_DEV)):
        rebuilt = client.post(f"{API_ROOT}/catalog/selectors/refresh", headers=_MASTER)
    assert rebuilt.status_code == status.HTTP_200_OK, rebuilt.text
    index = selectors.current_selector_index()
    assert index.offerings[catalog_id] == (deployment,)
    [view] = index.organizations.values()
    assert view.full == frozenset({own})
    # The view already lists the key first, which is why the ranking is also tested without its tier.
    assert view.offerings[catalog_id] == (own, deployment)


def _api_key_header(client: TestClient) -> dict[str, str]:
    created = client.post(f"{API_ROOT}/keys", json={"key_name": "app"}, headers=_MASTER)
    assert created.status_code == status.HTTP_200_OK, created.text
    return {API_KEY_HEADER: f"Bearer {created.json()['key']}"}


def _other_organization_key_header(client: TestClient) -> dict[str, str]:
    """An API key in another organization's workspace, an organization with no provider key of its own."""
    created = client.post(f"{API_ROOT}/organizations", json={"name": "Other Co"}, headers=_MASTER)
    assert created.status_code == status.HTTP_201_CREATED, created.text
    switched = client.post(
        f"{API_ROOT}/organizations/me/switch", json={"organization_id": created.json()["id"]}, headers=_MASTER
    )
    assert switched.status_code == status.HTTP_200_OK, switched.text
    return _api_key_header(client)


@pytest.fixture
def glm_client(postgres_url: str, clean_database: None, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    """The deployment's Fireworks GLM at a cent per million, the organization's own Nebius GLM at its default rate."""
    _discover(monkeypatch, _NEBIUS_MODEL)
    for client in _client(postgres_url, monkeypatch, {"fireworks": {"api_key": _DEPLOYMENT_KEY}}):
        _price(client, _FIREWORKS, _DEPLOYMENT_RATE)
        _offer_own_key(client, "nebius", _NEBIUS_MODEL)
        _refresh_index(client, _GLM, _FIREWORKS, _NEBIUS)
        yield client


@pytest.fixture
def gpt_client(postgres_url: str, clean_database: None, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    """The deployment's ``fleet`` OpenAI instance at a cent per million, the organization's OpenAI key at default."""
    _discover(monkeypatch, "gpt-4o")
    providers = {"fleet": {"provider_type": "openai", "api_key": _DEPLOYMENT_KEY}}
    for client in _client(postgres_url, monkeypatch, providers):
        _price(client, _FLEET, _DEPLOYMENT_RATE)
        _offer_own_key(client, "openai", "gpt-4o")
        _refresh_index(client, _GPT, _FLEET, _OPENAI)
        yield client


def _completion(**kwargs: Any) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-own",
        object="chat.completion",
        created=0,
        model=kwargs["model"].partition(":")[2],
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


async def _serve(**kwargs: Any) -> ChatCompletion:
    return _completion(**kwargs)


async def _own_key_down(**kwargs: Any) -> ChatCompletion:
    """A 503 from the upstream behind the organization's key; the deployment's answers."""
    if kwargs["api_key"] == _OWN_KEY:
        request = httpx.Request("POST", "http://upstream")
        raise httpx.HTTPStatusError("503", request=request, response=httpx.Response(503, request=request))
    return _completion(**kwargs)


def _chat(
    client: TestClient, headers: dict[str, str], catalog_id: str, dispatch: Dispatch = _serve
) -> tuple[Any, list[tuple[str, str]]]:
    """Send the catalog ID; return the response and each provider call's ``(model, api_key)``."""
    calls: list[tuple[str, str]] = []

    async def _recorded(**kwargs: Any) -> ChatCompletion:
        calls.append((kwargs["model"], kwargs["api_key"]))
        return await dispatch(**kwargs)

    with patch("gateway.api.routes.chat.acompletion", new=_recorded):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": catalog_id, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
            headers=headers,
        )
    return response, calls


def _usage(client: TestClient) -> list[tuple[str, str, int | None]]:
    """Each usage row's status, provider and attempt position, in attempt order."""
    rows = client.get(f"{API_ROOT}/usage", headers=_MASTER).json()
    return sorted(
        ((row["status"], row["provider"], row["attempt_position"]) for row in rows), key=lambda row: row[2] or 0
    )


def test_the_own_key_serves_the_catalog_id_ahead_of_a_cheaper_deployment_offering(glm_client: TestClient) -> None:
    response, calls = _chat(glm_client, _api_key_header(glm_client), _GLM)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json()["model"] == _GLM
    assert calls == [(_NEBIUS, _OWN_KEY)]
    assert _usage(glm_client) == [("success", "nebius", 1)]


def test_without_the_own_key_tier_the_router_ranks_the_cheaper_deployment_offering_first(
    glm_client: TestClient,
) -> None:
    """The organization's view lists its key first, so this proves the ranking decides, not that order."""
    with patch("gateway.api.routes._pipeline.organization_offerings", return_value=frozenset()):
        response, calls = _chat(glm_client, _api_key_header(glm_client), _GLM)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert calls == [(_FIREWORKS, _DEPLOYMENT_KEY)]
    assert _usage(glm_client) == [("success", "fireworks", 1)]


def test_a_caller_whose_organization_has_no_key_is_served_by_the_deployment(glm_client: TestClient) -> None:
    response, calls = _chat(glm_client, _other_organization_key_header(glm_client), _GLM)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert calls == [(_FIREWORKS, _DEPLOYMENT_KEY)]


def test_a_failing_own_key_falls_back_to_the_deployment_offering(glm_client: TestClient) -> None:
    response, calls = _chat(glm_client, _api_key_header(glm_client), _GLM, _own_key_down)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert calls == [(_NEBIUS, _OWN_KEY), (_FIREWORKS, _DEPLOYMENT_KEY)]
    assert _usage(glm_client) == [("absorbed", "nebius", 1), ("success", "fireworks", 2)]


def test_an_own_key_in_cooldown_yields_to_the_deployment_offering(glm_client: TestClient) -> None:
    """The failure above cools the key down, so the next request starts on the deployment's offering."""
    headers = _api_key_header(glm_client)
    _chat(glm_client, headers, _GLM, _own_key_down)

    response, calls = _chat(glm_client, headers, _GLM)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert calls == [(_FIREWORKS, _DEPLOYMENT_KEY)]


def test_an_own_openai_key_serves_openai_gpt_4o_ahead_of_a_cheaper_deployment_instance(
    gpt_client: TestClient,
) -> None:
    """Both offerings dispatch as ``openai:gpt-4o``; the credential says which one served."""
    response, calls = _chat(gpt_client, _api_key_header(gpt_client), _GPT)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert calls == [(_OPENAI, _OWN_KEY)]
    assert _usage(gpt_client) == [("success", "openai", 1)]


def test_a_failing_own_openai_key_falls_back_to_the_deployment_instance(gpt_client: TestClient) -> None:
    response, calls = _chat(gpt_client, _api_key_header(gpt_client), _GPT, _own_key_down)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert calls == [(_OPENAI, _OWN_KEY), (_OPENAI, _DEPLOYMENT_KEY)]
    assert _usage(gpt_client) == [("absorbed", "openai", 1), ("success", "fleet", 2)]
