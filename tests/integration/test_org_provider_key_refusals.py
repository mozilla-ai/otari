"""The organization-key gate says why it turns a model away.

Three answers, pinned on the completion pipeline and on the pass-through
scaffold: a model the organization offers with its switch off is refused as not
serving (403 ``model_not_serving``), a model no key of the provider offers is not
found (404 ``model_not_found``), and a model the API key's own allow-list
excludes keeps the permission refusal (403 ``model_not_allowed``). Each refusal
lands in the usage log as the gate always did.

The overlay is primed directly, the way ``tests/unit/test_routing_compiler.py``
does, because the gate reads the cache and nothing else: the service tests in
``test_org_provider_key_models.py`` pin that a refresh fills it. The default
workspace is the one a key created with the master key bills to. ``anthropic``
names no configured instance in the test config, so a bare selector resolves
through the organization overlay.
"""

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT
from gateway.core.error_codes import ERROR_CODE_HEADER, MODEL_NOT_ALLOWED, MODEL_NOT_FOUND, MODEL_NOT_SERVING
from gateway.services.tenancy import org_provider_key_service as org_store
from gateway.services.tenancy.org_provider_key_service import KeyOffer

_MESSAGES = [{"role": "user", "content": "hi"}]


def _default_workspace_id(client: TestClient, master_key_header: dict[str, str]) -> uuid.UUID:
    rows = client.get(f"{API_ROOT}/workspaces", headers=master_key_header).json()["data"]
    assert len(rows) == 1, rows
    return uuid.UUID(rows[0]["id"])


@pytest.fixture
def narrowed_overlay(client: TestClient, master_key_header: dict[str, str]) -> Iterator[None]:
    """One anthropic key named ``primary`` serving ``claude-on`` with ``claude-off`` switched off."""
    key = (_default_workspace_id(client, master_key_header), "anthropic")
    org_store._org_cache[key] = {"api_key": "test"}
    org_store._org_model_restrictions[key] = ["claude-on"]
    org_store._org_key_offers[key] = KeyOffer(key_name="primary", offered={"claude-on": True, "claude-off": False})
    yield
    org_store._org_cache.pop(key, None)
    org_store._org_model_restrictions.pop(key, None)
    org_store._org_key_offers.pop(key, None)


def _api_key(
    client: TestClient, master_key_header: dict[str, str], allowed_models: list[str] | None = None
) -> dict[str, str]:
    body: dict[str, Any] = {"key_name": "tenant"}
    if allowed_models is not None:
        body["allowed_models"] = allowed_models
    resp = client.post(f"{API_ROOT}/keys", json=body, headers=master_key_header)
    assert resp.status_code == 200, resp.text
    return {"Otari-Key": f"Bearer {resp.json()['key']}"}


def _one_error(client: TestClient, master_key_header: dict[str, str]) -> dict[str, Any]:
    rows = client.get(f"{API_ROOT}/usage", params={"status": "error"}, headers=master_key_header).json()
    assert len(rows) == 1, rows
    return dict(rows[0])


@pytest.mark.usefixtures("narrowed_overlay")
def test_chat_refuses_a_switched_off_model_as_not_serving(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    resp = client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": "anthropic:claude-off", "messages": _MESSAGES},
        headers=_api_key(client, master_key_header),
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["code"] == MODEL_NOT_SERVING
    assert resp.headers[ERROR_CODE_HEADER] == MODEL_NOT_SERVING
    assert resp.json()["detail"] == (
        "Model 'anthropic:claude-off' is offered on provider key 'primary' but not serving. "
        "Turn it on under Organization > Providers."
    )
    row = _one_error(client, master_key_header)
    assert row["status_code"] == 403
    assert "not serving" in row["error_message"]


@pytest.mark.usefixtures("narrowed_overlay")
def test_chat_refuses_a_model_no_key_offers_as_not_found(client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": "anthropic:claude-nope", "messages": _MESSAGES},
        headers=_api_key(client, master_key_header),
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["code"] == MODEL_NOT_FOUND
    assert resp.json()["detail"] == (
        "Provider key 'primary' does not offer 'anthropic:claude-nope'. Refresh the key's models or add it."
    )
    assert _one_error(client, master_key_header)["status_code"] == 404


@pytest.mark.usefixtures("narrowed_overlay")
def test_chat_keeps_the_permission_refusal_for_the_keys_own_allowlist(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The API key's allow-list is checked first and is a permission, whatever the organization offers."""
    resp = client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": "anthropic:claude-on", "messages": _MESSAGES},
        headers=_api_key(client, master_key_header, allowed_models=["openai:*"]),
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["code"] == MODEL_NOT_ALLOWED
    assert resp.json()["detail"] == "Model 'anthropic:claude-on' is not permitted for this API key."


@pytest.mark.usefixtures("narrowed_overlay")
def test_embeddings_refuse_a_switched_off_model_as_not_serving(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    resp = client.post(
        f"{API_ROOT}/embeddings",
        json={"model": "anthropic:claude-off", "input": "hi"},
        headers=_api_key(client, master_key_header),
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["code"] == MODEL_NOT_SERVING
    assert resp.headers[ERROR_CODE_HEADER] == MODEL_NOT_SERVING
    row = _one_error(client, master_key_header)
    assert row["endpoint"] == "/v1/embeddings"
    assert row["status_code"] == 403
    assert "not serving" in row["error_message"]


@pytest.mark.usefixtures("narrowed_overlay")
def test_embeddings_refuse_a_model_no_key_offers_as_not_found(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    resp = client.post(
        f"{API_ROOT}/embeddings",
        json={"model": "anthropic:claude-nope", "input": "hi"},
        headers=_api_key(client, master_key_header),
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["code"] == MODEL_NOT_FOUND
    assert _one_error(client, master_key_header)["status_code"] == 404


@pytest.mark.usefixtures("narrowed_overlay")
def test_passthrough_allowlist_refusal_carries_its_code(client: TestClient, master_key_header: dict[str, str]) -> None:
    """The pass-through scaffold's own allow-list refusal sends ``model_not_allowed`` like the pipeline does."""
    resp = client.post(
        f"{API_ROOT}/embeddings",
        json={"model": "anthropic:claude-on", "input": "hi"},
        headers=_api_key(client, master_key_header, allowed_models=["openai:*"]),
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["code"] == MODEL_NOT_ALLOWED
    assert resp.headers[ERROR_CODE_HEADER] == MODEL_NOT_ALLOWED
