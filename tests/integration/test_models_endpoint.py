"""Tests for the GET /api/v1/models endpoint."""

from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.services import model_catalog_service as mcs

from .conftest import build_test_client


def test_list_models_empty(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """GET /api/v1/models returns empty list when no pricing is configured."""
    resp = client.get(f"{API_ROOT}/models", headers=master_key_header)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert data["data"] == []


def test_list_models_returns_configured_models(
    client: TestClient,
    master_key_header: dict[str, str],
    model_pricing: dict[str, Any],
) -> None:
    """GET /api/v1/models returns models from the pricing table."""
    resp = client.get(f"{API_ROOT}/models", headers=master_key_header)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1

    model = data["data"][0]
    assert model["object"] == "model"
    assert "id" in model
    assert "created" in model
    assert isinstance(model["created"], int)
    assert "owned_by" in model


def test_list_models_deduplicates_effective_entries(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Only the latest effective pricing per model is returned."""

    model_key = "openai:gpt-4o"
    older = datetime(2025, 1, 1, tzinfo=UTC)
    newer = datetime(2025, 2, 1, tzinfo=UTC)

    for effective_at, input_price in ((older, 1.0), (newer, 2.0)):
        resp = client.post(
            f"{API_ROOT}/pricing",
            json={
                "model_key": model_key,
                "input_price_per_million": input_price,
                "output_price_per_million": 3.0,
                "effective_at": effective_at.isoformat(),
            },
            headers=master_key_header,
        )
        assert resp.status_code == 200

    resp = client.get(f"{API_ROOT}/models", headers=master_key_header)
    assert resp.status_code == 200
    models = [m for m in resp.json()["data"] if m["id"] == model_key]
    assert len(models) == 1


def test_list_models_owned_by_from_provider(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Models owned_by field is derived from the provider prefix."""
    client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "openai:gpt-4o",
            "input_price_per_million": 2.5,
            "output_price_per_million": 10.0,
        },
        headers=master_key_header,
    )

    resp = client.get(f"{API_ROOT}/models", headers=master_key_header)
    assert resp.status_code == 200
    models = resp.json()["data"]
    gpt4 = next(m for m in models if m["id"] == "openai:gpt-4o")
    assert gpt4["owned_by"] == "openai"


def test_list_models_exposes_cache_pricing(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Cache read/write rates set via /api/v1/pricing surface on the model catalog."""
    client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "anthropic:claude-sonnet-4",
            "input_price_per_million": 3.0,
            "output_price_per_million": 15.0,
            "cache_read_price_per_million": 0.3,
            "cache_write_price_per_million": 3.75,
        },
        headers=master_key_header,
    )

    resp = client.get(f"{API_ROOT}/models", headers=master_key_header)
    assert resp.status_code == 200
    model = next(m for m in resp.json()["data"] if m["id"] == "anthropic:claude-sonnet-4")
    assert model["pricing"]["cache_read_price_per_million"] == 0.3
    assert model["pricing"]["cache_write_price_per_million"] == 3.75

    # A model priced without cache rates reports them as null, not absent.
    client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": "openai:gpt-4o", "input_price_per_million": 2.5, "output_price_per_million": 10.0},
        headers=master_key_header,
    )
    resp = client.get(f"{API_ROOT}/models", headers=master_key_header)
    gpt4 = next(m for m in resp.json()["data"] if m["id"] == "openai:gpt-4o")
    assert gpt4["pricing"]["cache_read_price_per_million"] is None
    assert gpt4["pricing"]["cache_write_price_per_million"] is None


def test_get_model_found(
    client: TestClient,
    master_key_header: dict[str, str],
    model_pricing: dict[str, Any],
) -> None:
    """GET /api/v1/models/{model_id} returns the model when it exists."""
    model_key = model_pricing["model_key"]
    resp = client.get(f"{API_ROOT}/models/{model_key}", headers=master_key_header)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == model_key
    assert data["object"] == "model"
    assert isinstance(data["created"], int)
    assert data["owned_by"] == model_key.split(":")[0]


def test_get_model_not_found(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """GET /api/v1/models/{model_id} returns 404 for unknown models."""
    resp = client.get(f"{API_ROOT}/models/nonexistent:model", headers=master_key_header)
    assert resp.status_code == 404


def test_list_models_requires_auth(client: TestClient) -> None:
    """GET /api/v1/models requires authentication."""
    resp = client.get(f"{API_ROOT}/models")
    assert resp.status_code == 401


def test_get_model_requires_auth(client: TestClient) -> None:
    """GET /api/v1/models/{model_id} requires authentication."""
    resp = client.get(f"{API_ROOT}/models/openai:gpt-4o")
    assert resp.status_code == 401


def test_list_models_with_api_key(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """GET /api/v1/models works with API key authentication (not just master key)."""
    resp = client.get(f"{API_ROOT}/models", headers=api_key_header)
    assert resp.status_code == 200
    assert resp.json()["object"] == "list"


def test_list_models_sorted_by_key(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """GET /api/v1/models returns models sorted by model_key."""
    for model_key in ["openai:gpt-4o", "anthropic:claude-3-haiku"]:
        client.post(
            f"{API_ROOT}/pricing",
            json={
                "model_key": model_key,
                "input_price_per_million": 1.0,
                "output_price_per_million": 1.0,
            },
            headers=master_key_header,
        )

    resp = client.get(f"{API_ROOT}/models", headers=master_key_header)
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert ids == sorted(ids)


@pytest.fixture
def instance_config(postgres_url: str) -> GatewayConfig:
    """A deployment with one provider instance of its own.

    ``home_lab`` is a configured instance, so a key addressed through it
    dispatches on this deployment's credential; ``openai`` is not, so that prefix
    stays a bare provider name resolved against whatever BYO key an organization
    supplies.
    """
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        providers={"home_lab": {"provider_type": "openai", "api_base": "https://box.ts.net/v1", "api_key": "t"}},
    )


@pytest.fixture
def instance_client(instance_config: GatewayConfig, clean_database: None) -> Generator[TestClient]:
    yield from build_test_client(instance_config)


def _price(client: TestClient, model_key: str) -> None:
    resp = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": model_key, "input_price_per_million": 1.0, "output_price_per_million": 2.0},
        headers={API_KEY_HEADER: "Bearer test-master-key"},
    )
    assert resp.status_code == 200, resp.text


def test_the_catalog_says_which_models_the_deployment_supplies(instance_client: TestClient) -> None:
    """``deployment_managed`` is what lets the dashboard withhold a rate override.

    An organization may set its own rate for a model it supplies the provider key
    for, and not for one the deployment holds the credential to, so the catalog
    has to distinguish them rather than leave a client guessing from the prefix.
    """
    headers = {API_KEY_HEADER: "Bearer test-master-key"}
    _price(instance_client, "home_lab:llama-3")
    _price(instance_client, "openai:gpt-4o")

    listed = instance_client.get(f"{API_ROOT}/models", headers=headers)
    assert listed.status_code == 200, listed.text
    managed = {model["id"]: model["deployment_managed"] for model in listed.json()["data"]}

    assert managed["home_lab:llama-3"] is True
    assert managed["openai:gpt-4o"] is False

    one = instance_client.get(f"{API_ROOT}/models/home_lab:llama-3", headers=headers)
    assert one.status_code == 200, one.text
    assert one.json()["deployment_managed"] is True


_QWEN = "nebius:Qwen/Qwen3-30B-A3B-Instruct-2507"
_GEMINI = "gemini:models/gemini-3.5-flash-lite"
_UNKNOWN = "nebius:made-up-model-xyz"

# models.dev as the listing reads it: Gemini under ``google`` and keyed by the
# bare id, where the gateway's selector carries Gemini's own ``models/`` prefix.
_MODELS_DEV: dict[str, Any] = {
    "nebius": {
        "id": "nebius",
        "name": "Nebius",
        "models": {
            "Qwen/Qwen3-30B-A3B-Instruct-2507": {
                "id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "name": "Qwen3 30B A3B Instruct 2507",
                "limit": {"context": 262144, "output": 8192},
            },
        },
    },
    "google": {
        "id": "google",
        "name": "Google",
        "models": {
            "gemini-3.5-flash-lite": {
                "id": "gemini-3.5-flash-lite",
                "name": "Gemini 3.5 Flash Lite",
                "limit": {"context": 1048576, "output": 65536},
            },
        },
    },
}


@pytest.fixture
def limits_config(postgres_url: str) -> GatewayConfig:
    """Nebius, with a models.dev-only output limit; Gemini, whose ids carry ``models/``; and an alias to Gemini."""
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        providers={"nebius": {"api_key": "sk-fake"}, "gemini": {"api_key": "sk-fake"}},
        aliases={"house-gemini": _GEMINI},
    )


@pytest.fixture
def limits_client(limits_config: GatewayConfig, clean_database: None) -> Generator[TestClient]:
    """Each model gets a price row, which is what lists it while discovery is off."""
    for client in build_test_client(limits_config):
        for key in (_QWEN, _GEMINI, _UNKNOWN, "openai:gpt-4o"):
            _price(client, key)
        yield client


def _read(client: TestClient, path: str) -> Any:
    """GET ``path`` as the master key, with models.dev answering ``_MODELS_DEV``."""
    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=_MODELS_DEV)):
        response = client.get(path, headers={API_KEY_HEADER: "Bearer test-master-key"})
    assert response.status_code == 200, response.text
    return response.json()


def _limits(model: dict[str, Any]) -> tuple[int | None, int | None]:
    return model["context_window"], model["max_output_tokens"]


def test_the_listing_carries_each_models_token_limits(limits_client: TestClient) -> None:
    """The issue's reproduction: a Nebius model reports both limits."""
    listed = {model["id"]: _limits(model) for model in _read(limits_client, f"{API_ROOT}/models")["data"]}

    assert listed[_QWEN] == (262144, 8192)
    assert listed[_UNKNOWN] == (None, None)
    # An alias reports its target's limits; the listing still withholds the target itself.
    assert listed["house-gemini"] == (1048576, 65536)
    assert _GEMINI not in listed

    assert _limits(_read(limits_client, f"{API_ROOT}/models/{_QWEN}")) == (262144, 8192)
    assert _limits(_read(limits_client, f"{API_ROOT}/models/{_GEMINI}")) == (1048576, 65536)
    assert _limits(_read(limits_client, f"{API_ROOT}/models/house-gemini")) == (1048576, 65536)
    assert _limits(_read(limits_client, f"{API_ROOT}/models/{_UNKNOWN}")) == (None, None)


def test_the_listing_and_the_catalog_report_the_same_limits(limits_client: TestClient) -> None:
    """Both surfaces read one merged catalog, so no selector can carry two answers."""
    listed = {model["id"]: _limits(model) for model in _read(limits_client, f"{API_ROOT}/models")["data"]}
    offered: dict[str, tuple[int | None, int | None]] = {}
    for summary in _read(limits_client, f"{API_ROOT}/catalog/models")["models"]:
        detail = _read(limits_client, f"{API_ROOT}/catalog/models/{summary['id']}")
        offered.update({offering["selector"]: _limits(offering) for offering in detail["offerings"]})

    shared = listed.keys() & offered.keys()
    assert {_QWEN, _UNKNOWN, "openai:gpt-4o"} <= shared
    assert {selector: listed[selector] for selector in shared} == {selector: offered[selector] for selector in shared}


def test_without_models_dev_the_window_comes_from_the_snapshot(limits_client: TestClient) -> None:
    """models.dev unreachable: the pricing snapshot still answers the window, and nothing lists an output limit."""
    response = limits_client.get(f"{API_ROOT}/models", headers={API_KEY_HEADER: "Bearer test-master-key"})
    assert response.status_code == 200, response.text
    listed = {model["id"]: _limits(model) for model in response.json()["data"]}

    assert listed["openai:gpt-4o"] == (128000, None)
    assert listed[_QWEN] == (262144, None)
    assert listed[_UNKNOWN] == (None, None)


def test_disabled_models_dev_is_never_fetched_by_the_listing(postgres_url: str, clean_database: None) -> None:
    """Switched off, models.dev is not fetched at all, so only the snapshot's window is listed."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        models_dev_metadata=False,
        providers={"nebius": {"api_key": "sk-fake"}},
    )
    fetch = AsyncMock(return_value=_MODELS_DEV)
    for client in build_test_client(config):
        _price(client, _QWEN)
        with patch.object(mcs, "_fetch", new=fetch):
            response = client.get(f"{API_ROOT}/models", headers={API_KEY_HEADER: "Bearer test-master-key"})
        assert response.status_code == 200, response.text
        assert {model["id"]: _limits(model) for model in response.json()["data"]}[_QWEN] == (262144, None)
    fetch.assert_not_called()


def test_an_unknown_model_is_refused_without_fetching_models_dev(limits_client: TestClient) -> None:
    """A 404 has no limits to report, so it never waits on a models.dev fetch."""
    fetch = AsyncMock(return_value=_MODELS_DEV)
    with patch.object(mcs, "_fetch", new=fetch):
        response = limits_client.get(
            f"{API_ROOT}/models/nebius:not-a-model", headers={API_KEY_HEADER: "Bearer test-master-key"}
        )

    assert response.status_code == 404, response.text
    fetch.assert_not_called()
