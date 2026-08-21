"""Integration tests for model auto-discovery in the GET /v1/models endpoint."""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, GatewayConfig
from gateway.services.model_discovery_service import get_model_cache

from .conftest import build_test_client


def _make_openai_model(model_id: str, owned_by: str = "openai", created: int = 1700000000) -> dict[str, Any]:
    """Build a dict that Model.model_validate accepts."""
    return {"id": model_id, "object": "model", "created": created, "owned_by": owned_by}


@pytest.fixture
def discovery_config(postgres_url: str) -> GatewayConfig:
    """Config with a fake openai provider and discovery enabled."""
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        model_discovery=True,
        model_cache_ttl_seconds=300,
        providers={"openai": {"api_key": "sk-fake-for-test"}},
    )


@pytest.fixture
def no_discovery_config(postgres_url: str) -> GatewayConfig:
    """Config with discovery disabled."""
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        model_discovery=False,
        providers={"openai": {"api_key": "sk-fake-for-test"}},
    )


@pytest.fixture
def discovery_client(discovery_config: GatewayConfig) -> Generator[TestClient]:
    """Test client with model discovery enabled."""
    # Clear the module-level model cache before each test.
    get_model_cache().clear()
    yield from build_test_client(discovery_config)


@pytest.fixture
def no_discovery_client(no_discovery_config: GatewayConfig) -> Generator[TestClient]:
    """Test client with model discovery disabled."""
    get_model_cache().clear()
    yield from build_test_client(no_discovery_config)


@pytest.fixture
def discovery_master_header() -> dict[str, str]:
    return {API_KEY_HEADER: "Bearer test-master-key"}


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


def test_list_models_with_discovery(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """Discovered models appear in GET /v1/models."""
    from any_llm.types.model import Model

    fake_models = [
        Model(**_make_openai_model("gpt-4o")),
        Model(**_make_openai_model("gpt-4o-mini")),
    ]

    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            return_value=fake_models,
        ),
    ):
        resp = discovery_client.get("/v1/models", headers=discovery_master_header)

    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "openai:gpt-4o" in ids
    assert "openai:gpt-4o-mini" in ids


def test_list_models_excludes_env_only_provider(
    postgres_url: str,
    discovery_master_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider with only a credential env var (not configured) is NOT listed.

    Discovery is scoped to the configured provider instances: a provider only
    sources models once an operator configures it (config.yml or the Providers
    page). An empty ``providers`` block therefore lists nothing even when a
    provider's credential env var (here ANTHROPIC_API_KEY) is present.
    """
    from any_llm.types.model import Model

    # anthropic is NOT configured; only its credential env var is present.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        model_discovery=True,
        model_cache_ttl_seconds=300,
        providers={},
    )

    async def fake_alist(provider: Any, **kwargs: Any) -> list[Model]:
        if provider.value == "anthropic":
            return [Model(**_make_openai_model("claude-3-5-sonnet", owned_by="anthropic"))]
        return []

    get_model_cache().clear()
    client_gen = build_test_client(config)
    client = next(client_gen)
    try:
        with (
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", side_effect=fake_alist),
        ):
            resp = client.get("/v1/models", headers=discovery_master_header)
    finally:
        client_gen.close()

    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "anthropic:claude-3-5-sonnet" not in ids


def test_list_models_includes_keyless_local_provider(
    postgres_url: str,
    discovery_master_header: dict[str, str],
) -> None:
    """A keyless local backend configured with a valueless entry IS listed.

    Regression for mozilla-ai/otari#389. Ollama, llama.cpp, and llamafile take no
    credential, so ``ollama:`` with no body under ``providers`` is the whole
    config. That entry used to fail to load (YAML gives ``None``), leaving a
    running local server callable but permanently absent from GET /v1/models.
    """
    from any_llm.types.model import Model

    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        model_discovery=True,
        model_cache_ttl_seconds=300,
        providers={"ollama": None},  # type: ignore[dict-item]
    )
    assert config.providers == {"ollama": {}}

    async def fake_alist(provider: Any, **kwargs: Any) -> list[Model]:
        if provider.value == "ollama":
            return [Model(**_make_openai_model("llama3", owned_by="ollama"))]
        return []

    get_model_cache().clear()
    client_gen = build_test_client(config)
    client = next(client_gen)
    try:
        with (
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", side_effect=fake_alist),
        ):
            resp = client.get("/v1/models", headers=discovery_master_header)
    finally:
        client_gen.close()

    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "ollama:llama3" in ids


def test_list_models_discovery_disabled(
    no_discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """When discovery is disabled, only pricing-table models appear."""
    resp = no_discovery_client.get("/v1/models", headers=discovery_master_header)
    assert resp.status_code == 200
    assert resp.json()["data"] == []


def test_list_models_pricing_enrichment(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """Discovered models are enriched with pricing data when available."""
    from any_llm.types.model import Model

    fake_models = [Model(**_make_openai_model("gpt-4o"))]

    # First, set pricing for this model.
    resp = discovery_client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:gpt-4o",
            "input_price_per_million": 2.5,
            "output_price_per_million": 10.0,
        },
        headers=discovery_master_header,
    )
    assert resp.status_code == 200

    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            return_value=fake_models,
        ),
    ):
        resp = discovery_client.get("/v1/models", headers=discovery_master_header)

    assert resp.status_code == 200
    models = resp.json()["data"]
    gpt4 = next(m for m in models if m["id"] == "openai:gpt-4o")
    assert gpt4["pricing"] is not None
    assert gpt4["pricing"]["input_price_per_million"] == 2.5
    assert gpt4["pricing"]["output_price_per_million"] == 10.0


def test_list_models_no_pricing_returns_null(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """Discovered models without pricing have pricing=None."""
    from any_llm.types.model import Model

    fake_models = [Model(**_make_openai_model("gpt-4o-mini"))]

    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            return_value=fake_models,
        ),
    ):
        resp = discovery_client.get("/v1/models", headers=discovery_master_header)

    assert resp.status_code == 200
    models = resp.json()["data"]
    gpt4_mini = next(m for m in models if m["id"] == "openai:gpt-4o-mini")
    assert gpt4_mini["pricing"] is None


def test_list_models_pricing_only_still_appears(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """Models only in the pricing table still appear even when discovery returns nothing."""
    # Add a pricing-only model (for a provider not in config.providers).
    resp = discovery_client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:legacy-model",
            "input_price_per_million": 1.0,
            "output_price_per_million": 2.0,
        },
        headers=discovery_master_header,
    )
    assert resp.status_code == 200

    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        resp = discovery_client.get("/v1/models", headers=discovery_master_header)

    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "openai:legacy-model" in ids


def test_list_models_provider_filter(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """The ?provider= query param filters results to a single provider."""
    from any_llm.types.model import Model

    fake_models = [Model(**_make_openai_model("gpt-4o"))]

    # Add a pricing entry for a different provider.
    discovery_client.post(
        "/v1/pricing",
        json={
            "model_key": "anthropic:claude-3-opus",
            "input_price_per_million": 15.0,
            "output_price_per_million": 75.0,
        },
        headers=discovery_master_header,
    )

    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            return_value=fake_models,
        ),
    ):
        resp = discovery_client.get("/v1/models?provider=openai", headers=discovery_master_header)

    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    # OpenAI models should appear; Anthropic should not.
    assert "openai:gpt-4o" in ids
    assert all("anthropic:" not in mid for mid in ids)


def test_list_models_discovery_error_graceful(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """When discovery fails for a provider, the endpoint still succeeds."""
    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            side_effect=ConnectionError("upstream unreachable"),
        ),
    ):
        resp = discovery_client.get("/v1/models", headers=discovery_master_header)

    assert resp.status_code == 200
    assert resp.json()["object"] == "list"


def test_list_models_tolerates_a_provider_without_created(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """A provider reporting no ``created`` must not fail the whole listing.

    ``Model.created`` is typed ``int`` but the OpenAI SDK builds response models
    without validation, so an OpenAI-compatible provider answering ``null`` gets
    that far. Constructing the ModelObject used to raise and 500 the endpoint.
    """
    from any_llm.types.model import Model

    fake_models = [
        Model.construct(id="local-model", object="model", created=None, owned_by="openai"),
        Model(**_make_openai_model("gpt-4o")),
    ]

    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            return_value=fake_models,
        ),
    ):
        resp = discovery_client.get("/v1/models", headers=discovery_master_header)

    assert resp.status_code == 200
    by_id = {m["id"]: m for m in resp.json()["data"]}
    assert by_id["openai:local-model"]["created"] == 0
    assert by_id["openai:gpt-4o"]["created"] == 1700000000


def test_get_model_tolerates_a_provider_without_created(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """The same missing ``created`` must not fail GET /v1/models/{id}."""
    from any_llm.types.model import Model

    get_model_cache().set(
        "openai",
        [Model.construct(id="local-model", object="model", created=None, owned_by="openai")],
    )

    resp = discovery_client.get("/v1/models/openai:local-model", headers=discovery_master_header)
    assert resp.status_code == 200
    assert resp.json()["created"] == 0


def test_get_model_from_discovery_cache(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """GET /v1/models/{id} finds a model that exists only in the discovery cache."""
    from any_llm.types.model import Model

    fake_models = [Model(**_make_openai_model("gpt-4o"))]

    # Pre-populate the discovery cache.
    cache = get_model_cache()
    cache.set("openai", fake_models)

    resp = discovery_client.get("/v1/models/openai:gpt-4o", headers=discovery_master_header)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "openai:gpt-4o"
    assert data["owned_by"] == "openai"


def test_get_model_not_found_with_discovery(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """GET /v1/models/{id} returns 404 when not in cache or pricing table."""
    resp = discovery_client.get(
        "/v1/models/openai:nonexistent-model",
        headers=discovery_master_header,
    )
    assert resp.status_code == 404


def test_list_models_sorted(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """Discovered models are sorted alphabetically by id."""
    from any_llm.types.model import Model

    fake_models = [
        Model(**_make_openai_model("gpt-4o")),
        Model(**_make_openai_model("dall-e-3")),
        Model(**_make_openai_model("gpt-3.5-turbo")),
    ]

    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            return_value=fake_models,
        ),
    ):
        resp = discovery_client.get("/v1/models", headers=discovery_master_header)

    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# GET /v1/models/discoverable
# ---------------------------------------------------------------------------


@pytest.fixture
def two_provider_config(postgres_url: str) -> GatewayConfig:
    """Config with two live providers and discovery disabled.

    Discovery is off to prove the discoverable endpoint ignores the flag.
    """
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        model_discovery=False,
        model_cache_ttl_seconds=300,
        providers={
            "openai": {"api_key": "sk-fake-for-test"},
            "anthropic": {"api_key": "sk-ant-fake-for-test"},
        },
    )


@pytest.fixture
def two_provider_client(two_provider_config: GatewayConfig) -> Generator[TestClient]:
    get_model_cache().clear()
    yield from build_test_client(two_provider_config)


def _alist_models_per_provider(**kwargs: Any) -> list[Any]:
    """Serve openai, fail anthropic, so one response carries both outcomes."""
    from any_llm.types.model import Model

    provider = kwargs["provider"]
    if provider.value == "openai":
        return [Model(**_make_openai_model("gpt-4o")), Model(**_make_openai_model("gpt-4o-mini"))]
    raise RuntimeError("authentication failed: invalid x-api-key")


def test_discoverable_reports_each_provider_separately(
    two_provider_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """A failing provider carries its error without blanking the working one."""
    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            side_effect=_alist_models_per_provider,
        ),
    ):
        resp = two_provider_client.get("/v1/models/discoverable", headers=discovery_master_header)

    assert resp.status_code == 200
    providers = {p["provider"]: p for p in resp.json()["providers"]}
    assert set(providers) == {"anthropic", "openai"}

    assert providers["openai"]["ok"] is True
    assert providers["openai"]["error"] is None
    assert [m["id"] for m in providers["openai"]["models"]] == ["gpt-4o", "gpt-4o-mini"]
    # The selector is what the dashboard must send back as `model`.
    assert [m["key"] for m in providers["openai"]["models"]] == ["openai:gpt-4o", "openai:gpt-4o-mini"]

    assert providers["anthropic"]["ok"] is False
    assert providers["anthropic"]["models"] == []
    assert "authentication failed" in providers["anthropic"]["error"]


def test_discoverable_queries_providers_when_discovery_disabled(
    two_provider_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """model_discovery is a listing policy for /v1/models, not a query switch.

    The fixture config sets model_discovery=False; the operator still gets the
    models their credentials can reach.
    """
    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            side_effect=_alist_models_per_provider,
        ),
    ):
        discoverable = two_provider_client.get("/v1/models/discoverable", headers=discovery_master_header)
        listed = two_provider_client.get("/v1/models", headers=discovery_master_header)

    openai_models = next(p for p in discoverable.json()["providers"] if p["provider"] == "openai")
    assert [m["id"] for m in openai_models["models"]] == ["gpt-4o", "gpt-4o-mini"]
    # Same config, same call: /v1/models still publishes nothing, as configured.
    assert listed.json()["data"] == []


def test_discoverable_falls_back_to_declared_models(
    postgres_url: str,
    discovery_master_header: dict[str, str],
) -> None:
    """A backend with no /v1/models reports the operator's declared models."""
    get_model_cache().clear()
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        model_cache_ttl_seconds=300,
        providers={
            "local": {
                "provider_type": "openai",
                "api_key": "unused",
                "api_base": "http://127.0.0.1:9/v1",
                "models": ["llama-3.1-8b"],
            },
        },
    )

    client_gen = build_test_client(config)
    client = next(client_gen)
    try:
        with patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=False,
        ):
            resp = client.get("/v1/models/discoverable", headers=discovery_master_header)

        assert resp.status_code == 200
        provider = resp.json()["providers"][0]
        assert provider["provider"] == "local"
        assert provider["ok"] is True
        assert provider["models"] == [{"id": "llama-3.1-8b", "key": "local:llama-3.1-8b"}]
    finally:
        client_gen.close()


def test_discoverable_explains_a_provider_that_cannot_list(
    two_provider_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """No listing support and no declared models is reported, not silently empty."""
    with patch(
        "gateway.services.model_discovery_service._supports_list_models",
        return_value=False,
    ):
        resp = two_provider_client.get("/v1/models/discoverable", headers=discovery_master_header)

    assert resp.status_code == 200
    for provider in resp.json()["providers"]:
        assert provider["ok"] is False
        assert provider["models"] == []
        assert "cannot list models" in provider["error"]
        assert "config.yml" in provider["error"]


def test_discoverable_requires_master_key(
    two_provider_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """A working non-master API key is rejected: errors describe the config."""
    created = two_provider_client.post(
        "/v1/keys",
        json={"key_name": "discoverable-probe"},
        headers=discovery_master_header,
    )
    assert created.status_code == 200
    api_key = created.json()["key"]

    resp = two_provider_client.get(
        "/v1/models/discoverable",
        headers={API_KEY_HEADER: f"Bearer {api_key}"},
    )
    # 401 rather than 403: verify_master_key treats a non-master key as
    # unauthenticated, matching every other master-key-gated route.
    assert resp.status_code == 401

    # Same key on the caller-facing listing still works, so the 401 is the gate
    # and not a broken key.
    assert two_provider_client.get("/v1/models", headers={API_KEY_HEADER: f"Bearer {api_key}"}).status_code == 200


def test_discoverable_is_not_shadowed_by_get_model(
    two_provider_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """The path must not be captured by GET /v1/models/{model_id:path}.

    Registration order decides this, so a reordering would silently turn the
    response into a 404 ModelObject lookup for a model named "discoverable".
    """
    with patch(
        "gateway.services.model_discovery_service._supports_list_models",
        return_value=False,
    ):
        resp = two_provider_client.get("/v1/models/discoverable", headers=discovery_master_header)

    assert resp.status_code == 200
    body = resp.json()
    assert "providers" in body
    assert "object" not in body


# ---------------------------------------------------------------------------
# GET /v1/providers/health (provider health monitor)
# ---------------------------------------------------------------------------


def test_provider_health_reports_each_provider(
    two_provider_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """A failing provider is unhealthy and carries its error; a working one is healthy."""
    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            side_effect=_alist_models_per_provider,
        ),
    ):
        resp = two_provider_client.get("/v1/providers/health", headers=discovery_master_header)

    assert resp.status_code == 200
    body = resp.json()
    # Summary counts are precomputed so the overview tile (issue #302) need not re-derive them.
    assert body["total"] == 2
    assert body["healthy"] == 1
    assert body["checked_at"] is not None

    providers = {p["instance"]: p for p in body["providers"]}
    # Sorted by instance name for a stable table.
    assert [p["instance"] for p in body["providers"]] == ["anthropic", "openai"]

    assert providers["openai"]["ok"] is True
    assert providers["openai"]["model_count"] == 2
    assert providers["openai"]["error"] is None
    assert providers["openai"]["checked_at"] is not None

    assert providers["anthropic"]["ok"] is False
    assert providers["anthropic"]["model_count"] == 0
    assert "authentication failed" in providers["anthropic"]["error"]
    # A real credential failure is not a discovery gap, so it stays out of `degraded`.
    assert providers["anthropic"]["discovery_unsupported"] is False
    assert body["degraded"] == 0


def _alist_models_missing_endpoint(**kwargs: Any) -> list[Any]:
    """Serve openai; answer 404 for anthropic, as a backend with no /v1/models does."""
    from any_llm.types.model import Model

    provider = kwargs["provider"]
    if provider.value == "openai":
        return [Model(**_make_openai_model("gpt-4o"))]
    error = RuntimeError("Error code: 404 - {'detail': 'Not Found'}")
    error.status_code = 404  # type: ignore[attr-defined]
    raise error


def test_provider_health_flags_a_missing_models_endpoint_instead_of_unreachable(
    two_provider_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """A backend without /v1/models is degraded, not unreachable (otari#447).

    The credentials may well be fine; only discovery is unavailable, so the
    dashboard needs to tell the two apart.
    """
    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            side_effect=_alist_models_missing_endpoint,
        ),
    ):
        resp = two_provider_client.get("/v1/providers/health", headers=discovery_master_header)

    assert resp.status_code == 200
    body = resp.json()
    assert body["healthy"] == 1
    assert body["degraded"] == 1
    assert body["total"] == 2

    providers = {p["instance"]: p for p in body["providers"]}
    assert providers["anthropic"]["ok"] is False
    assert providers["anthropic"]["discovery_unsupported"] is True
    assert providers["openai"]["discovery_unsupported"] is False


def test_discoverable_flags_a_missing_models_endpoint(
    two_provider_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """The operator model picker distinguishes the same two failures (otari#447)."""
    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            side_effect=_alist_models_missing_endpoint,
        ),
    ):
        resp = two_provider_client.get("/v1/models/discoverable", headers=discovery_master_header)

    assert resp.status_code == 200
    providers = {p["provider"]: p for p in resp.json()["providers"]}
    assert providers["anthropic"]["ok"] is False
    assert providers["anthropic"]["discovery_unsupported"] is True
    assert providers["openai"]["discovery_unsupported"] is False


def test_provider_health_refresh_forces_a_live_recheck(
    two_provider_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """refresh=true clears the cache, so a now-broken provider is re-dialed, not served a stale verdict."""
    from datetime import UTC, datetime, timedelta

    from any_llm.types.model import Model

    # Prime a healthy listing whose dial is older than the refresh debounce window,
    # as if a check had succeeded a while ago, so a forced refresh actually re-dials.
    get_model_cache().set("openai", [Model(**_make_openai_model("gpt-4o"))])
    get_model_cache()._store["openai"].checked_at = datetime.now(UTC) - timedelta(seconds=60)

    def always_fail(**kwargs: Any) -> list[Any]:
        raise RuntimeError("provider down")

    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            side_effect=always_fail,
        ),
    ):
        # Without refresh, openai is served from the cached healthy listing (not dialed).
        cached = two_provider_client.get("/v1/providers/health", headers=discovery_master_header)
        # With refresh, the cache is cleared, openai is dialed live, and the outage surfaces.
        refreshed = two_provider_client.get(
            "/v1/providers/health?refresh=true", headers=discovery_master_header
        )

    cached_openai = next(p for p in cached.json()["providers"] if p["instance"] == "openai")
    assert cached_openai["ok"] is True
    assert cached_openai["model_count"] == 1

    refreshed_openai = next(p for p in refreshed.json()["providers"] if p["instance"] == "openai")
    assert refreshed_openai["ok"] is False
    assert "provider down" in refreshed_openai["error"]


def test_provider_health_refresh_is_debounced_within_the_window(
    two_provider_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """A refresh moments after a recent dial is coalesced onto it, not re-dialed."""
    from any_llm.types.model import Model

    # A freshly dialed healthy listing (checked_at == now); a refresh within the
    # debounce window must reuse it rather than clear the cache and dial again.
    get_model_cache().set("openai", [Model(**_make_openai_model("gpt-4o"))])

    def always_fail(**kwargs: Any) -> list[Any]:
        raise RuntimeError("provider down")

    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            side_effect=always_fail,
        ),
    ):
        refreshed = two_provider_client.get(
            "/v1/providers/health?refresh=true", headers=discovery_master_header
        )

    # Debounced: the recent healthy listing is served, not a fresh failed dial.
    refreshed_openai = next(p for p in refreshed.json()["providers"] if p["instance"] == "openai")
    assert refreshed_openai["ok"] is True
    assert refreshed_openai["model_count"] == 1


def test_provider_health_requires_master_key(
    two_provider_client: TestClient,
) -> None:
    """The endpoint describes gateway config, so it is master-key gated."""
    resp = two_provider_client.get("/v1/providers/health")
    assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Background refresh: no read dials a provider on the request path
# ---------------------------------------------------------------------------


@pytest.fixture
def cached_discovery_client(postgres_url: str) -> Generator[TestClient]:
    """A client on the production defaults, whose tests seed the cache themselves.

    Identical to ``discovery_client``; named apart because these tests assert
    what the *read* path does once the cache is warm, and each seeds it. The
    refresher that fills it in production is suppressed suite-wide by the
    ``_no_background_refresh`` fixture in conftest.
    """
    get_model_cache().clear()
    yield from build_test_client(
        GatewayConfig(
            database_url=postgres_url,
            master_key="test-master-key",
            host="127.0.0.1",
            port=8000,
            auto_migrate=False,
            model_discovery=True,
            model_cache_ttl_seconds=300,
            providers={"openai": {"api_key": "sk-fake-for-test"}},
        )
    )


def _warm_cache_with_expired_entry(model_id: str) -> None:
    """Seed the discovery cache and backdate it well past model_cache_ttl_seconds."""
    from any_llm.types.model import Model

    cache = get_model_cache()
    cache.set("openai", [Model(**_make_openai_model(model_id))])
    cache._store["openai"].cached_at -= 10_000


def test_models_read_serves_an_expired_cache_without_dialing(
    cached_discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """The fix: an expired entry is served, not re-dialed, on GET /v1/models.

    Before the background refresher, whichever request arrived after the TTL
    lapsed paid ``model_discovery_timeout_seconds`` per unreachable provider and
    held its database session open for the whole dial. That is what made an
    operator's next click sit behind a page load.
    """
    _warm_cache_with_expired_entry("gpt-4o")

    with patch(
        "gateway.services.model_discovery_service.alist_models",
        new_callable=AsyncMock,
    ) as mock_alist:
        resp = cached_discovery_client.get("/v1/models", headers=discovery_master_header)

    assert resp.status_code == 200
    assert "openai:gpt-4o" in [model["id"] for model in resp.json()["data"]]
    mock_alist.assert_not_awaited()


def test_discoverable_read_serves_an_expired_cache_without_dialing(
    cached_discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """Same for the operator listing, and it reports when it was last dialed."""
    _warm_cache_with_expired_entry("gpt-4o")

    with patch(
        "gateway.services.model_discovery_service.alist_models",
        new_callable=AsyncMock,
    ) as mock_alist:
        resp = cached_discovery_client.get("/v1/models/discoverable", headers=discovery_master_header)

    assert resp.status_code == 200
    provider = resp.json()["providers"][0]
    assert provider["provider"] == "openai"
    assert [model["id"] for model in provider["models"]] == ["gpt-4o"]
    # The age of the answer is reported rather than implied, so a cached verdict
    # is never mistaken for a live one.
    assert provider["checked_at"] is not None
    mock_alist.assert_not_awaited()


def test_discoverable_refresh_still_dials(
    cached_discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """?refresh=true is the operator's escape hatch and must reach the provider."""
    from any_llm.types.model import Model

    _warm_cache_with_expired_entry("stale-model")

    with (
        patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            return_value=[Model(**_make_openai_model("fresh-model"))],
        ) as mock_alist,
    ):
        resp = cached_discovery_client.get(
            "/v1/models/discoverable?refresh=true", headers=discovery_master_header
        )

    assert resp.status_code == 200
    assert [model["id"] for model in resp.json()["providers"][0]["models"]] == ["fresh-model"]
    assert mock_alist.await_count == 1


def test_provider_health_read_serves_an_expired_cache_without_dialing(
    cached_discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """The hourly health poll fans out over every provider; it must not dial."""
    _warm_cache_with_expired_entry("gpt-4o")

    with patch(
        "gateway.services.model_discovery_service.alist_models",
        new_callable=AsyncMock,
    ) as mock_alist:
        resp = cached_discovery_client.get("/v1/providers/health", headers=discovery_master_header)

    assert resp.status_code == 200
    body = resp.json()
    assert body["healthy"] == 1
    assert body["total"] == 1
    mock_alist.assert_not_awaited()


@pytest.fixture
def uncached_discovery_client(postgres_url: str) -> Generator[TestClient]:
    """A client with discovery caching disabled (`model_cache_ttl_seconds = 0`)."""
    get_model_cache().clear()
    yield from build_test_client(
        GatewayConfig(
            database_url=postgres_url,
            master_key="test-master-key",
            host="127.0.0.1",
            port=8000,
            auto_migrate=False,
            model_discovery=True,
            model_cache_ttl_seconds=0,
            providers={"openai": {"api_key": "sk-fake-for-test"}},
        )
    )


def test_models_read_dials_when_caching_is_disabled(
    uncached_discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """`model_cache_ttl_seconds = 0` still means dial on every read.

    It is the only switch for this, so it must not quietly become "serve a cache
    nothing refreshes": a cached entry is ignored here, not served.
    """
    from any_llm.types.model import Model

    _warm_cache_with_expired_entry("stale-model")

    with (
        patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            return_value=[Model(**_make_openai_model("fresh-model"))],
        ) as mock_alist,
    ):
        resp = uncached_discovery_client.get("/v1/models", headers=discovery_master_header)

    assert resp.status_code == 200
    assert "openai:fresh-model" in [model["id"] for model in resp.json()["data"]]
    assert mock_alist.await_count == 1


def test_model_detail_agrees_with_the_listing_on_a_stale_entry(
    cached_discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """GET /v1/models/{id} must not 404 a model GET /v1/models is listing.

    Nothing on the request path renews ``cached_at`` any more, and the refresher
    sleeps its interval *after* each round, so an entry is expired from the moment
    the next round starts until its dials finish. A TTL-bounded peek in the detail
    endpoint would disagree with the listing for that whole window, for any
    provider model with no pricing row and no genai-prices fallback.
    """
    _warm_cache_with_expired_entry("gpt-4o")

    with patch(
        "gateway.services.model_discovery_service.alist_models",
        new_callable=AsyncMock,
    ) as mock_alist:
        listed = cached_discovery_client.get("/v1/models", headers=discovery_master_header)
        detail = cached_discovery_client.get("/v1/models/openai:gpt-4o", headers=discovery_master_header)

    assert "openai:gpt-4o" in [model["id"] for model in listed.json()["data"]]
    assert detail.status_code == 200
    assert detail.json()["id"] == "openai:gpt-4o"
    # Neither read dialed: the detail endpoint never dials at all, and the
    # listing serves the cache.
    mock_alist.assert_not_awaited()


def test_model_detail_does_not_serve_a_cached_failure_as_a_model(
    cached_discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """A negatively cached provider has no models to report, at any age."""
    from datetime import UTC, datetime

    from gateway.services.model_discovery_service import ProviderDiscovery, _CacheEntry

    cache = get_model_cache()
    cache.clear()
    cache._store["openai"] = _CacheEntry(
        result=ProviderDiscovery(provider="openai", models=[], error="bad key"),
        cached_at=0.0,
        checked_at=datetime.now(UTC),
    )

    resp = cached_discovery_client.get("/v1/models/openai:never-seen", headers=discovery_master_header)

    # Falls through to the pricing/genai-prices answer rather than inventing a
    # discovered model out of a failed dial.
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        assert resp.json().get("pricing_source") != "discovered"
