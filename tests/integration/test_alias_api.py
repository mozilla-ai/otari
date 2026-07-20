"""End-to-end behavior for runtime (stored) model aliases.

A stored alias is a row in ``model_aliases``, created through /v1/aliases. It
means the same thing to a request as a ``config.yml`` alias, but it can appear
without a restart, so the interesting cases are the ones where the two kinds
have to agree: routing, listing, pricing, and validation.
"""

from collections.abc import Generator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from gateway.core.config import API_KEY_HEADER, GatewayConfig
from gateway.db import Base, get_db
from gateway.main import create_app

from .conftest import _run_alembic_migrations, build_async_session_override

ALIASES = "/v1/aliases"
HEADERS = {API_KEY_HEADER: "Bearer test-master-key"}


class _MockCompletionError(Exception):
    """Raised to short-circuit the mocked acompletion after capturing kwargs."""


@pytest.fixture
def alias_config(postgres_url: str) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        providers={
            "anthropic": {"api_key": "sk-ant"},
            "home_lab": {"provider_type": "openai", "api_base": "https://box.ts.net/v1", "api_key": "home-lab-token"},
        },
        aliases={"configalias": "anthropic:claude-opus-4"},
    )


@pytest.fixture
def client(alias_config: GatewayConfig) -> Generator[TestClient]:
    _run_alembic_migrations(alias_config.database_url)
    engine = create_engine(alias_config.database_url, pool_pre_ping=True)
    app = create_app(alias_config)
    override_get_db, dispose_override = build_async_session_override(alias_config.database_url)
    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        dispose_override()
        Base.metadata.drop_all(bind=engine)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version CASCADE"))
            conn.commit()
        engine.dispose()


def _create(client: TestClient, name: str, target: str) -> None:
    resp = client.post(ALIASES, json={"name": name, "target": target}, headers=HEADERS)
    assert resp.status_code == 200, resp.text


def _create_user(client: TestClient) -> None:
    assert client.post("/v1/users", json={"user_id": "u1", "alias": "u1"}, headers=HEADERS).status_code == 200


def _post_chat_capture(client: TestClient, model: str) -> dict[str, object]:
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> None:
        captured.update(kwargs)
        raise _MockCompletionError

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "Hi"}], "user": "u1"},
            headers=HEADERS,
        )
    assert captured, f"acompletion was never called for {model!r}"
    return captured


# ---------------------------------------------------------------------------
# Auth and CRUD
# ---------------------------------------------------------------------------


def test_alias_routes_require_master_key(client: TestClient) -> None:
    assert client.get(ALIASES).status_code == 401
    assert client.post(ALIASES, json={"name": "a", "target": "anthropic:claude-opus-4"}).status_code == 401


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("POST", "/v1/aliases", {"name": "probe", "target": "anthropic:claude-opus-4"}),
        ("DELETE", "/v1/aliases/probe", None),
        ("PATCH", "/v1/settings", {"model_discovery": False}),
        ("GET", "/v1/providers", None),
        ("GET", "/v1/models/metadata", None),
    ],
)
def test_management_endpoints_reject_a_valid_non_master_key(
    client: TestClient, method: str, path: str, json_body: dict[str, object] | None
) -> None:
    """A working tenant key must not reach any master-gated management surface.

    ``verify_master_key`` treats a non-master key as unauthenticated (401), so a
    one-token swap to ``verify_api_key_or_master_key`` on any of these routes,
    which would hand a tenant key control of routing and billing toggles, would
    fail here.
    """
    created = client.post("/v1/keys", json={"key_name": "probe"}, headers=HEADERS)
    assert created.status_code == 200
    key_header = {API_KEY_HEADER: f"Bearer {created.json()['key']}"}

    resp = client.request(method, path, json=json_body, headers=key_header)
    assert resp.status_code == 401

    # The same key still works on the caller-facing listing, so the 401 above is
    # the master-key gate, not a broken key.
    assert client.get("/v1/models", headers=key_header).status_code == 200


def test_create_and_list(client: TestClient) -> None:
    _create(client, "fast", "anthropic:claude-haiku-4")

    body = client.get(ALIASES, headers=HEADERS).json()
    by_name = {row["name"]: row for row in body}
    # Both kinds are listed, and each says where it came from, because only a
    # stored one can be edited here.
    assert by_name["fast"] == {
        "name": "fast",
        "target": "anthropic:claude-haiku-4",
        "source": "stored",
        "created_at": by_name["fast"]["created_at"],
        "updated_at": by_name["fast"]["updated_at"],
    }
    assert by_name["configalias"]["source"] == "config"
    assert by_name["configalias"]["target"] == "anthropic:claude-opus-4"


def test_create_is_idempotent_and_retargets(client: TestClient) -> None:
    _create(client, "fast", "anthropic:claude-haiku-4")
    _create(client, "fast", "home_lab:qwen3")

    rows = [row for row in client.get(ALIASES, headers=HEADERS).json() if row["name"] == "fast"]
    assert len(rows) == 1
    assert rows[0]["target"] == "home_lab:qwen3"


def test_delete_removes_the_alias(client: TestClient) -> None:
    _create(client, "fast", "anthropic:claude-haiku-4")

    assert client.delete(f"{ALIASES}/fast", headers=HEADERS).status_code == 204
    assert "fast" not in {row["name"] for row in client.get(ALIASES, headers=HEADERS).json()}
    assert client.delete(f"{ALIASES}/fast", headers=HEADERS).status_code == 404


def test_delete_of_a_config_alias_explains_itself(client: TestClient) -> None:
    # It is in the listing, so "not found" would read as a bug rather than as
    # "this one lives in a file".
    resp = client.delete(f"{ALIASES}/configalias", headers=HEADERS)
    assert resp.status_code == 404
    assert "config.yml" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Validation: the startup rules, enforced at runtime
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "target", "expected"),
    [
        ("has:colon", "anthropic:claude-opus-4", "must not contain"),
        ("has/slash", "anthropic:claude-opus-4", "must not contain"),
        ("anthropic", "anthropic:claude-opus-4", "collides with a configured provider instance"),
        ("fast", "no-prefix", "must be of the form"),
        ("fast", "not_a_provider:model", "neither a configured"),
        ("fast", "", "non-empty target"),
    ],
)
def test_invalid_alias_is_rejected(client: TestClient, name: str, target: str, expected: str) -> None:
    resp = client.post(ALIASES, json={"name": name, "target": target}, headers=HEADERS)
    assert resp.status_code == 400
    assert expected in resp.json()["detail"]


def test_cannot_shadow_a_config_alias(client: TestClient) -> None:
    # Config wins during resolution, so storing this name would be accepted and
    # then never take effect.
    resp = client.post(ALIASES, json={"name": "configalias", "target": "home_lab:qwen3"}, headers=HEADERS)

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "config.yml" in detail
    assert "never be used" in detail


def test_cannot_chain_onto_a_stored_alias(client: TestClient) -> None:
    _create(client, "fast", "anthropic:claude-haiku-4")

    resp = client.post(ALIASES, json={"name": "faster", "target": "fast:something"}, headers=HEADERS)
    assert resp.status_code == 400
    assert "chaining is not supported" in resp.json()["detail"]


def test_cannot_chain_onto_a_config_alias(client: TestClient) -> None:
    resp = client.post(ALIASES, json={"name": "chained", "target": "configalias:x"}, headers=HEADERS)
    assert resp.status_code == 400
    assert "chaining is not supported" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# A stored alias behaves like a configured one
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stored_alias_routes_to_its_target(client: TestClient) -> None:
    # The point of the whole feature: no restart between creating the alias and
    # it routing a request.
    _create_user(client)
    _create(client, "fast", "anthropic:claude-haiku-4")

    captured = _post_chat_capture(client, "fast")

    assert captured["model"] == "anthropic:claude-haiku-4"
    assert captured["api_key"] == "sk-ant"


@pytest.mark.asyncio
async def test_stored_alias_to_a_named_instance_uses_its_credentials(client: TestClient) -> None:
    _create_user(client)
    _create(client, "house", "home_lab:qwen3")

    captured = _post_chat_capture(client, "house")

    assert captured["model"] == "openai:qwen3"
    assert captured["api_base"] == "https://box.ts.net/v1"
    assert captured["api_key"] == "home-lab-token"


@pytest.mark.asyncio
async def test_deleted_alias_stops_routing(client: TestClient) -> None:
    _create_user(client)
    _create(client, "fast", "anthropic:claude-haiku-4")
    assert client.delete(f"{ALIASES}/fast", headers=HEADERS).status_code == 204

    dispatch = AsyncMock()
    with patch("gateway.api.routes.chat.acompletion", new=dispatch):
        # "fast" is no longer an alias and was never a valid selector on its own,
        # so it now fails to resolve, the same as any unknown model would. It is
        # surfaced as a 400 (not a bare 500) and, crucially, does not quietly
        # keep reaching the old target.
        response = client.post(
            "/v1/chat/completions",
            json={"model": "fast", "messages": [{"role": "user", "content": "Hi"}], "user": "u1"},
            headers=HEADERS,
        )

    assert response.status_code == 400
    assert "fast" in response.json()["detail"]
    dispatch.assert_not_awaited()


def test_stored_alias_appears_in_the_model_listing(client: TestClient) -> None:
    _create(client, "fast", "anthropic:claude-haiku-4")

    data = client.get("/v1/models", headers=HEADERS).json()["data"]
    entry = next(m for m in data if m["id"] == "fast")

    assert entry["owned_by"] == "otari"
    # The target stays hidden, same as for a config alias.
    assert "anthropic:claude-haiku-4" not in {m["id"] for m in data}


def test_stored_alias_inherits_its_targets_price(client: TestClient) -> None:
    _create(client, "fast", "anthropic:claude-haiku-4")
    client.post(
        "/v1/pricing",
        json={"model_key": "anthropic:claude-haiku-4", "input_price_per_million": 1.0, "output_price_per_million": 5.0},
        headers=HEADERS,
    )

    entry = client.get("/v1/models/fast", headers=HEADERS).json()
    assert entry["pricing"] == {"input_price_per_million": 1.0, "output_price_per_million": 5.0}


def test_pricing_a_stored_alias_is_rejected(client: TestClient) -> None:
    # Same reason as for a config alias: the row would key on the alias name and
    # never be read, so accepting it would report success and change nothing.
    _create(client, "fast", "anthropic:claude-haiku-4")

    resp = client.post(
        "/v1/pricing",
        json={"model_key": "fast", "input_price_per_million": 99.0, "output_price_per_million": 99.0},
        headers=HEADERS,
    )

    assert resp.status_code == 400
    assert "anthropic:claude-haiku-4" in resp.json()["detail"]
    assert client.get("/v1/pricing/fast", headers=HEADERS).status_code == 404
