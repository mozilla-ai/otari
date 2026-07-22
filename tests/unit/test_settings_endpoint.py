from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from gateway.api.routes import settings as settings_route
from gateway.api.routes.settings import _config_fields
from gateway.core.config import GatewayConfig
from gateway.main import create_app
from gateway.services import master_key_service
from gateway.services.pricing_service import configure_default_pricing, default_pricing_enabled

AUTH = {"Authorization": "Bearer sk-test-master"}


def _client(tmp_path: Path, *, default_pricing: bool = False, require_pricing: bool = True) -> TestClient:
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'settings-test.db'}",
        master_key="sk-test-master",
        default_pricing=default_pricing,
        require_pricing=require_pricing,
    )
    return TestClient(create_app(config))


def test_settings_requires_auth(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.get("/v1/settings").status_code == 401


def test_settings_rejects_non_master_key(tmp_path: Path) -> None:
    # The settings route is admin-only: a token that is not the master key is rejected.
    with _client(tmp_path) as client:
        response = client.get("/v1/settings", headers={"Authorization": "Bearer not-the-master-key"})
    assert response.status_code == 401


def test_rotate_master_key_rejects_configured_key(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post("/v1/settings/master-key/rotate", headers=AUTH)
    assert response.status_code == 409
    assert "configured master key" in response.json()["detail"]


def test_rotate_generated_master_key_invalidates_old_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tokens = iter(["otari-mk-old", "otari-mk-new"])
    monkeypatch.setattr(master_key_service, "generate_master_key", lambda: next(tokens))
    config = GatewayConfig(database_url=f"sqlite:///{tmp_path / 'generated-master.db'}", require_pricing=False)

    with TestClient(create_app(config)) as client:
        old_auth = {"Authorization": "Bearer otari-mk-old"}
        rotated = client.post("/v1/settings/master-key/rotate", headers=old_auth)
        assert rotated.status_code == 200, rotated.text
        assert rotated.json() == {"master_key": "otari-mk-new"}

        old_response = client.get("/v1/settings", headers=old_auth)
        assert old_response.status_code == 401
        new_response = client.get("/v1/settings", headers={"Authorization": "Bearer otari-mk-new"})
        assert new_response.status_code == 200
        assert new_response.json()["master_key_source"] == "generated"


def test_rotate_generated_master_key_rejects_a_stale_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(master_key_service, "generate_master_key", lambda: "otari-mk-old")

    async def _stale_rotation(*_: object) -> tuple[str, str]:
        raise master_key_service.MasterKeyRotationConflictError(
            "The master key was already rotated. Reload and try again."
        )

    monkeypatch.setattr(settings_route, "stage_generated_master_key_rotation", _stale_rotation)
    config = GatewayConfig(database_url=f"sqlite:///{tmp_path / 'stale-master.db'}", require_pricing=False)

    with TestClient(create_app(config)) as client:
        response = client.post("/v1/settings/master-key/rotate", headers={"Authorization": "Bearer otari-mk-old"})
    assert response.status_code == 409
    assert response.json()["detail"] == "The master key was already rotated. Reload and try again."


def test_rotation_invalidates_the_old_generated_key_on_another_replica(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokens = iter(["otari-mk-old", "otari-mk-new"])
    monkeypatch.setattr(master_key_service, "generate_master_key", lambda: next(tokens))
    database_url = f"sqlite:///{tmp_path / 'shared-master.db'}"
    first = GatewayConfig(database_url=database_url, require_pricing=False)
    second = GatewayConfig(database_url=database_url, require_pricing=False)

    with TestClient(create_app(first)) as first_client:
        with TestClient(create_app(second)) as second_client:
            old_auth = {"Authorization": "Bearer otari-mk-old"}
            rotated = first_client.post("/v1/settings/master-key/rotate", headers=old_auth)
            assert rotated.status_code == 200, rotated.text

            assert second_client.get("/v1/settings", headers=old_auth).status_code == 401
            assert second_client.get(
                "/v1/settings", headers={"Authorization": "Bearer otari-mk-new"}
            ).status_code == 200


def test_settings_reports_pricing_flags(tmp_path: Path) -> None:
    with _client(tmp_path, default_pricing=True, require_pricing=False) as client:
        response = client.get("/v1/settings", headers={"Authorization": "Bearer sk-test-master"})

    assert response.status_code == 200
    body = response.json()
    assert body["default_pricing"] is True
    assert body["require_pricing"] is False
    assert body["mode"] == "standalone"
    assert body["master_key_source"] == "configured"
    assert "version" in body


def test_settings_defaults(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.get("/v1/settings", headers={"Authorization": "Bearer sk-test-master"})

    assert response.status_code == 200
    body = response.json()
    # default_pricing is off by default; require_pricing is fail-closed by default.
    assert body["default_pricing"] is False
    assert body["require_pricing"] is True
    assert body["master_key_source"] == "configured"


def test_settings_patch_does_not_apply_when_commit_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A toggle that is persisted but never committed must not mutate this
    # worker's in-memory config or the process-wide pricing flag; otherwise a
    # failed write would leave the gateway metering against an unpersisted value.
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'settings-rollback.db'}",
        master_key="sk-test-master",
        default_pricing=False,
        require_pricing=False,
    )
    configure_default_pricing(False)  # establish a known baseline for the global flag

    async def _boom(self: object) -> None:
        raise OperationalError("commit failed", None, Exception("boom"))

    # raise_server_exceptions=False so we observe the 500 the operator would see,
    # rather than the exception being re-raised into the test.
    with TestClient(create_app(config), raise_server_exceptions=False) as client:
        monkeypatch.setattr("sqlalchemy.ext.asyncio.AsyncSession.commit", _boom)
        response = client.patch(
            "/v1/settings",
            headers={"Authorization": "Bearer sk-test-master"},
            json={"default_pricing": True},
        )

    assert response.status_code == 500
    # The in-memory config and the global pricing flag stay at their pre-request value.
    assert config.default_pricing is False
    assert default_pricing_enabled() is False


def test_settings_includes_full_config_view(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        body = client.get("/v1/settings", headers=AUTH).json()

    fields = body["config"]
    by_key = {field["key"]: field for field in fields}

    # The startup-only fields the issue calls out are present and clearly not settable.
    startup_only = (
        "host",
        "port",
        "database_url",
        "mode",
        "db_pool_size",
        "cors_allow_origins",
        "budget_strategy",
        "rate_limit_rpm",
    )
    for name in startup_only:
        assert name in by_key, name
        assert by_key[name]["settable"] is False, name

    # The widened settable set is present and marked settable.
    for name in (
        "model_discovery",
        "default_pricing",
        "require_pricing",
        "reject_user_mismatch",
        "model_cache_ttl_seconds",
        "stream_missing_usage_policy",
        "model_discovery_negative_ttl_seconds",
        "model_discovery_timeout_seconds",
        "models_dev_metadata",
        "models_dev_cache_ttl_seconds",
        "file_understanding_enabled",
        "vision_strategy",
        "vision_describe_model",
        "vision_describe_max_tokens",
        "budget_estimate_default_output_tokens",
    ):
        assert by_key[name]["settable"] is True, name

    # The outbound network-safety gates stay read-only on purpose.
    for name in ("mcp_allow_private_hosts", "web_search_allow_private_hosts", "sandbox_url", "guardrails_url"):
        assert by_key[name]["settable"] is False, name

    # Fields carry a group, description, and a display type.
    assert by_key["require_pricing"]["group"] == "Metering & budgets"
    assert by_key["require_pricing"]["description"]
    assert by_key["port"]["type"] == "int"
    assert by_key["cors_allow_origins"]["type"] == "list"
    assert by_key["stream_missing_usage_policy"]["type"] == "str"
    assert by_key["model_discovery_timeout_seconds"]["type"] == "float"
    # Enum fields carry their options so the dashboard can render a picker.
    assert by_key["vision_strategy"]["options"] == ["describe", "ocr", "off"]

    # Secrets and complex catalog fields are never surfaced here.
    for secret in ("master_key", "providers", "pricing", "aliases", "platform"):
        assert secret not in by_key, secret


def test_config_view_redacts_url_credentials() -> None:
    # A production database_url (or sandbox/guardrails url) can embed a secret;
    # the config view must show the host/db but never echo the secret back, even
    # to the master-key holder. Built directly (no app), so no live DB is needed.
    config = GatewayConfig(
        database_url="postgresql+asyncpg://otari:s3cr3t-pw@db.internal:5432/otari",
        sandbox_url="https://svc:tok3n@sandbox.internal",
    )
    by_key = {field.key: field.value for field in _config_fields(config)}

    assert by_key["database_url"] == "postgresql+asyncpg://otari:***@db.internal:5432/otari"
    assert "s3cr3t-pw" not in str(by_key["database_url"])
    assert by_key["sandbox_url"] == "https://svc:***@sandbox.internal"
    assert "tok3n" not in str(by_key["sandbox_url"])
    # A credential-free URL (the sqlite default) is left untouched.
    assert GatewayConfig().database_url.startswith("sqlite:///")


def test_config_view_redacts_token_in_username_and_query() -> None:
    # A bearer token can live in the username position (no colon) or in a query
    # param, not just the password slot. All must be masked. Query values are
    # masked regardless of the param name (a denylist of key names cannot be
    # complete), while the keys stay visible.
    config = GatewayConfig(
        guardrails_url="https://s3cr3t-token@guardrails.internal/scan",
        sandbox_url="https://sandbox.internal/run?client_secret=leaked&mode=fast",
    )
    by_key = {field.key: str(field.value) for field in _config_fields(config)}

    assert "s3cr3t-token" not in by_key["guardrails_url"]
    assert by_key["guardrails_url"] == "https://***@guardrails.internal/scan"
    # A param name not on any keyword list is still masked; keys stay visible.
    assert "leaked" not in by_key["sandbox_url"]
    assert by_key["sandbox_url"] == "https://sandbox.internal/run?client_secret=***&mode=***"
    # The mask is human-readable, not percent-encoded (%2A).
    assert "%2A" not in by_key["sandbox_url"]


def test_config_view_preserves_ipv6_host_when_masking() -> None:
    # Masking the userinfo must not corrupt an IPv6 host literal (keep brackets).
    config = GatewayConfig(database_url="postgresql+asyncpg://u:p@[2001:db8::1]:5432/otari")
    by_key = {field.key: str(field.value) for field in _config_fields(config)}
    assert by_key["database_url"] == "postgresql+asyncpg://u:***@[2001:db8::1]:5432/otari"


def test_config_view_exposes_numeric_bounds() -> None:
    # Settable numeric fields carry their lower bound so the dashboard can gate a
    # number input the same way the backend validator does.
    by_key = {field.key: field for field in _config_fields(GatewayConfig())}

    # gt=0 field: exclusive lower bound of 0.
    assert by_key["model_discovery_timeout_seconds"].exclusive_minimum == 0
    assert by_key["model_discovery_timeout_seconds"].minimum is None
    # ge=0 field: inclusive lower bound of 0.
    assert by_key["model_cache_ttl_seconds"].minimum == 0
    assert by_key["model_cache_ttl_seconds"].exclusive_minimum is None
    # A read-only field carries no bounds.
    assert by_key["host"].minimum is None
    assert by_key["host"].exclusive_minimum is None


def test_patch_applies_widened_settable_fields(tmp_path: Path) -> None:
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'settings-widen.db'}",
        master_key="sk-test-master",
        require_pricing=True,
        reject_user_mismatch=True,
        model_cache_ttl_seconds=300,
        stream_missing_usage_policy="estimate",
    )
    with TestClient(create_app(config)) as client:
        response = client.patch(
            "/v1/settings",
            headers=AUTH,
            json={
                "require_pricing": False,
                "reject_user_mismatch": False,
                "model_cache_ttl_seconds": 30,
                "stream_missing_usage_policy": "fail",
            },
        )

    assert response.status_code == 200
    # Each hot-changeable field mutated the running config immediately.
    assert config.require_pricing is False
    assert config.reject_user_mismatch is False
    assert config.model_cache_ttl_seconds == 30
    assert config.stream_missing_usage_policy == "fail"

    by_key = {field["key"]: field for field in response.json()["config"]}
    assert by_key["require_pricing"]["value"] is False
    assert by_key["model_cache_ttl_seconds"]["value"] == 30
    assert by_key["stream_missing_usage_policy"]["value"] == "fail"


def test_patch_applies_float_and_clears_nullable_field(tmp_path: Path) -> None:
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'settings-float.db'}",
        master_key="sk-test-master",
        model_discovery_negative_ttl_seconds=30.0,
        vision_describe_model="ollama/qwen2-vl",
    )
    with TestClient(create_app(config)) as client:
        response = client.patch(
            "/v1/settings",
            headers=AUTH,
            json={
                "model_discovery_negative_ttl_seconds": 5.5,
                # An explicit null clears the describe model (distinct from omitting it).
                "vision_describe_model": None,
            },
        )

    assert response.status_code == 200
    assert config.model_discovery_negative_ttl_seconds == 5.5
    assert config.vision_describe_model is None

    by_key = {field["key"]: field for field in response.json()["config"]}
    assert by_key["model_discovery_negative_ttl_seconds"]["value"] == 5.5
    assert by_key["vision_describe_model"]["value"] is None


def test_new_type_overrides_survive_restart(tmp_path: Path) -> None:
    # The serialize -> DB string -> parse -> apply roundtrip must hold for the
    # non-bool types too. Write overrides through one app, then start a fresh app
    # (fresh config) against the same database: apply_overrides_from_db runs on
    # startup and must reconstruct each typed value exactly.
    db_url = f"sqlite:///{tmp_path / 'settings-restart.db'}"

    first = GatewayConfig(database_url=db_url, master_key="sk-test-master")
    with TestClient(create_app(first)) as client:
        response = client.patch(
            "/v1/settings",
            headers=AUTH,
            json={
                "model_cache_ttl_seconds": 45,
                "model_discovery_negative_ttl_seconds": 7.5,
                "vision_strategy": "ocr",
                "vision_describe_model": "ollama/qwen2-vl",
                "models_dev_metadata": False,
            },
        )
        assert response.status_code == 200

    # A brand-new config object, so nothing carries over in memory: the values
    # can only come from the persisted overrides applied at startup.
    second = GatewayConfig(database_url=db_url, master_key="sk-test-master")
    assert second.model_cache_ttl_seconds != 45  # sanity: not already the target
    with TestClient(create_app(second)):
        pass
    assert second.model_cache_ttl_seconds == 45
    assert second.model_discovery_negative_ttl_seconds == 7.5
    assert second.vision_strategy == "ocr"
    assert second.vision_describe_model == "ollama/qwen2-vl"
    assert second.models_dev_metadata is False


def test_patch_ignores_startup_only_field(tmp_path: Path) -> None:
    # A startup-only field is not part of the writable schema, so it is ignored
    # (never applied) rather than silently mutating the running config.
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'settings-startup.db'}",
        master_key="sk-test-master",
        host="0.0.0.0",  # noqa: S104
    )
    with TestClient(create_app(config)) as client:
        response = client.patch("/v1/settings", headers=AUTH, json={"host": "10.0.0.1"})

    assert response.status_code == 200
    assert config.host == "0.0.0.0"  # noqa: S104


def test_patch_rejects_invalid_stream_policy(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.patch(
            "/v1/settings",
            headers=AUTH,
            json={"stream_missing_usage_policy": "bogus"},
        )
    # Rejected by the request schema (Literal), before any write.
    assert response.status_code == 422


def test_patch_rejects_negative_cache_ttl(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.patch(
            "/v1/settings",
            headers=AUTH,
            json={"model_cache_ttl_seconds": -1},
        )
    assert response.status_code == 422
