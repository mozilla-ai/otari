"""Every setting keeps the name a deployment configures it by."""

import json
import os
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import pytest
import yaml
from pydantic import BaseModel

from gateway.core.config import GatewayConfig, load_config

# Each name is a config.yml key, and OTARI_<NAME> is its environment variable.
_SETTING_NAMES = frozenset(
    {
        "activation_guide",
        "aliases",
        "auto_migrate",
        "bootstrap",
        "bootstrap_api_key",
        "budget_estimate_default_output_tokens",
        "budget_reservation_retention_sec",
        "budget_reservation_sweep_batch",
        "budget_reservation_sweep_interval_sec",
        "budget_reservation_ttl_sec",
        "budget_strategy",
        "capture_agent_telemetry",
        "code_execution_executor",
        "cors_allow_origins",
        "dashboard_login_rate_limit_per_minute",
        "dashboard_session_ttl_hours",
        "data_plane_url",
        "database_url",
        "db_command_timeout",
        "db_connect_timeout",
        "db_log_pool_size",
        "db_max_overflow",
        "db_pool_recycle",
        "db_pool_size",
        "db_pool_timeout",
        "db_statement_timeout_ms",
        "default_pricing",
        "docs_url",
        "email_verification_expiry_hours",
        "enable_docs",
        "enable_metrics",
        "file_understanding_enabled",
        "files_backend",
        "files_enabled",
        "files_local_dir",
        "files_max_bytes",
        "files_output_max_bytes",
        "files_output_max_files",
        "files_retention_hours",
        "files_s3_bucket",
        "files_s3_endpoint_url",
        "files_s3_region",
        "files_storage_options",
        "files_sweep_interval_sec",
        "files_url",
        "guardrail_thread_pool_size",
        "guardrails_url",
        "host",
        "invitation_expiry_hours",
        "log_writer_strategy",
        "mail_from_email",
        "mail_from_name",
        "mail_transport",
        "master_key",
        "mcp_allow_loopback",
        "mcp_allow_private_hosts",
        "mode",
        "model_cache_ttl_seconds",
        "model_capabilities",
        "model_discovery",
        "model_discovery_negative_ttl_seconds",
        "model_discovery_timeout_seconds",
        "models_dev_cache_ttl_seconds",
        "models_dev_metadata",
        "oauth_github_client_id",
        "oauth_github_client_secret",
        "oauth_google_client_id",
        "oauth_google_client_secret",
        "open_signup",
        "password_reset_expiry_hours",
        "platform",
        "port",
        "pricing",
        "pricing_refresh",
        "pricing_refresh_interval_seconds",
        "privacy_url",
        "provider_allow_private_hosts",
        "providers",
        "public_base_url",
        "public_catalog",
        "public_catalog_rate_limit_per_minute",
        "rate_limit_rpm",
        "reject_user_mismatch",
        "require_pricing",
        "router_alpha",
        "router_confidence_floor",
        "router_embedding_model",
        "router_granularity",
        "router_k",
        "router_max_records_per_user",
        "router_seed_count",
        "routing",
        "sandbox_allowed_session_images",
        "sandbox_container_idle_ttl_sec",
        "sandbox_container_max_lifetime_sec",
        "sandbox_provider",
        "sandbox_purpose_hint",
        "sandbox_session_image",
        "sandbox_url",
        "search_tools",
        "smtp_host",
        "smtp_password",
        "smtp_port",
        "smtp_tls",
        "smtp_user",
        "stream_missing_usage_policy",
        "streaming_keepalive_interval_ms",
        "terms_url",
        "tools_header",
        "ui_base_url",
        "vision_describe_max_tokens",
        "vision_describe_model",
        "vision_strategy",
        "web_fetch_enabled",
        "web_retrieval_trust_env_proxy",
        "web_search_allow_private_hosts",
        "web_search_backend_token",
        "web_search_engines",
        "web_search_extract",
        "web_search_intercept",
        "web_search_max_results",
        "web_search_provider",
        "web_search_provider_api_key",
        "web_search_purpose_hint",
        "web_search_url",
        "webauthn_allowed_origins",
        "webauthn_rp_id",
        "webauthn_rp_name",
    }
)


@pytest.fixture(autouse=True)
def _isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Hide OTARI_ variables and .env files, and restore os.environ, which loading writes to."""
    monkeypatch.chdir(tmp_path)
    with mock.patch.dict(os.environ):
        for key in [key for key in os.environ if key.startswith("OTARI_")]:
            del os.environ[key]
        yield


def _default(name: str) -> object:
    value = GatewayConfig.model_fields[name].get_default(call_default_factory=True)
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def _environment_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict | list):
        return json.dumps(value)
    return str(value)


def test_every_setting_keeps_its_name() -> None:
    assert set(GatewayConfig.model_fields) == _SETTING_NAMES


def test_an_empty_config_sets_nothing(tmp_path: Path) -> None:
    """The checks below read ``model_fields_set``, so an empty load must leave it empty."""
    config_file = tmp_path / "config.yml"
    config_file.write_text("{}\n", encoding="utf-8")

    assert load_config(str(config_file)).model_fields_set == set()


@pytest.mark.parametrize("name", sorted(GatewayConfig.model_fields))
def test_every_setting_is_read_from_its_yaml_key(name: str, tmp_path: Path) -> None:
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.safe_dump({name: _default(name)}), encoding="utf-8")

    assert name in load_config(str(config_file)).model_fields_set


@pytest.mark.parametrize("name", sorted(name for name in GatewayConfig.model_fields if _default(name) is not None))
def test_every_setting_with_a_default_is_read_from_its_environment_variable(name: str) -> None:
    """Skips settings whose default is None: an environment variable cannot hold None."""
    os.environ[f"OTARI_{name.upper()}"] = _environment_value(_default(name))

    assert name in load_config().model_fields_set
