import base64
import os
from pathlib import Path
from unittest import mock

import pytest

import gateway.core.config as config_module
from gateway.core.config import load_config
from gateway.core.env import otari_env
from gateway.services.url_safety import UnsafeURLError, validate_mcp_url, validate_outbound_fetch_url


def test_load_config_loads_provider_env_vars_from_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=from-dotenv\nOTARI_MASTER_KEY=gateway-master\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OTARI_MASTER_KEY", raising=False)

    config = load_config()

    assert os.getenv("ANTHROPIC_API_KEY") == "from-dotenv"
    assert config.master_key == "gateway-master"


def test_load_config_does_not_override_existing_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=from-dotenv\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "already-set")

    load_config()

    assert os.getenv("ANTHROPIC_API_KEY") == "already-set"


def test_load_config_prefers_dotenv_near_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / ".env").write_text("ANTHROPIC_API_KEY=from-config-dir\n", encoding="utf-8")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=from-cwd\n", encoding="utf-8")
    config_file = config_dir / "gateway.yml"
    config_file.write_text("providers:\n  anthropic:\n    api_key: ${ANTHROPIC_API_KEY}\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    config = load_config(str(config_file))

    assert os.getenv("ANTHROPIC_API_KEY") == "from-config-dir"
    assert config.providers["anthropic"]["api_key"] == "from-config-dir"


def test_load_config_skips_duplicate_dotenv_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OTARI_MASTER_KEY=gateway-master\n", encoding="utf-8")
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("{}\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    calls: list[Path] = []

    def _fake_load_dotenv(*, dotenv_path: Path, override: bool) -> None:
        calls.append(dotenv_path)
        assert override is False

    monkeypatch.setattr(config_module, "load_dotenv", _fake_load_dotenv)

    load_config(str(config_file))

    assert calls == [env_file]


def test_load_config_reads_otari_prefixed_env_aliases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("OTARI_MASTER_KEY", "otari-master")
    monkeypatch.setenv("OTARI_DATABASE_URL", "sqlite:///./otari.db")
    monkeypatch.setenv("OTARI_HOST", "127.0.0.1")
    monkeypatch.setenv("OTARI_PORT", "9001")
    monkeypatch.setenv("OTARI_AUTO_MIGRATE", "false")
    monkeypatch.setenv("OTARI_BOOTSTRAP_API_KEY", "false")

    config = load_config(str(config_file))

    assert config.master_key == "otari-master"
    assert config.database_url == "sqlite:///./otari.db"
    assert config.host == "127.0.0.1"
    assert config.port == 9001
    assert config.auto_migrate is False
    assert config.bootstrap_api_key is False


def test_load_config_ignores_legacy_gateway_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The GATEWAY_ prefix was removed after the Otari rename deprecation window;
    # it no longer configures anything, so a value set only under GATEWAY_ falls
    # back to the field default.
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("{}\n", encoding="utf-8")

    monkeypatch.delenv("OTARI_PORT", raising=False)
    monkeypatch.delenv("OTARI_BUDGET_STRATEGY", raising=False)
    monkeypatch.setenv("GATEWAY_PORT", "7000")
    monkeypatch.setenv("GATEWAY_BUDGET_STRATEGY", "cas")

    config = load_config(str(config_file))

    assert config.port == 8000
    assert config.budget_strategy == "for_update"


def test_load_config_otari_prefix_covers_all_scalar_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fields beyond the original hand-listed aliases must also resolve from OTARI_*.
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("OTARI_BUDGET_STRATEGY", "cas")
    monkeypatch.setenv("OTARI_REQUIRE_PRICING", "false")
    monkeypatch.setenv("OTARI_MODEL_CACHE_TTL_SECONDS", "42")
    monkeypatch.setenv("OTARI_DB_POOL_TIMEOUT", "12.5")

    config = load_config(str(config_file))

    assert config.budget_strategy == "cas"
    assert config.require_pricing is False
    assert config.model_cache_ttl_seconds == 42
    assert config.db_pool_timeout == 12.5


def test_load_config_promotes_service_level_fields_from_otari_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The service-level knobs that used to be read only via otari_env() deep in
    # route/service code are now typed GatewayConfig fields, validated at startup
    # and populated by the same OTARI_ override machinery as every scalar field.
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("OTARI_SANDBOX_URL", "http://sandbox:9000")
    monkeypatch.setenv("OTARI_GUARDRAILS_URL", "http://guardrails:8000")
    monkeypatch.setenv("OTARI_WEB_SEARCH_MAX_RESULTS", "7")
    monkeypatch.setenv("OTARI_WEB_SEARCH_EXTRACT", "false")
    monkeypatch.setenv("OTARI_MCP_ALLOW_LOOPBACK", "false")

    config = load_config(str(config_file))

    assert config.sandbox_url == "http://sandbox:9000"
    assert config.guardrails_url == "http://guardrails:8000"
    assert config.web_search_max_results == 7
    assert config.web_search_extract is False
    assert config.mcp_allow_loopback is False


def test_load_config_service_level_fields_ignore_legacy_gateway_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The legacy GATEWAY_ names for these service-level knobs were removed and no
    # longer populate the fields; the defaults stand.
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("{}\n", encoding="utf-8")

    monkeypatch.delenv("OTARI_SANDBOX_URL", raising=False)
    monkeypatch.delenv("OTARI_MCP_ALLOW_PRIVATE_HOSTS", raising=False)
    monkeypatch.setenv("GATEWAY_SANDBOX_URL", "http://legacy-sandbox:9000")
    monkeypatch.setenv("GATEWAY_MCP_ALLOW_PRIVATE_HOSTS", "true")

    config = load_config(str(config_file))

    assert config.sandbox_url is None
    assert config.mcp_allow_private_hosts is False


def test_load_config_rejects_invalid_service_level_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The promotion buys startup validation: a value the old code silently
    # ignored (a non-int web-search cap) now fails fast at config load.
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("OTARI_WEB_SEARCH_MAX_RESULTS", "not-an-int")

    with pytest.raises(ValueError):
        load_config(str(config_file))


def test_load_config_bridges_yaml_service_fields_into_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The otari_env() read sites only see environment variables, so load_config
    # bridges YAML-set promoted fields into the process env; without this a
    # YAML-set sandbox_url would validate at startup and be ignored at request
    # time.
    config_file = tmp_path / "gateway.yml"
    config_file.write_text(
        "sandbox_url: http://yaml-sandbox:9000\nmcp_allow_loopback: false\nweb_search_max_results: 5\n",
        encoding="utf-8",
    )
    for var in (
        "OTARI_SANDBOX_URL",
        "OTARI_MCP_ALLOW_LOOPBACK",
        "OTARI_WEB_SEARCH_MAX_RESULTS",
    ):
        monkeypatch.delenv(var, raising=False)

    with mock.patch.dict(os.environ):
        config = load_config(str(config_file))

        assert config.sandbox_url == "http://yaml-sandbox:9000"
        assert otari_env("SANDBOX_URL") == "http://yaml-sandbox:9000"
        assert otari_env("MCP_ALLOW_LOOPBACK") == "false"
        assert otari_env("WEB_SEARCH_MAX_RESULTS") == "5"


def test_load_config_env_wins_over_yaml_for_bridged_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The bridge uses setdefault: an OTARI_ variable that is already set keeps
    # its value (and also wins in the field, via the env override machinery).
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("sandbox_url: http://yaml-sandbox:9000\n", encoding="utf-8")

    monkeypatch.setenv("OTARI_SANDBOX_URL", "http://env-sandbox:9000")

    config = load_config(str(config_file))

    assert config.sandbox_url == "http://env-sandbox:9000"
    assert otari_env("SANDBOX_URL") == "http://env-sandbox:9000"


def test_load_config_does_not_bridge_fields_absent_from_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The bridge only injects YAML-set promoted fields into the environment. A
    # field populated only from its OTARI_ env var is already visible to the
    # otari_env() read sites, so load_config leaves the environment untouched.
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("OTARI_SANDBOX_URL", "http://env-sandbox:9000")

    with mock.patch.dict(os.environ):
        config = load_config(str(config_file))

        assert config.sandbox_url == "http://env-sandbox:9000"
        assert otari_env("SANDBOX_URL") == "http://env-sandbox:9000"


@pytest.mark.asyncio
async def test_yaml_ssrf_gates_round_trip_through_env_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A YAML-set SSRF gate boolean must reach the url_safety gate parsers with
    # the spelling they accept: `true` opens the private-host gates for both
    # the MCP and web-search fetch paths.
    config_file = tmp_path / "gateway.yml"
    config_file.write_text(
        "mcp_allow_private_hosts: true\nweb_search_allow_private_hosts: true\n",
        encoding="utf-8",
    )
    for var in (
        "OTARI_MCP_ALLOW_PRIVATE_HOSTS",
        "OTARI_WEB_SEARCH_ALLOW_PRIVATE_HOSTS",
    ):
        monkeypatch.delenv(var, raising=False)

    with mock.patch.dict(os.environ):
        load_config(str(config_file))

        await validate_mcp_url("https://10.0.0.5/mcp", has_authorization_token=False)
        await validate_outbound_fetch_url("https://10.0.0.5/page")

    # Outside the bridged environment the gates are closed again.
    with pytest.raises(UnsafeURLError, match="private"):
        await validate_outbound_fetch_url("https://10.0.0.5/page")


def test_load_config_otari_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("budget_strategy: for_update\n", encoding="utf-8")

    monkeypatch.setenv("OTARI_BUDGET_STRATEGY", "disabled")

    config = load_config(str(config_file))

    assert config.budget_strategy == "disabled"


def test_load_config_platform_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("mode: hybrid\n", encoding="utf-8")

    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    monkeypatch.setenv("PLATFORM_BASE_URL", "http://localhost:8100/api/v1")
    monkeypatch.setenv("PLATFORM_RESOLVE_TIMEOUT_MS", "1234")
    monkeypatch.setenv("PLATFORM_USAGE_TIMEOUT_MS", "2345")
    monkeypatch.setenv("PLATFORM_USAGE_MAX_RETRIES", "7")

    config = load_config(str(config_file))

    assert config.is_hybrid_mode
    assert config.platform["base_url"] == "http://localhost:8100/api/v1"
    assert config.platform["resolve_timeout_ms"] == 1234
    assert config.platform["usage_timeout_ms"] == 2345
    assert config.platform["usage_max_retries"] == 7


def test_load_config_sets_default_platform_base_url_when_token_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("mode: hybrid\n", encoding="utf-8")

    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    monkeypatch.delenv("PLATFORM_BASE_URL", raising=False)

    config = load_config(str(config_file))

    assert config.is_hybrid_mode
    assert config.platform["base_url"] == "https://api.otari.ai/api/v1"


def test_load_config_rejects_hybrid_mode_without_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("mode: hybrid\n", encoding="utf-8")
    monkeypatch.delenv("OTARI_AI_TOKEN", raising=False)

    with pytest.raises(ValueError, match="OTARI_AI_TOKEN"):
        load_config(str(config_file))


def test_load_config_derives_hybrid_mode_from_token_when_mode_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Token present and no explicit mode: the runtime mode is derived as hybrid.
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")

    config = load_config(str(config_file))

    assert config.mode is None
    assert config.is_hybrid_mode
    assert config.effective_mode == "hybrid"


def test_load_config_rejects_standalone_mode_when_token_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Explicitly asserting standalone while a platform token selects hybrid is a
    # conflicting configuration and must fail at startup rather than silently
    # running hybrid.
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("mode: standalone\n", encoding="utf-8")
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")

    with pytest.raises(ValueError, match="conflicts with OTARI_AI_TOKEN"):
        load_config(str(config_file))


def test_load_config_rejects_standalone_mode_from_env_when_token_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The conflict check also covers OTARI_MODE supplied via the environment.
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("OTARI_MODE", "standalone")
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")

    with pytest.raises(ValueError, match="conflicts with OTARI_AI_TOKEN"):
        load_config(str(config_file))


def test_load_config_honors_explicit_standalone_without_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("mode: standalone\n", encoding="utf-8")
    monkeypatch.delenv("OTARI_AI_TOKEN", raising=False)

    config = load_config(str(config_file))

    assert config.mode == "standalone"
    assert not config.is_hybrid_mode
    assert config.effective_mode == "standalone"


def test_load_config_unset_mode_without_token_is_standalone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("", encoding="utf-8")
    monkeypatch.delenv("OTARI_AI_TOKEN", raising=False)

    config = load_config(str(config_file))

    assert config.mode is None
    assert not config.is_hybrid_mode
    assert config.effective_mode == "standalone"


def test_load_config_accepts_legacy_platform_mode_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # "platform" is the pre-rename alias for the "hybrid" mode; it must keep
    # working so existing configs do not break.
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("mode: platform\n", encoding="utf-8")
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")

    config = load_config(str(config_file))

    assert config.is_hybrid_mode
    assert config.effective_mode == "hybrid"


@pytest.mark.parametrize("legacy_token_var", ["OTARI_PLATFORM_TOKEN", "ANY_LLM_PLATFORM_TOKEN"])
def test_load_config_ignores_legacy_platform_token_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_token_var: str
) -> None:
    # The pre-rename platform-token env aliases were removed: only OTARI_AI_TOKEN
    # selects hybrid mode now, so a legacy alias alone leaves the gateway
    # standalone.
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("", encoding="utf-8")

    monkeypatch.delenv("OTARI_AI_TOKEN", raising=False)
    monkeypatch.setenv(legacy_token_var, "legacy-token")

    config = load_config(str(config_file))

    assert not config.is_hybrid_mode
    assert config.platform_token is None


def _isolate_structured_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Clear any structured-config vars and run from an empty dir so load_config()
    # cannot pick up a developer's repo-root .env.
    monkeypatch.delenv("OTARI_CONFIG_YAML", raising=False)
    monkeypatch.delenv("OTARI_CONFIG_B64", raising=False)
    monkeypatch.chdir(tmp_path)


def test_load_config_reads_full_provider_pricing_from_env_yaml_without_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Env-only deploy: full provider + pricing config, no config.yml present.
    _isolate_structured_env(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "OTARI_CONFIG_YAML",
        (
            "providers:\n"
            "  openai:\n"
            "    api_key: env-openai-key\n"
            "    api_base: https://proxy.example/v1\n"
            "pricing:\n"
            "  openai:gpt-4o:\n"
            "    input_price_per_million: 2.5\n"
            "    output_price_per_million: 10\n"
        ),
    )

    config = load_config()

    assert config.providers["openai"]["api_key"] == "env-openai-key"
    assert config.providers["openai"]["api_base"] == "https://proxy.example/v1"
    assert config.pricing["openai:gpt-4o"].input_price_per_million == 2.5
    assert config.pricing["openai:gpt-4o"].output_price_per_million == 10


def test_load_config_structured_env_base64(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_structured_env(monkeypatch, tmp_path)
    yaml_text = "providers:\n  mistral:\n    api_key: b64-key\n"
    monkeypatch.setenv("OTARI_CONFIG_B64", base64.b64encode(yaml_text.encode("utf-8")).decode("ascii"))

    config = load_config()

    assert config.providers["mistral"]["api_key"] == "b64-key"


def test_load_config_structured_env_base64_tolerates_newline_wrapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The standard `base64` CLI and many env-var UIs wrap output at 76 columns.
    _isolate_structured_env(monkeypatch, tmp_path)
    yaml_text = (
        "providers:\n"
        "  openai:\n"
        "    api_key: a-fairly-long-key-value-to-force-base64-line-wrapping-aaaaaaaaaaaaaa\n"
    )
    wrapped = base64.encodebytes(yaml_text.encode("utf-8")).decode("ascii")
    assert "\n" in wrapped.strip()
    monkeypatch.setenv("OTARI_CONFIG_B64", wrapped)

    config = load_config()

    assert config.providers["openai"]["api_key"].startswith("a-fairly-long-key")


def test_load_config_structured_env_prefers_raw_yaml_over_base64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_structured_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OTARI_CONFIG_YAML", "providers:\n  openai:\n    api_key: from-raw\n")
    monkeypatch.setenv(
        "OTARI_CONFIG_B64",
        base64.b64encode(b"providers:\n  openai:\n    api_key: from-b64\n").decode("ascii"),
    )

    config = load_config()

    assert config.providers["openai"]["api_key"] == "from-raw"


def test_load_config_structured_env_whitespace_yaml_falls_back_to_base64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A whitespace-only OTARI_CONFIG_YAML must not suppress the base64 fallback.
    _isolate_structured_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OTARI_CONFIG_YAML", "   \n  ")
    monkeypatch.setenv(
        "OTARI_CONFIG_B64",
        base64.b64encode(b"providers:\n  openai:\n    api_key: from-b64\n").decode("ascii"),
    )

    config = load_config()

    assert config.providers["openai"]["api_key"] == "from-b64"


def test_load_config_precedence_file_then_structured_then_scalar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # file < env-structured < scalar OTARI_<FIELD>
    _isolate_structured_env(monkeypatch, tmp_path)
    config_file = tmp_path / "gateway.yml"
    config_file.write_text(
        "budget_strategy: for_update\nproviders:\n  openai:\n    api_key: file-key\n",
        encoding="utf-8",
    )

    monkeypatch.setenv(
        "OTARI_CONFIG_YAML",
        "budget_strategy: cas\nproviders:\n  openai:\n    api_key: structured-key\n",
    )
    monkeypatch.setenv("OTARI_BUDGET_STRATEGY", "disabled")

    config = load_config(str(config_file))

    # Scalar OTARI_<FIELD> wins over both file and env-structured.
    assert config.budget_strategy == "disabled"
    # env-structured providers win over the file's providers.
    assert config.providers["openai"]["api_key"] == "structured-key"


def test_load_config_structured_env_resolves_var_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_structured_env(monkeypatch, tmp_path)
    monkeypatch.setenv("MY_OPENAI_KEY", "resolved-secret")
    monkeypatch.setenv("OTARI_CONFIG_YAML", "providers:\n  openai:\n    api_key: ${MY_OPENAI_KEY}\n")

    config = load_config()

    assert config.providers["openai"]["api_key"] == "resolved-secret"


def test_load_config_structured_env_invalid_yaml_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_structured_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OTARI_CONFIG_YAML", "providers: [unclosed\n")

    with pytest.raises(ValueError, match="OTARI_CONFIG_YAML is not valid YAML"):
        load_config()


def test_load_config_structured_env_invalid_base64_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_structured_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OTARI_CONFIG_B64", "not!valid!base64!")

    with pytest.raises(ValueError, match="OTARI_CONFIG_B64 is not valid base64"):
        load_config()


def test_load_config_structured_env_non_mapping_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_structured_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OTARI_CONFIG_YAML", "- just\n- a\n- list\n")

    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_config()
