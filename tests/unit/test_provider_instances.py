"""Unit tests for named provider-instance resolution (mozilla-ai/otari#213)."""

from __future__ import annotations

import logging

import pytest
import yaml
from any_llm import LLMProvider
from any_llm.exceptions import AnyLLMError

from gateway.api.routes.pricing import _candidate_model_keys
from gateway.core.config import GatewayConfig, ModelCapabilityConfig, provider_credential_env_names
from gateway.log_config import logger as gateway_logger
from gateway.services.model_capabilities import resolve_capabilities
from gateway.services.provider_kwargs import (
    _KEYLESS_PLACEHOLDER_API_KEY,
    get_provider_kwargs,
    keyless_placeholder_api_key,
    normalize_pricing_key,
    resolve_provider_selector,
    split_selector,
)

# ---------------------------------------------------------------------------
# config.provider_instance_type
# ---------------------------------------------------------------------------


def test_instance_type_defaults_to_key() -> None:
    config = GatewayConfig(providers={"openai": {"api_key": "sk"}})
    assert config.provider_instance_type("openai") == "openai"


def test_instance_type_uses_declared_provider_type() -> None:
    config = GatewayConfig(providers={"home_lab": {"provider_type": "openai", "api_base": "http://x/v1"}})
    assert config.provider_instance_type("home_lab") == "openai"


def test_instance_type_normalizes_openai_compatible_alias() -> None:
    config = GatewayConfig(providers={"home_lab": {"provider_type": "openai-compatible"}})
    assert config.provider_instance_type("home_lab") == "openai"
    config2 = GatewayConfig(providers={"home_lab": {"provider_type": "openai_compatible"}})
    assert config2.provider_instance_type("home_lab") == "openai"


def test_instance_type_normalizes_anthropic_compatible_alias() -> None:
    config = GatewayConfig(providers={"proxy": {"provider_type": "anthropic-compatible"}})
    assert config.provider_instance_type("proxy") == "anthropic"
    config2 = GatewayConfig(providers={"proxy": {"provider_type": "anthropic_compatible"}})
    assert config2.provider_instance_type("proxy") == "anthropic"


def test_instance_type_unknown_instance_returns_input() -> None:
    assert GatewayConfig().provider_instance_type("anthropic") == "anthropic"


# ---------------------------------------------------------------------------
# config.validate_provider_instances
# ---------------------------------------------------------------------------


def test_validate_rejects_unknown_provider_type() -> None:
    config = GatewayConfig(providers={"home_lab": {"provider_type": "not-a-real-provider"}})
    with pytest.raises(ValueError, match="not a known provider"):
        config.validate_provider_instances()


def test_validate_accepts_alias_provider_type() -> None:
    config = GatewayConfig(providers={"home_lab": {"provider_type": "openai-compatible"}})
    config.validate_provider_instances()  # no raise


def test_validate_rejects_non_list_models() -> None:
    config = GatewayConfig(providers={"home_lab": {"provider_type": "openai", "models": "deepseek"}})
    with pytest.raises(ValueError, match="must be a list"):
        config.validate_provider_instances()


def test_validate_allows_instance_without_provider_type() -> None:
    # Backward compatible: keys that are real providers need no provider_type and
    # are not hard-validated here.
    GatewayConfig(providers={"openai": {"api_key": "sk"}}).validate_provider_instances()


def test_valueless_entry_loads_as_empty_config() -> None:
    # Regression (mozilla-ai/otari#389): a keyless local backend has no credential
    # to declare, so `ollama:` with no body is the natural config. YAML parses that
    # as None, which the dict[str, dict] annotation rejected outright.
    config = GatewayConfig(providers={"ollama": None})  # type: ignore[dict-item]
    assert config.providers == {"ollama": {}}
    config.validate_provider_instances()  # no raise


def test_valueless_providers_block_loads_as_no_providers() -> None:
    # Commenting out every entry leaves a bare `providers:`, which YAML also reads
    # as None. That should mean "no providers", not the same pydantic type error.
    config = GatewayConfig(**yaml.safe_load("providers:\n"))
    assert config.providers == {}


def test_valueless_entry_from_yaml_is_a_configured_instance() -> None:
    # The whole point of the entry is opting the local backend into discovery,
    # which is scoped to config.providers, so the key must survive the load.
    config = GatewayConfig(**yaml.safe_load("providers:\n  ollama:\n"))
    assert "ollama" in config.providers
    assert config.provider_instance_type("ollama") == "ollama"


def _capture_gateway_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Route the ``gateway`` logger (which does not propagate) into caplog."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="gateway")


def _validate_capturing_warnings(config: GatewayConfig, caplog: pytest.LogCaptureFixture) -> str:
    _capture_gateway_logs(caplog)
    try:
        config.validate_provider_instances()  # no raise
    finally:
        gateway_logger.removeHandler(caplog.handler)
    return caplog.text


def test_bare_keyless_local_entry_warns_nothing(caplog: pytest.LogCaptureFixture) -> None:
    # A keyless backend has no credential to declare, so the bare entry is the
    # intended config: it must stay silent, and stay discoverable.
    config = GatewayConfig(**yaml.safe_load("providers:\n  ollama:\n  llamacpp:\n  llamafile:\n"))
    assert _validate_capturing_warnings(config, caplog) == ""
    assert set(config.providers) == {"ollama", "llamacpp", "llamafile"}


def test_bare_keyed_entry_warns(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    # `openai:` with nothing beneath it used to be a startup type error. It now
    # loads, so warn: with no credential anywhere it is far more likely a
    # truncated YAML edit than an intentionally uncredentialed instance.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = GatewayConfig(**yaml.safe_load("providers:\n  openai:\n"))
    text = _validate_capturing_warnings(config, caplog)
    assert "providers.openai" in text
    assert "OPENAI_API_KEY" in text


def test_bare_keyed_entry_with_env_credential_warns_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # any-llm falls back to the provider's own env var, so a bare entry backed by
    # OPENAI_API_KEY is a working config, not a typo.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    config = GatewayConfig(**yaml.safe_load("providers:\n  openai:\n"))
    assert _validate_capturing_warnings(config, caplog) == ""


def test_bare_entry_of_unknown_instance_warns_nothing(caplog: pytest.LogCaptureFixture) -> None:
    # An instance that is not a known any-llm implementation is left alone here
    # (validate_provider_instances is lenient about it), so there is no basis for
    # claiming it needs a credential.
    config = GatewayConfig(**yaml.safe_load("providers:\n  mystery_box:\n"))
    assert _validate_capturing_warnings(config, caplog) == ""


def test_entry_with_settings_but_no_credential_warns_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Only a settings-less entry looks like a truncated edit; an entry with a body
    # is a deliberate config that predates this check.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = GatewayConfig(providers={"openai": {"models": ["gpt-4o"]}})
    assert _validate_capturing_warnings(config, caplog) == ""


# ---------------------------------------------------------------------------
# provider_credential_env_names
# ---------------------------------------------------------------------------


def test_credential_env_names_single() -> None:
    assert provider_credential_env_names("openai") == ("OPENAI_API_KEY",)


def test_credential_env_names_splits_alternatives() -> None:
    # gemini declares "GEMINI_API_KEY/GOOGLE_API_KEY"; either one authenticates.
    assert provider_credential_env_names("gemini") == ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def test_credential_env_names_empty_for_keyless_backends() -> None:
    # any-llm spells "no credential" as the literal string "None".
    for keyless in ("ollama", "llamacpp", "llamafile"):
        assert provider_credential_env_names(keyless) == ()


def test_credential_env_names_none_for_unknown_provider() -> None:
    # Not "keyless": unknowable, which callers treat as no basis for a warning.
    assert provider_credential_env_names("not-a-real-provider") is None


def test_validate_rejects_instance_name_with_separator() -> None:
    # A name containing ':' or '/' could never match the selector split and would
    # be silently unreachable; reject it at startup instead.
    for bad in ("my:lab", "my/lab"):
        with pytest.raises(ValueError, match="must not contain"):
            GatewayConfig(providers={bad: {"provider_type": "openai"}}).validate_provider_instances()


# ---------------------------------------------------------------------------
# split_selector
# ---------------------------------------------------------------------------


def test_split_selector_colon_first() -> None:
    assert split_selector("home_lab:deepseek-v4") == ("home_lab", "deepseek-v4")


def test_split_selector_slash() -> None:
    assert split_selector("openai/gpt-4o") == ("openai", "gpt-4o")


def test_split_selector_no_delimiter() -> None:
    assert split_selector("gpt-4o") is None


# ---------------------------------------------------------------------------
# resolve_provider_selector
# ---------------------------------------------------------------------------


def test_resolve_plain_provider_unchanged() -> None:
    config = GatewayConfig(providers={"openai": {"api_key": "sk-real"}})
    resolved = resolve_provider_selector(config, "openai:gpt-4o")
    assert resolved.instance == "openai"
    assert resolved.provider == LLMProvider.OPENAI
    assert resolved.model == "gpt-4o"
    assert resolved.dispatch_model == "openai:gpt-4o"
    assert resolved.kwargs["api_key"] == "sk-real"


def test_resolve_named_instance_routes_to_implementation() -> None:
    config = GatewayConfig(
        providers={
            "openai": {"api_key": "sk-real"},
            "home_lab": {
                "provider_type": "openai",
                "api_base": "https://box.ts.net/v1",
                "api_key": "home-token",
            },
        }
    )
    resolved = resolve_provider_selector(config, "home_lab:deepseek-v4-flash")
    # any-llm is dispatched against the implementation, never the instance name.
    assert resolved.provider == LLMProvider.OPENAI
    assert resolved.model == "deepseek-v4-flash"
    assert resolved.dispatch_model == "openai:deepseek-v4-flash"
    # ...but billing/pricing key on the instance, with the instance's credentials.
    assert resolved.instance == "home_lab"
    assert resolved.kwargs["api_base"] == "https://box.ts.net/v1"
    assert resolved.kwargs["api_key"] == "home-token"


def test_resolve_two_instances_do_not_collide() -> None:
    config = GatewayConfig(
        providers={
            "openai": {"api_key": "sk-real", "api_base": "https://api.openai.com/v1"},
            "home_lab": {"provider_type": "openai", "api_base": "https://box/v1", "api_key": "ht"},
        }
    )
    real = resolve_provider_selector(config, "openai:gpt-4o")
    local = resolve_provider_selector(config, "home_lab:gpt-4o")
    assert real.kwargs["api_key"] == "sk-real"
    assert local.kwargs["api_key"] == "ht"
    assert real.dispatch_model == local.dispatch_model == "openai:gpt-4o"


def test_resolve_alias_instance() -> None:
    config = GatewayConfig(providers={"vllm_box": {"provider_type": "openai-compatible", "api_base": "http://v/v1"}})
    resolved = resolve_provider_selector(config, "vllm_box:qwen3")
    assert resolved.provider == LLMProvider.OPENAI
    assert resolved.dispatch_model == "openai:qwen3"


def test_resolve_unknown_provider_raises() -> None:
    # An unconfigured prefix that is not a real provider surfaces any-llm's error
    # (caught as (ValueError, AnyLLMError) by the budget gate).
    with pytest.raises(AnyLLMError):
        resolve_provider_selector(GatewayConfig(), "not_a_provider:model")


# ---------------------------------------------------------------------------
# get_provider_kwargs strips instance-only metadata
# ---------------------------------------------------------------------------


def test_get_provider_kwargs_strips_provider_type_and_models() -> None:
    config = GatewayConfig(
        providers={
            "home_lab": {
                "provider_type": "openai",
                "models": ["a", "b"],
                "api_base": "http://x/v1",
                "api_key": "k",
            }
        }
    )
    kwargs = get_provider_kwargs(config, LLMProvider.OPENAI, instance="home_lab")
    assert "provider_type" not in kwargs
    assert "models" not in kwargs
    assert kwargs == {"api_base": "http://x/v1", "api_key": "k"}


# ---------------------------------------------------------------------------
# get_provider_kwargs: keyless custom-endpoint placeholder (mozilla-ai/otari#421)
# ---------------------------------------------------------------------------


def test_keyless_custom_endpoint_gets_placeholder_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # A custom OpenAI-compatible endpoint with no key would otherwise be rejected
    # by any-llm before it is even dialed; the placeholder makes the key optional.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = GatewayConfig(providers={"home_lab": {"provider_type": "openai", "api_base": "http://x/v1"}})
    kwargs = get_provider_kwargs(config, LLMProvider.OPENAI, instance="home_lab")
    assert kwargs == {"api_base": "http://x/v1", "api_key": _KEYLESS_PLACEHOLDER_API_KEY}


def test_explicit_key_is_not_overridden_by_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = GatewayConfig(
        providers={"home_lab": {"provider_type": "openai", "api_base": "http://x/v1", "api_key": "sk-real"}}
    )
    kwargs = get_provider_kwargs(config, LLMProvider.OPENAI, instance="home_lab")
    assert kwargs["api_key"] == "sk-real"


def test_no_placeholder_without_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    # A default hosted endpoint (no custom api_base) still relies on the provider's
    # native env var; injecting a placeholder there would shadow it.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = GatewayConfig(providers={"openai": {}})
    kwargs = get_provider_kwargs(config, LLMProvider.OPENAI, instance="openai")
    assert "api_key" not in kwargs


def test_env_var_fallback_preserved_over_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    # any-llm falls back to OPENAI_API_KEY before raising, so the placeholder must
    # not shadow a key the operator supplied that way.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    config = GatewayConfig(providers={"home_lab": {"provider_type": "openai", "api_base": "http://x/v1"}})
    kwargs = get_provider_kwargs(config, LLMProvider.OPENAI, instance="home_lab")
    assert "api_key" not in kwargs


def test_keyless_placeholder_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Keyless custom endpoint: placeholder.
    assert keyless_placeholder_api_key(LLMProvider.ANTHROPIC, "http://x/v1", None) == _KEYLESS_PLACEHOLDER_API_KEY
    # No api_base (hosted default): no placeholder.
    assert keyless_placeholder_api_key(LLMProvider.ANTHROPIC, None, None) is None
    # Key already present: no placeholder.
    assert keyless_placeholder_api_key(LLMProvider.ANTHROPIC, "http://x/v1", "sk-real") is None


def test_env_var_fallback_honors_compound_key_label(monkeypatch: pytest.MonkeyPatch) -> None:
    # Some providers expose ``ENV_API_KEY_NAME`` as a ``/``-joined list of
    # alternatives rather than one variable (gemini: "GEMINI_API_KEY/GOOGLE_API_KEY").
    # A key set under any alternative must still suppress the placeholder, or it
    # would shadow the operator's real env-var credential.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "goog-from-env")
    assert keyless_placeholder_api_key(LLMProvider.GEMINI, "http://x/v1", None) is None
    # With neither alternative set, a keyless custom endpoint still gets the placeholder.
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert keyless_placeholder_api_key(LLMProvider.GEMINI, "http://x/v1", None) == _KEYLESS_PLACEHOLDER_API_KEY


# ---------------------------------------------------------------------------
# normalize_pricing_key
# ---------------------------------------------------------------------------


def test_normalize_pricing_key_instance() -> None:
    config = GatewayConfig(providers={"home_lab": {"provider_type": "openai"}})
    assert normalize_pricing_key(config, "home_lab:deepseek") == "home_lab:deepseek"


def test_normalize_pricing_key_provider_slash_to_colon() -> None:
    assert normalize_pricing_key(GatewayConfig(), "openai/gpt-4o") == "openai:gpt-4o"


def test_normalize_pricing_key_unparseable_returned_unchanged() -> None:
    assert normalize_pricing_key(GatewayConfig(), "bare-model") == "bare-model"


def test_normalize_pricing_key_orphaned_instance_does_not_raise() -> None:
    # Regression: a pricing row keyed on an instance that is no longer configured
    # (e.g. a stored provider that could not be decrypted and was skipped) must be
    # returned unchanged, not raise AnyLLMError and 500 the models listing.
    assert normalize_pricing_key(GatewayConfig(), "home-lab:qwen3") == "home-lab:qwen3"


# ---------------------------------------------------------------------------
# capabilities key on the instance name
# ---------------------------------------------------------------------------


def test_capabilities_match_instance_scoped_key() -> None:
    config = GatewayConfig(
        providers={"home_lab": {"provider_type": "openai"}},
        model_capabilities={"home_lab:qwen2-vl": ModelCapabilityConfig(supports_image=True)},
    )
    caps = resolve_capabilities(config, LLMProvider.OPENAI, "qwen2-vl", instance="home_lab")
    assert caps.source == "config"
    assert caps.image is True


# ---------------------------------------------------------------------------
# pricing read endpoints tolerate instance-scoped keys (no 500)
# ---------------------------------------------------------------------------


def test_candidate_model_keys_handles_instance_key_without_raising() -> None:
    # Regression: an instance name is not an any-llm provider, so the underlying
    # split raises AnyLLMError (not ValueError). _candidate_model_keys must catch
    # it and still return the stored key, rather than 500 on pricing reads.
    assert _candidate_model_keys("home_lab:deepseek") == ["home_lab:deepseek"]


def test_candidate_model_keys_normalizes_instance_slash_to_colon() -> None:
    assert _candidate_model_keys("home_lab/deepseek") == ["home_lab/deepseek", "home_lab:deepseek"]


def test_candidate_model_keys_real_provider_unchanged() -> None:
    assert _candidate_model_keys("openai:gpt-4o") == ["openai:gpt-4o", "openai/gpt-4o"]
