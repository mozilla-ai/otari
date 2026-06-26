"""Unit tests for named provider-instance resolution (mozilla-ai/otari#213)."""

from __future__ import annotations

import pytest
from any_llm import LLMProvider
from any_llm.exceptions import AnyLLMError

from gateway.api.routes.pricing import _candidate_model_keys
from gateway.core.config import GatewayConfig, ModelCapabilityConfig
from gateway.services.model_capabilities import resolve_capabilities
from gateway.services.provider_kwargs import (
    get_provider_kwargs,
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
# normalize_pricing_key
# ---------------------------------------------------------------------------


def test_normalize_pricing_key_instance() -> None:
    config = GatewayConfig(providers={"home_lab": {"provider_type": "openai"}})
    assert normalize_pricing_key(config, "home_lab:deepseek") == "home_lab:deepseek"


def test_normalize_pricing_key_provider_slash_to_colon() -> None:
    assert normalize_pricing_key(GatewayConfig(), "openai/gpt-4o") == "openai:gpt-4o"


def test_normalize_pricing_key_unparseable_returned_unchanged() -> None:
    assert normalize_pricing_key(GatewayConfig(), "bare-model") == "bare-model"


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
