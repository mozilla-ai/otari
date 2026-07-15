"""Unit tests for model aliases (display name -> target selector)."""

from __future__ import annotations

import pytest
from any_llm import LLMProvider

from gateway.api.routes.models import _alias_target_keys
from gateway.core.config import GatewayConfig
from gateway.model_labeling import relabel_model
from gateway.services.provider_kwargs import normalize_pricing_key, resolve_provider_selector

# ---------------------------------------------------------------------------
# config.resolve_alias
# ---------------------------------------------------------------------------


def test_resolve_alias_returns_target() -> None:
    config = GatewayConfig(aliases={"myopusmodel": "anthropic:claude-opus-4"})
    assert config.resolve_alias("myopusmodel") == "anthropic:claude-opus-4"


def test_resolve_alias_unknown_returns_none() -> None:
    assert GatewayConfig().resolve_alias("nope") is None


def test_resolve_alias_empty_target_returns_none() -> None:
    assert GatewayConfig(aliases={"x": ""}).resolve_alias("x") is None


# ---------------------------------------------------------------------------
# config.validate_aliases
# ---------------------------------------------------------------------------


def test_validate_accepts_provider_target() -> None:
    GatewayConfig(aliases={"fastmodel": "openai:gpt-5"}).validate_aliases()  # no raise


def test_validate_accepts_named_instance_target() -> None:
    config = GatewayConfig(
        providers={"home_lab": {"provider_type": "openai", "api_base": "http://x/v1"}},
        aliases={"housemodel": "home_lab:qwen3"},
    )
    config.validate_aliases()  # no raise


def test_validate_rejects_empty_target() -> None:
    with pytest.raises(ValueError, match="non-empty target"):
        GatewayConfig(aliases={"a": ""}).validate_aliases()


def test_validate_rejects_target_without_prefix() -> None:
    with pytest.raises(ValueError, match="instance:model"):
        GatewayConfig(aliases={"a": "gpt-5"}).validate_aliases()


def test_validate_rejects_unknown_provider_prefix() -> None:
    with pytest.raises(ValueError, match="neither a configured"):
        GatewayConfig(aliases={"a": "not_a_provider:model"}).validate_aliases()


@pytest.mark.parametrize("prefix", ["openai-compatible", "openai_compatible"])
def test_validate_rejects_provider_type_alias_prefix(prefix: str) -> None:
    # A provider_type alias names an instance's implementation, not a selector
    # prefix: any-llm does not know it, so request-time routing would raise. It
    # must fail at startup rather than on the first request.
    with pytest.raises(ValueError, match="neither a configured"):
        GatewayConfig(aliases={"fastmodel": f"{prefix}:gpt-4o"}).validate_aliases()


def test_validate_accepts_instance_named_like_provider_type_alias() -> None:
    # The prefix is a configured instance here, so it stays valid: only the
    # unconfigured provider_type-alias prefix is rejected.
    config = GatewayConfig(
        providers={"openai-compatible": {"provider_type": "openai", "api_base": "http://x/v1"}},
        aliases={"fastmodel": "openai-compatible:gpt-4o"},
    )
    config.validate_aliases()  # no raise


@pytest.mark.parametrize("name", ["openai:gpt-4o", "openai/gpt-4o", "my:model"])
def test_validate_rejects_selector_shaped_name(name: str) -> None:
    # Alias lookup runs before selector resolution, so a name containing a
    # delimiter would silently reroute requests for a real provider:model.
    config = GatewayConfig(aliases={name: "anthropic:claude-opus-4"})
    with pytest.raises(ValueError, match="must not contain"):
        config.validate_aliases()


def test_validate_rejects_collision_with_provider_instance() -> None:
    config = GatewayConfig(
        providers={"openai": {"api_key": "sk"}},
        aliases={"openai": "anthropic:claude-opus-4"},
    )
    with pytest.raises(ValueError, match="collides with a configured provider"):
        config.validate_aliases()


def test_validate_rejects_alias_chaining() -> None:
    config = GatewayConfig(
        aliases={"a": "anthropic:claude-opus-4", "b": "a:whatever"},
    )
    with pytest.raises(ValueError, match="alias chaining is not supported"):
        config.validate_aliases()


# ---------------------------------------------------------------------------
# resolve_provider_selector with aliases
# ---------------------------------------------------------------------------


def test_resolve_alias_to_provider_model() -> None:
    config = GatewayConfig(
        providers={"anthropic": {"api_key": "sk-ant"}},
        aliases={"myopusmodel": "anthropic:claude-opus-4"},
    )
    resolved = resolve_provider_selector(config, "myopusmodel")
    # Dispatch + billing target the real model...
    assert resolved.provider == LLMProvider.ANTHROPIC
    assert resolved.model == "claude-opus-4"
    assert resolved.instance == "anthropic"
    assert resolved.dispatch_model == "anthropic:claude-opus-4"
    assert resolved.kwargs["api_key"] == "sk-ant"
    # ...but the alias is carried for response relabeling.
    assert resolved.alias == "myopusmodel"


def test_resolve_alias_to_named_instance() -> None:
    config = GatewayConfig(
        providers={"home_lab": {"provider_type": "openai", "api_base": "http://x/v1", "api_key": "ht"}},
        aliases={"housemodel": "home_lab:qwen3"},
    )
    resolved = resolve_provider_selector(config, "housemodel")
    assert resolved.provider == LLMProvider.OPENAI
    assert resolved.instance == "home_lab"
    assert resolved.dispatch_model == "openai:qwen3"
    assert resolved.kwargs["api_base"] == "http://x/v1"
    assert resolved.alias == "housemodel"


def test_resolve_non_alias_has_no_alias() -> None:
    config = GatewayConfig(
        providers={"openai": {"api_key": "sk"}},
        aliases={"fastmodel": "openai:gpt-5"},
    )
    resolved = resolve_provider_selector(config, "openai:gpt-4o")
    assert resolved.alias is None


# ---------------------------------------------------------------------------
# relabel_model
# ---------------------------------------------------------------------------


class _HasModel:
    def __init__(self, model: str) -> None:
        self.model = model


class _MessageEvent:
    """Mimics an Anthropic ``message_start`` event (nested ``.message.model``)."""

    def __init__(self, model: str) -> None:
        self.message = _HasModel(model)


class _ResponseEvent:
    """Mimics a Responses stream event (nested ``.response.model``)."""

    def __init__(self, model: str) -> None:
        self.response = _HasModel(model)


def test_alias_target_keys_are_canonical() -> None:
    # Targets are collected in canonical "instance:model" form so a pricing row
    # written in the legacy "provider/model" shape still matches its alias and is
    # withheld from the listing.
    config = GatewayConfig(
        providers={"anthropic": {"api_key": "sk-ant"}},
        aliases={"myopusmodel": "anthropic:claude-opus-4"},
    )
    assert _alias_target_keys(config) == {"anthropic:claude-opus-4"}
    assert normalize_pricing_key(config, "anthropic/claude-opus-4") in _alias_target_keys(config)


def test_relabel_top_level_model() -> None:
    obj = _HasModel("anthropic:claude-opus-4")
    relabel_model(obj, "myopusmodel")
    assert obj.model == "myopusmodel"


def test_relabel_nested_message_model() -> None:
    chunk = _MessageEvent("claude-opus-4")
    relabel_model(chunk, "myopusmodel")
    assert chunk.message.model == "myopusmodel"


def test_relabel_nested_response_model() -> None:
    event = _ResponseEvent("gpt-5")
    relabel_model(event, "fastmodel")
    assert event.response.model == "fastmodel"


def test_relabel_no_model_field_is_noop() -> None:
    class _Bare:
        pass

    obj = _Bare()
    relabel_model(obj, "fastmodel")  # must not raise
    assert not hasattr(obj, "model")
