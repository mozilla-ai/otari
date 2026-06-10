"""Unit tests for the model-capability resolver."""

from __future__ import annotations

from any_llm import LLMProvider

from gateway.core.config import GatewayConfig, ModelCapabilityConfig
from gateway.services.model_capabilities import resolve_capabilities


def test_hosted_provider_trusts_metadata() -> None:
    caps = resolve_capabilities(GatewayConfig(), LLMProvider.OPENAI, "gpt-4o")
    assert caps.source == "provider_metadata"
    assert caps.image is True
    assert caps.pdf is True


def test_local_provider_defaults_to_extract() -> None:
    # Ollama over-reports image/pdf support at the class level; we must NOT trust
    # it, or document blocks get silently dropped.
    caps = resolve_capabilities(GatewayConfig(), LLMProvider.OLLAMA, "llama3")
    assert caps.source == "default"
    assert caps.image is False
    assert caps.pdf is False


def test_config_override_wins_with_provider_model_key() -> None:
    config = GatewayConfig(
        model_capabilities={"ollama/qwen2-vl": ModelCapabilityConfig(supports_image=True, supports_pdf=False)}
    )
    caps = resolve_capabilities(config, LLMProvider.OLLAMA, "qwen2-vl")
    assert caps.source == "config"
    assert caps.image is True
    assert caps.pdf is False


def test_config_override_accepts_colon_separator() -> None:
    # Users write model selectors and pricing keys with a colon, so the colon
    # form must match too — not just the slash form.
    config = GatewayConfig(
        model_capabilities={"ollama:qwen2-vl": ModelCapabilityConfig(supports_image=True)}
    )
    caps = resolve_capabilities(config, LLMProvider.OLLAMA, "qwen2-vl")
    assert caps.source == "config"
    assert caps.image is True


def test_config_override_matches_bare_model_key() -> None:
    config = GatewayConfig(model_capabilities={"my-vision": ModelCapabilityConfig(supports_image=True)})
    caps = resolve_capabilities(config, LLMProvider.VLLM, "my-vision")
    assert caps.source == "config"
    assert caps.image is True


def test_override_beats_hosted_metadata() -> None:
    # An operator can also downgrade a hosted model (e.g. a text-only deployment).
    config = GatewayConfig(model_capabilities={"openai/gpt-4o": ModelCapabilityConfig()})
    caps = resolve_capabilities(config, LLMProvider.OPENAI, "gpt-4o")
    assert caps.source == "config"
    assert caps.image is False
    assert caps.pdf is False
