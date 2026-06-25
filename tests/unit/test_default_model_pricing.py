"""Tests for genai-prices-backed default pricing (default_model_pricing)."""

from datetime import UTC, datetime

import pytest

from gateway.services import pricing_service
from gateway.services.pricing_service import (
    configure_default_pricing,
    default_model_pricing,
    default_pricing_enabled,
)


def test_default_pricing_known_model_provider_scoped() -> None:
    """A well-known provider/model resolves to positive per-million rates."""
    as_of = datetime.now(UTC)
    pricing = default_model_pricing("openai", "gpt-4o", as_of)

    assert pricing is not None
    assert pricing.model_key == "openai:gpt-4o"
    assert pricing.effective_at == as_of
    assert pricing.input_price_per_million > 0
    assert pricing.output_price_per_million > 0


def test_default_pricing_without_provider() -> None:
    """A bare model name (no provider) still resolves when unambiguous."""
    pricing = default_model_pricing(None, "gpt-4o", datetime.now(UTC))

    assert pricing is not None
    assert pricing.model_key == "gpt-4o"
    assert pricing.input_price_per_million > 0


def test_default_pricing_input_only_model_prices_output_at_zero() -> None:
    """Input-only models (embeddings) price with a real input rate and 0 output."""
    pricing = default_model_pricing("openai", "text-embedding-3-small", datetime.now(UTC))

    assert pricing is not None
    assert pricing.input_price_per_million > 0
    assert pricing.output_price_per_million == 0.0


def test_default_pricing_huggingface_pinned_backend_is_priced() -> None:
    """A pinned HuggingFace backend maps to genai-prices' per-backend provider."""
    pricing = default_model_pricing("huggingface", "zai-org/GLM-4.6:together", datetime.now(UTC))

    assert pricing is not None
    # The key preserves the caller's full pinned selector.
    assert pricing.model_key == "huggingface:zai-org/GLM-4.6:together"
    assert pricing.input_price_per_million > 0
    assert pricing.output_price_per_million > 0


def test_default_pricing_huggingface_policy_suffix_not_priced() -> None:
    """Policy suffixes (auto routing) do not resolve to a single backend, so None."""
    assert default_model_pricing("huggingface", "zai-org/GLM-4.6:cheapest", datetime.now(UTC)) is None


def test_default_pricing_huggingface_bare_model_not_priced() -> None:
    """A bare HuggingFace model (no pinned backend) cannot be priced from the id."""
    assert default_model_pricing("huggingface", "meta-llama/Llama-3-70b", datetime.now(UTC)) is None


def test_default_pricing_unknown_model_returns_none() -> None:
    """An unknown model yields None so require_pricing can still fail closed."""
    pricing = default_model_pricing("openai", "totally-made-up-model-xyz", datetime.now(UTC))

    assert pricing is None


def test_default_pricing_fails_safe_on_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-LookupError from genai-prices degrades to None, not a request error."""

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("genai-prices exploded")

    monkeypatch.setattr(pricing_service, "calc_price", boom)

    assert default_model_pricing("openai", "gpt-4o", datetime.now(UTC)) is None


def test_configure_default_pricing_toggles_enabled_flag() -> None:
    """configure_default_pricing flips the process-wide enabled flag."""
    configure_default_pricing(False)
    assert default_pricing_enabled() is False

    configure_default_pricing(True)
    assert default_pricing_enabled() is True


def test_default_pricing_unknown_provider_falls_back_to_model_match() -> None:
    """An unrecognized provider id still resolves via a model-name-only match."""
    pricing = default_model_pricing("self-hosted-proxy", "gpt-4o", datetime.now(UTC))

    assert pricing is not None
    # The model_key preserves the caller's provider even though the rate was
    # resolved via the provider-agnostic fallback.
    assert pricing.model_key == "self-hosted-proxy:gpt-4o"
    assert pricing.input_price_per_million > 0


def test_default_pricing_is_transient_not_a_session_object() -> None:
    """The returned ModelPricing carries the requested timestamp and rates."""
    as_of = datetime(2025, 1, 1, tzinfo=UTC)
    pricing = default_model_pricing("anthropic", "claude-sonnet-4-20250514", as_of)

    assert pricing is not None
    assert pricing.effective_at == as_of
    assert pricing.input_price_per_million > 0
    assert pricing.output_price_per_million > 0
