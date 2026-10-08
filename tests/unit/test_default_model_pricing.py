"""Tests for models.dev-backed default pricing (default_model_pricing)."""

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from gateway.core.config import GatewayConfig
from gateway.services.pricing import ModelsDevPriceIndex, PriceGeneration, reset_generations, set_accepted_generations
from gateway.services.pricing_service import (
    configure_default_pricing,
    configure_provider_types,
    default_model_pricing,
    default_pricing_enabled,
    default_pricing_reference,
    model_context_window,
)

Install = Callable[[dict[str, Any] | None], ModelsDevPriceIndex]
NOW = datetime(2026, 6, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _mini_catalog(install_models_dev: Install) -> None:
    install_models_dev(None)


def test_default_pricing_known_model_provider_scoped() -> None:
    """A provider and model the catalog lists resolve to its per-million rates."""
    pricing = default_model_pricing("openai", "gpt-4.1", NOW)

    assert pricing is not None
    assert pricing.model_key == "openai:gpt-4.1"
    assert pricing.effective_at == NOW
    assert pricing.input_price_per_million == Decimal("2")
    assert pricing.output_price_per_million == Decimal("8")
    assert pricing.cache_read_price_per_million == Decimal("0.5")
    assert pricing.cache_write_price_per_million is None
    assert pricing.pricing_tiers == []


def test_default_pricing_carries_cache_rates() -> None:
    pricing = default_model_pricing("anthropic", "claude-sonnet-4-5", NOW)

    assert pricing is not None
    assert pricing.cache_read_price_per_million == Decimal("0.3")
    assert pricing.cache_write_price_per_million == Decimal("3.75")


def test_default_pricing_without_provider() -> None:
    """A bare model name (no provider) still resolves when unambiguous."""
    pricing = default_model_pricing(None, "gpt-4.1", NOW)

    assert pricing is not None
    assert pricing.model_key == "gpt-4.1"
    assert pricing.input_price_per_million == Decimal("2")


def test_default_pricing_bare_id_priced_differently_by_providers_is_not_priced() -> None:
    """A name two providers price differently is ambiguous, so it stays unpriced."""
    assert default_model_pricing(None, "shared/diff-price", NOW) is None
    assert default_model_pricing("self-hosted-proxy", "shared/diff-price", NOW) is None
    assert default_model_pricing(None, "shared/same-price", NOW) is not None


def test_default_pricing_input_only_model_prices_output_at_zero(install_models_dev: Install) -> None:
    """Input-only models (embeddings) price with a real input rate and 0 output."""
    install_models_dev(
        {
            "openai": {
                "id": "openai",
                "models": {"text-embedding-3-small": {"id": "text-embedding-3-small", "cost": {"input": 0.02}}},
            }
        }
    )
    pricing = default_model_pricing("openai", "text-embedding-3-small", NOW)

    assert pricing is not None
    assert pricing.input_price_per_million == Decimal("0.02")
    assert pricing.output_price_per_million == 0


def test_default_pricing_model_without_a_rate_is_not_priced() -> None:
    """A catalog entry with no cost cannot price a request."""
    assert default_model_pricing("openai", "gpt-image-1", NOW) is None


def test_default_pricing_huggingface_pinned_backend_is_priced() -> None:
    """A pinned HuggingFace backend prices at that backend's own listing."""
    pricing = default_model_pricing("huggingface", "meta-llama/Llama-3.3-70B-Instruct:together", NOW)

    assert pricing is not None
    # The key preserves the caller's full pinned selector.
    assert pricing.model_key == "huggingface:meta-llama/Llama-3.3-70B-Instruct:together"
    assert pricing.input_price_per_million == Decimal("0.88")
    assert pricing.output_price_per_million == Decimal("0.88")


def test_default_pricing_huggingface_policy_suffix_not_priced() -> None:
    """Policy suffixes (auto routing) do not resolve to a single backend, so None."""
    assert default_model_pricing("huggingface", "meta-llama/Llama-3.3-70B-Instruct:cheapest", NOW) is None


def test_default_pricing_unknown_model_returns_none() -> None:
    """An unknown model yields None so require_pricing can still fail closed."""
    assert default_model_pricing("openai", "totally-made-up-model-xyz", NOW) is None


@pytest.mark.parametrize(
    ("model", "expected_input"),
    [
        ("us.anthropic.claude-sonnet-4-20250514-v1:0", Decimal("3.3")),
        ("eu.anthropic.claude-sonnet-4-20250514-v1:0", Decimal("3")),
        ("global.anthropic.claude-sonnet-4-20250514-v1:0", Decimal("3")),
        ("anthropic.claude-sonnet-4-20250514", Decimal("3")),
    ],
)
def test_default_pricing_bedrock_geo_ids(model: str, expected_input: Decimal) -> None:
    """Bedrock geo prefixes and version suffixes resolve to the listing that carries the rate."""
    pricing = default_model_pricing("bedrock", model, NOW)

    assert pricing is not None
    assert pricing.input_price_per_million == expected_input


def test_default_pricing_context_tiers_are_preserved() -> None:
    """Long-context cliffs are kept as whole-request thresholds, not flattened."""
    pricing = default_model_pricing("anthropic", "claude-haiku-5-5", NOW)

    assert pricing is not None
    assert pricing.input_price_per_million == Decimal("0.1")
    assert pricing.pricing_tiers == [
        {
            "min_input_tokens": 100_000,
            "input_price_per_million": 0.5,
            "output_price_per_million": 2.5,
            "cache_read_price_per_million": 0.05,
            "cache_write_price_per_million": 0.625,
        }
    ]


def test_default_pricing_over_200k_is_one_tier() -> None:
    pricing = default_model_pricing("gemini", "gemini-2.5-pro", NOW)

    assert pricing is not None
    assert pricing.input_price_per_million == Decimal("1.25")
    assert pricing.pricing_tiers == [
        {
            "min_input_tokens": 200_000,
            "input_price_per_million": 2.5,
            "output_price_per_million": 15.0,
            "cache_read_price_per_million": 0.25,
        }
    ]


def test_configure_default_pricing_toggles_enabled_flag() -> None:
    """configure_default_pricing flips the process-wide enabled flag."""
    configure_default_pricing(False)
    assert default_pricing_enabled() is False

    configure_default_pricing(True)
    assert default_pricing_enabled() is True


def test_default_pricing_unknown_provider_falls_back_to_model_match() -> None:
    """An unrecognized provider id still resolves via an unambiguous model-name match."""
    pricing = default_model_pricing("self-hosted-proxy", "gpt-4.1", NOW)

    assert pricing is not None
    # The model_key preserves the caller's provider even though the rate was
    # resolved via the provider-agnostic fallback.
    assert pricing.model_key == "self-hosted-proxy:gpt-4.1"
    assert pricing.input_price_per_million == Decimal("2")


def test_default_pricing_falls_back_to_the_backing_implementation() -> None:
    """A custom-named instance prices under the provider_type it dispatches to."""
    configure_provider_types(lambda instance: "bedrock" if instance == "aws-prod" else instance)
    model = "us.anthropic.claude-sonnet-4-20250514-v1:0"

    pricing = default_model_pricing("aws-prod", model, NOW)

    assert pricing is not None
    assert pricing.model_key == f"aws-prod:{model}"
    assert pricing.input_price_per_million == Decimal("3.3")
    assert default_pricing_reference("aws-prod", model, NOW) == f"models.dev:amazon-bedrock/{model}"


def test_backing_implementation_beats_the_provider_agnostic_fallback() -> None:
    """The serving provider's rate wins where the bare name is ambiguous across providers."""
    configure_provider_types(lambda instance: "nebius" if instance == "edge-prod" else instance)

    pricing = default_model_pricing("edge-prod", "shared/diff-price", NOW)

    assert pricing is not None
    assert pricing.input_price_per_million == Decimal("1.5")


def test_openai_compatible_instance_does_not_resolve_its_implementation() -> None:
    """A self-hosted endpoint's wire protocol is not the vendor serving it."""
    config = GatewayConfig(providers={"local-vllm": {"provider_type": "openai-compatible"}})
    configure_provider_types(config.provider_pricing_implementation)

    assert default_model_pricing("local-vllm", "shared/diff-price", NOW) is None


def test_default_pricing_prefers_the_instance_over_the_implementation() -> None:
    """A resolvable instance name wins, so the implementation is only a fallback."""
    configure_provider_types(lambda _instance: "bedrock")

    pricing = default_model_pricing("anthropic", "claude-sonnet-4-5", NOW)

    assert pricing is not None
    assert pricing.input_price_per_million == Decimal("3")


@pytest.mark.parametrize("model", ["anthropic.claude-sonnet-4-5", "us.anthropic.claude-sonnet-4-5"])
def test_default_pricing_vendor_prefixed_model_under_unknown_provider(model: str) -> None:
    """A vendor-prefixed model id resolves under the vendor even when the serving provider is unknown."""
    pricing = default_model_pricing("sagemaker", model, NOW)

    assert pricing is not None
    assert pricing.model_key == f"sagemaker:{model}"
    assert pricing.input_price_per_million == Decimal("3")


@pytest.mark.parametrize(("provider", "model"), [("openai", "gpt-4.1"), ("gemini", "gemini-2.5-pro")])
def test_dotted_version_numbers_are_not_read_as_vendor_prefixes(provider: str, model: str) -> None:
    """A dot inside a version number must not change how a model resolves."""
    pricing = default_model_pricing(None, model, NOW)
    scoped = default_model_pricing(provider, model, NOW)

    assert pricing is not None
    assert scoped is not None
    assert pricing.input_price_per_million == scoped.input_price_per_million


def test_default_pricing_reference_names_the_catalog_entry() -> None:
    assert default_pricing_reference("openai", "gpt-4.1", NOW) == "models.dev:openai/gpt-4.1"
    assert default_pricing_reference(None, "gpt-4.1", NOW) == "models.dev:openai/gpt-4.1"
    assert default_pricing_reference("openai", "nope", NOW) is None


def test_model_context_window_reads_the_catalog() -> None:
    assert model_context_window("openai", "gpt-4.1") == 1_047_576
    assert model_context_window(None, "gpt-4.1") == 1_047_576
    assert model_context_window("openai", "nope") is None


def test_as_of_prices_from_the_snapshot_in_force_then(install_models_dev: Install) -> None:
    """The rate follows the accepted snapshot in effect at the lookup time, the oldest before the first."""
    old = ModelsDevPriceIndex.from_catalog(
        {"openai": {"id": "openai", "models": {"gpt-4.1": {"id": "gpt-4.1", "cost": {"input": 5, "output": 9}}}}}
    )
    new = install_models_dev(None)
    set_accepted_generations(
        [
            PriceGeneration(effective_at=datetime(2026, 1, 1, tzinfo=UTC), index=old),
            PriceGeneration(effective_at=datetime(2026, 3, 1, tzinfo=UTC), index=new),
        ]
    )

    before = default_model_pricing("openai", "gpt-4.1", datetime(2025, 1, 1, tzinfo=UTC))
    middle = default_model_pricing("openai", "gpt-4.1", datetime(2026, 2, 1, tzinfo=UTC))
    after = default_model_pricing("openai", "gpt-4.1", datetime(2026, 4, 1, tzinfo=UTC))

    assert before is not None and middle is not None and after is not None
    assert (before.input_price_per_million, middle.input_price_per_million) == (Decimal("5"), Decimal("5"))
    assert after.input_price_per_million == Decimal("2")


def test_the_bundled_snapshot_serves_when_nothing_was_accepted() -> None:
    """With no accepted snapshot the packaged one prices well-known models."""
    reset_generations()
    pricing = default_model_pricing("openai", "gpt-4o", NOW)

    assert pricing is not None
    assert pricing.input_price_per_million > 0
    assert pricing.output_price_per_million > 0
