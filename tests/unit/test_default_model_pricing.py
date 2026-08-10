"""Tests for genai-prices-backed default pricing (default_model_pricing)."""

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest
from genai_prices.types import Tier, TieredPrices

from gateway.services import pricing_service
from gateway.services.pricing_service import (
    configure_default_pricing,
    configure_provider_types,
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


def test_transient_failure_is_not_cached_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient genai-prices failure is retried on the next lookup.

    Guards against caching the ``None`` from the exception branch, which would pin
    the model to unpriced until the date rolls over. A genuine ``LookupError`` miss
    is still cacheable; only a raised exception must be retried.
    """
    as_of = datetime(2025, 1, 1, tzinfo=UTC)

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("genai-prices hiccup")

    monkeypatch.setattr(pricing_service, "calc_price", boom)

    # First lookup hits the transient failure and degrades to None...
    assert default_model_pricing("openai", "gpt-4o", as_of) is None

    # ...and because the failure was not cached, restoring the real client and
    # retrying the same key resolves a price (a cached None would stay None here).
    monkeypatch.undo()
    pricing = default_model_pricing("openai", "gpt-4o", as_of)
    assert pricing is not None
    assert pricing.input_price_per_million > 0


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


def test_default_pricing_falls_back_to_the_backing_implementation() -> None:
    """A custom-named instance prices under the provider_type it dispatches to.

    Pricing keys on the instance name, and genai-prices only recognizes an instance
    name by accident (``bedrock-eu`` contains "bedrock"; ``aws-prod`` does not), so
    without the implementation attempt every Bedrock-style model id under a
    differently named instance went unpriced.
    """
    as_of = datetime.now(UTC)
    configure_provider_types(lambda instance: "bedrock" if instance == "aws-prod" else instance)

    pricing = default_model_pricing("aws-prod", "anthropic.claude-sonnet-5", as_of)

    assert pricing is not None
    assert pricing.model_key == "aws-prod:anthropic.claude-sonnet-5"
    # The Bedrock rate, not Anthropic's own: the serving provider sets the price.
    direct = default_model_pricing("bedrock", "anthropic.claude-sonnet-5", as_of)
    assert direct is not None
    assert pricing.input_price_per_million == direct.input_price_per_million


def test_default_pricing_prefers_the_instance_over_the_implementation() -> None:
    """A resolvable instance name wins, so the implementation is only a fallback."""
    as_of = datetime.now(UTC)
    configure_provider_types(lambda _instance: "bedrock")

    pricing = default_model_pricing("anthropic", "claude-sonnet-5", as_of)
    anthropic_direct = default_model_pricing(None, "claude-sonnet-5", as_of)

    assert pricing is not None
    assert anthropic_direct is not None
    assert pricing.input_price_per_million == anthropic_direct.input_price_per_million


@pytest.mark.parametrize(
    "model",
    ["anthropic.claude-sonnet-5", "us.anthropic.claude-sonnet-5-v1:0"],
)
def test_default_pricing_vendor_prefixed_model_under_unknown_provider(model: str) -> None:
    """A vendor-prefixed model id resolves even when the serving provider is unknown.

    genai-prices files ``anthropic.claude-sonnet-5`` only under ``aws``, and its
    provider-agnostic fallback picks ``anthropic`` from the "claude" in the name and
    then finds no such id there. Splitting on the vendor prefix prices it instead of
    leaving it unpriced under a provider genai-prices does not know at all.
    """
    pricing = default_model_pricing("sagemaker", model, datetime.now(UTC))

    assert pricing is not None
    assert pricing.model_key == f"sagemaker:{model}"
    assert pricing.input_price_per_million > 0
    assert pricing.output_price_per_million > 0


@pytest.mark.parametrize("model", ["gpt-4.1", "claude-3.5-sonnet"])
def test_dotted_version_numbers_are_not_read_as_vendor_prefixes(model: str) -> None:
    """A dot inside a version number must not change how a model resolves."""
    pricing = default_model_pricing(None, model, datetime.now(UTC))
    scoped = default_model_pricing("openai" if model.startswith("gpt") else "anthropic", model, datetime.now(UTC))

    assert pricing is not None
    assert scoped is not None
    assert pricing.input_price_per_million == scoped.input_price_per_million


def test_vendor_prefix_attempts_walk_each_dot_boundary() -> None:
    """Each dot boundary is offered, so a region prefix does not stop the search."""
    assert pricing_service._vendor_prefixed_attempts("us.anthropic.claude-sonnet-5-v1:0") == [
        ("us", "anthropic.claude-sonnet-5-v1:0"),
        ("anthropic", "claude-sonnet-5-v1:0"),
    ]
    assert pricing_service._vendor_prefixed_attempts("gpt-5") == []


def test_default_pricing_is_transient_not_a_session_object() -> None:
    """The returned ModelPricing carries the requested timestamp and rates."""
    as_of = datetime(2025, 1, 1, tzinfo=UTC)
    pricing = default_model_pricing("anthropic", "claude-sonnet-4-20250514", as_of)

    assert pricing is not None
    assert pricing.effective_at == as_of
    assert pricing.input_price_per_million > 0
    assert pricing.output_price_per_million > 0


def test_genai_tiers_are_preserved_as_whole_request_thresholds() -> None:
    """Default pricing must retain, rather than flatten, long-context cliffs."""
    price = SimpleNamespace(
        input_mtok=TieredPrices(Decimal("2"), [Tier(start=200_000, price=Decimal("4"))]),
        output_mtok=TieredPrices(Decimal("8"), [Tier(start=200_000, price=Decimal("12"))]),
        cache_read_mtok=None,
        cache_write_mtok=None,
    )

    assert pricing_service._pricing_tiers(price) == [
        {
            "min_input_tokens": 200_000,
            "input_price_per_million": 4.0,
            "output_price_per_million": 12.0,
        }
    ]
