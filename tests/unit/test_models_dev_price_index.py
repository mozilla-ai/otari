"""The models.dev price index: parsing, matching order, tiers and as_of generations."""

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from gateway.services.pricing import ModelsDevPriceIndex, PriceGeneration, resolve_as_of, select_generation

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "models_dev_mini.json"


@pytest.fixture(scope="module")
def raw() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(FIXTURE.read_text())
    return data


@pytest.fixture(scope="module")
def index(raw: dict[str, Any]) -> ModelsDevPriceIndex:
    return ModelsDevPriceIndex.from_catalog(raw)


def test_parses_rates_metadata_and_reference(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("anthropic", "claude-sonnet-4-5")
    assert price is not None
    assert (price.input, price.output, price.cache_read, price.cache_write) == (
        Decimal("3"),
        Decimal("15"),
        Decimal("0.3"),
        Decimal("3.75"),
    )
    assert price.context_window == 200000
    assert price.model_name == "Claude Sonnet 4.5"
    assert price.provider_name == "Anthropic"
    assert price.provider_doc == "https://docs.anthropic.com/en/docs/about-claude/models"
    assert price.reference == "models.dev:anthropic/claude-sonnet-4-5"
    assert price.pricing_tiers() == []


def test_rates_are_exact_decimals(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("amazon-bedrock", "us.anthropic.claude-sonnet-4-20250514-v1:0")
    assert price is not None
    assert price.input == Decimal("3.3")
    assert price.cache_write == Decimal("4.125")


def test_missing_output_stays_none(raw: dict[str, Any]) -> None:
    raw = json.loads(json.dumps(raw))
    del raw["openai"]["models"]["gpt-4.1"]["cost"]["output"]
    price = ModelsDevPriceIndex.from_catalog(raw).resolve("openai", "gpt-4.1")
    assert price is not None
    assert price.input == Decimal("2")
    assert price.output is None


def test_model_without_cost_resolves_unpriced(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("openai", "gpt-image-1")
    assert price is not None
    assert not price.priced
    assert price.input is None
    assert price.context_window is None


def test_deprecated_status_still_prices(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("openai", "gpt-4")
    assert price is not None
    assert price.input == Decimal("30")


def test_context_tiers_map_to_pricing_tiers_sorted(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("deepinfra", "Qwen/Qwen3.7-Max")
    assert price is not None
    assert price.pricing_tiers() == [
        {
            "min_input_tokens": 32000,
            "input_price_per_million": 5.0,
            "output_price_per_million": 15.0,
            "cache_read_price_per_million": 1.0,
        },
        {
            "min_input_tokens": 128000,
            "input_price_per_million": 6.25,
            "output_price_per_million": 18.5,
            "cache_read_price_per_million": 1.25,
        },
    ]


def test_non_context_tier_types_are_ignored_and_over_200k_ignored_beside_tiers(
    index: ModelsDevPriceIndex, raw: dict[str, Any]
) -> None:
    price = index.resolve("anthropic", "claude-haiku-5-5")
    assert price is not None
    assert [t["min_input_tokens"] for t in price.pricing_tiers()] == [100000]
    assert price.pricing_tiers()[0]["cache_write_price_per_million"] == 0.625

    raw = json.loads(json.dumps(raw))
    raw["google"]["models"]["gemini-2.5-pro"]["cost"]["tiers"] = []
    both = ModelsDevPriceIndex.from_catalog(raw).resolve("google", "gemini-2.5-pro")
    assert both is not None
    assert both.pricing_tiers() == []


def test_context_over_200k_alone_is_one_tier(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("google", "gemini-2.5-pro")
    assert price is not None
    assert price.pricing_tiers() == [
        {
            "min_input_tokens": 200000,
            "input_price_per_million": 2.5,
            "output_price_per_million": 15.0,
            "cache_read_price_per_million": 0.25,
        }
    ]


def test_provider_scoped_exact_wins_over_other_providers(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("togetherai", "meta-llama/Llama-3.3-70B-Instruct")
    assert price is not None
    assert price.provider_id == "togetherai"
    assert price.input == Decimal("0.88")


def test_provider_alias_maps_to_models_dev_id(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("together", "meta-llama/Llama-3.3-70B-Instruct")
    assert price is not None
    assert price.provider_id == "togetherai"


def test_implementation_behind_instance_is_scoped_before_fallback(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("aws-prod", "anthropic.claude-sonnet-4-20250514-v1:0", implementation="bedrock")
    assert price is not None
    assert price.provider_id == "amazon-bedrock"
    assert price.input == Decimal("3")


def test_implementation_beats_unambiguous_agnostic_match(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("my-bedrock", "anthropic.claude-sonnet-4-20250514-v1:0", implementation="bedrock")
    assert price is not None
    assert price.provider_id == "amazon-bedrock"


@pytest.mark.parametrize(
    ("selector", "provider_id", "input_rate"),
    [
        ("meta-llama/Llama-3.3-70B-Instruct:together", "togetherai", "0.88"),
        ("meta-llama/Llama-3.3-70B-Instruct:groq", None, None),
        ("meta-llama/Llama-3.3-70B-Instruct:cheapest", None, None),
    ],
)
def test_huggingface_pinned_backend(
    index: ModelsDevPriceIndex, selector: str, provider_id: str | None, input_rate: str | None
) -> None:
    price = index.resolve("huggingface", selector)
    if provider_id is None:
        assert price is None
    else:
        assert price is not None
        assert price.provider_id == provider_id
        assert price.input == Decimal(str(input_rate))


def test_huggingface_backend_via_implementation(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("hf-prod", "meta-llama/Llama-3.3-70B-Instruct:together", implementation="huggingface")
    assert price is not None
    assert price.provider_id == "togetherai"


def test_huggingface_unpinned_uses_the_router_listing(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("huggingface", "meta-llama/Llama-3.3-70B-Instruct")
    assert price is not None
    assert price.provider_id == "huggingface"


@pytest.mark.parametrize(
    ("model", "expected_id"),
    [
        ("us.anthropic.claude-sonnet-4-20250514-v1:0", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
        ("eu.anthropic.claude-sonnet-4-20250514-v1:0", "anthropic.claude-sonnet-4-20250514-v1:0"),
        ("global.anthropic.claude-sonnet-4-20250514-v1:0", "anthropic.claude-sonnet-4-20250514-v1:0"),
        ("anthropic.claude-sonnet-4-20250514", "anthropic.claude-sonnet-4-20250514-v1:0"),
        ("apac.anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-haiku-20240307-v1:0"),
        ("amazon.nova-pro-v1:0", "eu.amazon.nova-pro"),
        ("amazon.nova-lite", "amazon.nova-lite-v1:0"),
        ("us.amazon.nova-lite-v1:0", "amazon.nova-lite-v1:0"),
    ],
)
def test_bedrock_id_normalization(index: ModelsDevPriceIndex, model: str, expected_id: str) -> None:
    price = index.resolve("bedrock", model)
    assert price is not None
    assert (price.provider_id, price.model_id) == ("amazon-bedrock", expected_id)


def test_agnostic_match_answers_when_name_is_unambiguous(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("unknown-gateway", "gpt-4.1")
    assert price is not None
    assert price.provider_id == "openai"


def test_agnostic_match_with_no_provider(index: ModelsDevPriceIndex) -> None:
    price = index.resolve(None, "gpt-4.1")
    assert price is not None
    assert price.reference == "models.dev:openai/gpt-4.1"


def test_agnostic_match_is_case_insensitive(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("x", "GPT-4.1")
    assert price is not None
    assert price.model_id == "gpt-4.1"


def test_agnostic_match_same_price_across_providers_is_unambiguous(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("x", "shared/same-price")
    assert price is not None
    assert price.provider_id == "groq"


def test_agnostic_match_differing_prices_is_ambiguous(index: ModelsDevPriceIndex) -> None:
    assert index.resolve("x", "shared/diff-price") is None
    assert index.resolve("x", "meta-llama/Llama-3.3-70B-Instruct") is None


def test_agnostic_match_by_canonical_model_id(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("x", "anthropic/claude-sonnet-4-5")
    assert price is not None
    assert price.provider_id == "anthropic"


def test_agnostic_match_by_canonical_tail_with_differing_prices_is_ambiguous(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("x", "claude-sonnet-4")
    assert price is None


def test_vendor_prefixed_id_priced_under_vendor(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("some-aggregator", "anthropic.claude-sonnet-4-5")
    assert price is not None
    assert price.provider_id == "anthropic"


def test_vendor_prefixed_id_behind_region_prefix(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("some-aggregator", "us.openai.gpt-4.1")
    assert price is not None
    assert price.provider_id == "openai"


def test_unknown_model_is_none(index: ModelsDevPriceIndex) -> None:
    assert index.resolve("openai", "no-such-model") is None
    assert index.resolve(None, "no.such.model") is None


def test_malformed_catalog_entries_are_skipped() -> None:
    index = ModelsDevPriceIndex.from_catalog(
        {
            "a": "nope",
            "b": {"id": "b", "models": []},
            "c": {"id": "c", "models": {"m": {"id": "m", "cost": {"input": "x", "output": -1, "tiers": "bad"}}, "n": 3}},
        }
    )
    price = index.resolve("c", "m")
    assert price is not None
    assert not price.priced
    assert len(index) == 1


def test_index_is_immutable(index: ModelsDevPriceIndex) -> None:
    price = index.resolve("openai", "gpt-4.1")
    assert price is not None
    with pytest.raises(AttributeError):
        price.input = Decimal(0)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        index.extra = 1  # type: ignore[attr-defined]


def _generation(day: int, input_rate: int) -> PriceGeneration:
    catalog = {"p": {"id": "p", "models": {"m": {"id": "m", "cost": {"input": input_rate, "output": 1}}}}}
    return PriceGeneration(datetime(2026, 1, day, tzinfo=UTC), ModelsDevPriceIndex.from_catalog(catalog))


def test_as_of_picks_the_generation_in_effect() -> None:
    generations = [_generation(20, 3), _generation(5, 1), _generation(10, 2)]

    def rate(day: int) -> Decimal | None:
        found = resolve_as_of(generations, datetime(2026, 1, day, 12, tzinfo=UTC), "p", "m")
        return found.input if found else None

    assert rate(1) == Decimal(1)
    assert rate(5) == Decimal(1)
    assert rate(9) == Decimal(1)
    assert rate(10) == Decimal(2)
    assert rate(25) == Decimal(3)


def test_as_of_with_no_generations_is_none() -> None:
    assert select_generation([], datetime(2026, 1, 1, tzinfo=UTC)) is None
    assert resolve_as_of([], datetime(2026, 1, 1, tzinfo=UTC), "p", "m") is None
