"""The models.dev price index: parsing, matching order, tiers and as_of generations."""

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from gateway.services.pricing import (
    ModelsDevPriceIndex,
    PriceGeneration,
    PriceTimeline,
    parse_rate,
    resolve_as_of,
    select_generation,
)

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


def test_non_context_tier_types_are_ignored_and_over_200k_ignored_beside_context_tiers(
    index: ModelsDevPriceIndex, raw: dict[str, Any]
) -> None:
    price = index.resolve("anthropic", "claude-haiku-5-5")
    assert price is not None
    assert [t["min_input_tokens"] for t in price.pricing_tiers()] == [100000]
    assert price.pricing_tiers()[0]["cache_write_price_per_million"] == 0.625

    raw = json.loads(json.dumps(raw))
    raw["google"]["models"]["gemini-2.5-pro"]["cost"]["tiers"] = [
        {"tier": {"type": "context", "size": 100000}, "input": 9}
    ]
    both = ModelsDevPriceIndex.from_catalog(raw).resolve("google", "gemini-2.5-pro")
    assert both is not None
    assert [t["min_input_tokens"] for t in both.pricing_tiers()] == [100000]


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
            "c": {
                "id": "c",
                "models": {"m": {"id": "m", "cost": {"input": "x", "output": -1, "tiers": "bad"}}, "n": 3},
            },
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


def _generation(day: int, input_rate: int, *, baseline: bool = False) -> PriceGeneration:
    catalog = {"p": {"id": "p", "models": {"m": {"id": "m", "cost": {"input": input_rate, "output": 1}}}}}
    return PriceGeneration(
        datetime(2026, 1, day, tzinfo=UTC), ModelsDevPriceIndex.from_catalog(catalog), baseline=baseline
    )


def _rate_at(generations: list[PriceGeneration] | PriceTimeline, day: int) -> Decimal | None:
    found = resolve_as_of(generations, datetime(2026, 1, day, 12, tzinfo=UTC), "p", "m")
    return found.input if found else None


def test_as_of_picks_the_generation_in_effect() -> None:
    generations = [_generation(20, 3), _generation(5, 1), _generation(10, 2)]

    assert _rate_at(generations, 5) == Decimal(1)
    assert _rate_at(generations, 9) == Decimal(1)
    assert _rate_at(generations, 10) == Decimal(2)
    assert _rate_at(generations, 25) == Decimal(3)


def test_as_of_before_the_first_generation_is_unpriced() -> None:
    generations = [_generation(5, 1), _generation(10, 2)]
    assert _rate_at(generations, 1) is None
    assert select_generation(generations, datetime(2026, 1, 1, tzinfo=UTC)) is None


def test_baseline_generation_answers_every_earlier_date() -> None:
    generations = [_generation(5, 1, baseline=True), _generation(10, 2)]
    assert _rate_at(generations, 1) == Decimal(1)
    assert _rate_at(generations, 12) == Decimal(2)


def test_naive_datetimes_are_utc() -> None:
    generations = [_generation(5, 1), _generation(10, 2)]
    found = resolve_as_of(generations, datetime(2026, 1, 7), "p", "m")
    assert found is not None
    assert found.input == Decimal(1)


def test_timeline_must_be_sorted() -> None:
    with pytest.raises(ValueError, match="sorted"):
        PriceTimeline((_generation(10, 2), _generation(5, 1)))
    assert _rate_at(PriceTimeline.build([_generation(10, 2), _generation(5, 1)]), 11) == Decimal(2)


def test_as_of_with_no_generations_is_none() -> None:
    assert select_generation([], datetime(2026, 1, 1, tzinfo=UTC)) is None
    assert resolve_as_of([], datetime(2026, 1, 1, tzinfo=UTC), "p", "m") is None


def _bedrock(*models: tuple[str, int]) -> ModelsDevPriceIndex:
    listing = {m: {"id": m, "cost": {"input": rate, "output": 1}} for m, rate in models}
    return ModelsDevPriceIndex.from_catalog({"amazon-bedrock": {"id": "amazon-bedrock", "models": listing}})


def test_bedrock_v2_is_never_priced_from_v1() -> None:
    index = _bedrock(("anthropic.claude-x-v1:0", 3))
    assert index.resolve("bedrock", "anthropic.claude-x-v2:0") is None
    assert index.resolve("bedrock", "anthropic.claude-x-v2") is None
    assert index.resolve("bedrock", "anthropic.claude-x") is not None


def test_bedrock_geo_is_never_swapped_for_another() -> None:
    index = _bedrock(("us.anthropic.claude-x-v1:0", 3))
    assert index.resolve("bedrock", "eu.anthropic.claude-x-v1:0") is None
    found = index.resolve("bedrock", "anthropic.claude-x-v1:0")
    assert found is not None
    assert found.model_id == "us.anthropic.claude-x-v1:0"


def test_bedrock_geo_less_request_with_disagreeing_geo_listings_is_unpriced() -> None:
    disagree = _bedrock(("us.anthropic.claude-x-v1:0", 3), ("eu.anthropic.claude-x-v1:0", 4))
    assert disagree.resolve("bedrock", "anthropic.claude-x-v1:0") is None
    agree = _bedrock(("us.anthropic.claude-x-v1:0", 3), ("eu.anthropic.claude-x-v1:0", 3))
    assert agree.resolve("bedrock", "anthropic.claude-x-v1:0") is not None


def test_bedrock_global_request_does_not_match_a_regional_listing() -> None:
    index = _bedrock(("eu.anthropic.claude-x", 3))
    assert index.resolve("bedrock", "global.anthropic.claude-x-v1:0") is None


def test_vendor_walk_uses_only_a_head_that_is_a_provider() -> None:
    index = ModelsDevPriceIndex.from_catalog(
        {
            "openai": {"id": "openai", "models": {"gpt-4.1": {"id": "gpt-4.1", "cost": {"input": 2, "output": 8}}}},
            "other": {"id": "other", "models": {"x.gpt-4.1": {"id": "x.gpt-4.1", "cost": {"input": 9, "output": 9}}}},
        }
    )
    assert index.resolve("agg", "openai.gpt-4.1") is not None
    assert index.resolve("agg", "nobody.gpt-4.1") is None


def test_agnostic_match_blocked_by_an_unpriced_listing() -> None:
    catalog = {
        "a": {"id": "a", "models": {"m": {"id": "m", "cost": {"input": 1, "output": 1}}}},
        "b": {"id": "b", "models": {"m": {"id": "m"}}},
    }
    assert ModelsDevPriceIndex.from_catalog(catalog).resolve("x", "m") is None


def test_agnostic_match_without_a_slash_in_the_canonical_id_is_one_candidate() -> None:
    catalog = {"a": {"id": "a", "models": {"m": {"id": "m", "canonical_model_id": "m", "cost": {"input": 1}}}}}
    found = ModelsDevPriceIndex.from_catalog(catalog).resolve("x", "m")
    assert found is not None
    assert found.provider_id == "a"


def _tiers_for(cost: dict[str, Any]) -> list[dict[str, float | int]]:
    catalog = {"p": {"id": "p", "models": {"m": {"id": "m", "cost": {"input": 1, "output": 1, **cost}}}}}
    found = ModelsDevPriceIndex.from_catalog(catalog).resolve("p", "m")
    assert found is not None
    return found.pricing_tiers()


def test_over_200k_is_used_when_no_context_tier_was_extracted() -> None:
    over = {"input": 2, "output": 2}
    assert _tiers_for({"tiers": [{"tier": {"type": "other", "size": 5}, "input": 3}], "context_over_200k": over}) == [
        {"min_input_tokens": 200000, "input_price_per_million": 2.0, "output_price_per_million": 2.0}
    ]
    assert _tiers_for({"tiers": [], "context_over_200k": over})[0]["min_input_tokens"] == 200000


@pytest.mark.parametrize("size", [0, -5, 1.5, True])
def test_unusable_tier_sizes_are_ignored(size: object) -> None:
    assert _tiers_for({"tiers": [{"tier": {"type": "context", "size": size}, "input": 3}]}) == []


def test_integral_float_tier_size_is_accepted() -> None:
    assert (
        _tiers_for({"tiers": [{"tier": {"type": "context", "size": 32000.0}, "input": 3}]})[0]["min_input_tokens"]
        == 32000
    )


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), 1_000_000.5, 1e12, True, "3"])
def test_implausible_rates_are_rejected(value: object) -> None:
    assert parse_rate(value) is None


def test_rates_inside_the_bounds_parse() -> None:
    assert parse_rate(0) == Decimal(0)
    assert parse_rate(0.3) == Decimal("0.3")
    assert parse_rate(1_000_000) == Decimal(1_000_000)
