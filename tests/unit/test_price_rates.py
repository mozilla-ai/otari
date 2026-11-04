"""A rate as a ``cost`` block, and the six columns as its text view."""

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from gateway.core.metered_pricing import RATE_FIELDS, TIER_THRESHOLD_FIELD
from gateway.services.pricing.models_dev_index import ModelsDevPriceIndex
from gateway.types.models_dev import parse_catalog_v2, parse_pricing, pricing_document
from gateway.types.price_rates import (
    TIER_RATES,
    TIER_THRESHOLD,
    RateColumns,
    RatesDocument,
    columns_from_rates,
    dump_rates,
    load_rates,
    rates_from_columns,
)
from gateway.types.price_unit import PRICE_UNITS, PriceUnit

SAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "models_dev_v2_sample.json"

# The units whose slot holds one price, so their output column has nowhere to go.
ONE_PRICE = ("image", "megapixel", "web_search")


def _blocks() -> list[tuple[str, dict[str, Any]]]:
    document = json.loads(SAMPLE.read_text("utf-8"))
    return [
        (f"{provider}:{model}", entry["cost"])
        for provider, models in document.items()
        for model, entry in models["models"].items()
        if entry.get("cost") is not None
    ]


def _columns(unit: PriceUnit, **given: Any) -> RateColumns:
    return RateColumns(
        unit=unit,
        input_price_per_million=Decimal(40000),
        output_price_per_million=Decimal(0) if unit in ONE_PRICE else Decimal(10),
        **given,
    )


def test_the_tier_names_are_the_column_shape() -> None:
    assert TIER_THRESHOLD == TIER_THRESHOLD_FIELD
    assert tuple(TIER_RATES.values()) == RATE_FIELDS


@pytest.mark.parametrize("unit", PRICE_UNITS)
def test_every_unit_round_trips_through_a_block(unit: PriceUnit) -> None:
    tier = {TIER_THRESHOLD: 200_000, "input_price_per_million": 5.0}
    if unit not in ONE_PRICE:
        tier["output_price_per_million"] = 15.0
    columns = _columns(
        unit,
        cache_read_price_per_million=Decimal("1.25"),
        cache_write_price_per_million=Decimal("3.75"),
        pricing_tiers=(tier,),
    )

    read = columns_from_rates(rates_from_columns(columns))

    assert read == columns


@pytest.mark.parametrize("unit", PRICE_UNITS)
def test_a_tier_keeps_the_base_rates_it_does_not_replace(unit: PriceUnit) -> None:
    columns = _columns(unit, pricing_tiers=({TIER_THRESHOLD: 200_000, "cache_read_price_per_million": 2.5},))

    read = columns_from_rates(rates_from_columns(columns))

    base = {} if unit in ONE_PRICE else {"output_price_per_million": 10.0}
    assert read.pricing_tiers == (
        {TIER_THRESHOLD: 200_000, "input_price_per_million": 40000.0, **base, "cache_read_price_per_million": 2.5},
    )


def test_a_token_rate_is_written_as_the_catalog_writes_it() -> None:
    columns = RateColumns(
        unit="token",
        input_price_per_million=Decimal("2.5"),
        output_price_per_million=Decimal(10),
        cache_read_price_per_million=Decimal("1.25"),
        pricing_tiers=({TIER_THRESHOLD: 200_000, "input_price_per_million": 5.0, "output_price_per_million": 15.0},),
    )

    assert rates_from_columns(columns) == {
        "text": {
            "input": {
                "price": Decimal("2.5"),
                "unit": "token",
                "cache": {"read": Decimal("1.25")},
                "overrides": [{"when": {"context": {"gte": 200_000}}, "price": Decimal("5.0")}],
            },
            "output": {
                "price": Decimal(10),
                "unit": "token",
                "overrides": [{"when": {"context": {"gte": 200_000}}, "price": Decimal("15.0")}],
            },
        }
    }


def test_a_per_unit_price_is_written_as_published() -> None:
    image = rates_from_columns(_columns("image"))
    search = rates_from_columns(_columns("web_search"))

    assert image == {"image": {"output": {"price": Decimal("0.04"), "unit": "image"}}}
    assert search == {"web_search": {"price": Decimal(40), "unit": "web_search"}}


def test_the_columns_of_a_block_scale_each_price_back() -> None:
    columns = columns_from_rates({"image": {"output": {"price": 0.04, "unit": "image"}}})
    search = columns_from_rates({"web_search": {"price": 2, "unit": "web_search"}})

    assert (columns.unit, columns.input_price_per_million, columns.output_price_per_million) == (
        "image",
        Decimal(40000),
        Decimal(0),
    )
    assert (search.unit, search.input_price_per_million) == ("web_search", Decimal(2000))


def test_the_text_prices_come_first_and_a_missing_unit_is_token() -> None:
    columns = columns_from_rates(
        {
            "image": {"output": {"price": 0.04, "unit": "image"}},
            "text": {"input": {"price": 3}, "output": {"price": 15}},
        }
    )

    assert (columns.unit, columns.input_price_per_million, columns.output_price_per_million) == (
        "token",
        Decimal(3),
        Decimal(15),
    )


@pytest.mark.parametrize(
    ("block", "reason"),
    [
        ({}, "price nothing"),
        ({"text": {"input": {"price": "3"}}}, "cannot be read"),
        ({"text": {"input": {"price": 3, "unit": "furlong"}}}, "not known here"),
        (
            {"text": {"input": {"price": 3, "unit": "token"}, "output": {"price": 1, "unit": "character"}}},
            "different units",
        ),
        ({"text": {"input": {"price": 1e11}}}, "does not fit"),
        ({"image": {"input": {"price": 0.003, "unit": "image"}}}, "price nothing"),
    ],
)
def test_a_block_no_column_can_hold_is_refused(block: dict[str, Any], reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        columns_from_rates(block)


def test_a_key_the_reader_does_not_know_is_ignored_and_a_bad_value_is_an_issue() -> None:
    pricing, issues = parse_pricing(
        {"text": {"input": {"price": 3, "added_later": {"x": 1}}, "output": {"price": "15"}}}, path="rates"
    )

    text = pricing.modalities["text"]
    assert text.input is not None and text.input.price == 3
    assert text.output is not None and text.output.price is None
    assert (issues.count, issues.examples) == (1, ("rates.text.output.price: expected a number",))


@pytest.mark.parametrize(("name", "block"), _blocks())
def test_every_sample_block_round_trips_through_the_reader(name: str, block: dict[str, Any]) -> None:
    pricing, issues = parse_pricing(block)

    assert issues.count == 0, name
    assert pricing_document(pricing) == block, name


@pytest.mark.parametrize(("name", "block"), _blocks())
def test_the_wire_model_takes_every_sample_block(name: str, block: dict[str, Any]) -> None:
    assert RatesDocument.model_validate(block).document() == block, name


@pytest.mark.parametrize(
    "block",
    [
        {"text": {"input": {"prize": 3}}},
        {"text": {"input": {"price": -3}}},
        {"text": {"input": {"price": 3, "unit": "tokens"}}},
        {"text": {"imput": {"price": 3}}},
        {"web_search": {"price": 10, "cache": {"write": {"5m": -1}}}},
        {"multipliers": [{"factor": 1.5}]},
    ],
)
def test_the_wire_model_refuses_a_typo_and_a_negative_price(block: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RatesDocument.model_validate(block)


def test_the_derived_columns_agree_with_the_price_index() -> None:
    """A seeded row's columns, read out of its block, are the ones the index gives the model."""
    catalog = parse_catalog_v2(json.loads(SAMPLE.read_text("utf-8")))
    index = ModelsDevPriceIndex.from_catalog(catalog)
    checked = 0
    for provider in catalog.providers.values():
        for model in provider.models.values():
            entry = index.get(provider.id, model.id)
            if model.pricing is None or entry is None or not entry.priced:
                continue
            columns = columns_from_rates(pricing_document(model.pricing))
            assert columns == RateColumns(
                unit="token",
                input_price_per_million=entry.input,  # type: ignore[arg-type]
                output_price_per_million=entry.output,  # type: ignore[arg-type]
                cache_read_price_per_million=entry.cache_read,
                cache_write_price_per_million=entry.cache_write,
                pricing_tiers=tuple(entry.pricing_tiers()),
            ), model.id
            checked += 1
    assert checked >= 4


def test_a_rate_is_written_exactly_and_read_back_as_decimal() -> None:
    document = {
        "text": {
            "input": {"price": Decimal("123456789.12345678"), "cache": {"write": {"5m": 3.75, "1h": 6}}},
            "output": {"price": 0.1},
        },
        "multipliers": [{"when": {"service_tier": {"equals": "flex"}}, "factor": {"text": 0.5}}],
    }

    text = dump_rates(document)

    assert text == (
        '{"text":{"input":{"price":123456789.12345678,"cache":{"write":{"5m":3.75,"1h":6}}},"output":{"price":0.1}},'
        '"multipliers":[{"when":{"service_tier":{"equals":"flex"}},"factor":{"text":0.5}}]}'
    )
    read = load_rates(text)
    assert read["text"]["input"]["price"] == Decimal("123456789.12345678")
    assert isinstance(read["text"]["output"]["price"], Decimal)
    assert list(read["text"]["input"]["cache"]["write"]) == ["5m", "1h"]
    assert dump_rates(read) == text


@pytest.mark.parametrize("value", [Decimal("1E+2"), Decimal("2.50"), 1e-07, 0, True])
def test_a_number_is_written_in_plain_digits(value: object) -> None:
    assert (
        dump_rates({"n": value})
        == {
            Decimal("1E+2"): '{"n":100}',
            Decimal("2.50"): '{"n":2.5}',
            1e-07: '{"n":0.0000001}',
            0: '{"n":0}',
            True: '{"n":true}',
        }[value]
    )


def test_a_text_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(ValueError, match="not a JSON object"):
        load_rates("[1]")
