"""A rate's block with the overrides and multipliers its request meets applied, in models.dev's order."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from gateway.models.pricing import ModelPricing
from gateway.services.pricing import PriceFacts, apply_rules, resolve_rates
from gateway.services.pricing import _price_rules as price_rules

_MONDAY_NOON = datetime(2026, 11, 2, 12, 30, tzinfo=UTC)


def _text(price: float, *, overrides: list[dict[str, Any]] | None = None, **extra: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {"price": price, **extra}
    if overrides:
        entry["overrides"] = overrides
    return entry


def _resolved(document: dict[str, Any], **facts: Any) -> dict[str, Any]:
    return resolve_rates(document, PriceFacts(at=_MONDAY_NOON, **facts)).document


def _input_price(document: dict[str, Any], **facts: Any) -> Decimal:
    return Decimal(str(_resolved(document, **facts)["text"]["input"]["price"]))


def _on(when: dict[str, Any], price: float = 9) -> dict[str, Any]:
    return {"text": {"input": _text(1, overrides=[{"when": when, "price": price}])}}


@pytest.mark.parametrize(
    ("when", "facts", "applies"),
    [
        ({"service_tier": {"equals": "priority"}}, {"service_tier": "priority"}, True),
        ({"service_tier": {"equals": "priority"}}, {"service_tier": "flex"}, False),
        ({"resolution": {"equals": "4K"}}, {"resolution": "4k"}, True),
        ({"service_tier": {"in": ["flex", "priority"]}}, {"service_tier": "flex"}, True),
        ({"context": {"gt": 200_000}}, {"context": 200_000}, False),
        ({"context": {"gt": 200_000}}, {"context": 200_001}, True),
        ({"context": {"gte": 200_000}}, {"context": 200_000}, True),
        ({"context": {"lt": 1000}}, {"context": 999}, True),
        ({"context": {"lte": 1000}}, {"context": 1001}, False),
        ({"duration": {"gt": 5}}, {"duration": Decimal("5.5")}, True),
        ({"utc_hours": {"gte": "12:00"}}, {}, True),
        ({"utc_hours": {"lt": "12:00"}}, {}, False),
        ({"utc_days": {"in": ["sat", "sun"]}}, {}, False),
        ({"utc_days": {"equals": "mon"}}, {}, True),
        ({"date": {"gte": "2026-11-01"}}, {}, True),
        ({"date": {"gte": "2027-01-01"}}, {}, False),
        ({"input_video": {"equals": False}}, {}, True),
        ({"websocket": {"equals": True}}, {}, False),
        ({"input_video": {"equals": True}}, {"input_video": True}, True),
        ({"generate_audio": {"equals": False}}, {"generate_audio": False}, True),
        ({"generate_audio": {"equals": False}}, {"generate_audio": True}, False),
        ({"region": {"in": ["us", "eu"]}}, {"region": "US"}, True),
        ({"not": {"region": {"equals": "global"}}}, {"region": "global"}, False),
        ({"context": {"equals": 200_000}}, {"context": 200_000}, True),
        ({"duration": {"equals": 5}}, {"duration": Decimal("5.0")}, True),
        ({"duration": {"in": [4, 5.5]}}, {"duration": Decimal("5.50")}, True),
        ({"input_video": {"equals": 1}}, {"input_video": True}, False),
        ({"context": {"equals": True}}, {"context": 1}, False),
        ({"context": {"gt": "100"}}, {"context": 200}, False),
    ],
)
def test_a_condition_holds_by_its_operator(when: dict[str, Any], facts: dict[str, Any], applies: bool) -> None:
    assert _input_price(_on(when), **facts) == (9 if applies else 1)


@pytest.mark.parametrize(
    "when",
    [
        {"input_video": {"equals": False}},
        {"not": {"websocket": {"equals": True}}},
        {"generate_audio": {"equals": False}},
        {"not": {"generate_audio": {"equals": True}}},
        {"not": {"region": {"equals": "global"}}},
    ],
)
def test_a_rule_on_a_flag_otari_cannot_tell_is_skipped(when: dict[str, Any]) -> None:
    assert _input_price(_on(when), input_video=None, websocket=None) == 1


def test_a_known_key_with_no_value_does_not_hold() -> None:
    assert _input_price(_on({"service_tier": {"equals": "priority"}})) == 1
    assert _input_price(_on({"not": {"service_tier": {"equals": "priority"}}})) == 9


@pytest.mark.parametrize(
    "when",
    [
        {"region": {"equals": "us"}},
        {"not": {"region": {"equals": "global"}}},
        {"all": [{"service_tier": {"equals": "flex"}}, {"any": [{"holiday": {"equals": "US"}}]}]},
        {"service_tier": {"equals": "flex"}, "mode": {"equals": "pro"}},
    ],
)
def test_a_rule_naming_a_key_no_fact_answers_is_skipped(when: dict[str, Any]) -> None:
    assert _input_price(_on(when), service_tier="flex") == 1


@pytest.mark.parametrize(
    "when",
    [
        {"service_tier": {"equals": "flex", "in": ["flex"]}},
        {"service_tier": {}},
        {"service_tier": {"like": "flex"}},
        {"service_tier": "flex"},
        {"service_tier": {"in": "flex"}},
        {"all": {"service_tier": {"equals": "flex"}}},
        "flex",
        {"service_tier": {"equals": ["flex"]}},
        {"context": {"gt": True}},
        {"context": {"gt": None}},
        {"context": {"min_items": -1}},
        {"context": {"min_items": True}},
    ],
)
def test_a_rule_whose_condition_cannot_be_read_is_skipped(when: Any) -> None:
    # Negating a skipped rule is still skipped, which a rule read as false is not.
    assert _input_price(_on(when), service_tier="flex", context=5) == 1
    assert _input_price(_on({"not": when}), service_tier="flex", context=5) == 1


def test_all_any_and_not_combine() -> None:
    off_peak = {"all": [{"utc_hours": {"gte": "12:00"}}, {"utc_hours": {"lt": "18:00"}}]}
    assert _input_price(_on(off_peak)) == 9
    either = {"any": [{"utc_days": {"equals": "sat"}}, {"service_tier": {"equals": "flex"}}]}
    assert _input_price(_on(either), service_tier="flex") == 9
    assert _input_price(_on(either)) == 1
    assert _input_price(_on({"not": either})) == 9


def test_min_items_counts_a_list_value() -> None:
    # No fact holds a list today, so the operator reads a scalar as not holding.
    assert _input_price(_on({"context": {"min_items": 1}}), context=5) == 1


def test_matching_overrides_merge_field_by_field_in_order() -> None:
    document = {
        "text": {
            "input": {
                "price": 1,
                "cache": {"read": 0.1, "write": {"5m": 1.25, "1h": 2}, "storage": 4},
                "overrides": [
                    {"when": {"context": {"gt": 10}}, "price": 2, "cache": {"read": 0.2, "write": {"1h": 4}}},
                    {"when": {"context": {"gt": 20}}, "price": 3},
                    {"when": {"context": {"gt": 99}}, "price": 7},
                ],
            }
        }
    }

    resolved = _resolved(document, context=50)["text"]["input"]

    assert resolved == {"price": 3, "cache": {"read": 0.2, "write": {"5m": 1.25, "1h": 4}, "storage": 4}}


def test_an_override_writing_one_cache_write_replaces_the_ttls() -> None:
    document = {
        "text": {
            "input": {
                "price": 1,
                "cache": {"write": {"5m": 1.25, "1h": 2}},
                "overrides": [{"when": {"context": {"gt": 10}}, "cache": {"write": 3}}],
            }
        }
    }

    assert _resolved(document, context=50)["text"]["input"]["cache"] == {"write": 3}


def test_an_override_reaches_every_modality_and_web_search() -> None:
    document = {
        "text": {"output": _text(10, overrides=[{"when": {"thinking": {"equals": True}}, "price": 20}])},
        "audio": {"input": _text(3, overrides=[{"when": {"context": {"gt": 100}}, "price": 6}])},
        "image": {
            "output": _text(
                0.04,
                unit="image",
                overrides=[
                    {"when": {"resolution": {"equals": "2048x2048"}}, "price": 0.08},
                    {"when": {"quality": {"equals": "low"}}, "price": 0.02},
                    {"when": {"style": {"equals": "vector_illustration"}}, "price": 0.09},
                ],
            )
        },
        "video": {
            "output": _text(
                0.1,
                unit="second",
                overrides=[
                    {"when": {"resolution": {"equals": "1080p"}}, "price": 0.3},
                    {"when": {"aspect_ratio": {"equals": "9:16"}}, "price": 0.4},
                    {"when": {"duration": {"gt": 8}}, "price": 0.5},
                ],
            )
        },
        "web_search": _text(10, overrides=[{"when": {"service_tier": {"equals": "priority"}}, "price": 30}]),
    }

    resolved = _resolved(
        document, context=101, resolution="1080p", quality="low", aspect_ratio="9:16", service_tier="priority"
    )

    assert resolved["text"]["output"] == {"price": 10}
    assert resolved["audio"]["input"] == {"price": 6}
    assert resolved["image"]["output"] == {"price": 0.02, "unit": "image"}
    assert resolved["video"]["output"] == {"price": 0.4, "unit": "second"}
    assert resolved["web_search"] == {"price": 30}


def test_a_number_factor_scales_every_price_of_the_block() -> None:
    document = {
        "text": {
            "input": {"price": 1, "cache": {"read": 0.1, "write": {"5m": 1.25, "1h": 2}, "explicit": {"read": 0.3}}},
            "output": {"price": 10, "reasoning": 12},
        },
        "image": {"output": {"price": 0.04, "unit": "image"}},
        "web_search": {"price": 10},
        "multipliers": [{"when": {"service_tier": {"equals": "priority"}}, "factor": 2}],
    }

    resolved = _resolved(document, service_tier="priority")

    assert resolved == {
        "text": {
            "input": {
                "price": Decimal(2),
                "cache": {
                    "read": Decimal("0.2"),
                    "write": {"5m": Decimal("2.50"), "1h": Decimal(4)},
                    "explicit": {"read": Decimal("0.6")},
                },
            },
            "output": {"price": Decimal(20), "reasoning": Decimal(24)},
        },
        "image": {"output": {"price": Decimal("0.08"), "unit": "image"}},
        "web_search": {"price": Decimal(20)},
    }


def test_an_object_factor_scales_only_the_prices_it_names() -> None:
    document = {
        "text": {"input": {"price": 1, "cache": {"read": 0.1}}, "output": {"price": 10}},
        "audio": {"input": {"price": 3}},
        "web_search": {"price": 10},
        "multipliers": [
            {"when": {"service_tier": {"equals": "flex"}}, "factor": {"text": 0.5}},
            {"when": {"service_tier": {"equals": "flex"}}, "factor": {"audio": {"input": {"price": 3}}}},
            {"when": {"service_tier": {"equals": "flex"}}, "factor": {"video": 9}},
        ],
    }

    resolved = _resolved(document, service_tier="flex")

    assert resolved["text"] == {
        "input": {"price": Decimal("0.5"), "cache": {"read": Decimal("0.05")}},
        "output": {"price": Decimal("5.0")},
    }
    assert resolved["audio"] == {"input": {"price": Decimal(9)}}
    assert resolved["web_search"] == {"price": 10}


def test_overrides_apply_before_every_matching_multiplier() -> None:
    document = {
        "text": {"input": _text(1, overrides=[{"when": {"context": {"gt": 100}}, "price": 2}])},
        "multipliers": [
            {"when": {"service_tier": {"equals": "priority"}}, "factor": 2},
            {"when": {"utc_days": {"equals": "mon"}}, "factor": {"text": 1.5}},
            {"when": {"region": {"equals": "us"}}, "factor": 10},
        ],
    }

    assert _input_price(document, context=101, service_tier="priority") == Decimal(6)


def test_a_negative_or_unreadable_factor_is_skipped() -> None:
    document = {
        "text": {"input": {"price": 1}},
        "multipliers": [
            {"when": {"service_tier": {"equals": "flex"}}, "factor": -1},
            {"when": {"service_tier": {"equals": "flex"}}, "factor": {"text": "half"}},
            {"when": {"service_tier": {"equals": "flex"}}, "factor": True},
        ],
    }

    assert _input_price(document, service_tier="flex") == 1


def test_the_rules_that_changed_each_price_are_named_by_its_path() -> None:
    tier = {"service_tier": {"equals": "priority"}}
    long = {"context": {"gt": 100}}
    document = {
        "text": {"input": _text(1, overrides=[{"when": long, "price": 2}]), "output": _text(10)},
        "image": {"output": {"price": 0.04, "unit": "image"}},
        "web_search": {"price": 10},
        "multipliers": [{"when": tier, "factor": {"text": {"output": 2}}}, {"when": tier, "factor": 3}],
    }

    rules = resolve_rates(document, PriceFacts(context=101, service_tier="priority")).rules

    assert rules == {
        "text.input": (long, tier),
        "text.output": (tier, tier),
        "image.output": (tier,),
        "web_search": (tier,),
    }


def test_a_cache_price_names_only_the_rules_that_changed_it() -> None:
    long = {"context": {"gt": 100}}
    tier = {"service_tier": {"equals": "priority"}}
    document = {
        "text": {
            "input": _text(
                1,
                cache={"read": 0.1, "write": 1.25},
                overrides=[{"when": long, "price": 2}, {"when": long, "cache": {"write_ttls": [["1h", 2]]}}],
            )
        },
        "multipliers": [{"when": tier, "factor": {"text": {"input": {"cache": {"read": 2}}}}}],
    }

    rules = resolve_rates(document, PriceFacts(context=101, service_tier="priority")).rules

    assert rules == {"text.input": (long,), "text.input.cache.write": (long,), "text.input.cache.read": (tier,)}


def test_a_rule_read_from_a_stored_block_is_named_as_written() -> None:
    long = {"context": {"gt": Decimal("100")}, "duration": {"lte": Decimal("2.5")}}
    document = {"text": {"input": _text(1, overrides=[{"when": long, "price": 2}])}}

    rules = resolve_rates(document, PriceFacts(context=101, duration=2)).rules

    assert rules == {"text.input": ({"context": {"gt": 100}, "duration": {"lte": 2.5}},)}
    ((named,),) = rules.values()
    assert type(named["context"]["gt"]) is int
    assert type(named["duration"]["lte"]) is float


def test_resolving_leaves_the_block_it_read_unchanged() -> None:
    document = {
        "text": {"input": _text(1, overrides=[{"when": {"context": {"gt": 1}}, "price": 2}], cache={"read": 0.1})},
        "multipliers": [{"when": {"context": {"gt": 1}}, "factor": 2}],
    }
    before = repr(document)

    resolve_rates(document, PriceFacts(context=5))

    assert repr(document) == before


def test_the_clock_is_now_when_the_request_gives_no_start(monkeypatch: pytest.MonkeyPatch) -> None:
    instant = datetime(2026, 11, 8, 23, 59, 59, tzinfo=UTC)

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
            return instant

    monkeypatch.setattr(price_rules, "datetime", _Frozen)

    assert _input_price_now(_on({"date": {"equals": "2026-11-08"}, "utc_hours": {"equals": "23:59"}})) == 9


def _input_price_now(document: dict[str, Any]) -> Decimal:
    return Decimal(str(resolve_rates(document, PriceFacts()).document["text"]["input"]["price"]))


def _row(rates: dict[str, Any] | None, **columns: Any) -> ModelPricing:
    return ModelPricing(
        model_key="anthropic:claude-x",
        input_price_per_million=Decimal(columns.get("input", 1)),
        output_price_per_million=Decimal(columns.get("output", 10)),
        cache_read_price_per_million=columns.get("cache_read"),
        pricing_tiers=columns.get("tiers", []),
        unit=columns.get("unit", "token"),
        rates=rates,
    )


_LONG_CONTEXT = {
    "text": {
        "input": _text(1, cache={"read": 0.1}, overrides=[{"when": {"context": {"gt": 200_000}}, "price": 2}]),
        "output": _text(10, overrides=[{"when": {"context": {"gt": 200_000}}, "price": 15}]),
    }
}


def test_a_row_takes_its_columns_from_the_resolved_text_prices() -> None:
    row = _row(
        _LONG_CONTEXT, cache_read=Decimal("0.1"), tiers=[{"min_input_tokens": 200_000, "input_price_per_million": 2}]
    )

    at_bound = apply_rules(row, PriceFacts(context=200_000)).rate
    past_bound = apply_rules(row, PriceFacts(context=200_001)).rate

    assert (at_bound.input_price_per_million, at_bound.output_price_per_million) == (Decimal(1), Decimal(10))
    assert (past_bound.input_price_per_million, past_bound.output_price_per_million) == (Decimal(2), Decimal(15))
    assert past_bound.cache_read_price_per_million == Decimal("0.1")
    assert past_bound.pricing_tiers == []
    assert past_bound.model_key == row.model_key
    assert "overrides" not in past_bound.rates["text"]["input"]


def test_an_operator_tier_still_starts_at_its_threshold() -> None:
    operator = {
        "text": {"input": _text(1, overrides=[{"when": {"context": {"gte": 1000}}, "price": 2}]), "output": _text(10)}
    }

    assert apply_rules(_row(operator), PriceFacts(context=1000)).rate.input_price_per_million == Decimal(2)


def test_a_multiplier_reaches_a_row_priced_in_another_unit() -> None:
    per_image = {
        "image": {"output": {"price": 0.04, "unit": "image"}},
        "multipliers": [{"when": {"service_tier": {"equals": "flex"}}, "factor": 0.5}],
    }
    row = _row(per_image, input=40_000, output=0, unit="image")

    applied = apply_rules(row, PriceFacts(service_tier="flex"))

    assert applied.rate.input_price_per_million == Decimal(20_000)
    assert applied.rate.unit == "image"
    assert applied.rules == {"image.output": ({"service_tier": {"equals": "flex"}},)}


def test_a_row_with_no_block_is_priced_by_its_columns() -> None:
    row = _row(None, tiers=[{"min_input_tokens": 10, "input_price_per_million": 2}])

    applied = apply_rules(row, PriceFacts(context=50))

    assert applied.rate is row
    assert applied.rules == {}


def test_a_block_in_a_unit_no_column_holds_keeps_the_row_columns() -> None:
    row = _row({"video": {"output": {"price": 0.1, "unit": "frame"}}}, input=7, output=8, unit="token")

    applied = apply_rules(row, PriceFacts())

    assert (applied.rate.input_price_per_million, applied.rate.output_price_per_million) == (Decimal(7), Decimal(8))


def test_a_block_whose_columns_name_another_unit_keeps_the_row_columns() -> None:
    row = _row({"audio": {"input": {"price": 0.006, "unit": "minute"}}}, input=7, output=8, unit="token")

    applied = apply_rules(row, PriceFacts())

    assert (applied.rate.input_price_per_million, applied.rate.output_price_per_million) == (Decimal(7), Decimal(8))
    assert applied.rate.rates == {"audio": {"input": {"price": 0.006, "unit": "minute"}}}
