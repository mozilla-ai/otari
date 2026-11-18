"""The one list of price units, pinned to models.dev's and to the API."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from gateway.types.models_dev import MODELS_DEV_UNITS
from gateway.types.price_unit import (
    PRICE_PER,
    PRICE_UNITS,
    RETIRED_UNIT_NAMES,
    AcceptedPriceUnit,
    current_unit_name,
    models_dev_unit,
)

SPEC = Path(__file__).resolve().parents[2] / "docs" / "public" / "openapi.json"

# ``CostUnitV2Values`` in models.dev's ``packages/core/src/schema-v2.ts``, as of 2026-11-03.
# ``web_search`` is not among them: it is the ``web_search.price`` field, USD per 1,000 searches.
COST_UNIT_V2_VALUES = ("token", "character", "image", "megapixel", "second", "minute")


def test_every_unit_has_a_quantity_one_price_buys() -> None:
    assert tuple(PRICE_PER) == PRICE_UNITS
    assert all(per > 0 for per in PRICE_PER.values())


@pytest.mark.parametrize(
    ("unit", "per"),
    [
        ("token", 1_000_000),
        ("character", 1_000_000),
        ("request", 1),
        ("image", 1),
        ("megapixel", 1),
        ("second", 1),
        ("minute", 1),
        ("web_search", 1_000),
    ],
)
def test_a_published_price_buys_this_many(unit: str, per: int) -> None:
    assert PRICE_PER[unit] == per  # type: ignore[index]


def test_the_reader_knows_every_models_dev_unit() -> None:
    assert MODELS_DEV_UNITS == COST_UNIT_V2_VALUES


def test_every_models_dev_unit_is_a_price_unit_spelled_the_same() -> None:
    """One vocabulary: the document's name is the stored name, with no map between them."""
    assert set(MODELS_DEV_UNITS) <= set(PRICE_UNITS)
    assert [models_dev_unit(value) for value in MODELS_DEV_UNITS] == list(MODELS_DEV_UNITS)


@pytest.mark.parametrize(
    ("value", "unit"),
    [
        (None, "token"),
        ("", "token"),
        ("token", "token"),
        ("second", "second"),
        # Otari's own unit, the field that is not a unit, and the retired plurals are not the document's.
        ("request", None),
        ("web_search", None),
        ("tokens", None),
        ("images", None),
        ("a-unit-models-dev-adds-later", None),
    ],
)
def test_a_models_dev_unit_names_otari_unit(value: str | None, unit: str | None) -> None:
    assert models_dev_unit(value) == unit


def test_a_retired_name_has_a_current_one_and_nothing_else_is_touched() -> None:
    assert RETIRED_UNIT_NAMES == {"tokens": "token", "requests": "request", "images": "image"}
    assert set(RETIRED_UNIT_NAMES.values()) <= set(PRICE_UNITS)
    assert set(RETIRED_UNIT_NAMES).isdisjoint(PRICE_UNITS)
    assert [current_unit_name(name) for name in ("tokens", "token", "second", "seconds")] == [
        "token",
        "token",
        "second",
        "seconds",
    ]
    assert current_unit_name(None) is None
    assert current_unit_name(3) == 3


class _Body(BaseModel):
    unit: AcceptedPriceUnit = "token"


@pytest.mark.parametrize(
    ("sent", "read"),
    [("tokens", "token"), ("requests", "request"), ("images", "image"), ("token", "token"), ("minute", "minute")],
)
def test_a_body_still_takes_a_retired_name_and_reads_it_as_the_current_one(sent: str, read: str) -> None:
    assert _Body.model_validate({"unit": sent}).unit == read


@pytest.mark.parametrize("sent", ["seconds", "Token", "", None, 1])
def test_a_body_refuses_what_is_neither_a_current_nor_a_retired_name(sent: object) -> None:
    with pytest.raises(ValidationError):
        _Body.model_validate({"unit": sent})


def test_a_body_publishes_the_current_names_only() -> None:
    assert tuple(_Body.model_json_schema()["properties"]["unit"]["enum"]) == PRICE_UNITS


def _unit_enums(node: Any, path: str = "") -> dict[str, list[str]]:
    """Every ``unit`` property of the spec that carries an enum, by its path."""
    found: dict[str, list[str]] = {}
    if isinstance(node, dict):
        unit = node.get("properties", {}).get("unit") if isinstance(node.get("properties"), dict) else None
        if isinstance(unit, dict):
            enums = [item["enum"] for item in (unit, *unit.get("anyOf", ())) if isinstance(item.get("enum"), list)]
            if enums:
                found[path] = enums[0]
        for key, value in node.items():
            found.update(_unit_enums(value, f"{path}/{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.update(_unit_enums(value, f"{path}/{index}"))
    return found


def test_every_unit_enum_in_the_api_is_the_one_list() -> None:
    enums = _unit_enums(json.loads(SPEC.read_text("utf-8")))

    assert enums, "the spec names no unit enum"
    assert {path: tuple(values) for path, values in enums.items() if tuple(values) != PRICE_UNITS} == {}
