"""What a price is per: the one list of units a rate can be in, and the quantity one price buys.

A stored rate is always USD per million of its unit (``input_price_per_million``),
whatever the unit. A published price is not: models.dev and a provider's price
list write a token price per million and an image price per one, so a reader
that moves a price into a rate column scales it by ``PRICE_PER``.

The names are models.dev's (``CostUnitV2Values`` in ``packages/core/src/schema-v2.ts``),
singular, so a v2 price's ``unit`` is a ``PriceUnit`` as the document writes it.
``request`` is Otari's own and follows the same rule. ``web_search`` is not a unit
there: it is the ``web_search.price`` field, USD per 1,000 searches.
"""

from typing import Annotated, Literal, cast, get_args

from pydantic import BeforeValidator

from gateway.types.models_dev import MODELS_DEV_UNITS

PriceUnit = Literal[
    "token",
    "request",
    "image",
    "character",
    "megapixel",
    "second",
    "minute",
    "web_search",
]

PRICE_UNITS: tuple[PriceUnit, ...] = get_args(PriceUnit)

# How many of the unit one published price buys.
PRICE_PER: dict[PriceUnit, int] = {
    "token": 1_000_000,
    "request": 1,
    "image": 1,
    "character": 1_000_000,
    "megapixel": 1,
    "second": 1,
    "minute": 1,
    "web_search": 1_000,
}

# The plural names Otari used before it took models.dev's. Read for one release,
# from a caller and from a row an older writer stored, and written nowhere.
RETIRED_UNIT_NAMES: dict[str, PriceUnit] = {"tokens": "token", "requests": "request", "images": "image"}


def current_unit_name(value: object) -> object:
    """``value`` with a retired unit name replaced by its current one, and anything else as given."""
    if isinstance(value, str):
        return RETIRED_UNIT_NAMES.get(value, value)
    return value


# A unit as a request body or the config file may still spell it. The schema it
# publishes is ``PriceUnit``'s, so a generated client learns the current names only.
AcceptedPriceUnit = Annotated[PriceUnit, BeforeValidator(current_unit_name)]


def models_dev_unit(value: str | None) -> PriceUnit | None:
    """The unit a v2 price names, or ``None`` for one no meter knows, so a new unit is never billed as tokens."""
    unit = value or "token"
    if unit not in MODELS_DEV_UNITS:
        return None
    return cast(PriceUnit, unit)
