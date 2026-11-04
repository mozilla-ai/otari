"""What a price is per: the one list of units a rate can be in, and the quantity one price buys.

A stored rate is always USD per million of its unit (``input_price_per_million``),
whatever the unit. A published price is not: models.dev and a provider's price
list write a token price per million and an image price per one, so a reader
that moves a price into a rate column scales it by ``PRICE_PER``.
"""

from typing import Literal, cast, get_args

from gateway.types.models_dev import MODELS_DEV_UNITS

PriceUnit = Literal[
    "tokens",
    "requests",
    "images",
    "character",
    "megapixel",
    "second",
    "minute",
    "web_search",
]

PRICE_UNITS: tuple[PriceUnit, ...] = get_args(PriceUnit)

# How many of the unit one published price buys. ``web_search`` is models.dev's
# ``web_search.price``, USD per 1,000 searches, and ``requests`` is Otari's own.
PRICE_PER: dict[PriceUnit, int] = {
    "tokens": 1_000_000,
    "requests": 1,
    "images": 1,
    "character": 1_000_000,
    "megapixel": 1,
    "second": 1,
    "minute": 1,
    "web_search": 1_000,
}

# Where Otari's name for a models.dev unit is not the document's.
_OTARI_NAMES: dict[str, PriceUnit] = {"token": "tokens", "image": "images"}


def models_dev_unit(value: str | None) -> PriceUnit | None:
    """The unit a v2 price's ``unit`` names: ``tokens`` where the document gives none, ``None`` for one not known."""
    unit = value or "token"
    if unit not in MODELS_DEV_UNITS:
        return None
    return _OTARI_NAMES.get(unit) or cast(PriceUnit, unit)
