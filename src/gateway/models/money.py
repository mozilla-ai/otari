"""Exact USD column types for rates and settled costs.

``model_pricing`` rates and ``usage_logs.cost`` were ``float`` columns, which
made every stored rate and every settled amount a binary approximation of the
decimal an operator typed. These types make them exact
(mozilla-ai/otari#661): ``NUMERIC`` in the DDL, ``Decimal`` in Python, with the
rounding the cost core defines applied on the way in.

Two things this deliberately does, both at the bind step so no writer can skip
them:

- **A ``float`` is accepted and coerced through its shortest decimal
  representation.** Config files, the pricing API, and the seed scripts all
  hand over floats, and ``Decimal(str(0.075))`` is ``0.075`` where
  ``Decimal(0.075)`` is not. Refusing floats would only push the same
  conversion out to a dozen call sites.
- **The value is quantized to the column's scale, half-up.** PostgreSQL rounds
  half away from zero on a numeric that exceeds the column scale, SQLite has no
  numeric type at all and stores a float, so leaving the rounding to the engine
  would make the stored amount depend on which one is underneath. Rounding here
  makes it the same on both.

SQLite still stores the value in a REAL column, because that is the only thing
its NUMERIC affinity does; exactness at rest needs PostgreSQL, which is what a
deployment that cares about accounting runs. What the quantization buys on
SQLite is that the value read back is the value written: SQLAlchemy formats the
float to the column's scale on the way out, which recovers a 6 or 8 decimal
place number exactly for any amount this schema will hold.
"""

from decimal import Decimal
from typing import Any

from sqlalchemy import Numeric
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.types import TypeDecorator

from gateway.core.metered_pricing import COST_SCALE, RATE_SCALE, quantize_cost, quantize_rate

# Room for a rate of ~$100M per million tokens and a cost of ~$1T. Both are far
# past anything real; the point of the width is that no arithmetic here can
# overflow the column and abort a settlement.
RATE_PRECISION = 18
COST_PRECISION = 18

# The largest USD amount a cost column holds, ~$1T less one micro-dollar.
MAX_USD_COST = Decimal(10) ** (COST_PRECISION - COST_SCALE) - Decimal(1).scaleb(-COST_SCALE)

# The largest budget limit a request body may set, and the bound the OpenAPI
# schema advertises. A *cap* is operator-typed where a settled cost is not, and
# an operator reaching for "no limit" reaches for a big round number: above the
# column's ceiling PostgreSQL refuses the write with a bare numeric overflow,
# which a route can only render as a 500 (mozilla-ai/otari#691). A whole dollar
# below :data:`MAX_USD_COST`, because the ceiling itself is not representable in
# the float the wire carries: ``float(MAX_USD_COST)`` rounds *up* to 1e12, which
# the column would then refuse. Spend counters need no such bound; they are sums
# of real settled costs.
MAX_USD_LIMIT = 999_999_999_999.0


def to_usd(value: Decimal | float | int | str) -> Decimal:
    """Coerce a value to ``Decimal`` without inheriting float error.

    Applied automatically to everything bound to one of these columns, and
    exported for the write paths that assign a ``float`` to one: spelling the
    conversion at the assignment keeps the declared column type honest, and
    performs the same shortest-decimal conversion the bind would.

    Unlike :func:`gateway.core.metered_pricing.to_decimal`, this raises rather
    than returning ``None`` for a value it cannot read, and it passes a negative
    through: a write of an unusable amount is a bug to surface, and the sign is
    the check constraints' business, not the type's.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise TypeError("A boolean is not a USD amount")
    if isinstance(value, int | float | str):
        return Decimal(str(value))
    raise TypeError(f"Cannot store {type(value).__name__} as a USD amount")


class UsdRate(TypeDecorator[Decimal]):
    """A price in USD per million tokens, stored exactly to 1e-8."""

    impl = Numeric(RATE_PRECISION, RATE_SCALE)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> Decimal | None:
        if value is None:
            return None
        return quantize_rate(to_usd(value))


class UsdCost(TypeDecorator[Decimal]):
    """A settled USD amount, stored exactly to the micro-dollar."""

    impl = Numeric(COST_PRECISION, COST_SCALE)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> Decimal | None:
        if value is None:
            return None
        return quantize_cost(to_usd(value))


def to_usd_or_none(value: Decimal | float | int | str | None) -> Decimal | None:
    """:func:`to_usd`, passing ``None`` through for a nullable rate."""
    return None if value is None else to_usd(value)


def as_float(value: Decimal | float | None) -> float | None:
    """Narrow a stored amount for a response body or a metric.

    JSON has no exact decimal and the dashboard's client is generated from
    ``float`` schemas, so a rate or a cost is narrowed on the way out. This is a
    display conversion: it never feeds arithmetic that settles anything.
    """
    return None if value is None else float(value)
