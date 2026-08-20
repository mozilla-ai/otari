"""The exact USD column types: what they accept, and where the rounding happens.

These are the bind-side guarantees ``gateway.models.money`` makes, tested
without a database because they are properties of the type rather than of any
engine: a float writer keeps working, its value converts through its shortest
decimal, and the amount that reaches the column is rounded half-up to the
column's scale rather than by whatever the engine underneath would have done.
"""

from decimal import Decimal
from typing import Any

import pytest

from gateway.models.money import COST_PRECISION, RATE_PRECISION, UsdCost, UsdRate, as_float, to_usd, to_usd_or_none


def _bind(column: UsdRate | UsdCost, value: Any) -> Decimal | None:
    return column.process_bind_param(value, dialect=None)  # type: ignore[arg-type]


def test_a_float_writer_keeps_working_and_loses_no_precision() -> None:
    """Config files and the pricing API hand over floats; they convert exactly."""
    assert _bind(UsdRate(), 0.075) == Decimal("0.075")
    assert _bind(UsdRate(), 3) == Decimal("3")
    assert _bind(UsdRate(), "1.25") == Decimal("1.25")
    assert _bind(UsdCost(), Decimal("0.000001")) == Decimal("0.000001")


def test_none_stays_none() -> None:
    """A model without a cache rate leaves the column NULL, which is not zero."""
    assert _bind(UsdRate(), None) is None
    assert _bind(UsdCost(), None) is None


def test_a_cost_is_rounded_half_up_to_the_micro_dollar_on_the_way_in() -> None:
    """No engine in the path, so this is the gateway's rule and nothing else's."""
    assert _bind(UsdCost(), Decimal("0.1234565")) == Decimal("0.123457")
    assert _bind(UsdCost(), Decimal("0.1234564")) == Decimal("0.123456")
    # A tie Python's default rounding would settle the other way: half-even
    # gives 0.000002 here, and half-away-from-zero gives -0.000003 below.
    assert _bind(UsdCost(), Decimal("0.0000025")) == Decimal("0.000003")
    assert _bind(UsdCost(), Decimal("-0.0000025")) == Decimal("-0.000003")


def test_a_rate_is_rounded_half_up_to_its_own_scale() -> None:
    assert _bind(UsdRate(), Decimal("0.123456785")) == Decimal("0.12345679")


def test_a_value_that_is_not_an_amount_is_refused_rather_than_stored() -> None:
    """A bug that would otherwise land in a money column as 0 or as True."""
    with pytest.raises(TypeError):
        _bind(UsdCost(), True)
    with pytest.raises(TypeError):
        _bind(UsdRate(), object())


def test_a_negative_amount_is_left_to_the_check_constraints() -> None:
    """The type does not silently correct a sign; the table refuses it."""
    assert _bind(UsdRate(), -1) == Decimal("-1")


def test_the_column_scales_match_what_the_ddl_declares() -> None:
    assert (UsdRate().impl.precision, UsdRate().impl.scale) == (RATE_PRECISION, 8)
    assert (UsdCost().impl.precision, UsdCost().impl.scale) == (COST_PRECISION, 6)


def test_the_edge_helpers_narrow_and_widen_only_where_they_should() -> None:
    assert to_usd(0.1) == Decimal("0.1")
    assert to_usd_or_none(None) is None
    assert as_float(Decimal("0.123457")) == 0.123457
    assert as_float(None) is None
