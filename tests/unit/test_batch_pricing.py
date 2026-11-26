from decimal import Decimal

from gateway.core.metered_pricing import allocate_rounded_total


def test_batch_cost_allocation_uses_result_order_for_equal_remainders() -> None:
    total, costs = allocate_rounded_total([Decimal("0.0000005"), Decimal("0.0000005")])

    assert total == Decimal("0.000001")
    assert costs == [Decimal("0.000001"), Decimal("0.000000")]


def test_batch_cost_allocation_preserves_unpriced_lines() -> None:
    total, costs = allocate_rounded_total([None, Decimal("0.0000014"), None])

    assert total == Decimal("0.000001")
    assert costs == [None, Decimal("0.000001"), None]


def test_batch_cost_allocation_without_priced_lines() -> None:
    assert allocate_rounded_total([None, None]) == (None, [None, None])
