from decimal import Decimal

from gateway.services.pricing import BatchPricingService


def test_batch_cost_allocation_uses_result_order_for_equal_remainders() -> None:
    total, costs = BatchPricingService.allocate_costs([Decimal("0.0000005"), Decimal("0.0000005")])

    assert total == Decimal("0.000001")
    assert costs == [Decimal("0.000001"), Decimal("0.000000")]


def test_batch_cost_allocation_preserves_unpriced_lines() -> None:
    total, costs = BatchPricingService.allocate_costs([None, Decimal("0.0000014"), None])

    assert total == Decimal("0.000001")
    assert costs == [None, Decimal("0.000001"), None]


def test_batch_cost_allocation_without_priced_lines() -> None:
    assert BatchPricingService.allocate_costs([None, None]) == (None, [None, None])
