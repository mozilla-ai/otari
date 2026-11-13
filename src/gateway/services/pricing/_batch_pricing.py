"""Pricing helpers for batch results."""

from decimal import ROUND_FLOOR, Decimal

from gateway.core.metered_pricing import COST_QUANTUM, quantize_cost


class BatchPricingService:
    """Price batch result lines and reconcile their rounded total."""

    @staticmethod
    def allocate_costs(line_costs: list[Decimal | None]) -> tuple[Decimal | None, list[Decimal | None]]:
        """Round a batch total once and allocate its micro-dollars across priced lines."""
        allocations: list[Decimal | None] = [None] * len(line_costs)
        priced = [(index, cost) for index, cost in enumerate(line_costs) if cost is not None]
        if not priced:
            return None, allocations

        total = quantize_cost(sum((cost for _, cost in priced), Decimal(0)))
        rounded_down = {index: cost.quantize(COST_QUANTUM, rounding=ROUND_FLOOR) for index, cost in priced}
        remaining = int((total - sum(rounded_down.values(), Decimal(0))) / COST_QUANTUM)
        ranked = sorted(
            priced,
            key=lambda entry: (entry[1] - rounded_down[entry[0]], -entry[0]),
            reverse=True,
        )
        for index, _ in ranked[:remaining]:
            rounded_down[index] += COST_QUANTUM
        for index, amount in rounded_down.items():
            allocations[index] = amount
        return total, allocations
