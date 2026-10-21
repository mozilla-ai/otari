"""What one subagent recommendation costs at the operator's rates.

The scaffold holds an amount before the call and never charges past it, so
:meth:`RecommenderPrice.hold` prices the most one call can use, and the call
is then charged for the tokens it used.
"""

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal

from gateway.core.metered_pricing import COST_QUANTUM, meter_cost, quantize_cost

# Twice the 32,000 characters of task a recommendation sends (``MAX_TASK_CHARS``),
# which leaves room for the spawn's other fields, the candidates, the
# instructions, and text that splits into more than one token a character. The
# answer is one label.
HOLD_INPUT_TOKENS = 65_536
HOLD_OUTPUT_TOKENS = 1_024


@dataclass(frozen=True)
class RecommenderPrice:
    """The operator's rates, in USD per million of the decision model's tokens."""

    input_price_per_million: Decimal = Decimal(0)
    output_price_per_million: Decimal = Decimal(0)

    def _exact(self, input_tokens: int, output_tokens: int) -> Decimal:
        return meter_cost(input_tokens, self.input_price_per_million) + meter_cost(
            output_tokens, self.output_price_per_million
        )

    def charge(self, *, input_tokens: int, output_tokens: int) -> Decimal:
        """The price of one recommendation that used these tokens."""
        return quantize_cost(self._exact(input_tokens, output_tokens))

    def hold(self) -> Decimal:
        """The most one recommendation can be charged, rounded up so no charge passes it."""
        return self._exact(HOLD_INPUT_TOKENS, HOLD_OUTPUT_TOKENS).quantize(COST_QUANTUM, rounding=ROUND_CEILING)
