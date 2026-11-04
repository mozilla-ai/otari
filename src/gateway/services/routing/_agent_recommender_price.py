"""What a recommendation costs the caller: the decision model's tokens at the operator's rates.

The decisions scaffold holds an amount before the call and never charges past
it, so :meth:`RecommenderPrice.hold` prices the most one call can use, and the
call is then charged for the tokens it used. An operator who wants a margin
puts it in the rates.
"""

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal

from gateway.core.metered_pricing import COST_QUANTUM

# Twice the 32,000 characters of task the question carries (``MAX_TASK_CHARS``).
# That leaves room for the spawn's other fields, the candidates and the
# instructions, and for text that splits into more than one token a character.
# The answer is one label.
HOLD_INPUT_TOKENS = 65_536
HOLD_OUTPUT_TOKENS = 1_024

_PER_MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class RecommenderPrice:
    """The operator's rates, in USD per million tokens."""

    input_price_per_million: Decimal = Decimal(0)
    output_price_per_million: Decimal = Decimal(0)

    def _exact(self, input_tokens: int, output_tokens: int) -> Decimal:
        tokens = input_tokens * self.input_price_per_million + output_tokens * self.output_price_per_million
        return tokens / _PER_MILLION

    def charge(self, *, input_tokens: int, output_tokens: int) -> Decimal:
        """The price of one recommendation that used these tokens."""
        return self._exact(input_tokens, output_tokens).quantize(COST_QUANTUM, rounding=ROUND_HALF_UP)

    def hold(self) -> Decimal:
        """The most one recommendation can be charged, rounded up so no charge passes it."""
        return self._exact(HOLD_INPUT_TOKENS, HOLD_OUTPUT_TOKENS).quantize(COST_QUANTUM, rounding=ROUND_CEILING)
