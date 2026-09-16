"""Wire shapes for prices that more than one layer needs.

``PricingTier`` is written on a pricing request, read back on a price row, and
carried on every catalog entry, so it sits below the routes that serve it and
the services that build those entries.
"""

from pydantic import BaseModel, Field, model_validator


class PricingTier(BaseModel):
    """Whole-request price cliff selected by total billable input tokens."""

    min_input_tokens: int = Field(gt=0)
    input_price_per_million: float | None = Field(default=None, ge=0)
    output_price_per_million: float | None = Field(default=None, ge=0)
    cache_read_price_per_million: float | None = Field(default=None, ge=0)
    cache_write_price_per_million: float | None = Field(default=None, ge=0)
    cache_write_1h_price_per_million: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_has_rate_override(self) -> "PricingTier":
        rates = (
            self.input_price_per_million,
            self.output_price_per_million,
            self.cache_read_price_per_million,
            self.cache_write_price_per_million,
            self.cache_write_1h_price_per_million,
        )
        if all(rate is None for rate in rates):
            raise ValueError("pricing tier must override at least one price field")
        return self
