"""Wire shapes for prices that more than one layer needs.

``PricingTier`` is written on a pricing request, read back on a price row, and
carried on every catalog entry, so it sits below the routes that serve it and
the services that build those entries. ``PricedKind`` carries the rules a price
row's kind adds, for the deployment's and the organizations' request bodies.
"""

from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, Field, model_validator

from gateway.models.pricing import MODEL_KIND, PricingKind, is_tool_price_key

PRICING_KIND_DESCRIPTION = (
    "What the row prices: 'model' (the default), or 'search' or 'fetch' for a search or fetch instance, "
    "keyed by the instance's name and priced per call. Only the deployment operator sets a search or fetch rate."
)


class PricedKind(BaseModel):
    """A price request's ``kind``, mixed into a body that has a ``model_key`` and a ``unit``.

    A search or fetch price is keyed by the instance's name and is per call, so
    its unit defaults to ``requests`` and may be nothing else.
    """

    kind: PricingKind | None = Field(default=None, description=PRICING_KIND_DESCRIPTION)

    if TYPE_CHECKING:
        model_key: str
        unit: Literal["tokens", "requests", "images"]

    @property
    def row_kind(self) -> PricingKind:
        """The kind of row this writes: an omitted kind is a model's."""
        return self.kind or MODEL_KIND

    @model_validator(mode="after")
    def validate_tool_kind(self) -> Self:
        """Hold a search or fetch price to a bare instance name and a per-call unit, ``requests`` by default."""
        if self.row_kind == MODEL_KIND:
            return self
        if not is_tool_price_key(self.model_key):
            raise ValueError(f"a {self.kind} price is keyed by the instance's name, with no ':' or '/'")
        if "unit" not in self.model_fields_set:
            self.unit = "requests"
        elif self.unit != "requests":
            raise ValueError(f"a {self.kind} price is per call, so its unit is 'requests'")
        return self


class PricingTier(BaseModel):
    """Whole-request price cliff selected by total billable input tokens."""

    min_input_tokens: int = Field(gt=0)
    input_price_per_million: float | None = Field(default=None, ge=0)
    output_price_per_million: float | None = Field(default=None, ge=0)
    cache_read_price_per_million: float | None = Field(default=None, ge=0)
    cache_write_price_per_million: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_has_rate_override(self) -> "PricingTier":
        rates = (
            self.input_price_per_million,
            self.output_price_per_million,
            self.cache_read_price_per_million,
            self.cache_write_price_per_million,
        )
        if all(rate is None for rate in rates):
            raise ValueError("pricing tier must override at least one price field")
        return self
