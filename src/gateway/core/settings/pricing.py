"""Pricing settings."""

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from gateway.core.settings_view import OMITTED, SettingsGroup, Shown

PRICING_REFRESH_POLICIES: tuple[str, ...] = ("manual", "review", "auto")


class PricingTierConfig(BaseModel):
    """One whole-request context threshold price rule from configuration."""

    min_input_tokens: int = Field(gt=0)
    input_price_per_million: float | None = Field(default=None, ge=0)
    output_price_per_million: float | None = Field(default=None, ge=0)
    cache_read_price_per_million: float | None = Field(default=None, ge=0)
    cache_write_price_per_million: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_has_rate_override(self) -> "PricingTierConfig":
        rates = (
            self.input_price_per_million,
            self.output_price_per_million,
            self.cache_read_price_per_million,
            self.cache_write_price_per_million,
        )
        if all(rate is None for rate in rates):
            raise ValueError("pricing tier must override at least one price field")
        return self


# The fields a flat price per request leaves no room for: a per-request rate has no tokens to price.
_TOKEN_PRICE_FIELDS = (
    "output_price_per_million",
    "cache_read_price_per_million",
    "cache_write_price_per_million",
    "pricing_tiers",
)
PRICE_PER_REQUEST_DESCRIPTION = (
    "USD for each request, whatever it uses: a completion model billed per call, such as Exa's or Liner's "
    "answers. Shorthand for unit 'requests' with input_price_per_million at a million times this; not "
    "combined with token prices."
)


def _finite_decimal(value: Any) -> Decimal | None:
    """``value`` as an exact finite decimal, or None when field validation should refuse it instead."""
    if isinstance(value, bool) or not isinstance(value, int | float | str | Decimal):
        return None
    try:
        parsed = Decimal(str(value))
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def price_per_request_shorthand(data: Any) -> Any:
    """Expand ``price_per_request`` into the per-request rate it stands for, refusing a mix with token prices.

    ``{"price_per_request": 0.005}`` becomes ``unit: requests`` with
    ``input_price_per_million: 5000`` and ``output_price_per_million: 0``,
    which is how a per-request rate is stored and billed. Exact in decimal, so
    a price like 0.003 does not come out as 3000.0000000000005 per million.
    """
    if not isinstance(data, dict) or data.get("price_per_request") is None:
        return data
    price = _finite_decimal(data["price_per_request"])
    if price is None:
        return data  # Left for the field's own validation to refuse.
    if data.get("unit", "requests") != "requests":
        raise ValueError("price_per_request is a price per request: leave unit out or set it to 'requests'")
    per_million = price * 1_000_000
    mixed = [name for name in _TOKEN_PRICE_FIELDS if data.get(name) not in (None, 0, [])]
    # An input rate equal to the expansion is this same price validated again, as a dump of it reads.
    given = data.get("input_price_per_million")
    if given is not None and _finite_decimal(given) != per_million:
        mixed.insert(0, "input_price_per_million")
    if mixed:
        raise ValueError(f"price_per_request replaces token prices; remove {', '.join(mixed)}")
    return {
        **data,
        "unit": "requests",
        "input_price_per_million": float(per_million),
        "output_price_per_million": 0.0,
    }


class PricingConfig(BaseModel):
    """Model pricing configuration."""

    input_price_per_million: float = Field(ge=0)
    output_price_per_million: float = Field(ge=0)
    cache_read_price_per_million: float | None = Field(
        default=None,
        ge=0,
        description="Price per 1M cached-input tokens (OpenAI/Gemini discount rate or Anthropic cache-read rate).",
    )
    cache_write_price_per_million: float | None = Field(
        default=None,
        ge=0,
        description="Price per 1M cache-write (creation) tokens. Anthropic only.",
    )
    pricing_tiers: list[PricingTierConfig] = Field(
        default_factory=list,
        description="Whole-request context threshold pricing rules.",
    )
    effective_at: datetime | None = Field(
        default=None,
        description="ISO 8601 datetime from which this price applies. Defaults to now if omitted.",
    )
    unit: Literal["tokens", "requests", "images"] = Field(
        default="tokens",
        description=(
            "What the rates are per: 'tokens' for a model, 'requests' for a gateway-run tool, a "
            "moderation call, or a completion model billed per call (USD per million requests), "
            "'images' for image generation (USD per million images)."
        ),
    )
    price_per_request: float | None = Field(
        default=None, ge=0, allow_inf_nan=False, description=PRICE_PER_REQUEST_DESCRIPTION
    )

    @model_validator(mode="before")
    @classmethod
    def _expand_price_per_request(cls, data: Any) -> Any:
        return price_per_request_shorthand(data)

    @model_validator(mode="after")
    def validate_unique_tier_thresholds(self) -> "PricingConfig":
        thresholds = [tier.min_input_tokens for tier in self.pricing_tiers]
        if len(thresholds) != len(set(thresholds)):
            raise ValueError("pricing_tiers must not repeat min_input_tokens")
        return self


class PricingSettings(BaseModel):
    """The deployment's configured prices, and how a model without one is served."""

    pricing: Annotated[dict[str, PricingConfig], OMITTED] = Field(
        default_factory=dict,
        description=(
            "Pre-configured model USD pricing (model_key -> {input_price_per_million, output_price_per_million})"
        ),
    )
    require_pricing: Annotated[bool, Shown(SettingsGroup.METERING)] = Field(
        default=True,
        description=(
            "Reject requests for models, direct searches and gateway-run tools that have no configured pricing "
            "(fail-closed, default). When False, unpriced models are served and logged without cost (legacy "
            "behavior), and unpriced searches and tool calls are served at the cost their provider reports, if any. "
            "Audio and moderation endpoints are always exempt: they have no token-based pricing."
        ),
    )
    default_pricing: Annotated[bool, Shown(SettingsGroup.METERING)] = Field(
        default=False,
        description=(
            "When a model has no pricing in the database, fall back to community-maintained "
            "default pricing from the models.dev catalog. Off by default: a billing "
            "gateway should price from rates you control, and these community estimates can lag "
            "or differ from real provider rates. Database pricing always takes precedence. Enable "
            "to auto-price common models without configuring each one; while off, require_pricing "
            "stays fail-closed for any model you have not priced explicitly."
        ),
    )
    pricing_refresh: Annotated[Literal["manual", "review", "auto"], Shown(SettingsGroup.METERING)] = Field(
        default="manual",
        description=(
            "How the models.dev defaults are kept current. 'manual': only when an operator checks for "
            "updates on Model pricing. 'review': fetch upstream every pricing_refresh_interval_seconds and "
            "hold the update for an operator to accept or reject. 'auto': fetch and apply on that schedule."
        ),
    )
    pricing_refresh_interval_seconds: Annotated[int, Shown(SettingsGroup.METERING)] = Field(
        default=86400,
        ge=300,
        description="How often the scheduled models.dev price check runs when pricing_refresh is review or auto.",
    )
