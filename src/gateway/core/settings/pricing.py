"""Pricing settings."""

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from gateway.core.settings_view import OMITTED, SettingsGroup, Shown
from gateway.types.price_rates import MODALITIES, RATES_DESCRIPTION, RatesDocument, columns_from_rates
from gateway.types.price_unit import AcceptedPriceUnit, current_unit_name

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
    "answers. Shorthand for unit 'request' with input_price_per_million at a million times this; not "
    "combined with token prices."
)

PRICE_UNIT_DESCRIPTION = (
    "What the rates are per: 'token' for a model, 'request' for a gateway-run tool, a "
    "moderation call, or a completion model billed per call, 'image' for image generation, "
    "'character' for speech."
    " Every rate is USD per million of its unit. A rate in a unit that has no meter yet is stored "
    "and prices nothing. The plural names 'tokens', 'requests' and 'images' are still read for one release."
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

    ``{"price_per_request": 0.005}`` becomes ``unit: request`` with
    ``input_price_per_million: 5000`` and ``output_price_per_million: 0``,
    which is how a per-request rate is stored and billed. Exact in decimal, so
    a price like 0.003 does not come out as 3000.0000000000005 per million.
    """
    if not isinstance(data, dict) or data.get("price_per_request") is None:
        return data
    price = _finite_decimal(data["price_per_request"])
    if price is None:
        return data  # Left for the field's own validation to refuse.
    if current_unit_name(data.get("unit", "request")) != "request":
        raise ValueError("price_per_request is a price per request: leave unit out or set it to 'request'")
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
        "unit": "request",
        "input_price_per_million": float(per_million),
        "output_price_per_million": 0.0,
    }


# The fields ``rates`` carries, so a body that gives it gives none of them.
_RATES_REPLACES = (
    "input_price_per_million",
    "output_price_per_million",
    "cache_read_price_per_million",
    "cache_write_price_per_million",
    "pricing_tiers",
    "unit",
    "price_per_request",
)


def rates_shorthand(data: Any) -> Any:
    """Read the per-million fields out of ``rates``, refusing a mix of the two.

    A body that gives ``rates`` gives every price there, so the six fields are
    written from it: the unit, the two rates, the cache rates (``None`` where
    the block has none, so an earlier version's are not inherited) and the
    tiers. A block no column can hold is refused with its reason.
    """
    if not isinstance(data, dict) or data.get("rates") is None:
        return data
    mixed = [name for name in _RATES_REPLACES if data.get(name) not in (None, [])]
    if mixed:
        raise ValueError(f"rates carries every price; remove {', '.join(mixed)}")
    if isinstance(data["rates"], dict):
        unknown = [key for key in data["rates"] if key not in (*MODALITIES, "web_search", "multipliers")]
        if unknown:
            raise ValueError(
                f"rates price a modality not known here: {', '.join(unknown)}; one of {', '.join(MODALITIES)}"
            )
    try:
        # The strict shape first, so a typo or a negative price is named as such before the columns are read.
        RatesDocument.model_validate(data["rates"])
        columns = columns_from_rates(data["rates"])
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from None
    return {
        **data,
        "unit": columns.unit,
        "input_price_per_million": float(columns.input_price_per_million),
        "output_price_per_million": float(columns.output_price_per_million),
        "cache_read_price_per_million": _float_or_none(columns.cache_read_price_per_million),
        "cache_write_price_per_million": _float_or_none(columns.cache_write_price_per_million),
        "pricing_tiers": [dict(tier) for tier in columns.pricing_tiers],
    }


def _float_or_none(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


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
    unit: AcceptedPriceUnit = Field(default="token", description=PRICE_UNIT_DESCRIPTION)
    price_per_request: float | None = Field(
        default=None, ge=0, allow_inf_nan=False, description=PRICE_PER_REQUEST_DESCRIPTION
    )
    rates: RatesDocument | None = Field(default=None, description=RATES_DESCRIPTION)

    @model_validator(mode="before")
    @classmethod
    def _expand_shorthands(cls, data: Any) -> Any:
        return price_per_request_shorthand(rates_shorthand(data))

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
            "Audio transcription and moderation endpoints are always exempt: they have no token-based pricing."
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
            "Whether the catalog update (the latest models.dev prices, then every hosted provider's models) "
            "runs on its own. 'manual': only when an operator runs Update catalog on Hosted providers. "
            "'review': run every pricing_refresh_interval_seconds and hold changed prices for an operator "
            "to approve. 'auto': run on that schedule and apply them."
        ),
    )
    pricing_refresh_interval_seconds: Annotated[int, Shown(SettingsGroup.METERING)] = Field(
        default=86400,
        ge=300,
        description="How often the scheduled catalog update runs when pricing_refresh is review or auto.",
    )
