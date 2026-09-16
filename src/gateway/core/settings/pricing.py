"""Pricing settings."""

from datetime import datetime
from typing import Annotated, Literal

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
    cache_write_1h_price_per_million: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_has_rate_override(self) -> "PricingTierConfig":
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
    cache_write_1h_price_per_million: float | None = Field(
        default=None,
        ge=0,
        description="Price per 1M Anthropic 1-hour cache-write tokens.",
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
            "What the rates are per: 'tokens' for a model, 'requests' for a gateway-run tool or a "
            "moderation call (USD per million requests), 'images' for image generation."
        ),
    )

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
            "Reject requests for models that have no configured pricing (fail-closed, default). "
            "When False, unpriced models are served and logged without cost (legacy behavior). "
            "Audio and moderation endpoints are always exempt — they have no token-based pricing."
        ),
    )
    default_pricing: Annotated[bool, Shown(SettingsGroup.METERING)] = Field(
        default=False,
        description=(
            "When a model has no pricing in the database, fall back to community-maintained "
            "default pricing from the bundled genai-prices dataset. Off by default: a billing "
            "gateway should price from rates you control, and these community estimates can lag "
            "or differ from real provider rates. Database pricing always takes precedence. Enable "
            "to auto-price common models without configuring each one; while off, require_pricing "
            "stays fail-closed for any model you have not priced explicitly."
        ),
    )
    pricing_refresh: Annotated[Literal["manual", "review", "auto"], Shown(SettingsGroup.METERING)] = Field(
        default="manual",
        description=(
            "How the genai-prices defaults are kept current. 'manual': only when an operator checks for "
            "updates on Model pricing. 'review': fetch upstream every pricing_refresh_interval_seconds and "
            "hold the update for an operator to accept or reject. 'auto': fetch and apply on that schedule."
        ),
    )
    pricing_refresh_interval_seconds: Annotated[int, Shown(SettingsGroup.METERING)] = Field(
        default=86400,
        ge=300,
        description="How often the scheduled genai-prices check runs when pricing_refresh is review or auto.",
    )
