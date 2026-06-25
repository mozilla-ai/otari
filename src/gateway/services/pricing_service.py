"""Shared pricing lookup utilities."""

from datetime import UTC, datetime
from decimal import Decimal

from genai_prices import Usage, calc_price
from genai_prices.types import TieredPrices
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.log_config import logger
from gateway.models.entities import ModelPricing

# A zero-token usage is enough to resolve a model's per-million rates from
# genai-prices without depending on real token counts.
_ZERO_USAGE = Usage(input_tokens=0, output_tokens=0)


# Process-wide toggle for the genai-prices default fallback, set once at startup
# from ``GatewayConfig.default_pricing`` (see ``configure_default_pricing``). It
# mirrors the module-level engine/session pattern in ``core.database``: pricing
# lookups happen deep in request/budget code that does not carry the config
# object, so the resolved flag lives here rather than being threaded through
# every call site. Defaults to off, matching the config field's opt-in default.
_default_pricing_enabled = False


def configure_default_pricing(enabled: bool) -> None:
    """Set whether default pricing is consulted, from ``config.default_pricing``."""

    global _default_pricing_enabled
    _default_pricing_enabled = enabled


def default_pricing_enabled() -> bool:
    """Whether the genai-prices default fallback is consulted on a DB miss."""

    return _default_pricing_enabled


def _flat_rate(value: Decimal | TieredPrices) -> float:
    """Collapse a genai-prices rate to a single USD-per-million float.

    Tiered models (threshold "cliff" pricing) are flattened to their ``base``
    rate, the price that applies below the first tier, which is the right default
    for the typical request that never crosses a tier boundary.
    """

    if isinstance(value, TieredPrices):
        return float(value.base)
    return float(value)


def normalize_effective_at(value: datetime | None) -> datetime:
    """Normalize a datetime to an aware UTC timestamp, defaulting to now."""

    normalized = value or datetime.now(UTC)
    if normalized.tzinfo is None:
        return normalized.replace(tzinfo=UTC)
    return normalized.astimezone(UTC)


def default_model_pricing(provider: str | None, model: str, as_of: datetime) -> ModelPricing | None:
    """Resolve community-maintained default pricing for a model via genai-prices.

    Returns a *transient* (unpersisted) ``ModelPricing`` carrying the per-million
    input/output rates from the bundled ``genai-prices`` dataset, or ``None`` when
    no matching model is found. The returned object is never added to a session:
    it is a lookup result, not a stored price, so explicit config/API pricing
    always wins (the DB is consulted first) and ``require_pricing`` still fails
    closed for genuinely unknown models.

    Whether this fallback runs at all is the caller's decision (the
    ``default_pricing`` config field, gating ``find_model_pricing``).

    Caveats: a model with tiered ("cliff") pricing is billed at its base rate, and
    a provider-agnostic match (below) may resolve an ambiguous model *name* to a
    different provider's rate.
    """

    # Build the genai-prices lookups to try, most specific first:
    #   1. HuggingFace pinned-backend selectors (`huggingface:<model>:<backend>`,
    #      see docs/models.md) map to genai-prices' per-backend provider ids
    #      (`huggingface_<backend>`), which is where HF rates live; a bare
    #      `huggingface` provider has no rates. Auto/policy suffixes (`:cheapest`,
    #      ...) simply fail to match and fall through to require_pricing.
    #   2. The provider-scoped lookup.
    #   3. A provider-agnostic match, so a model under a provider id genai-prices
    #      does not recognize still gets priced when its name is unambiguous.
    attempts: list[tuple[str | None, str]] = []
    if provider == "huggingface" and ":" in model:
        base_model, backend = model.rsplit(":", 1)
        attempts.append((f"huggingface_{backend}", base_model))
    attempts.append((provider, model))
    if provider is not None:
        attempts.append((None, model))

    for provider_id, model_ref in attempts:
        try:
            calc = calc_price(
                _ZERO_USAGE, model_ref=model_ref, provider_id=provider_id, genai_request_timestamp=as_of
            )
        except LookupError:
            continue
        except Exception:
            # genai-prices runs on the per-request hot path; a data/API hiccup
            # must degrade to "unpriced" (require_pricing decides) rather than
            # turn into a request error for that model.
            logger.warning("genai-prices lookup failed for model_ref=%r provider_id=%r", model_ref, provider_id)
            return None

        price = calc.model_price
        if price.input_mtok is None:
            return None
        # Input-only models (embeddings, rerank) legitimately have no output
        # rate; price output at 0 rather than rejecting the whole model.
        output_rate = _flat_rate(price.output_mtok) if price.output_mtok is not None else 0.0

        model_key = f"{provider}:{model}" if provider else model
        logger.debug(
            "Using genai-prices default pricing for '%s' (matched %s/%s)",
            model_key,
            getattr(calc.provider, "id", None),
            getattr(calc.model, "id", None),
        )
        return ModelPricing(
            model_key=model_key,
            effective_at=as_of,
            input_price_per_million=_flat_rate(price.input_mtok),
            output_price_per_million=output_rate,
        )

    return None


async def _find_by_model_key(db: AsyncSession, model_key: str, as_of: datetime) -> ModelPricing | None:
    stmt = (
        select(ModelPricing)
        .where(
            ModelPricing.model_key == model_key,
            ModelPricing.effective_at <= as_of,
        )
        .order_by(ModelPricing.effective_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def find_model_pricing(
    db: AsyncSession,
    provider: str | None,
    model: str,
    *,
    as_of: datetime | None = None,
) -> ModelPricing | None:
    """Look up model pricing as of a timestamp.

    Resolution order: the canonical ``provider:model`` key, then the legacy
    ``provider/model`` key, then (when default pricing is enabled) community-
    maintained default pricing from genai-prices. Explicit pricing stored in the
    database always takes precedence over defaults. The default fallback is gated
    by ``GatewayConfig.default_pricing`` via :func:`configure_default_pricing`.
    """

    lookup_time = normalize_effective_at(as_of)
    model_key = f"{provider}:{model}" if provider else model
    pricing = await _find_by_model_key(db, model_key, lookup_time)

    if pricing is None and provider:
        pricing = await _find_by_model_key(db, f"{provider}/{model}", lookup_time)

    if pricing is None and default_pricing_enabled():
        pricing = default_model_pricing(provider, model, lookup_time)

    return pricing


def pricing_required_but_missing(pricing: ModelPricing | None, *, require_pricing: bool) -> bool:
    """Return True when the request must be rejected for lacking pricing.

    This is the predicate behind the ``require_pricing`` config: an unpriced
    model would otherwise be served free and unmetered (the budget cap cannot
    restrain it). Callers evaluate this *after* reserving budget — so a missing
    user, a blocked user, or an exhausted budget (404/403) take precedence over
    the missing-pricing rejection (402) — and refund the reservation before
    raising. When ``require_pricing`` is False, the legacy behavior is preserved
    (the request is served and logged without cost).
    """
    return pricing is None and require_pricing
