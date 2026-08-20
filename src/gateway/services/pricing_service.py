"""Shared pricing lookup utilities."""

import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import Decimal

from genai_prices import Usage, calc_price
from genai_prices.types import PriceCalculation, TieredPrices
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.log_config import logger
from gateway.models.entities import ModelPricing, OrganizationModelPricing

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


# Process-wide resolver from a provider *instance* name to the any-llm
# implementation backing it, set once at startup from
# ``GatewayConfig.provider_pricing_implementation`` (see
# ``configure_provider_types``). That method, and not
# ``provider_instance_type``, because a ``*-compatible`` provider_type names the
# wire protocol an endpoint speaks rather than the vendor serving it.
# It lives here for the same reason the toggle above does: pricing keys on the
# instance name, and the lookups below run deep in request/budget/catalog code
# that does not carry the config object. A callable rather than a snapshot map,
# because a provider added in the dashboard rewrites ``config.providers`` while
# the worker runs.
_provider_type_resolver: Callable[[str], str | None] | None = None


def configure_provider_types(resolver: Callable[[str], str | None] | None) -> None:
    """Register the instance to any-llm implementation lookup used for pricing."""

    global _provider_type_resolver
    _provider_type_resolver = resolver


def _provider_implementation(instance: str | None) -> str | None:
    """The any-llm implementation behind ``instance``, when it differs from it.

    ``None`` when no resolver is registered, when the instance is unconfigured, or
    when the instance name already *is* the implementation name (the common case,
    which the instance-scoped lookup covers on its own).
    """

    if instance is None or _provider_type_resolver is None:
        return None
    implementation = _provider_type_resolver(instance)
    if not implementation or implementation == instance:
        return None
    return implementation


def _flat_rate(value: Decimal | TieredPrices) -> float:
    """Collapse a genai-prices rate to a single USD-per-million float.

    Tiered models (threshold "cliff" pricing) are flattened to their ``base``
    rate, the price that applies below the first tier, which is the right default
    for the typical request that never crosses a tier boundary.
    """

    if isinstance(value, TieredPrices):
        return float(value.base)
    return float(value)


def _rate_at(value: Decimal | TieredPrices | None, threshold: int) -> float | None:
    if value is None:
        return None
    if not isinstance(value, TieredPrices):
        return float(value)
    rate = value.base
    for tier in value.tiers:
        if tier.start <= threshold:
            rate = tier.price
        else:
            break
    return float(rate)


def _pricing_tiers(price: object) -> list[dict[str, float | int]]:
    fields = {
        "input_price_per_million": getattr(price, "input_mtok"),
        "output_price_per_million": getattr(price, "output_mtok"),
        "cache_read_price_per_million": getattr(price, "cache_read_mtok"),
        "cache_write_price_per_million": getattr(price, "cache_write_mtok"),
    }
    thresholds = sorted(
        {tier.start for value in fields.values() if isinstance(value, TieredPrices) for tier in value.tiers}
    )
    return [
        {
            "min_input_tokens": threshold,
            **{field: rate for field, value in fields.items() if (rate := _rate_at(value, threshold)) is not None},
        }
        for threshold in thresholds
    ]


def normalize_effective_at(value: datetime | None) -> datetime:
    """Normalize a datetime to an aware UTC timestamp, defaulting to now."""

    normalized = value or datetime.now(UTC)
    if normalized.tzinfo is None:
        return normalized.replace(tzinfo=UTC)
    return normalized.astimezone(UTC)


# genai-prices rates and metadata are date-granular (period boundaries fall on
# dates, not times), so a model resolves to the same calculation for any instant
# within a day. Memoize by (provider, model, day) so a single GET /v1/models,
# which resolves each model twice (context window in one phase, default price in
# another), and repeated same-day billing lookups do not re-walk the dataset each
# time. Bounded by clearing at a cap; the distinct key count is roughly
# providers x models x recent days, so the cap is a backstop, not a normal path.
_PRICE_CACHE_MAX = 16384
_price_cache: dict[tuple[str | None, str | None, str, date], PriceCalculation | None] = {}


class _TransientFailure:
    """A genai-prices lookup raised rather than missing cleanly.

    Distinguishes a transient dataset/API hiccup from a genuine ``LookupError``
    miss: a miss is cached for the day, but a transient failure must be retried on
    the next request instead of pinning the model to unpriced until the date rolls.
    """


_TRANSIENT_FAILURE = _TransientFailure()


def reset_price_cache() -> None:
    """Clear the memoized genai-prices resolutions (used by tests)."""

    _price_cache.clear()


def _resolve_genai_price(provider: str | None, model: str, as_of: datetime) -> PriceCalculation | None:
    """Resolve a genai-prices calculation for a model, or ``None`` on a miss.

    Memoized per (provider, implementation, model, day); see
    ``_resolve_genai_price_uncached`` for the matching rules. The implementation is
    part of the key because it is registered state rather than a function of the
    provider name, so re-typing an instance in the dashboard must not keep serving
    the resolution made under its old ``provider_type`` for the rest of the day.
    """
    implementation = _provider_implementation(provider)
    key = (provider, implementation, model, as_of.date())
    if key in _price_cache:
        return _price_cache[key]
    result = _resolve_genai_price_uncached(provider, model, as_of, implementation)
    if isinstance(result, _TransientFailure):
        # A transient failure is not memoized: the next request retries rather
        # than inheriting a stale "unpriced" for the rest of the day.
        return None
    if len(_price_cache) >= _PRICE_CACHE_MAX:
        _price_cache.clear()
    _price_cache[key] = result
    return result


def _vendor_prefixed_attempts(model: str) -> list[tuple[str | None, str]]:
    """Split a vendor-prefixed model id into ``(vendor, model)`` candidates.

    Aggregating providers name a model after the vendor that built it
    (``anthropic.claude-sonnet-5`` on Bedrock, ``openai.gpt-oss-120b``), sometimes
    behind a region or routing prefix (``us.anthropic.claude-sonnet-5-v1:0``).
    genai-prices files those ids under the *serving* provider, so a serving
    provider it does not recognize leaves them unpriced: the provider-agnostic
    fallback matches a provider on the vendor's name appearing in the model
    (``claude`` selects ``anthropic``) and then finds no such dotted model id
    there, which no amount of retrying that lookup can fix.

    Each dot boundary is offered in turn, so a region prefix is skipped once it
    fails to name a provider. Pricing under the vendor is an approximation of the
    serving provider's rate, which is why this is tried only after every lookup
    that could be exact; a name with no vendor prefix (``gpt-4.1``,
    ``claude-3.5-sonnet``) yields candidates whose provider does not resolve, so it
    is unaffected.
    """

    attempts: list[tuple[str | None, str]] = []
    head, separator, rest = model.partition(".")
    while separator and rest:
        attempts.append((head, rest))
        head, separator, rest = rest.partition(".")
    return attempts


def _resolve_genai_price_uncached(
    provider: str | None, model: str, as_of: datetime, implementation: str | None = None
) -> PriceCalculation | None | _TransientFailure:
    """Resolve a genai-prices calculation for a model, or ``None`` on a miss.

    Shared by pricing and by metadata lookups (e.g. context window) so both apply
    the same model matching: HuggingFace pinned-backend selectors, a
    provider-scoped lookup, the backing implementation, then two fallbacks.
    """

    # Build the genai-prices lookups to try, most specific first:
    #   1. HuggingFace pinned-backend selectors (`huggingface:<model>:<backend>`,
    #      see docs/models.md) map to genai-prices' per-backend provider ids
    #      (`huggingface_<backend>`), which is where HF rates live; a bare
    #      `huggingface` provider has no rates. Auto/policy suffixes (`:cheapest`,
    #      ...) simply fail to match and fall through to require_pricing.
    #   2. The provider-scoped lookup. Note this is scoped to the *instance* name,
    #      which is what pricing keys on and is only sometimes a provider id
    #      genai-prices knows.
    #   3. The any-llm implementation backing that instance, so an instance named
    #      anything else (`aws-prod` over `provider_type: bedrock`) still resolves.
    #      Rates differ per serving provider, so this must precede any fallback:
    #      Bedrock's Sonnet is not priced like Anthropic's.
    #   4. A provider-agnostic match, so a model under a provider id genai-prices
    #      does not recognize still gets priced when its name is unambiguous.
    #   5. Vendor-prefixed model ids, which the agnostic match cannot resolve.
    attempts: list[tuple[str | None, str]] = []
    if provider == "huggingface" and ":" in model:
        base_model, backend = model.rsplit(":", 1)
        attempts.append((f"huggingface_{backend}", base_model))
    attempts.append((provider, model))
    if implementation is not None:
        attempts.append((implementation, model))
    if provider is not None:
        attempts.append((None, model))
    attempts.extend(_vendor_prefixed_attempts(model))

    for provider_id, model_ref in attempts:
        try:
            return calc_price(_ZERO_USAGE, model_ref=model_ref, provider_id=provider_id, genai_request_timestamp=as_of)
        except LookupError:
            continue
        except Exception:
            # genai-prices runs on the per-request hot path; a data/API hiccup
            # must degrade to "unpriced"/"unknown" rather than turn into a request
            # error for that model. Signal a transient failure so the caller does
            # not memoize it (the next request retries).
            logger.warning("genai-prices lookup failed for model_ref=%r provider_id=%r", model_ref, provider_id)
            return _TRANSIENT_FAILURE

    return None


def model_context_window(provider: str | None, model: str, as_of: datetime | None = None) -> int | None:
    """Context-window token limit for a model from genai-prices, or ``None``.

    Metadata, not pricing: this is resolved regardless of the ``default_pricing``
    toggle (a context window is not a cost), and many models in the dataset simply
    have no value, in which case ``None`` is returned.
    """

    calc = _resolve_genai_price(provider, model, normalize_effective_at(as_of))
    if calc is None:
        return None
    return calc.model.context_window


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

    Tiered ("cliff") pricing retains its context thresholds. A provider-agnostic
    match (below) may resolve an ambiguous model *name* to a different provider's
    rate.
    """

    calc = _resolve_genai_price(provider, model, as_of)
    if calc is None:
        return None

    price = calc.model_price
    if price.input_mtok is None:
        return None
    # Input-only models (embeddings, rerank) legitimately have no output rate;
    # price output at 0 rather than rejecting the whole model.
    output_rate = _flat_rate(price.output_mtok) if price.output_mtok is not None else 0.0
    cache_read_rate = _flat_rate(price.cache_read_mtok) if price.cache_read_mtok is not None else None
    cache_write_rate = _flat_rate(price.cache_write_mtok) if price.cache_write_mtok is not None else None

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
        cache_read_price_per_million=cache_read_rate,
        cache_write_price_per_million=cache_write_rate,
        pricing_tiers=_pricing_tiers(price),
    )


def _override_as_model_pricing(override: OrganizationModelPricing) -> ModelPricing:
    """Present an organization's override as a transient ``ModelPricing``.

    The same trick :func:`default_model_pricing` uses for a genai-prices match,
    and for the same reason: every caller of :func:`find_model_pricing`, and the
    whole cost-math core behind them, reads a ``ModelPricing``. Returning the
    override row itself would make the override the one pricing source that needs
    a different accessor at fourteen call sites.

    Never added to a session. It is a lookup result, not a stored price.

    ``effective_at`` carries the override's ``effective_from``, which is what the
    row's own "this price applies from" means. The UTC stamp is not cosmetic:
    ``DateTime(timezone=True)`` is a no-op on SQLite, so a value read back there
    is naive, and a caller comparing it against an aware timestamp would raise
    rather than compare.
    """
    effective_at = override.effective_from
    if effective_at.tzinfo is None:
        effective_at = effective_at.replace(tzinfo=UTC)
    return ModelPricing(
        model_key=override.model_key,
        effective_at=effective_at,
        input_price_per_million=override.input_price_per_million,
        output_price_per_million=override.output_price_per_million,
        cache_read_price_per_million=override.cache_read_price_per_million,
        cache_write_price_per_million=override.cache_write_price_per_million,
        cache_write_1h_price_per_million=override.cache_write_1h_price_per_million,
        pricing_tiers=override.pricing_tiers or [],
    )


async def _find_organization_override(
    db: AsyncSession,
    organization_id: uuid.UUID,
    model_keys: list[str],
    as_of: datetime,
) -> ModelPricing | None:
    """An organization's rate for the first of ``model_keys`` that has one.

    The keys are tried in the caller's order (canonical ``provider:model`` before
    the legacy ``provider/model``) so an override matches on the same key the
    deployment row would, rather than on whichever form happened to be stored.

    The period test is half-open: ``effective_from`` is inclusive and
    ``effective_to`` exclusive, so two adjacent periods that share an instant
    (one ending exactly where the next begins) resolve to the later one and are
    not an overlap. The service applies the same rule when it refuses one, so
    what is storable and what is resolvable agree.
    """
    for model_key in model_keys:
        stmt = (
            select(OrganizationModelPricing)
            .where(
                OrganizationModelPricing.organization_id == organization_id,
                OrganizationModelPricing.model_key == model_key,
                OrganizationModelPricing.effective_from <= as_of,
                or_(
                    OrganizationModelPricing.effective_to.is_(None),
                    OrganizationModelPricing.effective_to > as_of,
                ),
            )
            # At most one row can match once the overlap rule holds. The ordering
            # is what makes the read deterministic anyway, so a row that predates
            # the rule, or slipped through the write race the model documents,
            # resolves to the newest applicable period instead of an arbitrary one.
            .order_by(OrganizationModelPricing.effective_from.desc())
            .limit(1)
        )
        override = (await db.execute(stmt)).scalar_one_or_none()
        if override is not None:
            return _override_as_model_pricing(override)
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
    use_defaults: bool = True,
    organization_id: uuid.UUID | None = None,
) -> ModelPricing | None:
    """Look up model pricing as of a timestamp.

    Resolution order: the requesting organization's own override (when
    ``organization_id`` is given), then the canonical ``provider:model`` key, then
    the legacy ``provider/model`` key, then (when default pricing is enabled)
    community-maintained default pricing from genai-prices. Explicit pricing
    stored in the database always takes precedence over defaults. The default
    fallback is gated by ``GatewayConfig.default_pricing`` via
    :func:`configure_default_pricing`.

    ``organization_id`` is what makes a rate tenant-specific. Omitted, this
    resolves exactly as it did before the override table existed, which is what
    keeps every deployment-wide caller (the startup pricing warning, the catalog)
    reading the deployment's own list rather than some organization's negotiated
    one. Request-path callers pass it, resolved from the authenticating key by
    ``services.workspace_scope.organization_for_key_id``, never from a header.

    ``use_defaults=False`` skips that fallback for any caller whose billable unit
    is not a token, because every dataset rate is quoted per million *tokens*.
    Two kinds of caller need it. A key that is not a model at all (a search tool,
    a gateway-run tool): the genai-prices lookup falls back to a provider-agnostic
    match on the bare name, so a tool an operator happened to name after a real
    model would pick up that model's rate. And a real model billed under a
    non-token unit (audio and moderations per request, images per image):
    ``gpt-4o-transcribe`` and ``gpt-image-1`` are both in the dataset, and their
    per-million-token rates, read under :func:`flat_request_cost` or
    :func:`per_image_cost`, become a per-request or per-image rate, writing a
    charge line at the wrong unit for a rate nobody configured.
    """

    lookup_time = normalize_effective_at(as_of)
    model_key = f"{provider}:{model}" if provider else model
    legacy_keys = [f"{provider}/{model}"] if provider else []

    if organization_id is not None:
        override = await _find_organization_override(db, organization_id, [model_key, *legacy_keys], lookup_time)
        if override is not None:
            return override

    pricing = await _find_by_model_key(db, model_key, lookup_time)

    for legacy_key in legacy_keys:
        if pricing is not None:
            break
        pricing = await _find_by_model_key(db, legacy_key, lookup_time)

    if pricing is None and use_defaults and default_pricing_enabled():
        pricing = default_model_pricing(provider, model, lookup_time)

    return pricing


# ``ModelPricing`` only has per-million-token rate columns, so endpoints whose
# billable unit is not a token overload ``input_price_per_million`` with a
# different unit convention. Each convention gets a named helper below so the
# unit is visible at the call site instead of an anonymous expression that can
# be miscopied into a new route and misbill by a factor of a million. Dedicated
# per-unit columns would need a schema migration (deferred; see issue #259).


def input_token_cost(tokens: int, pricing: ModelPricing) -> float:
    """USD cost of ``tokens`` input tokens at the per-million-token rate.

    The standard convention: ``input_price_per_million`` is USD per million
    input tokens. Used by embeddings and rerank, which bill input tokens only.
    """
    return (tokens / 1_000_000) * pricing.input_price_per_million


def flat_request_cost(pricing: ModelPricing | None) -> float:
    """Flat USD cost of one request for a model priced per request.

    Moderations convention: ``input_price_per_million`` stores the per-request
    rate scaled by 1e6 (USD per million requests), so one request costs the
    stored rate divided by 1e6. Unpriced models are treated as free.
    """
    if pricing is None or not pricing.input_price_per_million:
        return 0.0
    return pricing.input_price_per_million / 1_000_000


PerRequestMeters = tuple[dict[str, int], list[dict[str, float | int | str]]]


def per_request_meters(cost: float) -> PerRequestMeters | None:
    """This request's billing meters and charge line, priced per request.

    One request is one billed meter, so the per-request rate is the cost itself.
    Charge lines carry ``unit_rate`` rather than ``rate_per_million``, the same
    shape :func:`price_tool_calls` writes, which is what tells a reader and the
    dashboard which unit convention applies.

    Returns ``None`` when the request is free, the common case on these routes
    since they are exempt from ``require_pricing`` and an unset or ``0.0`` rate
    both settle at $0: a zero charge line would render in Activity as a billed
    meter explaining a charge that never happened. Shared by every route billing
    per request (audio transcription, audio speech, moderations) so the shape
    cannot drift between them, for the reason the unit conventions above are
    named helpers rather than inline expressions.
    """
    if not cost:
        return None
    return {"requests": 1}, [{"meter": "request", "units": 1, "unit_rate": cost, "cost": cost}]


GATEWAY_TOOL_PRICING_PROVIDER = "otari"


def gateway_tool_pricing_key(tool: str) -> str:
    """The ``ModelPricing.model_key`` an operator prices a gateway-run tool under.

    ``model_key`` is a free-form ``provider:model`` string, so a tool the gateway
    runs itself is priced as ``otari:<tool>`` (for example ``otari:web_search``).
    Note this is a different key from the one ``POST /v1/search`` uses for the same
    search: that endpoint prices ``<search-provider>:<tool>`` because it knows which
    commercial API it called, while the tool loop only knows the operator-configured
    backend URL.
    """
    return f"{GATEWAY_TOOL_PRICING_PROVIDER}:{tool}"


async def price_tool_calls(
    db: AsyncSession,
    billable_calls: dict[str, int],
    *,
    as_of: datetime | None = None,
    organization_id: uuid.UUID | None = None,
) -> tuple[float, list[dict[str, float | int | str]], list[str]]:
    """Price a request's successful gateway-run tool calls.

    Returns the total USD cost, one auditable charge line per tool, and the names
    of the tools that had no pricing row. Charge lines use the same shape
    ``calculate_metered_cost`` produces so both kinds share
    ``UsageLog.pricing_breakdown`` and the dashboard's renderer, except that a tool
    line carries ``unit_rate`` (USD per call) where a token line carries
    ``rate_per_million``. That key is what tells a reader, and the UI, which unit
    convention applies.

    A tool with no pricing row contributes units at a zero rate, so the work stays
    on the row and in the audit trail even when the operator has not priced it.
    Lookups pass ``use_defaults=False``: MCP tool names come from a caller-supplied
    server, and the genai-prices fallback matches on a bare name, so a tool named
    after a real model would otherwise be billed at that model's
    per-million-token rate divided by a million.
    """
    tools = [tool for tool in sorted(billable_calls) if billable_calls[tool] > 0]
    if not tools:
        return 0.0, [], []
    rates = await _tool_rates(db, tools, as_of=as_of, organization_id=organization_id)

    total = 0.0
    lines: list[dict[str, float | int | str]] = []
    unpriced: list[str] = []
    for tool in tools:
        units = billable_calls[tool]
        pricing = rates.get(tool)
        if pricing is None:
            unpriced.append(tool)
        unit_rate = flat_request_cost(pricing)
        cost = units * unit_rate
        total += cost
        lines.append({"meter": f"{tool}_calls", "units": units, "unit_rate": unit_rate, "cost": cost})
    return total, lines, unpriced


async def _tool_rates(
    db: AsyncSession,
    tools: list[str],
    *,
    as_of: datetime | None,
    organization_id: uuid.UUID | None = None,
) -> dict[str, ModelPricing]:
    """Latest-as-of pricing for several gateway tools, in one query.

    One statement rather than a lookup per tool: an MCP pool can put up to
    ``MAX_TOOL_NAMES`` distinct names on a single request, and this runs on the
    settlement path. Rows come back oldest-first so the newest effective row for a key
    wins the dict assignment, the same "latest as of" rule :func:`find_model_pricing`
    applies one key at a time. The genai-prices fallback is deliberately not consulted
    (see the note in :func:`price_tool_calls`).
    """
    lookup_time = normalize_effective_at(as_of)
    keys = {f"{GATEWAY_TOOL_PRICING_PROVIDER}:{tool}": tool for tool in tools}
    stmt = (
        select(ModelPricing)
        .where(ModelPricing.model_key.in_(keys), ModelPricing.effective_at <= lookup_time)
        .order_by(ModelPricing.effective_at)
    )
    found: dict[str, ModelPricing] = {}
    for row in (await db.execute(stmt)).scalars():
        found[keys[row.model_key]] = row

    if organization_id is not None:
        # Overrides win, resolved in a second batched statement for the same
        # reason the first one is batched. This has to happen even though the
        # gate in ``_pipeline`` already consulted overrides: the gate decides
        # whether a tool *has* a rate and this decides what it *is*, so a tool
        # priced only by an override would otherwise pass the gate and then be
        # charged at zero.
        override_stmt = (
            select(OrganizationModelPricing)
            .where(
                OrganizationModelPricing.organization_id == organization_id,
                OrganizationModelPricing.model_key.in_(keys),
                OrganizationModelPricing.effective_from <= lookup_time,
                or_(
                    OrganizationModelPricing.effective_to.is_(None),
                    OrganizationModelPricing.effective_to > lookup_time,
                ),
            )
            # Oldest first, so the newest applicable period wins the assignment,
            # matching the rule the statement above uses for deployment rows.
            .order_by(OrganizationModelPricing.effective_from)
        )
        for override in (await db.execute(override_stmt)).scalars():
            found[keys[override.model_key]] = _override_as_model_pricing(override)

    return found


def per_image_cost(n_images: int, pricing: ModelPricing) -> float:
    """USD cost of ``n_images`` generated images.

    Images convention: despite the name, ``input_price_per_million`` stores raw
    USD per image (no scaling, no division).
    """
    return n_images * pricing.input_price_per_million


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


def no_pricing_error_detail(model: str) -> str:
    """The 402 body for an unpriced model: what went wrong, why, and how to fix it.

    A new operator adding a provider in the dashboard hits this on their first
    request; the cause and both fixes live only in a startup log they never see,
    so spell them out in the response.
    """
    return (
        f"No pricing is configured for model '{model}', and require_pricing is on, so it cannot be billed. "
        "Fix it either way: add pricing (POST /v1/pricing, or the pricing section of config.yml), "
        'or enable the default-pricing fallback (PATCH /v1/settings {"default_pricing": true}) '
        "to meter with public list prices."
    )
