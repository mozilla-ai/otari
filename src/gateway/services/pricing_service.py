"""Shared pricing lookup utilities."""

import uuid
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import NamedTuple

from sqlalchemy import ColumnElement, case, distinct, func, inspect, not_, or_, select, true
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import API_ROOT
from gateway.core.metered_pricing import priced_per_request
from gateway.log_config import logger
from gateway.models.pricing import OPERATOR_ORIGIN, ModelPricing, OrganizationModelPricing, PriceSource
from gateway.models.usage import PRICING_REFERENCE_MAX_LENGTH
from gateway.services.pricing import (
    PROVIDER_TOOL_LIST_PRICES,
    ModelsDevPrice,
    ToolCharges,
    ToolRate,
    ToolRateLookup,
    active_timeline,
    billed_search_units,
    charge_tool_calls,
    paid_by_the_deployment,
    request_rate,
    resolve_as_of,
    tool_rate_lookup,
)
from gateway.types.tool_charges import (
    GATEWAY_TOOL_PRICING_PROVIDER as GATEWAY_TOOL_PRICING_PROVIDER,
)
from gateway.types.tool_charges import (
    LabelCalls,
)
from gateway.types.tool_charges import (
    gateway_tool_pricing_key as gateway_tool_pricing_key,
)

# Bound on the model keys named in one ``IN()`` list, so a batched override load
# stays under SQLite's default limit of 999 bind parameters in a statement.
_KEY_CHUNK = 500


# Process-wide toggle for the models.dev default fallback, set once at startup
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
    """Whether the models.dev default fallback is consulted on a DB miss."""

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


def normalize_effective_at(value: datetime | None) -> datetime:
    """Normalize a datetime to an aware UTC timestamp, defaulting to now."""

    normalized = value or datetime.now(UTC)
    if normalized.tzinfo is None:
        return normalized.replace(tzinfo=UTC)
    return normalized.astimezone(UTC)


def _resolve_default(provider: str | None, model: str, as_of: datetime) -> ModelsDevPrice | None:
    """The models.dev entry pricing ``model`` at ``as_of``, or ``None`` on a miss.

    A date no generation covers is a miss too, so callers degrade to unpriced.

    Matching rules live in ``ModelsDevPriceIndex.resolve``. The any-llm
    implementation behind an instance is looked up per call because it is
    registered state: re-typing an instance in the dashboard takes effect at once.
    """
    return resolve_as_of(active_timeline(), as_of, provider, model, _provider_implementation(provider))


def model_context_window(provider: str | None, model: str, as_of: datetime | None = None) -> int | None:
    """Context-window token limit for a model from models.dev, or ``None``.

    Metadata, not pricing: this is resolved regardless of the ``default_pricing``
    toggle (a context window is not a cost), and some catalog entries carry no
    value, in which case ``None`` is returned.
    """

    entry = _resolve_default(provider, model, normalize_effective_at(as_of))
    return None if entry is None else entry.context_window


def default_pricing_reference(provider: str | None, model: str, as_of: datetime | None = None) -> str | None:
    """Which models.dev entry :func:`default_model_pricing` would price a model from.

    ``models.dev:<provider>/<model>`` in the catalog's own spelling, or ``None``
    on a miss. The match walks several fallbacks, so the entry that answers is
    often not the one the selector named (a Bedrock id priced under a geo-less
    spelling, a bare name matched across providers); saying which one is what
    lets a reader judge whether the default is the right rate rather than a
    plausible one.
    """
    entry = _resolve_default(provider, model, normalize_effective_at(as_of))
    return None if entry is None else entry.reference


def default_model_pricing(provider: str | None, model: str, as_of: datetime) -> ModelPricing | None:
    """Resolve default pricing for a model from the models.dev catalog.

    Returns a *transient* (unpersisted) ``ModelPricing`` carrying the per-million
    rates of the snapshot in force at ``as_of``, or ``None`` when no matching
    model is found, no snapshot covers ``as_of``, or the match carries no input
    or no output rate: a missing rate is never read as free. The returned object
    is never added to a session: it is a lookup result, not a stored price, so
    explicit config/API pricing always wins (the DB is consulted first) and
    ``require_pricing`` still fails closed for genuinely unknown models.

    Whether this fallback runs at all is the caller's decision (the
    ``default_pricing`` config field, gating ``find_model_pricing``).

    Context-tiered pricing keeps its thresholds. A provider-agnostic match may
    resolve an ambiguous model *name* to a different provider's rate only when
    every provider listing it agrees on the price.
    """
    resolved = default_pricing_match(provider, model, as_of)
    return None if resolved is None else resolved[0]


def default_pricing_match(provider: str | None, model: str, as_of: datetime) -> tuple[ModelPricing, str] | None:
    """:func:`default_model_pricing` with the reference of the entry that answered, from one resolution."""
    entry = _resolve_default(provider, model, as_of)
    if entry is None or entry.input is None or entry.output is None:
        return None

    model_key = f"{provider}:{model}" if provider else model
    logger.debug(
        "Using models.dev default pricing for '%s' (matched %s)",
        # codeql[py/clear-text-logging-sensitive-data]
        model_key,
        entry.reference,
    )
    pricing = ModelPricing(
        model_key=model_key,
        effective_at=as_of,
        input_price_per_million=entry.input,
        output_price_per_million=entry.output,
        cache_read_price_per_million=entry.cache_read,
        cache_write_price_per_million=entry.cache_write,
        pricing_tiers=entry.pricing_tiers(),
    )
    return pricing, entry.reference


def override_as_model_pricing(override: OrganizationModelPricing) -> ModelPricing:
    """Present an organization's override as a transient ``ModelPricing``.

    The same trick :func:`default_model_pricing` uses for a models.dev match,
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
    effective_at = normalize_effective_at(override.effective_from)
    return ModelPricing(
        model_key=override.model_key,
        effective_at=effective_at,
        input_price_per_million=override.input_price_per_million,
        output_price_per_million=override.output_price_per_million,
        cache_read_price_per_million=override.cache_read_price_per_million,
        cache_write_price_per_million=override.cache_write_price_per_million,
        pricing_tiers=override.pricing_tiers or [],
        unit=override.unit or "tokens",
    )


async def _find_organization_override(
    db: AsyncSession,
    organization_id: uuid.UUID,
    model_keys: list[str],
    as_of: datetime,
    *,
    operator_set_only: bool = False,
) -> OrganizationModelPricing | None:
    """An organization's override for the first of ``model_keys`` that has one.

    ``operator_set_only`` leaves out every override the organization wrote
    itself, for a request the deployment paid for (see
    :func:`resolve_model_pricing`).

    One statement over every candidate key, not one per key. This runs on the
    request path ahead of the deployment lookup, so an organization with no
    override at all pays for it on every request and it has to be a single index
    probe rather than a probe per key form.

    Preference, not just a filter: ``model_keys`` is ordered (canonical
    ``provider:model`` before the legacy ``provider/model``), and the ``CASE``
    below carries that order into SQL so an override matches on the same key the
    deployment row would rather than on whichever form happened to be stored.
    A per-key loop expressed the same preference by asking twice.

    The period test is half-open: ``effective_from`` is inclusive and
    ``effective_to`` exclusive, so two adjacent periods that share an instant
    (one ending exactly where the next begins) resolve to the later one and are
    not an overlap. The service applies the same rule when it refuses one, so
    what is storable and what is resolvable agree.
    """
    key_preference = case(
        {model_key: rank for rank, model_key in enumerate(model_keys)},
        value=OrganizationModelPricing.model_key,
        else_=len(model_keys),
    )
    stmt = (
        select(OrganizationModelPricing)
        .where(
            OrganizationModelPricing.organization_id == organization_id,
            OrganizationModelPricing.model_key.in_(model_keys),
            OrganizationModelPricing.effective_from <= as_of,
            or_(
                OrganizationModelPricing.effective_to.is_(None),
                OrganizationModelPricing.effective_to > as_of,
            ),
            OrganizationModelPricing.origin == OPERATOR_ORIGIN if operator_set_only else true(),
        )
        # At most one row per key can match once the overlap rule holds, so the
        # second ordering term only decides between a row that predates the rule
        # and one that slipped through the write race the model documents; it
        # resolves to the newest applicable period rather than an arbitrary one.
        .order_by(key_preference, OrganizationModelPricing.effective_from.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


class OverridePeriod(NamedTuple):
    """One organization override, normalized for in-memory resolution.

    The stored row carries a period and a rate; this carries the period beside
    the rate already presented as a ``ModelPricing``, so a caller resolving many
    timestamps against one preloaded set does the conversion once per row rather
    than once per timestamp.
    """

    effective_from: datetime
    effective_to: datetime | None
    pricing: ModelPricing
    origin: str | None = None


async def load_organization_override_index(
    db: AsyncSession,
    organization_id: uuid.UUID,
    model_keys: Iterable[str],
) -> dict[str, list[OverridePeriod]]:
    """Every override period this organization holds for ``model_keys``.

    The bulk counterpart to :func:`_find_organization_override`, for a caller
    pricing a batch of events that share an organization but each carry their own
    timestamp: one query for the whole batch instead of one per event. The
    periods are returned rather than a rate, because which one applies is decided
    per event by :func:`resolve_organization_override`.

    Chunked over the key list for the same reason the batch's other ``IN()``
    lookups are: a batch naming many models would otherwise exceed SQLite's
    default bound on bind parameters in one statement.
    """
    keys = sorted(set(model_keys))
    index: dict[str, list[OverridePeriod]] = {}
    for start in range(0, len(keys), _KEY_CHUNK):
        chunk = keys[start : start + _KEY_CHUNK]
        stmt = select(OrganizationModelPricing).where(
            OrganizationModelPricing.organization_id == organization_id,
            OrganizationModelPricing.model_key.in_(chunk),
        )
        for row in (await db.execute(stmt)).scalars():
            # ``effective_to`` keeps its ``None``: an open-ended period means "no
            # end", where ``normalize_effective_at`` would read it as "ends now".
            effective_to = normalize_effective_at(row.effective_to) if row.effective_to is not None else None
            index.setdefault(row.model_key, []).append(
                OverridePeriod(
                    normalize_effective_at(row.effective_from),
                    effective_to,
                    override_as_model_pricing(row),
                    row.origin,
                )
            )
    for periods in index.values():
        periods.sort(key=lambda period: period.effective_from)
    return index


def resolve_organization_override(
    index: dict[str, list[OverridePeriod]],
    model_keys: Sequence[str],
    as_of: datetime,
    *,
    deployment_paid: bool = False,
) -> ModelPricing | None:
    """The override applying at ``as_of``, resolved from a preloaded index.

    ``deployment_paid`` reads only the overrides the deployment operator set,
    as :func:`resolve_model_pricing` does for such a request.

    The in-memory statement of the same rule :func:`_find_organization_override`
    expresses in SQL, and it has to stay the same rule: an event priced through
    this path and the same event priced through the request path must resolve to
    one rate (``tests/unit/test_organization_pricing_resolution.py`` pins that).

    Key preference before period: ``model_keys`` is ordered (canonical
    ``provider:model`` before the legacy ``provider/model``), and the first key
    holding *any* applicable period wins, exactly as the SQL ``CASE`` plus
    ``LIMIT 1`` decides it. The period test is half-open, ``effective_from``
    inclusive and ``effective_to`` exclusive, so two adjacent periods that share
    an instant resolve to the later one. Within a key the newest applicable
    period wins, which is the in-memory form of ``effective_from DESC``.
    """
    for model_key in model_keys:
        match: ModelPricing | None = None
        for period in index.get(model_key, ()):
            if period.effective_from > as_of:
                break
            if deployment_paid and period.origin != OPERATOR_ORIGIN:
                continue
            if period.effective_to is None or period.effective_to > as_of:
                match = period.pricing
        if match is not None:
            return match
    return None


def pricing_key_forms(model_key: str) -> list[str]:
    """The stored key forms a lookup offers for a selector, canonical first.

    A row written before keys were canonicalized may still spell the legacy
    ``provider/model``. Settlement, an organization override and the catalog all
    offer the same forms in the same order, so a rate one of them finds cannot be
    a rate another misses.
    """
    instance, separator, model = model_key.partition(":")
    return [model_key, f"{instance}/{model}"] if separator else [model_key]


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


def _canonical_key_form(model_key: str) -> str:
    """The ``provider:model`` spelling of a key stored in the legacy form."""
    instance, separator, model = model_key.partition("/")
    return f"{instance}:{model}" if separator and ":" not in model_key else model_key


async def rates_in_force(
    db: AsyncSession,
    *,
    as_of: datetime | None = None,
    limit: int,
    exclude_key_prefix: str | None = None,
) -> list[ModelPricing]:
    """Each priced model's stored rate in force at ``as_of``, newest key first.

    The bulk form of what :func:`find_model_pricing` resolves for one model, and
    it has to agree with it: a model stored under both the canonical
    ``provider:model`` key and the legacy ``provider/model`` one is reported
    once, under the key a lookup would return, so a report cannot name a row
    settlement would never pick.
    """
    lookup_time = normalize_effective_at(as_of)
    latest_effective = (
        select(ModelPricing.model_key.label("model_key"), func.max(ModelPricing.effective_at).label("effective_at"))
        .where(ModelPricing.effective_at <= lookup_time)
        .group_by(ModelPricing.model_key)
        .subquery()
    )
    stmt = select(ModelPricing).join(
        latest_effective,
        (ModelPricing.model_key == latest_effective.c.model_key)
        & (ModelPricing.effective_at == latest_effective.c.effective_at),
    )
    if exclude_key_prefix is not None:
        stmt = stmt.where(ModelPricing.model_key.notlike(f"{exclude_key_prefix}%"))
    # One more than asked for is never returned; the caller's bound is the wire
    # bound. Ordered by key so the page is stable between reads.
    rows = list((await db.execute(stmt.order_by(ModelPricing.model_key))).scalars())
    stored = {row.model_key for row in rows}
    in_force = [row for row in rows if _canonical_key_form(row.model_key) not in stored - {row.model_key}]
    return in_force[:limit]


async def _keys_shadowed_by_their_canonical_form(db: AsyncSession) -> set[str]:
    """Legacy ``provider/model`` keys that also exist as ``provider:model``.

    A lookup resolves such a model to the canonical row, so the legacy one is a
    rate nothing is ever metered at and listing it would report one model twice.
    :func:`rates_in_force` drops it after reading everything, which a paged read
    cannot do: dropping rows after the window makes a short page and a count
    that disagrees with it. So the keys are resolved first and excluded in SQL.

    Two statements rather than string surgery in the query: splitting on the
    *first* separator is what :func:`_canonical_key_form` means, and neither
    ``replace`` (which takes every occurrence) nor a portable ``position`` says
    that across both PostgreSQL and SQLite. A write normalizes its key
    (``normalize_pricing_key``), so the legacy form only survives in older rows
    and the first statement usually answers empty.
    """

    legacy = set(
        (
            await db.scalars(
                select(distinct(ModelPricing.model_key)).where(
                    ModelPricing.model_key.like("%/%"),
                    ModelPricing.model_key.notlike("%:%"),
                )
            )
        ).all()
    )
    if not legacy:
        return set()
    canonical = {key: _canonical_key_form(key) for key in legacy}
    wanted = sorted(set(canonical.values()))
    present: set[str] = set()
    for start in range(0, len(wanted), _KEY_CHUNK):
        chunk = wanted[start : start + _KEY_CHUNK]
        present.update(
            (await db.scalars(select(distinct(ModelPricing.model_key)).where(ModelPricing.model_key.in_(chunk)))).all()
        )
    return {key for key, form in canonical.items() if form in present}


def _outside_providers(providers: Collection[str]) -> ColumnElement[bool]:
    """Keys priced under none of ``providers``, in either spelling."""
    if not providers:
        return true()
    return not_(
        or_(
            *(
                ModelPricing.model_key.startswith(f"{provider}{separator}", autoescape=True)
                for provider in sorted(providers)
                for separator in (":", "/")
            )
        )
    )


async def pricing_versions_page(
    db: AsyncSession, *, skip: int, limit: int, hidden_providers: Collection[str] = ()
) -> list[ModelPricing]:
    """One page of every stored rate version, newest first per key, leaving out ``hidden_providers``' keys."""
    stmt = (
        select(ModelPricing)
        .where(_outside_providers(hidden_providers))
        .order_by(ModelPricing.model_key, ModelPricing.effective_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return list((await db.execute(stmt)).scalars())


async def current_rates_page(
    db: AsyncSession,
    *,
    skip: int,
    limit: int,
    as_of: datetime | None = None,
    hidden_providers: Collection[str] = (),
) -> tuple[list[ModelPricing], int]:
    """One page of each priced model's current rate, with the total model count.

    :func:`rates_in_force` answers what settlement would pick, so a key whose
    only row is scheduled for later has none. The catalog needs the wider view:
    a rate an operator has queued is one the table has to show, so this falls
    back to the earliest scheduled row where nothing has taken effect yet. The
    two agree wherever a rate is live.

    ``hidden_providers`` leaves out every key priced under those providers, and
    the count with them.
    """

    lookup_time = normalize_effective_at(as_of)
    shadowed = await _keys_shadowed_by_their_canonical_form(db)
    # One grouped pass rather than a join of a past and a future subquery: the
    # group-wide MIN is the earliest row, and COALESCE only reaches it when no
    # row has taken effect. A FULL OUTER JOIN would say the same thing and is
    # not portable to the SQLite the OSS edition runs on.
    chosen = (
        select(
            ModelPricing.model_key.label("model_key"),
            func.coalesce(
                func.max(case((ModelPricing.effective_at <= lookup_time, ModelPricing.effective_at))),
                func.min(ModelPricing.effective_at),
            ).label("effective_at"),
        )
        .where(ModelPricing.model_key.notin_(shadowed) if shadowed else true())
        .where(_outside_providers(hidden_providers))
        .group_by(ModelPricing.model_key)
        .subquery()
    )
    # ``(model_key, effective_at)`` is the primary key, so the join keeps one row
    # per model. Ordered by key so a page is stable between reads.
    stmt = (
        select(ModelPricing)
        .join(
            chosen,
            (ModelPricing.model_key == chosen.c.model_key) & (ModelPricing.effective_at == chosen.c.effective_at),
        )
        .order_by(ModelPricing.model_key)
        .offset(skip)
        .limit(limit)
    )
    rows = list((await db.execute(stmt)).scalars())
    total = select(func.count(distinct(ModelPricing.model_key))).where(_outside_providers(hidden_providers))
    if shadowed:
        total = total.where(ModelPricing.model_key.notin_(shadowed))
    count = await db.scalar(total)
    return rows, count or 0


def is_free_pricing(pricing: ModelPricing | None) -> bool:
    """Whether a resolved rate charges nothing: zero for input and for output."""
    return pricing is not None and pricing.input_price_per_million == 0 and pricing.output_price_per_million == 0


async def find_model_pricing(
    db: AsyncSession,
    provider: str | None,
    model: str,
    *,
    as_of: datetime | None = None,
    use_defaults: bool = True,
    organization_id: uuid.UUID | None = None,
    deployment_paid: bool = False,
) -> ModelPricing | None:
    """Look up model pricing as of a timestamp.

    Resolution order: the requesting organization's own override (when
    ``organization_id`` is given), then the canonical ``provider:model`` key, then
    the legacy ``provider/model`` key, then (when default pricing is enabled)
    community-maintained default pricing from models.dev. Explicit pricing
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
    a gateway-run tool): the models.dev lookup falls back to a provider-agnostic
    match on the bare name, so a tool an operator happened to name after a real
    model would pick up that model's rate. And a real model billed under a
    non-token unit (audio and moderations per request, images per image):
    ``gpt-4o-transcribe`` and ``gpt-image-1`` are both in the dataset, and their
    per-million-token rates, read under :func:`flat_request_cost` or
    :func:`per_image_cost`, become a per-request or per-image rate, writing a
    charge line at the wrong unit for a rate nobody configured.
    """

    resolved = await resolve_model_pricing(
        db,
        provider,
        model,
        as_of=as_of,
        use_defaults=use_defaults,
        organization_id=organization_id,
        deployment_paid=deployment_paid,
    )
    return resolved.pricing if resolved is not None else None


class ResolvedPricing(NamedTuple):
    """A resolved rate, which step of the lookup order supplied it, and which entry.

    ``reference`` names the entry within that source: an override's row id, the
    stored ``model_key`` a deployment rate matched on (which may be the legacy
    spelling), or the models.dev ``provider_id:model_id`` that answered.
    ``effective_at`` is when that entry took effect, and is ``None`` for a
    default, whose dataset carries no such date for the rate.
    """

    pricing: ModelPricing
    source: PriceSource
    reference: str | None = None
    effective_at: datetime | None = None


class PricingProvenance(NamedTuple):
    """The usage-row columns that say which rate priced a row, in column order."""

    source: PriceSource
    reference: str | None
    effective_at: datetime | None


def pricing_provenance(resolved: ResolvedPricing) -> PricingProvenance:
    """What a usage row priced at ``resolved`` records about the rate.

    A model key is unbounded where the reference column is not, and an
    oversized value would fail the row's insert on PostgreSQL rather than lose a
    label, so a reference past the limit is recorded as unknown.
    """
    reference = resolved.reference
    if reference is not None and len(reference) > PRICING_REFERENCE_MAX_LENGTH:
        reference = None
    return PricingProvenance(resolved.source, reference, resolved.effective_at)


async def resolve_model_pricing(
    db: AsyncSession,
    provider: str | None,
    model: str,
    *,
    as_of: datetime | None = None,
    use_defaults: bool = True,
    organization_id: uuid.UUID | None = None,
    deployment_paid: bool = False,
) -> ResolvedPricing | None:
    """:func:`find_model_pricing`, also reporting which source supplied the rate.

    ``deployment_paid`` is a request served on the deployment's own credential.
    Its organization's override prices it only when the deployment operator set
    that override: one the organization wrote, whether under a provider key it
    has since archived or before the deployment served the model, would let a
    tenant name the deployment's cost basis, down to zero.
    """
    lookup_time = normalize_effective_at(as_of)
    model_key = f"{provider}:{model}" if provider else model
    key_forms = pricing_key_forms(model_key)
    legacy_keys = key_forms[1:]

    if organization_id is not None:
        override = await _find_organization_override(
            db, organization_id, key_forms, lookup_time, operator_set_only=deployment_paid
        )
        if override is not None:
            pricing = override_as_model_pricing(override)
            return ResolvedPricing(pricing, "organization", str(override.id), pricing.effective_at)

    stored = await _find_by_model_key(db, model_key, lookup_time)

    for legacy_key in legacy_keys:
        if stored is not None:
            break
        stored = await _find_by_model_key(db, legacy_key, lookup_time)
    if stored is not None:
        return ResolvedPricing(stored, "deployment", stored.model_key, normalize_effective_at(stored.effective_at))

    if use_defaults and default_pricing_enabled():
        matched = default_pricing_match(provider, model, lookup_time)
        if matched is not None:
            return ResolvedPricing(matched[0], "defaults", matched[1])

    return None


# ``ModelPricing`` only has per-million-token rate columns, so a rate in another
# unit overloads ``input_price_per_million``. Which meter a rate prices, and how
# it reads that column, is the pricing domain's rule (``gateway.services.pricing``);
# these two read a per-request rate through it for the callers that only hold one.


def flat_request_cost(pricing: ModelPricing | None) -> Decimal:
    """USD for one request at a per-request rate, or zero when unpriced: USD per million requests, read once."""
    return request_rate(pricing)


def search_unit_cost(units: int, pricing: ModelPricing | None) -> Decimal:
    """USD for ``units`` search units, each priced as one request is."""
    return max(units, 0) * request_rate(pricing)


def metered_search_units(pricing: ModelPricing | None, billed_units: Mapping[str, float] | None) -> int | None:
    """The search units a rerank request was billed for, or ``None`` when it was not billed per unit.

    Only a model priced per request bills search units, at the count
    :func:`billed_search_units` derives from the provider's report, so this is
    the count the request's ``search_units`` billing meter records.
    """
    if not priced_per_request(pricing):
        return None
    return billed_search_units(billed_units)


def _detach(row: object) -> None:
    """Take an ORM row out of its session, keeping what it has loaded; anything else is left alone."""
    state = inspect(row, raiseerr=False)
    if state is not None and state.session is not None:
        state.session.expunge(row)


class RequestPrices:
    """The rates one request has resolved, so each model it bills is priced once.

    Admission, a fallover's top-up and settlement each price the model they are
    about to bill. A request is short, so they share the first resolution rather
    than reading the price list again; a rate that takes effect while a request
    is in flight applies from the next request on.
    """

    def __init__(self, organization_id: uuid.UUID | None) -> None:
        self.organization_id = organization_id
        self._resolved: dict[tuple[str | None, str, bool], ResolvedPricing | None] = {}

    async def resolve(
        self, db: AsyncSession, provider: str | None, model: str, *, deployment_paid: bool = False
    ) -> ResolvedPricing | None:
        """:func:`resolve_model_pricing` for this request's organization, once per model and payer."""
        key = (provider, model, deployment_paid)
        if key not in self._resolved:
            resolved = await resolve_model_pricing(
                db, provider, model, organization_id=self.organization_id, deployment_paid=deployment_paid
            )
            # Detached, so a rollback between admission and settlement, which
            # expires every instance in the session, cannot expire the rate too.
            if resolved is not None:
                _detach(resolved.pricing)
            self._resolved[key] = resolved
        return self._resolved[key]


async def price_tool_calls(
    db: AsyncSession,
    calls: Sequence[LabelCalls],
    *,
    as_of: datetime | None = None,
    organization_id: uuid.UUID | None = None,
) -> ToolCharges:
    """Price a request's successful gateway-run tool calls, per tool and label.

    The charge rule is :func:`gateway.services.pricing.charge_tool_calls`; this
    looks up the rates it needs, only for the labels with a call that reported
    no cost.
    """
    lookups = {lookup for entry in calls if (lookup := tool_rate_lookup(entry)) is not None}
    rates = await _tool_rates(db, lookups, as_of=as_of, organization_id=organization_id)
    return charge_tool_calls(calls, rates)


async def resolve_tool_rate(
    db: AsyncSession,
    rate_keys: Sequence[str],
    *,
    label: str | None = None,
    as_of: datetime | None = None,
    organization_id: uuid.UUID | None = None,
) -> ToolRate | None:
    """The rate one call's rate keys resolve to, or ``None`` when none has a pricing row.

    The lookup :func:`price_tool_calls` prices each label with, for a caller that
    holds one resolved call, such as a pricing gate. ``label`` decides whose
    overrides count, as it does at settlement.
    """
    lookup = ToolRateLookup(tuple(rate_keys), paid_by_the_deployment(label))
    return (await _tool_rates(db, {lookup}, as_of=as_of, organization_id=organization_id))[lookup]


async def _tool_rates(
    db: AsyncSession,
    lookups: Collection[ToolRateLookup],
    *,
    as_of: datetime | None,
    organization_id: uuid.UUID | None,
) -> dict[ToolRateLookup, ToolRate | None]:
    """The rate each lookup resolves to: its first rate key with a pricing row.

    Each key resolves as :func:`resolve_model_pricing` resolves one: the
    organization's override first, through :func:`resolve_organization_override`,
    the in-memory statement of the same rule, then the deployment price list in
    both spellings, canonical first. The models.dev fallback is not consulted:
    MCP tool names come from a caller-supplied server, and the fallback matches
    on a bare name, so a tool named after a real model would otherwise be billed
    at that model's per-million-token rate divided by a million.

    Every key of every lookup is read in one batch per table, because an MCP
    pool can put up to ``MAX_TOOL_NAMES`` names on one request, and this runs on
    the settlement path.
    """
    if not lookups:
        return {}
    lookup_time = normalize_effective_at(as_of)
    forms = {form for lookup in lookups for key in lookup.rate_keys for form in pricing_key_forms(key)}
    overrides = (
        await load_organization_override_index(db, organization_id, forms) if organization_id is not None else {}
    )
    deployment = await _deployment_rates(db, forms, lookup_time)
    return {lookup: _first_tool_rate(lookup, overrides, deployment, lookup_time) for lookup in lookups}


async def _deployment_rates(db: AsyncSession, model_keys: Collection[str], as_of: datetime) -> dict[str, ModelPricing]:
    """The deployment price list's version in force at ``as_of`` for each stored key that has one.

    Chunked over the key list, as :func:`load_organization_override_index` is.
    """
    keys = sorted(set(model_keys))
    found: dict[str, ModelPricing] = {}
    for start in range(0, len(keys), _KEY_CHUNK):
        chunk = keys[start : start + _KEY_CHUNK]
        stmt = (
            select(ModelPricing)
            .where(ModelPricing.model_key.in_(chunk), ModelPricing.effective_at <= as_of)
            # Oldest first, so the last row kept for a key is its newest version.
            .order_by(ModelPricing.effective_at)
        )
        for row in (await db.execute(stmt)).scalars():
            found[row.model_key] = row
    return found


def _first_tool_rate(
    lookup: ToolRateLookup,
    overrides: dict[str, list[OverridePeriod]],
    deployment: Mapping[str, ModelPricing],
    as_of: datetime,
) -> ToolRate | None:
    for rate_key in lookup.rate_keys:
        forms = pricing_key_forms(rate_key)
        override = resolve_organization_override(overrides, forms, as_of, deployment_paid=lookup.operator_set_only)
        if override is not None:
            return ToolRate(rate_key, flat_request_cost(override), "organization")
        stored = next((deployment[form] for form in forms if form in deployment), None)
        if stored is not None:
            return ToolRate(rate_key, flat_request_cost(stored), "deployment")
    if default_pricing_enabled():
        for rate_key in lookup.rate_keys:
            list_price = PROVIDER_TOOL_LIST_PRICES.get(rate_key)
            if list_price is not None:
                return ToolRate(rate_key, list_price, "defaults")
    return None


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
    request; the cause and its fixes live only in a startup log they never see,
    so spell them out in the response. With default pricing already on, the
    bundled price list missed the name, most often because it is misspelled, so
    that is named first and enabling the fallback is not offered again.
    """
    cause = f"No pricing is configured for model '{model}', and require_pricing is on, so it cannot be billed. "
    if default_pricing_enabled():
        return (
            f"{cause}Default pricing is on, and its bundled price list has no entry for this model. "
            f"Check the model name for a typo (GET {API_ROOT}/models lists the models this gateway serves), "
            f"or add pricing for a model the list does not cover (POST {API_ROOT}/pricing, "
            "or the pricing section of config.yml)."
        )
    return (
        f"{cause}Fix it either way: add pricing (POST {API_ROOT}/pricing, or the pricing section of config.yml), "
        f'or enable the default-pricing fallback (PATCH {API_ROOT}/settings {{"default_pricing": true}}) '
        "to meter with public list prices."
    )
