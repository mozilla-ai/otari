"""Ingest externally-observed usage events into the usage log.

Subscription-backed agents (e.g. Claude Code) never route through the gateway, so
their usage is invisible to Otari. This service accepts normalized, content-free
usage events, prices them at the rate effective for the event's own timestamp
(the importing workspace's organization override where it has one, otherwise the
deployment price list), and records them as ``usage_logs`` rows with a ``source``
provenance tag. Imported rows are attributed to a user and carry their real
(API-equivalent) cost, but are always ``counts_toward_budget=False``:
retrospective usage cannot be reserved, so it is recorded and shown in cost
analytics without ever gating or mutating ``users.spend``.

Idempotency: rows are keyed by ``(source, source_event_id)`` (a unique constraint).
Re-submitting a batch counts already-present events as duplicates and never creates
a second row. The endpoint accepts only metadata and numeric token counts; prompts,
completions, and tool payloads are rejected by the request schema, not stored.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import NamedTuple

from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.metered_pricing import BillableUsage, ChargeLine, billable_usage, price_billable_usage
from gateway.log_config import logger
from gateway.models.entities import APIKey, ModelPricing, UsageLog, User
from gateway.services.pricing_service import (
    OverridePeriod,
    default_model_pricing,
    default_pricing_enabled,
    load_organization_override_index,
    normalize_effective_at,
    resolve_organization_override,
)
from gateway.services.workspace_scope import organization_for_workspace_id, resolve_workspace_id

# Bounds. Batch size mirrors the /v1/usage list `limit` cap; the error list is
# capped so one bad batch can't return an unbounded payload; the IN() list is
# chunked to stay under SQLite's default variable limit (999).
MAX_EVENTS_PER_BATCH = 1000
_MAX_ERRORS = 100
_IN_CHUNK = 500
# Slug pattern for a source: keep provenance identifiers boring so they are safe to
# render and group on.
_SLUG_PATTERN = r"^[a-zA-Z0-9._:-]+$"
# Identifier grammar for the other stored metadata (event id, provider, model,
# session label). Deliberately narrow: token-like values only, so an importer
# cannot smuggle prompt/response prose into these fields despite the content-free
# contract. Allows the punctuation real ids and model names use (`msg_...`,
# `openai/gpt-4o`, `project:otari`, git branches) but rejects free text.
_IDENT_PATTERN = r"^[A-Za-z0-9._:/\-]+$"
# Cap numeric fields at the usage_logs 32-bit integer column width so absurd
# counts are a 422, not a database error that fails the whole batch.
_MAX_TOKENS = 2_147_483_647
# "gateway" tags rows Otari served itself; an import claiming it would masquerade
# as native traffic in every provenance breakdown.
RESERVED_SOURCES = {"gateway"}


class ExternalUsageEvent(BaseModel):
    """One imported usage event. Content-free: token counts and metadata only.

    ``extra="forbid"`` rejects any unexpected field (e.g. a stray ``prompt`` or
    ``completion``) with a 422 rather than silently dropping it, so no prompt or
    completion text can ever be accepted here.
    """

    model_config = ConfigDict(extra="forbid")

    source_event_id: str = Field(min_length=1, max_length=256, pattern=_IDENT_PATTERN)
    timestamp: datetime
    provider: str = Field(min_length=1, max_length=128, pattern=_IDENT_PATTERN)
    model: str = Field(min_length=1, max_length=256, pattern=_IDENT_PATTERN)
    status: str = Field(default="success", pattern=r"^(success|error)$")
    # Token counts are capped at the usage_logs column width (32-bit signed int);
    # anything larger would fail the INSERT with a 500 instead of a 422, and no
    # real request has ever been within orders of magnitude of the cap.
    input_tokens: int = Field(default=0, ge=0, le=_MAX_TOKENS)
    output_tokens: int = Field(default=0, ge=0, le=_MAX_TOKENS)
    cache_read_tokens: int = Field(default=0, ge=0, le=_MAX_TOKENS)
    cache_write_tokens: int = Field(default=0, ge=0, le=_MAX_TOKENS)
    cache_write_1h_tokens: int = Field(default=0, ge=0, le=_MAX_TOKENS)
    # Whether ``input_tokens`` already includes the cache counts (OpenAI shape,
    # where ``cached_tokens`` is a subset of ``prompt_tokens``) or excludes them
    # (Anthropic / Claude Code shape, where the cache buckets are additive). The
    # cost calculation reads this to avoid double-counting cached tokens. Defaults
    # to the Anthropic shape, matching this endpoint's documented additive counts.
    cache_tokens_in_prompt: bool = Field(default=False)
    duration_ms: int | None = Field(default=None, ge=0, le=_MAX_TOKENS)
    session_label: str | None = Field(default=None, max_length=256, pattern=_IDENT_PATTERN)
    # Optional per-event attribution overriding the batch default. A single
    # collector feed carries many users, so per-event user_id lets one pipeline
    # serve a whole team.
    user_id: str | None = Field(default=None, max_length=256)


class ExternalEventsRequest(BaseModel):
    """A batch of imported usage events sharing a source and default user.

    ``extra="forbid"`` here mirrors the per-event schema: a stray content field at
    the batch level (e.g. a top-level ``prompt``) is a 422, not silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    source: str = Field(pattern=_SLUG_PATTERN, max_length=64)
    # Default attribution for events without their own user_id. Optional: when the
    # caller authenticates with an API key, usage binds to that key's user and this
    # may be omitted. The master key must supply it (it can name any user).
    user_id: str | None = Field(default=None, min_length=1, max_length=256)
    events: list[ExternalUsageEvent] = Field(min_length=1, max_length=MAX_EVENTS_PER_BATCH)

    @field_validator("source")
    @classmethod
    def _not_reserved(cls, value: str) -> str:
        if value.lower() in RESERVED_SOURCES:
            raise ValueError("source 'gateway' is reserved for usage Otari served itself; pick another slug.")
        return value


class ExternalIngestError(BaseModel):
    """A single rejected event, with enough context to fix it and retry."""

    index: int
    source_event_id: str | None
    detail: str


class ExternalIngestResult(BaseModel):
    """Per-batch outcome. Re-submitting is safe: prior events count as duplicates."""

    accepted: int = 0
    duplicate: int = 0
    rejected: int = 0
    errors: list[ExternalIngestError] = Field(default_factory=list)


def _resolve_target_user(
    event_user: str | None,
    batch_user: str | None,
    key_user: str | None,
    *,
    is_master: bool,
    reject_mismatch: bool,
) -> tuple[str | None, str | None]:
    """Resolve the user an imported event is attributed to, returning (user, error).

    Mirrors the request-path binding rule (see resolve_user_id): the master key may
    name any user, but an API key binds usage to its own user. Naming a different
    user through an API key is rejected (strict) or ignored (lenient) so a key can
    never attribute cost to someone else.
    """
    requested = event_user or batch_user
    if is_master:
        if requested is None:
            return None, "user_id is required for master-key ingestion (set the batch user_id or a per-event user_id)."
        return requested, None
    if key_user is None:
        return None, "This API key has no associated user to attribute usage to."
    if requested is not None and requested != key_user and reject_mismatch:
        return None, (
            f"user_id '{requested}' does not match this API key's user '{key_user}'. "
            "Omit user_id to attribute to the key's user, or use the master key to import for other users."
        )
    return key_user, None


def _reject(result: ExternalIngestResult, index: int, source_event_id: str | None, detail: str) -> None:
    result.rejected += 1
    if len(result.errors) < _MAX_ERRORS:
        result.errors.append(ExternalIngestError(index=index, source_event_id=source_event_id, detail=detail))


async def _existing_event_ids(db: AsyncSession, source: str, event_ids: list[str]) -> set[str]:
    """Return the subset of ``event_ids`` already present for ``source``.

    Chunked so a full 1000-event batch stays under SQLite's default bound on the
    number of bind parameters in a single statement.
    """
    found: set[str] = set()
    for start in range(0, len(event_ids), _IN_CHUNK):
        chunk = event_ids[start : start + _IN_CHUNK]
        if not chunk:
            continue
        rows = (
            await db.execute(
                select(UsageLog.source_event_id).where(
                    UsageLog.source == source,
                    UsageLog.source_event_id.in_(chunk),
                )
            )
        ).scalars().all()
        found.update(row for row in rows if row is not None)
    return found


def _model_keys(provider: str, model: str) -> list[str]:
    """Candidate pricing keys, canonical then legacy, matching find_model_pricing."""
    return [f"{provider}:{model}", f"{provider}/{model}"]


class BatchPricing(NamedTuple):
    """Every rate a batch could resolve against, preloaded.

    Two shapes because the two tables are two shapes: ``model_pricing`` is a
    version series per key (the newest row effective at or before the timestamp
    wins) and ``organization_model_pricing`` is a set of periods per key (the
    period containing the timestamp wins). Both are kept whole rather than
    collapsed to a rate, because a batch carries one timestamp per event.
    """

    deployment: dict[str, list[tuple[datetime, ModelPricing]]]
    overrides: dict[str, list[OverridePeriod]]


async def _load_pricing_index(
    db: AsyncSession, pairs: set[tuple[str, str]], organization_id: uuid.UUID | None
) -> BatchPricing:
    """Load every stored rate for the batch's (provider, model) pairs up front, so
    pricing is resolved in memory instead of once per event.

    A Claude Code batch carries a few models but a distinct timestamp per event, so
    a per-event ``find_model_pricing`` would issue ~one query per event (an N+1). We
    instead pull the full history for every model key in a fixed handful of queries
    and pick what applies per timestamp below; the genai default fallback is
    memoized in pricing_service and hits no database.

    ``organization_id`` is the importing workspace's organization, resolved from
    the authenticating key rather than from the request, exactly as the request
    path resolves it. ``None`` skips the override query entirely, but that is a
    guard rather than a common path: ``workspace.organization_id`` is NOT NULL
    and a key's workspace is a RESTRICT foreign key, so an import that reaches
    here has an organization and pays for the one extra indexed lookup whether or
    not any override exists.
    """
    keys: set[str] = set()
    for provider, model in pairs:
        keys.update(_model_keys(provider, model))
    index: dict[str, list[tuple[datetime, ModelPricing]]] = {}
    for start in range(0, len(keys), _IN_CHUNK):
        chunk = list(keys)[start : start + _IN_CHUNK]
        if not chunk:
            continue
        rows = (
            await db.execute(select(ModelPricing).where(ModelPricing.model_key.in_(chunk)))
        ).scalars().all()
        for row in rows:
            index.setdefault(row.model_key, []).append((normalize_effective_at(row.effective_at), row))
    for entries in index.values():
        entries.sort(key=lambda pair: pair[0])
    overrides: dict[str, list[OverridePeriod]] = {}
    if organization_id is not None:
        overrides = await load_organization_override_index(db, organization_id, keys)
    return BatchPricing(deployment=index, overrides=overrides)


def _resolve_pricing(
    index: BatchPricing,
    provider: str,
    model: str,
    timestamp: datetime,
) -> ModelPricing | None:
    """Effective pricing at the event timestamp, resolved from the preloaded index.

    Mirrors ``find_model_pricing``, whole: the importing organization's own
    override, then the canonical key, then the legacy key, then the genai default
    (when enabled). Stored rows win over the default, and the newest row effective
    at or before the timestamp is chosen, so historical rates are honored.

    **The timestamp is the event's own, not the import's.** Usage imported today
    for last month prices at the rate that was in force last month, on both
    layers, which is what makes an imported row comparable to a gateway row from
    the same instant. Note this is about the *first* import of an event, not a
    re-import: ingest is idempotent on ``(source, source_event_id)`` and never
    updates a row, so a row's cost is settled once, by the rate table as it stood
    when that row landed.

    An override wins on either key spelling before the deployment list is
    consulted on either, because that is the order ``find_model_pricing`` applies:
    it asks ``_find_organization_override`` for all candidate keys at once and
    only falls through to ``_find_by_model_key`` when that misses.
    """
    lookup = normalize_effective_at(timestamp)
    override = resolve_organization_override(index.overrides, _model_keys(provider, model), lookup)
    if override is not None:
        return override
    for key in _model_keys(provider, model):
        match: ModelPricing | None = None
        for effective_at, row in index.deployment.get(key, ()):
            if effective_at <= lookup:
                match = row
            else:
                break
        if match is not None:
            return match
    if default_pricing_enabled():
        return default_model_pricing(provider, model, lookup)
    return None


def _build_usage(event: ExternalUsageEvent) -> BillableUsage:
    """Normalize an imported event's token counts for pricing.

    The submitter states the cached-token convention per event (the default is
    the additive Anthropic / Claude Code shape), and it is passed straight
    through to the cost core, which has no default for it. An ingest that
    stopped carrying the flag would therefore fail to compile rather than
    quietly price one shape as the other. See ``core/metered_pricing`` for what
    the two conventions are.
    """
    return billable_usage(
        input_tokens=event.input_tokens,
        output_tokens=event.output_tokens,
        cache_read_tokens=event.cache_read_tokens,
        cache_write_tokens=event.cache_write_tokens,
        cache_write_1h_tokens=event.cache_write_1h_tokens,
        cache_tokens_included=event.cache_tokens_in_prompt,
    )


def _build_row(
    source: str,
    user_id: str,
    event: ExternalUsageEvent,
    pricing: ModelPricing | None,
    api_key_id: str | None,
    workspace_id: uuid.UUID,
) -> UsageLog:
    cost: Decimal | None = None
    meters: dict[str, int] | None = None
    breakdown: list[ChargeLine] | None = None
    if pricing is not None:
        cost, meters, breakdown = price_billable_usage(pricing, _build_usage(event))
    return UsageLog(
        workspace_id=workspace_id,
        api_key_id=api_key_id,
        user_id=user_id,
        timestamp=event.timestamp,
        model=event.model,
        provider=event.provider,
        endpoint="external",
        source=source,
        source_event_id=event.source_event_id,
        source_label=event.session_label,
        counts_toward_budget=False,
        prompt_tokens=event.input_tokens,
        completion_tokens=event.output_tokens,
        total_tokens=event.input_tokens + event.output_tokens,
        cache_read_tokens=event.cache_read_tokens,
        cache_write_tokens=event.cache_write_tokens,
        cache_write_1h_tokens=event.cache_write_1h_tokens,
        # Persist the convention the submitter stated, so a row repriced later
        # (POST /v1/usage/set-price) is priced the way it was reported rather than
        # the way its numbers happen to look. A row that arrives with no rate to
        # price it against has no billing meters for the recovery to read.
        cache_tokens_in_prompt=event.cache_tokens_in_prompt,
        billing_meters=meters,
        pricing_breakdown=breakdown,
        cost=cost,
        status=event.status,
        latency_ms=event.duration_ms,
    )


async def _insert_rows(
    db: AsyncSession,
    source: str,
    rows: list[UsageLog],
    result: ExternalIngestResult,
) -> None:
    """Commit new rows, treating unique-constraint collisions as duplicates.

    A concurrent import of the same event races on ``(source, source_event_id)``.
    On IntegrityError the whole batch rolls back, so re-query what now exists and
    retry only the survivors. If the retry commit also collides (a still-racing
    writer landed more ids in the window), fall back to row-at-a-time inserts so
    events that collide with nothing are not misreported as duplicates.
    """
    if not rows:
        return
    db.add_all(rows)
    try:
        await db.commit()
        result.accepted += len(rows)
        return
    except IntegrityError:
        await db.rollback()

    existing = await _existing_event_ids(db, source, [r.source_event_id for r in rows if r.source_event_id])
    survivors = [r for r in rows if r.source_event_id not in existing]
    result.duplicate += len(rows) - len(survivors)
    if not survivors:
        return
    db.add_all(survivors)
    try:
        await db.commit()
        result.accepted += len(survivors)
        return
    except IntegrityError:
        await db.rollback()

    still_colliding = 0
    for row in survivors:
        db.add(row)
        try:
            await db.commit()
            result.accepted += 1
        except IntegrityError:
            await db.rollback()
            # This specific event is landing via a concurrent request (or, rarely,
            # violates another constraint such as a just-deleted API key). Count it
            # as duplicate rather than retry indefinitely.
            result.duplicate += 1
            still_colliding += 1
    if still_colliding:
        logger.warning("external usage: %d events still colliding after retry for source=%s", still_colliding, source)


async def ingest_external_events(
    db: AsyncSession,
    request: ExternalEventsRequest,
    *,
    api_key: APIKey | None,
    is_master_key: bool,
    reject_user_mismatch: bool,
) -> ExternalIngestResult:
    """Validate, price, and persist a batch of imported usage events.

    Attribution binds to the authenticated principal: the master key may name any
    user, an API key binds usage to its own user (and stamps its id on the rows).
    Never touches ``users.spend``/budgets: rows are written ``counts_toward_budget``
    False. Idempotent by ``(source, source_event_id)``.

    An API key used for import MUST be budget-exempt. Imported usage is
    retrospective, so it can never be blocked before it happens; letting it count
    toward an enforced budget would let a user silently exceed a budget we had no
    way to enforce. Requiring an exempt key keeps the invariant honest and explicit.
    The master key (admin) is always allowed and imports as observability.
    """
    if api_key is not None and not api_key.exclude_from_budget:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "External usage can only be imported with a budget-exempt API key (or the master key). "
                "Imported usage is retrospective and cannot be budget-enforced, so it must not count toward "
                "a budget. Set exclude_from_budget on this key (Keys page toggle, or PATCH /v1/keys/{id}), "
                "or import with a key that already has it."
            ),
        )
    result = ExternalIngestResult()
    # A key's own ``reject_user_mismatch`` overrides the deployment-wide setting
    # here too, so the per-key override means the same thing on the import path as
    # on the request path (see resolve_user_id). Either way usage binds to the
    # key's own user, never to the one the request named.
    key_override = api_key.reject_user_mismatch if api_key is not None else None
    reject_mismatch = reject_user_mismatch if key_override is None else key_override
    key_user = str(api_key.user_id) if api_key and api_key.user_id else None
    api_key_id = api_key.id if api_key else None
    seen_in_batch: set[str] = set()
    # The key names the workspace; an imported batch under the master key is a
    # deployment-wide write and lands in the default one.
    usage_workspace_id = await resolve_workspace_id(db, api_key)

    rows: list[UsageLog] = []

    # Resolve users and pricing in bulk before the row loop, so ingestion issues a
    # fixed handful of queries regardless of batch size (no per-event N+1). Candidate
    # attribution targets are the batch default, any per-event override, and the key's
    # own user; pricing is loaded for every distinct (provider, model) at once.
    candidate_users = sorted(u for u in ({request.user_id, key_user} | {e.user_id for e in request.events}) if u)
    active_users: set[str] = set()
    # Chunked like the event-id and pricing lookups: a full batch with per-event
    # user_ids can carry more candidates than SQLite's bind-variable limit.
    for start in range(0, len(candidate_users), _IN_CHUNK):
        chunk = candidate_users[start : start + _IN_CHUNK]
        active_users.update(
            (
                await db.execute(select(User.user_id).where(User.user_id.in_(chunk), User.deleted_at.is_(None)))
            ).scalars().all()
        )
    # The organization comes off the workspace the key named, never off the
    # request: the organization decides what an event costs, so taking it from
    # something the importer controls would let one import at another
    # organization's negotiated rate.
    pricing_organization_id = await organization_for_workspace_id(db, usage_workspace_id)
    pricing_index = await _load_pricing_index(
        db, {(e.provider, e.model) for e in request.events}, pricing_organization_id
    )

    for index, event in enumerate(request.events):
        target_user, bind_error = _resolve_target_user(
            event.user_id,
            request.user_id,
            key_user,
            is_master=is_master_key,
            reject_mismatch=reject_mismatch,
        )
        if bind_error is not None or target_user is None:
            _reject(result, index, event.source_event_id, bind_error or "Could not attribute usage to a user.")
            continue
        if target_user not in active_users:
            _reject(
                result,
                index,
                event.source_event_id,
                f"user_id '{target_user}' not found. Create the user via POST /v1/users first.",
            )
            continue

        if event.source_event_id in seen_in_batch:
            result.duplicate += 1
            continue
        seen_in_batch.add(event.source_event_id)

        # Imported usage is always budget-exempt (counts_toward_budget=False), so
        # require_pricing does not apply: it is a budget-enforcement safety gate, and
        # there is no budget to protect here. A model with no configured price simply
        # records cost=null (the row still lands); add pricing later to see the cost.
        pricing = _resolve_pricing(pricing_index, event.provider, event.model, event.timestamp)
        rows.append(_build_row(request.source, target_user, event, pricing, api_key_id, usage_workspace_id))

    # Drop events already imported in a prior batch before inserting.
    if rows:
        existing = await _existing_event_ids(db, request.source, [r.source_event_id for r in rows if r.source_event_id])
        fresh = [r for r in rows if r.source_event_id not in existing]
        result.duplicate += len(rows) - len(fresh)
        await _insert_rows(db, request.source, fresh, result)

    logger.info(
        "external usage ingest: source=%s accepted=%d duplicate=%d rejected=%d",
        request.source,
        result.accepted,
        result.duplicate,
        result.rejected,
    )
    return result
