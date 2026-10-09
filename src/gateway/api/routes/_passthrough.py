"""Shared request scaffold for the pass-through provider routes.

The pass-through endpoints (audio, images, embeddings, moderations, rerank)
follow one scaffold: resolve the billed user, rate limit, resolve the provider
selector, reserve budget, call the provider, write a usage log, and reconcile
(success) or refund (failure) the reservation. :func:`run_passthrough` owns
that scaffold; each route supplies only its endpoint-specific pieces (budget
estimate, provider call, token extraction, cost computation) as callbacks.

Provider failures are classified with the same helper the hybrid fallback path
uses (``_classify_upstream_error``) and surface as HTTP 502: an upstream outage
is an upstream failure, not a gateway bug, matching the chat, messages, and
responses routes. The raw provider message is never included in the response
detail (it is preserved on the usage log's ``error_message``).

Every row this scaffold writes comes from one of two places: ``_usage_row`` for
the outcomes of an attempted provider call (success, provider error), and the
shared ``log_gateway_rejection`` for requests the gateway refused before ever
calling a provider, which the chat/messages/responses pipeline uses too so both
scaffolds record a rejection identically.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any, Generic, Protocol, TypeVar

from any_llm.exceptions import AnyLLMError
from fastapi import Depends, HTTPException, Request, Response, status
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import ContainerDep, get_config, get_db, get_log_writer
from gateway.api.routes._helpers import resolve_user_id
from gateway.api.routes._pipeline import (
    _elapsed_ms,
    _raise_for_unresolvable_model,
    failure_status_code,
    log_gateway_rejection,
    rate_limit_headers,
    throttle_early_rejection,
    unresolvable_model_detail,
)
from gateway.api.routes._platform import _classify_upstream_error
from gateway.core.config import GatewayConfig
from gateway.core.database import release_session
from gateway.core.error_codes import PROVIDER_NOT_CONFIGURED, error_headers
from gateway.core.metered_pricing import billable_usage, price_billable_usage, quantize_cost
from gateway.inflight import track_request
from gateway.log_config import logger
from gateway.model_labeling import relabel_model
from gateway.models.api_keys import APIKey
from gateway.models.pricing import ModelPricing
from gateway.models.usage import UsageLog
from gateway.ports.billing_port import BillingPort, InsufficientFundsError
from gateway.rate_limit import check_rate_limit
from gateway.schemas.inference import DecisionRequest, DecisionResponse
from gateway.services.budgets import (
    ZERO,
    BudgetScopeRequest,
    ReservationHandle,
    estimate_cost,
    estimate_tokens,
    reconcile_reservation,
    refund_reservation,
    reserve_budget,
)
from gateway.services.inference import (
    DecisionProvider,
    DecisionProviderError,
    UnknownDecisionProviderError,
    decision_body,
    reported_charge,
    request_decision,
    resolve_decision_provider,
)
from gateway.services.log_writer import LogWriter
from gateway.services.model_access import is_model_allowed, model_not_allowed_detail, resolve_request_allowlist
from gateway.services.pricing_service import (
    find_model_pricing,
    no_pricing_error_detail,
    pricing_required_but_missing,
)
from gateway.services.provider_kwargs import ResolvedProvider, missing_credential, resolve_provider_selector
from gateway.services.tenancy.org_provider_key_service import cached_org_model_restriction
from gateway.services.workspace_scope import organization_for_workspace_id, resolve_workspace_id

ResultT = TypeVar("ResultT")

PASSTHROUGH_PROVIDER_ERROR_DETAIL = "The request could not be completed by the provider"

# A route's non-token charge lines: the meters dict and the auditable breakdown,
# matching the shape ``price_billable_usage`` and ``price_tool_calls`` already
# write for the chat and tool-charge paths.
BillingMeters = tuple[dict[str, Any], list[dict[str, Any]]]


def resolve_passthrough_user_id(
    auth_result: tuple[APIKey | None, bool],
    user: str | None,
    *,
    reject_mismatch: bool,
) -> str:
    """Resolve the billed user with the standard pass-through error responses."""
    api_key, is_master_key = auth_result
    return resolve_user_id(
        user_id_from_request=user,
        api_key=api_key,
        is_master_key=is_master_key,
        master_key_error=HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="When using master key, 'user' field is required in request body",
        ),
        no_api_key_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key validation failed",
        ),
        no_user_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key has no associated user",
        ),
        forbidden_user_error=HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="'user' field does not match the authenticated API key's user",
        ),
        reject_mismatch=reject_mismatch,
    )


@dataclass
class PassthroughOutcome(Generic[ResultT]):
    """A successful pass-through provider call plus response metadata."""

    result: ResultT
    """The provider result, relabeled to the request alias when applicable."""
    resolved: ResolvedProvider
    """The resolved selector the call was dispatched against."""
    headers: dict[str, str]
    """Rate-limit headers for routes that build their own response object."""


async def run_passthrough(
    *,
    endpoint: str,
    raw_request: Request,
    response: Response | None,
    auth_result: tuple[APIKey | None, bool],
    db: AsyncSession,
    config: GatewayConfig,
    log_writer: LogWriter,
    model: str,
    user: str | None,
    call_provider: Callable[[ResolvedProvider], Awaitable[ResultT]],
    lookup_pricing: bool = True,
    pricing_use_defaults: bool = True,
    estimate: Callable[[ModelPricing | None], Decimal] | None = None,
    enforce_require_pricing: bool = False,
    usage_tokens: Callable[[ResultT], tuple[int | None, int | None, int | None]] | None = None,
    compute_cost: Callable[[ResultT, ModelPricing | None], Decimal | None] | None = None,
    compute_meters: Callable[[ResultT, ModelPricing | None, Decimal], BillingMeters | None] | None = None,
    map_provider_error: Callable[[Exception], HTTPException | None] | None = None,
    reserve_before_resolve: bool = False,
    relabel: bool = True,
) -> PassthroughOutcome[ResultT]:
    """Run the shared pass-through scaffold around a single provider call.

    Steps: resolve the billed user (honoring ``config.reject_user_mismatch``),
    rate limit, resolve the provider selector, look up pricing, reserve the
    estimated cost, invoke ``call_provider``, write the usage log, and
    reconcile (success) or refund (failure) the reservation.

    Args:
        endpoint: Path recorded on usage log rows (e.g. ``"/v1/embeddings"``).
        raw_request: Incoming request, used for rate limiting.
        response: When given, rate-limit headers are set on it. Routes that
            return their own response object pass ``None`` and read
            ``PassthroughOutcome.headers`` instead.
        auth_result: The ``verify_api_key_or_master_key`` dependency result.
        model: The raw request selector; used for the reservation and error
            text, while the resolved short name reaches the provider and logs.
        user: The request's ``user`` field, if any.
        call_provider: Awaits the provider call for the resolved selector. An
            ``HTTPException`` raised here (e.g. an upload size check) refunds
            the reservation and propagates unchanged.
        lookup_pricing: Whether to resolve :class:`ModelPricing` for the model.
            Audio resolves it for per-request charge lines but the reservation
            estimate stays 0 (no measurable cost unit yet, so no pre-call spend).
        pricing_use_defaults: Whether the pricing lookup may fall back to the
            genai-prices dataset. A route whose billable unit is not a token
            passes False for the reason :func:`find_model_pricing` documents:
            those rates are USD per million *tokens*, so a per-request route
            would charge them as USD per million *requests* and a per-image route
            as USD per image, writing a charge line at the wrong unit for a rate
            nobody configured.
        estimate: Maps the pricing row to the reservation estimate in USD.
            Defaults to zero, which still enforces per-user state (user exists,
            not blocked, not already over budget).
        enforce_require_pricing: When True and ``config.require_pricing`` is
            set, reject unpriced models with 402. The check runs after the
            reservation (so its 404/403 rejections take precedence) and the
            reservation is refunded before raising. Honored only on the
            resolve-first path: a ``reserve_before_resolve`` route resolves
            pricing after its reservation and skips this gate, so setting both
            silently serves an unpriced model. No route sets both today.
        usage_tokens: Maps the provider result to ``(prompt, completion,
            total)`` token counts for the usage log. Defaults to ``(0, 0, 0)``.
        compute_cost: Maps the result and pricing to the final USD cost, or
            ``None`` to leave the log's cost unset and reconcile at 0.0.
        compute_meters: Maps the result, pricing, and the cost ``compute_cost``
            just returned to this request's billing meters and charge lines, or
            ``None`` to leave both unset. Only called when ``compute_cost``
            returned a cost, so a route with no priced unit never needs to guard
            against a missing cost itself.
        map_provider_error: Route-specific provider-exception mapping checked
            before the generic 502 (the error log and refund happen either way).
        reserve_before_resolve: Preserve the audio routes' historical ordering,
            reserving budget before the selector is resolved. Routes that need
            pricing resolve first (the pricing key is the resolved instance).
        relabel: Rewrite the result's ``model`` field to the configured alias
            the caller used, so responses do not echo the aliased target.

    Returns:
        The provider result plus the resolved selector and rate-limit headers.
    """
    # Anchor request latency at the earliest point in the scaffold (monotonic,
    # so it is immune to wall-clock steps); recorded on the usage log below.
    started_at = time.monotonic()
    api_key, is_master_key = auth_result
    api_key_id = api_key.id if api_key else None
    # Zero I/O for a keyed request (see `_pipeline.resolve_request_context`'s
    # equivalent comment); only a master-key request pays a lookup. Fixed for
    # the whole request, so this also becomes the usage row's workspace below.
    workspace_id = await resolve_workspace_id(db, api_key)
    # Derived from the workspace already resolved above, not via
    # `organization_for_key_id` (which would re-derive the same workspace from
    # `api_key_id` internally): a master-key request's workspace lookup is
    # deliberately never memoized, so re-deriving it here would pay that cost
    # twice for one request. Decides both what the model costs and whether the
    # free-model shortcut applies.
    organization_id = await organization_for_workspace_id(db, workspace_id)
    # A key flagged exclude_from_budget logs cost but is never reserved or folded
    # into users.spend. Threaded through the reservation handle (so reconcile skips
    # the spend write) and stamped on the usage row.
    budget_exempt = api_key is not None and api_key.exclude_from_budget

    try:
        user_id = resolve_passthrough_user_id(auth_result, user, reject_mismatch=config.reject_user_mismatch)
    except HTTPException as exc:
        # Only the user/key mismatch (403) has a user to attribute the drop to;
        # see log_gateway_rejection for the rejections that deliberately do not
        # log. Like its counterpart in the pipeline, this row carries the raw
        # selector and no provider: nothing is resolved this early, and resolving
        # purely to shape a log row is not worth it on a refusal path.
        # Also like its counterpart, this gate precedes check_rate_limit, so the
        # write is charged to the key's own bucket and skipped once throttled
        # (see throttle_early_rejection). The response stays 403.
        if (
            exc.status_code == status.HTTP_403_FORBIDDEN
            and api_key is not None
            and not await throttle_early_rejection(raw_request, str(api_key.user_id))
        ):
            await log_gateway_rejection(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                user_id=api_key.user_id,
                model=model,
                provider=None,
                endpoint=endpoint,
                detail=str(exc.detail),
                status_code=exc.status_code,
                started_at=started_at,
            )
        raise

    rate_limit_info = await check_rate_limit(raw_request, user_id)

    async def _log_rejection(detail: str, *, row_model: str, row_provider: str | None, status_code: int) -> None:
        """Record a gateway-side rejection of this request.

        Call after refunding the reservation, if one is held; this only writes a
        row (see :func:`log_gateway_rejection`) and never touches the budget.
        ``status_code`` is the status this rejection is about to return, which the
        row keeps so the failure taxonomy can tell a refusal from a provider fault.
        """
        await log_gateway_rejection(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            user_id=user_id,
            model=row_model,
            provider=row_provider,
            endpoint=endpoint,
            detail=detail,
            status_code=status_code,
            started_at=started_at,
        )

    async def _reserve(estimated_cost: Decimal, *, row_model: str, row_provider: str | None) -> ReservationHandle:
        """Reserve the estimate, recording a blocked/over-budget refusal.

        ``reserve_budget`` reserves nothing on the paths that raise, so there is
        nothing to refund here. The 404 for an unknown user is left unlogged:
        ``usage_logs.user_id`` is a foreign key to ``users``, so a row naming a
        user that does not exist could not be inserted.
        """
        try:
            return await reserve_budget(
                db,
                user_id,
                estimated_cost,
                model=model,
                strategy=config.budget_strategy,
                counts_toward_budget=not budget_exempt,
                scope=BudgetScopeRequest(api_key=api_key, provider_instance=row_provider),
                organization_id=organization_id,
            )
        except HTTPException as exc:
            if exc.status_code != status.HTTP_404_NOT_FOUND:
                await _log_rejection(
                    str(exc.detail),
                    row_model=row_model,
                    row_provider=row_provider,
                    status_code=exc.status_code,
                )
            raise

    pricing: ModelPricing | None = None
    if reserve_before_resolve:
        # Nothing is resolved yet, so a rejection here records the requested
        # selector with no provider.
        reservation = await _reserve(
            estimate(None) if estimate else ZERO,
            row_model=model,
            row_provider=None,
        )
        # The reservation is already held, so refund it before mapping an
        # unresolvable selector to 400; otherwise the estimate leaks.
        try:
            resolved = resolve_provider_selector(config, model, user_id, workspace_id=workspace_id)
        except (ValueError, AnyLLMError) as exc:
            await refund_reservation(db, reservation)
            await _log_rejection(
                unresolvable_model_detail(model),
                row_model=model,
                row_provider=None,
                status_code=status.HTTP_400_BAD_REQUEST,
            )
            _raise_for_unresolvable_model(model, exc)
        if lookup_pricing:
            # Unlike the branch below, the reservation is already held here, so a
            # failed lookup must refund before propagating or the estimate leaks.
            try:
                pricing = await find_model_pricing(
                    db,
                    resolved.instance,
                    resolved.model,
                    use_defaults=pricing_use_defaults,
                    organization_id=organization_id,
                )
            except Exception:
                # The realistic failure is a DB error, which leaves the session
                # needing a rollback: without one the refund's own UPDATE raises
                # PendingRollbackError, masking this exception and leaking the
                # hold this block exists to release. ``reserve_budget`` already
                # committed, so the rollback discards nothing of its own.
                await db.rollback()
                await refund_reservation(db, reservation)
                raise
    else:
        try:
            resolved = resolve_provider_selector(config, model, user_id, workspace_id=workspace_id)
        except (ValueError, AnyLLMError) as exc:
            # Nothing is reserved yet on this branch, so there is no refund to do.
            await _log_rejection(
                unresolvable_model_detail(model),
                row_model=model,
                row_provider=None,
                status_code=status.HTTP_400_BAD_REQUEST,
            )
            _raise_for_unresolvable_model(model, exc)
        if lookup_pricing:
            pricing = await find_model_pricing(
                db,
                resolved.instance,
                resolved.model,
                use_defaults=pricing_use_defaults,
                organization_id=organization_id,
            )
        # Reserve first so user/blocked/budget rejections (404/403) precede the
        # missing-pricing rejection (402); refund if we then reject for no pricing.
        reservation = await _reserve(
            estimate(pricing) if estimate else ZERO,
            row_model=resolved.model,
            row_provider=resolved.instance,
        )
        # A budget-exempt key is never debited, so the require_pricing safety gate
        # does not apply: the call proceeds and logs cost=null when unpriced.
        if (
            enforce_require_pricing
            and not budget_exempt
            and pricing_required_but_missing(pricing, require_pricing=config.require_pricing)
        ):
            await refund_reservation(db, reservation)
            no_pricing_detail = no_pricing_error_detail(model)
            # Record the rejection so dropped traffic is visible in the activity
            # log and countable as an error, rather than only reaching the
            # operator as a user complaint. cost stays null: nothing was spent.
            await _log_rejection(
                no_pricing_detail,
                row_model=resolved.model,
                row_provider=resolved.instance,
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
            )
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=no_pricing_detail,
            )

    # Model access control (per-key). The reservation is already taken above (the
    # audio branch reserves before resolve), so refund before rejecting. A key with
    # no list of its own inherits its user's default.
    key_allowlist = await resolve_request_allowlist(db, api_key)
    if key_allowlist is not None and not is_model_allowed(key_allowlist, f"{resolved.instance}:{resolved.model}"):
        await refund_reservation(db, reservation)
        not_allowed_detail = model_not_allowed_detail(model)
        await _log_rejection(
            not_allowed_detail,
            row_model=resolved.model,
            row_provider=resolved.instance,
            status_code=status.HTTP_403_FORBIDDEN,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=not_allowed_detail,
        )

    # Organization-scoped model restriction (otari#643): mirrors `_pipeline.py`'s
    # equivalent gate. Applies only when the selector did not name a
    # configured instance, the same condition `provider_kwargs.get_provider_kwargs`
    # uses to decide whether to consult the organization overlay at all.
    if workspace_id is not None and resolved.instance not in config.providers:
        org_allowlist = cached_org_model_restriction(workspace_id, resolved.provider.value)
        if org_allowlist is not None and resolved.model not in org_allowlist:
            await refund_reservation(db, reservation)
            not_allowed_detail = model_not_allowed_detail(model)
            await _log_rejection(
                not_allowed_detail,
                row_model=resolved.model,
                row_provider=resolved.instance,
                status_code=status.HTTP_403_FORBIDDEN,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=not_allowed_detail,
            )

    # Already resolved above (needed there for organization-scoped provider
    # credentials); reused here rather than re-derived, since it is fixed for
    # the whole request either way.
    usage_workspace_id = workspace_id

    def _usage_row(row_status: str, **outcome: Any) -> UsageLog:
        """Build this request's usage row, varying only the outcome columns.

        The identity and attribution columns are identical for every outcome, so
        they live here once: a new column is added in one place rather than at
        each of the call sites below.
        """
        return UsageLog(
            id=str(uuid.uuid4()),
            workspace_id=usage_workspace_id,
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=resolved.model,
            provider=resolved.instance,
            endpoint=endpoint,
            status=row_status,
            latency_ms=_elapsed_ms(started_at),
            counts_toward_budget=not budget_exempt,
            **outcome,
        )

    # Every gate has passed and the provider is about to be called, so the request
    # is genuinely in flight from here until its response has been sent. Registered
    # for the same reason as on the chat/messages/responses path, and it matters as
    # much: an image generation routinely runs longer than a completion, and until
    # it settles the activity log has nothing to show for it. The entry is dropped
    # by InFlightMiddleware, not here.
    #
    # The upstream call starts here, and these endpoints held their connection
    # across it exactly as the completion routes did: the allow-list read above
    # is the last statement on the request session and nothing commits it. An
    # image generation or a long transcription outlasts a completion, so this
    # path reaches the pool ceiling sooner rather than later.
    await release_session(db)

    track_request(
        raw_request,
        endpoint=endpoint,
        # The same pair `_usage_row` stamps, so the row does not appear to change
        # model when it settles.
        model=resolved.model,
        provider=resolved.instance,
        user_id=user_id,
        api_key_id=api_key_id,
    )

    try:
        result = await call_provider(resolved)

        prompt_tokens, completion_tokens, total_tokens = usage_tokens(result) if usage_tokens else (0, 0, 0)
        usage_log = _usage_row(
            "success",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        # Rounded here, once, so the row and the spend it reconciles settle at the
        # same amount: the column would round the row on its own, and the ledger
        # would then hold a fraction of a micro-dollar the row does not show.
        raw_cost = compute_cost(result, pricing) if compute_cost else None
        cost = quantize_cost(raw_cost) if raw_cost is not None else None
        if cost is not None and raw_cost is not None:
            usage_log.cost = cost
            # The charge lines get the unrounded amount, the same way the token
            # path's do: the column is the amount, the lines explain it.
            billing = compute_meters(result, pricing, raw_cost) if compute_meters else None
            if billing is not None:
                usage_log.billing_meters, usage_log.pricing_breakdown = billing

        await log_writer.put(usage_log)
        # The measured total, so a token ceiling counts this endpoint's traffic.
        # These paths hold no token estimate at admission (the estimate callback is
        # priced in dollars), so a token cap refuses them once it is exhausted
        # rather than reserving headroom for each one.
        await reconcile_reservation(
            db,
            reservation,
            cost if cost is not None else 0.0,
            actual_tokens=max(total_tokens or 0, 0),
        )

    except HTTPException:
        await refund_reservation(db, reservation)
        raise
    except Exception as e:
        await log_writer.put(_usage_row("error", error_message=str(e), status_code=failure_status_code(e)))
        await refund_reservation(db, reservation)

        missing = missing_credential(e)
        if missing is not None:
            # Nothing reached the provider: the deployment holds no credential
            # for it. Answered before any route-specific mapping, and as a 424
            # rather than a 502, because a retrying SDK cannot supply a key.
            logger.warning("No credential configured for %s:%s", resolved.provider, resolved.model)
            raise HTTPException(
                status_code=status.HTTP_424_FAILED_DEPENDENCY,
                detail=missing.detail,
                headers=error_headers(PROVIDER_NOT_CONFIGURED),
            ) from e

        mapped = map_provider_error(e) if map_provider_error else None
        if mapped is not None:
            raise mapped from e

        _, error_class = _classify_upstream_error(e)
        logger.error("Provider call failed for %s:%s (%s): %s", resolved.provider, resolved.model, error_class, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=PASSTHROUGH_PROVIDER_ERROR_DETAIL,
        ) from e

    headers = rate_limit_headers(rate_limit_info) if rate_limit_info else {}
    if response is not None:
        for key, value in headers.items():
            response.headers[key] = value

    if relabel and resolved.alias is not None:
        relabel_model(result, resolved.alias)

    return PassthroughOutcome(result=result, resolved=resolved, headers=headers)


DECISIONS_ENDPOINT = "/v1/decisions"
"""The usage-log label for every decisions path. See chat.USAGE_ENDPOINT."""

# An answer is a handful of tokens (one label, or a number), so the reservation
# holds this many output tokens per question rather than a completion's default.
ESTIMATED_OUTPUT_TOKENS_PER_QUESTION = 32
# An image costs the model a fixed token budget, not its base64 length, so the
# estimate counts each one as this many characters of prompt (about 1024 tokens).
_ESTIMATED_CHARS_PER_IMAGE = 4096

_DECISION_INVALID_DETAIL = "The provider rejected the request as invalid"
_DECISION_RATE_LIMITED_DETAIL = "The provider is rate limiting requests; retry with backoff"
_DECISION_UNSUPPORTED_DETAIL = (
    "The model cannot answer this request; it may not be a decision model or may not accept images"
)
_DECISION_NO_ORGANIZATION_DETAIL = "This request belongs to no organization the call could be billed to"


@dataclass(frozen=True)
class DecisionQuote:
    """What a decision call knows before it is made.

    ``provider`` and ``model`` are the label the call is priced, allow-listed and
    recorded under. ``charge`` is the price of one call where whoever answers
    owns the price: the scaffold holds and charges it, through the budgets and
    the billing port, and looks up no rate. ``None`` leaves the estimate to the
    deployment's pricing row, which is the open-source case, where the caller
    pays whatever the upstream charged.
    """

    provider: str
    model: str
    charge: Decimal | None = None


@dataclass(frozen=True)
class DecisionOutcome(Generic[ResultT]):
    """A decision call that returned, and what it consumed."""

    result: ResultT
    input_tokens: int
    output_tokens: int
    charge: Decimal | None = None
    """The final price where the call names one. ``None`` prices the tokens, or settles the quote."""


class DecisionUnavailableError(Exception):
    """The call resolved to nothing to dispatch to. The message is the caller's detail."""


class DecisionCallError(Exception):
    """The call failed. ``logged_status`` goes on the usage row and ``response`` to the caller."""

    def __init__(self, message: str, *, logged_status: int, response: HTTPException) -> None:
        super().__init__(message)
        self.logged_status = logged_status
        self.response = response


class DecisionCall(Protocol[ResultT]):
    """One decision call, split where the scaffold steps in between.

    :meth:`resolve` runs before anything is held, so a call with nothing to
    dispatch to is refused without a reservation. :meth:`dispatch` runs once
    the budget is held and the request is in flight.
    """

    def resolve(self) -> DecisionQuote:
        """Name what the call is metered under.

        Raises:
            DecisionUnavailableError: there is nothing to dispatch to.

        """
        ...

    async def dispatch(self) -> DecisionOutcome[ResultT]:
        """Make the call.

        Raises:
            DecisionCallError: the call failed, with what to record and what to answer.

        """
        ...


@dataclass(frozen=True)
class _Funding:
    """A deployment-paid call's hold on the billing port, and whose funds it is on."""

    organization_id: uuid.UUID
    hold: Decimal


def decision_status_error(status_code: int | None) -> HTTPException:
    """Return the caller-facing status for an upstream failure, with no upstream text."""
    if status_code in (status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_CONTENT):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_DECISION_INVALID_DETAIL)
    # llama-server's answer to a model that is not a decision model, or to images it cannot read.
    if status_code == status.HTTP_501_NOT_IMPLEMENTED:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_DECISION_UNSUPPORTED_DETAIL)
    if status_code == status.HTTP_429_TOO_MANY_REQUESTS:
        return HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=_DECISION_RATE_LIMITED_DETAIL)
    return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=PASSTHROUGH_PROVIDER_ERROR_DETAIL)


def decision_input_chars(request: DecisionRequest) -> int:
    """How much of ``request`` reaches the decision model, in the characters the budget estimate counts.

    The state and the questions as they are sent, plus a fixed allowance per image.
    """
    body = decision_body(request, request.model)
    return (
        len(json.dumps(body["state"]))
        + len(json.dumps(body["questions"]))
        + _ESTIMATED_CHARS_PER_IMAGE * len(request.images or ())
    )


class SelectorDecision:
    """The call behind a decisions request: the selector names a ``decision_providers`` entry, asked as is."""

    def __init__(self, config: GatewayConfig, request: DecisionRequest) -> None:
        self._config = config
        self._request = request
        self.prompt_chars = decision_input_chars(request)
        self.default_output_tokens = ESTIMATED_OUTPUT_TOKENS_PER_QUESTION * len(request.questions)
        self._provider: DecisionProvider | None = None
        self._model = ""

    def resolve(self) -> DecisionQuote:
        try:
            self._provider, self._model = resolve_decision_provider(self._config, self._request.model)
        except UnknownDecisionProviderError as exc:
            raise DecisionUnavailableError(str(exc)) from exc
        return DecisionQuote(provider=self._provider.name, model=self._model)

    async def dispatch(self) -> DecisionOutcome[DecisionResponse]:
        if self._provider is None:
            msg = "dispatch() before resolve()"
            raise RuntimeError(msg)
        body = decision_body(self._request, self._model)
        try:
            result = DecisionResponse.model_validate(await request_decision(self._provider, body))
        except DecisionProviderError as exc:
            raise DecisionCallError(
                str(exc), logged_status=failure_status_code(exc), response=decision_status_error(exc.status_code)
            ) from exc
        except ValidationError as exc:
            # The error's own text quotes the upstream body, so only its size is kept.
            detail = f"{self._provider.provider} decisions returned an answer with {exc.error_count()} invalid field(s)"
            unreadable = HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=PASSTHROUGH_PROVIDER_ERROR_DETAIL
            )
            raise DecisionCallError(detail, logged_status=status.HTTP_502_BAD_GATEWAY, response=unreadable) from exc
        usage = result.usage
        return DecisionOutcome(
            result=result,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            charge=reported_charge(result),
        )


async def run_decision(
    *,
    raw_request: Request,
    response: Response,
    auth_result: tuple[APIKey | None, bool],
    db: AsyncSession,
    config: GatewayConfig,
    log_writer: LogWriter,
    billing: BillingPort,
    selector: str,
    user: str | None,
    prompt_chars: int,
    default_output_tokens: int,
    call: DecisionCall[ResultT],
) -> ResultT:
    """Run the decisions scaffold around ``call``: resolve, gate, reserve, call, log, settle.

    The same steps as :func:`run_passthrough`, which cannot run them: it resolves the model
    against the any-llm provider instances, and a decision is answered by whatever ``call``
    resolves to. Billing is per token, priced from the ``<provider>:<model>`` rate when one
    is configured, otherwise from the charge the call reports, which only OpenRouter does.
    A call whose quote names a price of its own is deployment-paid instead: that price is
    held and charged through ``billing`` as well as the budgets, and no rate is looked up.
    """
    started_at = time.monotonic()
    api_key, _ = auth_result
    api_key_id = api_key.id if api_key else None
    budget_exempt = api_key is not None and api_key.exclude_from_budget
    workspace_id = await resolve_workspace_id(db, api_key)
    organization_id = await organization_for_workspace_id(db, workspace_id)

    try:
        user_id = resolve_passthrough_user_id(auth_result, user, reject_mismatch=config.reject_user_mismatch)
    except HTTPException as exc:
        # As in run_passthrough: only the mismatch has a user to attribute the refusal to.
        if (
            exc.status_code == status.HTTP_403_FORBIDDEN
            and api_key is not None
            and not await throttle_early_rejection(raw_request, str(api_key.user_id))
        ):
            await log_gateway_rejection(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                user_id=api_key.user_id,
                model=selector,
                provider=None,
                endpoint=DECISIONS_ENDPOINT,
                detail=str(exc.detail),
                status_code=exc.status_code,
                started_at=started_at,
            )
        raise

    rate_limit_info = await check_rate_limit(raw_request, user_id)

    async def log_rejection(detail: str, *, row_model: str, row_provider: str | None, status_code: int) -> None:
        await log_gateway_rejection(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            user_id=user_id,
            model=row_model,
            provider=row_provider,
            endpoint=DECISIONS_ENDPOINT,
            detail=detail,
            status_code=status_code,
            started_at=started_at,
        )

    try:
        quote = call.resolve()
    except DecisionUnavailableError as exc:
        await log_rejection(str(exc), row_model=selector, row_provider=None, status_code=400)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    provider_name, model = quote.provider, quote.model

    key_allowlist = await resolve_request_allowlist(db, api_key)
    if key_allowlist is not None and not is_model_allowed(key_allowlist, f"{provider_name}:{model}"):
        not_allowed_detail = model_not_allowed_detail(selector)
        await log_rejection(not_allowed_detail, row_model=model, row_provider=provider_name, status_code=403)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=not_allowed_detail)

    pricing: ModelPricing | None = None
    if quote.charge is None:
        # A decisions model is not in the community pricing dataset, which could
        # only produce a false match on the bare model name.
        pricing = await find_model_pricing(
            db, provider_name, model, use_defaults=False, organization_id=organization_id
        )
    elif organization_id is None:
        # Deployment-paid, and nobody to bill: refused before anything is held.
        no_organization_detail = _DECISION_NO_ORGANIZATION_DETAIL
        await log_rejection(no_organization_detail, row_model=model, row_provider=provider_name, status_code=402)
        raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail=no_organization_detail)
    try:
        reservation = await reserve_budget(
            db,
            user_id,
            quote.charge
            if quote.charge is not None
            else estimate_cost(
                pricing,
                prompt_chars=prompt_chars,
                max_output_tokens=None,
                default_output_tokens=default_output_tokens,
            ),
            estimated_tokens=estimate_tokens(
                prompt_chars=prompt_chars,
                max_output_tokens=None,
                default_output_tokens=default_output_tokens,
            ),
            # Not the selector: ``model`` only drives reserve_budget's free-model
            # shortcut, which splits it through any-llm (see search.py).
            model=None,
            strategy=config.budget_strategy,
            counts_toward_budget=not budget_exempt,
            scope=BudgetScopeRequest(api_key=api_key, provider_instance=provider_name),
            organization_id=organization_id,
        )
    except HTTPException as exc:
        # As in run_passthrough's _reserve: an unknown user's 404 cannot satisfy usage_logs.user_id's foreign key.
        if exc.status_code != status.HTTP_404_NOT_FOUND:
            await log_rejection(
                str(exc.detail), row_model=model, row_provider=provider_name, status_code=exc.status_code
            )
        raise
    if (
        quote.charge is None
        and not budget_exempt
        and pricing_required_but_missing(pricing, require_pricing=config.require_pricing)
    ):
        await refund_reservation(db, reservation)
        no_pricing_detail = no_pricing_error_detail(selector)
        await log_rejection(no_pricing_detail, row_model=model, row_provider=provider_name, status_code=402)
        raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail=no_pricing_detail)

    funding: _Funding | None = None
    if quote.charge is not None and organization_id is not None:
        funding = _Funding(organization_id=organization_id, hold=quote.charge)
        try:
            await billing.hold(organization_id=funding.organization_id, amount=funding.hold)
        except InsufficientFundsError as exc:
            await refund_reservation(db, reservation)
            await log_rejection(str(exc), row_model=model, row_provider=provider_name, status_code=402)
            raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail=str(exc)) from exc

    async def undo() -> None:
        """Give back what the request holds: the billing hold, then the budget reservation.

        The session is rolled back first. The port commits nothing, so whatever
        it wrote before the failure (a release that raised midway, a charge the
        request will not keep) goes with the rollback, and the release here
        starts from the hold as it was committed and lands in the refund's
        commit. The refund runs even when the release fails, and a hold still
        claimed after that is logged for releasing by hand.
        """
        await db.rollback()
        try:
            if funding is not None:
                try:
                    await billing.release_hold(organization_id=funding.organization_id, amount=funding.hold)
                except BaseException:
                    logger.exception(
                        "The hold of %s on organization %s is still claimed; release it by hand",
                        funding.hold,
                        funding.organization_id,
                    )
                    await db.rollback()
                    raise
        finally:
            await refund_reservation(db, reservation)

    def usage_row(**outcome: Any) -> UsageLog:
        return UsageLog(
            id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model,
            provider=provider_name,
            endpoint=DECISIONS_ENDPOINT,
            latency_ms=_elapsed_ms(started_at),
            counts_toward_budget=not budget_exempt,
            **outcome,
        )

    # As in run_passthrough: nothing else runs on the session before settlement.
    await release_session(db)
    track_request(
        raw_request,
        endpoint=DECISIONS_ENDPOINT,
        model=model,
        provider=provider_name,
        user_id=user_id,
        api_key_id=api_key_id,
    )

    try:
        outcome = await call.dispatch()
        row = usage_row(
            status="success",
            prompt_tokens=outcome.input_tokens,
            completion_tokens=outcome.output_tokens,
            total_tokens=outcome.input_tokens + outcome.output_tokens,
        )
        cost: Decimal | None = None
        if pricing is not None:
            cost, row.billing_meters, row.pricing_breakdown = price_billable_usage(
                pricing,
                billable_usage(
                    input_tokens=outcome.input_tokens, output_tokens=outcome.output_tokens, cache_tokens_included=True
                ),
            )
        elif outcome.charge is not None:
            cost = outcome.charge
        elif quote.charge is not None:
            cost = quote.charge
        row.cost = cost
        if funding is not None:
            # The port bounds a charge by its hold, so a figure past the quote is charged as the quote.
            if cost is not None and cost > funding.hold:
                logger.warning(
                    "%s:%s reported a charge of %s above its quote of %s", provider_name, model, cost, funding.hold
                )
            cost = funding.hold if cost is None else min(cost, funding.hold)
            row.cost = cost
            # The hold ends before the charge, and both before the budget settles,
            # whose commit is what lands them: the request session is closed
            # without one. A failure at any step leaves the earlier steps
            # uncommitted for ``undo`` to roll back, and the reservation active
            # for its refund, where a charge that failed after settlement would
            # find nothing to undo and go uncharged.
            await billing.release_hold(organization_id=funding.organization_id, amount=funding.hold)
            try:
                await billing.charge(
                    organization_id=funding.organization_id,
                    amount=cost,
                    description=f"{provider_name}:{model}",
                    api_key_id=api_key_id,
                )
            except Exception:
                logger.exception(
                    "Charge for %s:%s failed after the call was served; it goes unanswered", provider_name, model
                )
                raise
        await log_writer.put(row)
        await reconcile_reservation(
            db,
            reservation,
            cost if cost is not None else 0.0,
            actual_tokens=outcome.input_tokens + outcome.output_tokens,
        )
    except DecisionCallError as exc:
        await log_writer.put(usage_row(status="error", error_message=str(exc), status_code=exc.logged_status))
        await undo()
        logger.error("Decision failed for %s:%s: %s", provider_name, model, exc)
        raise exc.response from exc
    except BaseException:
        # Cancellation, a failed release, charge or settlement, or an unexpected error.
        await undo()
        raise

    if rate_limit_info:
        for header, value in rate_limit_headers(rate_limit_info).items():
            response.headers[header] = value
    return outcome.result


@dataclass(frozen=True)
class DecisionScaffold:
    """:func:`run_decision`, bound to the request's session, so the route never names one."""

    db: AsyncSession
    config: GatewayConfig
    log_writer: LogWriter
    billing: BillingPort

    async def run(
        self,
        *,
        raw_request: Request,
        response: Response,
        request: DecisionRequest,
        auth_result: tuple[APIKey | None, bool],
    ) -> DecisionResponse:
        """Run the scaffold for one decisions request, against the provider its selector names."""
        call = SelectorDecision(self.config, request)
        return await self.run_call(
            raw_request=raw_request,
            response=response,
            auth_result=auth_result,
            selector=request.model,
            user=request.user,
            prompt_chars=call.prompt_chars,
            default_output_tokens=call.default_output_tokens,
            call=call,
        )

    async def run_call(
        self,
        *,
        raw_request: Request,
        response: Response,
        auth_result: tuple[APIKey | None, bool],
        selector: str,
        user: str | None,
        prompt_chars: int,
        default_output_tokens: int,
        call: DecisionCall[ResultT],
    ) -> ResultT:
        """Run the scaffold around ``call``, with ``selector`` as the name a refusal reports."""
        return await run_decision(
            raw_request=raw_request,
            response=response,
            auth_result=auth_result,
            db=self.db,
            config=self.config,
            log_writer=self.log_writer,
            billing=self.billing,
            selector=selector,
            user=user,
            prompt_chars=prompt_chars,
            default_output_tokens=default_output_tokens,
            call=call,
        )


def get_decision_scaffold(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
    container: ContainerDep,
) -> DecisionScaffold:
    """Build the decisions scaffold on the request's session.

    The billing port is resolved on that same session rather than through
    ``get_billing_port``, which would open a second one: a deployment-paid
    call's wallet entries have to land in the transaction its usage row and
    budget settlement do.
    """
    return DecisionScaffold(db=db, config=config, log_writer=log_writer, billing=container.resolve(BillingPort, db))
