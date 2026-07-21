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
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Generic, TypeVar

from any_llm.exceptions import AnyLLMError
from fastapi import HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes._helpers import resolve_user_id
from gateway.api.routes._pipeline import _elapsed_ms, _raise_for_unresolvable_model, rate_limit_headers
from gateway.api.routes._platform import _classify_upstream_error
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.model_labeling import relabel_model
from gateway.models.entities import APIKey, ModelPricing, UsageLog
from gateway.rate_limit import check_rate_limit
from gateway.services.budget_service import (
    reconcile_reservation,
    refund_reservation,
    reserve_budget,
)
from gateway.services.log_writer import LogWriter
from gateway.services.model_access import is_model_allowed, model_not_allowed_detail, resolve_request_allowlist
from gateway.services.pricing_service import (
    find_model_pricing,
    no_pricing_error_detail,
    pricing_required_but_missing,
)
from gateway.services.provider_kwargs import ResolvedProvider, resolve_provider_selector

ResultT = TypeVar("ResultT")

PASSTHROUGH_PROVIDER_ERROR_DETAIL = "The request could not be completed by the provider"


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
    estimate: Callable[[ModelPricing | None], float] | None = None,
    enforce_require_pricing: bool = False,
    usage_tokens: Callable[[ResultT], tuple[int | None, int | None, int | None]] | None = None,
    compute_cost: Callable[[ResultT, ModelPricing | None], float | None] | None = None,
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
            Audio has no measurable cost unit yet and skips the lookup.
        estimate: Maps the pricing row to the reservation estimate in USD.
            Defaults to 0.0, which still enforces per-user state (user exists,
            not blocked, not already over budget).
        enforce_require_pricing: When True and ``config.require_pricing`` is
            set, reject unpriced models with 402. The check runs after the
            reservation (so its 404/403 rejections take precedence) and the
            reservation is refunded before raising.
        usage_tokens: Maps the provider result to ``(prompt, completion,
            total)`` token counts for the usage log. Defaults to ``(0, 0, 0)``.
        compute_cost: Maps the result and pricing to the final USD cost, or
            ``None`` to leave the log's cost unset and reconcile at 0.0.
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

    user_id = resolve_passthrough_user_id(auth_result, user, reject_mismatch=config.reject_user_mismatch)

    rate_limit_info = check_rate_limit(raw_request, user_id)

    pricing: ModelPricing | None = None
    if reserve_before_resolve:
        reservation = await reserve_budget(
            db, user_id, estimate(None) if estimate else 0.0, model=model, strategy=config.budget_strategy
        )
        # The reservation is already held, so refund it before mapping an
        # unresolvable selector to 400; otherwise the estimate leaks.
        try:
            resolved = resolve_provider_selector(config, model)
        except (ValueError, AnyLLMError) as exc:
            await refund_reservation(db, reservation)
            _raise_for_unresolvable_model(model, exc)
    else:
        try:
            resolved = resolve_provider_selector(config, model)
        except (ValueError, AnyLLMError) as exc:
            _raise_for_unresolvable_model(model, exc)
        if lookup_pricing:
            pricing = await find_model_pricing(db, resolved.instance, resolved.model)
        # Reserve first so user/blocked/budget rejections (404/403) precede the
        # missing-pricing rejection (402); refund if we then reject for no pricing.
        reservation = await reserve_budget(
            db, user_id, estimate(pricing) if estimate else 0.0, model=model, strategy=config.budget_strategy
        )
        if enforce_require_pricing and pricing_required_but_missing(pricing, require_pricing=config.require_pricing):
            await refund_reservation(db, reservation)
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=no_pricing_error_detail(model),
            )

    # Model access control (per-key). The reservation is already taken above (the
    # audio branch reserves before resolve), so refund before rejecting. A key with
    # no list of its own inherits its user's default.
    key_allowlist = await resolve_request_allowlist(db, api_key)
    if key_allowlist is not None and not is_model_allowed(
        key_allowlist, f"{resolved.instance}:{resolved.model}"
    ):
        await refund_reservation(db, reservation)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=model_not_allowed_detail(model),
        )

    try:
        result = await call_provider(resolved)

        prompt_tokens, completion_tokens, total_tokens = usage_tokens(result) if usage_tokens else (0, 0, 0)
        usage_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=resolved.model,
            provider=resolved.instance,
            endpoint=endpoint,
            status="success",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=_elapsed_ms(started_at),
        )

        cost = compute_cost(result, pricing) if compute_cost else None
        if cost is not None:
            usage_log.cost = cost

        await log_writer.put(usage_log)
        await reconcile_reservation(db, reservation, cost if cost is not None else 0.0)

    except HTTPException:
        await refund_reservation(db, reservation)
        raise
    except Exception as e:
        error_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=resolved.model,
            provider=resolved.instance,
            endpoint=endpoint,
            status="error",
            error_message=str(e),
            latency_ms=_elapsed_ms(started_at),
        )
        await log_writer.put(error_log)
        await refund_reservation(db, reservation)

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
