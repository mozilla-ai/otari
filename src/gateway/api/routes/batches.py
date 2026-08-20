"""Batch API endpoints for asynchronous LLM processing."""

import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any

from any_llm import AnyLLM, LLMProvider
from any_llm.api import acancel_batch, acreate_batch, alist_batches, aretrieve_batch, aretrieve_batch_results
from any_llm.exceptions import AnyLLMError, BatchNotCompleteError, UnsupportedProviderError
from any_llm.types.batch import Batch
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._helpers import resolve_user_id
from gateway.api.routes._pipeline import _raise_for_unresolvable_model, failure_status_code
from gateway.api.routes.chat import rate_limit_headers
from gateway.core.config import GatewayConfig
from gateway.core.metered_pricing import calculate_token_cost
from gateway.log_config import logger
from gateway.models.entities import APIKey, BatchRecord, UsageLog
from gateway.rate_limit import check_rate_limit
from gateway.services.batch_service import (
    claim_batch_accounting,
    get_batch_record,
    get_batch_records,
    record_batch,
)
from gateway.services.budget_service import (
    reconcile_reservation,
    record_external_spend,
    refund_reservation,
    reserve_budget,
)
from gateway.services.log_writer import LogWriter
from gateway.services.model_access import is_model_allowed, model_not_allowed_detail, resolve_request_allowlist
from gateway.services.pricing_service import find_model_pricing
from gateway.services.provider_kwargs import get_provider_kwargs, resolve_provider_selector
from gateway.services.scoped_budget_service import BudgetScopeRequest
from gateway.services.tenancy.org_provider_key_service import cached_org_model_restriction
from gateway.services.workspace_scope import (
    organization_for_workspace_id,
    resolve_workspace_id,
    workspace_for_key_id,
)

router = APIRouter(prefix="/v1/batches", tags=["batches"])

# Metadata key stamped on provider batches at creation time so ownership can be
# checked on retrieve/cancel/results. The gateway stores no batch table, so the
# provider-side metadata is the ownership anchor.
_OWNER_METADATA_KEY = "otari_user_id"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class BatchRequestItem(BaseModel):
    custom_id: str
    body: dict[str, Any]


class CreateBatchRequest(BaseModel):
    model: str
    requests: list[BatchRequestItem] = Field(min_length=1, max_length=10_000)
    completion_window: str = "24h"
    metadata: dict[str, str] | None = None
    user: str | None = None


# ---------------------------------------------------------------------------
# Usage logging helper
# ---------------------------------------------------------------------------


async def log_batch_usage(
    db: AsyncSession,
    log_writer: LogWriter,
    api_key_id: str | None,
    model: str,
    provider: str,
    endpoint: str,
    user_id: str | None = None,
    error: str | None = None,
    status_code: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cost: Decimal | None = None,
    counts_toward_budget: bool = True,
    workspace_id: uuid.UUID | None = None,
) -> None:
    """Log batch API usage, including token counts and cost when derivable.

    ``workspace_id`` overrides the workspace derived from ``api_key_id``. A batch
    is retrieved by a key that need not be the one that created it, and the
    accounting scopes follow the creator: the row is attributed to the creator's
    user, priced at the creator's organization's rates, and so must land in the
    creator's workspace, or per-workspace usage stops reconciling against the
    rate list it was billed under. ``api_key_id`` stays the key that actually
    made the request, because that is what it records.
    """
    # latency_ms is intentionally left NULL: a batch is an asynchronous job that
    # is submitted and later polled, so there is no synchronous request duration
    # to record (unlike the chat/passthrough paths).
    usage_log = UsageLog(
        id=str(uuid.uuid4()),
        workspace_id=workspace_id if workspace_id is not None else await workspace_for_key_id(db, api_key_id),
        api_key_id=api_key_id,
        user_id=user_id,
        timestamp=datetime.now(UTC),
        model=model,
        provider=provider,
        endpoint=endpoint,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost=cost,
        status="success" if error is None else "error",
        error_message=error,
        status_code=status_code,
        counts_toward_budget=counts_toward_budget,
    )
    await log_writer.put(usage_log)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_provider(provider: str) -> LLMProvider:
    """Parse a provider string into an LLMProvider enum, raising 400 on invalid values."""
    try:
        return LLMProvider.from_string(provider)
    except UnsupportedProviderError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


def _resolve_batch_provider(
    config: GatewayConfig, provider: str, *, workspace_id: uuid.UUID | None = None
) -> tuple[LLMProvider, dict[str, Any]]:
    """Resolve a batch ``provider`` query param into (implementation, kwargs).

    The value is either a configured instance name (resolved to its
    ``provider_type`` with the instance's credentials) or a real any-llm
    provider, mirroring what ``create_batch`` echoes back as the batch's
    ``provider`` field. ``workspace_id`` reaches ``get_provider_kwargs``
    unchanged: consulted only when ``provider`` names no configured instance,
    the same organization-scoped fallback ``create_batch`` uses, so a batch
    created against an organization key stays reachable by every later
    lifecycle call (retrieve, cancel, list, results) rather than only at
    creation. For retrieve/cancel/results the caller passes the batch's own
    stored ``workspace_id`` (see ``_lifecycle_workspace_id``), not necessarily
    its own current one, so credentials resolve from the organization that
    created the batch even for a master-key or cross-workspace call;
    ``list_batches`` has no batch id to key off of and keeps using the
    caller's own workspace.
    """
    if provider in config.providers:
        impl = LLMProvider(config.provider_instance_type(provider))
        return impl, get_provider_kwargs(config, impl, instance=provider)
    impl = _parse_provider(provider)
    return impl, get_provider_kwargs(config, impl, workspace_id=workspace_id)


async def _lifecycle_workspace_id(
    db: AsyncSession, record: BatchRecord | None, api_key: APIKey | None
) -> uuid.UUID:
    """Which workspace's organization-scoped credentials a lifecycle call should use.

    The batch's own stored ``workspace_id`` when there is a record that has one
    (the workspace it was *created* in), so a master-key or legitimately
    cross-workspace retrieve/cancel/results still resolves the right
    organization's key. Falls back to the caller's own workspace only for
    legacy batches with no record, or one predating this column -- the same
    fallback shape ``_owns_batch``/``_authorize_legacy_batch`` already use for
    ownership.
    """
    if record is not None and record.workspace_id is not None:
        return record.workspace_id
    return await resolve_workspace_id(db, api_key)


def _owns_batch(batch: Batch, api_key: APIKey | None, is_master_key: bool) -> bool:
    """Whether the requester may access ``batch``.

    Ownership is anchored on the :data:`_OWNER_METADATA_KEY` metadata entry
    stamped at creation time. Batches without the marker (created before
    stamping existed, or via providers that do not round-trip metadata) remain
    accessible to any authenticated key. The master key may access any batch.
    """
    if is_master_key:
        return True
    owner = (batch.metadata or {}).get(_OWNER_METADATA_KEY)
    if owner is None:
        return True
    requester = str(api_key.user_id) if api_key and api_key.user_id else None
    return owner == requester


def _authorize_record(
    record: BatchRecord,
    batch_id: str,
    api_key: APIKey | None,
    is_master_key: bool,
) -> None:
    """Authorize access to a batch using only its stored record. No upstream call needed.

    Deliberately callable *before* the batch is retrieved from the provider:
    ``record.workspace_id`` decides which organization's BYO credential a
    lifecycle call dispatches with (see :func:`_lifecycle_workspace_id`), so
    checking ownership only after that dispatch would let an unauthorized
    caller cause the gateway to spend another organization's provider
    credential before ever learning the batch is not theirs. Only the
    strict, record-based check needs to run this early; the legacy,
    metadata-anchored fallback (:func:`_authorize_legacy_batch`) has no such
    risk, because a record-less batch has no stored workspace to leak --
    credentials there already resolve to the caller's own.

    Raises 404 (not 403) on denial so a foreign key cannot probe which batch
    ids exist.
    """
    if is_master_key:
        return
    requester = str(api_key.user_id) if api_key and api_key.user_id else None
    if record.user_id != requester:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch '{batch_id}' not found",
        )


def _authorize_legacy_batch(
    batch: Batch,
    batch_id: str,
    api_key: APIKey | None,
    is_master_key: bool,
) -> None:
    """Authorize a record-less (legacy) batch via the provider-side metadata marker.

    Only reached when :func:`get_batch_record` found no row, so this is
    necessarily after the batch has already been fetched from the provider
    (the metadata marker lives on the provider's own object); see
    :func:`_authorize_record` for why that ordering is safe only in this case.
    """
    if not _owns_batch(batch, api_key, is_master_key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch '{batch_id}' not found",
        )


async def _retrieve_batch_or_502(
    provider_enum: LLMProvider,
    batch_id: str,
    provider: str,
    provider_kwargs: dict[str, Any],
) -> Batch:
    """Retrieve a batch from the provider, mapping failures to 502."""
    try:
        batch: Batch = await aretrieve_batch(
            provider=provider_enum,
            batch_id=batch_id,
            **provider_kwargs,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch retrieve failed for %s: %s", provider, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider error",
        ) from e
    return batch


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.post("", response_model=None)
async def create_batch(
    raw_request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    request: CreateBatchRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any]:
    """Create a batch of LLM requests for asynchronous processing.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use the shared "default" user
    """
    api_key, is_master_key = auth_result
    api_key_id = api_key.id if api_key else None

    user_id = resolve_user_id(
        user_id_from_request=request.user,
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
        reject_mismatch=config.reject_user_mismatch,
    )

    rate_limit_info = check_rate_limit(raw_request, user_id)

    # Zero I/O for a keyed request (see `_pipeline.resolve_request_context`'s
    # equivalent comment); only a master-key request pays a lookup.
    workspace_id = await resolve_workspace_id(db, api_key)
    try:
        resolved = resolve_provider_selector(config, request.model, user_id, workspace_id=workspace_id)
    except (ValueError, AnyLLMError) as exc:
        _raise_for_unresolvable_model(request.model, exc)
    provider, model = resolved.provider, resolved.model

    # Model access control (per-key). Runs before the reservation, so nothing to
    # refund. Master-key callers have api_key None -> unrestricted. A key with no
    # list of its own inherits its user's default.
    key_allowlist = await resolve_request_allowlist(db, api_key)
    if key_allowlist is not None and not is_model_allowed(
        key_allowlist, f"{resolved.instance}:{resolved.model}"
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=model_not_allowed_detail(request.model),
        )

    # Organization-scoped model restriction (otari#643): mirrors `_pipeline.py`'s
    # equivalent gate. Applies only when the selector did not name a
    # configured instance, the same condition `provider_kwargs.get_provider_kwargs`
    # uses to decide whether to consult the organization overlay at all.
    if workspace_id is not None and resolved.instance not in config.providers:
        org_allowlist = cached_org_model_restriction(workspace_id, provider.value)
        if org_allowlist is not None and model not in org_allowlist:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=model_not_allowed_detail(request.model),
            )

    # Validate provider supports batch operations
    provider_class = AnyLLM.get_provider_class(provider)
    if not getattr(provider_class, "SUPPORTS_BATCH", False):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Provider '{provider.value}' does not support batch operations",
        )

    provider_kwargs = resolved.kwargs

    # A key flagged exclude_from_budget logs cost but is never reserved or folded
    # into users.spend, at create time and again when results are accounted.
    budget_exempt = api_key is not None and api_key.exclude_from_budget

    # Batch cost is unknown until results are retrieved, so the reservation
    # estimate is 0; it still enforces per-user state (user exists, not
    # blocked, not already over budget), matching the audio routes.
    reservation = await reserve_budget(
        db,
        user_id,
        0.0,
        model=request.model,
        strategy=config.budget_strategy,
        counts_toward_budget=not budget_exempt,
        scope=BudgetScopeRequest(api_key=api_key, provider_instance=resolved.instance),
        # So the free-model shortcut reads this organization's rate, the same one
        # the results are priced at when they are retrieved. Derived from the
        # workspace already resolved above (not `organization_for_key_id`,
        # which would re-derive it from `api_key_id` and pay a master-key
        # request's un-memoized workspace lookup a second time).
        organization_id=await organization_for_workspace_id(db, workspace_id),
    )

    # Stamp the billed user into the provider-side metadata so ownership can be
    # enforced on retrieve/cancel/results; a client-supplied value never wins.
    metadata = {**(request.metadata or {}), _OWNER_METADATA_KEY: user_id}

    # Build JSONL temp file from requests
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for req_item in request.requests:
            line = {
                "custom_id": req_item.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {**req_item.body, "model": model},
            }
            tmp.write(json.dumps(line) + "\n")
        tmp_path = tmp.name

    try:
        batch: Batch = await acreate_batch(
            provider=provider,
            input_file_path=tmp_path,
            endpoint="/v1/chat/completions",
            completion_window=request.completion_window,
            metadata=metadata,
            **provider_kwargs,
        )
    except HTTPException:
        await refund_reservation(db, reservation)
        raise
    except Exception as e:
        await log_batch_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=resolved.instance,
            endpoint="/v1/batches",
            user_id=user_id,
            error=str(e),
            status_code=failure_status_code(e),
            counts_toward_budget=not budget_exempt,
        )
        await refund_reservation(db, reservation)
        logger.error("Batch create failed for %s: %s", provider, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider error",
        ) from e
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            logger.warning("Failed to remove temp file %s", tmp_path)

    await log_batch_usage(
        db=db,
        log_writer=log_writer,
        api_key_id=api_key_id,
        model=model,
        provider=resolved.instance,
        endpoint="/v1/batches",
        user_id=user_id,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        counts_toward_budget=not budget_exempt,
    )
    await reconcile_reservation(db, reservation, 0.0)

    # Persist the ownership record so results accounting can be made idempotent,
    # spend can be folded once, and ownership enforced without depending on the
    # provider round-tripping the metadata marker. Best-effort: the batch already
    # exists upstream, so a persistence failure must not fail the request.
    await record_batch(
        db,
        batch_id=batch.id,
        provider=resolved.instance,
        user_id=user_id,
        api_key_id=api_key_id,
        model=model,
        workspace_id=workspace_id,
    )

    if rate_limit_info:
        for key, value in rate_limit_headers(rate_limit_info).items():
            response.headers[key] = value

    response_data = batch.model_dump()
    response_data["provider"] = resolved.instance
    return response_data


@router.get("/{batch_id}", response_model=None)
async def retrieve_batch(
    batch_id: str,
    provider: str,
    raw_request: Request,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> dict[str, Any]:
    """Retrieve the status of a batch."""
    api_key, is_master_key = auth_result
    record = await get_batch_record(db, batch_id)
    if record is not None:
        _authorize_record(record, batch_id, api_key, is_master_key)
    provider_enum, provider_kwargs = _resolve_batch_provider(
        config, provider, workspace_id=await _lifecycle_workspace_id(db, record, api_key)
    )

    batch = await _retrieve_batch_or_502(provider_enum, batch_id, provider, provider_kwargs)
    if record is None:
        _authorize_legacy_batch(batch, batch_id, api_key, is_master_key)

    response_data = batch.model_dump()
    response_data["provider"] = provider
    return response_data


@router.post("/{batch_id}/cancel", response_model=None)
async def cancel_batch(
    batch_id: str,
    provider: str,
    raw_request: Request,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> dict[str, Any]:
    """Cancel a batch."""
    api_key, is_master_key = auth_result
    record = await get_batch_record(db, batch_id)
    if record is not None:
        _authorize_record(record, batch_id, api_key, is_master_key)
    provider_enum, provider_kwargs = _resolve_batch_provider(
        config, provider, workspace_id=await _lifecycle_workspace_id(db, record, api_key)
    )

    existing = await _retrieve_batch_or_502(provider_enum, batch_id, provider, provider_kwargs)
    if record is None:
        _authorize_legacy_batch(existing, batch_id, api_key, is_master_key)

    try:
        batch: Batch = await acancel_batch(
            provider=provider_enum,
            batch_id=batch_id,
            **provider_kwargs,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch cancel failed for %s: %s", provider, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider error",
        ) from e

    response_data = batch.model_dump()
    response_data["provider"] = provider
    return response_data


@router.get("", response_model=None)
async def list_batches(
    provider: str,
    raw_request: Request,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    after: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """List batches for a provider.

    Non-master keys only see batches they own (plus legacy batches without an
    ownership marker); the page is filtered after the provider call, so a page
    may contain fewer than ``limit`` items.
    """
    api_key, is_master_key = auth_result
    provider_enum, provider_kwargs = _resolve_batch_provider(
        config, provider, workspace_id=await resolve_workspace_id(db, api_key)
    )

    list_kwargs: dict[str, Any] = {**provider_kwargs}
    if after is not None:
        list_kwargs["after"] = after
    if limit is not None:
        list_kwargs["limit"] = limit

    try:
        batches = await alist_batches(
            provider=provider_enum,
            **list_kwargs,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch list failed for %s: %s", provider, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider error",
        ) from e

    # Prefer the local records (strict) and fall back to the metadata marker for
    # legacy batches. One bulk lookup avoids a per-batch query.
    records = {} if is_master_key else await get_batch_records(db, [batch.id for batch in batches])
    requester = str(api_key.user_id) if api_key and api_key.user_id else None

    def _visible(batch: Batch) -> bool:
        if is_master_key:
            return True
        record = records.get(batch.id)
        if record is not None:
            return record.user_id == requester
        return _owns_batch(batch, api_key, is_master_key)

    return {"data": [{**batch.model_dump(), "provider": provider} for batch in batches if _visible(batch)]}


@router.get(
    "/{batch_id}/results",
    response_model=None,
    responses={
        status.HTTP_409_CONFLICT: {"description": "Batch is not yet complete"},
        status.HTTP_502_BAD_GATEWAY: {"description": "LLM provider error"},
    },
)
async def retrieve_batch_results(
    batch_id: str,
    provider: str,
    raw_request: Request,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any]:
    """Retrieve the results of a completed batch."""
    api_key, is_master_key = auth_result
    api_key_id = api_key.id if api_key else None

    record = await get_batch_record(db, batch_id)
    if record is not None:
        _authorize_record(record, batch_id, api_key, is_master_key)
    # The batch's originating workspace, reused below for pricing and the usage
    # row: both attribute to the creator, exactly like the credentials just
    # resolved from it, so one resolve serves all three rather than each
    # re-deriving its own notion of "whose batch is this" from `api_key_id`.
    origin_workspace_id = await _lifecycle_workspace_id(db, record, api_key)
    provider_enum, provider_kwargs = _resolve_batch_provider(config, provider, workspace_id=origin_workspace_id)

    batch = await _retrieve_batch_or_502(provider_enum, batch_id, provider, provider_kwargs)
    if record is None:
        _authorize_legacy_batch(batch, batch_id, api_key, is_master_key)

    # Attribute usage to the batch owner: the stored record when present (strict,
    # so a master-key retrieval bills the true owner), else the metadata marker
    # stamped at creation, else the key's own user for legacy batches with neither.
    metadata_owner = (batch.metadata or {}).get(_OWNER_METADATA_KEY)
    user_id = (record.user_id if record else None) or metadata_owner or (api_key.user_id if api_key else None)

    # Budget exemption follows the key that CREATED the batch (billing attributes to
    # its owner), not the retriever. Resolve it from the stored record; a deleted
    # originating key (api_key_id NULL) defaults to enforced, the conservative choice.
    batch_exempt = False
    if record is not None and record.api_key_id is not None:
        origin_key = await db.get(APIKey, record.api_key_id)
        batch_exempt = origin_key is not None and origin_key.exclude_from_budget

    try:
        result = await aretrieve_batch_results(
            provider=provider_enum,
            batch_id=batch_id,
            **provider_kwargs,
        )
    except HTTPException:
        raise
    except BatchNotCompleteError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Batch '{batch_id}' is not yet complete (status: {e.batch_status}). "
                f"Call GET /v1/batches/{batch_id}?provider={provider} to check the current status."
            ),
        ) from e
    except Exception as e:
        logger.error("Batch results retrieve failed for %s: %s", provider, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider error",
        ) from e

    # Extract model from the first successful result if available
    batch_model = "batch"
    for item in result.results:
        if item.result is not None:
            batch_model = item.result.model
            break

    # Sum per-request token usage so batch spend is visible in usage reporting.
    # The per-line counts are kept as well as summed: the row reports the batch's
    # totals, but pricing has to happen a line at a time (see below).
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    line_tokens: list[tuple[int, int]] = []
    for item in result.results:
        usage = item.result.usage if item.result is not None else None
        if usage is not None:
            line_prompt = usage.prompt_tokens or 0
            line_completion = usage.completion_tokens or 0
            prompt_tokens += line_prompt
            completion_tokens += line_completion
            total_tokens += usage.total_tokens or 0
            line_tokens.append((line_prompt, line_completion))

    cost: Decimal | None = None
    if total_tokens:
        # Rates follow the batch's originating workspace (`origin_workspace_id`,
        # resolved above), for the same reason the budget exemption above and
        # the owner attribution before it do: the cost is billed to the
        # creator, so it has to be priced at the creator's organization.
        # Scoping to the retriever would let a master-key retrieval, or a
        # second key of the same user in another organization, settle the
        # batch at rates its owner never negotiated.
        pricing = await find_model_pricing(
            db,
            provider_enum.value,
            batch_model,
            organization_id=await organization_for_workspace_id(db, origin_workspace_id),
        )
        if pricing:
            # Through the cost core rather than a local formula, so a batch is
            # priced by the same arithmetic as a live request. **One line at a
            # time**, because a threshold tier is a per-request cliff: charging
            # the batch's summed input against it would put a thousand short
            # requests on the long-context rate that not one of them reached. A
            # batch result reports OpenAI-shaped usage, so a line's cached
            # tokens, when it reports any, are already inside its
            # ``prompt_tokens``.
            cost = sum(
                (
                    calculate_token_cost(
                        pricing,
                        input_tokens=line_prompt,
                        output_tokens=line_completion,
                        cache_tokens_included=True,
                    )
                    for line_prompt, line_completion in line_tokens
                ),
                Decimal(0),
            )

    # Recorded batches are accounted exactly once: the first completed retrieval
    # claims the accounting slot, logs usage, and folds the cost into the owner's
    # spend; later retrievals (retries, polling clients) return results without
    # re-billing. The claim and the spend fold commit together (below), so if the
    # spend fold fails or the process dies mid-flight the claim rolls back and a
    # later retrieval re-accounts, rather than marking the batch accounted but
    # never billing it.
    #
    # A retrieval that finds billable tokens but no pricing row (cost is None)
    # does not claim the slot, so once pricing exists a later retrieval still
    # folds the spend once. Legacy batches with no record keep the prior behavior
    # (log every retrieval, no spend fold), since they cannot be deduplicated.
    # Exempt batches never fold spend, so there is nothing to defer for: they
    # claim and log immediately with cost null, like every other exempt path.
    pricing_missing = record is not None and bool(total_tokens) and cost is None and not batch_exempt
    if pricing_missing:
        should_account = False
    elif record is not None:
        should_account = await claim_batch_accounting(db, batch_id)
    else:
        should_account = True

    if should_account:
        await log_batch_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=batch_model,
            provider=provider,
            endpoint="/v1/batches/results",
            user_id=user_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            counts_toward_budget=not batch_exempt,
            # The creator's workspace, matching the user and the rates this row
            # was priced at; see the note on ``log_batch_usage``.
            workspace_id=origin_workspace_id,
        )
        if record is not None:
            if cost and user_id and not batch_exempt:
                # Folds the spend and commits the claim in one transaction.
                await record_external_spend(db, user_id, cost)
            else:
                # Nothing to fold (no cost/owner, or the originating key is
                # budget-exempt); still persist the one-time accounting claim.
                await db.commit()

    return {
        "results": [
            {
                "custom_id": item.custom_id,
                "result": item.result.model_dump() if item.result else None,
                "error": {"code": item.error.code, "message": item.error.message} if item.error else None,
            }
            for item in result.results
        ]
    }
