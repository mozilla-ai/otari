"""Batch API endpoints for asynchronous LLM processing."""

import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
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
from gateway.api.routes._pipeline import _raise_for_unresolvable_model
from gateway.api.routes.chat import rate_limit_headers
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey, UsageLog
from gateway.rate_limit import check_rate_limit
from gateway.services.budget_service import (
    reconcile_reservation,
    refund_reservation,
    reserve_budget,
)
from gateway.services.log_writer import LogWriter
from gateway.services.model_access import is_model_allowed, model_not_allowed_detail, resolve_request_allowlist
from gateway.services.pricing_service import find_model_pricing
from gateway.services.provider_kwargs import get_provider_kwargs, resolve_provider_selector

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
    log_writer: LogWriter,
    api_key_id: str | None,
    model: str,
    provider: str,
    endpoint: str,
    user_id: str | None = None,
    error: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cost: float | None = None,
) -> None:
    """Log batch API usage, including token counts and cost when derivable."""
    usage_log = UsageLog(
        id=str(uuid.uuid4()),
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


def _resolve_batch_provider(config: GatewayConfig, provider: str) -> tuple[LLMProvider, dict[str, Any]]:
    """Resolve a batch ``provider`` query param into (implementation, kwargs).

    The value is either a configured instance name (resolved to its
    ``provider_type`` with the instance's credentials) or a real any-llm
    provider, mirroring what ``create_batch`` echoes back as the batch's
    ``provider`` field.
    """
    if provider in config.providers:
        impl = LLMProvider(config.provider_instance_type(provider))
        return impl, get_provider_kwargs(config, impl, instance=provider)
    impl = _parse_provider(provider)
    return impl, get_provider_kwargs(config, impl)


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


def _check_batch_ownership(batch: Batch, api_key: APIKey | None, is_master_key: bool, batch_id: str) -> None:
    """Raise 404 when the requester does not own ``batch``.

    404 rather than 403 so a foreign key cannot probe which batch ids exist.
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

    try:
        resolved = resolve_provider_selector(config, request.model)
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

    # Validate provider supports batch operations
    provider_class = AnyLLM.get_provider_class(provider)
    if not getattr(provider_class, "SUPPORTS_BATCH", False):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Provider '{provider.value}' does not support batch operations",
        )

    provider_kwargs = resolved.kwargs

    # Batch cost is unknown until results are retrieved, so the reservation
    # estimate is 0; it still enforces per-user state (user exists, not
    # blocked, not already over budget), matching the audio routes.
    reservation = await reserve_budget(db, user_id, 0.0, model=request.model, strategy=config.budget_strategy)

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
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=resolved.instance,
            endpoint="/v1/batches",
            user_id=user_id,
            error=str(e),
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
        log_writer=log_writer,
        api_key_id=api_key_id,
        model=model,
        provider=resolved.instance,
        endpoint="/v1/batches",
        user_id=user_id,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )
    await reconcile_reservation(db, reservation, 0.0)

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
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> dict[str, Any]:
    """Retrieve the status of a batch."""
    api_key, is_master_key = auth_result
    provider_enum, provider_kwargs = _resolve_batch_provider(config, provider)

    batch = await _retrieve_batch_or_502(provider_enum, batch_id, provider, provider_kwargs)
    _check_batch_ownership(batch, api_key, is_master_key, batch_id)

    response_data = batch.model_dump()
    response_data["provider"] = provider
    return response_data


@router.post("/{batch_id}/cancel", response_model=None)
async def cancel_batch(
    batch_id: str,
    provider: str,
    raw_request: Request,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> dict[str, Any]:
    """Cancel a batch."""
    api_key, is_master_key = auth_result
    provider_enum, provider_kwargs = _resolve_batch_provider(config, provider)

    existing = await _retrieve_batch_or_502(provider_enum, batch_id, provider, provider_kwargs)
    _check_batch_ownership(existing, api_key, is_master_key, batch_id)

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
    provider_enum, provider_kwargs = _resolve_batch_provider(config, provider)

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

    return {
        "data": [
            {**batch.model_dump(), "provider": provider}
            for batch in batches
            if _owns_batch(batch, api_key, is_master_key)
        ]
    }


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

    provider_enum, provider_kwargs = _resolve_batch_provider(config, provider)

    batch = await _retrieve_batch_or_502(provider_enum, batch_id, provider, provider_kwargs)
    _check_batch_ownership(batch, api_key, is_master_key, batch_id)

    # Attribute usage to the batch owner stamped at creation time (so a
    # master-key retrieval bills the owner, not user_id=None), falling back to
    # the key's user for legacy batches without the marker.
    owner = (batch.metadata or {}).get(_OWNER_METADATA_KEY)
    user_id = owner or (api_key.user_id if api_key else None)

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
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for item in result.results:
        usage = item.result.usage if item.result is not None else None
        if usage is not None:
            prompt_tokens += usage.prompt_tokens or 0
            completion_tokens += usage.completion_tokens or 0
            total_tokens += usage.total_tokens or 0

    cost: float | None = None
    if total_tokens:
        pricing = await find_model_pricing(db, provider_enum.value, batch_model)
        if pricing:
            cost = (prompt_tokens / 1_000_000) * pricing.input_price_per_million + (
                completion_tokens / 1_000_000
            ) * pricing.output_price_per_million

    await log_batch_usage(
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
    )

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
