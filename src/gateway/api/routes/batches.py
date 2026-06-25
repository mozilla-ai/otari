"""Batch API endpoints for asynchronous LLM processing."""

import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from any_llm import AnyLLM, LLMProvider
from any_llm.api import acancel_batch, acreate_batch, alist_batches, aretrieve_batch, aretrieve_batch_results
from any_llm.exceptions import BatchNotCompleteError, UnsupportedProviderError
from any_llm.types.batch import Batch
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from gateway.api.deps import get_config, get_log_writer, verify_api_key_or_master_key
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey, UsageLog
from gateway.services.log_writer import LogWriter
from gateway.services.provider_kwargs import get_provider_kwargs, resolve_provider_selector

router = APIRouter(prefix="/v1/batches", tags=["batches"])


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
) -> None:
    """Log batch API usage."""
    usage_log = UsageLog(
        id=str(uuid.uuid4()),
        api_key_id=api_key_id,
        user_id=user_id,
        timestamp=datetime.now(UTC),
        model=model,
        provider=provider,
        endpoint=endpoint,
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


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.post("", response_model=None)
async def create_batch(
    raw_request: Request,
    background_tasks: BackgroundTasks,
    request: CreateBatchRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any]:
    """Create a batch of LLM requests for asynchronous processing."""
    api_key, is_master_key = auth_result
    api_key_id = api_key.id if api_key else None
    user_id = api_key.user_id if api_key else None

    try:
        resolved = resolve_provider_selector(config, request.model)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {e}",
        ) from e
    provider, model = resolved.provider, resolved.model

    # Validate provider supports batch operations
    provider_class = AnyLLM.get_provider_class(provider)
    if not getattr(provider_class, "SUPPORTS_BATCH", False):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Provider '{provider.value}' does not support batch operations",
        )

    provider_kwargs = resolved.kwargs

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
            metadata=request.metadata,
            **provider_kwargs,
        )
    except HTTPException:
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
    )

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
    provider_enum, provider_kwargs = _resolve_batch_provider(config, provider)

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
    provider_enum, provider_kwargs = _resolve_batch_provider(config, provider)

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
    """List batches for a provider."""
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

    return {"data": [{**batch.model_dump(), "provider": provider} for batch in batches]}


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
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any]:
    """Retrieve the results of a completed batch."""
    api_key, is_master_key = auth_result
    api_key_id = api_key.id if api_key else None
    user_id = api_key.user_id if api_key else None

    provider_enum, provider_kwargs = _resolve_batch_provider(config, provider)

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

    await log_batch_usage(
        log_writer=log_writer,
        api_key_id=api_key_id,
        model=batch_model,
        provider=provider,
        endpoint="/v1/batches/results",
        user_id=user_id,
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
