"""OpenAI-compatible moderations endpoint."""

from typing import Annotated, Any

from any_llm import amoderation
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._passthrough import run_passthrough
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey, ModelPricing
from gateway.services.log_writer import LogWriter
from gateway.services.pricing_service import flat_request_cost
from gateway.services.provider_kwargs import ResolvedProvider
from gateway.types.moderation import ModerationResponse

# Locked phrasing — cross-SDK error contract. Do not reword.
UNSUPPORTED_MODERATION_SUBSTRING = "does not support moderation"

router = APIRouter(prefix="/v1", tags=["moderations"])


class ModerationRequest(BaseModel):
    """OpenAI-compatible moderation request."""

    model: str
    input: str | list[str] | list[dict[str, Any]] = Field(
        description="Text, list of texts, or list of content-part dicts to moderate",
    )
    user: str | None = None


def _map_unsupported_moderation(e: Exception) -> HTTPException | None:
    """Surface any-llm's "provider does not support moderation" as a 400."""
    if isinstance(e, NotImplementedError) and UNSUPPORTED_MODERATION_SUBSTRING in str(e):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    return None


@router.post("/moderations", response_model=ModerationResponse)
async def create_moderation(
    raw_request: Request,
    response: Response,
    request: ModerationRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
    include_raw: Annotated[bool, Query()] = False,
) -> ModerationResponse:
    """OpenAI-compatible moderations endpoint.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use the shared "default" user
    """

    # Moderations is exempt from require_pricing: it is free at most providers
    # and is intentionally treated as $0 when unpriced (no "No pricing
    # configured" warning). Pricing, when present, is a flat per-request rate
    # (see flat_request_cost); moderation has no token usage.
    def compute_cost(result: ModerationResponse, pricing: ModelPricing | None) -> float:
        return flat_request_cost(pricing)

    def usage_tokens(result: ModerationResponse) -> tuple[int | None, int | None, int | None]:
        return (None, 0, None)

    async def call_provider(resolved: ResolvedProvider) -> ModerationResponse:
        moderation_kwargs: dict[str, Any] = {
            "model": resolved.model,
            "input": request.input,
            "provider": resolved.provider,
            "include_raw": include_raw,
            **resolved.kwargs,
        }
        return await amoderation(**moderation_kwargs)

    outcome = await run_passthrough(
        endpoint="/v1/moderations",
        raw_request=raw_request,
        response=response,
        auth_result=auth_result,
        db=db,
        config=config,
        log_writer=log_writer,
        model=request.model,
        user=request.user,
        call_provider=call_provider,
        estimate=flat_request_cost,
        usage_tokens=usage_tokens,
        compute_cost=compute_cost,
        map_provider_error=_map_unsupported_moderation,
    )
    return outcome.result
