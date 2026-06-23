"""Router preference-collection + status endpoints (standalone only).

These power the onboarding loop for the kNN routing memory:

* ``POST /v1/router/preferences/compare`` fans a prompt out to several candidate
  models concurrently and returns their responses, so a caller (human or judge)
  can see the answers side by side.
* ``POST /v1/router/preferences/rank`` records per-model quality scores for a
  prompt. Each submission writes one routing-memory record (the prompt embedding
  plus the per-model score map) plus one audit row, which is what the kNN router
  votes over at request time.
* ``GET /v1/router/status`` reports routing-memory progress per pool, the default
  pool plus each task partition, so onboarding can show progress toward the seed
  count.

Routing memory is per-tenant; records are scoped to the authenticated key's
user, the same identity the chat route bills and routes under.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, cast

from any_llm import AnyLLM, acompletion
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_api_key
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey, RouterPreference, RoutingMemory
from gateway.services.knn_router import KnnRoutingMemory
from gateway.services.provider_kwargs import get_provider_kwargs
from gateway.services.router_backend import get_router_backend

router = APIRouter(prefix="/v1/router", tags=["router"])

_KNN_REQUIRED = "Routing preferences require OTARI_ROUTER_BACKEND=knn"
# Cap the compare fan-out so one request cannot launch unbounded concurrent
# provider calls.
_MAX_COMPARE_MODELS = 8


def _tenant_id(api_key: APIKey) -> str:
    return api_key.user_id or ""


class CompareRequest(BaseModel):
    """Fan a single prompt out to a set of candidate models."""

    prompt: str = Field(description="User prompt to send to every candidate model.")
    models: list[str] = Field(
        min_length=1,
        max_length=_MAX_COMPARE_MODELS,
        description="provider/model identifiers to compare.",
    )
    system: str | None = Field(default=None, description="Optional system prompt applied to every model.")
    max_tokens: int | None = Field(default=None, description="Optional max output tokens per model.")


class ModelResponse(BaseModel):
    model: str
    content: str | None
    error: str | None = None


class CompareResponse(BaseModel):
    prompt: str
    responses: list[ModelResponse]


class RankRequest(BaseModel):
    """Submit per-model quality scores over models for a prompt."""

    prompt: str = Field(description="The prompt that was compared.")
    scores: dict[str, Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        min_length=1,
        description="Map of provider/model -> quality score in [0.0, 1.0] (1.0 = best). Models may tie.",
    )
    task_id: str | None = Field(default=None, description="Optional task label that scopes retrieval/scoring.")
    label_source: str = Field(default="human", description="Provenance of the label: 'human' or 'judge'.")


class RankResponse(BaseModel):
    recorded: int
    task_id: str | None = None
    warm: bool  # whether the pool this submission contributes to has crossed the seed count


class PoolStatus(BaseModel):
    records: int
    warm: bool


class RouterTaskInfo(BaseModel):
    task_id: str
    records: int
    warm: bool


class RouterStatus(BaseModel):
    """Routing-memory overview for a tenant.

    Routing memory has no single warmth, it is a set of independent pools, so the
    status reports each one. ``default_pool`` is what a request with no
    ``Otari-Router-Task`` header routes over (every record the tenant has, tagged
    or not); ``tasks`` lists each task partition, which a request carrying that
    task routes over. Every pool warms on its own once it crosses ``seed_count``.
    """

    backend: str
    seed_count: int
    default_pool: PoolStatus
    tasks: list[RouterTaskInfo]


def _require_knn(config: GatewayConfig) -> KnnRoutingMemory:
    backend = get_router_backend(config)
    if not isinstance(backend, KnnRoutingMemory):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_KNN_REQUIRED)
    return backend


@router.post("/preferences/compare", response_model=CompareResponse)
async def compare_models(
    request: CompareRequest,
    api_key: Annotated[APIKey, Depends(verify_api_key)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> CompareResponse:
    """Fan the prompt out to every candidate model concurrently."""
    messages: list[dict[str, Any]] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.prompt})

    async def _one(model: str) -> ModelResponse:
        try:
            provider, bare = AnyLLM.split_model_provider(model)
            kwargs = get_provider_kwargs(config, provider)
            if request.max_tokens is not None:
                kwargs["max_tokens"] = request.max_tokens
            result = await acompletion(model=bare, provider=provider, messages=cast(Any, messages), **kwargs)
            choices = getattr(result, "choices", None)
            content = choices[0].message.content if choices else None
            return ModelResponse(model=model, content=content)
        except Exception as exc:  # one model failing must not sink the comparison
            logger.warning("compare: model %s failed (%s)", model, type(exc).__name__)
            return ModelResponse(model=model, content=None, error="model invocation failed")

    responses = await asyncio.gather(*[_one(m) for m in request.models])
    return CompareResponse(prompt=request.prompt, responses=list(responses))


@router.post("/preferences/rank", response_model=RankResponse)
async def rank_models(
    request: RankRequest,
    api_key: Annotated[APIKey, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> RankResponse:
    """Persist per-model scores as a routing-memory record plus an audit row."""
    backend = _require_knn(config)
    tenant = _tenant_id(api_key)

    # Write the routing-memory record first: it is the load-bearing record (the
    # one the router votes over), and embedding it can fail. Writing the audit row
    # only afterwards means a failure here never leaves an orphan audit row.
    recorded = await backend.record_preference(
        tenant_id=tenant,
        prompt=request.prompt,
        scores=request.scores,
        task_id=request.task_id,
        label_source=request.label_source,
    )

    try:
        db.add(
            RouterPreference(
                tenant_id=tenant,
                prompt=request.prompt,
                task_id=request.task_id,
                scores=request.scores,
                label_source=request.label_source,
            )
        )
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise
    # Warmth of the pool this submission contributes to: the named task partition,
    # or (no task) the default pool of every record the tenant has.
    stmt = (
        select(func.count())
        .select_from(RoutingMemory)
        .where(RoutingMemory.tenant_id == tenant, RoutingMemory.embedding_model == backend.embedding_model)
    )
    if request.task_id is not None:
        stmt = stmt.where(RoutingMemory.task_id == request.task_id)
    count = (await db.execute(stmt)).scalar_one()
    return RankResponse(recorded=recorded, task_id=request.task_id, warm=int(count) >= backend.seed_count)


@router.get("/status", response_model=RouterStatus)
async def router_status(
    api_key: Annotated[APIKey, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> RouterStatus:
    """Report routing-memory progress for the authenticated tenant.

    Routing memory is a set of independent pools, so there is no single warmth.
    ``default_pool`` covers a request with no ``Otari-Router-Task`` header (every
    record the tenant has); ``tasks`` lists each task partition. Each pool warms
    on its own once it crosses ``seed_count``.
    """
    tenant = _tenant_id(api_key)
    backend = get_router_backend(config)
    if not isinstance(backend, KnnRoutingMemory):
        return RouterStatus(
            backend=config.router_backend, seed_count=0, default_pool=PoolStatus(records=0, warm=False), tasks=[]
        )
    base = (
        RoutingMemory.tenant_id == tenant,
        RoutingMemory.embedding_model == backend.embedding_model,
    )
    total = (await db.execute(select(func.count()).select_from(RoutingMemory).where(*base))).scalar_one()
    rows = (
        await db.execute(
            select(RoutingMemory.task_id, func.count())
            .where(*base, RoutingMemory.task_id.is_not(None))
            .group_by(RoutingMemory.task_id)
            .order_by(func.count().desc(), RoutingMemory.task_id)
        )
    ).all()
    seed = backend.seed_count
    tasks = [RouterTaskInfo(task_id=str(t), records=int(n), warm=int(n) >= seed) for t, n in rows]
    return RouterStatus(
        backend="knn",
        seed_count=seed,
        default_pool=PoolStatus(records=int(total), warm=int(total) >= seed),
        tasks=tasks,
    )
