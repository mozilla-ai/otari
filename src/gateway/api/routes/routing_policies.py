from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, verify_master_key
from gateway.models.entities import RoutingPolicy, RoutingPolicyRevision
from gateway.services.routing_policy_service import ACTIVE_ROUTING_POLICY_STATUS, ROUTING_STRATEGIES

router = APIRouter(prefix="/v1/routing-policies", tags=["routing-policies"])

ROUTING_POLICY_STATUSES = {"draft", ACTIVE_ROUTING_POLICY_STATUS, "archived"}
_MERGE_DEFAULT_STRATEGY_TYPES = {"fallback", "intelligent", "weighted_score"}
_MERGE_INTELLIGENT_AXES = {"cost", "performance", "intelligence"}
_MERGE_AXIS_TIER_THRESHOLDS = {
    "cost": {"medium": 1600, "complex": 6000, "reasoning": 12000},
    "performance": {"medium": 800, "complex": 3000, "reasoning": 9000},
    "intelligence": {"medium": 300, "complex": 1400, "reasoning": 5000},
}
_MERGE_PROVIDER_KEYS = {
    "provider",
    "model",
    "priority",
    "tier",
    "input_price_per_million",
    "output_price_per_million",
}
_WEIGHTED_SCORE_KEYS = {
    "weights",
    "quality_weight",
    "cost_weight",
    "latency_weight",
    "default_quality_score",
    "unknown_cost_score",
    "unknown_latency_score",
}


class CreateRoutingPolicyRequest(BaseModel):
    """Request model for creating a routing policy."""

    name: str = Field(min_length=1)
    strategy: str = Field(default="cost_tier")
    config: dict[str, Any] = Field(default_factory=dict)
    default_strategy: dict[str, Any] | None = None
    is_default: bool = False
    status: str = ACTIVE_ROUTING_POLICY_STATUS
    change_note: str | None = None


class CloneRoutingPolicyRequest(BaseModel):
    """Request model for cloning a routing policy into a draft."""

    name: str | None = Field(default=None, min_length=1)
    change_note: str | None = None


class ApplyRoutingPolicyRevisionRequest(BaseModel):
    """Request model for applying a previous routing policy revision."""

    change_note: str | None = None


class RoutingPolicyEvalScoreItem(BaseModel):
    """One uploaded eval or benchmark score for a candidate model."""

    model: str = Field(min_length=1)
    provider: str | None = Field(default=None, min_length=1)
    score: float | None = Field(default=None, ge=0, le=100)
    quality_score: float | None = Field(default=None, ge=0, le=100)
    benchmark_score: float | None = Field(default=None, ge=0, le=100)
    metric: str | None = Field(default=None, min_length=1)
    sample_count: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApplyRoutingPolicyEvalScoresRequest(BaseModel):
    """Request model for applying eval scores to weighted routing candidates."""

    scores: list[RoutingPolicyEvalScoreItem] = Field(min_length=1)
    change_note: str | None = None


class UpdateRoutingPolicyRequest(BaseModel):
    """Request model for updating a routing policy."""

    name: str | None = Field(default=None, min_length=1)
    strategy: str | None = None
    config: dict[str, Any] | None = None
    default_strategy: dict[str, Any] | None = None
    is_default: bool | None = None
    status: str | None = None
    change_note: str | None = None


class RoutingPolicyResponse(BaseModel):
    """Response model for routing policy information."""

    policy_id: str
    name: str
    strategy: str
    config: dict[str, Any]
    default_strategy: dict[str, Any] | None
    is_default: bool
    revision: int
    status: str
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, policy: RoutingPolicy) -> "RoutingPolicyResponse":
        """Create a response from an ORM model."""
        return cls(
            policy_id=policy.policy_id,
            name=policy.name,
            strategy=policy.strategy,
            config=dict(policy.config_) if policy.config_ else {},
            default_strategy=_default_strategy_from_internal(policy.strategy, policy.config_ or {}),
            is_default=bool(policy.is_default),
            revision=int(policy.revision or 0),
            status=policy.status,
            created_at=policy.created_at.isoformat(),
            updated_at=policy.updated_at.isoformat(),
        )


class RoutingPolicyRevisionResponse(BaseModel):
    """Response model for routing policy revision history."""

    revision_id: str
    policy_id: str
    revision: int
    action: str
    name: str
    strategy: str
    config: dict[str, Any]
    default_strategy: dict[str, Any] | None
    is_default: bool
    status: str
    change_note: str | None
    created_at: str

    @classmethod
    def from_model(cls, revision: RoutingPolicyRevision) -> "RoutingPolicyRevisionResponse":
        """Create a response from an ORM model."""
        return cls(
            revision_id=revision.revision_id,
            policy_id=revision.policy_id,
            revision=revision.revision,
            action=revision.action,
            name=revision.name,
            strategy=revision.strategy,
            config=dict(revision.config_) if revision.config_ else {},
            default_strategy=_default_strategy_from_internal(revision.strategy, revision.config_ or {}),
            is_default=bool(revision.is_default),
            status=revision.status,
            change_note=revision.change_note,
            created_at=revision.created_at.isoformat(),
        )


class AppliedRoutingPolicyEvalScoreResponse(BaseModel):
    """Candidate score update returned after eval ingestion."""

    model: str
    previous_quality_score: float | None
    quality_score: float
    sample_count: int
    metrics: list[str]


class ApplyRoutingPolicyEvalScoresResponse(BaseModel):
    """Response model for eval score ingestion."""

    policy: RoutingPolicyResponse
    applied_count: int
    unmatched_models: list[str]
    applied_scores: list[AppliedRoutingPolicyEvalScoreResponse]


@dataclass(frozen=True)
class _EvalScoreAggregate:
    model: str
    quality_score: float
    sample_count: int
    metrics: list[str]
    metadata: dict[str, Any]


def _validate_strategy(strategy: str) -> None:
    if strategy not in ROUTING_STRATEGIES:
        supported = ", ".join(sorted(ROUTING_STRATEGIES))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported routing strategy '{strategy}'. Supported strategies: {supported}",
        )


def _unprocessable(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)


def _number_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _score_value(value: Any) -> float | None:
    parsed = _number_value(value)
    if parsed is None or parsed < 0:
        return None
    if parsed <= 1:
        return parsed
    if parsed <= 100:
        return parsed / 100
    return None


def _model_selector(provider: str | None, model: str) -> str:
    model_value = model.strip()
    if not provider:
        if "/" in model_value:
            return model_value.replace("/", ":", 1)
        return model_value
    provider_value = provider.strip()
    if model_value.startswith(f"{provider_value}:") or model_value.startswith(f"{provider_value}/"):
        return model_value.replace("/", ":", 1)
    return f"{provider_value}:{model_value}"


def _normalized_model_selector(provider: str | None, model: str) -> str:
    return _model_selector(provider, model).replace("/", ":", 1).strip()


def _split_model_selector(model_selector: str) -> tuple[str | None, str]:
    if ":" in model_selector:
        provider, model = model_selector.split(":", 1)
        return provider or None, model
    if "/" in model_selector:
        provider, model = model_selector.split("/", 1)
        return provider or None, model
    return None, model_selector


def _provider_priority(item: Mapping[str, Any], position: int) -> float:
    priority = _number_value(item.get("priority"))
    return priority if priority is not None else float(position)


def _candidate_from_merge_provider(item: Mapping[str, Any]) -> dict[str, Any]:
    provider = item.get("provider")
    model = item.get("model")
    if not isinstance(model, str) or not model.strip():
        raise _unprocessable("default_strategy.providers entries must include a non-empty model")
    if provider is not None and (not isinstance(provider, str) or not provider.strip()):
        raise _unprocessable("default_strategy.providers entries must include a non-empty provider when set")

    candidate: dict[str, Any] = {"model": _model_selector(provider if isinstance(provider, str) else None, model)}
    for key in ("tier", "input_price_per_million", "output_price_per_million"):
        if key in item:
            candidate[key] = item[key]

    metadata = {key: value for key, value in item.items() if key not in _MERGE_PROVIDER_KEYS}
    if "priority" in item:
        metadata["priority"] = item["priority"]
    if metadata:
        candidate["metadata"] = metadata
    return candidate


def _merge_strategy_providers(default_strategy: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    providers = default_strategy.get("providers")
    if not isinstance(providers, list) or not providers:
        raise _unprocessable("default_strategy.providers must be a non-empty list")
    provider_items: list[Mapping[str, Any]] = []
    for item in providers:
        if not isinstance(item, dict):
            raise _unprocessable("default_strategy.providers entries must be objects")
        provider_items.append(item)
    return provider_items


def _merge_config_from_default_strategy(
    default_strategy: Mapping[str, Any],
    *,
    base_config: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    strategy_type_value = default_strategy.get("type")
    if not isinstance(strategy_type_value, str):
        raise _unprocessable("default_strategy.type is required")
    strategy_type = strategy_type_value.strip().lower()
    if strategy_type not in _MERGE_DEFAULT_STRATEGY_TYPES:
        supported = ", ".join(sorted(_MERGE_DEFAULT_STRATEGY_TYPES))
        raise _unprocessable(f"Unsupported default_strategy.type '{strategy_type_value}'. Supported types: {supported}")

    provider_items = _merge_strategy_providers(default_strategy)
    config = dict(base_config or {})
    for key in ("constraints", "health", "match", "tier_thresholds"):
        if key in default_strategy:
            config[key] = default_strategy[key]

    if strategy_type == "fallback":
        ordered_providers = sorted(
            enumerate(provider_items, start=1),
            key=lambda indexed: (_provider_priority(indexed[1], indexed[0]), indexed[0]),
        )
        config["candidates"] = [
            _candidate_from_merge_provider(provider_item)
            for _, provider_item in ordered_providers
        ]
        if "fallback_enabled" in default_strategy:
            config["fallback_enabled"] = bool(default_strategy["fallback_enabled"])
        strategy = "single" if len(provider_items) == 1 else "priority"
        return strategy, config

    if strategy_type == "weighted_score":
        config["candidates"] = [_candidate_from_merge_provider(provider_item) for provider_item in provider_items]
        scoring = default_strategy.get("scoring")
        scoring_config = dict(scoring) if isinstance(scoring, dict) else {}
        for key in _WEIGHTED_SCORE_KEYS:
            if key in default_strategy:
                scoring_config[key] = default_strategy[key]
        if scoring_config:
            config["scoring"] = scoring_config
        config.setdefault("fallback_enabled", True)
        return "weighted_score", config

    axis_value = default_strategy.get("axis", "performance")
    if not isinstance(axis_value, str):
        raise _unprocessable("default_strategy.axis must be a string")
    axis = axis_value.strip().lower()
    if axis not in _MERGE_INTELLIGENT_AXES:
        supported = ", ".join(sorted(_MERGE_INTELLIGENT_AXES))
        raise _unprocessable(f"Unsupported default_strategy.axis '{axis_value}'. Supported axes: {supported}")
    config["candidates"] = [_candidate_from_merge_provider(provider_item) for provider_item in provider_items]
    config["axis"] = axis
    config.setdefault("fallback_enabled", True)
    config.setdefault("tier_thresholds", _MERGE_AXIS_TIER_THRESHOLDS[axis])
    return "intelligent", config


def _candidate_item_to_provider(item: Any, *, position: int, tier: str | None = None) -> dict[str, Any] | None:
    if isinstance(item, str):
        model_selector = item
        metadata: Mapping[str, Any] = {}
        input_price = None
        output_price = None
        candidate_tier = tier
    elif isinstance(item, dict):
        model_value = item.get("model")
        if not isinstance(model_value, str) or not model_value.strip():
            return None
        model_selector = model_value
        metadata_value = item.get("metadata")
        metadata = metadata_value if isinstance(metadata_value, dict) else {}
        input_price = item.get("input_price_per_million")
        output_price = item.get("output_price_per_million")
        candidate_tier = item.get("tier") if isinstance(item.get("tier"), str) else tier
    else:
        return None

    provider, model = _split_model_selector(model_selector)
    provider_item: dict[str, Any] = {"provider": provider, "model": model, "priority": position}
    if candidate_tier is not None:
        provider_item["tier"] = candidate_tier
    if input_price is not None:
        provider_item["input_price_per_million"] = input_price
    if output_price is not None:
        provider_item["output_price_per_million"] = output_price
    for key, value in metadata.items():
        if key == "priority":
            provider_item["priority"] = value
        elif key not in provider_item:
            provider_item[key] = value
    return provider_item


def _providers_from_config(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    providers: list[dict[str, Any]] = []
    candidates = config.get("candidates")
    if isinstance(candidates, list):
        for item in candidates:
            provider_item = _candidate_item_to_provider(item, position=len(providers) + 1)
            if provider_item is not None:
                providers.append(provider_item)

    tiers = config.get("tiers")
    if isinstance(tiers, dict):
        for tier_name, items in tiers.items():
            if not isinstance(tier_name, str) or not isinstance(items, list):
                continue
            for item in items:
                provider_item = _candidate_item_to_provider(item, position=len(providers) + 1, tier=tier_name)
                if provider_item is not None:
                    providers.append(provider_item)
    return providers


def _default_strategy_from_internal(strategy: str, config: Mapping[str, Any]) -> dict[str, Any] | None:
    providers = _providers_from_config(config)
    if not providers:
        return None
    if strategy in {"single", "priority"}:
        return {
            "type": "fallback",
            "providers": providers,
            "fallback_enabled": strategy != "single" and bool(config.get("fallback_enabled", True)),
        }
    if strategy == "intelligent":
        axis = config.get("axis")
        return {
            "type": "intelligent",
            "axis": axis if isinstance(axis, str) else "performance",
            "providers": providers,
            "fallback_enabled": bool(config.get("fallback_enabled", True)),
        }
    if strategy == "weighted_score":
        response: dict[str, Any] = {
            "type": "weighted_score",
            "providers": providers,
            "fallback_enabled": bool(config.get("fallback_enabled", True)),
        }
        scoring = config.get("scoring")
        if isinstance(scoring, dict):
            response["scoring"] = dict(scoring)
        return response
    return {
        "type": strategy,
        "providers": providers,
        "fallback_enabled": bool(config.get("fallback_enabled", strategy != "single")),
    }


def _eval_item_score(item: RoutingPolicyEvalScoreItem) -> float:
    for value in (item.quality_score, item.score, item.benchmark_score):
        score = _score_value(value)
        if score is not None:
            return score
    raise _unprocessable("Each eval score must include score, quality_score, or benchmark_score")


def _aggregate_eval_scores(items: list[RoutingPolicyEvalScoreItem]) -> dict[str, _EvalScoreAggregate]:
    totals: dict[str, float] = {}
    sample_counts: dict[str, int] = {}
    metrics: dict[str, set[str]] = {}
    metadata_by_model: dict[str, dict[str, Any]] = {}

    for item in items:
        model_key = _normalized_model_selector(item.provider, item.model)
        score = _eval_item_score(item)
        sample_count = item.sample_count or 1
        totals[model_key] = totals.get(model_key, 0.0) + (score * sample_count)
        sample_counts[model_key] = sample_counts.get(model_key, 0) + sample_count
        if item.metric is not None:
            metrics.setdefault(model_key, set()).add(item.metric)
        if item.metadata:
            metadata = metadata_by_model.setdefault(model_key, {})
            metadata.update(item.metadata)

    return {
        model_key: _EvalScoreAggregate(
            model=model_key,
            quality_score=totals[model_key] / sample_counts[model_key],
            sample_count=sample_counts[model_key],
            metrics=sorted(metrics.get(model_key, set())),
            metadata=metadata_by_model.get(model_key, {}),
        )
        for model_key in totals
    }


def _candidate_quality_score(candidate: Any) -> float | None:
    if not isinstance(candidate, dict):
        return None
    score = _score_value(candidate.get("quality_score"))
    if score is not None:
        return score
    metadata = candidate.get("metadata")
    if isinstance(metadata, dict):
        return _score_value(metadata.get("quality_score"))
    return None


def _candidate_model_key(candidate: Any) -> str | None:
    if isinstance(candidate, str):
        return _normalized_model_selector(None, candidate)
    if isinstance(candidate, dict):
        model = candidate.get("model")
        if isinstance(model, str) and model.strip():
            return _normalized_model_selector(None, model)
    return None


def _apply_eval_score_to_candidate(
    candidate: Any,
    *,
    scores_by_model: Mapping[str, _EvalScoreAggregate],
    applied_scores: list[AppliedRoutingPolicyEvalScoreResponse],
    applied_model_keys: set[str],
    updated_at: str,
) -> Any:
    model_key = _candidate_model_key(candidate)
    if model_key is None or model_key not in scores_by_model:
        return candidate

    aggregate = scores_by_model[model_key]
    previous_quality_score = _candidate_quality_score(candidate)
    updated_candidate = {"model": candidate} if isinstance(candidate, str) else dict(candidate)
    metadata_value = updated_candidate.get("metadata")
    metadata = dict(metadata_value) if isinstance(metadata_value, dict) else {}
    metadata["eval_score"] = {
        "quality_score": aggregate.quality_score,
        "sample_count": aggregate.sample_count,
        "metrics": aggregate.metrics,
        "metadata": aggregate.metadata,
        "updated_at": updated_at,
    }
    updated_candidate["quality_score"] = aggregate.quality_score
    updated_candidate["metadata"] = metadata
    applied_model_keys.add(model_key)
    applied_scores.append(
        AppliedRoutingPolicyEvalScoreResponse(
            model=model_key,
            previous_quality_score=previous_quality_score,
            quality_score=aggregate.quality_score,
            sample_count=aggregate.sample_count,
            metrics=aggregate.metrics,
        )
    )
    return updated_candidate


def _apply_eval_scores_to_policy_config(
    config: Mapping[str, Any],
    *,
    scores_by_model: Mapping[str, _EvalScoreAggregate],
) -> tuple[dict[str, Any], list[AppliedRoutingPolicyEvalScoreResponse], list[str]]:
    updated_config = dict(config)
    applied_scores: list[AppliedRoutingPolicyEvalScoreResponse] = []
    applied_model_keys: set[str] = set()
    updated_at = datetime.now(UTC).isoformat()

    candidates = updated_config.get("candidates")
    if isinstance(candidates, list):
        updated_config["candidates"] = [
            _apply_eval_score_to_candidate(
                candidate,
                scores_by_model=scores_by_model,
                applied_scores=applied_scores,
                applied_model_keys=applied_model_keys,
                updated_at=updated_at,
            )
            for candidate in candidates
        ]

    tiers = updated_config.get("tiers")
    if isinstance(tiers, dict):
        updated_tiers: dict[str, Any] = {}
        for tier_name, tier_candidates in tiers.items():
            if not isinstance(tier_candidates, list):
                updated_tiers[str(tier_name)] = tier_candidates
                continue
            updated_tiers[str(tier_name)] = [
                _apply_eval_score_to_candidate(
                    candidate,
                    scores_by_model=scores_by_model,
                    applied_scores=applied_scores,
                    applied_model_keys=applied_model_keys,
                    updated_at=updated_at,
                )
                for candidate in tier_candidates
            ]
        updated_config["tiers"] = updated_tiers

    unmatched_models = sorted(set(scores_by_model) - applied_model_keys)
    return updated_config, applied_scores, unmatched_models


def _create_policy_shape(request: CreateRoutingPolicyRequest) -> tuple[str, dict[str, Any]]:
    if request.default_strategy is not None:
        return _merge_config_from_default_strategy(request.default_strategy, base_config=request.config)
    return request.strategy, dict(request.config)


def _update_policy_shape(
    policy: RoutingPolicy,
    payload: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    if "default_strategy" not in payload or payload["default_strategy"] is None:
        strategy = str(payload["strategy"]) if payload.get("strategy") is not None else None
        config = dict(payload["config"] or {}) if "config" in payload else None
        return strategy, config

    base_config = dict(payload["config"] or {}) if "config" in payload else dict(policy.config_ or {})
    default_strategy = payload["default_strategy"]
    if not isinstance(default_strategy, dict):
        raise _unprocessable("default_strategy must be an object")
    return _merge_config_from_default_strategy(default_strategy, base_config=base_config)


def _validate_status(policy_status: str) -> None:
    if policy_status not in ROUTING_POLICY_STATUSES:
        supported = ", ".join(sorted(ROUTING_POLICY_STATUSES))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unsupported routing policy status '{policy_status}'. Supported statuses: {supported}",
        )


def _ensure_default_policy_is_active(*, policy_status: str, is_default: bool) -> None:
    if is_default and policy_status != ACTIVE_ROUTING_POLICY_STATUS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Only active routing policies can be default",
        )


def _record_policy_revision(
    db: AsyncSession,
    policy: RoutingPolicy,
    *,
    action: str,
    change_note: str | None,
) -> None:
    db.add(
        RoutingPolicyRevision(
            policy_id=policy.policy_id,
            revision=policy.revision,
            action=action,
            name=policy.name,
            strategy=policy.strategy,
            config_=dict(policy.config_) if policy.config_ else {},
            is_default=bool(policy.is_default),
            status=policy.status,
            change_note=change_note,
        )
    )


def _bump_policy_revision(policy: RoutingPolicy) -> None:
    policy.revision = int(policy.revision or 0) + 1


async def _get_policy_revision(
    db: AsyncSession,
    *,
    policy_id: str,
    revision: int,
) -> RoutingPolicyRevision:
    result = await db.execute(
        select(RoutingPolicyRevision).where(
            RoutingPolicyRevision.policy_id == policy_id,
            RoutingPolicyRevision.revision == revision,
        )
    )
    policy_revision = result.scalar_one_or_none()
    if policy_revision is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing policy revision '{policy_id}@{revision}' not found",
        )
    return policy_revision


async def _unset_default_policies(
    db: AsyncSession,
    *,
    except_policy_id: str | None = None,
    change_note: str | None = None,
) -> None:
    stmt = select(RoutingPolicy).where(RoutingPolicy.is_default.is_(True))
    if except_policy_id is not None:
        stmt = stmt.where(RoutingPolicy.policy_id != except_policy_id)
    result = await db.execute(stmt)
    for policy in result.scalars().all():
        policy.is_default = False
        _bump_policy_revision(policy)
        _record_policy_revision(
            db,
            policy,
            action="unset_default",
            change_note=change_note,
        )


@router.post("", dependencies=[Depends(verify_master_key)])
async def create_routing_policy(
    request: CreateRoutingPolicyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RoutingPolicyResponse:
    """Create a routing policy."""
    policy_strategy, policy_config = _create_policy_shape(request)
    _validate_strategy(policy_strategy)
    _validate_status(request.status)
    _ensure_default_policy_is_active(policy_status=request.status, is_default=request.is_default)
    if request.is_default:
        await _unset_default_policies(
            db,
            change_note=request.change_note or f"Unset default before creating routing policy '{request.name}'",
        )

    policy = RoutingPolicy(
        name=request.name,
        strategy=policy_strategy,
        config_=policy_config,
        is_default=request.is_default,
        revision=1,
        status=request.status,
    )
    db.add(policy)
    await db.flush()
    _record_policy_revision(
        db,
        policy,
        action="create",
        change_note=request.change_note,
    )
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(policy)
    return RoutingPolicyResponse.from_model(policy)


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_routing_policies(
    db: Annotated[AsyncSession, Depends(get_db)],
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[RoutingPolicyResponse]:
    """List routing policies."""
    if status_filter is not None:
        _validate_status(status_filter)
    stmt = select(RoutingPolicy)
    if status_filter is not None:
        stmt = stmt.where(RoutingPolicy.status == status_filter)
    result = await db.execute(stmt.order_by(RoutingPolicy.created_at.desc()).offset(skip).limit(limit))
    policies = result.scalars().all()
    return [RoutingPolicyResponse.from_model(policy) for policy in policies]


@router.get("/{policy_id}/revisions", dependencies=[Depends(verify_master_key)])
async def list_routing_policy_revisions(
    policy_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[RoutingPolicyRevisionResponse]:
    """List immutable revisions for a routing policy."""
    result = await db.execute(
        select(RoutingPolicyRevision)
        .where(RoutingPolicyRevision.policy_id == policy_id)
        .order_by(RoutingPolicyRevision.revision.desc())
        .offset(skip)
        .limit(limit)
    )
    revisions = result.scalars().all()
    return [RoutingPolicyRevisionResponse.from_model(revision) for revision in revisions]


@router.get("/{policy_id}/revisions/{revision}", dependencies=[Depends(verify_master_key)])
async def get_routing_policy_revision(
    policy_id: str,
    revision: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RoutingPolicyRevisionResponse:
    """Get one immutable routing policy revision."""
    policy_revision = await _get_policy_revision(db, policy_id=policy_id, revision=revision)
    return RoutingPolicyRevisionResponse.from_model(policy_revision)


@router.post("/{policy_id}/revisions/{revision}/apply", dependencies=[Depends(verify_master_key)])
async def apply_routing_policy_revision(
    policy_id: str,
    revision: int,
    request: ApplyRoutingPolicyRevisionRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RoutingPolicyResponse:
    """Apply a previous policy revision as a new audited revision."""
    policy = await db.get(RoutingPolicy, policy_id)
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing policy '{policy_id}' not found",
        )
    policy_revision = await _get_policy_revision(db, policy_id=policy_id, revision=revision)
    _validate_strategy(policy_revision.strategy)
    _validate_status(policy_revision.status)
    _ensure_default_policy_is_active(
        policy_status=policy_revision.status,
        is_default=bool(policy_revision.is_default),
    )

    change_note = request.change_note or f"Applied routing policy revision {revision}"
    if policy_revision.is_default:
        await _unset_default_policies(
            db,
            except_policy_id=policy.policy_id,
            change_note=f"Unset default before applying routing policy revision '{policy_id}@{revision}'",
        )

    policy.name = policy_revision.name
    policy.strategy = policy_revision.strategy
    policy.config_ = dict(policy_revision.config_) if policy_revision.config_ else {}
    policy.is_default = bool(policy_revision.is_default)
    policy.status = policy_revision.status
    _bump_policy_revision(policy)
    _record_policy_revision(
        db,
        policy,
        action="apply_revision",
        change_note=change_note,
    )
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(policy)
    return RoutingPolicyResponse.from_model(policy)


@router.post("/{policy_id}/eval-scores", dependencies=[Depends(verify_master_key)])
async def apply_routing_policy_eval_scores(
    policy_id: str,
    request: ApplyRoutingPolicyEvalScoresRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ApplyRoutingPolicyEvalScoresResponse:
    """Apply uploaded eval or benchmark scores to weighted routing candidates."""
    policy = await db.get(RoutingPolicy, policy_id)
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing policy '{policy_id}' not found",
        )
    if policy.strategy != "weighted_score":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Eval score ingestion requires a weighted_score routing policy",
        )

    scores_by_model = _aggregate_eval_scores(request.scores)
    updated_config, applied_scores, unmatched_models = _apply_eval_scores_to_policy_config(
        policy.config_ or {},
        scores_by_model=scores_by_model,
    )
    if not applied_scores:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No eval scores matched routing policy candidates",
        )

    policy.config_ = updated_config
    _bump_policy_revision(policy)
    _record_policy_revision(
        db,
        policy,
        action="apply_eval_scores",
        change_note=request.change_note or "Applied routing policy eval scores",
    )
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(policy)
    return ApplyRoutingPolicyEvalScoresResponse(
        policy=RoutingPolicyResponse.from_model(policy),
        applied_count=len(applied_scores),
        unmatched_models=unmatched_models,
        applied_scores=applied_scores,
    )


@router.get("/{policy_id}", dependencies=[Depends(verify_master_key)])
async def get_routing_policy(
    policy_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RoutingPolicyResponse:
    """Get a routing policy."""
    policy = await db.get(RoutingPolicy, policy_id)
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing policy '{policy_id}' not found",
        )
    return RoutingPolicyResponse.from_model(policy)


@router.post("/{policy_id}/clone", dependencies=[Depends(verify_master_key)])
async def clone_routing_policy(
    policy_id: str,
    request: CloneRoutingPolicyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RoutingPolicyResponse:
    """Clone a routing policy into an inactive draft for safe editing."""
    source = await db.get(RoutingPolicy, policy_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing policy '{policy_id}' not found",
        )

    clone = RoutingPolicy(
        name=request.name or f"{source.name} draft",
        strategy=source.strategy,
        config_=dict(source.config_) if source.config_ else {},
        is_default=False,
        revision=1,
        status="draft",
    )
    db.add(clone)
    await db.flush()
    _record_policy_revision(
        db,
        clone,
        action="clone",
        change_note=request.change_note or f"Cloned from routing policy '{source.policy_id}'",
    )
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(clone)
    return RoutingPolicyResponse.from_model(clone)


@router.patch("/{policy_id}", dependencies=[Depends(verify_master_key)])
async def update_routing_policy(
    policy_id: str,
    request: UpdateRoutingPolicyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RoutingPolicyResponse:
    """Update a routing policy."""
    policy = await db.get(RoutingPolicy, policy_id)
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing policy '{policy_id}' not found",
        )

    payload = request.model_dump(exclude_unset=True)
    change_note = request.change_note
    next_status = str(payload["status"]) if "status" in payload and payload["status"] is not None else policy.status
    next_is_default = (
        bool(payload["is_default"])
        if "is_default" in payload and payload["is_default"] is not None
        else bool(policy.is_default)
    )
    _validate_status(next_status)
    _ensure_default_policy_is_active(policy_status=next_status, is_default=next_is_default)
    next_strategy, next_config = _update_policy_shape(policy, payload)
    if next_strategy is not None:
        _validate_strategy(next_strategy)
        policy.strategy = next_strategy
    if "name" in payload and payload["name"] is not None:
        policy.name = str(payload["name"])
    if next_config is not None:
        policy.config_ = next_config
    if "status" in payload and payload["status"] is not None:
        policy.status = str(payload["status"])
    if "is_default" in payload and payload["is_default"] is not None:
        policy.is_default = bool(payload["is_default"])
        if policy.is_default:
            await _unset_default_policies(
                db,
                except_policy_id=policy.policy_id,
                change_note=change_note or f"Unset default before promoting routing policy '{policy.policy_id}'",
            )
    _bump_policy_revision(policy)
    _record_policy_revision(
        db,
        policy,
        action="update",
        change_note=change_note,
    )

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(policy)
    return RoutingPolicyResponse.from_model(policy)


@router.delete(
    "/{policy_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_master_key)],
)
async def delete_routing_policy(
    policy_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    change_note: str | None = None,
) -> None:
    """Delete a routing policy."""
    policy = await db.get(RoutingPolicy, policy_id)
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing policy '{policy_id}' not found",
        )

    _bump_policy_revision(policy)
    _record_policy_revision(
        db,
        policy,
        action="delete",
        change_note=change_note,
    )
    await db.flush()
    await db.delete(policy)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
