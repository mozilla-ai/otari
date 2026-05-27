"""Database-backed routing policy resolution for standalone gateway mode."""

import copy
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any

import httpx
from any_llm import AnyLLM
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.log_config import logger
from gateway.models.entities import Project, RouteTrace, RoutingPolicy
from gateway.services.pricing_service import find_model_pricing

DEFAULT_ROUTING_MODEL = "default_routing"
DEFAULT_ROUTE_TRACE_ENDPOINT = "/v1/chat/completions"
ROUTING_STRATEGIES = {
    "single",
    "priority",
    "lowest_cost",
    "least_latency",
    "cost_tier",
    "intelligent",
    "weighted_score",
}
ACTIVE_ROUTING_POLICY_STATUS = "active"

_DEFAULT_OUTPUT_TOKENS = 700
_TIER_ORDER = ("simple", "medium", "complex", "reasoning")
_INFERRED_TIER_BY_OUTPUT_PRICE = (
    (0.10, "simple"),
    (1.50, "simple"),
    (2.00, "medium"),
    (5.00, "complex"),
)
_HEALTH_MODES = {"observe", "downrank", "skip_unhealthy"}
_HEALTH_RANK = {"healthy": 0, "unknown": 1, "degraded": 2, "unhealthy": 3}
_REASONING_HINTS = (
    "prove",
    "proof",
    "derive",
    "theorem",
    "formal",
    "optimization",
    "multi-step",
    "step by step",
)
_COMPLEX_HINTS = (
    "architecture",
    "design doc",
    "debug",
    "refactor",
    "implement",
    "migration",
    "security",
    "compliance",
    "analyze",
)
_MEDIUM_HINTS = (
    "summarize",
    "compare",
    "rewrite",
    "extract",
    "classify",
    "explain",
)
_GUARDRAIL_ACTIONS = {"block", "observe"}
_CONTEXT_STRATEGIES = {"trim_messages", "summarize_messages"}
_PROMPT_INJECTION_PHRASES = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard previous instructions",
    "reveal the system prompt",
    "show me the system prompt",
    "print the system prompt",
    "developer message",
    "jailbreak",
)
_PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}
_CREDENTIAL_BLOCKED_PATTERNS = (
    {"name": "openai_api_key", "pattern": r"\bsk-[A-Za-z0-9_-]{20,}\b"},
    {"name": "aws_access_key_id", "pattern": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"},
    {"name": "github_token", "pattern": r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"},
    {"name": "slack_token", "pattern": r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"},
    {"name": "private_key_block", "pattern": r"-----BEGIN [A-Z ]*PRIVATE KEY-----"},
)
_GUARDRAIL_PRESETS: dict[str, dict[str, Any]] = {
    "prompt_injection": {"prompt_injection": {"enabled": True}},
    "pii": {"pii": {"enabled": True}},
    "credential_leak": {"blocked_patterns": list(_CREDENTIAL_BLOCKED_PATTERNS)},
    "secrets": {"blocked_patterns": list(_CREDENTIAL_BLOCKED_PATTERNS)},
    "dlp": {
        "pii": {"enabled": True},
        "blocked_patterns": list(_CREDENTIAL_BLOCKED_PATTERNS),
    },
    "baseline": {
        "prompt_injection": {"enabled": True},
        "blocked_patterns": list(_CREDENTIAL_BLOCKED_PATTERNS),
    },
    "strict": {
        "pii": {"enabled": True},
        "prompt_injection": {"enabled": True},
        "blocked_patterns": list(_CREDENTIAL_BLOCKED_PATTERNS),
    },
}
_GUARDRAIL_PRESET_ALIASES = {
    "credentials": "credential_leak",
    "credential": "credential_leak",
    "secret": "secrets",
    "data_loss_prevention": "dlp",
    "data_loss": "dlp",
    "prompt_shield": "prompt_injection",
    "prompt_injection_detection": "prompt_injection",
}


def normalize_routing_model_selector(model: Any) -> str:
    """Normalize Merge-style routing sentinels while preserving direct model selectors."""
    if model is None:
        return DEFAULT_ROUTING_MODEL
    if not isinstance(model, str):
        return str(model)
    normalized = model.strip()
    if normalized.lower() == DEFAULT_ROUTING_MODEL:
        return DEFAULT_ROUTING_MODEL
    return normalized


class RoutingPolicyError(Exception):
    """Error that can be surfaced as an HTTP routing-policy failure."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class _CandidateSpec:
    model: str
    tier: str | None
    input_price_per_million: float | None
    output_price_per_million: float | None
    quality_score: float | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ProviderHealth:
    """Passive provider health summary derived from recent route traces."""

    provider: str
    status: str
    sample_count: int
    success_count: int
    error_count: int
    failure_rate: float | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe health representation."""
        return {
            "provider": self.provider,
            "status": self.status,
            "sample_count": self.sample_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "failure_rate": self.failure_rate,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RoutingCandidate:
    """Resolved candidate model with pricing and ordering metadata."""

    model: str
    provider: str
    provider_model: str
    position: int
    tier: str | None
    estimated_cost: float | None
    input_price_per_million: float | None
    output_price_per_million: float | None
    quality_score: float | None
    average_latency_ms: float | None
    latency_sample_count: int
    routing_score: float | None
    score_components: dict[str, float] | None
    provider_health: ProviderHealth | None
    metadata: dict[str, Any]

    def to_trace_dict(self) -> dict[str, Any]:
        """Return the JSON-safe trace representation."""
        return {
            "model": self.model,
            "provider": self.provider,
            "provider_model": self.provider_model,
            "position": self.position,
            "tier": self.tier,
            "estimated_cost": self.estimated_cost,
            "input_price_per_million": self.input_price_per_million,
            "output_price_per_million": self.output_price_per_million,
            "quality_score": self.quality_score,
            "average_latency_ms": self.average_latency_ms,
            "latency_sample_count": self.latency_sample_count,
            "routing_score": self.routing_score,
            "score_components": self.score_components,
            "provider_health": self.provider_health.to_dict() if self.provider_health else None,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RoutingPlan:
    """Resolved model-routing plan for one chat completion request."""

    requested_model: str
    project: Project | None
    policy: RoutingPolicy
    strategy: str
    target_tier: str
    prompt_tokens: int
    output_tokens: int
    candidates: list[RoutingCandidate]
    fallback_enabled: bool
    policy_source: str
    reason: str
    tags: dict[str, str]
    rejected_candidates: list[dict[str, Any]]
    policy_rollout: dict[str, Any] | None
    guardrails: dict[str, Any] | None
    context: dict[str, Any] | None

    @property
    def selected_candidate(self) -> RoutingCandidate:
        """Return the first candidate the gateway should try."""
        return self.candidates[0]

    @property
    def project_id(self) -> str | None:
        """Return the project id attached to the request."""
        return self.project.project_id if self.project is not None else None


@dataclass(frozen=True)
class _PolicyMatch:
    policy: RoutingPolicy
    source: str
    rollout: dict[str, Any] | None


def _jsonable_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_jsonable_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_jsonable_text(item) for item in value.values())
    if value is None:
        return ""
    return str(value)


def estimate_prompt_tokens(request_body: Mapping[str, Any]) -> int:
    """Estimate prompt tokens without pulling in a tokenizer dependency."""
    text_parts: list[str] = []
    messages = request_body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                text_parts.append(_jsonable_text(message.get("content")))

    tools = request_body.get("tools")
    if tools:
        try:
            text_parts.append(json.dumps(tools, sort_keys=True))
        except TypeError:
            text_parts.append(_jsonable_text(tools))

    joined = " ".join(part for part in text_parts if part)
    return max(1, len(joined) // 4)


def estimate_output_tokens(request_body: Mapping[str, Any]) -> int:
    """Estimate expected output tokens from request limits."""
    for key in ("max_completion_tokens", "max_tokens"):
        value = request_body.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return _DEFAULT_OUTPUT_TOKENS


def classify_request_tier(
    request_body: Mapping[str, Any],
    *,
    prompt_tokens: int,
    config: Mapping[str, Any],
) -> str:
    """Classify a request into a ClawSwitch-style complexity tier."""
    thresholds = config.get("tier_thresholds")
    threshold_map = thresholds if isinstance(thresholds, dict) else {}
    medium_threshold = _int_config(threshold_map.get("medium"), 800)
    complex_threshold = _int_config(threshold_map.get("complex"), 3000)
    reasoning_threshold = _int_config(threshold_map.get("reasoning"), 9000)

    request_text = _jsonable_text(request_body.get("messages")).lower()
    if prompt_tokens >= reasoning_threshold or any(hint in request_text for hint in _REASONING_HINTS):
        return "reasoning"
    if prompt_tokens >= complex_threshold or any(hint in request_text for hint in _COMPLEX_HINTS):
        return "complex"
    if prompt_tokens >= medium_threshold or any(hint in request_text for hint in _MEDIUM_HINTS):
        return "medium"
    return "simple"


def _int_config(value: Any, default: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    return default


def _non_negative_int_config(value: Any, default: int) -> int:
    if isinstance(value, int) and value >= 0:
        return value
    return default


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _non_negative_float_or_none(value: Any) -> float | None:
    parsed = _float_or_none(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def _score_or_none(value: Any) -> float | None:
    parsed = _non_negative_float_or_none(value)
    if parsed is None:
        return None
    if parsed <= 1.0:
        return parsed
    if parsed <= 100.0:
        return parsed / 100.0
    return None


def _bool_config(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _normalize_tier(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in _TIER_ORDER else None


def _string_set(value: Any) -> set[str]:
    if isinstance(value, str) and value.strip():
        return {value.strip()}
    if not isinstance(value, list):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _constraint_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    constraints = config.get("constraints")
    return constraints if isinstance(constraints, dict) else {}


def _normalize_model_key_for_constraint(value: str) -> str:
    try:
        _, _, normalized = split_model_selector(value)
    except ValueError:
        return value
    return normalized


def _constraint_model_set(value: Any) -> set[str]:
    return {_normalize_model_key_for_constraint(item) for item in _string_set(value)}


def _region_set(value: Any) -> set[str]:
    return {item.strip().lower() for item in _string_set(value) if item.strip()}


def _candidate_regions(candidate: RoutingCandidate) -> set[str]:
    regions = _region_set(candidate.metadata.get("regions"))
    region = candidate.metadata.get("region")
    if isinstance(region, str) and region.strip():
        regions.add(region.strip().lower())
    return regions


def _request_region(constraints: Mapping[str, Any], tags: Mapping[str, str]) -> str | None:
    region_tag = constraints.get("region_tag", "region")
    if not isinstance(region_tag, str) or not region_tag.strip():
        return None
    value = tags.get(region_tag.strip())
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


def _constraint_failure(
    candidate: RoutingCandidate,
    constraints: Mapping[str, Any],
    *,
    tags: Mapping[str, str],
) -> str | None:
    allowed_providers = _string_set(constraints.get("allowed_providers"))
    blocked_providers = _string_set(constraints.get("blocked_providers"))
    allowed_models = _constraint_model_set(constraints.get("allowed_models"))
    blocked_models = _constraint_model_set(constraints.get("blocked_models"))
    allowed_regions = _region_set(constraints.get("allowed_regions"))
    blocked_regions = _region_set(constraints.get("blocked_regions"))

    if allowed_providers and candidate.provider not in allowed_providers:
        return "provider_not_allowed"
    if candidate.provider in blocked_providers:
        return "provider_blocked"
    if allowed_models and candidate.model not in allowed_models:
        return "model_not_allowed"
    if candidate.model in blocked_models:
        return "model_blocked"

    candidate_regions = _candidate_regions(candidate)
    if allowed_regions:
        if not candidate_regions:
            return "region_unknown"
        if not candidate_regions & allowed_regions:
            return "region_not_allowed"
    if blocked_regions and candidate_regions & blocked_regions:
        return "region_blocked"

    if _bool_config(constraints.get("require_region_match"), False):
        requested_region = _request_region(constraints, tags)
        if requested_region is not None:
            if not candidate_regions:
                return "region_unknown"
            if requested_region not in candidate_regions:
                return "region_not_supported"

    max_estimated_cost = _non_negative_float_or_none(constraints.get("max_estimated_cost"))
    if max_estimated_cost is None:
        return None

    allow_unknown_cost = _bool_config(constraints.get("allow_unknown_cost"), False)
    if candidate.estimated_cost is None:
        return None if allow_unknown_cost else "estimated_cost_unknown"
    if candidate.estimated_cost > max_estimated_cost:
        return "estimated_cost_exceeds_max"
    return None


def _apply_constraints(
    candidates: Sequence[RoutingCandidate],
    *,
    config: Mapping[str, Any],
    tags: Mapping[str, str],
) -> tuple[list[RoutingCandidate], list[dict[str, Any]]]:
    constraints = _constraint_config(config)
    if not constraints:
        return list(candidates), []

    allowed: list[RoutingCandidate] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        reason = _constraint_failure(candidate, constraints, tags=tags)
        if reason is None:
            allowed.append(candidate)
            continue
        candidate_regions = sorted(_candidate_regions(candidate))
        rejected.append(
            {
                "model": candidate.model,
                "provider": candidate.provider,
                "reason": reason,
                "estimated_cost": candidate.estimated_cost,
                "regions": candidate_regions,
            }
        )
    return allowed, rejected


def _candidate_spec_from_item(item: Any, *, tier: str | None) -> _CandidateSpec | None:
    if isinstance(item, str):
        model = item.strip()
        if not model:
            return None
        return _CandidateSpec(
            model=model,
            tier=tier,
            input_price_per_million=None,
            output_price_per_million=None,
            quality_score=None,
            metadata={},
        )

    if not isinstance(item, dict):
        return None
    model_value = item.get("model")
    if not isinstance(model_value, str) or not model_value.strip():
        return None

    metadata_value = item.get("metadata")
    metadata = dict(metadata_value) if isinstance(metadata_value, dict) else {}
    for key in ("region", "regions"):
        if key in item and key not in metadata:
            metadata[key] = item[key]
    quality_score = _score_or_none(item.get("quality_score"))
    if quality_score is None:
        for key in ("benchmark_score", "score", "intelligence_score"):
            quality_score = _score_or_none(item.get(key))
            if quality_score is not None:
                break
    if quality_score is None:
        for key in ("quality_score", "benchmark_score", "score", "intelligence_score"):
            quality_score = _score_or_none(metadata.get(key))
            if quality_score is not None:
                break
    return _CandidateSpec(
        model=model_value.strip(),
        tier=_normalize_tier(item.get("tier")) or tier,
        input_price_per_million=_float_or_none(item.get("input_price_per_million")),
        output_price_per_million=_float_or_none(item.get("output_price_per_million")),
        quality_score=quality_score,
        metadata=metadata,
    )


def _infer_tier_from_output_price(output_price_per_million: float | None) -> str | None:
    """Infer an internal complexity tier from output-token pricing."""
    if output_price_per_million is None:
        return None
    for max_output_price, tier in _INFERRED_TIER_BY_OUTPUT_PRICE:
        if output_price_per_million < max_output_price:
            return tier
    return "reasoning"


def _configured_candidate_specs(config: Mapping[str, Any]) -> list[_CandidateSpec]:
    specs: list[_CandidateSpec] = []

    candidates = config.get("candidates")
    if isinstance(candidates, list):
        for item in candidates:
            spec = _candidate_spec_from_item(item, tier=None)
            if spec is not None:
                specs.append(spec)

    tiers = config.get("tiers")
    if isinstance(tiers, dict):
        for tier_name, items in tiers.items():
            tier = _normalize_tier(tier_name)
            if tier is None or not isinstance(items, list):
                continue
            for item in items:
                spec = _candidate_spec_from_item(item, tier=tier)
                if spec is not None:
                    specs.append(spec)

    deduped: list[_CandidateSpec] = []
    seen: set[str] = set()
    for spec in specs:
        provider, provider_model, normalized = split_model_selector(spec.model)
        dedupe_key = f"{provider}:{provider_model}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(
            _CandidateSpec(
                model=normalized,
                tier=spec.tier,
                input_price_per_million=spec.input_price_per_million,
                output_price_per_million=spec.output_price_per_million,
                quality_score=spec.quality_score,
                metadata=spec.metadata,
            )
        )
    return deduped


def split_model_selector(model_selector: str) -> tuple[str, str, str]:
    """Split and normalize a provider:model selector."""
    provider, model_name = AnyLLM.split_model_provider(model_selector)
    provider_name = provider.value
    return provider_name, model_name, f"{provider_name}:{model_name}"


async def _build_candidates(
    db: AsyncSession,
    specs: Sequence[_CandidateSpec],
    *,
    prompt_tokens: int,
    output_tokens: int,
) -> list[RoutingCandidate]:
    candidates: list[RoutingCandidate] = []
    for index, spec in enumerate(specs, start=1):
        provider, provider_model, normalized_model = split_model_selector(spec.model)
        pricing = await find_model_pricing(db, provider, provider_model)
        input_price = spec.input_price_per_million
        output_price = spec.output_price_per_million
        if pricing is not None:
            if input_price is None:
                input_price = pricing.input_price_per_million
            if output_price is None:
                output_price = pricing.output_price_per_million

        estimated_cost: float | None = None
        if input_price is not None and output_price is not None:
            estimated_cost = (prompt_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price

        candidates.append(
            RoutingCandidate(
                model=normalized_model,
                provider=provider,
                provider_model=provider_model,
                position=index,
                tier=spec.tier or _infer_tier_from_output_price(output_price),
                estimated_cost=estimated_cost,
                input_price_per_million=input_price,
                output_price_per_million=output_price,
                quality_score=spec.quality_score,
                average_latency_ms=None,
                latency_sample_count=0,
                routing_score=None,
                score_components=None,
                provider_health=None,
                metadata=spec.metadata,
            )
        )
    return candidates


def _by_cost(candidate: RoutingCandidate) -> tuple[bool, float, int]:
    return (candidate.estimated_cost is None, candidate.estimated_cost or 0.0, candidate.position)


def _by_latency(candidate: RoutingCandidate) -> tuple[bool, float, int]:
    return (candidate.average_latency_ms is None, candidate.average_latency_ms or 0.0, candidate.position)


def _by_weighted_score(candidate: RoutingCandidate) -> tuple[bool, float, int]:
    return (candidate.routing_score is None, -(candidate.routing_score or 0.0), candidate.position)


def _by_health(candidate: RoutingCandidate) -> tuple[int, int]:
    status_value = candidate.provider_health.status if candidate.provider_health else "unknown"
    return (_HEALTH_RANK.get(status_value, _HEALTH_RANK["unknown"]), candidate.position)


def _weighted_scoring_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    scoring = config.get("scoring")
    if isinstance(scoring, dict):
        return scoring
    score_weights = config.get("score_weights")
    return score_weights if isinstance(score_weights, dict) else {}


def _score_weight(scoring: Mapping[str, Any], key: str, default: float) -> float:
    weights = scoring.get("weights")
    weight_value: Any = None
    if isinstance(weights, dict):
        weight_value = weights.get(key, weights.get(f"{key}_weight"))
    if weight_value is None:
        weight_value = scoring.get(f"{key}_weight", scoring.get(key))
    parsed = _non_negative_float_or_none(weight_value)
    return default if parsed is None else parsed


def _weighted_score_weights(config: Mapping[str, Any]) -> dict[str, float]:
    scoring = _weighted_scoring_config(config)
    weights = {
        "quality": _score_weight(scoring, "quality", 0.5),
        "cost": _score_weight(scoring, "cost", 0.3),
        "latency": _score_weight(scoring, "latency", 0.2),
    }
    if sum(weights.values()) <= 0:
        return {"quality": 0.5, "cost": 0.3, "latency": 0.2}
    return weights


def _normalized_lower_is_better(
    value: float | None,
    known_values: Sequence[float],
    *,
    unknown_score: float,
) -> float:
    if value is None:
        return unknown_score
    if not known_values:
        return 1.0
    minimum = min(known_values)
    maximum = max(known_values)
    if maximum == minimum:
        return 1.0
    return max(0.0, min(1.0, 1.0 - ((value - minimum) / (maximum - minimum))))


def _attach_weighted_scores(
    candidates: Sequence[RoutingCandidate],
    *,
    config: Mapping[str, Any],
) -> list[RoutingCandidate]:
    scoring = _weighted_scoring_config(config)
    weights = _weighted_score_weights(config)
    weight_total = sum(weights.values())
    default_quality = _score_or_none(scoring.get("default_quality_score"))
    unknown_cost = _score_or_none(scoring.get("unknown_cost_score"))
    unknown_latency = _score_or_none(scoring.get("unknown_latency_score"))
    default_quality_score = 0.5 if default_quality is None else default_quality
    unknown_cost_score = 0.0 if unknown_cost is None else unknown_cost
    unknown_latency_score = 0.5 if unknown_latency is None else unknown_latency

    known_costs = [
        candidate.estimated_cost
        for candidate in candidates
        if candidate.estimated_cost is not None
    ]
    known_latencies = [
        candidate.average_latency_ms
        for candidate in candidates
        if candidate.average_latency_ms is not None
    ]

    scored: list[RoutingCandidate] = []
    for candidate in candidates:
        quality_score = candidate.quality_score if candidate.quality_score is not None else default_quality_score
        cost_score = _normalized_lower_is_better(
            candidate.estimated_cost,
            known_costs,
            unknown_score=unknown_cost_score,
        )
        latency_score = _normalized_lower_is_better(
            candidate.average_latency_ms,
            known_latencies,
            unknown_score=unknown_latency_score,
        )
        routing_score = (
            (quality_score * weights["quality"])
            + (cost_score * weights["cost"])
            + (latency_score * weights["latency"])
        ) / weight_total
        scored.append(
            replace(
                candidate,
                routing_score=routing_score,
                score_components={
                    "quality": quality_score,
                    "cost": cost_score,
                    "latency": latency_score,
                    "quality_weight": weights["quality"],
                    "cost_weight": weights["cost"],
                    "latency_weight": weights["latency"],
                },
            )
        )
    return scored


def _tier_fallback_order(target_tier: str) -> list[str]:
    target_index = _TIER_ORDER.index(target_tier)
    return [*_TIER_ORDER[target_index:], *reversed(_TIER_ORDER[:target_index])]


def _order_candidates(
    candidates: Sequence[RoutingCandidate],
    *,
    strategy: str,
    target_tier: str,
) -> list[RoutingCandidate]:
    if strategy in {"single", "priority"}:
        return list(candidates)
    if strategy == "lowest_cost":
        return sorted(candidates, key=_by_cost)
    if strategy == "least_latency":
        return sorted(candidates, key=_by_latency)
    if strategy == "weighted_score":
        return sorted(candidates, key=_by_weighted_score)

    ordered: list[RoutingCandidate] = []
    consumed: set[str] = set()
    for tier in _tier_fallback_order(target_tier):
        tier_candidates = [candidate for candidate in candidates if candidate.tier == tier]
        for candidate in sorted(tier_candidates, key=_by_cost):
            ordered.append(candidate)
            consumed.add(candidate.model)

    remaining = [candidate for candidate in candidates if candidate.model not in consumed]
    ordered.extend(sorted(remaining, key=_by_cost))
    return ordered


def _fallback_enabled(config: Mapping[str, Any], *, strategy: str) -> bool:
    if strategy == "single":
        return False
    return _bool_config(config.get("fallback_enabled"), True)


def _attempt_model_key(attempt: Mapping[str, Any]) -> str | None:
    model_key = attempt.get("model_key")
    if isinstance(model_key, str) and model_key:
        return model_key

    provider = attempt.get("provider")
    model = attempt.get("model")
    if isinstance(provider, str) and provider and isinstance(model, str) and model:
        return f"{provider}:{model}"
    return None


def _attempt_duration_ms(attempt: Mapping[str, Any]) -> float | None:
    duration = attempt.get("duration_ms")
    if isinstance(duration, bool):
        return None
    if isinstance(duration, int | float) and duration >= 0:
        return float(duration)
    return None


def _attempt_provider(attempt: Mapping[str, Any]) -> str | None:
    provider = attempt.get("provider")
    return provider if isinstance(provider, str) and provider else None


def _attempt_outcome(attempt: Mapping[str, Any]) -> str | None:
    status = attempt.get("status")
    if status in {"success", "error"}:
        return str(status)
    return None


async def _attach_latency_stats(
    db: AsyncSession,
    candidates: Sequence[RoutingCandidate],
    *,
    config: Mapping[str, Any],
) -> list[RoutingCandidate]:
    candidate_models = {candidate.model for candidate in candidates}
    if not candidate_models:
        return list(candidates)

    sample_limit = _int_config(config.get("latency_sample_limit"), 200)
    min_samples = _int_config(config.get("latency_min_samples"), 1)
    result = await db.execute(
        select(RouteTrace)
        .where(RouteTrace.status == "success")
        .order_by(RouteTrace.timestamp.desc())
        .limit(sample_limit)
    )
    durations_by_model: dict[str, list[float]] = {model: [] for model in candidate_models}
    for trace in result.scalars().all():
        attempts = trace.attempts if isinstance(trace.attempts, list) else []
        for attempt in attempts:
            if not isinstance(attempt, dict) or attempt.get("status") != "success":
                continue
            model_key = _attempt_model_key(attempt)
            duration_ms = _attempt_duration_ms(attempt)
            if model_key in durations_by_model and duration_ms is not None:
                durations_by_model[model_key].append(duration_ms)

    enriched: list[RoutingCandidate] = []
    for candidate in candidates:
        durations = durations_by_model.get(candidate.model, [])
        if len(durations) < min_samples:
            enriched.append(candidate)
            continue
        enriched.append(
            replace(
                candidate,
                average_latency_ms=sum(durations) / len(durations),
                latency_sample_count=len(durations),
            )
        )
    return enriched


def _provider_health_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    health = config.get("health")
    return health if isinstance(health, dict) else {}


def _provider_health_enabled(config: Mapping[str, Any]) -> bool:
    return _bool_config(_provider_health_config(config).get("enabled"), False)


def _provider_health_mode(config: Mapping[str, Any]) -> str:
    mode = _provider_health_config(config).get("mode")
    return mode if isinstance(mode, str) and mode in _HEALTH_MODES else "downrank"


def _provider_health_rate(config: Mapping[str, Any], key: str, default: float) -> float:
    rate = _non_negative_float_or_none(_provider_health_config(config).get(key))
    if rate is None:
        return default
    return min(rate, 1.0)


def _provider_health_from_counts(
    provider: str,
    *,
    success_count: int,
    error_count: int,
    config: Mapping[str, Any],
) -> ProviderHealth:
    sample_count = success_count + error_count
    min_samples = _int_config(_provider_health_config(config).get("min_samples"), 3)
    degraded_rate = _provider_health_rate(config, "degraded_failure_rate", 0.25)
    unhealthy_rate = _provider_health_rate(config, "unhealthy_failure_rate", 0.50)
    failure_rate = None if sample_count == 0 else error_count / sample_count

    if sample_count < min_samples or failure_rate is None:
        return ProviderHealth(
            provider=provider,
            status="unknown",
            sample_count=sample_count,
            success_count=success_count,
            error_count=error_count,
            failure_rate=failure_rate,
            reason="insufficient_samples",
        )
    if failure_rate >= unhealthy_rate:
        return ProviderHealth(
            provider=provider,
            status="unhealthy",
            sample_count=sample_count,
            success_count=success_count,
            error_count=error_count,
            failure_rate=failure_rate,
            reason="failure_rate_exceeds_unhealthy_threshold",
        )
    if failure_rate >= degraded_rate:
        return ProviderHealth(
            provider=provider,
            status="degraded",
            sample_count=sample_count,
            success_count=success_count,
            error_count=error_count,
            failure_rate=failure_rate,
            reason="failure_rate_exceeds_degraded_threshold",
        )
    return ProviderHealth(
        provider=provider,
        status="healthy",
        sample_count=sample_count,
        success_count=success_count,
        error_count=error_count,
        failure_rate=failure_rate,
        reason="failure_rate_below_threshold",
    )


async def _attach_provider_health(
    db: AsyncSession,
    candidates: Sequence[RoutingCandidate],
    *,
    config: Mapping[str, Any],
) -> list[RoutingCandidate]:
    if not _provider_health_enabled(config):
        return list(candidates)

    candidate_providers = {candidate.provider for candidate in candidates}
    if not candidate_providers:
        return list(candidates)

    sample_limit = _int_config(_provider_health_config(config).get("sample_limit"), 200)
    counts_by_provider = {
        provider: {"success": 0, "error": 0}
        for provider in candidate_providers
    }
    result = await db.execute(
        select(RouteTrace)
        .order_by(RouteTrace.timestamp.desc())
        .limit(sample_limit)
    )
    for trace in result.scalars().all():
        attempts = trace.attempts if isinstance(trace.attempts, list) else []
        if attempts:
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue
                provider = _attempt_provider(attempt)
                outcome = _attempt_outcome(attempt)
                if provider in counts_by_provider and outcome is not None:
                    counts_by_provider[provider][outcome] += 1
            continue

        if trace.selected_provider in counts_by_provider and trace.status in {"success", "error"}:
            counts_by_provider[trace.selected_provider][trace.status] += 1

    health_by_provider = {
        provider: _provider_health_from_counts(
            provider,
            success_count=counts["success"],
            error_count=counts["error"],
            config=config,
        )
        for provider, counts in counts_by_provider.items()
    }
    return [
        replace(candidate, provider_health=health_by_provider.get(candidate.provider))
        for candidate in candidates
    ]


def _apply_provider_health_gate(
    candidates: Sequence[RoutingCandidate],
    *,
    config: Mapping[str, Any],
) -> tuple[list[RoutingCandidate], list[dict[str, Any]]]:
    if not _provider_health_enabled(config) or _provider_health_mode(config) != "skip_unhealthy":
        return list(candidates), []

    allowed: list[RoutingCandidate] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.provider_health is None or candidate.provider_health.status != "unhealthy":
            allowed.append(candidate)
            continue
        rejected.append(
            {
                "model": candidate.model,
                "provider": candidate.provider,
                "reason": "provider_unhealthy",
                "estimated_cost": candidate.estimated_cost,
                "provider_health": candidate.provider_health.to_dict(),
            }
        )
    return allowed, rejected


def _apply_provider_health_order(
    candidates: Sequence[RoutingCandidate],
    *,
    config: Mapping[str, Any],
) -> list[RoutingCandidate]:
    if not _provider_health_enabled(config) or _provider_health_mode(config) != "downrank":
        return list(candidates)
    return sorted(candidates, key=_by_health)


def _policy_match_tags(config: Mapping[str, Any]) -> dict[str, str]:
    match_config = config.get("match")
    if not isinstance(match_config, dict):
        return {}
    tags = match_config.get("tags")
    if not isinstance(tags, dict):
        return {}
    return {str(key): str(value) for key, value in tags.items()}


def _policy_match_priority(config: Mapping[str, Any]) -> int:
    match_config = config.get("match")
    if not isinstance(match_config, dict):
        return 0
    priority = match_config.get("priority")
    if isinstance(priority, int) and not isinstance(priority, bool):
        return priority
    return 0


def _policy_match_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    match_config = config.get("match")
    return match_config if isinstance(match_config, dict) else {}


def _policy_match_rollout_percentage(config: Mapping[str, Any]) -> float:
    match_config = _policy_match_config(config)
    value = match_config.get("rollout_percentage", match_config.get("percentage"))
    if isinstance(value, bool):
        return 100.0
    if isinstance(value, int | float):
        return min(max(float(value), 0.0), 100.0)
    return 100.0


def _policy_match_bucket_key(config: Mapping[str, Any], request_tags: Mapping[str, str]) -> str:
    match_config = _policy_match_config(config)
    bucket_by = match_config.get("bucket_by")
    if isinstance(bucket_by, str) and bucket_by in request_tags:
        return f"{bucket_by}:{request_tags[bucket_by]}"
    if request_tags:
        return json.dumps(sorted(request_tags.items()), separators=(",", ":"))
    return "__empty__"


def _policy_match_bucket(config: Mapping[str, Any], policy_id: str, request_tags: Mapping[str, str]) -> float:
    match_config = _policy_match_config(config)
    salt = match_config.get("salt")
    salt_value = salt if isinstance(salt, str) else policy_id
    bucket_key = _policy_match_bucket_key(config, request_tags)
    digest = sha256(f"{salt_value}:{policy_id}:{bucket_key}".encode()).hexdigest()
    return (int(digest[:8], 16) % 10_000) / 100


def _policy_rollout_info(policy: RoutingPolicy, request_tags: Mapping[str, str]) -> dict[str, Any] | None:
    config = policy.config_ or {}
    match_config = _policy_match_config(config)
    if "rollout_percentage" not in match_config and "percentage" not in match_config:
        return None
    percentage = _policy_match_rollout_percentage(config)
    bucket_key = _policy_match_bucket_key(config, request_tags)
    bucket = _policy_match_bucket(config, policy.policy_id, request_tags)
    return {
        "percentage": percentage,
        "bucket": bucket,
        "bucket_key": bucket_key,
        "matched": bucket < percentage,
    }


def _condition_tag_key(condition: Mapping[str, Any]) -> str | None:
    for key in ("tag", "key", "field", "name"):
        value = condition.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _tag_values(value: Any) -> set[str]:
    if isinstance(value, list | tuple | set):
        return {str(item) for item in value}
    return {str(value)}


def _evaluate_tag_condition(condition: Mapping[str, Any], request_tags: Mapping[str, str]) -> bool:
    tag_key = _condition_tag_key(condition)
    if tag_key is None:
        return False
    operator = condition.get("operator", condition.get("op", "eq"))
    op = str(operator).strip().lower()
    expected = condition.get("value")
    exists = tag_key in request_tags
    actual = request_tags.get(tag_key)

    if op == "exists":
        expected_exists = expected if isinstance(expected, bool) else True
        return exists is expected_exists
    if actual is None:
        return False

    actual_value = str(actual)
    if op in {"eq", "equals", "=="}:
        return actual_value == str(expected)
    if op in {"ne", "neq", "not_eq", "!="}:
        return actual_value != str(expected)
    if op == "in":
        return actual_value in _tag_values(expected)
    if op in {"not_in", "nin"}:
        return actual_value not in _tag_values(expected)
    if op == "contains":
        return str(expected) in actual_value
    if op == "starts_with":
        return actual_value.startswith(str(expected))
    if op == "ends_with":
        return actual_value.endswith(str(expected))

    actual_number = _numeric_value(actual_value)
    expected_number = _numeric_value(expected)
    if actual_number is None or expected_number is None:
        return False
    if op == "gt":
        return actual_number > expected_number
    if op in {"gte", "ge"}:
        return actual_number >= expected_number
    if op == "lt":
        return actual_number < expected_number
    if op in {"lte", "le"}:
        return actual_number <= expected_number
    return False


def _evaluate_condition_group(
    conditions: Any,
    request_tags: Mapping[str, str],
    *,
    logic: str,
) -> bool:
    if not isinstance(conditions, list) or not conditions:
        return False
    results = [_matches_tag_condition(condition, request_tags) for condition in conditions]
    return any(results) if logic == "or" else all(results)


def _matches_tag_condition(condition: Any, request_tags: Mapping[str, str]) -> bool:
    if not isinstance(condition, dict):
        return False
    for key, logic in (("any", "or"), ("or", "or"), ("all", "and"), ("and", "and")):
        if key in condition:
            return _evaluate_condition_group(condition[key], request_tags, logic=logic)
    return _evaluate_tag_condition(condition, request_tags)


def _matches_condition_config(match_config: Mapping[str, Any], request_tags: Mapping[str, str]) -> bool:
    if "any" in match_config:
        return _evaluate_condition_group(match_config["any"], request_tags, logic="or")
    if "or" in match_config:
        return _evaluate_condition_group(match_config["or"], request_tags, logic="or")
    if "all" in match_config:
        return _evaluate_condition_group(match_config["all"], request_tags, logic="and")
    if "and" in match_config:
        return _evaluate_condition_group(match_config["and"], request_tags, logic="and")
    conditions = match_config.get("conditions")
    if "conditions" not in match_config:
        return False
    logic_value = match_config.get("logic", "and")
    logic = str(logic_value).strip().lower()
    return _evaluate_condition_group(conditions, request_tags, logic="or" if logic in {"or", "any"} else "and")


def _matches_request_tags(policy_tags: Mapping[str, str], request_tags: Mapping[str, str]) -> bool:
    if not policy_tags:
        return False
    return all(request_tags.get(key) == value for key, value in policy_tags.items())


def _matches_policy_match_config(config: Mapping[str, Any], request_tags: Mapping[str, str]) -> bool:
    match_config = _policy_match_config(config)
    legacy_tags = _policy_match_tags(config)
    has_conditions = any(key in match_config for key in ("conditions", "all", "any", "and", "or"))
    if not legacy_tags and not has_conditions:
        return False
    if legacy_tags and not _matches_request_tags(legacy_tags, request_tags):
        return False
    if has_conditions and not _matches_condition_config(match_config, request_tags):
        return False
    return True


def _guardrails_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    guardrails = config.get("guardrails")
    return guardrails if isinstance(guardrails, dict) else {}


def _normalize_guardrail_preset_name(value: Any) -> str | None:
    name: Any = value
    if isinstance(value, dict):
        name = value.get("name") or value.get("preset")
    if not isinstance(name, str) or not name.strip():
        return None
    normalized = name.strip().lower().replace("-", "_")
    return _GUARDRAIL_PRESET_ALIASES.get(normalized, normalized)


def _guardrail_preset_values(guardrails: Mapping[str, Any]) -> list[Any]:
    presets = guardrails.get("presets")
    if presets is None:
        presets = guardrails.get("managed_presets")
    if isinstance(presets, list):
        return presets
    if isinstance(presets, str) and presets.strip():
        return [presets]
    return []


def _merge_guardrail_list(existing: Any, incoming: Any) -> list[Any]:
    values: list[Any] = []
    seen: set[str] = set()
    for item in [*_guardrail_list_items(existing), *_guardrail_list_items(incoming)]:
        try:
            key = json.dumps(item, sort_keys=True)
        except TypeError:
            key = str(item)
        if key in seen:
            continue
        seen.add(key)
        values.append(copy.deepcopy(item))
    return values


def _guardrail_list_items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _merge_guardrail_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in {"blocked_terms", "blocked_patterns", "external_classifiers"}:
            merged[key] = _merge_guardrail_list(merged.get(key), value)
            continue
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            nested = dict(merged[key])
            nested.update(copy.deepcopy(value))
            merged[key] = nested
            continue
        merged[key] = copy.deepcopy(value)
    return merged


def _guardrail_preset_expansion(
    guardrails: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[str]] | None]:
    preset_config: dict[str, Any] = {}
    applied: list[str] = []
    ignored: list[str] = []
    seen: set[str] = set()
    for value in _guardrail_preset_values(guardrails):
        normalized = _normalize_guardrail_preset_name(value)
        if normalized is None:
            continue
        if normalized not in _GUARDRAIL_PRESETS:
            ignored.append(normalized)
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        applied.append(normalized)
        preset_config = _merge_guardrail_config(preset_config, _GUARDRAIL_PRESETS[normalized])

    if not applied and not ignored:
        return preset_config, None
    metadata = {"applied": applied, "ignored": ignored}
    return preset_config, metadata


def _effective_guardrails_config(
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, list[str]] | None]:
    guardrails = _guardrails_config(config)
    preset_config, preset_metadata = _guardrail_preset_expansion(guardrails)
    if not preset_config:
        return guardrails, preset_metadata
    return _merge_guardrail_config(preset_config, guardrails), preset_metadata


def _redactions_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    redactions = _guardrails_config(config).get("redactions")
    return redactions if isinstance(redactions, dict) else {}


def _redactions_enabled(config: Mapping[str, Any]) -> bool:
    redactions = _redactions_config(config)
    return _bool_config(redactions.get("enabled"), bool(redactions))


def _context_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    context = config.get("context")
    if not isinstance(context, dict):
        context = config.get("context_policy")
    return context if isinstance(context, dict) else {}


def _context_enabled(config: Mapping[str, Any]) -> bool:
    context = _context_config(config)
    return _bool_config(context.get("enabled"), bool(context))


def _message_role(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    role = message.get("role")
    return role.strip().lower() if isinstance(role, str) else ""


def _message_token_estimate(message: Any) -> int:
    if not isinstance(message, dict):
        return max(1, len(_jsonable_text(message)) // 4)
    text = _jsonable_text(message.get("content"))
    return max(1, len(text) // 4)


def _non_message_prompt_tokens(request_body: Mapping[str, Any]) -> int:
    tools = request_body.get("tools")
    if not tools:
        return 0
    try:
        text = json.dumps(tools, sort_keys=True)
    except TypeError:
        text = _jsonable_text(tools)
    return max(1, len(text) // 4)


def _context_kept_message_indexes(
    messages: Sequence[Any],
    *,
    preserve_system_messages: bool,
    preserve_last_messages: int,
) -> set[int]:
    kept_indexes: set[int] = set()
    if preserve_system_messages:
        for index, message in enumerate(messages):
            if _message_role(message) in {"system", "developer"}:
                kept_indexes.add(index)

    if preserve_last_messages:
        for index in range(max(0, len(messages) - preserve_last_messages), len(messages)):
            kept_indexes.add(index)
    return kept_indexes


def _context_summary_role(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"system", "developer", "user"}:
            return normalized
    return "system"


def _summary_line_for_message(message: Any, *, max_chars: int) -> str:
    role = _message_role(message) or "message"
    text = _jsonable_text(message.get("content") if isinstance(message, dict) else message)
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = f"{text[: max(0, max_chars - 3)].rstrip()}..."
    return f"- {role}: {text}" if text else f"- {role}: [empty]"


def _trim_summary_to_chars(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return ""
    return f"{text[: max_chars - 3].rstrip()}..."


def _build_context_summary(
    messages: Sequence[Any],
    summarized_indexes: Sequence[int],
    *,
    max_chars: int,
    prefix: str,
    max_message_chars: int,
) -> str:
    if max_chars <= 0:
        return ""
    lines = [prefix.strip() or "Earlier conversation summary:"]
    for index in summarized_indexes:
        lines.append(_summary_line_for_message(messages[index], max_chars=max_message_chars))
    return _trim_summary_to_chars("\n".join(lines), max_chars)


def _messages_with_summary(
    messages: Sequence[Any],
    kept_indexes: set[int],
    *,
    summary_message: dict[str, Any] | None,
) -> list[Any]:
    result: list[Any] = []
    summary_inserted = summary_message is None
    for index, message in enumerate(messages):
        if index not in kept_indexes:
            continue
        if not summary_inserted and _message_role(message) not in {"system", "developer"}:
            result.append(summary_message)
            summary_inserted = True
        result.append(message)
    if not summary_inserted and summary_message is not None:
        result.append(summary_message)
    return result


def apply_context_policy(
    config: Mapping[str, Any],
    request_body: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply deterministic prompt compression from policy config and return trace metadata."""
    body = copy.deepcopy(dict(request_body))
    if not _context_enabled(config):
        return body, None

    context_config = _context_config(config)
    strategy = context_config.get("strategy", "trim_messages")
    if not isinstance(strategy, str) or strategy.strip().lower() not in _CONTEXT_STRATEGIES:
        return body, {
            "enabled": True,
            "status": "skipped",
            "reason": "unsupported_strategy",
            "strategy": str(strategy),
        }
    strategy = strategy.strip().lower()

    max_prompt_tokens = _int_config(context_config.get("max_prompt_tokens"), 0)
    if max_prompt_tokens <= 0:
        return body, {
            "enabled": True,
            "status": "skipped",
            "reason": "missing_max_prompt_tokens",
            "strategy": strategy,
        }

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return body, {
            "enabled": True,
            "status": "skipped",
            "reason": "missing_messages",
            "strategy": strategy,
            "max_prompt_tokens": max_prompt_tokens,
        }

    original_prompt_tokens = estimate_prompt_tokens(body)
    original_message_count = len(messages)
    preserve_system_messages = _bool_config(context_config.get("preserve_system_messages"), True)
    preserve_last_messages = _non_negative_int_config(context_config.get("preserve_last_messages"), 4)
    if original_prompt_tokens <= max_prompt_tokens:
        return body, {
            "enabled": True,
            "status": "unchanged",
            "strategy": strategy,
            "max_prompt_tokens": max_prompt_tokens,
            "original_prompt_tokens": original_prompt_tokens,
            "final_prompt_tokens": original_prompt_tokens,
            "original_message_count": original_message_count,
            "final_message_count": original_message_count,
            "trimmed_message_count": 0,
            "preserve_system_messages": preserve_system_messages,
            "preserve_last_messages": preserve_last_messages,
        }

    kept_indexes = _context_kept_message_indexes(
        messages,
        preserve_system_messages=preserve_system_messages,
        preserve_last_messages=preserve_last_messages,
    )

    if strategy == "summarize_messages":
        summarized_indexes = [index for index in range(len(messages)) if index not in kept_indexes]
        if not summarized_indexes:
            body["messages"] = [messages[index] for index in sorted(kept_indexes)]
            final_prompt_tokens = estimate_prompt_tokens(body)
            return body, {
                "enabled": True,
                "status": "unchanged",
                "strategy": strategy,
                "max_prompt_tokens": max_prompt_tokens,
                "original_prompt_tokens": original_prompt_tokens,
                "final_prompt_tokens": final_prompt_tokens,
                "original_message_count": original_message_count,
                "final_message_count": len(body["messages"]),
                "summarized_message_count": 0,
                "preserve_system_messages": preserve_system_messages,
                "preserve_last_messages": preserve_last_messages,
            }

        kept_tokens = _non_message_prompt_tokens(body) + sum(
            _message_token_estimate(messages[index]) for index in kept_indexes
        )
        summary_token_budget = max_prompt_tokens - kept_tokens
        configured_summary_tokens = _int_config(
            context_config.get("summary_max_tokens"),
            min(512, max(1, max_prompt_tokens // 4)),
        )
        summary_token_budget = min(configured_summary_tokens, summary_token_budget)
        summary_text = _build_context_summary(
            messages,
            summarized_indexes,
            max_chars=max(0, summary_token_budget * 4),
            prefix=str(context_config.get("summary_prefix") or "Earlier conversation summary:"),
            max_message_chars=_int_config(context_config.get("summary_message_max_chars"), 240),
        )
        summary_role = _context_summary_role(context_config.get("summary_role"))
        summary_message = {"role": summary_role, "content": summary_text} if summary_text else None
        body["messages"] = _messages_with_summary(messages, kept_indexes, summary_message=summary_message)
        final_prompt_tokens = estimate_prompt_tokens(body)
        if summary_message is not None and final_prompt_tokens > max_prompt_tokens:
            overflow_chars = ((final_prompt_tokens - max_prompt_tokens) * 4) + 16
            summary_message["content"] = _trim_summary_to_chars(
                str(summary_message["content"]),
                max(0, len(str(summary_message["content"])) - overflow_chars),
            )
            if not summary_message["content"]:
                summary_message = None
            body["messages"] = _messages_with_summary(messages, kept_indexes, summary_message=summary_message)
            final_prompt_tokens = estimate_prompt_tokens(body)

        return body, {
            "enabled": True,
            "status": "summarized" if summary_message is not None else "trimmed",
            "strategy": strategy,
            "max_prompt_tokens": max_prompt_tokens,
            "original_prompt_tokens": original_prompt_tokens,
            "final_prompt_tokens": final_prompt_tokens,
            "original_message_count": original_message_count,
            "final_message_count": len(body["messages"]),
            "trimmed_message_count": original_message_count - len(body["messages"]),
            "summarized_message_count": len(summarized_indexes),
            "summary_message_role": summary_role,
            "summary_chars": len(str(summary_message["content"])) if summary_message is not None else 0,
            "preserve_system_messages": preserve_system_messages,
            "preserve_last_messages": preserve_last_messages,
            "over_budget_after_summarization": final_prompt_tokens > max_prompt_tokens,
        }

    message_budget = max(max_prompt_tokens - _non_message_prompt_tokens(body), 1)
    current_message_tokens = sum(_message_token_estimate(messages[index]) for index in kept_indexes)
    for index in range(len(messages) - 1, -1, -1):
        if index in kept_indexes:
            continue
        token_estimate = _message_token_estimate(messages[index])
        if current_message_tokens + token_estimate <= message_budget:
            kept_indexes.add(index)
            current_message_tokens += token_estimate

    body["messages"] = [messages[index] for index in sorted(kept_indexes)]
    final_prompt_tokens = estimate_prompt_tokens(body)
    trimmed_message_count = original_message_count - len(body["messages"])
    return body, {
        "enabled": True,
        "status": "trimmed" if trimmed_message_count else "unchanged",
        "strategy": strategy,
        "max_prompt_tokens": max_prompt_tokens,
        "original_prompt_tokens": original_prompt_tokens,
        "final_prompt_tokens": final_prompt_tokens,
        "original_message_count": original_message_count,
        "final_message_count": len(body["messages"]),
        "trimmed_message_count": trimmed_message_count,
        "preserve_system_messages": preserve_system_messages,
        "preserve_last_messages": preserve_last_messages,
        "over_budget_after_trimming": final_prompt_tokens > max_prompt_tokens,
    }


def _guardrails_enabled(config: Mapping[str, Any]) -> bool:
    guardrails = _guardrails_config(config)
    return _bool_config(guardrails.get("enabled"), bool(guardrails))


def _guardrail_action(config: Mapping[str, Any]) -> str:
    action = _guardrails_config(config).get("action")
    if isinstance(action, str) and action.strip().lower() in _GUARDRAIL_ACTIONS:
        return action.strip().lower()
    return "block"


def _request_guardrail_text(request_body: Mapping[str, Any]) -> str:
    parts = [
        _jsonable_text(request_body.get("messages")),
        _jsonable_text(request_body.get("input")),
        _jsonable_text(request_body.get("instructions")),
    ]
    return "\n".join(part for part in parts if part)


def _named_patterns(value: Any) -> list[tuple[str, re.Pattern[str]]]:
    if not isinstance(value, list):
        return []
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for index, item in enumerate(value, start=1):
        name = f"pattern_{index}"
        pattern_value: Any = item
        if isinstance(item, dict):
            name_value = item.get("name")
            if isinstance(name_value, str) and name_value.strip():
                name = name_value.strip()
            pattern_value = item.get("pattern")
        if not isinstance(pattern_value, str) or not pattern_value.strip():
            continue
        try:
            patterns.append((name, re.compile(pattern_value, re.IGNORECASE)))
        except re.error:
            logger.warning("Ignoring invalid routing guardrail regex pattern '%s'", name)
    return patterns


def _guardrail_violation(kind: str, rule: str) -> dict[str, str]:
    return {"type": kind, "rule": rule}


def _redaction_rules(config: Mapping[str, Any]) -> list[tuple[str, str, re.Pattern[str]]]:
    redactions = _redactions_config(config)
    rules: list[tuple[str, str, re.Pattern[str]]] = []

    pii_config = redactions.get("pii")
    pii_enabled = _bool_config(pii_config.get("enabled") if isinstance(pii_config, dict) else pii_config, False)
    if pii_enabled:
        type_config = pii_config.get("types") if isinstance(pii_config, dict) else redactions.get("pii_types")
        pii_types = _string_list(type_config) or sorted(_PII_PATTERNS)
        for pii_type in pii_types:
            pattern = _PII_PATTERNS.get(pii_type)
            if pattern is not None:
                rules.append(("pii", pii_type, pattern))

    for name, pattern in _named_patterns(redactions.get("patterns")):
        rules.append(("pattern", name, pattern))
    return rules


def _redact_text(
    value: str,
    *,
    rules: Sequence[tuple[str, str, re.Pattern[str]]],
    replacement: str,
    counts: dict[tuple[str, str], int],
) -> str:
    redacted = value
    for kind, rule, pattern in rules:
        redacted, count = pattern.subn(replacement, redacted)
        if count:
            key = (kind, rule)
            counts[key] = counts.get(key, 0) + count
    return redacted


def _redact_content(
    value: Any,
    *,
    rules: Sequence[tuple[str, str, re.Pattern[str]]],
    replacement: str,
    counts: dict[tuple[str, str], int],
) -> Any:
    if isinstance(value, str):
        return _redact_text(value, rules=rules, replacement=replacement, counts=counts)
    if isinstance(value, list):
        return [
            _redact_content(item, rules=rules, replacement=replacement, counts=counts)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _redact_content(item, rules=rules, replacement=replacement, counts=counts)
            for key, item in value.items()
        }
    return value


def _redaction_replacement(config: Mapping[str, Any]) -> str:
    replacement = _redactions_config(config).get("replacement")
    if isinstance(replacement, str):
        return replacement
    return "[REDACTED]"


def apply_guardrail_redactions(
    config: Mapping[str, Any],
    request_body: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply policy redactions to provider-bound request content."""
    body = copy.deepcopy(dict(request_body))
    if not _redactions_enabled(config):
        return body, None

    rules = _redaction_rules(config)
    redactions = _redactions_config(config)
    replacement = _redaction_replacement(config)
    if not rules:
        return body, {
            "enabled": True,
            "status": "skipped",
            "reason": "missing_rules",
            "replacement": replacement,
        }

    counts: dict[tuple[str, str], int] = {}
    messages = body.get("messages")
    if isinstance(messages, list):
        redacted_messages: list[Any] = []
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                redacted_message = dict(message)
                redacted_message["content"] = _redact_content(
                    message.get("content"),
                    rules=rules,
                    replacement=replacement,
                    counts=counts,
                )
                redacted_messages.append(redacted_message)
            else:
                redacted_messages.append(message)
        body["messages"] = redacted_messages

    for key in ("input", "instructions"):
        if key in body:
            body[key] = _redact_content(body[key], rules=rules, replacement=replacement, counts=counts)

    count_items = [
        {"type": kind, "rule": rule, "count": count}
        for (kind, rule), count in sorted(counts.items())
    ]
    total_replacements = sum(counts.values())
    return body, {
        "enabled": True,
        "status": "redacted" if total_replacements else "unchanged",
        "replacement": replacement,
        "total_replacements": total_replacements,
        "counts": count_items,
        "pattern_count": len(_named_patterns(redactions.get("patterns"))),
    }


def _external_classifier_configs(guardrails: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    classifiers = guardrails.get("external_classifiers")
    if isinstance(classifiers, dict):
        return [classifiers]
    if not isinstance(classifiers, list):
        return []
    return [classifier for classifier in classifiers if isinstance(classifier, dict)]


def _external_classifier_name(classifier: Mapping[str, Any], index: int) -> str:
    name = classifier.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return f"classifier_{index}"


def _external_classifier_headers(classifier: Mapping[str, Any]) -> dict[str, str] | None:
    headers = classifier.get("headers")
    if not isinstance(headers, dict):
        return None
    normalized = {str(key): str(value) for key, value in headers.items() if str(key).strip()}
    return normalized or None


async def _post_external_guardrail_classifier(
    *,
    url: str,
    request_text: str,
    timeout_seconds: float,
    headers: dict[str, str] | None,
) -> tuple[int | None, dict[str, Any] | None, str | None]:
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(url, json={"text": request_text}, headers=headers)
    except httpx.HTTPError as exc:
        return None, None, str(exc)

    payload: dict[str, Any] | None = None
    try:
        parsed = response.json()
        if isinstance(parsed, dict):
            payload = parsed
    except ValueError:
        payload = None

    if response.status_code < 200 or response.status_code >= 300:
        error = response.text
        if payload is not None:
            detail = payload.get("detail") or payload.get("error")
            if isinstance(detail, str) and detail.strip():
                error = detail.strip()
        return response.status_code, payload, f"HTTP {response.status_code}: {error}"
    if payload is None:
        return response.status_code, None, "classifier returned non-object JSON"
    return response.status_code, payload, None


def _classifier_rule(value: Any, *, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("rule", "type", "label", "category", "name"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
    return fallback


def _classifier_violations(name: str, payload: Mapping[str, Any], threshold: float | None) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    raw_violations = payload.get("violations")
    if isinstance(raw_violations, list):
        for item in raw_violations:
            violations.append(_guardrail_violation("external_classifier", _classifier_rule(item, fallback=name)))

    flagged = payload.get("blocked") is True or payload.get("flagged") is True
    score = _non_negative_float_or_none(payload.get("score"))
    if threshold is not None and score is not None and score >= threshold:
        flagged = True
    if flagged and not violations:
        label = _classifier_rule(payload.get("label"), fallback=name)
        violations.append(_guardrail_violation("external_classifier", label))
    return violations


async def _evaluate_external_classifiers(
    *,
    guardrails: Mapping[str, Any],
    request_text: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    violations: list[dict[str, str]] = []
    classifier_results: list[dict[str, Any]] = []
    for index, classifier in enumerate(_external_classifier_configs(guardrails), start=1):
        name = _external_classifier_name(classifier, index)
        url = classifier.get("url")
        if not isinstance(url, str) or not url.strip():
            classifier_results.append({"name": name, "status": "skipped", "reason": "missing_url"})
            continue
        timeout_seconds = _non_negative_float_or_none(classifier.get("timeout_seconds")) or 2.0
        threshold = _non_negative_float_or_none(classifier.get("threshold"))
        status_code, payload, error = await _post_external_guardrail_classifier(
            url=url.strip(),
            request_text=request_text,
            timeout_seconds=timeout_seconds,
            headers=_external_classifier_headers(classifier),
        )
        fail_closed = _bool_config(classifier.get("fail_closed"), False)
        if error is not None:
            classifier_results.append(
                {
                    "name": name,
                    "status": "error",
                    "status_code": status_code,
                    "error": error[:200],
                    "fail_closed": fail_closed,
                }
            )
            if fail_closed:
                violations.append(_guardrail_violation("external_classifier_error", name))
            continue
        assert payload is not None
        classifier_violations = _classifier_violations(name, payload, threshold)
        violations.extend(classifier_violations)
        score = _non_negative_float_or_none(payload.get("score"))
        label = payload.get("label") if isinstance(payload.get("label"), str) else None
        classifier_results.append(
            {
                "name": name,
                "status": "flagged" if classifier_violations else "passed",
                "status_code": status_code,
                "score": score,
                "threshold": threshold,
                "label": label,
                "violations": classifier_violations,
            }
        )
    return violations, classifier_results


async def _evaluate_guardrails(config: Mapping[str, Any], request_body: Mapping[str, Any]) -> dict[str, Any] | None:
    if not _guardrails_enabled(config):
        return None

    guardrails, preset_metadata = _effective_guardrails_config(config)
    request_text = _request_guardrail_text(request_body)
    normalized_text = request_text.lower()
    violations: list[dict[str, str]] = []
    classifier_results: list[dict[str, Any]] = []

    for term in _string_list(guardrails.get("blocked_terms")):
        if term.lower() in normalized_text:
            violations.append(_guardrail_violation("blocked_term", term))

    for name, pattern in _named_patterns(guardrails.get("blocked_patterns")):
        if pattern.search(request_text):
            violations.append(_guardrail_violation("blocked_pattern", name))

    pii_config = guardrails.get("pii")
    pii_enabled = _bool_config(pii_config.get("enabled") if isinstance(pii_config, dict) else pii_config, False)
    if pii_enabled:
        type_config = pii_config.get("types") if isinstance(pii_config, dict) else None
        pii_types = _string_list(type_config) or sorted(_PII_PATTERNS)
        for pii_type in pii_types:
            pii_pattern = _PII_PATTERNS.get(pii_type)
            if pii_pattern is not None and pii_pattern.search(request_text):
                violations.append(_guardrail_violation("pii", pii_type))

    injection_config = guardrails.get("prompt_injection")
    injection_enabled = _bool_config(
        injection_config.get("enabled") if isinstance(injection_config, dict) else injection_config,
        False,
    )
    if injection_enabled:
        phrases = _string_list(injection_config.get("phrases") if isinstance(injection_config, dict) else None)
        for phrase in [*phrases, *_PROMPT_INJECTION_PHRASES]:
            if phrase.lower() in normalized_text:
                violations.append(_guardrail_violation("prompt_injection", phrase))

    external_violations, classifier_results = await _evaluate_external_classifiers(
        guardrails=guardrails,
        request_text=request_text,
    )
    violations.extend(external_violations)

    action = _guardrail_action(config)
    status_value = "passed"
    if violations:
        status_value = "blocked" if action == "block" else "observed"
    result: dict[str, Any] = {
        "enabled": True,
        "status": status_value,
        "action": action,
        "violations": violations,
        "external_classifiers": classifier_results,
        "checked_text_chars": len(request_text),
    }
    if preset_metadata is not None:
        result["presets"] = preset_metadata
    return result


async def _default_policy(db: AsyncSession) -> RoutingPolicy | None:
    result = await db.execute(
        select(RoutingPolicy)
        .where(RoutingPolicy.is_default.is_(True))
        .where(RoutingPolicy.status == ACTIVE_ROUTING_POLICY_STATUS)
        .order_by(RoutingPolicy.updated_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _matching_policy(db: AsyncSession, tags: Mapping[str, str]) -> _PolicyMatch | None:
    if not tags:
        return None

    result = await db.execute(
        select(RoutingPolicy)
        .where(RoutingPolicy.is_default.is_(False))
        .where(RoutingPolicy.status == ACTIVE_ROUTING_POLICY_STATUS)
    )
    policies = result.scalars().all()
    matches: list[_PolicyMatch] = []
    for policy in policies:
        config = policy.config_ or {}
        if not _matches_policy_match_config(config, tags):
            continue
        rollout = _policy_rollout_info(policy, tags)
        if rollout is not None and not rollout["matched"]:
            continue
        source = "canary_match" if rollout is not None else "tag_match"
        matches.append(_PolicyMatch(policy=policy, source=source, rollout=rollout))
    if not matches:
        return None

    return sorted(
        matches,
        key=lambda match: (
            _policy_match_priority(match.policy.config_ or {}),
            match.policy.updated_at,
        ),
        reverse=True,
    )[0]


async def resolve_routing_plan(
    db: AsyncSession,
    *,
    request_body: Mapping[str, Any],
    project_id: str | None,
    tags: Mapping[str, str] | None,
    policy_id: str | None = None,
    allow_inactive_policy: bool = False,
) -> RoutingPlan:
    """Resolve a routing policy into an ordered model-attempt plan."""
    project: Project | None = None
    policy: RoutingPolicy | None = None
    request_tags = dict(tags or {})
    policy_source = "default"
    policy_rollout: dict[str, Any] | None = None

    if project_id:
        project = await db.get(Project, project_id)
        if project is None:
            raise RoutingPolicyError(404, f"Project '{project_id}' not found")
        if not project.is_active:
            raise RoutingPolicyError(403, f"Project '{project_id}' is inactive")
        if project.routing_policy_id and policy_id is None:
            policy = await db.get(RoutingPolicy, project.routing_policy_id)
            policy_source = "project"

    if policy_id is not None:
        policy = await db.get(RoutingPolicy, policy_id)
        if policy is None:
            raise RoutingPolicyError(404, f"Routing policy '{policy_id}' not found")
        policy_source = "policy_override"
        policy_rollout = _policy_rollout_info(policy, request_tags)

    if policy is None:
        match = await _matching_policy(db, request_tags)
        if match is not None:
            policy = match.policy
            policy_source = match.source
            policy_rollout = match.rollout

    if policy is None:
        policy = await _default_policy(db)
        policy_source = "default"
        policy_rollout = None
    if policy is None:
        raise RoutingPolicyError(404, "No routing policy is configured")
    if policy.status != ACTIVE_ROUTING_POLICY_STATUS and not allow_inactive_policy:
        raise RoutingPolicyError(422, f"Routing policy '{policy.policy_id}' is not active")

    strategy = policy.strategy
    if strategy not in ROUTING_STRATEGIES:
        raise RoutingPolicyError(422, f"Unsupported routing strategy '{strategy}'")

    config = policy.config_ or {}
    guardrails = await _evaluate_guardrails(config, request_body)
    if guardrails is not None and guardrails["status"] == "blocked":
        first_violation = guardrails["violations"][0] if guardrails["violations"] else {}
        violation_type = first_violation.get("type", "guardrail")
        violation_rule = first_violation.get("rule", "configured_rule")
        raise RoutingPolicyError(
            403,
            f"Routing policy '{policy.policy_id}' guardrail blocked request: {violation_type}:{violation_rule}",
        )
    try:
        specs = _configured_candidate_specs(config)
    except ValueError as exc:
        raise RoutingPolicyError(422, f"Invalid routing policy candidate: {exc}") from exc
    if not specs:
        raise RoutingPolicyError(422, f"Routing policy '{policy.policy_id}' has no candidates")

    redacted_request_body, redactions = apply_guardrail_redactions(config, request_body)
    if redactions is not None:
        if guardrails is None:
            guardrails = {"enabled": True, "status": "passed", "action": _guardrail_action(config), "violations": []}
        guardrails["redactions"] = redactions

    effective_request_body, context = apply_context_policy(config, redacted_request_body)
    prompt_tokens = estimate_prompt_tokens(effective_request_body)
    output_tokens = estimate_output_tokens(effective_request_body)
    target_tier = classify_request_tier(effective_request_body, prompt_tokens=prompt_tokens, config=config)
    try:
        candidates = await _build_candidates(
            db,
            specs,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
    except ValueError as exc:
        raise RoutingPolicyError(422, f"Invalid routing policy candidate: {exc}") from exc
    candidates, rejected_candidates = _apply_constraints(candidates, config=config, tags=request_tags)
    if not candidates:
        detail = f"Routing policy '{policy.policy_id}' has no candidates after constraints"
        if rejected_candidates:
            reasons = sorted({str(item["reason"]) for item in rejected_candidates})
            detail = f"{detail}: {', '.join(reasons)}"
        raise RoutingPolicyError(422, detail)
    candidates = await _attach_provider_health(db, candidates, config=config)
    candidates, health_rejected_candidates = _apply_provider_health_gate(candidates, config=config)
    rejected_candidates.extend(health_rejected_candidates)
    if not candidates:
        detail = f"Routing policy '{policy.policy_id}' has no candidates after provider health gate"
        if health_rejected_candidates:
            reasons = sorted({str(item["reason"]) for item in health_rejected_candidates})
            detail = f"{detail}: {', '.join(reasons)}"
        raise RoutingPolicyError(422, detail)
    if strategy in {"least_latency", "weighted_score"}:
        candidates = await _attach_latency_stats(db, candidates, config=config)
    if strategy == "weighted_score":
        candidates = _attach_weighted_scores(candidates, config=config)
    ordered_candidates = _order_candidates(candidates, strategy=strategy, target_tier=target_tier)
    ordered_candidates = _apply_provider_health_order(ordered_candidates, config=config)
    fallback_enabled = _fallback_enabled(config, strategy=strategy)
    if not fallback_enabled:
        ordered_candidates = ordered_candidates[:1]
    if not ordered_candidates:
        raise RoutingPolicyError(422, f"Routing policy '{policy.policy_id}' has no usable candidates")

    reason = f"{policy_source} policy {strategy} selected {ordered_candidates[0].model} for {target_tier} request"
    if not fallback_enabled:
        reason += " with fallback disabled"

    return RoutingPlan(
        requested_model=DEFAULT_ROUTING_MODEL,
        project=project,
        policy=policy,
        strategy=strategy,
        target_tier=target_tier,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        candidates=ordered_candidates,
        fallback_enabled=fallback_enabled,
        policy_source=policy_source,
        reason=reason,
        tags=request_tags,
        rejected_candidates=rejected_candidates,
        policy_rollout=policy_rollout,
        guardrails=guardrails,
        context=context,
    )


async def record_route_trace(
    db: AsyncSession,
    *,
    plan: RoutingPlan,
    api_key_id: str | None,
    user_id: str | None,
    status: str,
    attempts: list[dict[str, Any]],
    error_message: str | None,
    selected_candidate: RoutingCandidate | None,
    endpoint: str = DEFAULT_ROUTE_TRACE_ENDPOINT,
) -> str:
    """Persist a best-effort route trace and return its generated id."""
    trace_id = str(uuid.uuid4())
    candidate = selected_candidate
    trace = RouteTrace(
        trace_id=trace_id,
        api_key_id=api_key_id,
        user_id=user_id,
        project_id=plan.project_id,
        policy_id=plan.policy.policy_id,
        requested_model=plan.requested_model,
        endpoint=endpoint,
        selected_model=candidate.model if candidate else None,
        selected_provider=candidate.provider if candidate else None,
        strategy=plan.strategy,
        status=status,
        error_message=error_message,
        selected_reason=plan.reason if candidate else None,
        estimated_prompt_tokens=plan.prompt_tokens,
        estimated_output_tokens=plan.output_tokens,
        estimated_cost=candidate.estimated_cost if candidate else None,
        fallback_enabled=plan.fallback_enabled,
        policy_source=plan.policy_source,
        tags=plan.tags,
        guardrails=plan.guardrails,
        context=plan.context,
        candidates=[candidate_item.to_trace_dict() for candidate_item in plan.candidates],
        attempts=attempts,
    )
    db.add(trace)
    try:
        await db.commit()
    except SQLAlchemyError as exc:
        await db.rollback()
        logger.warning("Failed to record route trace %s: %s", trace_id, exc)
    return trace_id
