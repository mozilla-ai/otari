"""Pluggable model-routing backend interface (interface seam only).

Defines the contract a routing backend implements so the gateway can pick a
model from a candidate pool per request, plus a default-off wiring point. This
module owns the protocol and the inert backends; the learned kNN backend lives
in :mod:`gateway.services.knn_router` and is imported lazily by
:func:`get_router_backend` so the default path never loads it.

Selection is operator-configurable via ``OTARI_ROUTER_BACKEND`` (scalar
``router_backend`` field on :class:`gateway.core.config.GatewayConfig`):

* ``"none"`` (default) → :func:`get_router_backend` returns ``None`` and the
  chat handler behaves byte-for-byte as it does today.
* ``"noop"`` → a :class:`NoOpRouterBackend` that returns the requested model.
* ``"knn"`` → a :class:`gateway.services.knn_router.KnnRoutingMemory`.
* anything else → :class:`NotImplementedError` (strategy backends are not built
  yet).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from gateway.log_config import logger

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig


@dataclass
class RoutingContext:
    """Inputs a backend may use to pick a model for a single request."""

    tenant_id: str
    messages: list[dict[str, Any]]
    requested_model: str
    candidate_pool: list[str]
    task_id: str | None = None
    has_tools: bool = False
    is_trace_continuation: bool = False
    trace_key: str | None = None


@dataclass
class RoutingDecision:
    """A backend's ordered model preference for one request."""

    ordered_models: list[str]
    confidence: float
    rationale: str


@runtime_checkable
class RouterBackend(Protocol):
    """Contract a routing backend implements."""

    async def route(self, ctx: RoutingContext) -> RoutingDecision: ...


class NoOpRouterBackend:
    """Routing backend that always returns the requested model unchanged."""

    async def route(self, ctx: RoutingContext) -> RoutingDecision:
        return RoutingDecision(
            ordered_models=[ctx.requested_model],
            confidence=1.0,
            rationale="routing disabled (noop)",
        )


# The kNN backend carries per-process mutable state (the trace-sticky decision
# cache). The chat handler resolves the backend once per request, so a fresh
# instance every call would reset that cache and break trace stickiness across
# the requests of one conversation. Cache one instance per backend-config
# signature so the state persists; cleared by clear_router_backend_cache().
_KNN_CACHE: dict[tuple[Any, ...], RouterBackend] = {}


def _knn_signature(config: GatewayConfig) -> tuple[Any, ...]:
    return (
        config.router_alpha,
        config.router_k,
        config.router_embedding_model,
        config.router_confidence_floor,
        config.router_seed_count,
        config.router_granularity,
        config.router_max_vectors_per_tenant,
    )


def clear_router_backend_cache() -> None:
    """Drop cached backend instances (test isolation; called from reset_config)."""
    _KNN_CACHE.clear()


def get_router_backend(config: GatewayConfig) -> RouterBackend | None:
    """Resolve the configured routing backend.

    Returns ``None`` when routing is disabled (``router_backend == "none"``),
    a :class:`NoOpRouterBackend` for ``"noop"``, a cached
    :class:`gateway.services.knn_router.KnnRoutingMemory` for ``"knn"``, and
    raises :class:`NotImplementedError` for any other value.
    """
    backend = config.router_backend.strip().lower()
    if backend == "none":
        return None
    if backend == "noop":
        logger.debug("Router backend enabled: noop (returns requested model unchanged)")
        return NoOpRouterBackend()
    if backend == "knn":
        # Imported lazily: the kNN backend pulls in any_llm embeddings and the
        # ORM store, neither of which the default ('none') path should load.
        from gateway.services.knn_router import KnnRoutingMemory

        signature = _knn_signature(config)
        cached = _KNN_CACHE.get(signature)
        if cached is None:
            logger.debug("Router backend enabled: knn (routing memory)")
            cached = KnnRoutingMemory(config)
            _KNN_CACHE[signature] = cached
        return cached
    msg = (
        f"Unknown router_backend '{config.router_backend}': only 'none', 'noop', and 'knn' are implemented. "
        "Strategy (cheapest/fastest) backends are not built yet."
    )
    raise NotImplementedError(msg)
