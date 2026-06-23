"""Unit tests for the pluggable router-backend interface seam.

Proves the seam is inert by default: ``get_router_backend`` returns ``None``
for the default ``"none"`` setting, a :class:`NoOpRouterBackend` for ``"noop"``
(which echoes the requested model), and that an unimplemented value is rejected
at config load.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gateway.core.config import GatewayConfig
from gateway.services.router_backend import (
    NoOpRouterBackend,
    RouterBackend,
    RoutingContext,
    get_router_backend,
)


def _ctx(model: str = "openai/gpt-4o") -> RoutingContext:
    return RoutingContext(
        tenant_id="user-1",
        messages=[{"role": "user", "content": "hi"}],
        requested_model=model,
        candidate_pool=[model],
    )


def test_get_router_backend_none_returns_none() -> None:
    config = GatewayConfig(router_backend="none")
    assert get_router_backend(config) is None


def test_get_router_backend_noop_returns_noop_backend() -> None:
    config = GatewayConfig(router_backend="noop")
    backend = get_router_backend(config)
    assert isinstance(backend, NoOpRouterBackend)
    assert isinstance(backend, RouterBackend)


def test_unknown_router_backend_is_rejected_at_config_load() -> None:
    # Unimplemented strategy backends fail fast at config construction rather
    # than turning every chat request into a 500 at resolution time.
    with pytest.raises(ValidationError, match="router_backend must be one of"):
        GatewayConfig(router_backend="cheapest")


@pytest.mark.asyncio
async def test_noop_route_echoes_requested_model() -> None:
    backend = NoOpRouterBackend()
    decision = await backend.route(_ctx("anthropic/claude-3-5-sonnet"))
    assert decision.ordered_models == ["anthropic/claude-3-5-sonnet"]
    assert decision.confidence == 1.0
