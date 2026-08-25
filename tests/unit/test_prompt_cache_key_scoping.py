"""Tests for authenticated prompt-cache routing namespaces."""

from typing import Any, cast

from gateway.api.routes._pipeline import RequestContext, scope_prompt_cache_key
from gateway.api.routes._platform import ResolvedRoute
from gateway.core.config import GatewayConfig


def _context(
    *,
    user_id: str | None = "user-a",
    route: ResolvedRoute | None = None,
) -> RequestContext:
    return RequestContext(
        config=GatewayConfig(),
        db=None,
        log_writer=cast(Any, None),
        hybrid_mode=route is not None,
        route=route,
        user_token=None,
        api_key_id=None,
        user_id=user_id if route is None else None,
        rate_limit_info=None,
        reservation=None,
        started_at=0.0,
    )


def _scoped_key(ctx: RequestContext, caller_key: str = "shared-key") -> str:
    fields: dict[str, Any] = {"prompt_cache_key": caller_key}
    assert scope_prompt_cache_key(fields, ctx) is fields
    scoped = fields["prompt_cache_key"]
    assert isinstance(scoped, str)
    return scoped


def test_prompt_cache_key_is_stable_and_fixed_length() -> None:
    first = _scoped_key(_context())
    second = _scoped_key(_context())

    assert first == second
    assert first != "shared-key"
    assert len(first) == 64


def test_prompt_cache_key_isolated_between_standalone_users() -> None:
    assert _scoped_key(_context(user_id="user-a")) != _scoped_key(_context(user_id="user-b"))


def test_prompt_cache_key_uses_hybrid_resolved_user() -> None:
    route_a = ResolvedRoute(
        request_id="request-a",
        fallback_enabled=False,
        attempts=[],
        user_id="platform-user-a",
        workspace_id="workspace-shared",
    )
    route_b = route_a.model_copy(update={"user_id": "platform-user-b"})

    assert _scoped_key(_context(route=route_a)) != _scoped_key(_context(route=route_b))


def test_prompt_cache_key_falls_back_to_hybrid_workspace() -> None:
    route_a = ResolvedRoute(
        request_id="request-a",
        fallback_enabled=False,
        attempts=[],
        workspace_id="workspace-a",
    )
    route_b = route_a.model_copy(update={"workspace_id": "workspace-b"})

    assert _scoped_key(_context(route=route_a)) != _scoped_key(_context(route=route_b))


def test_prompt_cache_key_is_dropped_without_authenticated_scope() -> None:
    route = ResolvedRoute(request_id="request-a", fallback_enabled=False, attempts=[])
    fields: dict[str, Any] = {"prompt_cache_key": "shared-key", "model": "openai:gpt-5"}

    scope_prompt_cache_key(fields, _context(route=route))

    assert fields == {"model": "openai:gpt-5"}
