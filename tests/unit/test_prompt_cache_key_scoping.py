"""Tests for authenticated prompt-cache routing namespaces."""

import uuid
from typing import Any, cast

from gateway.api.routes._pipeline import RequestContext, scope_prompt_cache_key
from gateway.core.config import GatewayConfig


def _context(*, user_id: str | None = "user-a") -> RequestContext:
    return RequestContext(
        config=GatewayConfig(),
        db=cast(Any, object()),
        uow=cast(Any, object()),
        log_writer=cast(Any, None),
        api_key_id=None,
        user_id=user_id,
        rate_limit_info=None,
        reservation=None,
        started_at=0.0,
        workspace_id=uuid.uuid4(),
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


def test_prompt_cache_key_isolated_between_users() -> None:
    assert _scoped_key(_context(user_id="user-a")) != _scoped_key(_context(user_id="user-b"))


def test_prompt_cache_key_is_dropped_without_authenticated_scope() -> None:
    fields: dict[str, Any] = {"prompt_cache_key": "shared-key", "model": "openai:gpt-5"}

    scope_prompt_cache_key(fields, _context(user_id=None))

    assert fields == {"model": "openai:gpt-5"}
