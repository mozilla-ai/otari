"""In-flight request registry and its cleanup middleware."""

from collections.abc import MutableMapping
from typing import Any

import pytest
from starlette.types import Receive, Scope, Send

from gateway.inflight import (
    INFLIGHT_SCOPE_KEY,
    InFlightMiddleware,
    InFlightRegistry,
)


def _begin(registry: InFlightRegistry, model: str = "openai:gpt-4o", **extra: Any) -> str:
    return registry.begin(endpoint="/v1/chat/completions", model=model, **extra)


class TestRegistry:
    def test_begin_makes_a_request_visible(self) -> None:
        registry = InFlightRegistry()
        request_id = _begin(registry, provider="openai", user_id="u1", api_key_id="k1", policy_name="cheap")

        (entry,) = registry.snapshot()
        assert entry.id == request_id
        assert entry.endpoint == "/v1/chat/completions"
        assert entry.model == "openai:gpt-4o"
        assert entry.provider == "openai"
        assert entry.user_id == "u1"
        assert entry.api_key_id == "k1"
        assert entry.policy_name == "cheap"
        assert len(registry) == 1

    def test_finish_removes_the_entry(self) -> None:
        registry = InFlightRegistry()
        request_id = _begin(registry)

        registry.finish(request_id)

        assert registry.snapshot() == []
        assert len(registry) == 0

    def test_finish_tolerates_none_and_a_repeat(self) -> None:
        """The middleware cleans up unconditionally, so neither case is an error."""
        registry = InFlightRegistry()
        request_id = _begin(registry)

        registry.finish(None)
        registry.finish(request_id)
        registry.finish(request_id)

        assert len(registry) == 0

    def test_snapshot_puts_the_longest_running_first(self) -> None:
        registry = InFlightRegistry()
        first = _begin(registry, model="a")
        second = _begin(registry, model="b")

        assert [entry.id for entry in registry.snapshot()] == [first, second]

    def test_elapsed_ms_grows_and_never_goes_negative(self) -> None:
        registry = InFlightRegistry()
        _begin(registry)
        (entry,) = registry.snapshot()

        assert entry.elapsed_ms(entry.started_monotonic) == 0
        assert entry.elapsed_ms(entry.started_monotonic + 1.5) == 1500
        # A clock reading taken before the entry (a caller passing a stale `now`)
        # must not report a request that started in the future.
        assert entry.elapsed_ms(entry.started_monotonic - 5) == 0


async def _receive() -> MutableMapping[str, Any]:
    return {"type": "http.request"}


async def _send(message: MutableMapping[str, Any]) -> None:
    return None


class TestMiddleware:
    @staticmethod
    def _scope() -> Scope:
        return {"type": "http", "method": "POST", "path": "/v1/chat/completions"}

    @pytest.mark.asyncio
    async def test_entry_is_dropped_once_the_response_is_sent(self) -> None:
        registry = InFlightRegistry()
        scope = self._scope()

        async def app(scope: Scope, receive: Receive, send: Send) -> None:
            # Stands in for the route preamble registering the request.
            scope[INFLIGHT_SCOPE_KEY] = _begin(registry)
            assert len(registry) == 1

        await InFlightMiddleware(app, registry)(scope, _receive, _send)

        assert len(registry) == 0

    @pytest.mark.asyncio
    async def test_entry_is_dropped_when_the_request_fails(self) -> None:
        """Otherwise a failing request would sit in the list forever."""
        registry = InFlightRegistry()

        async def app(scope: Scope, receive: Receive, send: Send) -> None:
            scope[INFLIGHT_SCOPE_KEY] = _begin(registry)
            raise RuntimeError("upstream exploded")

        with pytest.raises(RuntimeError, match="upstream exploded"):
            await InFlightMiddleware(app, registry)(self._scope(), _receive, _send)

        assert len(registry) == 0

    @pytest.mark.asyncio
    async def test_unregistered_request_is_a_no_op(self) -> None:
        """Most requests (dashboard reads, /health) never register anything."""
        registry = InFlightRegistry()

        async def app(scope: Scope, receive: Receive, send: Send) -> None:
            return None

        await InFlightMiddleware(app, registry)(self._scope(), _receive, _send)

        assert len(registry) == 0

    @pytest.mark.asyncio
    async def test_non_http_scopes_pass_straight_through(self) -> None:
        registry = InFlightRegistry()
        seen: list[str] = []

        async def app(scope: Scope, receive: Receive, send: Send) -> None:
            seen.append(scope["type"])

        await InFlightMiddleware(app, registry)({"type": "lifespan"}, _receive, _send)

        assert seen == ["lifespan"]
