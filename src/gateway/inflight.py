"""In-process registry of the requests the gateway is currently serving.

A usage row is written when a request *settles*, so the activity log can only
ever describe the past. On a slow backend (a local model answering in 30 seconds
or more) that leaves the operator with nothing to look at while the request is
actually running. This module holds the other half: what is in flight right now.

Two deliberate properties:

* **In-memory, per worker.** Nothing is persisted and nothing is shared between
  processes, so a gateway run with several uvicorn workers reports only the
  requests the answering worker is serving. Persisting would mean a write per
  request on the hot path to record something that is stale a second later.
* **Removal is owned by the middleware, not the handler.** A streaming response
  outlives its route handler (the body is produced after the handler returns),
  and the settlement paths that write usage rows branch a dozen ways. The one
  place that always runs exactly once per request is the ASGI middleware's
  ``finally``, which is where the entry is dropped. This mirrors how the
  ``gateway_active_requests`` gauge is kept honest in :mod:`gateway.metrics`.

The entry map is therefore not capped: an entry's lifetime is a request's
lifetime, so its size is bounded by real concurrency. The read endpoint caps
what it serializes instead, and reports the true count alongside it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from starlette.requests import Request
    from starlette.types import ASGIApp, Receive, Scope, Send

# Namespaced ASGI scope key holding the id of this request's registry entry, set
# by the route preamble and read by the middleware that removes it. The scope
# dict is the one object both halves are guaranteed to share.
INFLIGHT_SCOPE_KEY = "otari.inflight_id"


@dataclass(frozen=True, slots=True)
class InFlightRequest:
    """One request the gateway is serving right now.

    The fields mirror their ``UsageLog`` counterparts, so the row an operator
    watches in flight reads the same way as the row it becomes. ``id`` does not:
    it is an ephemeral tracking id, not the id of the usage row this request will
    eventually write.
    """

    id: str
    endpoint: str
    model: str
    provider: str | None
    user_id: str | None
    api_key_id: str | None
    policy_name: str | None
    started_at: datetime
    # Monotonic reading, so elapsed time is immune to a wall-clock step.
    started_monotonic: float

    def elapsed_ms(self, now: float | None = None) -> int:
        """Milliseconds since this request entered the registry."""
        return max(0, round(((now if now is not None else monotonic()) - self.started_monotonic) * 1000))


class InFlightRegistry:
    """The set of requests currently being served by this process.

    No lock: every mutation is a single dict operation on the event loop's
    thread, and the registry is never read from a worker thread.
    """

    def __init__(self) -> None:
        self._entries: dict[str, InFlightRequest] = {}

    def begin(
        self,
        *,
        endpoint: str,
        model: str,
        provider: str | None = None,
        user_id: str | None = None,
        api_key_id: str | None = None,
        policy_name: str | None = None,
    ) -> str:
        """Record a request as in flight and return its tracking id."""
        entry = InFlightRequest(
            id=str(uuid.uuid4()),
            endpoint=endpoint,
            model=model,
            provider=provider,
            user_id=user_id,
            api_key_id=api_key_id,
            policy_name=policy_name,
            started_at=datetime.now(UTC),
            started_monotonic=monotonic(),
        )
        self._entries[entry.id] = entry
        return entry.id

    def finish(self, request_id: str | None) -> None:
        """Drop an entry. Tolerates ``None`` and an id already dropped.

        Both cases are normal rather than exceptional: a request rejected before
        the preamble registered anything has no id, and the middleware runs its
        cleanup unconditionally.
        """
        if request_id is None:
            return
        self._entries.pop(request_id, None)

    def snapshot(self) -> list[InFlightRequest]:
        """Every tracked request, longest-running first.

        Oldest-first rather than newest-first: the reason to look at this list is
        to find what a caller is still waiting on, and that is the front of it.
        """
        return sorted(self._entries.values(), key=lambda entry: entry.started_monotonic)

    def __len__(self) -> int:
        return len(self._entries)


def get_registry(request: Request) -> InFlightRegistry | None:
    """The app's registry, or None when there is nowhere to record anything.

    Reads ``scope["app"]`` rather than ``request.app``, which raises on a scope
    assembled without one: tracking is an observability side effect, so a bare
    ASGI harness (or a unit test driving the preamble directly) must degrade to
    "not tracked" rather than fail the request it was watching.
    """
    state = getattr(request.scope.get("app"), "state", None)
    registry: InFlightRegistry | None = getattr(state, "inflight", None)
    return registry


def track_request(
    request: Request,
    *,
    endpoint: str,
    model: str,
    provider: str | None = None,
    user_id: str | None = None,
    api_key_id: str | None = None,
    policy_name: str | None = None,
) -> None:
    """Mark ``request`` as in flight until its response has been fully sent.

    The id is stashed on the ASGI scope for :class:`InFlightMiddleware` to clean
    up. Called once the request is authorized and about to be dispatched, so a
    request the gateway refused never appears as in flight.
    """
    registry = get_registry(request)
    if registry is None:
        return
    request.scope[INFLIGHT_SCOPE_KEY] = registry.begin(
        endpoint=endpoint,
        model=model,
        provider=provider,
        user_id=user_id,
        api_key_id=api_key_id,
        policy_name=policy_name,
    )


class InFlightMiddleware:
    """Removes a request's registry entry once its response is fully sent.

    Pure ASGI rather than ``BaseHTTPMiddleware`` for the same reason
    :class:`gateway.metrics.MetricsMiddleware` is: the ``finally`` has to run
    after a streaming body is exhausted, not when the route handler returns.
    """

    def __init__(self, app: ASGIApp, registry: InFlightRegistry) -> None:
        self.app = app
        self.registry = registry

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            mapping: MutableMapping[str, Any] = scope
            self.registry.finish(mapping.get(INFLIGHT_SCOPE_KEY))
