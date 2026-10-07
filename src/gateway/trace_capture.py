"""Opens a request's trace in the preamble and hands it to the trace writer once the response is sent.

The middleware's ``finally`` is the one place every request passes exactly once:
a streamed response outlives its route handler, and the settlement paths branch
a dozen ways, which is why the in-flight registry drops its entries here too.
The trace is closed with what the client actually got: the status line, and
whether the body was sent in full.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gateway.services.traces import RequestTrace, RequestTraces, TraceWriter

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from starlette.requests import Request
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Namespaced ASGI scope key holding the request id whose trace is open.
TRACE_SCOPE_KEY = "otari.trace_request_id"


def _state(scope: MutableMapping[str, Any]) -> Any:
    return getattr(scope.get("app"), "state", None)


def begin_request_trace(request: Request, trace: RequestTrace) -> None:
    """Register ``trace`` as the open trace of ``request``. A no-op where nothing collects traces."""
    traces: RequestTraces | None = getattr(_state(request.scope), "request_traces", None)
    if traces is None:
        return
    traces.begin(trace)
    request.scope[TRACE_SCOPE_KEY] = trace.request_id


class TraceCaptureMiddleware:
    """Closes each request's trace after its response, and submits it to the writer."""

    def __init__(self, app: ASGIApp, *, traces: RequestTraces) -> None:
        self.app = app
        self.traces = traces

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        status_code: int | None = None
        completed = False

        async def observing_send(message: Message) -> None:
            nonlocal status_code, completed
            if message["type"] == "http.response.start":
                status_code = message["status"]
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                completed = True
            await send(message)

        try:
            await self.app(scope, receive, observing_send)
        finally:
            mapping: MutableMapping[str, Any] = scope
            trace = self.traces.finish(mapping.get(TRACE_SCOPE_KEY))
            writer: TraceWriter | None = getattr(_state(mapping), "trace_writer", None)
            if trace is not None and writer is not None:
                writer.submit(trace.finish(status_code=status_code, completed=completed), truncated=trace.dropped)
