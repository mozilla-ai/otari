"""Opens a request's trace in the preamble and hands it to the trace writer once the response is sent.

The middleware's ``finally`` is the one place every request passes exactly once:
a streamed response outlives its route handler, and the settlement paths branch
a dozen ways, which is why the in-flight registry drops its entries here too.
The trace is closed with what the client actually got: the status line, and
whether the body was sent in full. In a workspace that keeps everything, the
body itself is kept too (bounded), so the step's content holds the model's
output for this request and not only what the next request replays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gateway.log_config import logger
from gateway.services.traces import ContentKeys, RequestTrace, RequestTraces, TraceWriter

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
        is_stream = False

        async def observing_send(message: Message) -> None:
            nonlocal status_code, completed, is_stream
            if message["type"] == "http.response.start":
                status_code = message["status"]
                is_stream = any(
                    name.lower() == b"content-type" and value.startswith(b"text/event-stream")
                    for name, value in message.get("headers", [])
                )
            elif message["type"] == "http.response.body":
                if not message.get("more_body", False):
                    completed = True
                trace = self.traces.get(scope.get(TRACE_SCOPE_KEY))
                if trace is not None:
                    trace.keep_response_bytes(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, receive, observing_send)
        finally:
            mapping: MutableMapping[str, Any] = scope
            trace = self.traces.finish(mapping.get(TRACE_SCOPE_KEY))
            if trace is not None:
                # After the response, so nothing here may raise into the app's own
                # outcome: a trace that cannot be written is dropped, and logged
                # without its payload.
                try:
                    trace.read_response(is_stream=is_stream)
                    await _submit(mapping, trace, status_code=status_code, completed=completed)
                except Exception:  # noqa: BLE001
                    logger.warning("Trace for request %s dropped: %s", trace.request_id, "could not be closed")
                finally:
                    trace.discard_content()


async def _submit(
    scope: MutableMapping[str, Any], trace: RequestTrace, *, status_code: int | None, completed: bool
) -> None:
    writer: TraceWriter | None = getattr(_state(scope), "trace_writer", None)
    if writer is None:
        return
    write = trace.finish(status_code=status_code, completed=completed)
    keys: ContentKeys | None = getattr(_state(scope), "trace_content_keys", None)
    if trace.content and keys is not None:
        try:
            write = await keys.seal(write, trace.content)
        # Sealing reaches a key store this module cannot know the failures of. Whatever
        # it raises, the content is dropped and the trace is still kept: content is the
        # optional part, and plaintext never travels on.
        except Exception as exc:  # noqa: BLE001
            logger.warning("Trace content could not be sealed (%s); keeping the trace without it", type(exc).__name__)
    writer.submit(write, truncated=trace.dropped)
