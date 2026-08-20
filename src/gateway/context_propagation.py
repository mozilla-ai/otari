"""OpenTelemetry context propagation middleware.

This middleware extracts incoming context using the propagator configured by
OpenTelemetry, and sets up the context so that subsequent spans created in the
request are properly linked to the parent trace. OpenTelemetry's default
propagator set includes W3C Trace Context (`traceparent` and `tracestate`), but
the `OTEL_PROPAGATORS` environment variable may select another configured set.

If the header is not present, the middleware allows normal OpenTelemetry behavior
(new traces are created as needed).

Threat model: incoming propagation headers are unauthenticated, since this
middleware runs before any route's auth dependency. Honoring them lets any caller
(with no credential at all) choose the trace context that reaches the operator's
collector. That is standard OpenTelemetry instrumentation behavior, not a
vulnerability in itself, but it is a trust decision: this middleware is only
installed when `accept_incoming_trace_context` is set (default off). Enable it
only for backend/service-to-service deployments where callers are trusted,
ideally behind a proxy that strips propagation headers at the edge.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from opentelemetry import context as otel_context
from opentelemetry import propagate
from opentelemetry.context import Context
from starlette.datastructures import Headers

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send


class TraceContextPropagationMiddleware:
    """Pure ASGI middleware that extracts W3C Trace Context from incoming requests.

    When a request contains supported propagation headers, this middleware
    extracts the context and sets it in the OpenTelemetry context, so subsequent
    spans are linked to the parent trace. Implemented as raw ASGI (like
    `MetricsMiddleware`) rather than `BaseHTTPMiddleware`, whose `call_next`
    returns before a streaming response body finishes sending, which would
    detach the context before streaming spans are done.

    Note: This middleware assumes incoming HTTP requests are the starting point
    for traces, and that any relevant baggage comes from the caller's headers
    (via the propagator's extraction logic). Existing context is preserved, so
    if an outer layer (e.g., auto-instrumentation) has already set up a context
    and spans, the incoming trace context becomes the new parent.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process incoming request and extract trace context.

        Extracts incoming context if supported headers are present and sets the
        OpenTelemetry context before passing the request to the next
        middleware/handler. Uses pure ASGI so context detachment happens after
        streaming responses are fully sent.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        extracted_context = extract_trace_context(Headers(scope=scope))

        token = otel_context.attach(extracted_context)
        try:
            # Awaiting here spans the full response, including a streamed body,
            # so the context stays attached until streaming completes.
            await self.app(scope, receive, send)
        finally:
            otel_context.detach(token)


def extract_trace_context(carrier: Mapping[str, str]) -> Context:
    """Extract incoming context using OpenTelemetry's configured propagators.

    The default OpenTelemetry configuration includes W3C Trace Context
    (`traceparent` and `tracestate`). The `OTEL_PROPAGATORS` environment
    variable controls the global propagator set used here, which may use
    different carriers.

    Args:
        carrier: A mapping of request headers (e.g. `request.headers`).

    Returns:
        An OpenTelemetry Context with whatever the configured propagator could
        extract from the carrier.
    """
    # Extract into the current context so we preserve any existing baggage,
    # suppress-instrumentation flags, or enclosing spans that may already
    # be set (e.g., by an outer instrumentation layer).
    context = propagate.extract(
        carrier=carrier, context=otel_context.get_current()
    )

    return context
