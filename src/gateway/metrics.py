"""Prometheus registry, metric types, and HTTP request instrumentation for the gateway.

The metric types are re-exported so that code declaring a metric need not depend on ``prometheus_client`` directly.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    ProcessCollector,
    generate_latest,
)
from starlette.responses import Response

from gateway.core.config import API_ROOT, API_VERSION

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

__all__ = ["Counter", "Gauge", "Histogram", "MetricsMiddleware", "REGISTRY", "metrics_endpoint"]

REGISTRY = CollectorRegistry()

# process_resident_memory_bytes and friends. The gateway keeps its own registry
# rather than prometheus_client's default one, which is where these live by
# default, so without this /metrics reports no process memory at all and the
# only way to see the resident set is a shell inside the container.
PROCESS_COLLECTOR = ProcessCollector(registry=REGISTRY)

REQUESTS = Counter(
    "gateway_requests",
    "Total number of HTTP requests",
    ["method", "endpoint", "api_version", "status"],
    registry=REGISTRY,
)

REQUEST_DURATION_SECONDS = Histogram(
    "gateway_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint", "api_version"],
    registry=REGISTRY,
)

ACTIVE_REQUESTS = Gauge(
    "gateway_active_requests",
    "Number of currently in-flight requests",
    registry=REGISTRY,
)


_PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

_UNMATCHED_ENDPOINT = "unmatched"
# What a request outside the API root reports for its version.
_NO_VERSION = ""


def _endpoint_label(scope: Scope) -> tuple[str, str]:
    """Return the resource a request matched and the API version it came in under.

    Labeling metrics with the raw ``scope["path"]`` mints a new Prometheus
    series per distinct path parameter value (file id, batch id, per-user
    lookups, ...), which is unbounded label cardinality. FastAPI records the
    matched route on the scope once routing finishes, so its ``path`` template
    (for example ``/files/{file_id}``) collapses those into a single series.
    Requests that match no route land in the ``unmatched`` bucket.

    The API root is split off into its own label so a series survives the root
    moving, and so two versions served side by side stay distinguishable
    instead of summing into one. A path outside the root keeps its whole
    template: ``/metrics`` is ours to name, and an OTel signal path is not.
    """
    route = scope.get("route")
    template = getattr(route, "path", None)
    if not isinstance(template, str) or not template:
        return _UNMATCHED_ENDPOINT, _NO_VERSION
    if template == API_ROOT:
        return "/", API_VERSION
    if template.startswith(f"{API_ROOT}/"):
        return template[len(API_ROOT) :], API_VERSION
    return template, _NO_VERSION


async def metrics_endpoint(request: Request) -> Response:
    """Serve Prometheus metrics."""
    body = generate_latest(REGISTRY)
    return Response(content=body, media_type=_PROMETHEUS_CONTENT_TYPE)


class MetricsMiddleware:
    """ASGI middleware that records request count, duration, and active requests."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope["path"]
        if path == "/metrics":
            await self.app(scope, receive, send)
            return

        method: str = scope["method"]
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        ACTIVE_REQUESTS.inc()
        start = time.monotonic()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.monotonic() - start
            ACTIVE_REQUESTS.dec()
            endpoint, api_version = _endpoint_label(scope)
            REQUESTS.labels(
                method=method, endpoint=endpoint, api_version=api_version, status=str(status_code)
            ).inc()
            REQUEST_DURATION_SECONDS.labels(method=method, endpoint=endpoint, api_version=api_version).observe(
                duration
            )
