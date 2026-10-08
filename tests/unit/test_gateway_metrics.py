"""Unit tests for gateway Prometheus metrics — no database required."""

import os

import pytest
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from gateway.core.config import API_ROOT, API_VERSION, OTLP_ROOT, GatewayConfig
from gateway.metrics import (
    REGISTRY,
    MetricsMiddleware,
    _endpoint_label,
    _route_template,
    metrics_endpoint,
)


def _sample(name: str, labels: dict[str, str] | None = None) -> float:
    """Read a metric sample value from the registry, returning 0.0 if not found."""
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


# Every family the gateway registers, as (name, type, label names). A metric
# may move to the module that increments it, but the scrape is an external
# contract: dashboards, recording rules and alerts outside this repository
# key on these names and labels.
_EXPOSED_FAMILIES: set[tuple[str, str, tuple[str, ...]]] = {
    ("gateway_abandoned_attempts", "counter", ("provider", "model", "reason", "position")),
    ("gateway_active_requests", "gauge", ()),
    ("gateway_auth_failures", "counter", ("reason",)),
    ("gateway_budget_exceeded", "counter", ()),
    ("gateway_db_pool_capacity", "gauge", ("pool",)),
    ("gateway_db_pool_connections_checked_out", "gauge", ("pool",)),
    ("gateway_db_pool_connections_idle", "gauge", ("pool",)),
    ("gateway_db_pool_overflow_connections", "gauge", ("pool",)),
    ("gateway_inline_cost_settlements", "counter", ("outcome",)),
    ("gateway_rate_limit_hits", "counter", ()),
    ("gateway_rate_limit_model_full", "counter", ("rule", "model")),
    ("gateway_request_cost_dollars", "histogram", ("provider", "model")),
    ("gateway_request_duration_seconds", "histogram", ("method", "endpoint", "api_version")),
    ("gateway_requests", "counter", ("method", "endpoint", "api_version", "status")),
    ("gateway_tokens", "counter", ("provider", "model", "type")),
    ("gateway_usage_log_batch_size", "histogram", ("writer",)),
    ("gateway_usage_log_flush_duration_seconds", "histogram", ("writer", "result")),
    ("gateway_usage_log_queue_depth", "gauge", ()),
    ("gateway_usage_log_rows", "counter", ("writer", "result")),
}


def test_scrape_exposes_the_pinned_families() -> None:
    """The set of gateway metric families, with their types and label names, is fixed.

    A labeled family with no series yet shows only its HELP and TYPE lines in a
    scrape, so the label names are read off the collector, or off the family it
    yields where the collector is a custom one that keeps no label names.
    """
    import gateway.main  # noqa: F401  # imports every module that registers a metric

    families: set[tuple[str, str, tuple[str, ...]]] = set()
    for collector in REGISTRY._collector_to_names:
        describe = getattr(collector, "describe", collector.collect)
        for metric in describe():
            if metric.name.startswith("gateway_"):
                labelnames = getattr(collector, "_labelnames", ()) or getattr(metric, "_labelnames", ())
                families.add((metric.name, metric.type, tuple(labelnames)))

    assert families == _EXPOSED_FAMILIES


def test_config_enable_metrics_defaults_to_false() -> None:
    config = GatewayConfig()
    assert config.enable_metrics is False


def test_config_enable_metrics_accepted() -> None:
    config = GatewayConfig(enable_metrics=True)
    assert config.enable_metrics is True


def _make_test_app(*, enable_metrics: bool = True) -> FastAPI:
    """Build a minimal FastAPI app with /ok and optionally the metrics middleware + endpoint."""
    app = FastAPI()

    @app.get("/ok")
    async def ok() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/error")
    async def error() -> None:
        raise HTTPException(status_code=503, detail="boom")

    if enable_metrics:
        app.add_middleware(MetricsMiddleware)
        app.add_route("/metrics", metrics_endpoint, methods=["GET"])

    return app


def test_middleware_increments_request_counter() -> None:
    app = _make_test_app()
    client = TestClient(app)
    labels = {"method": "GET", "endpoint": "/ok", "api_version": "", "status": "200"}
    before = _sample("gateway_requests_total", labels)

    client.get("/ok")

    assert _sample("gateway_requests_total", labels) - before == 1.0


def test_middleware_records_duration() -> None:
    app = _make_test_app()
    client = TestClient(app)
    labels = {"method": "GET", "endpoint": "/ok", "api_version": ""}
    before = _sample("gateway_request_duration_seconds_count", labels)

    client.get("/ok")

    assert _sample("gateway_request_duration_seconds_count", labels) - before == 1.0
    assert _sample("gateway_request_duration_seconds_sum", labels) > 0


def test_middleware_tracks_error_status_codes() -> None:
    app = _make_test_app()
    client = TestClient(app, raise_server_exceptions=False)
    labels = {"method": "GET", "endpoint": "/error", "api_version": "", "status": "503"}
    before = _sample("gateway_requests_total", labels)

    client.get("/error")

    assert _sample("gateway_requests_total", labels) - before == 1.0


def test_middleware_labels_parameterized_route_with_template() -> None:
    """Different path params collapse to one series; unknown paths bucket as 'unmatched'."""
    app = FastAPI()

    @app.get(f"{API_ROOT}/files/{{file_id}}")
    async def get_file(file_id: str) -> dict[str, str]:
        return {"id": file_id}

    app.add_middleware(MetricsMiddleware)
    client = TestClient(app, raise_server_exceptions=False)

    template_labels = {"method": "GET", "endpoint": "/files/{file_id}", "api_version": "v1", "status": "200"}
    raw_labels_a = {"method": "GET", "endpoint": "/files/aaa", "api_version": "v1", "status": "200"}
    raw_labels_b = {"method": "GET", "endpoint": "/files/bbb", "api_version": "v1", "status": "200"}
    unmatched_labels = {"method": "GET", "endpoint": "unmatched", "api_version": "", "status": "404"}

    before_template = _sample("gateway_requests_total", template_labels)
    before_unmatched = _sample("gateway_requests_total", unmatched_labels)

    assert client.get(f"{API_ROOT}/files/aaa").status_code == 200
    assert client.get(f"{API_ROOT}/files/bbb").status_code == 200
    assert client.get("/no/such/route").status_code == 404

    # Two distinct ids produce a single labeled series keyed by the route template.
    assert _sample("gateway_requests_total", template_labels) - before_template == 2.0
    # The raw per-id paths must never appear as their own series.
    assert _sample("gateway_requests_total", raw_labels_a) == 0.0
    assert _sample("gateway_requests_total", raw_labels_b) == 0.0
    # Paths that match no route land in the bounded fallback bucket.
    assert _sample("gateway_requests_total", unmatched_labels) - before_unmatched == 1.0


def test_the_label_carries_every_prefix_an_included_router_is_mounted_under() -> None:
    """The gateway mounts its routers as a tree, and the series is keyed on the full path.

    FastAPI records only the route's own path on the request, so a label read
    from it would drop the API root and fold every series into the unversioned
    one. This mounts the way ``register_routers`` does: a prefixed aggregate
    router that includes a prefixed resource router.
    """
    app = FastAPI()
    api = APIRouter(prefix=API_ROOT)
    chat = APIRouter(prefix="/chat")

    @chat.post("/completions")
    async def completions() -> dict[str, str]:
        return {"ok": "yes"}

    @chat.get("/{thread_id}")
    async def thread(thread_id: str) -> dict[str, str]:
        return {"id": thread_id}

    @chat.api_route("/stub/{path:path}", methods=["GET", "POST"])
    async def stub(path: str) -> dict[str, str]:
        return {"path": path}

    api.include_router(chat)
    app.include_router(api)
    app.add_middleware(MetricsMiddleware)
    client = TestClient(app)

    cases = {
        ("POST", f"{API_ROOT}/chat/completions"): "/chat/completions",
        ("GET", f"{API_ROOT}/chat/abc"): "/chat/{thread_id}",
        ("POST", f"{API_ROOT}/chat/stub/deep/er/path"): "/chat/stub/{path:path}",
    }
    before = {
        label: _sample("gateway_requests_total", {"method": m, "endpoint": label, "api_version": "v1", "status": "200"})
        for (m, _), label in cases.items()
    }
    for method, path in cases:
        assert client.request(method, path).status_code == 200
    for (method, _), label in cases.items():
        labels = {"method": method, "endpoint": label, "api_version": "v1", "status": "200"}
        assert _sample("gateway_requests_total", labels) - before[label] == 1.0, label


def test_the_route_template_is_recovered_from_the_request_path() -> None:
    """The prefix is the request path up to the tail the route matched, chosen by its parameters."""
    from starlette.routing import Route

    def scope(route_path: str, path: str, params: dict[str, str] | None = None) -> dict[str, object]:
        route = Route(route_path, endpoint=lambda request: None)
        return {"route": route, "path": path, "path_params": params or {}}

    # A fixed tail, and a tail with a parameter, each behind two prefixes.
    assert _route_template(scope("/completions", "/api/v1/chat/completions")) == "/api/v1/chat/completions"
    assert _route_template(scope("/{id}", "/api/v1/chat/abc", {"id": "abc"})) == "/api/v1/chat/{id}"
    # A catch-all also matches from the root; the parsed parameter picks the real tail.
    assert _route_template(scope("/{path:path}", "/api/v1/chat/a/b", {"path": "a/b"})) == "/api/v1/chat/{path:path}"
    # A route with no prefix, as older FastAPI recorded every route.
    unprefixed = scope("/api/v1/files/{file_id}", "/api/v1/files/x", {"file_id": "x"})
    assert _route_template(unprefixed) == "/api/v1/files/{file_id}"
    # The root path a proxy strips is not part of any template.
    with_root = scope("/completions", "/gw/api/v1/chat/completions")
    with_root["root_path"] = "/gw"
    assert _route_template(with_root) == "/api/v1/chat/completions"
    # No route recorded is no template, which the label reports as unmatched.
    assert _route_template({"path": "/no/such/route"}) is None


def test_the_endpoint_label_does_not_carry_the_api_root() -> None:
    """A metric series has to outlive the root moving, which is why the root is not in it.

    The label is what a dashboard, a recording rule and an alert expression are
    keyed on, and those live outside this repository and far longer than any
    one prefix. Splitting the root off means moving the API renames no series,
    and two roots served side by side stay countable apart instead of summing
    into one.
    """
    endpoint_for = _endpoint_label

    assert endpoint_for(f"{API_ROOT}/chat/completions") == ("/chat/completions", "v1")
    assert endpoint_for(f"{API_ROOT}/files/{{file_id}}") == ("/files/{file_id}", "v1")
    # Outside the root, the template is the whole identity: /metrics is ours to
    # name and an OTel signal path belongs to OTel.
    assert endpoint_for("/metrics") == ("/metrics", "")
    assert endpoint_for(f"{OTLP_ROOT}/v1/traces") == (f"{OTLP_ROOT}/v1/traces", "")
    # A sibling root is not silently folded into this one, or a v2 rollout would
    # be invisible: both versions would sum into one series.
    assert endpoint_for("/api/v2/chat/completions") == ("/api/v2/chat/completions", "")
    # The root is matched on the segment boundary, not as a byte prefix. This is
    # the bug class that has bitten this migration more than once.
    assert endpoint_for(f"{API_ROOT}beta/chat") == (f"{API_ROOT}beta/chat", "")
    assert endpoint_for(f"{API_ROOT}-internal/x") == (f"{API_ROOT}-internal/x", "")
    # The root itself is a resource, not an empty label.
    assert endpoint_for(API_ROOT) == ("/", API_VERSION)
    # The version reported is the one the app was built with, not a literal.
    assert endpoint_for(f"{API_ROOT}/chat/completions")[1] == API_VERSION


def test_middleware_skips_metrics_endpoint() -> None:
    app = _make_test_app()
    client = TestClient(app)
    labels = {"method": "GET", "endpoint": "/metrics", "api_version": "", "status": "200"}
    before = _sample("gateway_requests_total", labels)

    client.get("/metrics")
    client.get("/metrics")

    assert _sample("gateway_requests_total", labels) == before


def test_metrics_endpoint_returns_prometheus_format() -> None:
    app = _make_test_app()
    client = TestClient(app)
    resp = client.get("/metrics")

    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "gateway_requests" in resp.text
    assert "gateway_active_requests" in resp.text


def test_metrics_endpoint_not_present_when_disabled() -> None:
    app = _make_test_app(enable_metrics=False)
    client = TestClient(app)
    resp = client.get("/metrics")

    assert resp.status_code in (404, 405)


def test_active_requests_returns_to_zero() -> None:
    """After a request completes, active_requests gauge should be back to its prior value."""
    app = _make_test_app()
    client = TestClient(app)
    before = _sample("gateway_active_requests")

    client.get("/ok")

    assert _sample("gateway_active_requests") == before


@pytest.mark.skipif(not os.path.exists("/proc/stat"), reason="ProcessCollector needs /proc")
def test_metrics_expose_process_memory() -> None:
    """Without this the only way to read the resident set is a shell in the container."""
    body = generate_latest(REGISTRY).decode()

    assert "process_resident_memory_bytes" in body
    assert _sample("process_resident_memory_bytes") > 0
