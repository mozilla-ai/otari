"""A search's query and its hits' URLs reach no record of the gateway's own (design, section 7).

Any-search's and any-fetch's ``fake`` providers raise with the query or the URL
in their message, as a careless adapter might. Neither may reach a log record,
a span attribute, a tally entry, a kept failure or the text the model reads.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import any_fetch
import any_search
import gateway.services.tools._web_search._backend as backend_module
from gateway.core.config import GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.services.tool_usage import ToolUsageTally
from gateway.services.tools import WEB_SEARCH_TOOL_NAME, WebSearchBackend, WebToolCallCap, resolve_in_loop_search
from gateway.services.traces import RequestTrace

_QUERY = "sentinel-query-5d1e"
_URL = "https://example.com/sentinel-path-5d1e"


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Every record at DEBUG, the ``gateway`` logger's included, which does not propagate."""
    any_search.install_log_filter()
    any_fetch.install_log_filter()
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="gateway")
    try:
        yield caplog
    finally:
        gateway_logger.removeHandler(caplog.handler)


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exporter, provider = InMemorySpanExporter(), TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(backend_module, "_tracer", provider.get_tracer(backend_module.__name__))
    return exporter


def _trace() -> RequestTrace:
    return RequestTrace(
        request_id="req-1",
        workspace_id=uuid.uuid4(),
        user_id="alice",
        api_key_id="key-1",
        endpoint="/v1/chat/completions",
        started_at=datetime.now(UTC),
        max_spans=10,
    )


async def _searched(config: GatewayConfig, *, tally: ToolUsageTally, trace: RequestTrace) -> tuple[str, str]:
    search = resolve_in_loop_search(config, None)
    assert search is not None
    backend = WebSearchBackend(search, cap=WebToolCallCap(10), max_results=5, tally=tally, trace=trace)
    async with backend:
        text = await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": _QUERY})
    return text, str(backend.take_last_failure())


@pytest.mark.asyncio
async def test_a_query_a_provider_raises_with_reaches_no_record(
    logs: pytest.LogCaptureFixture, spans: InMemorySpanExporter
) -> None:
    config = GatewayConfig(
        search_tools={"leaky": {"provider": "fake", "options": {"leak_query": True}}}, web_search_default_tool="leaky"
    )
    tally, trace = ToolUsageTally(), _trace()

    text, failure = await _searched(config, tally=tally, trace=trace)

    assert text == "[tool error] fake search failed (internal_error)"
    records = [
        text,
        failure,
        str(tally.meters()),
        str(tally.label_calls()),
        str(trace.spans),
        str([dict(span.attributes or {}) for span in spans.get_finished_spans()]),
        logs.text,
    ]
    assert not any("sentinel" in record for record in records)
    assert "failed: RuntimeError" in logs.text, "the failure is logged, by its class alone"


@pytest.mark.asyncio
async def test_a_url_an_enrichment_fetcher_raises_with_reaches_no_record(
    logs: pytest.LogCaptureFixture, spans: InMemorySpanExporter
) -> None:
    config = GatewayConfig(
        search_tools={
            "search": {
                "provider": "fake",
                "options": {"hits": [{"url": _URL, "title": "Hit", "snippet": "Its snippet."}]},
                "fetch_tool": "leaky",
            }
        },
        fetch_tools={"leaky": {"provider": "fake", "options": {"leak_url": True}}},
        web_search_default_tool="search",
    )
    tally, trace = ToolUsageTally(), _trace()

    text, failure = await _searched(config, tally=tally, trace=trace)

    # The hit's own URL is what the model reads of it; the fetcher's exception text is not.
    assert text == f"[1] Hit\n{_URL}\nIts snippet."
    records = [
        failure,
        str(tally.meters()),
        str(tally.label_calls()),
        str(trace.spans),
        str([dict(span.attributes or {}) for span in spans.get_finished_spans()]),
        logs.text,
    ]
    assert not any("sentinel" in record for record in records)
    assert "failed: RuntimeError" in logs.text, "the failure is logged, by its class alone"
