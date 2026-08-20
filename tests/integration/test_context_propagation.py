"""Integration tests for HTTP trace-context propagation."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
)
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_chat_request_propagates_trace_context_to_provider_span(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """A real chat request makes provider-created spans use its traceparent."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)

    async def mock_acompletion(**_: Any) -> ChatCompletion:
        with tracer.start_as_current_span("provider_request"):
            return ChatCompletion(
                id="chatcmpl-test",
                object="chat.completion",
                created=1700000000,
                model="test-model",
                choices=[
                    Choice(
                        index=0,
                        message=ChatCompletionMessage(role="assistant", content="ok"),
                        finish_reason="stop",
                    )
                ],
                usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        response = client.post(
            "/v1/chat/completions",
            headers={
                **master_key_header,
                "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                "tracestate": "vendor=value",
            },
            json={
                "model": "ollama:test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "user": test_user["user_id"],
            },
        )

    assert response.status_code == 200, response.text
    spans = exporter.get_finished_spans()
    provider_span = next(span for span in spans if span.name == "provider_request")
    assert format(provider_span.context.trace_id, "032x") == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert provider_span.context.trace_state.get("vendor") == "value"
