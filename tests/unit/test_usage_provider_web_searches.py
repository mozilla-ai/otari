"""Unit tests for the searches a provider ran itself: read off a response, kept across a stream, tallied."""

from types import SimpleNamespace
from typing import Any

import pytest
from anthropic.types.message_delta_usage import MessageDeltaUsage
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionUsage
from any_llm.types.messages import MessageDeltaEvent, MessageResponse, MessageUsage
from openai.types.responses import ResponseUsage

from gateway.api.routes.chat import _ChatAdapter
from gateway.api.routes.messages import _messages_stream_usage, _MessagesAdapter
from gateway.api.routes.responses import _ResponsesAdapter
from gateway.core.usage import GatewayUsage, provider_web_searches_of
from gateway.services.pricing import billed_provider_searches, tool_rate_lookup
from gateway.services.tool_usage import ToolUsageTally
from gateway.services.tools import WEB_SEARCH_CALL_ID_PREFIX
from gateway.streaming import merge_stream_usage


def test_a_response_carries_the_searches_anthropic_ran() -> None:
    result = MessageResponse.model_construct(
        usage=MessageUsage.model_validate(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 1},
            }
        )
    )

    usage = _MessagesAdapter().extract_usage(result)

    assert isinstance(usage, GatewayUsage)
    assert usage.provider_web_searches == 2


def test_a_response_with_no_server_tool_use_carries_none() -> None:
    result = MessageResponse.model_construct(usage=MessageUsage.model_validate({"input_tokens": 1, "output_tokens": 1}))

    usage = _MessagesAdapter().extract_usage(result)

    assert usage is not None
    assert provider_web_searches_of(usage) == 0
    assert provider_web_searches_of(CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)) == 0


def test_a_stream_keeps_the_count_its_final_delta_reports() -> None:
    """Anthropic's message_delta usage is cumulative, so the last report is the count, not a sum."""
    delta = MessageDeltaEvent.model_construct(
        usage=MessageDeltaUsage.model_validate(
            {"output_tokens": 30, "server_tool_use": {"web_search_requests": 3, "web_fetch_requests": 0}}
        )
    )
    start = GatewayUsage(prompt_tokens=100, completion_tokens=0, total_tokens=100)

    reported = _messages_stream_usage(delta)
    assert reported is not None
    merged = merge_stream_usage(
        merge_stream_usage(start, reported), GatewayUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    )

    assert provider_web_searches_of(merged) == 3


def _responses_usage(**extra: Any) -> ResponseUsage:
    return ResponseUsage.model_validate(
        {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 50,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 150,
            **extra,
        }
    )


def _search_call(item_id: str) -> dict[str, Any]:
    return {"type": "web_search_call", "id": item_id, "status": "completed"}


def test_a_responses_response_carries_the_searches_openai_reports() -> None:
    """The reported count wins over the items, which a response may not list one per search."""
    result = SimpleNamespace(
        usage=_responses_usage(tool_usage={"web_search": {"num_requests": 2}}), output=[_search_call("ws_1")]
    )

    usage = _ResponsesAdapter().extract_usage(result)  # type: ignore[arg-type]

    assert usage is not None
    assert provider_web_searches_of(usage) == 2


def test_a_responses_response_with_no_reported_count_counts_its_provider_search_items() -> None:
    """The gateway's own ``web_search_call`` items are its searches, billed apart, never the provider's."""
    result = SimpleNamespace(
        usage=_responses_usage(),
        output=[
            _search_call("ws_1"),
            _search_call("ws_2"),
            _search_call(f"{WEB_SEARCH_CALL_ID_PREFIX}abc"),
            {"type": "message", "id": "msg_1"},
        ],
    )

    usage = _ResponsesAdapter().extract_usage(result)  # type: ignore[arg-type]

    assert usage is not None
    assert provider_web_searches_of(usage) == 2


@pytest.mark.parametrize(
    "tool_usage",
    [None, "two", {"web_search": None}, {"web_search": {"num_requests": "2"}}, {"web_search": {"num_requests": -1}}],
)
def test_a_malformed_reported_count_falls_back_to_the_items(tool_usage: Any) -> None:
    """A provider-controlled ``tool_usage`` that is not a count is ignored, never billed as one."""
    result = SimpleNamespace(usage=_responses_usage(tool_usage=tool_usage), output=[_search_call("ws_1")])

    usage = _ResponsesAdapter().extract_usage(result)  # type: ignore[arg-type]

    assert usage is not None
    assert provider_web_searches_of(usage) == 1


def test_a_responses_stream_reads_the_searches_off_its_terminal_response() -> None:
    """A stream reports its searches on the terminal event's response, where its usage is."""
    completed = SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(usage=_responses_usage(tool_usage={"web_search": {"num_requests": 3}}), output=[]),
    )

    usage = _ResponsesAdapter().extract_stream_usage(completed)  # type: ignore[arg-type]

    assert usage is not None
    assert provider_web_searches_of(usage) == 3


def test_the_tally_counts_provider_searches_under_their_own_label() -> None:
    tally = ToolUsageTally()
    tally.record_result("web_search", "results")
    tally.record_provider_searches("anthropic", 2, deployment_paid=False)
    tally.record_provider_searches("anthropic", 0, deployment_paid=False)

    assert tally.meters() == {"web_search": {"billed": 3, "errors": 0}}
    [own, provider] = sorted(tally.label_calls(), key=lambda entry: entry.label or "")
    assert (own.label, own.calls) == (None, 1)
    assert (provider.label, provider.rate_keys, provider.calls, provider.unreported_calls) == (
        "anthropic:web_search",
        ("anthropic:web_search",),
        2,
        2,
    )
    assert provider.deployment_paid is False


def test_whose_overrides_price_a_provider_search_follows_the_call_not_the_label() -> None:
    """A search on a tenant's own Anthropic key takes the tenant's override.

    One on the deployment's own key takes only the operator's, as the call's tokens do.
    """
    for deployment_paid in (False, True):
        tally = ToolUsageTally()
        tally.record_provider_searches("anthropic", 1, deployment_paid=deployment_paid)
        [entry] = tally.label_calls()

        lookup = tool_rate_lookup(entry)

        assert lookup is not None
        assert lookup.rate_keys == ("anthropic:web_search",)
        assert lookup.operator_set_only is deployment_paid


def _grounded_usage(searches: int) -> dict[str, Any]:
    return {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "server_tool_use": {"web_search_requests": searches},
    }


def test_a_chat_completion_carries_the_searches_gemini_grounding_ran() -> None:
    """any-llm reports Gemini's Google Search queries as ``server_tool_use.web_search_requests``."""
    result = ChatCompletion.model_validate(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "gemini-3-flash",
            "choices": [],
            "usage": _grounded_usage(2),
        }
    )

    usage = _ChatAdapter().extract_usage(result)

    assert usage is not None
    assert provider_web_searches_of(usage) == 2


def test_a_chat_stream_reads_the_searches_off_its_usage_chunk() -> None:
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gemini-3-flash",
            "choices": [],
            "usage": _grounded_usage(3),
        }
    )

    usage = _ChatAdapter().extract_stream_usage(chunk)

    assert usage is not None
    assert provider_web_searches_of(usage) == 3


@pytest.mark.parametrize(
    ("model", "billed"),
    [
        ("gemini-2.5-flash", 1),
        ("publishers/google/models/gemini-2.0-flash-001", 1),
        ("gemini-3-flash-preview", 3),
        ("gemini-3.1-flash-lite", 3),
        ("gpt-5", 3),
        ("claude-sonnet-4-5", 3),
    ],
)
def test_gemini_before_3_bills_one_search_per_grounded_prompt(model: str, billed: int) -> None:
    """Gemini 2.5 and earlier bill a grounded prompt once, however many queries it ran; Gemini 3 bills each query."""
    assert billed_provider_searches(model, 3) == billed
