"""Per-round usage snapshots for the Reprise v0 fingerprint (otari-ai#1647).

The saving estimate v0 exists to produce is mostly the cache re-read, so a
snapshot that records "whatever this provider calls input" understates the
number in the direction that cancels the project. The hooks therefore return a
``GatewayUsage``, which already carries cache reads, cache writes, the 1-hour
write subset, and the ``cache_tokens_in_prompt`` flag that reconciles OpenAI's
"cached tokens are a slice of the prompt" with Anthropic's additive buckets.

Zero and unknown stay distinguishable: a provider that reported nothing yields
``None``, not a zero-filled snapshot.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
    PromptTokensDetails,
)
from any_llm.types.messages import (
    MessageDeltaEvent,
    MessageResponse,
    MessageStartEvent,
    MessageStreamEvent,
    MessageUsage,
)
from any_llm.types.responses import Response
from openai.types.responses import ResponseCompletedEvent, ResponseIncompleteEvent, ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from gateway.core.usage import GatewayUsage
from gateway.services._tool_loop import ToolBackend
from gateway.services.mcp_loop import _CHAT_STRATEGY
from gateway.services.mcp_loop_messages import _MESSAGES_STRATEGY
from gateway.services.mcp_loop_responses import _RESPONSES_STRATEGY


class _NoTools:
    """Minimal :class:`ToolBackend` for driving ``observe`` over raw events."""

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return []

    def owns_tool(self, name: str) -> bool:
        return False

    def purpose_hints(self) -> list[tuple[str, str]]:
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:  # pragma: no cover
        raise AssertionError("no tools in these fixtures")


_POOL: ToolBackend = _NoTools()


def _effective_prompt_tokens(usage: GatewayUsage) -> int:
    """The real input the request paid for, with ``cache_tokens_in_prompt`` applied."""
    if usage.cache_tokens_in_prompt:
        return usage.prompt_tokens
    return usage.prompt_tokens + usage.cache_read_tokens + usage.cache_write_tokens


# ---------- fixtures per format ----------


def _chat_completion(usage: CompletionUsage | None) -> ChatCompletion:
    return ChatCompletion(
        id="cmpl-1",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="done"),
            )
        ],
        created=0,
        model="fake",
        object="chat.completion",
        usage=usage,
    )


def _chat_usage_chunk(usage: CompletionUsage) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="cmpl-1",
        choices=[],
        created=0,
        model="fake",
        object="chat.completion.chunk",
        usage=usage,
    )


def _chat_terminal_chunk() -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "cmpl-1",
            "choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}],
            "created": 0,
            "model": "fake",
            "object": "chat.completion.chunk",
        }
    )


def _message_usage(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read: int | None = None,
    cache_write: int | None = None,
    cache_write_1h: int | None = None,
) -> MessageUsage:
    return MessageUsage.model_validate(
        {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_write,
            "cache_creation": (
                {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": cache_write_1h}
                if cache_write_1h is not None
                else None
            ),
        }
    )


def _message_response(usage: MessageUsage | None) -> MessageResponse:
    if usage is None:
        # The SDK model requires ``usage``; a provider adapter that hands one back
        # without it is exactly the "reported nothing" case worth covering.
        return MessageResponse.model_construct(usage=None)
    return MessageResponse(
        id="msg_1",
        type="message",
        role="assistant",
        model="fake",
        content=[],
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=cast(Any, usage),
        container=None,
    )


def _message_start(usage: MessageUsage) -> MessageStartEvent:
    return MessageStartEvent(type="message_start", message=cast(Any, _message_response(usage)))


def _message_delta(output_tokens: int) -> MessageDeltaEvent:
    return MessageDeltaEvent.model_validate(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"input_tokens": None, "output_tokens": output_tokens},
        }
    )


def _message_delta_with_iterations(iterations: list[dict[str, Any]]) -> MessageDeltaEvent:
    """A ``message_delta`` whose ``iterations`` sum to what the round is billed.

    The shape ``tests/integration/test_messages_streaming_usage.py`` pins: Anthropic
    reports compaction sampling here and not on ``message_start``, which only ever
    saw the first pass.
    """
    return MessageDeltaEvent.model_validate(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {
                "input_tokens": iterations[-1]["input_tokens"],
                "output_tokens": iterations[-1]["output_tokens"],
                "iterations": iterations,
            },
        }
    )


def _responses_usage(*, input_tokens: int, output_tokens: int, cached: int = 0) -> ResponseUsage:
    return ResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached),
        output_tokens=output_tokens,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=input_tokens + output_tokens,
    )


def _response(usage: ResponseUsage | None, status: str = "completed") -> Response:
    return Response(
        id="resp_1",
        created_at=0.0,
        model="fake",
        object="response",
        status=cast(Any, status),
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=usage,
    )


# ---------- the cross-provider reconciliation ----------


def test_openai_and_anthropic_agree_on_identical_real_token_counts() -> None:
    """1000 in, 800 of it a cache read, 50 out, expressed in each provider's own fields."""
    openai = _CHAT_STRATEGY.usage_snapshot(
        _chat_completion(
            CompletionUsage(
                prompt_tokens=1000,
                completion_tokens=50,
                total_tokens=1050,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=800),
            )
        )
    )
    anthropic = _MESSAGES_STRATEGY.usage_snapshot(
        _message_response(_message_usage(input_tokens=200, output_tokens=50, cache_read=800))
    )

    assert openai is not None
    assert anthropic is not None
    assert openai.cache_tokens_in_prompt is True
    assert anthropic.cache_tokens_in_prompt is False
    assert _effective_prompt_tokens(openai) == _effective_prompt_tokens(anthropic) == 1000
    assert openai.completion_tokens == anthropic.completion_tokens == 50
    assert openai.cache_read_tokens == anthropic.cache_read_tokens == 800
    assert openai.cache_write_tokens == anthropic.cache_write_tokens == 0


def test_anthropic_cache_writes_and_the_1h_breakdown_survive() -> None:
    """Cache writes are the priciest input class, so a snapshot that drops them lies."""
    snapshot = _MESSAGES_STRATEGY.usage_snapshot(
        _message_response(
            _message_usage(
                input_tokens=100,
                output_tokens=20,
                cache_read=40,
                cache_write=15,
                cache_write_1h=10,
            )
        )
    )

    assert snapshot is not None
    assert snapshot.cache_write_tokens == 15
    assert snapshot.cache_write_1h_tokens == 10
    assert _effective_prompt_tokens(snapshot) == 155


# ---------- streaming agrees with non-streaming ----------


def test_chat_streamed_round_matches_the_non_streamed_round() -> None:
    usage = CompletionUsage(
        prompt_tokens=1000,
        completion_tokens=50,
        total_tokens=1050,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=800),
    )
    state = _CHAT_STRATEGY.new_stream_state()
    acc: None = None  # chat's stream accumulator is None by design
    for event in (_chat_terminal_chunk(), _chat_usage_chunk(usage)):
        _CHAT_STRATEGY.observe(state, event, _POOL, acc)

    assert _CHAT_STRATEGY.stream_usage_snapshot(state) == _CHAT_STRATEGY.usage_snapshot(
        _chat_completion(usage)
    )


def test_messages_streamed_round_matches_the_non_streamed_round() -> None:
    """Anthropic splits one round's usage over message_start and message_delta."""
    state = _MESSAGES_STRATEGY.new_stream_state()
    acc = _MESSAGES_STRATEGY.new_stream_accumulator()
    events: list[MessageStreamEvent] = [
        _message_start(
            _message_usage(
                input_tokens=100, output_tokens=0, cache_read=40, cache_write=15, cache_write_1h=10
            )
        ),
        _message_delta(output_tokens=20),
    ]
    for event in events:
        _MESSAGES_STRATEGY.observe(state, event, _POOL, acc)

    assert _MESSAGES_STRATEGY.stream_usage_snapshot(state) == _MESSAGES_STRATEGY.usage_snapshot(
        _message_response(
            _message_usage(
                input_tokens=100, output_tokens=20, cache_read=40, cache_write=15, cache_write_1h=10
            )
        )
    )


def test_responses_streamed_round_matches_the_non_streamed_round() -> None:
    usage = _responses_usage(input_tokens=1000, output_tokens=50, cached=800)
    state = _RESPONSES_STRATEGY.new_stream_state()
    acc = _RESPONSES_STRATEGY.new_stream_accumulator()
    _RESPONSES_STRATEGY.observe(
        state,
        ResponseCompletedEvent(type="response.completed", response=_response(usage), sequence_number=0),
        _POOL,
        acc,
    )

    assert _RESPONSES_STRATEGY.stream_usage_snapshot(state) == _RESPONSES_STRATEGY.usage_snapshot(
        _response(usage)
    )


def test_messages_streamed_compaction_round_matches_the_non_streamed_round() -> None:
    """Only ``message_delta`` carries the iterations, so the input side has to read it.

    Taking the input from ``message_start`` wholesale reports a compaction round's
    output summed over every iteration and its input from only the first, which
    understates the cache re-read the estimate is mostly made of, and disagrees with
    what the same stream is billed.
    """
    iterations: list[dict[str, Any]] = [
        {
            "type": "compaction",
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 30,
        },
        {
            "type": "message",
            "input_tokens": 42,
            "output_tokens": 7,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 5,
        },
    ]
    state = _MESSAGES_STRATEGY.new_stream_state()
    acc = _MESSAGES_STRATEGY.new_stream_accumulator()
    events: list[MessageStreamEvent] = [
        _message_start(_message_usage(input_tokens=42, output_tokens=0)),
        _message_delta_with_iterations(iterations),
    ]
    for event in events:
        _MESSAGES_STRATEGY.observe(state, event, _POOL, acc)

    streamed = _MESSAGES_STRATEGY.stream_usage_snapshot(state)
    non_streamed = _MESSAGES_STRATEGY.usage_snapshot(
        _message_response(
            MessageUsage.model_validate(
                {"input_tokens": 42, "output_tokens": 7, "iterations": iterations}
            )
        )
    )

    assert streamed == non_streamed
    assert streamed is not None
    assert (streamed.prompt_tokens, streamed.completion_tokens) == (142, 27)
    assert streamed.cache_read_tokens == 35


def test_messages_stream_keeps_the_start_input_when_the_delta_reports_none() -> None:
    """The ordinary round: only ``message_start`` knows the input side."""
    state = _MESSAGES_STRATEGY.new_stream_state()
    acc = _MESSAGES_STRATEGY.new_stream_accumulator()
    events: list[MessageStreamEvent] = [
        _message_start(_message_usage(input_tokens=100, output_tokens=0, cache_read=40)),
        _message_delta(output_tokens=20),
    ]
    for event in events:
        _MESSAGES_STRATEGY.observe(state, event, _POOL, acc)

    snapshot = _MESSAGES_STRATEGY.stream_usage_snapshot(state)

    assert snapshot is not None
    assert (snapshot.prompt_tokens, snapshot.cache_read_tokens, snapshot.completion_tokens) == (100, 40, 20)


def test_responses_stream_reports_usage_for_a_round_that_ended_incomplete() -> None:
    """A truncated round was still billed, so it is not an unknown-usage round."""
    usage = _responses_usage(input_tokens=8000, output_tokens=512)
    state = _RESPONSES_STRATEGY.new_stream_state()
    acc = _RESPONSES_STRATEGY.new_stream_accumulator()
    _RESPONSES_STRATEGY.observe(
        state,
        ResponseIncompleteEvent(
            type="response.incomplete",
            response=_response(usage, status="incomplete"),
            sequence_number=0,
        ),
        _POOL,
        acc,
    )

    assert _RESPONSES_STRATEGY.stream_usage_snapshot(state) == _RESPONSES_STRATEGY.usage_snapshot(
        _response(usage, status="incomplete")
    )


# ---------- unknown is not zero ----------


def test_chat_stream_without_include_usage_yields_none() -> None:
    """A caller that set ``include_usage: False`` gets no usage, which is not zero usage."""
    state = _CHAT_STRATEGY.new_stream_state()
    acc: None = None  # chat's stream accumulator is None by design
    _CHAT_STRATEGY.observe(state, _chat_terminal_chunk(), _POOL, acc)

    assert _CHAT_STRATEGY.stream_usage_snapshot(state) is None


@pytest.mark.parametrize(
    ("snapshot", "result"),
    [
        pytest.param(_CHAT_STRATEGY.usage_snapshot, _chat_completion(None), id="chat"),
        pytest.param(_MESSAGES_STRATEGY.usage_snapshot, _message_response(None), id="messages"),
        pytest.param(_RESPONSES_STRATEGY.usage_snapshot, _response(None), id="responses"),
    ],
)
def test_a_result_without_usage_yields_none(snapshot: Any, result: Any) -> None:
    assert snapshot(result) is None


@pytest.mark.parametrize(
    "strategy",
    [
        pytest.param(_MESSAGES_STRATEGY, id="messages"),
        pytest.param(_RESPONSES_STRATEGY, id="responses"),
    ],
)
def test_a_stream_that_reported_nothing_yields_none(strategy: Any) -> None:
    assert strategy.stream_usage_snapshot(strategy.new_stream_state()) is None


# ---------- the loop's own accounting is untouched ----------


def test_snapshot_reads_the_round_while_the_loop_still_folds_the_total() -> None:
    """The snapshot reads the provider's own fields and leaves the accumulators alone.

    The loop's usage accounting exists to fold every round's totals into the one
    response the client sees, and it must keep doing exactly that.
    """
    rounds = [
        _chat_completion(CompletionUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13)),
        _chat_completion(CompletionUsage(prompt_tokens=20, completion_tokens=5, total_tokens=25)),
    ]
    snapshots = [_CHAT_STRATEGY.usage_snapshot(result) for result in rounds]

    acc = _CHAT_STRATEGY.new_usage_accumulator()
    for result in rounds:
        _CHAT_STRATEGY.accumulate_usage(acc, result)
    _CHAT_STRATEGY.fold_usage(rounds[-1], acc)

    assert [(s.prompt_tokens, s.completion_tokens) for s in snapshots if s is not None] == [(10, 3), (20, 5)]
    assert rounds[-1].usage is not None
    assert (rounds[-1].usage.prompt_tokens, rounds[-1].usage.completion_tokens) == (30, 8)
