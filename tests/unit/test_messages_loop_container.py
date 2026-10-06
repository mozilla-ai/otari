"""The Messages loop reports the sandbox a request holds where Anthropic reports its own.

Anthropic puts ``container`` on the message, both on the final response and on
the ``message_start`` event, and its SDK clients read it there to resume the
workspace next turn. The loop strategy is what puts the gateway's there.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

from any_llm.types.messages import MessageResponse, MessageStartEvent, MessageUsage, TextBlock

from gateway.services.code_execution import ContainerLease
from gateway.services.mcp_loop_messages import _MESSAGES_STRATEGY, _MessagesToolLoopStrategy, _strategy_for


def _lease() -> ContainerLease:
    now = datetime.now(UTC)
    return ContainerLease(
        container_id="otari_cntr_abc",
        provider="e2b",
        provider_session_id="sbx_1",
        expires_at=now + timedelta(minutes=10),
        hard_expires_at=now + timedelta(hours=1),
    )


def _message() -> MessageResponse:
    return MessageResponse(
        id="msg_1",
        type="message",
        role="assistant",
        model="m",
        content=[TextBlock(type="text", text="done", citations=None)],
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=MessageUsage(input_tokens=1, output_tokens=1),
    )


def test_the_final_message_carries_the_container_in_anthropics_shape() -> None:
    lease = _lease()
    strategy = _MessagesToolLoopStrategy(container=lease)
    result = _message()

    strategy.fold_usage(result, strategy.new_usage_accumulator())

    assert result.container is not None
    assert result.container.id == "otari_cntr_abc"
    assert result.container.expires_at == lease.expires_at
    # The wire shape an Anthropic SDK client parses back.
    dumped = result.model_dump(exclude_none=True)
    assert dumped["container"]["id"] == "otari_cntr_abc"


def test_message_start_carries_the_container_too() -> None:
    lease = _lease()
    strategy = _MessagesToolLoopStrategy(container=lease)
    event = MessageStartEvent(type="message_start", message=_message())

    state, acc = strategy.new_stream_state(), strategy.new_stream_accumulator()
    action, forwarded = strategy.observe(state, event, _NoTools(), acc)

    assert forwarded is event
    assert event.message.container is not None
    assert event.message.container.id == "otari_cntr_abc"


def test_without_a_container_nothing_is_added_and_the_shared_strategy_is_reused() -> None:
    result = _message()
    _MESSAGES_STRATEGY.fold_usage(result, _MESSAGES_STRATEGY.new_usage_accumulator())
    assert result.container is None
    assert _strategy_for(frozenset(), None) is _MESSAGES_STRATEGY
    assert _strategy_for(frozenset(), None, container=_lease()) is not _MESSAGES_STRATEGY


class _NoTools:
    openai_tools: list[dict[str, Any]] = []

    def owns_tool(self, name: str) -> bool:
        return False

    def purpose_hints(self) -> list[tuple[str, str]]:
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        raise AssertionError("no tool should run")
