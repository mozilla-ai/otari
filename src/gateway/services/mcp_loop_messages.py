"""Anthropic Messages API variant of the MCP tool-use loop.

Mirrors :mod:`gateway.services.mcp_loop` but speaks the Anthropic wire shape:
``content`` blocks instead of ``tool_calls``, ``tool_use`` / ``tool_result``
blocks, ``stop_reason == "tool_use"`` as the round-continuation signal.

The loop skeleton itself lives in :mod:`gateway.services._tool_loop`; this
module supplies the Anthropic strategy and thin public wrappers.

The duck-typed pool interface (``owns_tool`` / ``call_tool`` /
``openai_tools`` / ``purpose_hints``) is reused unchanged; the
``openai_tools`` shape is converted at the boundary in :mod:`tool_format`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from any_llm import amessages

from gateway.log_config import logger
from gateway.services._tool_loop import StreamAction, run_tool_loop, run_tool_loop_stream
from gateway.services.mcp_loop import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    MAX_TOOL_ITERATIONS_CAP,
    MaxToolIterationsExceeded,
    ToolBackend,
)
from gateway.services.tool_format import openai_to_anthropic_tools

if TYPE_CHECKING:
    from any_llm.types.messages import (
        MessageResponse,
        MessageStreamEvent,
    )


# Re-export so callers in routes/messages.py have a single import surface.
__all__ = [
    "DEFAULT_MAX_TOOL_ITERATIONS",
    "MAX_TOOL_ITERATIONS_CAP",
    "MaxToolIterationsExceeded",
    "anthropic_tool_loop",
    "anthropic_tool_loop_stream",
]


def _split_tool_uses(
    content: list[Any],
    pool: ToolBackend,
) -> tuple[list[Any], bool]:
    """Return (owned_tool_use_blocks, has_foreign).

    Walks ``content`` for blocks with ``type == "tool_use"`` and partitions
    them by ``pool.owns_tool(block.name)``. Foreign = caller-supplied tool the
    gateway can't execute itself; the caller dispatches it.
    """
    owned: list[Any] = []
    has_foreign = False
    for block in content:
        if getattr(block, "type", None) != "tool_use":
            continue
        if pool.owns_tool(block.name):
            owned.append(block)
        else:
            has_foreign = True
    return owned, has_foreign


async def _execute_tool_uses(
    pool: ToolBackend,
    blocks: list[Any],
) -> list[dict[str, Any]]:
    """Run each owned tool_use block and return the Anthropic tool_result blocks.

    Tool failures convert to a ``[tool error] ...`` string in the result so the
    model can recover. Only cancellation-class exceptions
    (``asyncio.CancelledError``, ``KeyboardInterrupt``) escape; they inherit
    from ``BaseException`` and skip the ``Exception`` clause. Same idiom as
    :func:`gateway.services.mcp_loop._execute_mcp_calls`.
    """
    out: list[dict[str, Any]] = []
    for block in blocks:
        try:
            text = await pool.call_tool(block.name, dict(block.input or {}))
        except Exception as exc:  # noqa: BLE001 — see docstring
            logger.warning("MCP tool %s execution failed: %s", block.name, exc)
            text = f"[tool error] {exc}"
        out.append({"type": "tool_result", "tool_use_id": block.id, "content": text})
    return out


def _content_to_dicts(content: list[Any]) -> list[dict[str, Any]]:
    """Serialize a list of Anthropic content blocks back to wire shape.

    The model returned them as pydantic objects (TextBlock, ToolUseBlock,
    ThinkingBlock, ...); when we feed them back as an assistant message on the
    next turn, Anthropic expects plain dicts.
    """
    out: list[dict[str, Any]] = []
    for block in content:
        if hasattr(block, "model_dump"):
            out.append(block.model_dump(exclude_none=True))
        elif isinstance(block, dict):
            out.append(block)
        else:
            # Defensive: any_llm should always hand us pydantic models, but
            # if a provider adapter returns a raw dict-like, accept it.
            out.append(dict(block))
    return out


def _fold_usage(result: MessageResponse, input_total: int, output_total: int) -> None:
    """Replace ``result.usage`` token counts with the loop's running totals.

    Mirrors :func:`gateway.services.mcp_loop._fold_usage` but in Anthropic
    field naming (``input_tokens`` / ``output_tokens`` instead of
    ``prompt_tokens`` / ``completion_tokens``).
    """
    if result.usage is None:
        return
    result.usage.input_tokens = input_total
    result.usage.output_tokens = output_total


def _maybe_fold_message_delta_usage(event: Any, acc_output_tokens: int) -> Any:
    """Return ``event`` with cumulative output_tokens folded in.

    Pass-through for any event type other than ``message_delta``. For a
    ``message_delta`` carrying a usage block, replaces ``output_tokens`` with
    ``current + acc_output_tokens`` so the stream consumer sees the full
    tool-loop output count instead of only the final iteration's. No-op when
    ``acc_output_tokens`` is zero (single-iteration streams stay byte-exact).
    """
    if acc_output_tokens <= 0:
        return event
    if getattr(event, "type", None) != "message_delta":
        return event
    usage = getattr(event, "usage", None)
    if usage is None or not hasattr(usage, "model_copy"):
        return event
    new_usage = usage.model_copy(
        update={"output_tokens": (getattr(usage, "output_tokens", 0) or 0) + acc_output_tokens}
    )
    return event.model_copy(update={"usage": new_usage})


class _MessagesStreamState:
    """Per-iteration bookkeeping for the Anthropic streaming loop.

    All blocks (text, tool_use, thinking, redacted_thinking, ...) are tracked
    by their original ``content_block_start.index`` so the assistant message
    fed back into the next round preserves the model's original ordering and
    doesn't silently drop non-text / non-tool_use block types. The per-tool_use
    JSON-arg buffer is stored separately so no internal field has to be
    stripped before serializing the assistant message.
    """

    def __init__(self) -> None:
        self.blocks_by_index: dict[int, dict[str, Any]] = {}
        self.tool_use_json_bufs: dict[int, str] = {}
        self.stop_reason: str | None = None
        self.deferred_terminal: list[MessageStreamEvent] = []
        self.owned_specs: list[dict[str, Any]] = []


class _MessagesToolLoopStrategy:
    """Anthropic Messages strategy for the generic tool loop.

    ``amessages`` is resolved as a module global at call time so tests can
    monkeypatch ``gateway.services.mcp_loop_messages.amessages``.
    """

    transcript_key = "messages"

    def coerce_transcript(self, value: Any) -> list[Any]:
        return list(value or [])

    def convert_pool_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return openai_to_anthropic_tools(tools)

    # ---- non-streaming hooks ----

    async def call(self, kwargs: dict[str, Any]) -> MessageResponse:
        result: MessageResponse = await amessages(**kwargs)  # type: ignore[assignment]
        return result

    def new_usage_accumulator(self) -> dict[str, int]:
        return {"input": 0, "output": 0}

    def accumulate_usage(self, acc: dict[str, int], result: MessageResponse) -> None:
        if result.usage:
            acc["input"] += result.usage.input_tokens or 0
            acc["output"] += result.usage.output_tokens or 0

    def fold_usage(self, result: MessageResponse, acc: dict[str, int]) -> None:
        _fold_usage(result, acc["input"], acc["output"])

    def exit_before_split(self, result: MessageResponse) -> bool:
        return False

    def split_owned(self, result: MessageResponse, pool: ToolBackend) -> tuple[list[Any], bool]:
        return _split_tool_uses(list(result.content or []), pool)

    def exit_after_split(self, result: MessageResponse) -> bool:
        # The model emitted tool_use blocks but stopped for another reason
        # (e.g. ``end_turn`` because ``max_tokens`` was hit mid-tool-call):
        # exit rather than try to execute them.
        return result.stop_reason != "tool_use"

    async def execute_owned(self, pool: ToolBackend, owned: list[Any]) -> list[dict[str, Any]]:
        return await _execute_tool_uses(pool, owned)

    def filter_owned(self, result: MessageResponse, owned: list[Any], pool: ToolBackend) -> None:
        # Mixed batch: the owned subset was executed for its side effects;
        # filter it from the returned content so the caller only sees blocks
        # it can dispatch. Mirrors the chat-completions mixed-batch handling.
        owned_ids = {b.id for b in owned}
        content = list(result.content or [])
        try:
            result.content = [
                b
                for b in content
                if not (getattr(b, "type", None) == "tool_use" and getattr(b, "id", None) in owned_ids)
            ]
        except (AttributeError, TypeError):
            logger.warning(
                "Anthropic-mixed: could not filter content on response; client will see tool_use "
                "blocks the gateway already executed (no-op on the client side).",
            )

    async def advance_transcript(
        self,
        transcript: list[Any],
        result: MessageResponse,
        owned: list[Any],
        pool: ToolBackend,
    ) -> None:
        # All-owned: continue the loop. Append the assistant turn (so the model
        # sees its own tool_use blocks) and a user turn carrying tool_result.
        content = list(result.content or [])
        transcript.append({"role": "assistant", "content": _content_to_dicts(content)})
        transcript.append({"role": "user", "content": await _execute_tool_uses(pool, owned)})

    # ---- streaming hooks ----

    async def open_stream(self, kwargs: dict[str, Any]) -> AsyncIterator[MessageStreamEvent]:
        stream: AsyncIterator[MessageStreamEvent] = await amessages(**kwargs)  # type: ignore[assignment]
        return stream

    def new_stream_state(self) -> _MessagesStreamState:
        return _MessagesStreamState()

    def new_stream_accumulator(self) -> dict[str, int]:
        # output_tokens from dropped intermediate ``message_delta`` events.
        # The final iteration's forwarded ``message_delta`` is modified to
        # carry the cumulative total so streaming usage reporting downstream
        # sees the full tool-loop output, not just the final round's tokens.
        return {"output_tokens": 0}

    def observe(self, state: _MessagesStreamState, event: MessageStreamEvent) -> StreamAction:
        event_type = getattr(event, "type", None)

        if event_type == "content_block_start":
            block = event.content_block  # type: ignore[union-attr]
            idx = event.index  # type: ignore[union-attr]
            if hasattr(block, "model_dump"):
                state.blocks_by_index[idx] = block.model_dump(exclude_none=True)
            else:
                state.blocks_by_index[idx] = dict(block) if isinstance(block, dict) else {}
            if state.blocks_by_index[idx].get("type") == "tool_use":
                state.tool_use_json_bufs[idx] = ""

        elif event_type == "content_block_delta":
            idx = event.index  # type: ignore[union-attr]
            delta = event.delta  # type: ignore[union-attr]
            dtype = getattr(delta, "type", None)
            block_dict = state.blocks_by_index.get(idx)
            if block_dict is None:
                pass  # delta for an unknown index; defensive no-op
            elif dtype == "input_json_delta" and idx in state.tool_use_json_bufs:
                state.tool_use_json_bufs[idx] += getattr(delta, "partial_json", "") or ""
            elif dtype == "text_delta":
                block_dict["text"] = (block_dict.get("text") or "") + (getattr(delta, "text", "") or "")
            elif dtype == "thinking_delta":
                block_dict["thinking"] = (block_dict.get("thinking") or "") + (
                    getattr(delta, "thinking", "") or ""
                )
            elif dtype == "signature_delta":
                block_dict["signature"] = (block_dict.get("signature") or "") + (
                    getattr(delta, "signature", "") or ""
                )

        elif event_type == "message_delta":
            state.stop_reason = getattr(event.delta, "stop_reason", None) or state.stop_reason  # type: ignore[union-attr]
            state.deferred_terminal.append(event)
            return StreamAction.DEFER

        elif event_type == "message_stop":
            state.deferred_terminal.append(event)
            return StreamAction.BREAK

        return StreamAction.FORWARD

    def stream_exiting(self, state: _MessagesStreamState, pool: ToolBackend) -> bool:
        owned_specs: list[dict[str, Any]] = []
        has_foreign = False
        for idx in sorted(state.blocks_by_index):
            block_dict = state.blocks_by_index[idx]
            if block_dict.get("type") != "tool_use":
                continue
            if pool.owns_tool(block_dict.get("name", "")):
                owned_specs.append({"index": idx, **block_dict})
            else:
                has_foreign = True
        state.owned_specs = owned_specs
        # Loop exits when: no tool_use blocks, stop_reason isn't tool_use,
        # the batch is mixed/foreign, or nothing owned. In all of those cases
        # the deferred terminal events get forwarded (and the message_delta
        # is rewritten to carry cumulative output_tokens from any prior
        # dropped iterations).
        return not owned_specs or has_foreign or state.stop_reason != "tool_use"

    def terminal_events(self, state: _MessagesStreamState, acc: dict[str, int]) -> list[MessageStreamEvent]:
        return [_maybe_fold_message_delta_usage(term, acc["output_tokens"]) for term in state.deferred_terminal]

    def accumulate_stream_usage(self, acc: dict[str, int], state: _MessagesStreamState) -> None:
        # All-owned continuation: fold this iteration's output_tokens (from
        # the dropped terminal) into the running total for the final fold.
        for term in state.deferred_terminal:
            if getattr(term, "type", None) == "message_delta":
                usage = getattr(term, "usage", None)
                if usage is not None:
                    acc["output_tokens"] += getattr(usage, "output_tokens", 0) or 0

    async def advance_stream_transcript(
        self,
        transcript: list[Any],
        state: _MessagesStreamState,
        pool: ToolBackend,
    ) -> None:
        # Assistant message for the next round; preserve original block
        # ordering. tool_use blocks pick up the parsed input from their
        # JSON buffer.
        assistant_content: list[dict[str, Any]] = []
        for idx in sorted(state.blocks_by_index):
            block_dict = state.blocks_by_index[idx]
            if block_dict.get("type") == "tool_use":
                try:
                    parsed_input = json.loads(state.tool_use_json_bufs.get(idx, "") or "{}")
                except json.JSONDecodeError:
                    parsed_input = {}
                block_dict = {**block_dict, "input": parsed_input}
            assistant_content.append(block_dict)

        # Execute owned tool_use blocks and build tool_result blocks for the
        # next user message.
        tool_results: list[dict[str, Any]] = []
        for spec in state.owned_specs:
            try:
                parsed_input = json.loads(state.tool_use_json_bufs.get(spec["index"], "") or "{}")
            except json.JSONDecodeError:
                parsed_input = {}
            try:
                text = await pool.call_tool(spec["name"], parsed_input)
            except Exception as exc:  # noqa: BLE001 — same tool-error-as-message idiom as the non-stream loop
                logger.warning("MCP tool %s execution failed: %s", spec["name"], exc)
                text = f"[tool error] {exc}"
            tool_results.append({"type": "tool_result", "tool_use_id": spec["id"], "content": text})

        transcript.append({"role": "assistant", "content": assistant_content})
        transcript.append({"role": "user", "content": tool_results})


_MESSAGES_STRATEGY = _MessagesToolLoopStrategy()


async def anthropic_tool_loop(
    *,
    completion_kwargs: dict[str, Any],
    pool: ToolBackend,
    max_iterations: int,
    on_first_response: Callable[[], None] | None = None,
) -> MessageResponse:
    """Non-streaming Anthropic Messages tool-use loop.

    Each iteration calls ``amessages``, walks the response's content blocks for
    ``tool_use`` entries, and if any are gateway-owned, executes them and
    appends the assistant + tool_result messages for the next round.

    Loop terminates when:
      * the response has no ``tool_use`` blocks (final answer);
      * ``stop_reason != "tool_use"`` (model decided to stop);
      * the response contains foreign ``tool_use`` blocks, which are returned
        to the caller for client-side dispatch. If the batch is mixed
        (owned + foreign), the owned subset is executed for its side effects
        but the response is returned with the owned blocks filtered out so
        the caller only sees what it can dispatch.

    Accumulates usage across iterations into the returned ``MessageResponse``.

    ``on_first_response`` follows the provider lock-in contract documented on
    :func:`gateway.services._tool_loop.run_tool_loop`.
    """
    return await run_tool_loop(
        strategy=_MESSAGES_STRATEGY,
        completion_kwargs=completion_kwargs,
        pool=pool,
        max_iterations=max_iterations,
        on_first_response=on_first_response,
    )


async def anthropic_tool_loop_stream(
    *,
    completion_kwargs: dict[str, Any],
    pool: ToolBackend,
    max_iterations: int,
) -> AsyncIterator[MessageStreamEvent]:
    """Streaming Anthropic Messages tool-use loop.

    Forwards every Anthropic event downstream **except** the terminal
    ``message_delta`` / ``message_stop`` of an iteration that's about to
    continue (a new ``message_start`` after the client thought the message
    ended would confuse most SDK consumers).

    Per iteration:
      1. Set ``stream=True`` on ``amessages`` and iterate the event stream.
      2. Track tool_use content blocks by ``index`` from ``content_block_start``
         (when ``content_block.type == "tool_use"``). Buffer their
         ``input_json_delta`` chunks until ``content_block_stop``.
      3. Yield every event as it arrives (including the tool_use events, so the
         client sees the model's tool intent even mid-loop). Defer
         ``message_delta`` and ``message_stop`` until we know whether the loop
         will continue.
      4. On ``message_stop``: if any buffered tool_use blocks exist AND all
         owned by the pool, execute them, append messages, drop the terminal
         events, and continue. If foreign blocks exist OR no tool_use blocks
         were buffered, forward the terminal events and exit.

    Re-emitting a synthetic ``message_start`` for the next iteration is not
    needed because ``amessages`` produces a fresh stream; the next call's
    natural ``message_start`` arrives downstream as if nothing had happened.
    """
    async for event in run_tool_loop_stream(
        strategy=_MESSAGES_STRATEGY,
        completion_kwargs=completion_kwargs,
        pool=pool,
        max_iterations=max_iterations,
    ):
        yield event
