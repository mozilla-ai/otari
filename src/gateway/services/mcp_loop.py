"""Streaming-aware MCP tool-use loop (OpenAI chat-completions format).

Wraps one or more `acompletion` calls so that when the model emits tool_calls
for tools owned by the MCPClientPool, the loop executes them against the MCP
servers, appends the assistant + tool result messages to the conversation, and
re-calls the provider for the next iteration. Tool calls for user-supplied
(non-MCP) tools end the loop and bubble up to the caller untouched.

Both streaming and non-streaming variants are provided. The streaming variant
yields `ChatCompletionChunk` objects across the entire loop as a single
`AsyncIterator`, which can be fed into the existing `streaming_generator`.

The format-agnostic loop skeleton lives in
:mod:`gateway.services._tool_loop`; this module supplies the chat-completions
strategy (wire-shape helpers, exit predicates, usage folding) and the public
wrappers the routes call.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

from any_llm import acompletion
from any_llm.types.completion import PromptTokensDetails

from gateway.core.env import otari_env
from gateway.log_config import logger
from gateway.services._tool_loop import (
    MaxToolIterationsExceeded,
    StreamAction,
    ToolBackend,
    run_tool_loop,
    run_tool_loop_stream,
)

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk

MAX_TOOL_ITERATIONS_CAP = 25
DEFAULT_MAX_TOOL_ITERATIONS = 10


__all__ = [
    "DEFAULT_MAX_TOOL_ITERATIONS",
    "MAX_TOOL_ITERATIONS_CAP",
    "PURPOSE_HINT_HEADER",
    "MaxToolIterationsExceeded",
    "ToolBackend",
    "inject_purpose_hints",
    "mcp_tool_loop",
    "mcp_tool_loop_stream",
]


# Lead-in for the per-source purpose-hint block we prepend to the system message.
# Generic across MCP servers, the sandbox code-execution tool, and any future
# tool source. Surfaced as a constant so phrasing can be tuned for different
# model families (open-weight models in particular benefit from more directive
# language).
PURPOSE_HINT_HEADER = "You have access to the following tools:"


def inject_purpose_hints(
    messages: list[dict[str, Any]],
    hints: list[tuple[str, str]],
    *,
    header: str | None = None,
) -> list[dict[str, Any]]:
    """Prepend or extend the system message with per-tool usage hints.

    Header resolution priority:
      1. ``header`` arg (per-request override, set from the request body)
      2. ``OTARI_TOOLS_HEADER`` env (per-deployment override)
      3. :data:`PURPOSE_HINT_HEADER` built-in default
    """
    if not hints:
        return messages

    effective_header = header or otari_env("TOOLS_HEADER") or PURPOSE_HINT_HEADER
    lines = [effective_header]
    for name, hint in hints:
        lines.append(f"- {name}: {hint}")
    block = "\n".join(lines)

    out = list(messages)
    if out and out[0].get("role") == "system":
        existing = out[0].get("content") or ""
        out[0] = {**out[0], "content": f"{existing}\n\n{block}" if existing else block}
    else:
        out.insert(0, {"role": "system", "content": block})
    return out


def _accumulate_tool_call_deltas(slots: dict[int, dict[str, Any]], deltas: list[Any]) -> None:
    """Merge incremental streaming tool_call deltas into per-index slots."""
    for delta in deltas:
        idx = delta.index
        slot = slots.setdefault(idx, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
        if getattr(delta, "id", None):
            slot["id"] = delta.id
        if getattr(delta, "type", None):
            slot["type"] = delta.type
        fn = getattr(delta, "function", None)
        if fn is not None:
            if getattr(fn, "name", None):
                slot["function"]["name"] += fn.name
            if getattr(fn, "arguments", None):
                slot["function"]["arguments"] += fn.arguments


def _finalize_tool_calls(slots: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [slots[i] for i in sorted(slots)]


def _execute_split(tool_calls: list[dict[str, Any]], pool: ToolBackend) -> tuple[list[dict[str, Any]], bool]:
    """Return (mcp_owned_calls, has_foreign_calls). Foreign = user-supplied, gateway can't execute."""
    mcp_calls: list[dict[str, Any]] = []
    has_foreign = False
    for tc in tool_calls:
        name = tc.get("function", {}).get("name", "")
        if pool.owns_tool(name):
            mcp_calls.append(tc)
        else:
            has_foreign = True
    return mcp_calls, has_foreign


async def _execute_mcp_calls(pool: ToolBackend, mcp_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run each MCP tool call and return the resulting tool-role messages.

    Tool failures (network errors, server errors, schema mismatches, MCP-specific
    or httpx-level transport errors) are converted to a ``[tool error] ...``
    message so the model can recover. Only cancellation/interrupt-class
    exceptions (``asyncio.CancelledError``, ``KeyboardInterrupt``) escape; they
    inherit from ``BaseException`` and never reach the ``Exception`` clause.
    That's the standard idiom for "treat tool failures as recoverable, let
    cancellation propagate".
    """
    out: list[dict[str, Any]] = []
    for tc in mcp_calls:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        try:
            text = await pool.call_tool(name, args)
        except Exception as exc:  # noqa: BLE001 — see docstring
            logger.warning("MCP tool %s execution failed: %s", name, exc)
            text = f"[tool error] {exc}"
        out.append({"role": "tool", "tool_call_id": tc["id"] or "", "content": text})
    return out


def _fold_usage(
    completion: ChatCompletion,
    prompt_total: int,
    completion_total: int,
    cache_read_total: int = 0,
) -> None:
    if completion.usage is None:
        return
    completion.usage.prompt_tokens = prompt_total
    completion.usage.completion_tokens = completion_total
    completion.usage.total_tokens = prompt_total + completion_total
    # OpenAI chat reports cached tokens as a subset of prompt_tokens. Fold the
    # accumulated read count back into prompt_tokens_details so the downstream
    # GatewayUsage wrapper forwards the loop-wide total, not just the last
    # iteration's slice. (Chat has no cache-write concept.)
    details = completion.usage.prompt_tokens_details
    if details is not None:
        details.cached_tokens = cache_read_total
    elif cache_read_total > 0:
        # The final iteration carried no prompt_tokens_details, but an earlier
        # one did; create the sub-object so the accumulated count is not lost.
        completion.usage.prompt_tokens_details = PromptTokensDetails(cached_tokens=cache_read_total)


class _ChatStreamState:
    """Per-iteration bookkeeping for the chat streaming loop."""

    def __init__(self) -> None:
        self.slots: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.pending_terminal: ChatCompletionChunk | None = None
        self.mcp_calls: list[dict[str, Any]] = []


class _ChatToolLoopStrategy:
    """Chat-completions strategy for the generic tool loop.

    ``acompletion`` is resolved as a module global at call time so tests can
    monkeypatch ``gateway.services.mcp_loop.acompletion``.
    """

    transcript_key = "messages"

    def coerce_transcript(self, value: Any) -> list[Any]:
        return list(value or [])

    def convert_pool_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return tools

    # ---- non-streaming hooks ----

    async def call(self, kwargs: dict[str, Any]) -> ChatCompletion:
        completion: ChatCompletion = await acompletion(**kwargs)  # type: ignore[assignment]
        return completion

    def new_usage_accumulator(self) -> dict[str, int]:
        return {"prompt": 0, "completion": 0, "cache_read": 0}

    def accumulate_usage(self, acc: dict[str, int], result: ChatCompletion) -> None:
        if result.usage:
            acc["prompt"] += result.usage.prompt_tokens or 0
            acc["completion"] += result.usage.completion_tokens or 0
            details = result.usage.prompt_tokens_details
            if details is not None:
                acc["cache_read"] += details.cached_tokens or 0

    def fold_usage(self, result: ChatCompletion, acc: dict[str, int]) -> None:
        _fold_usage(result, acc["prompt"], acc["completion"], acc["cache_read"])

    def exit_before_split(self, result: ChatCompletion) -> bool:
        return not result.choices or result.choices[0].finish_reason != "tool_calls"

    def split_owned(self, result: ChatCompletion, pool: ToolBackend) -> tuple[list[Any], bool]:
        sdk_calls = result.choices[0].message.tool_calls or []
        # Duck-type the SDK tool-call: any object with a `.function` attribute is a
        # function tool-call. We avoid `isinstance` here because `acompletion`'s
        # return type uses the OpenAI SDK's class, while any_llm exposes a
        # same-named but distinct class alias.
        tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in sdk_calls
            if hasattr(tc, "function")
        ]
        return _execute_split(tool_calls, pool)

    def exit_after_split(self, result: ChatCompletion) -> bool:
        return False

    async def execute_owned(self, pool: ToolBackend, owned: list[Any]) -> list[dict[str, Any]]:
        return await _execute_mcp_calls(pool, owned)

    def filter_owned(self, result: ChatCompletion, owned: list[Any], pool: ToolBackend) -> None:
        # Mixed batch: the MCP-owned subset was executed internally so its
        # work isn't wasted; filter those calls out of the returned completion
        # so the caller only sees tool_calls it can itself dispatch. The
        # conversation continues client-side; if the caller wants to keep
        # using the gateway's MCP tools they'll send the foreign-tool results
        # back on the next request.
        choice = result.choices[0]
        sdk_calls = choice.message.tool_calls or []
        foreign_sdk_calls = [
            tc for tc in sdk_calls if hasattr(tc, "function") and not pool.owns_tool(tc.function.name)
        ]
        try:
            choice.message.tool_calls = foreign_sdk_calls or None
        except (AttributeError, TypeError):
            # If the SDK model is frozen, fall back to leaving the
            # original list. Cleaner UX requires SDK mutability.
            logger.warning(
                "MCP-mixed: could not filter tool_calls on response; client will see MCP calls "
                "the gateway already executed (no-op on the client side).",
            )

    async def advance_transcript(
        self,
        transcript: list[Any],
        result: ChatCompletion,
        owned: list[Any],
        pool: ToolBackend,
    ) -> None:
        transcript.append({"role": "assistant", "tool_calls": owned})
        transcript.extend(await _execute_mcp_calls(pool, owned))

    # ---- streaming hooks ----

    async def open_stream(self, kwargs: dict[str, Any]) -> AsyncIterator[ChatCompletionChunk]:
        stream: AsyncIterator[ChatCompletionChunk] = await acompletion(**kwargs)  # type: ignore[assignment]
        return stream

    def new_stream_state(self) -> _ChatStreamState:
        return _ChatStreamState()

    def new_stream_accumulator(self) -> None:
        # Chat streaming does not fold cumulative usage into the terminal
        # chunk (parity with the pre-engine behavior); streaming usage
        # accounting happens downstream in `streaming_generator`.
        return None

    def observe(self, state: _ChatStreamState, event: ChatCompletionChunk) -> StreamAction:
        chunk_is_terminal = False
        if event.choices:
            choice = event.choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None and getattr(delta, "tool_calls", None):
                _accumulate_tool_call_deltas(state.slots, delta.tool_calls)
            if choice.finish_reason:
                # Sticky-tool-calls: a trailing ``stop`` chunk from
                # Anthropic must not override ``tool_calls`` we've
                # already seen on this same iteration.
                if not (state.finish_reason == "tool_calls" and choice.finish_reason != "tool_calls"):
                    state.finish_reason = choice.finish_reason
                chunk_is_terminal = True
        if chunk_is_terminal:
            state.pending_terminal = event
            return StreamAction.DEFER
        return StreamAction.FORWARD

    def stream_exiting(self, state: _ChatStreamState, pool: ToolBackend) -> bool:
        if state.finish_reason != "tool_calls":
            return True
        tool_calls = _finalize_tool_calls(state.slots)
        state.mcp_calls, has_foreign = _execute_split(tool_calls, pool)
        # Mixed (has_foreign with mcp_calls) is handled the same as
        # all-foreign here: the terminal chunk is forwarded so the
        # client sees the full tool_calls set, and any MCP-owned calls
        # in a mixed batch are left for a follow-up turn (the
        # non-streaming variant filters them out; rewriting deltas
        # mid-stream is too invasive to do here).
        return has_foreign or not state.mcp_calls

    def terminal_events(self, state: _ChatStreamState, acc: None) -> list[ChatCompletionChunk]:
        return [state.pending_terminal] if state.pending_terminal is not None else []

    def accumulate_stream_usage(self, acc: None, state: _ChatStreamState) -> None:
        return None

    async def advance_stream_transcript(
        self,
        transcript: list[Any],
        state: _ChatStreamState,
        pool: ToolBackend,
    ) -> None:
        # All-MCP: the terminal chunk was silently dropped so the client
        # doesn't think this iteration's response was the final answer.
        transcript.append({"role": "assistant", "tool_calls": state.mcp_calls})
        transcript.extend(await _execute_mcp_calls(pool, state.mcp_calls))


_CHAT_STRATEGY = _ChatToolLoopStrategy()


async def mcp_tool_loop_stream(
    *,
    completion_kwargs: dict[str, Any],
    pool: ToolBackend,
    max_iterations: int,
) -> AsyncGenerator[ChatCompletionChunk, None]:
    """Yield chunks across multiple `acompletion(stream=True)` calls, with MCP execution between rounds.

    Tool-call deltas from intermediate iterations are streamed straight through to the
    caller (so clients that want to render "thinking" still get those bytes), but the
    *terminal* chunk of each intermediate iteration, the one carrying
    ``finish_reason="tool_calls"``, is buffered and dropped if the loop is going to
    iterate again. Forwarding that chunk would tell an OpenAI-compatible client "this
    is the final answer", and most SDKs stop reading at that point; subsequent
    iterations' content would be silently truncated.

    The terminal chunk is forwarded in three cases:
      * the iteration ended with a non-``tool_calls`` finish_reason (e.g. ``stop``),
      * the model produced foreign tool_calls (caller needs to dispatch them), or
      * the model produced no MCP-owned calls at all (loop exits, terminal goes through).
    """
    # aclosing makes downstream closes (client disconnect) propagate to the
    # engine generator, and through it to the upstream provider stream,
    # instead of waiting for event-loop async-generator finalization.
    async with aclosing(
        run_tool_loop_stream(
            strategy=_CHAT_STRATEGY,
            completion_kwargs=completion_kwargs,
            pool=pool,
            max_iterations=max_iterations,
        )
    ) as inner:
        async for chunk in inner:
            yield chunk


async def mcp_tool_loop(
    *,
    completion_kwargs: dict[str, Any],
    pool: ToolBackend,
    max_iterations: int,
    on_first_response: Callable[[], None] | None = None,
) -> ChatCompletion:
    """Non-streaming variant. Accumulates usage across iterations into the returned completion.

    ``on_first_response`` follows the provider lock-in contract documented on
    :func:`gateway.services._tool_loop.run_tool_loop`; the hybrid-mode attempt
    loop in :mod:`gateway.api.routes.chat` is the consumer.
    """
    return await run_tool_loop(
        strategy=_CHAT_STRATEGY,
        completion_kwargs=completion_kwargs,
        pool=pool,
        max_iterations=max_iterations,
        on_first_response=on_first_response,
    )
