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
from any_llm.types.completion import CompletionUsage, PromptTokensDetails

from gateway.core.env import otari_env
from gateway.core.observation import NormalizedTool, normalized_tool
from gateway.core.usage import GatewayUsage
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


def _definition_of(tool: dict[str, Any]) -> dict[str, Any]:
    """The chat-completions tool definition, unwrapped from its ``function`` nesting."""
    fn = tool.get("function")
    return fn if isinstance(fn, dict) else tool


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


def _with_tool_calls(event: ChatCompletionChunk, tool_calls: list[Any] | None) -> ChatCompletionChunk | None:
    """Return ``event`` with its delta's ``tool_calls`` replaced, or ``None`` on failure.

    Used to strip gateway-owned fragments out of a chunk that also carries content or
    foreign fragments, so the client sees only what it can act on.

    ``None`` means the rewrite failed and the caller must drop the chunk rather than
    forward it. Forwarding the original would hand the client a gateway tool call it
    can never receive a result for, and one the gateway is about to execute itself, so
    losing a delta is the safer failure: the terminal chunk still carries the finish
    reason, and the caller's own tool calls survive in the following chunks.
    """
    try:
        choice = event.choices[0]
        new_delta = choice.delta.model_copy(update={"tool_calls": tool_calls})
        new_choice = choice.model_copy(update={"delta": new_delta})
        return event.model_copy(update={"choices": [new_choice]})
    except (AttributeError, TypeError, IndexError):
        logger.warning("Could not strip gateway tool_calls from a streamed chunk; dropping it")
        return None


def _stripped_tool_calls(event: ChatCompletionChunk) -> ChatCompletionChunk:
    """``event`` with its tool_calls removed, or unchanged if that cannot be built.

    Last resort for a terminal chunk whose rewrite failed: the finish reason has to
    reach the client, and a terminal carrying no tool_calls is a valid stream, while
    one carrying the gateway's own call is not.
    """
    stripped = _with_tool_calls(event, None)
    return stripped if stripped is not None else event


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
        # Upstream tool_call index -> the index the client sees. Gateway-owned calls
        # are dropped from the stream, so the surviving foreign calls have to be
        # renumbered into a gapless sequence: an SDK accumulator indexes its snapshot
        # array by the index it is handed, and a first fragment numbered 1 makes the
        # official OpenAI client raise IndexError.
        self.visible_tool_index: dict[int, int] = {}
        self.next_visible_tool_index = 0
        # This round's provider-reported usage, for the Reprise usage snapshot
        # (otari-ai#1647). Recorded only: the chunk carrying it is forwarded
        # untouched, and the client-facing usage accounting still happens
        # downstream in ``streaming_generator``.
        self.usage: CompletionUsage | None = None


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

    def normalize_tools(self, tools: list[dict[str, Any]]) -> list[NormalizedTool]:
        # Chat nests the definition under ``function``; an entry that is not a
        # function tool is read flat, so a provider-native tool still counts by name.
        return [normalized_tool(_definition_of(tool), "parameters") for tool in tools]

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

    def usage_snapshot(self, result: ChatCompletion) -> GatewayUsage | None:
        # OpenAI-shaped: cached tokens are a slice of prompt_tokens rather than an
        # additive bucket, which ``from_completion_usage`` records by leaving
        # ``cache_tokens_in_prompt`` at its True default.
        if result.usage is None:
            return None
        return GatewayUsage.from_completion_usage(result.usage)

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

    async def execute_owned(
        self, pool: ToolBackend, owned: list[Any], acc: Any = None
    ) -> list[dict[str, Any]]:
        # ``acc`` is accepted for interface parity and unused: this format has no
        # native vocabulary for a server-side tool call to report on a mixed batch.
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
        foreign_sdk_calls = [tc for tc in sdk_calls if hasattr(tc, "function") and not pool.owns_tool(tc.function.name)]
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
        acc: Any = None,
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

    def observe(
        self,
        state: _ChatStreamState,
        event: ChatCompletionChunk,
        pool: ToolBackend,
        acc: None,
    ) -> tuple[StreamAction, ChatCompletionChunk]:
        if event.usage is not None:
            # Usually a trailing choices-less chunk, but a provider any-llm
            # synthesizes streaming for may attach it to the terminal chunk
            # instead; last one wins, since the counts are cumulative.
            state.usage = event.usage
        chunk_is_terminal = False
        hide = False
        visible = event
        if event.choices:
            choice = event.choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None and getattr(delta, "tool_calls", None):
                _accumulate_tool_call_deltas(state.slots, delta.tool_calls)
                # Fragments for a gateway-owned call are not forwarded: the client
                # can never be sent the matching ``tool`` message, because the
                # gateway consumes the result itself. A fragment carrying only
                # arguments has no name of its own, so ownership is resolved from
                # the accumulated slot for its index.
                foreign = [tc for tc in delta.tool_calls if not self._owned_fragment(state, pool, tc)]
                renumbered = [self._renumbered(state, tc) for tc in foreign]
                rewritten: ChatCompletionChunk | None = None
                if len(foreign) != len(delta.tool_calls):
                    if foreign or getattr(delta, "content", None):
                        rewritten = _with_tool_calls(event, renumbered or None)
                        hide = rewritten is None
                    else:
                        hide = True
                elif any(tc is not original for tc, original in zip(renumbered, foreign, strict=True)):
                    # No owned call in this chunk, but an earlier one shifted the
                    # numbering, so the survivors still need rewriting.
                    rewritten = _with_tool_calls(event, renumbered)
                    hide = rewritten is None
                if rewritten is not None:
                    visible = rewritten
            if choice.finish_reason:
                # Sticky-tool-calls: a trailing ``stop`` chunk from
                # Anthropic must not override ``tool_calls`` we've
                # already seen on this same iteration.
                if not (state.finish_reason == "tool_calls" and choice.finish_reason != "tool_calls"):
                    state.finish_reason = choice.finish_reason
                chunk_is_terminal = True
        if chunk_is_terminal:
            # The rewritten chunk, not the upstream one: a provider that packs
            # tool_calls and finish_reason into a single chunk would otherwise leak
            # the gateway's own call through the deferred terminal, undoing the hiding
            # above for exactly the providers any-llm synthesizes streaming for. If
            # the rewrite failed, the terminal still has to reach the client (it
            # carries the finish reason), so send it with no tool_calls at all.
            state.pending_terminal = visible if not hide else _stripped_tool_calls(event)
            return StreamAction.DEFER, state.pending_terminal
        if hide:
            return StreamAction.DEFER, event
        return StreamAction.FORWARD, visible

    @staticmethod
    def _renumbered(state: _ChatStreamState, fragment: Any) -> Any:
        """Return ``fragment`` with its client-visible tool_call index.

        Identity is returned when the index already matches, so a stream with no
        hidden calls forwards byte-identical fragments.
        """
        index = getattr(fragment, "index", None)
        if not isinstance(index, int):
            return fragment
        if index not in state.visible_tool_index:
            state.visible_tool_index[index] = state.next_visible_tool_index
            state.next_visible_tool_index += 1
        visible = state.visible_tool_index[index]
        if visible == index or not hasattr(fragment, "model_copy"):
            return fragment
        return fragment.model_copy(update={"index": visible})

    @staticmethod
    def _owned_fragment(state: _ChatStreamState, pool: ToolBackend, fragment: Any) -> bool:
        """Whether a streamed tool_call fragment belongs to a gateway-run tool.

        A fragment that carries only arguments has no name of its own, so ownership
        is resolved from the accumulated slot for its index.
        """
        index = getattr(fragment, "index", None)
        if not isinstance(index, int):
            return False
        slot = state.slots.get(index)
        name = (slot or {}).get("function", {}).get("name") or ""
        return bool(name) and pool.owns_tool(name)

    def stream_exiting(self, state: _ChatStreamState, pool: ToolBackend) -> bool:
        if state.finish_reason != "tool_calls":
            return True
        tool_calls = _finalize_tool_calls(state.slots)
        state.mcp_calls, has_foreign = _execute_split(tool_calls, pool)
        # A mixed batch exits so the caller can dispatch its own tools, and the
        # gateway's calls were filtered out of the stream, so the client only ever
        # sees what it can act on. Those owned calls are still executed, in
        # ``finalize_exit``, which is the same "execute for side effects, return only
        # the foreign calls" contract the non-streaming loop applies.
        return has_foreign or not state.mcp_calls

    async def finalize_exit(self, state: _ChatStreamState, pool: ToolBackend) -> None:
        if state.mcp_calls:
            await _execute_mcp_calls(pool, state.mcp_calls)

    def terminal_events(self, state: _ChatStreamState, acc: None) -> list[ChatCompletionChunk]:
        return [state.pending_terminal] if state.pending_terminal is not None else []

    def accumulate_stream_usage(self, acc: None, state: _ChatStreamState) -> None:
        return None

    def stream_usage_snapshot(self, state: _ChatStreamState) -> GatewayUsage | None:
        # ``None`` rather than a zero-filled snapshot when the caller set
        # ``include_usage: False`` and the provider therefore sent nothing: this is
        # the one format where a round's usage can be genuinely unknown, and
        # unknown must not read as zero.
        if state.usage is None:
            return None
        return GatewayUsage.from_completion_usage(state.usage)

    def synthetic_events(self, state: _ChatStreamState, acc: None) -> list[Any]:
        # This format has no native vocabulary for a server-side tool call, so the
        # gateway's calls stay invisible on the wire. Documented in docs/tools.md.
        return []

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
