"""Format-agnostic engine for the gateway's agentic tool-use loop.

The chat-completions (:mod:`gateway.services.mcp_loop`), Anthropic Messages
(:mod:`gateway.services.mcp_loop_messages`), and OpenAI Responses
(:mod:`gateway.services.mcp_loop_responses`) tool loops run the same
algorithm: call the provider, split the tool calls it produced into
gateway-owned and caller-supplied (foreign) ones, execute the owned calls,
append the assistant turn plus tool results to the transcript, and re-call
the provider until a terminal answer arrives. Only the wire vocabulary
differs (message shapes, terminal events, usage field names). This module
owns the algorithm once; each format contributes a small strategy object
implementing :class:`ToolLoopStrategy` (non-streaming) and
:class:`StreamToolLoopStrategy` (streaming).

Provider functions are intentionally not called from here: each strategy's
``call`` / ``open_stream`` resolves ``acompletion`` / ``amessages`` /
``aresponses`` as a module global of its own format module at call time, so
tests can keep monkeypatching them there.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, aclosing, nullcontext
from enum import Enum, auto
from typing import Any, Generic, Protocol, TypeVar, cast

ResultT = TypeVar("ResultT")
ChunkT = TypeVar("ChunkT")
StateT = TypeVar("StateT")
AccT = TypeVar("AccT")


class ToolBackend(Protocol):
    """Subset of the tool-backend surface the loop drives.

    Structurally implemented by ``MCPClientPool``, ``SandboxBackend``, and
    ``WebSearchBackend``; each exposes the same members the loop and the route
    pipeline need to advertise tools, decide ownership, execute a call, and
    describe each tool's purpose for the system-message hint block. Widening
    ``pool`` to this Protocol lets the routes pass any of those backends
    without casts.
    """

    @property
    def openai_tools(self) -> list[dict[str, Any]]: ...

    def owns_tool(self, name: str) -> bool: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...

    def purpose_hints(self) -> list[tuple[str, str]]: ...


class MaxToolIterationsExceeded(Exception):
    """Raised when the loop fails to reach a non-tool-call finish in N rounds."""


class StreamAction(Enum):
    """What the engine does with one upstream stream event.

    ``FORWARD`` yields the event downstream, ``DEFER`` swallows it (the
    strategy stashed it as a pending terminal), and ``BREAK`` additionally
    stops consuming the current upstream stream.
    """

    FORWARD = auto()
    DEFER = auto()
    BREAK = auto()


class ToolLoopStrategy(Protocol, Generic[ResultT, AccT]):
    """Per-format hooks for :func:`run_tool_loop`.

    ``transcript_key`` names the kwargs entry carrying the conversation
    (``messages`` for chat/messages, ``input_data`` for responses). The exit
    hooks preserve each format's historical check ordering: chat inspects the
    finish_reason before splitting (``exit_before_split``), while messages
    checks the stop_reason only after the foreign-tool branch
    (``exit_after_split``); see :func:`run_tool_loop` for the consequences.
    """

    transcript_key: str

    def coerce_transcript(self, value: Any) -> list[Any]: ...

    def convert_pool_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

    async def call(self, kwargs: dict[str, Any]) -> ResultT: ...

    def new_usage_accumulator(self) -> AccT: ...

    def accumulate_usage(self, acc: AccT, result: ResultT) -> None: ...

    def fold_usage(self, result: ResultT, acc: AccT) -> None: ...

    def exit_before_split(self, result: ResultT) -> bool: ...

    def split_owned(self, result: ResultT, pool: ToolBackend) -> tuple[list[Any], bool]: ...

    def exit_after_split(self, result: ResultT) -> bool: ...

    async def execute_owned(self, pool: ToolBackend, owned: list[Any]) -> list[dict[str, Any]]: ...

    def filter_owned(self, result: ResultT, owned: list[Any], pool: ToolBackend) -> None: ...

    async def advance_transcript(
        self,
        transcript: list[Any],
        result: ResultT,
        owned: list[Any],
        pool: ToolBackend,
    ) -> None: ...


class StreamToolLoopStrategy(Protocol, Generic[ChunkT, StateT, AccT]):
    """Per-format hooks for :func:`run_tool_loop_stream`.

    ``observe`` performs the format's per-event bookkeeping (tool-call delta
    accumulation, terminal-event capture) on the per-iteration ``state`` and
    tells the engine what to do with the event. ``stream_exiting`` makes the
    loop-or-exit decision after the stream ends and stashes whatever the
    continuation needs (owned tool specs) on the state.
    """

    transcript_key: str

    def coerce_transcript(self, value: Any) -> list[Any]: ...

    def convert_pool_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

    async def open_stream(self, kwargs: dict[str, Any]) -> AsyncIterator[ChunkT]: ...

    def new_stream_state(self) -> StateT: ...

    def new_stream_accumulator(self) -> AccT: ...

    def observe(self, state: StateT, event: ChunkT) -> StreamAction: ...

    def stream_exiting(self, state: StateT, pool: ToolBackend) -> bool: ...

    def terminal_events(self, state: StateT, acc: AccT) -> list[ChunkT]: ...

    def accumulate_stream_usage(self, acc: AccT, state: StateT) -> None: ...

    async def advance_stream_transcript(
        self,
        transcript: list[Any],
        state: StateT,
        pool: ToolBackend,
    ) -> None: ...


def _prepare(
    strategy_key: str,
    coerce: Callable[[Any], list[Any]],
    convert: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    completion_kwargs: dict[str, Any],
    pool: ToolBackend,
    *,
    drop_stream: bool,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    """Split ``completion_kwargs`` into (transcript, merged tools, base kwargs)."""
    transcript = coerce(completion_kwargs.get(strategy_key))
    user_tools = list(completion_kwargs.get("tools") or [])
    merged_tools = user_tools + convert(pool.openai_tools)
    excluded = {strategy_key, "tools", "stream"} if drop_stream else {strategy_key, "tools"}
    base = {k: v for k, v in completion_kwargs.items() if k not in excluded}
    return transcript, merged_tools, base


def _stream_scope(stream: AsyncIterator[Any]) -> AbstractAsyncContextManager[Any]:
    """Close ``stream`` on scope exit when it supports ``aclose``.

    The streaming loops stop consuming the upstream stream as soon as they see
    the format's terminal event; without an explicit close, the async
    generator (and any underlying connection) would be left to garbage
    collection. The scope also closes the upstream when the downstream
    consumer closes the tool-loop generator mid-stream.
    """
    if hasattr(stream, "aclose"):
        return aclosing(cast(Any, stream))
    return nullcontext(stream)


async def run_tool_loop(
    *,
    strategy: ToolLoopStrategy[ResultT, Any],
    completion_kwargs: dict[str, Any],
    pool: ToolBackend,
    max_iterations: int,
    on_first_response: Callable[[], None] | None = None,
) -> ResultT:
    """Non-streaming tool loop, generic over the wire format.

    Each iteration calls the provider through the strategy, accumulates usage,
    and then walks the exit ladder:

    1. ``exit_before_split``: format-level terminal check that runs before any
       tool accounting (chat: no choices, or finish_reason is not
       ``tool_calls``). When it fires, nothing is executed even if the result
       carries gateway-owned calls.
    2. Foreign tools present: mixed batches execute the owned subset for its
       side effects and filter it from the returned result, so the caller only
       sees calls it can dispatch itself. Note the ordering divergence kept
       from the pre-engine modules: messages/responses run this branch before
       any stop-reason check, so a mixed batch executes owned calls even when
       the model stopped for another reason, while chat never reaches this
       branch unless finish_reason was ``tool_calls``.
    3. Nothing owned, or ``exit_after_split`` (messages: stop_reason is not
       ``tool_use``): return the result as the final answer.
    4. Otherwise execute the owned calls, extend the transcript, and iterate.

    ``on_first_response`` is invoked once, right after the first upstream call
    returns successfully. Callers use it to lock in the chosen provider: once
    the model has produced any assistant output, the conversation state is
    provider-specific and subsequent failures must not silently swap
    providers. See the hybrid-mode attempt loops in the route modules.
    """
    transcript, merged_tools, base = _prepare(
        strategy.transcript_key,
        strategy.coerce_transcript,
        strategy.convert_pool_tools,
        completion_kwargs,
        pool,
        drop_stream=True,
    )
    acc = strategy.new_usage_accumulator()
    first_response_signaled = False

    for _ in range(max_iterations):
        kwargs: dict[str, Any] = {**base, strategy.transcript_key: transcript, "stream": False}
        if merged_tools:
            kwargs["tools"] = merged_tools

        result = await strategy.call(kwargs)
        if not first_response_signaled:
            first_response_signaled = True
            if on_first_response is not None:
                on_first_response()
        strategy.accumulate_usage(acc, result)

        if strategy.exit_before_split(result):
            strategy.fold_usage(result, acc)
            return result

        owned, has_foreign = strategy.split_owned(result, pool)
        if has_foreign:
            if owned:
                await strategy.execute_owned(pool, owned)
                strategy.filter_owned(result, owned, pool)
            strategy.fold_usage(result, acc)
            return result
        if not owned or strategy.exit_after_split(result):
            strategy.fold_usage(result, acc)
            return result

        await strategy.advance_transcript(transcript, result, owned, pool)

    raise MaxToolIterationsExceeded(f"Exceeded max_tool_iterations={max_iterations}")


async def run_tool_loop_stream(
    *,
    strategy: StreamToolLoopStrategy[ChunkT, Any, Any],
    completion_kwargs: dict[str, Any],
    pool: ToolBackend,
    max_iterations: int,
) -> AsyncGenerator[ChunkT, None]:
    """Streaming tool loop, generic over the wire format.

    Every upstream event flows through ``strategy.observe``, which does the
    format's bookkeeping and decides whether the event is forwarded downstream
    or deferred as a pending terminal. After each upstream stream ends,
    ``strategy.stream_exiting`` decides between exiting (the deferred terminal
    events are forwarded, with cumulative usage folded in where the format
    supports it) and continuing (the terminal events are dropped, their usage
    accumulated, the owned calls executed, and the next round dispatched).

    The upstream stream is closed when the loop stops consuming it early;
    see :func:`_stream_scope`.
    """
    transcript, merged_tools, base = _prepare(
        strategy.transcript_key,
        strategy.coerce_transcript,
        strategy.convert_pool_tools,
        completion_kwargs,
        pool,
        drop_stream=False,
    )
    base["stream"] = True
    acc = strategy.new_stream_accumulator()

    for _ in range(max_iterations):
        kwargs: dict[str, Any] = {**base, strategy.transcript_key: transcript}
        if merged_tools:
            kwargs["tools"] = merged_tools

        stream = await strategy.open_stream(kwargs)
        state = strategy.new_stream_state()

        async with _stream_scope(stream):
            async for event in stream:
                action = strategy.observe(state, event)
                if action is StreamAction.BREAK:
                    break
                if action is StreamAction.FORWARD:
                    yield event

        if strategy.stream_exiting(state, pool):
            for terminal in strategy.terminal_events(state, acc):
                yield terminal
            return

        strategy.accumulate_stream_usage(acc, state)
        await strategy.advance_stream_transcript(transcript, state, pool)

    raise MaxToolIterationsExceeded(f"Exceeded max_tool_iterations={max_iterations}")
