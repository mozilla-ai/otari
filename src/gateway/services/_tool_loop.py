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

from gateway.core.observation import NormalizedTool
from gateway.core.usage import GatewayUsage

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

    ``normalize_tools`` and ``usage_snapshot`` are the odd ones out: nothing in
    the loop calls either. They exist so Reprise can read a round without
    reimplementing three wire vocabularies (otari-ai#1647), and they sit on the
    Protocol, and on its streaming twin, because that is where the format is
    known and because ``mypy --strict`` then names a format that forgot one.

    ``normalize_tools`` unwraps the format-shaped list :func:`_prepare` merges
    (nested ``function.parameters`` for chat, ``input_schema`` for messages, flat
    ``parameters`` for responses) into the format-neutral triples behind
    ``tool_set_hash`` and ``tool_definitions_hash``.

    ``usage_snapshot`` reads one round's usage off the provider's own result as a
    :class:`~gateway.core.usage.GatewayUsage`, cache reads and cache writes
    included, and its streaming twin ``stream_usage_snapshot`` reads the same
    figures off the events of one upstream stream. Both return ``None`` when the
    provider reported no usage at all, because zero and unknown are different
    answers. Neither touches the usage accumulators, which exist to fold totals
    into the client's response and must keep doing exactly that.
    """

    transcript_key: str

    def coerce_transcript(self, value: Any) -> list[Any]: ...

    def convert_pool_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

    def normalize_tools(self, tools: list[dict[str, Any]]) -> list[NormalizedTool]: ...

    async def call(self, kwargs: dict[str, Any]) -> ResultT: ...

    def new_usage_accumulator(self) -> AccT: ...

    def accumulate_usage(self, acc: AccT, result: ResultT) -> None: ...

    def fold_usage(self, result: ResultT, acc: AccT) -> None: ...

    def usage_snapshot(self, result: ResultT) -> GatewayUsage | None: ...

    def exit_before_split(self, result: ResultT) -> bool: ...

    def split_owned(self, result: ResultT, pool: ToolBackend) -> tuple[list[Any], bool]: ...

    def exit_after_split(self, result: ResultT) -> bool: ...

    async def execute_owned(
        self, pool: ToolBackend, owned: list[Any], acc: AccT | None = None
    ) -> list[dict[str, Any]]: ...

    def filter_owned(self, result: ResultT, owned: list[Any], pool: ToolBackend) -> None: ...

    async def advance_transcript(
        self,
        transcript: list[Any],
        result: ResultT,
        owned: list[Any],
        pool: ToolBackend,
        acc: AccT,
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

    def normalize_tools(self, tools: list[dict[str, Any]]) -> list[NormalizedTool]: ...

    async def open_stream(self, kwargs: dict[str, Any]) -> AsyncIterator[ChunkT]: ...

    def new_stream_state(self) -> StateT: ...

    def new_stream_accumulator(self) -> AccT: ...

    def observe(
        self, state: StateT, event: ChunkT, pool: ToolBackend, acc: AccT
    ) -> tuple[StreamAction, ChunkT]: ...

    def stream_exiting(self, state: StateT, pool: ToolBackend) -> bool: ...

    def terminal_events(self, state: StateT, acc: AccT) -> list[ChunkT]: ...

    def accumulate_stream_usage(self, acc: AccT, state: StateT) -> None: ...

    def stream_usage_snapshot(self, state: StateT) -> GatewayUsage | None: ...

    async def advance_stream_transcript(
        self,
        transcript: list[Any],
        state: StateT,
        pool: ToolBackend,
    ) -> None: ...

    def synthetic_events(self, state: StateT, acc: AccT) -> list[ChunkT]: ...

    async def finalize_exit(self, state: StateT, pool: ToolBackend) -> None: ...


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
                await strategy.execute_owned(pool, owned, acc)
                strategy.filter_owned(result, owned, pool)
            strategy.fold_usage(result, acc)
            return result
        if not owned or strategy.exit_after_split(result):
            strategy.fold_usage(result, acc)
            return result

        await strategy.advance_transcript(transcript, result, owned, pool, acc)

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
    format's bookkeeping and returns both what to do with the event and the
    event to forward. The returned event may differ from the upstream one:
    a format that stitches several provider messages into one client-visible
    response has to renumber the indices it forwards, and it must swallow the
    gateway's own tool calls entirely, since the client can never receive a
    result for a tool the gateway ran itself.

    ``acc`` is passed to ``observe`` because that stitching is cross-iteration
    state (has the envelope been opened, which index comes next), while ``state``
    is per-iteration. After each upstream stream ends,
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
                action, visible = strategy.observe(state, event, pool, acc)
                if action is StreamAction.BREAK:
                    break
                if action is StreamAction.FORWARD:
                    yield visible

        if strategy.stream_exiting(state, pool):
            # A mixed batch (the gateway's own tools plus the caller's) exits here so
            # the caller can dispatch its own, but the gateway's calls were hidden
            # from the stream and would otherwise never run: the model asked for a
            # search and would get silence. Executing them for their side effects is
            # what the non-streaming loop already does before returning a mixed
            # result (see ``run_tool_loop``), so the two agree.
            await strategy.finalize_exit(state, pool)
            for terminal in strategy.terminal_events(state, acc):
                yield terminal
            return

        strategy.accumulate_stream_usage(acc, state)
        await strategy.advance_stream_transcript(transcript, state, pool)
        # A format with a native vocabulary for server-side tool calls announces
        # the calls the gateway just ran, in place of the raw tool-call events that
        # were swallowed. Formats without one return nothing.
        for synthetic in strategy.synthetic_events(state, acc):
            yield synthetic

    raise MaxToolIterationsExceeded(f"Exceeded max_tool_iterations={max_iterations}")
