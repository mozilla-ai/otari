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
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, cast, runtime_checkable

from anthropic.types import ServerToolUseBlock, WebSearchResultBlock, WebSearchToolResultBlock
from anthropic.types.beta import BetaMCPToolResultBlock, BetaMCPToolUseBlock
from any_llm import amessages
from any_llm.types.messages import (
    BetaContextManagementResponse,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
)

from gateway.log_config import logger
from gateway.services._tool_loop import StreamAction, run_tool_loop, run_tool_loop_stream
from gateway.services.mcp_loop import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    MAX_TOOL_ITERATIONS_CAP,
    MaxToolIterationsExceeded,
    ToolBackend,
)
from gateway.services.tool_format import openai_to_anthropic_tools
from gateway.services.tool_usage import is_tool_error
from gateway.services.web_search_backend import WEB_SEARCH_TOOL_NAME

if TYPE_CHECKING:
    from any_llm.types.messages import (
        MessageResponse,
        MessageStreamEvent,
    )

    from gateway.services.mcp_client import MCPToolCallOutcome


@runtime_checkable
class MCPToolBackend(Protocol):
    """MCP-specific capabilities used to emit streaming activity blocks."""

    def server_name_for_tool(self, name: str) -> str | None: ...

    async def call_tool_outcome(self, name: str, arguments: dict[str, Any]) -> MCPToolCallOutcome: ...


# Re-export so callers in routes/messages.py have a single import surface.
__all__ = [
    "DEFAULT_MAX_TOOL_ITERATIONS",
    "MAX_TOOL_ITERATIONS_CAP",
    "MaxToolIterationsExceeded",
    "MCP_ACTIVITY_ID_PREFIX",
    "anthropic_tool_loop",
    "anthropic_tool_loop_stream",
]


# A search result's recency, collapsed to one line and length-capped before it
# goes on a native block. Same rendering hygiene the text formatter applies: the
# value is whatever a search-API-fronting adapter forwarded, so one overlong or
# multiline entry shouldn't be what makes a citations panel unreadable.
_PAGE_AGE_MAX_CHARS = 128

# Provenance marker for MCP activity blocks minted by this gateway. Clients may
# echo the accumulated assistant message on their next request; the Messages
# route uses this prefix to remove only our synthetic pair while preserving
# provider-native MCP blocks.
MCP_ACTIVITY_ID_PREFIX = "otari_mcptoolu_"


def _native_web_search_blocks(query: str, results: list[dict[str, Any]]) -> list[Any]:
    """A ``server_tool_use`` / ``web_search_tool_result`` pair for one gateway search.

    Emitted only for a caller that declared web search in Anthropic's native
    vocabulary, which is the contract that makes a client expect these blocks and
    render citations from them.

    ``encrypted_content`` is required by the schema but is an Anthropic-signed blob
    only Anthropic can mint, so the gateway sends it empty rather than forging one.
    A client that echoes the block back through the gateway has it stripped before
    the provider sees it (see ``routes/messages.py``); one that echoes it straight
    to Anthropic instead would be rejected there, which is the same trade-off the
    Responses path already accepts for its minted ``web_search_call`` items.
    """
    tool_use_id = f"srvtoolu_{uuid.uuid4().hex}"
    citations: list[WebSearchResultBlock] = []
    for result in results:
        url = str(result.get("url") or "").strip()
        if not url:
            # Nothing to cite. A hit with no URL is unusable to a citations panel.
            continue
        page_age = " ".join(str(result.get("published_date") or "").split())[:_PAGE_AGE_MAX_CHARS]
        citations.append(
            WebSearchResultBlock(
                type="web_search_result",
                url=url,
                title=str(result.get("title") or url).strip(),
                page_age=page_age or None,
                encrypted_content="",
            )
        )
    return [
        ServerToolUseBlock(
            id=tool_use_id,
            # Anthropic types this field as a Literal of its own server-tool names.
            # The gateway's tool name is one of them, but the shared constant is a
            # plain ``str``, so narrow it here rather than duplicating the literal.
            name=cast('Literal["web_search"]', WEB_SEARCH_TOOL_NAME),
            input={"query": query},
            type="server_tool_use",
        ),
        WebSearchToolResultBlock(
            tool_use_id=tool_use_id,
            type="web_search_tool_result",
            content=citations,
        ),
    ]


def _native_blocks_for_call(pool: ToolBackend, name: str, arguments: dict[str, Any]) -> list[Any]:
    """Native blocks describing one completed gateway tool call, if it has any.

    Only ``web_search`` does: a sandbox or MCP call has no Anthropic block that
    would be honest to emit (``code_execution_tool_result`` would claim Anthropic's
    own container ran the code), so those stay invisible, as they do on Responses.
    """
    if name != WEB_SEARCH_TOOL_NAME:
        return []
    take_last_results = getattr(pool, "take_last_results", None)
    if take_last_results is None:
        return []
    return _native_web_search_blocks(str(arguments.get("query") or ""), take_last_results())


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
    *,
    native_blocks: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Run each owned tool_use block and return the Anthropic tool_result blocks.

    Tool failures convert to a ``[tool error] ...`` string in the result so the
    model can recover. Only cancellation-class exceptions
    (``asyncio.CancelledError``, ``KeyboardInterrupt``) escape; they inherit
    from ``BaseException`` and skip the ``Exception`` clause. Same idiom as
    :func:`gateway.services.mcp_loop._execute_mcp_calls`.

    When ``native_blocks`` is given, each *successful* call appends the native
    server-tool blocks describing it. A failed call contributes none: the model
    still gets the ``[tool error]`` text, but there is no result to cite. Collecting
    immediately after each awaited call is what makes the backend's single-slot
    result buffer safe, since the calls run one at a time.
    """
    out: list[dict[str, Any]] = []
    for block in blocks:
        arguments = dict(block.input or {})
        try:
            text = await pool.call_tool(block.name, arguments)
        except Exception as exc:  # noqa: BLE001 — see docstring
            logger.warning("MCP tool %s execution failed: %s", block.name, exc)
            text = f"[tool error] {exc}"
        else:
            if native_blocks is not None:
                native_blocks.extend(_native_blocks_for_call(pool, block.name, arguments))
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


class _MessagesUsageAccumulator(TypedDict):
    input: int
    output: int
    iterations: list[Any]
    # Native server-tool blocks for the searches this loop ran, prepended to the
    # final content by ``fold_usage``. Always present, only filled when the
    # strategy was built with native emission on.
    native_blocks: list[Any]


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


class _MessagesStreamAccumulator(TypedDict):
    output_tokens: int
    started: int
    next_index: int
    iterations: list[Any]
    applied_edits: list[Any]


def _maybe_fold_message_delta(event: Any, acc: _MessagesStreamAccumulator) -> Any:
    """Fold usage and context-management telemetry from hidden iterations."""
    if getattr(event, "type", None) != "message_delta":
        return event
    usage = getattr(event, "usage", None)
    if usage is None or not hasattr(usage, "model_copy"):
        return event

    usage_update: dict[str, Any] = {}
    if acc["output_tokens"] > 0:
        usage_update["output_tokens"] = (getattr(usage, "output_tokens", 0) or 0) + acc["output_tokens"]
    if acc["iterations"]:
        usage_update["iterations"] = [*acc["iterations"], *(getattr(usage, "iterations", None) or [])]

    event_update: dict[str, Any] = {}
    if usage_update:
        event_update["usage"] = usage.model_copy(update=usage_update)
    if acc["applied_edits"]:
        context_management = getattr(event, "context_management", None)
        applied_edits = [
            *acc["applied_edits"],
            *(getattr(context_management, "applied_edits", None) or []),
        ]
        if context_management is not None and hasattr(context_management, "model_copy"):
            event_update["context_management"] = context_management.model_copy(
                update={"applied_edits": applied_edits}
            )
        else:
            event_update["context_management"] = BetaContextManagementResponse(applied_edits=applied_edits)

    return event.model_copy(update=event_update) if event_update else event


def _reindexed(event: Any, visible_index: int) -> Any:
    """Return ``event`` with its content-block ``index`` set to ``visible_index``.

    A no-op when the index already matches, which is every event of a
    single-iteration stream, so the common case stays byte-identical and no
    pydantic copy is made.
    """
    if getattr(event, "index", None) == visible_index or not hasattr(event, "model_copy"):
        return event
    return event.model_copy(update={"index": visible_index})


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
        # Native server-tool blocks for this iteration's searches, drained by
        # ``synthetic_events`` into content_block start/stop pairs.
        self.native_blocks: list[Any] = []
        self.stop_reason: str | None = None
        self.deferred_terminal: list[MessageStreamEvent] = []
        self.owned_specs: list[dict[str, Any]] = []
        # Blocks the gateway runs itself. Their caller-owned ``tool_use`` events are
        # swallowed because the gateway consumes them. MCP calls may instead get a
        # server-owned ``mcp_tool_use`` / ``mcp_tool_result`` representation.
        self.hidden_indices: set[int] = set()
        # Upstream block index -> the index the client sees. Each iteration's
        # blocks start again at 0 upstream, but the client is being shown one
        # single message, so forwarded blocks are renumbered.
        self.visible_index: dict[int, int] = {}


def _parsed_stream_input(state: _MessagesStreamState, spec: dict[str, Any]) -> dict[str, Any]:
    """Parse one buffered tool input, falling back to the loop's historical empty object."""
    try:
        value = json.loads(state.tool_use_json_bufs.get(spec["index"], "") or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _as_mcp_tool_backend(pool: ToolBackend) -> MCPToolBackend | None:
    """Narrow ``pool`` when it exposes the MCP streaming activity surface."""
    return pool if isinstance(pool, MCPToolBackend) else None


def _mcp_server_name(pool: MCPToolBackend | None, tool_name: str) -> str | None:
    """Return MCP ownership metadata when ``pool`` is an MCP client pool."""
    if pool is None:
        return None
    value = pool.server_name_for_tool(tool_name)
    return value if value else None


async def _call_stream_tool(
    pool: ToolBackend,
    name: str,
    arguments: dict[str, Any],
    *,
    mcp_backend: MCPToolBackend | None,
) -> tuple[str, bool, bool]:
    """Return ``(content, is_error, raised)`` for one gateway-owned call."""
    try:
        if mcp_backend is not None:
            outcome = await mcp_backend.call_tool_outcome(name, arguments)
            return outcome.content, outcome.is_error, False
        text = await pool.call_tool(name, arguments)
        return text, is_tool_error(text), False
    except Exception:  # noqa: BLE001 - recoverable tool failure is model input
        # Transport exceptions may include URLs, headers, or credentials. Neither
        # logs, client events, nor the model-facing result may receive that detail.
        logger.warning("Gateway tool %s execution failed", name)
        detail = "MCP tool execution failed" if mcp_backend is not None else "Gateway tool execution failed"
        return f"[tool error] {detail}", True, True


def _content_block_events(block: Any, acc: _MessagesStreamAccumulator) -> list[Any]:
    """A complete synthetic content block at the next client-visible index."""
    index = acc["next_index"]
    acc["next_index"] += 1
    return [
        ContentBlockStartEvent(content_block=block, index=index, type="content_block_start"),
        ContentBlockStopEvent(index=index, type="content_block_stop"),
    ]


async def _execute_stream_owned_events(
    state: _MessagesStreamState,
    pool: ToolBackend,
    acc: _MessagesStreamAccumulator,
    results: list[dict[str, Any]],
    *,
    native_blocks: list[Any] | None = None,
) -> AsyncGenerator[MessageStreamEvent, None]:
    """Execute owned calls and yield live MCP start/completion representations."""
    for spec in state.owned_specs:
        name = str(spec["name"])
        parsed_input = _parsed_stream_input(state, spec)
        mcp_backend = _as_mcp_tool_backend(pool)
        server_name = _mcp_server_name(mcp_backend, name)
        activity_id: str | None = None
        if server_name is not None:
            activity_id = f"{MCP_ACTIVITY_ID_PREFIX}{uuid.uuid4().hex}"
            start = BetaMCPToolUseBlock(
                type="mcp_tool_use",
                id=activity_id,
                name=name,
                server_name=server_name,
                input=parsed_input,
            )
            for event in _content_block_events(start, acc):
                yield event

        text, is_error, raised = await _call_stream_tool(
            pool,
            name,
            parsed_input,
            mcp_backend=mcp_backend if server_name is not None else None,
        )
        if not raised and native_blocks is not None:
            native_blocks.extend(_native_blocks_for_call(pool, name, parsed_input))
        results.append({"type": "tool_result", "tool_use_id": spec["id"], "content": text})

        if activity_id is not None:
            # A transport exception may embed URLs or headers. Expose only the
            # truthful failure state and the same fixed message sent to the model.
            activity_content = "[tool error] MCP tool execution failed" if raised else text
            completion = BetaMCPToolResultBlock(
                type="mcp_tool_result",
                tool_use_id=activity_id,
                content=activity_content,
                is_error=is_error,
            )
            for event in _content_block_events(completion, acc):
                yield event


class _MessagesToolLoopStrategy:
    """Anthropic Messages strategy for the generic tool loop.

    ``amessages`` is resolved as a module global at call time so tests can
    monkeypatch ``gateway.services.mcp_loop_messages.amessages``.

    ``emit_native_web_search`` is per-request (it depends on how the caller
    declared the tool), so a request that wants native blocks gets its own
    strategy instance rather than sharing the module-level default one.
    """

    transcript_key = "messages"

    def __init__(self, *, emit_native_web_search: bool = False) -> None:
        self._emit_native_web_search = emit_native_web_search

    def _native_sink(self, sink: list[Any]) -> list[Any] | None:
        """``sink`` when native emission is on, else ``None`` (collect nothing)."""
        return sink if self._emit_native_web_search else None

    def coerce_transcript(self, value: Any) -> list[Any]:
        return list(value or [])

    def convert_pool_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return openai_to_anthropic_tools(tools)

    # ---- non-streaming hooks ----

    async def call(self, kwargs: dict[str, Any]) -> MessageResponse:
        result: MessageResponse = await amessages(**kwargs)  # type: ignore[assignment]
        return result

    def new_usage_accumulator(self) -> _MessagesUsageAccumulator:
        return {"input": 0, "output": 0, "iterations": [], "native_blocks": []}

    def accumulate_usage(self, acc: _MessagesUsageAccumulator, result: MessageResponse) -> None:
        if result.usage:
            acc["input"] += result.usage.input_tokens or 0
            acc["output"] += result.usage.output_tokens or 0
            acc["iterations"].extend(result.usage.iterations or [])

    def fold_usage(self, result: MessageResponse, acc: _MessagesUsageAccumulator) -> None:
        _fold_usage(result, acc["input"], acc["output"])
        if result.usage is not None and acc["iterations"]:
            result.usage.iterations = acc["iterations"]
        # Prepend the native blocks for the searches this loop ran. The loop consumed
        # the model's own tool_use blocks, so without these a native client has no
        # way to know a search happened; they come first because they did.
        if acc["native_blocks"]:
            try:
                result.content = [*acc["native_blocks"], *(result.content or [])]
            except (AttributeError, TypeError):
                logger.warning("Could not add native web-search blocks to the response content")

    def exit_before_split(self, result: MessageResponse) -> bool:
        return False

    def split_owned(self, result: MessageResponse, pool: ToolBackend) -> tuple[list[Any], bool]:
        return _split_tool_uses(list(result.content or []), pool)

    def exit_after_split(self, result: MessageResponse) -> bool:
        # The model emitted tool_use blocks but stopped for another reason
        # (e.g. ``end_turn`` because ``max_tokens`` was hit mid-tool-call):
        # exit rather than try to execute them.
        return result.stop_reason != "tool_use"

    async def execute_owned(
        self, pool: ToolBackend, owned: list[Any], acc: _MessagesUsageAccumulator | None = None
    ) -> list[dict[str, Any]]:
        # Reached on the mixed-batch exit, where the owned subset runs for its side
        # effects. Collect its native blocks too: ``fold_usage`` runs on that path and
        # prepends them, so a native client still sees the search it paid for.
        native_sink = self._native_sink(acc["native_blocks"]) if acc is not None else None
        return await _execute_tool_uses(pool, owned, native_blocks=native_sink)

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
        acc: Any = None,
    ) -> None:
        # All-owned: continue the loop. Append the assistant turn (so the model
        # sees its own tool_use blocks) and a user turn carrying tool_result.
        content = list(result.content or [])
        transcript.append({"role": "assistant", "content": _content_to_dicts(content)})
        native_sink = self._native_sink(acc["native_blocks"]) if acc is not None else None
        transcript.append(
            {"role": "user", "content": await _execute_tool_uses(pool, owned, native_blocks=native_sink)}
        )

    # ---- streaming hooks ----

    async def open_stream(self, kwargs: dict[str, Any]) -> AsyncIterator[MessageStreamEvent]:
        stream: AsyncIterator[MessageStreamEvent] = await amessages(**kwargs)  # type: ignore[assignment]
        return stream

    def new_stream_state(self) -> _MessagesStreamState:
        return _MessagesStreamState()

    def new_stream_accumulator(self) -> _MessagesStreamAccumulator:
        # output_tokens and telemetry come from dropped intermediate
        # ``message_delta`` events. The final forwarded delta carries the
        # accumulated values so clients see the complete logical response.
        #
        # started / next_index: the client is shown ONE message even though the
        # gateway may have consumed several upstream ones, so only the first
        # ``message_start`` is forwarded and forwarded blocks are renumbered
        # continuously. Without this a tool-loop stream contains two
        # ``message_start`` events and reuses block index 0, which every SDK
        # stream accumulator rejects.
        return {
            "output_tokens": 0,
            "started": 0,
            "next_index": 0,
            "iterations": [],
            "applied_edits": [],
        }

    def observe(
        self,
        state: _MessagesStreamState,
        event: MessageStreamEvent,
        pool: ToolBackend,
        acc: _MessagesStreamAccumulator,
    ) -> tuple[StreamAction, MessageStreamEvent]:
        event_type = getattr(event, "type", None)

        if event_type == "message_start":
            # One envelope per response, no matter how many upstream messages the
            # tool loop consumed to produce it.
            if acc["started"]:
                return StreamAction.DEFER, event
            acc["started"] = 1
            return StreamAction.FORWARD, event

        if event_type == "content_block_start":
            block = event.content_block  # type: ignore[union-attr]
            idx = event.index  # type: ignore[union-attr]
            if block is not None and hasattr(block, "model_dump"):
                state.blocks_by_index[idx] = block.model_dump(exclude_none=True)
            else:
                state.blocks_by_index[idx] = dict(block) if isinstance(block, dict) else {}
            recorded = state.blocks_by_index[idx]
            if recorded.get("type") == "tool_use":
                state.tool_use_json_bufs[idx] = ""
                if pool.owns_tool(recorded.get("name", "")):
                    # Still recorded above: the loop needs the block and its
                    # buffered arguments to execute the call. Just not shown.
                    state.hidden_indices.add(idx)
                    return StreamAction.DEFER, event
            return StreamAction.FORWARD, _reindexed(event, self._visible_for(state, acc, idx))

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
            elif dtype == "compaction_delta":
                block_dict["content"] = (block_dict.get("content") or "") + (
                    getattr(delta, "content", "") or ""
                )
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
            return StreamAction.DEFER, event

        elif event_type == "message_stop":
            state.deferred_terminal.append(event)
            return StreamAction.BREAK, event

        idx = getattr(event, "index", None)
        if isinstance(idx, int):
            if idx in state.hidden_indices:
                return StreamAction.DEFER, event
            return StreamAction.FORWARD, _reindexed(event, self._visible_for(state, acc, idx))
        return StreamAction.FORWARD, event

    @staticmethod
    def _visible_for(state: _MessagesStreamState, acc: _MessagesStreamAccumulator, idx: int) -> int:
        """The client-visible index for an upstream block index, assigned in order."""
        if idx not in state.visible_index:
            state.visible_index[idx] = acc["next_index"]
            acc["next_index"] += 1
        return state.visible_index[idx]

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

    async def finalize_exit(
        self,
        state: _MessagesStreamState,
        pool: ToolBackend,
        acc: _MessagesStreamAccumulator,
    ) -> AsyncIterator[MessageStreamEvent]:
        # Mixed batch: the gateway's tool_use blocks were withheld from the stream, so
        # run them for their side effects rather than dropping the model's request.
        # Matches the non-streaming loop, which executes the owned subset and filters
        # it out of the returned content.
        if state.stop_reason == "tool_use" and state.owned_specs:
            # Collected, not discarded: the search ran, so a native client is owed the
            # pair describing it even though this round exits for the caller to
            # dispatch its own tool. ``terminal_events`` emits them. MCP activity is
            # yielded around the call itself so the start signal is truthful.
            discarded: list[dict[str, Any]] = []
            async for event in _execute_stream_owned_events(
                state,
                pool,
                acc,
                discarded,
                native_blocks=self._native_sink(state.native_blocks),
            ):
                yield event

    def terminal_events(
        self,
        state: _MessagesStreamState,
        acc: _MessagesStreamAccumulator,
    ) -> list[MessageStreamEvent]:
        # Any native blocks a mixed-batch exit collected go out first: they describe
        # work that happened before the message ended, and message_delta /
        # message_stop must stay last.
        blocks, state.native_blocks = state.native_blocks, []
        return [
            *self._native_block_events(blocks, acc),
            *(_maybe_fold_message_delta(term, acc) for term in state.deferred_terminal),
        ]

    def accumulate_stream_usage(self, acc: _MessagesStreamAccumulator, state: _MessagesStreamState) -> None:
        for term in state.deferred_terminal:
            if getattr(term, "type", None) != "message_delta":
                continue
            usage = getattr(term, "usage", None)
            if usage is not None:
                acc["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
                acc["iterations"].extend(getattr(usage, "iterations", None) or [])
            context_management = getattr(term, "context_management", None)
            if context_management is not None:
                acc["applied_edits"].extend(getattr(context_management, "applied_edits", None) or [])

    @staticmethod
    def _native_block_events(blocks: list[Any], acc: _MessagesStreamAccumulator) -> list[Any]:
        """content_block start/stop pairs for ``blocks``, numbered off ``acc``.

        Each block gets a ``content_block_start`` carrying the complete block plus a
        ``content_block_stop``, and no ``input_json_delta``: the SDK accumulator
        appends the start event's block wholesale and only overwrites ``input`` when a
        delta actually arrives, so a start/stop pair preserves the query.
        """
        return [event for block in blocks for event in _content_block_events(block, acc)]

    def synthetic_events(
        self, state: _MessagesStreamState, acc: _MessagesStreamAccumulator
    ) -> list[Any]:
        """Announce this iteration's gateway-run searches as native content blocks.

        The model's own ``tool_use`` events were swallowed, so a
        ``server_tool_use`` / ``web_search_tool_result`` pair takes their place for a
        caller that declared the tool natively. MCP activity is yielded directly
        around execution and therefore does not pass through this deferred hook.

        Each block gets a ``content_block_start`` carrying the complete block plus a
        ``content_block_stop``, and no ``input_json_delta``: the SDK accumulator
        appends the start event's block wholesale and only overwrites ``input`` when a
        delta actually arrives, so a start/stop pair preserves the query.
        """
        blocks, state.native_blocks = state.native_blocks, []
        return self._native_block_events(blocks, acc)

    async def advance_stream_transcript(
        self,
        transcript: list[Any],
        state: _MessagesStreamState,
        pool: ToolBackend,
        acc: _MessagesStreamAccumulator,
    ) -> AsyncIterator[MessageStreamEvent]:
        # Assistant message for the next round; preserve original block
        # ordering. tool_use blocks pick up the parsed input from their
        # JSON buffer.
        assistant_content: list[dict[str, Any]] = []
        for idx in sorted(state.blocks_by_index):
            block_dict = state.blocks_by_index[idx]
            if block_dict.get("type") == "tool_use":
                spec = {"index": idx}
                block_dict = {**block_dict, "input": _parsed_stream_input(state, spec)}
            assistant_content.append(block_dict)

        transcript.append({"role": "assistant", "content": assistant_content})
        tool_results: list[dict[str, Any]] = []
        async for event in _execute_stream_owned_events(
            state,
            pool,
            acc,
            tool_results,
            native_blocks=self._native_sink(state.native_blocks),
        ):
            yield event
        transcript.append({"role": "user", "content": tool_results})


_MESSAGES_STRATEGY = _MessagesToolLoopStrategy()


def _strategy_for(emit_native_web_search: bool) -> _MessagesToolLoopStrategy:
    """The shared strategy, or a per-request one when native emission is on.

    Strategies are stateless apart from that flag, so the common case keeps reusing
    the single module-level instance.
    """
    return _MessagesToolLoopStrategy(emit_native_web_search=True) if emit_native_web_search else _MESSAGES_STRATEGY


async def anthropic_tool_loop(
    *,
    completion_kwargs: dict[str, Any],
    pool: ToolBackend,
    max_iterations: int,
    on_first_response: Callable[[], None] | None = None,
    emit_native_web_search: bool = False,
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

    With ``emit_native_web_search``, the returned content is prefixed with a
    ``server_tool_use`` / ``web_search_tool_result`` pair per gateway-run search.
    """
    return await run_tool_loop(
        strategy=_strategy_for(emit_native_web_search),
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
    emit_native_web_search: bool = False,
) -> AsyncGenerator[MessageStreamEvent, None]:
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
      3. Forward caller-owned blocks, but hide gateway-owned ``tool_use`` events.
         Defer ``message_delta`` and ``message_stop`` until the loop knows whether
         the logical message will continue.
      4. On ``message_stop``: if all buffered tool uses are gateway-owned, execute
         them, append the hidden call and result to the internal transcript, and
         continue. MCP execution is represented live as server-owned
         ``mcp_tool_use`` / ``mcp_tool_result`` blocks. If foreign blocks exist or
         no owned blocks were buffered, forward the terminal events and exit.

    Re-emitting a synthetic ``message_start`` for the next iteration is not
    needed because ``amessages`` produces a fresh stream; the next call's
    natural ``message_start`` arrives downstream as if nothing had happened.
    """
    # aclosing makes downstream closes (client disconnect) propagate to the
    # engine generator, and through it to the upstream provider stream,
    # instead of waiting for event-loop async-generator finalization.
    async with aclosing(
        run_tool_loop_stream(
            strategy=_strategy_for(emit_native_web_search),
            completion_kwargs=completion_kwargs,
            pool=pool,
            max_iterations=max_iterations,
        )
    ) as inner:
        async for event in inner:
            yield event
