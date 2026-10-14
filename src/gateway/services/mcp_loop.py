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
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from any_llm.types.completion import ChatCompletionChunk, ChoiceDelta, ChunkChoice, PromptTokensDetails

from gateway.core.env import otari_env
from gateway.log_config import logger
from gateway.models.usage import PLAYGROUND_USAGE_ENDPOINT
from gateway.services._tool_loop import (
    MaxToolIterationsExceeded,
    StreamAction,
    ToolBackend,
    log_tool_failure,
    run_tool_loop,
    run_tool_loop_stream,
    tool_failure_detail,
)
from gateway.services.inference import require_choices
from gateway.services.providers import acompletion
from gateway.services.tool_usage import is_tool_error
from gateway.services.tools import (
    MAX_TOOL_ITERATIONS_CAP,
    MAX_USES_EXCEEDED_ERROR,
    OTARI_TOOL_CALLS_FIELD,
    Dialect,
    NativeCall,
    ToolUseBudget,
    is_capped_call,
    native_rendering,
    owning_backend,
)

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletion, CompletionUsage

DEFAULT_MAX_TOOL_ITERATIONS = 10


__all__ = [
    "DEFAULT_MAX_TOOL_ITERATIONS",
    "MAX_TOOL_ITERATIONS_CAP",
    "PURPOSE_HINT_HEADER",
    "TOOL_ACTIVITY_FIELD",
    "MaxToolIterationsExceeded",
    "ToolBackend",
    "inject_purpose_hints",
    "mcp_tool_loop",
    "mcp_tool_loop_stream",
    "reports_tool_activity",
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

    The merge target is the first message whose ``role`` is ``"system"``, wherever
    it sits (a ``developer`` message may legally precede it); one is inserted at
    the front when the conversation has none. ``content`` may be a string or a
    list of content parts, so a list keeps its shape and gains a leading text
    part rather than being stringified.

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
    system_idx = next((i for i, m in enumerate(out) if m.get("role") == "system"), None)
    if system_idx is None:
        out.insert(0, {"role": "system", "content": block})
        return out

    existing = out[system_idx].get("content")
    if existing is None:
        new_content: Any = block
    elif isinstance(existing, str):
        new_content = f"{existing}\n\n{block}" if existing else block
    elif isinstance(existing, list):
        new_content = [{"type": "text", "text": block}, *existing]
    else:
        new_content = block
    out[system_idx] = {**out[system_idx], "content": new_content}
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


async def _run_owned_calls(
    pool: ToolBackend,
    mcp_calls: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    budget: ToolUseBudget | None,
    native_tools: frozenset[str],
) -> AsyncGenerator[tuple[bool, list[dict[str, Any]]], None]:
    """Run each gateway-owned call, append its tool-role message to ``results``, and yield its report.

    Tool failures (network errors, server errors, schema mismatches, MCP-specific
    or httpx-level transport errors) are converted to a ``[tool error] ...``
    message so the model can recover. Only cancellation/interrupt-class
    exceptions (``asyncio.CancelledError``, ``KeyboardInterrupt``) escape; they
    inherit from ``BaseException`` and never reach the ``Exception`` clause.
    That's the standard idiom for "treat tool failures as recoverable, let
    cancellation propagate".

    A call past ``budget`` is refused the same way, as a tool error the model reads.

    For a call to one of ``native_tools``, the request's tools announced in
    ``otari_tool_calls``, this yields ``(False, entries)`` just before the call runs
    and ``(True, entries)`` right after it returns, so a stream can send each while
    the call waits or before the next one starts. A refused call yields only its
    ended entries, since nothing ran.
    """
    for tc in mcp_calls:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        rendering = native_rendering(name, Dialect.CHAT) if name in native_tools else None
        call = NativeCall(name, tc["id"] or "", args if isinstance(args, dict) else {})
        capped = is_capped_call(budget, pool, name)
        if capped and budget is not None and budget.exhausted(name):
            results.append({"role": "tool", "tool_call_id": tc["id"] or "", "content": MAX_USES_EXCEEDED_ERROR})
            if rendering is not None:
                yield True, rendering.refused(call)
            continue
        if rendering is not None:
            yield False, rendering.started(call)
        owner = owning_backend(pool, name)
        try:
            text = await owner.call_tool(name, args)
        except MaxToolIterationsExceeded:
            # The request's shared search and fetch limit refused the call inside the
            # backend, after its started entry went out, so the entry ends before the
            # request fails.
            if rendering is not None:
                yield True, rendering.refused(call)
            raise
        except Exception as exc:  # noqa: BLE001 — see docstring
            detail = tool_failure_detail(owner, name)
            log_tool_failure(name, detail, exc)
            text = f"[tool error] {detail}"
        else:
            if capped and budget is not None:
                budget.record(name, text)
        results.append({"role": "tool", "tool_call_id": tc["id"] or "", "content": text})
        if rendering is not None:
            yield True, rendering.ran(replace(call, failed=is_tool_error(text)), owner)


async def _execute_mcp_calls(
    pool: ToolBackend,
    mcp_calls: list[dict[str, Any]],
    *,
    budget: ToolUseBudget | None = None,
    native_tools: frozenset[str] = frozenset(),
    ended: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run each gateway-owned call (see :func:`_run_owned_calls`) and return the tool-role messages.

    The ``otari_tool_calls`` entries of each ended call are added to ``ended``, which
    a response that is not streamed carries; the started ones have no reader there.
    """
    results: list[dict[str, Any]] = []
    async with aclosing(
        _run_owned_calls(pool, mcp_calls, results, budget=budget, native_tools=native_tools)
    ) as reports:
        async for call_ended, entries in reports:
            if call_ended and ended is not None:
                ended.extend(entries)
    return results


def _tool_calls_chunk(first: ChatCompletionChunk, entries: list[dict[str, Any]]) -> ChatCompletionChunk:
    """A chunk carrying ``entries`` beside its ``choices``, with the round's ``id``, ``model`` and ``created``.

    One choice with an empty ``delta`` keeps the chunk ordinary to a client that reads
    only ``choices``; the entries sit at the top level for the reason
    ``services/tools/_chat_tool_calls.py`` gives.
    """
    chunk = ChatCompletionChunk(
        id=first.id,
        object="chat.completion.chunk",
        created=first.created,
        model=first.model,
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason=None)],
    )
    setattr(chunk, OTARI_TOOL_CALLS_FIELD, entries)
    return chunk


@dataclass
class _ChatTotals:
    """What the chat loop carries across rounds: the usage of the rounds it hid, and the report of its calls.

    ``tool_calls`` holds the ended ``otari_tool_calls`` entries a response that is not
    streamed carries. A stream sends each entry as its call starts and ends instead,
    so it stays empty there.
    """

    prompt: int = 0
    completion: int = 0
    cache_read: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def _add_usage(acc: _ChatTotals, usage: CompletionUsage | None) -> None:
    """Add one round's usage to the loop's running totals."""
    if usage is None:
        return
    acc.prompt += usage.prompt_tokens or 0
    acc.completion += usage.completion_tokens or 0
    details = usage.prompt_tokens_details
    if details is not None:
        acc.cache_read += details.cached_tokens or 0


def _fold_usage(
    completion: ChatCompletion | ChatCompletionChunk,
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


# The field a tool activity chunk carries. The Chat wire has no vocabulary for a
# gateway-run tool call, so an opted-in stream (the Playground's) reports it in a
# field of its own, beside ``choices`` for the reason ``otari_tool_calls`` sits
# there (``services/tools/_chat_tool_calls.py``). In the delta it would also be
# copied under ``provider_specific_fields``, sending every record twice.
TOOL_ACTIVITY_FIELD = "otari_tool_activity"

# A tool result can be a whole fetched page; the transcript shows a preview.
_ACTIVITY_RESULT_LIMIT = 8_000
_TOOL_ERROR_PREFIX = "[tool error]"


def reports_tool_activity(endpoint: str) -> bool:
    """Whether a completion served at ``endpoint`` streams gateway-run tool activity: the Playground's does."""
    return endpoint == PLAYGROUND_USAGE_ENDPOINT


def _activity_chunk(template: ChatCompletionChunk, payload: dict[str, Any]) -> ChatCompletionChunk:
    """A content-free chunk carrying one tool activity record, with ``template``'s ``id``, ``model`` and ``created``."""
    chunk = ChatCompletionChunk(
        id=template.id,
        object="chat.completion.chunk",
        created=template.created,
        model=template.model,
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason=None)],
    )
    setattr(chunk, TOOL_ACTIVITY_FIELD, payload)
    return chunk


def _call_activity(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function", {})
    return {
        "type": "tool_call",
        "id": call.get("id") or "",
        "name": function.get("name", ""),
        "arguments": function.get("arguments") or "",
    }


def _result_activity(call: dict[str, Any], message: dict[str, Any]) -> dict[str, Any]:
    content = str(message.get("content") or "")
    return {
        "type": "tool_result",
        "id": call.get("id") or "",
        "name": call.get("function", {}).get("name", ""),
        "content": content[:_ACTIVITY_RESULT_LIMIT],
        "truncated": len(content) > _ACTIVITY_RESULT_LIMIT,
        "is_error": content.startswith(_TOOL_ERROR_PREFIX),
    }


class _ChatStreamState:
    """Per-iteration bookkeeping for the chat streaming loop."""

    def __init__(self) -> None:
        self.slots: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.pending_terminal: ChatCompletionChunk | None = None
        # The ``include_usage`` chunks (usage set, no choices) that OpenAI, and
        # any-llm's Anthropic adapter, send after the finish chunk. Held back with
        # it, so the rounds the loop hides are billed on the one the client sees.
        self.usage_chunks: list[ChatCompletionChunk] = []
        self.mcp_calls: list[dict[str, Any]] = []
        # The round's first chunk with choices, whose id, model and created the
        # gateway's own ``otari_tool_calls`` chunks repeat. A provider's preamble
        # chunk with no choices may carry neither.
        self.first: ChatCompletionChunk | None = None
        # Upstream tool_call index -> the index the client sees. Gateway-owned calls
        # are dropped from the stream, so the surviving foreign calls have to be
        # renumbered into a gapless sequence: an SDK accumulator indexes its snapshot
        # array by the index it is handed, and a first fragment numbered 1 makes the
        # official OpenAI client raise IndexError.
        self.visible_tool_index: dict[int, int] = {}
        self.next_visible_tool_index = 0
        # The round's first upstream chunk: activity chunks copy its id, model and
        # timestamp so they read as part of the same stream.
        self.template: ChatCompletionChunk | None = None

    def deferred_terminal(self) -> list[ChatCompletionChunk]:
        """The round's held-back chunks, in stream order."""
        head = [self.pending_terminal] if self.pending_terminal is not None else []
        return [*head, *self.usage_chunks]

    def usage_carrier(self) -> ChatCompletionChunk | None:
        """The chunk carrying this round's usage report: the last one that has usage."""
        return next((c for c in reversed(self.deferred_terminal()) if c.usage is not None), None)


class _ChatToolLoopStrategy:
    """Chat-completions strategy for the generic tool loop.

    ``acompletion`` is resolved as a module global at call time so tests can
    monkeypatch ``gateway.services.mcp_loop.acompletion``.
    """

    transcript_key = "messages"

    def __init__(
        self,
        *,
        budget: ToolUseBudget | None = None,
        native_tools: frozenset[str] = frozenset(),
        emit_activity: bool = False,
    ) -> None:
        # Absent unless the caller capped the searches, is owed a report or asked
        # for activity, which keeps the shared instance in ``_strategy_for`` free
        # of per-request state.
        self._budget = budget
        # The gateway-run tools whose calls this request reports in ``otari_tool_calls``.
        self._native_tools = native_tools
        self._emit_activity = emit_activity

    def coerce_transcript(self, value: Any) -> list[Any]:
        return list(value or [])

    def convert_pool_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return tools

    # ---- non-streaming hooks ----

    async def call(self, kwargs: dict[str, Any]) -> ChatCompletion:
        completion: ChatCompletion = await acompletion(**kwargs)  # type: ignore[assignment]
        return completion

    def new_usage_accumulator(self) -> _ChatTotals:
        return _ChatTotals()

    def accumulate_usage(self, acc: _ChatTotals, result: ChatCompletion) -> None:
        _add_usage(acc, result.usage)

    def fold_usage(self, result: ChatCompletion, acc: _ChatTotals) -> None:
        _fold_usage(result, acc.prompt, acc.completion, acc.cache_read)
        # Every return path folds, so this is where the answer gains the report of
        # the calls the loop ran and the client never saw.
        if acc.tool_calls:
            setattr(result, OTARI_TOOL_CALLS_FIELD, list(acc.tool_calls))

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

    async def _run_into(self, pool: ToolBackend, owned: list[Any], acc: _ChatTotals | None) -> list[dict[str, Any]]:
        """Run ``owned``, adding each ended ``otari_tool_calls`` entry to ``acc`` for the answer to carry."""
        ended = acc.tool_calls if acc is not None else None
        return await _execute_mcp_calls(pool, owned, budget=self._budget, native_tools=self._native_tools, ended=ended)

    async def execute_owned(
        self, pool: ToolBackend, owned: list[Any], acc: _ChatTotals | None = None
    ) -> list[dict[str, Any]]:
        # A mixed batch's owned calls ran, so the answer reports them like any other.
        return await self._run_into(pool, owned, acc)

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
        acc: _ChatTotals | None = None,
    ) -> None:
        transcript.append({"role": "assistant", "tool_calls": owned})
        transcript.extend(await self._run_into(pool, owned, acc))

    # ---- streaming hooks ----

    async def open_stream(self, kwargs: dict[str, Any]) -> AsyncIterator[ChatCompletionChunk]:
        stream: AsyncIterator[ChatCompletionChunk] = await acompletion(**kwargs)  # type: ignore[assignment]
        return stream

    def new_stream_state(self) -> _ChatStreamState:
        return _ChatStreamState()

    def new_stream_accumulator(self) -> _ChatTotals:
        # Usage of the rounds the loop hid, folded into the final round's report
        # by ``terminal_events``: the client sees one usage chunk for the loop.
        return self.new_usage_accumulator()

    def observe(
        self,
        state: _ChatStreamState,
        event: ChatCompletionChunk,
        pool: ToolBackend,
        acc: _ChatTotals,
    ) -> tuple[StreamAction, ChatCompletionChunk]:
        if state.template is None:
            state.template = event
        if state.first is None and event.choices:
            state.first = event
        if not event.choices and event.usage is not None:
            state.usage_chunks.append(event)
            return StreamAction.DEFER, event
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

    async def finalize_exit(
        self, state: _ChatStreamState, pool: ToolBackend, acc: _ChatTotals
    ) -> AsyncIterator[ChatCompletionChunk]:
        del acc
        # A mixed batch's owned calls run here, before the terminal chunk, and are
        # reported like any other; their results are not needed, since the caller
        # continues the conversation.
        async for chunk in self._run_reported(state, pool, []):
            yield chunk

    def terminal_events(self, state: _ChatStreamState, acc: _ChatTotals) -> list[ChatCompletionChunk]:
        carrier = state.usage_carrier()
        if carrier is not None:
            totals = replace(acc, tool_calls=[])
            _add_usage(totals, carrier.usage)
            _fold_usage(carrier, totals.prompt, totals.completion, totals.cache_read)
        return state.deferred_terminal()

    def accumulate_stream_usage(self, acc: _ChatTotals, state: _ChatStreamState) -> None:
        carrier = state.usage_carrier()
        _add_usage(acc, carrier.usage if carrier is not None else None)

    def synthetic_events(self, state: _ChatStreamState, acc: _ChatTotals) -> list[Any]:
        # Each call was reported around its own execution, in ``_run_reported``, so
        # nothing is left to announce once the round's calls have run.
        return []

    async def advance_stream_transcript(
        self,
        transcript: list[Any],
        state: _ChatStreamState,
        pool: ToolBackend,
        acc: _ChatTotals,
    ) -> AsyncIterator[ChatCompletionChunk]:
        del acc
        # All-MCP: the terminal chunk was silently dropped so the client
        # doesn't think this iteration's response was the final answer.
        results: list[dict[str, Any]] = []
        async for chunk in self._run_reported(state, pool, results):
            yield chunk
        transcript.append({"role": "assistant", "tool_calls": state.mcp_calls})
        transcript.extend(results)

    async def _run_reported(
        self, state: _ChatStreamState, pool: ToolBackend, results: list[dict[str, Any]]
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Run the round's owned calls into ``results``, reporting each as it starts and ends.

        Every request gets its ``otari_tool_calls`` chunks; an opted-in one (the
        Playground's) also gets a :data:`TOOL_ACTIVITY_FIELD` chunk around each call.
        """
        template = state.template if self._emit_activity else None
        if template is None:
            async for chunk in self._run_calls(state, pool, state.mcp_calls, results):
                yield chunk
            return
        # One call at a time, so each result is reported as soon as it exists
        # instead of after the slowest call in the batch.
        for call in state.mcp_calls:
            yield _activity_chunk(template, _call_activity(call))
            ran: list[dict[str, Any]] = []
            async for chunk in self._run_calls(state, pool, [call], ran):
                yield chunk
            results.extend(ran)
            yield _activity_chunk(template, _result_activity(call, ran[0]))

    async def _run_calls(
        self, state: _ChatStreamState, pool: ToolBackend, calls: list[dict[str, Any]], results: list[dict[str, Any]]
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Run ``calls`` into ``results``, with an ``otari_tool_calls`` chunk as each starts and ends."""
        reports = _run_owned_calls(pool, calls, results, budget=self._budget, native_tools=self._native_tools)
        async with aclosing(reports):
            async for _ended, entries in reports:
                if entries and state.first is not None:
                    yield _tool_calls_chunk(state.first, entries)


_CHAT_STRATEGY = _ChatToolLoopStrategy()


def _strategy_for(
    budget: ToolUseBudget | None, native_tools: frozenset[str] = frozenset(), *, emit_activity: bool = False
) -> _ChatToolLoopStrategy:
    """The shared strategy, or a per-request one when the caller capped searches, is owed a report or wants activity.

    Only such a request has anything per-request to hold, so every other request
    keeps reusing the single module-level instance.
    """
    if budget is None and not native_tools and not emit_activity:
        return _CHAT_STRATEGY
    return _ChatToolLoopStrategy(budget=budget, native_tools=native_tools, emit_activity=emit_activity)


async def mcp_tool_loop_stream(
    *,
    completion_kwargs: dict[str, Any],
    pool: ToolBackend,
    max_iterations: int,
    use_budget: ToolUseBudget | None = None,
    native_tools: frozenset[str] = frozenset(),
    emit_activity: bool = False,
) -> AsyncGenerator[ChatCompletionChunk, None]:
    """Yield chunks across multiple `acompletion(stream=True)` calls, with MCP execution between rounds.

    Tool-call deltas for the gateway's own calls are held back, since the client can
    never send their results; the caller's own calls are forwarded, renumbered. The
    *terminal* chunk of each intermediate iteration, the one carrying
    ``finish_reason="tool_calls"``, is buffered and dropped if the loop is going to
    iterate again. Forwarding that chunk would tell an OpenAI-compatible client "this
    is the final answer", and most SDKs stop reading at that point; subsequent
    iterations' content would be silently truncated.

    A call to one of ``native_tools`` is reported in an ``otari_tool_calls`` chunk
    when it starts and another when it ends.

    The terminal chunk is forwarded in three cases:
      * the iteration ended with a non-``tool_calls`` finish_reason (e.g. ``stop``),
      * the model produced foreign tool_calls (caller needs to dispatch them), or
      * the model produced no MCP-owned calls at all (loop exits, terminal goes through).

    ``emit_activity`` reports each gateway-run call and its result as a chunk
    carrying :data:`TOOL_ACTIVITY_FIELD`, for a client (the Playground) that
    shows them. Off, those calls stay invisible on the wire.
    """
    # aclosing makes downstream closes (client disconnect) propagate to the
    # engine generator, and through it to the upstream provider stream,
    # instead of waiting for event-loop async-generator finalization.
    async with aclosing(
        run_tool_loop_stream(
            strategy=_strategy_for(use_budget, native_tools, emit_activity=emit_activity),
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
    use_budget: ToolUseBudget | None = None,
    native_tools: frozenset[str] = frozenset(),
) -> ChatCompletion:
    """Non-streaming variant. Accumulates usage across iterations into the returned completion.

    The completion carries the ended ``otari_tool_calls`` entry of each call to one of
    ``native_tools``, across every round.

    ``on_first_response`` follows the provider lock-in contract documented on
    :func:`gateway.services._tool_loop.run_tool_loop`; the hybrid-mode attempt
    loop in :mod:`gateway.api.routes.chat` is the consumer.

    A first response with no choices is a provider failure, as on a call without
    tools. A later one is returned as the answer, so the rounds before it are billed.
    """
    return await run_tool_loop(
        strategy=_strategy_for(use_budget, native_tools),
        completion_kwargs=completion_kwargs,
        pool=pool,
        max_iterations=max_iterations,
        on_first_response=on_first_response,
        check_first_response=require_choices,
    )
