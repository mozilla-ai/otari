"""OpenAI Responses API variant of the MCP tool-use loop.

Mirrors :mod:`gateway.services.mcp_loop` and
:mod:`gateway.services.mcp_loop_messages` but speaks the OpenAI Responses wire
shape: ``Response.output`` items (``function_call`` entries) instead of
``tool_calls``; ``function_call_output`` input items as tool results;
``response.output_item.*`` / ``response.function_call_arguments.*`` /
``response.completed`` stream events.

The loop skeleton itself lives in :mod:`gateway.services._tool_loop`; this
module supplies the Responses strategy and thin public wrappers.

The duck-typed pool interface (``owns_tool`` / ``call_tool`` /
``openai_tools`` / ``purpose_hints``) is reused unchanged; the
``openai_tools`` shape is converted at the boundary in :mod:`tool_format`.

Native server-managed tools that the Responses API can run upstream
(``web_search_call``, ``code_interpreter_call``, ``mcp_call``, etc.) are not
intercepted here; those output items belong to the provider's own tool
execution path and the gateway has nothing to dispatch against.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

from any_llm import aresponses
from openai.types.responses import ResponseFunctionWebSearch
from openai.types.responses.response_function_web_search import ActionSearch
from openai.types.responses.response_output_item_added_event import ResponseOutputItemAddedEvent
from openai.types.responses.response_output_item_done_event import ResponseOutputItemDoneEvent

from gateway.core.observation import NormalizedTool, normalized_tool
from gateway.core.usage import GatewayUsage, responses_usage
from gateway.log_config import logger
from gateway.services._tool_loop import StreamAction, run_tool_loop, run_tool_loop_stream
from gateway.services.mcp_loop import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    MAX_TOOL_ITERATIONS_CAP,
    MaxToolIterationsExceeded,
    ToolBackend,
)
from gateway.services.tool_format import openai_to_responses_tools
from gateway.services.web_search_backend import WEB_SEARCH_TOOL_NAME

if TYPE_CHECKING:
    from any_llm.types.responses import Response, ResponseStreamEvent


__all__ = [
    "DEFAULT_MAX_TOOL_ITERATIONS",
    "MAX_TOOL_ITERATIONS_CAP",
    "MaxToolIterationsExceeded",
    "responses_tool_loop",
    "responses_tool_loop_stream",
]


def _split_function_calls(
    output: list[Any],
    pool: ToolBackend,
) -> tuple[list[Any], bool]:
    """Return (owned_function_call_items, has_foreign).

    Walks ``output`` for items with ``type == "function_call"`` and partitions
    by ``pool.owns_tool(item.name)``. Non-function-call items (text messages,
    web_search_call, code_interpreter_call, mcp_call, ...) are ignored; they're
    not gateway-managed tool dispatch.
    """
    owned: list[Any] = []
    has_foreign = False
    for item in output:
        if getattr(item, "type", None) != "function_call":
            continue
        if pool.owns_tool(item.name):
            owned.append(item)
        else:
            has_foreign = True
    return owned, has_foreign


async def _execute_function_calls(
    pool: ToolBackend,
    items: list[Any],
) -> list[dict[str, Any]]:
    """Run each owned function_call and return the Responses function_call_output items.

    Tool failures convert to a ``[tool error] ...`` string in the output so the
    model can recover. Only cancellation-class exceptions escape; same idiom
    as :func:`gateway.services.mcp_loop._execute_mcp_calls`.
    """
    out: list[dict[str, Any]] = []
    for item in items:
        try:
            args = json.loads(item.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        try:
            text = await pool.call_tool(item.name, args)
        except Exception as exc:  # noqa: BLE001 — see docstring
            logger.warning("MCP tool %s execution failed: %s", item.name, exc)
            text = f"[tool error] {exc}"
        out.append({"type": "function_call_output", "call_id": item.call_id, "output": text})
    return out


def _items_to_dicts(items: list[Any]) -> list[dict[str, Any]]:
    """Serialize Response output items back to wire shape for the next round's input."""
    out: list[dict[str, Any]] = []
    for item in items:
        if hasattr(item, "model_dump"):
            out.append(item.model_dump(exclude_none=True))
        elif isinstance(item, dict):
            out.append(item)
        else:
            out.append(dict(item))
    return out


def _coerce_input_to_list(input_data: Any) -> list[Any]:
    """Normalize ``input_data`` to a list so the tool loop can append items.

    The Responses API accepts ``input`` as either a string (treated as a single
    user message) or a list of input items. To continue the conversation after
    a tool round we have to append items, which requires the list form.
    """
    if input_data is None:
        return []
    if isinstance(input_data, list):
        return list(input_data)
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]
    return [input_data]


def _fold_usage(result: Response, input_total: int, output_total: int, total_total: int) -> None:
    """Replace ``result.usage`` token counts with the loop's running totals."""
    if result.usage is None:
        return
    result.usage.input_tokens = input_total
    result.usage.output_tokens = output_total
    result.usage.total_tokens = total_total


def _maybe_fold_response_completed_usage(event: Any, acc_output_tokens: int) -> Any:
    """Return a ``response.completed`` event with cumulative output_tokens folded in.

    Pass-through for any other event type or when ``acc_output_tokens`` is
    zero. The Responses streaming usage report is read off
    ``event.response.usage``; we ``model_copy`` the Response with an updated
    Usage so consumers (``streaming_generator``) see the full tool-loop
    output count instead of only the final iteration's.
    """
    if acc_output_tokens <= 0:
        return event
    if getattr(event, "type", None) != "response.completed":
        return event
    response_obj = getattr(event, "response", None)
    usage = getattr(response_obj, "usage", None) if response_obj is not None else None
    if usage is None or not hasattr(usage, "model_copy") or response_obj is None:
        return event
    new_output = (getattr(usage, "output_tokens", 0) or 0) + acc_output_tokens
    new_total = (getattr(usage, "total_tokens", 0) or 0) + acc_output_tokens
    new_usage = usage.model_copy(update={"output_tokens": new_output, "total_tokens": new_total})
    new_response = response_obj.model_copy(update={"usage": new_usage})
    return event.model_copy(update={"response": new_response})


def _reoutput_indexed(event: Any, visible_index: int) -> Any:
    """Return ``event`` with its ``output_index`` set to ``visible_index``.

    A no-op when it already matches, so a single-iteration stream stays
    byte-identical to the upstream one.
    """
    if getattr(event, "output_index", None) == visible_index or not hasattr(event, "model_copy"):
        return event
    return event.model_copy(update={"output_index": visible_index})


def _web_search_call_item(call_id: str, query: str) -> ResponseFunctionWebSearch:
    """The Responses API's native "the server ran a search" output item.

    This is the one place the gateway's own tool work is expressible in a
    provider's native vocabulary: ``ResponseFunctionWebSearch`` needs only an id,
    an action, and a status, all of which the gateway legitimately knows. The
    Anthropic equivalent is not expressible, because its result block requires an
    Anthropic-signed ``encrypted_content`` blob (see docs/tools.md).
    """
    return ResponseFunctionWebSearch(
        id=call_id,
        action=ActionSearch(type="search", query=query),
        status="completed",
        type="web_search_call",
    )


def _web_search_items_for(owned: list[Any]) -> list[ResponseFunctionWebSearch]:
    """Native items for the gateway-run searches among ``owned``.

    Only ``web_search`` maps to a Responses item the gateway can emit honestly. A
    sandbox or MCP call has no native equivalent (``code_interpreter_call`` means
    OpenAI's own interpreter ran, which would be a lie), so those stay invisible.
    """
    items: list[ResponseFunctionWebSearch] = []
    for item in owned:
        if getattr(item, "name", None) != WEB_SEARCH_TOOL_NAME:
            continue
        try:
            query = str(json.loads(getattr(item, "arguments", "") or "{}").get("query") or "")
        except json.JSONDecodeError:
            query = ""
        items.append(_web_search_call_item(getattr(item, "call_id", "") or "", query))
    return items


def _compaction_items(output: list[Any]) -> list[Any]:
    """Return provider compaction items in output order."""
    return [item for item in output if getattr(item, "type", None) == "compaction"]


def _replay_items(output: list[Any], owned: list[Any]) -> list[Any]:
    """Keep compaction and owned function calls in provider output order."""
    owned_call_ids = {getattr(item, "call_id", None) for item in owned}
    return [
        item
        for item in output
        if getattr(item, "type", None) == "compaction"
        or (
            getattr(item, "type", None) == "function_call"
            and getattr(item, "call_id", None) in owned_call_ids
        )
    ]


async def _execute_stream_owned(state: "_ResponsesStreamState", pool: ToolBackend) -> list[dict[str, Any]]:
    """Run the stream's gateway-owned function calls, returning their output items.

    Shared by the continue path and the mixed-batch exit so both parse the buffered
    arguments identically.
    """
    results: list[dict[str, Any]] = []
    for spec in state.owned_specs:
        try:
            args = json.loads(spec.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        try:
            text = await pool.call_tool(spec["name"], args)
        except Exception as exc:  # noqa: BLE001 (same tool-error-as-message idiom as the non-stream loop)
            logger.warning("MCP tool %s execution failed: %s", spec["name"], exc)
            text = f"[tool error] {exc}"
        results.append({"type": "function_call_output", "call_id": spec["call_id"], "output": text})
    return results


def _hidden_call_ids(state: "_ResponsesStreamState") -> set[str]:
    """``call_id``s of the gateway-owned calls whose item events were withheld."""
    return {
        str(spec.get("call_id"))
        for idx, spec in state.function_calls.items()
        if idx in state.hidden_output_indices and spec.get("call_id")
    }


def _without_output_items(event: Any, call_ids: set[str]) -> Any:
    """Return a ``response.completed`` event with the named function calls removed."""
    response_obj = getattr(event, "response", None)
    output = getattr(response_obj, "output", None)
    if response_obj is None or not output:
        return event
    kept = [
        item
        for item in output
        if not (getattr(item, "type", None) == "function_call" and getattr(item, "call_id", None) in call_ids)
    ]
    if len(kept) == len(output):
        return event
    try:
        return event.model_copy(update={"response": response_obj.model_copy(update={"output": kept})})
    except (AttributeError, TypeError):
        logger.warning("Could not filter gateway function_call items from response.completed")
        return event


def _prepend_output_items(event: Any, items: list[Any]) -> Any:
    """Prepend hidden-iteration output items to a terminal response."""
    if not items:
        return event
    response_obj = getattr(event, "response", None)
    if response_obj is None:
        return event
    try:
        output = list(items) + list(getattr(response_obj, "output", None) or [])
        return event.model_copy(update={"response": response_obj.model_copy(update={"output": output})})
    except (AttributeError, TypeError):
        logger.warning("Could not add hidden Responses output items to response.completed")
        return event


class _ResponsesStreamState:
    """Per-iteration bookkeeping for the Responses streaming loop."""

    def __init__(self) -> None:
        # output_index -> {"call_id", "name", "arguments"}
        self.function_calls: dict[int, dict[str, Any]] = {}
        self.compaction_items: dict[int, Any] = {}
        self.deferred_completed: ResponseStreamEvent | None = None
        # This round's provider-reported usage, for the Reprise usage snapshot
        # (otari-ai#1647). Taken from any event that carries one rather than from
        # ``deferred_completed`` alone, because a round truncated by
        # ``max_output_tokens`` or a content filter ends on ``response.incomplete``,
        # which the gateway bills and which would otherwise read as unknown usage.
        # Recorded only: no event's handling changes.
        self.usage: Any = None
        self.owned_specs: list[dict[str, Any]] = []
        # Output items the gateway runs itself. Their events are swallowed: the
        # client can never be sent a ``function_call_output`` for a call the
        # gateway consumed, so showing it the call is a dead end.
        self.hidden_output_indices: set[int] = set()
        # Upstream output index -> the index the client sees. Each iteration
        # numbers its own output from 0, but the client is shown one response, so
        # forwarded items are renumbered continuously.
        self.visible_output_index: dict[int, int] = {}


class _ResponsesToolLoopStrategy:
    """OpenAI Responses strategy for the generic tool loop.

    ``aresponses`` is resolved as a module global at call time so tests can
    monkeypatch ``gateway.services.mcp_loop_responses.aresponses``.
    """

    transcript_key = "input_data"

    def coerce_transcript(self, value: Any) -> list[Any]:
        return _coerce_input_to_list(value)

    def convert_pool_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return openai_to_responses_tools(tools)

    def normalize_tools(self, tools: list[dict[str, Any]]) -> list[NormalizedTool]:
        return [normalized_tool(tool, "parameters") for tool in tools]

    # ---- non-streaming hooks ----

    async def call(self, kwargs: dict[str, Any]) -> Response:
        result: Response = await aresponses(**kwargs)  # type: ignore[assignment]
        return result

    def new_usage_accumulator(self) -> dict[str, Any]:
        # ``searches`` collects the gateway-run searches so the final response can
        # announce them natively; ``compactions`` keeps replay state produced by
        # hidden iterations available to the caller. See ``fold_usage``.
        return {"input": 0, "output": 0, "total": 0, "searches": [], "compactions": []}

    def accumulate_usage(self, acc: dict[str, Any], result: Response) -> None:
        if result.usage:
            acc["input"] += result.usage.input_tokens or 0
            acc["output"] += result.usage.output_tokens or 0
            acc["total"] += result.usage.total_tokens or 0

    def usage_snapshot(self, result: Response) -> GatewayUsage | None:
        return responses_usage(getattr(result, "usage", None))

    def fold_usage(self, result: Response, acc: dict[str, Any]) -> None:
        _fold_usage(result, acc["input"], acc["output"], acc["total"])
        # Prepend a native ``web_search_call`` item per gateway-run search. The
        # loop consumed the raw ``function_call`` items, so without this the caller
        # has no way to know a search happened; they come first because they did.
        hidden_output = list(acc["compactions"]) + list(acc["searches"])
        if hidden_output:
            try:
                result.output = hidden_output + list(result.output or [])
            except (AttributeError, TypeError):
                logger.warning("Could not add hidden tool-loop items to the response output")

    def exit_before_split(self, result: Response) -> bool:
        return False

    def split_owned(self, result: Response, pool: ToolBackend) -> tuple[list[Any], bool]:
        return _split_function_calls(list(result.output or []), pool)

    def exit_after_split(self, result: Response) -> bool:
        return False

    async def execute_owned(
        self, pool: ToolBackend, owned: list[Any], acc: Any = None
    ) -> list[dict[str, Any]]:
        # ``acc`` is accepted for interface parity and unused: this format has no
        # native vocabulary for a server-side tool call to report on a mixed batch.
        return await _execute_function_calls(pool, owned)

    def filter_owned(self, result: Response, owned: list[Any], pool: ToolBackend) -> None:
        # Mixed batch: the owned subset was executed for its side effects;
        # filter it from the returned output so the caller only sees the
        # foreign function_call items it can dispatch itself.
        owned_call_ids = {item.call_id for item in owned}
        output = list(result.output or [])
        try:
            result.output = [
                item
                for item in output
                if not (
                    getattr(item, "type", None) == "function_call"
                    and getattr(item, "call_id", None) in owned_call_ids
                )
            ]
        except (AttributeError, TypeError):
            logger.warning(
                "Responses-mixed: could not filter output on response; client will see function_call "
                "items the gateway already executed (no-op on the client side).",
            )

    async def advance_transcript(
        self,
        transcript: list[Any],
        result: Response,
        owned: list[Any],
        pool: ToolBackend,
        acc: dict[str, Any] | None = None,
    ) -> None:
        # All-owned: continue. Replay compaction items and the assistant's
        # function calls in their original output order, then append matching
        # function_call_output items. Stateless compaction requires the opaque
        # compaction item on every continuation.
        output = list(result.output or [])
        transcript.extend(_items_to_dicts(_replay_items(output, owned)))
        transcript.extend(await _execute_function_calls(pool, owned))
        if acc is not None:
            acc["compactions"].extend(_compaction_items(output))
            acc["searches"].extend(_web_search_items_for(owned))

    # ---- streaming hooks ----

    async def open_stream(self, kwargs: dict[str, Any]) -> AsyncIterator[ResponseStreamEvent]:
        stream: AsyncIterator[ResponseStreamEvent] = await aresponses(**kwargs)  # type: ignore[assignment]
        return stream

    def new_stream_state(self) -> _ResponsesStreamState:
        return _ResponsesStreamState()

    def new_stream_accumulator(self) -> dict[str, Any]:
        # Each iteration's ``response.completed`` carries that iteration's
        # usage on ``event.response.usage``. When the loop continues, that
        # event is dropped and its usage would be lost from streaming token
        # reporting. Accumulate the per-iteration ``output_tokens`` and fold
        # the running total into the final forwarded ``response.completed``
        # so downstream usage logging sees the full tool-loop output count.
        #
        # started / next_sequence: the client is shown ONE response even when the
        # loop consumed several upstream ones, so only the first
        # ``response.created`` is forwarded and every forwarded event's
        # ``sequence_number`` is renumbered continuously. Without this a tool-loop
        # stream repeats ``response.created`` and restarts sequence numbers, which
        # the SDK's stream helper treats as a protocol error.
        return {
            "output_tokens": 0,
            "started": 0,
            "next_sequence": 0,
            "next_output_index": 0,
            "compactions": [],
        }

    def observe(
        self,
        state: _ResponsesStreamState,
        event: ResponseStreamEvent,
        pool: ToolBackend,
        acc: dict[str, Any],
    ) -> tuple[StreamAction, ResponseStreamEvent]:
        etype = getattr(event, "type", None)
        usage = getattr(getattr(event, "response", None), "usage", None)
        if usage is not None:
            state.usage = usage

        if etype == "response.created":
            if acc["started"]:
                return StreamAction.DEFER, event
            acc["started"] = 1
            return StreamAction.FORWARD, self._resequenced(event, acc)

        if etype in {"response.output_item.added", "response.output_item.done"}:
            item = getattr(event, "item", None)
            output_index = getattr(event, "output_index", None)
            if item is not None and isinstance(output_index, int) and getattr(item, "type", None) == "compaction":
                # The done event replaces the added snapshot when both arrive,
                # ensuring the replay uses the complete encrypted payload.
                state.compaction_items[output_index] = item
            if (
                etype == "response.output_item.added"
                and item is not None
                and output_index is not None
                and getattr(item, "type", None) == "function_call"
            ):
                name = getattr(item, "name", "")
                state.function_calls[output_index] = {
                    "call_id": getattr(item, "call_id", ""),
                    "name": name,
                    "arguments": getattr(item, "arguments", "") or "",
                }
                if pool.owns_tool(name):
                    # Recorded above so the loop can execute it; hidden from the
                    # client because it will never see the matching output item.
                    state.hidden_output_indices.add(output_index)
                    return StreamAction.DEFER, event

        elif etype == "response.function_call_arguments.delta":
            idx = getattr(event, "output_index", None)
            if idx is not None and idx in state.function_calls:
                state.function_calls[idx]["arguments"] += getattr(event, "delta", "") or ""

        elif etype == "response.function_call_arguments.done":
            # The terminal arguments value overrides the running buffer;
            # the SDK uses this event for completeness even when the
            # deltas already concatenate cleanly.
            idx = getattr(event, "output_index", None)
            if idx is not None and idx in state.function_calls:
                final_args = getattr(event, "arguments", None)
                if final_args:
                    state.function_calls[idx]["arguments"] = final_args
                final_name = getattr(event, "name", None)
                if final_name:
                    state.function_calls[idx]["name"] = final_name

        elif etype == "response.completed":
            # Defer: whether it is forwarded or dropped depends on the
            # tool-call accounting in ``stream_exiting``.
            state.deferred_completed = event
            return StreamAction.BREAK, event

        raw_index = getattr(event, "output_index", None)
        if raw_index in state.hidden_output_indices:
            return StreamAction.DEFER, event
        visible = event
        if isinstance(raw_index, int):
            visible = _reoutput_indexed(event, self._visible_output_for(state, acc, raw_index))
        return StreamAction.FORWARD, self._resequenced(visible, acc)

    @staticmethod
    def _visible_output_for(state: _ResponsesStreamState, acc: dict[str, int], raw_index: int) -> int:
        """The client-visible output index for an upstream one, assigned in order."""
        if raw_index not in state.visible_output_index:
            state.visible_output_index[raw_index] = acc["next_output_index"]
            acc["next_output_index"] += 1
        return state.visible_output_index[raw_index]

    @staticmethod
    def _resequenced(event: Any, acc: dict[str, int]) -> Any:
        """Stamp the next client-visible ``sequence_number`` on a forwarded event.

        No-op when the number already matches, so a single-iteration stream is
        byte-identical to the upstream one.
        """
        nxt = acc["next_sequence"]
        acc["next_sequence"] = nxt + 1
        if getattr(event, "sequence_number", None) == nxt or not hasattr(event, "model_copy"):
            return event
        return event.model_copy(update={"sequence_number": nxt})

    def stream_exiting(self, state: _ResponsesStreamState, pool: ToolBackend) -> bool:
        if not state.function_calls:
            return True
        owned_specs: list[dict[str, Any]] = []
        has_foreign = False
        for idx in sorted(state.function_calls):
            spec = state.function_calls[idx]
            if pool.owns_tool(spec["name"]):
                owned_specs.append(spec)
            else:
                has_foreign = True
        state.owned_specs = owned_specs
        # Mixed batches in streaming mode forward everything as-is (rewriting
        # the output_items mid-stream to remove owned ones would be too
        # invasive); the client sees what it sees and the loop exits. Same
        # trade-off as the Anthropic streaming variant.
        return has_foreign or not owned_specs

    async def finalize_exit(self, state: _ResponsesStreamState, pool: ToolBackend) -> None:
        # Mixed batch: the gateway's function_call items were withheld from the
        # stream, so run them for their side effects rather than dropping the model's
        # request. Matches the non-streaming loop's mixed-batch handling.
        if state.owned_specs:
            await _execute_stream_owned(state, pool)

    def terminal_events(self, state: _ResponsesStreamState, acc: dict[str, Any]) -> list[ResponseStreamEvent]:
        if state.deferred_completed is None:
            return []
        # The terminal event carries the whole response, whose ``output`` still lists
        # the gateway's own function_call items even though their item events were
        # hidden. Left in, ``get_final_response()`` would contradict the stream the
        # client just accumulated, and hand it a call it cannot dispatch.
        hidden = _hidden_call_ids(state)
        folded = _without_output_items(state.deferred_completed, hidden) if hidden else state.deferred_completed
        folded = _prepend_output_items(folded, acc["compactions"])
        folded = _maybe_fold_response_completed_usage(folded, acc["output_tokens"])
        # The terminal event is the last thing the client sees, so it continues the
        # same sequence as the events forwarded before it.
        return [self._resequenced(folded, acc)]

    def accumulate_stream_usage(self, acc: dict[str, Any], state: _ResponsesStreamState) -> None:
        # All-owned continuation: fold this iteration's output_tokens from the
        # dropped ``response.completed`` event into the running total.
        if state.deferred_completed is not None:
            iter_response = getattr(state.deferred_completed, "response", None)
            iter_usage = getattr(iter_response, "usage", None) if iter_response is not None else None
            if iter_usage is not None:
                acc["output_tokens"] += getattr(iter_usage, "output_tokens", 0) or 0
        acc["compactions"].extend(state.compaction_items[index] for index in sorted(state.compaction_items))

    def stream_usage_snapshot(self, state: _ResponsesStreamState) -> GatewayUsage | None:
        # One round's whole usage object rides on its terminal event, recorded by
        # ``observe`` whichever terminal it was. Matches what the route's
        # ``extract_stream_usage`` bills off the same stream.
        return responses_usage(state.usage)

    def synthetic_events(
        self, state: _ResponsesStreamState, acc: dict[str, Any]
    ) -> list[ResponseStreamEvent]:
        """Announce gateway-run searches in the Responses API's native vocabulary.

        The raw ``function_call`` events were swallowed (the client can never be
        sent their output), so a ``web_search_call`` item takes their place: it is
        what an OpenAI-hosted search would have emitted, and unlike the Anthropic
        equivalent it is expressible without forging provider-signed content.

        Only ``web_search`` is announced. A sandbox or MCP call has no native item
        that would be honest to emit, so it stays invisible on the wire.
        """
        events: list[ResponseStreamEvent] = []
        for spec in state.owned_specs:
            if spec.get("name") != WEB_SEARCH_TOOL_NAME:
                continue
            try:
                query = str(json.loads(spec.get("arguments") or "{}").get("query") or "")
            except json.JSONDecodeError:
                query = ""
            item = _web_search_call_item(spec.get("call_id") or "", query)
            output_index = acc["next_output_index"]
            acc["next_output_index"] += 1
            for event_cls, event_type in (
                (ResponseOutputItemAddedEvent, "response.output_item.added"),
                (ResponseOutputItemDoneEvent, "response.output_item.done"),
            ):
                sequence = acc["next_sequence"]
                acc["next_sequence"] = sequence + 1
                events.append(
                    event_cls(
                        type=event_type,  # type: ignore[arg-type]
                        item=item,
                        output_index=output_index,
                        sequence_number=sequence,
                    )
                )
        return events

    async def advance_stream_transcript(
        self,
        transcript: list[Any],
        state: _ResponsesStreamState,
        pool: ToolBackend,
    ) -> None:
        replay_items: list[Any] = []
        for output_index in sorted(set(state.compaction_items) | set(state.function_calls)):
            if output_index in state.compaction_items:
                replay_items.append(state.compaction_items[output_index])
            if output_index in state.function_calls:
                spec = state.function_calls[output_index]
                replay_items.append(
                    {
                        "type": "function_call",
                        "call_id": spec["call_id"],
                        "name": spec["name"],
                        "arguments": spec["arguments"] or "{}",
                    }
                )
        transcript.extend(_items_to_dicts(replay_items))
        transcript.extend(await _execute_stream_owned(state, pool))


_RESPONSES_STRATEGY = _ResponsesToolLoopStrategy()


async def responses_tool_loop(
    *,
    completion_kwargs: dict[str, Any],
    pool: ToolBackend,
    max_iterations: int,
    on_first_response: Callable[[], None] | None = None,
) -> Response:
    """Non-streaming OpenAI Responses tool-use loop.

    Each iteration calls ``aresponses``, walks ``result.output`` for owned
    ``function_call`` items, executes them, and appends the originals plus
    matching ``function_call_output`` items to ``input_data`` for the next
    round. Loop terminates when:

    - the response has no owned ``function_call`` items (final answer);
    - the response includes foreign ``function_call`` items, which are
      returned to the caller for client-side dispatch. Mixed batches execute
      the owned subset for its side effects and filter those out of the
      returned ``output`` so the caller only sees the foreign ones.

    Accumulates ``input_tokens`` / ``output_tokens`` / ``total_tokens`` across
    iterations into the returned ``Response``.

    ``on_first_response`` follows the provider lock-in contract documented on
    :func:`gateway.services._tool_loop.run_tool_loop`. Lock-in matters even
    more here: a Responses transcript carries provider-specific call_ids and
    reasoning items that can't be replayed against another provider.
    """
    return await run_tool_loop(
        strategy=_RESPONSES_STRATEGY,
        completion_kwargs=completion_kwargs,
        pool=pool,
        max_iterations=max_iterations,
        on_first_response=on_first_response,
    )


async def responses_tool_loop_stream(
    *,
    completion_kwargs: dict[str, Any],
    pool: ToolBackend,
    max_iterations: int,
) -> AsyncGenerator[ResponseStreamEvent, None]:
    """Streaming OpenAI Responses tool-use loop.

    Forwards every event downstream and tracks ``function_call`` output items
    by buffering ``response.function_call_arguments.delta`` chunks per
    ``output_index``. ``response.function_call_arguments.done``, if present,
    overrides the running buffer with the terminal argument string.

    Loop continuation is decided on ``response.completed``: if any buffered
    function_call items exist AND all are owned by the pool, execute them,
    append the originals plus the ``function_call_output`` items to
    ``input_data``, drop the terminal ``response.completed`` event, and start
    the next round. If any foreign function_call is present (or none at all),
    forward ``response.completed`` and exit.
    """
    # aclosing makes downstream closes (client disconnect) propagate to the
    # engine generator, and through it to the upstream provider stream,
    # instead of waiting for event-loop async-generator finalization.
    async with aclosing(
        run_tool_loop_stream(
            strategy=_RESPONSES_STRATEGY,
            completion_kwargs=completion_kwargs,
            pool=pool,
            max_iterations=max_iterations,
        )
    ) as inner:
        async for event in inner:
            yield event
