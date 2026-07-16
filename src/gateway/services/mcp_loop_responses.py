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

from gateway.log_config import logger
from gateway.services._tool_loop import StreamAction, run_tool_loop, run_tool_loop_stream
from gateway.services.mcp_loop import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    MAX_TOOL_ITERATIONS_CAP,
    MaxToolIterationsExceeded,
    ToolBackend,
)
from gateway.services.tool_format import openai_to_responses_tools

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


class _ResponsesStreamState:
    """Per-iteration bookkeeping for the Responses streaming loop."""

    def __init__(self) -> None:
        # output_index -> {"call_id", "name", "arguments"}
        self.function_calls: dict[int, dict[str, Any]] = {}
        self.deferred_completed: ResponseStreamEvent | None = None
        self.owned_specs: list[dict[str, Any]] = []


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

    # ---- non-streaming hooks ----

    async def call(self, kwargs: dict[str, Any]) -> Response:
        result: Response = await aresponses(**kwargs)  # type: ignore[assignment]
        return result

    def new_usage_accumulator(self) -> dict[str, int]:
        return {"input": 0, "output": 0, "total": 0}

    def accumulate_usage(self, acc: dict[str, int], result: Response) -> None:
        if result.usage:
            acc["input"] += result.usage.input_tokens or 0
            acc["output"] += result.usage.output_tokens or 0
            acc["total"] += result.usage.total_tokens or 0

    def fold_usage(self, result: Response, acc: dict[str, int]) -> None:
        _fold_usage(result, acc["input"], acc["output"], acc["total"])

    def exit_before_split(self, result: Response) -> bool:
        return False

    def split_owned(self, result: Response, pool: ToolBackend) -> tuple[list[Any], bool]:
        return _split_function_calls(list(result.output or []), pool)

    def exit_after_split(self, result: Response) -> bool:
        return False

    async def execute_owned(self, pool: ToolBackend, owned: list[Any]) -> list[dict[str, Any]]:
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
    ) -> None:
        # All-owned: continue. Append the assistant's function_call items AND
        # the matching function_call_output items so the next call's input has
        # the full transcript.
        transcript.extend(_items_to_dicts(owned))
        transcript.extend(await _execute_function_calls(pool, owned))

    # ---- streaming hooks ----

    async def open_stream(self, kwargs: dict[str, Any]) -> AsyncIterator[ResponseStreamEvent]:
        stream: AsyncIterator[ResponseStreamEvent] = await aresponses(**kwargs)  # type: ignore[assignment]
        return stream

    def new_stream_state(self) -> _ResponsesStreamState:
        return _ResponsesStreamState()

    def new_stream_accumulator(self) -> dict[str, int]:
        # Each iteration's ``response.completed`` carries that iteration's
        # usage on ``event.response.usage``. When the loop continues, that
        # event is dropped and its usage would be lost from streaming token
        # reporting. Accumulate the per-iteration ``output_tokens`` and fold
        # the running total into the final forwarded ``response.completed``
        # so downstream usage logging sees the full tool-loop output count.
        return {"output_tokens": 0}

    def observe(self, state: _ResponsesStreamState, event: ResponseStreamEvent) -> StreamAction:
        etype = getattr(event, "type", None)

        if etype == "response.output_item.added":
            item = getattr(event, "item", None)
            output_index = getattr(event, "output_index", None)
            if item is not None and output_index is not None and getattr(item, "type", None) == "function_call":
                state.function_calls[output_index] = {
                    "call_id": getattr(item, "call_id", ""),
                    "name": getattr(item, "name", ""),
                    "arguments": getattr(item, "arguments", "") or "",
                }

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
            return StreamAction.BREAK

        return StreamAction.FORWARD

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

    def terminal_events(self, state: _ResponsesStreamState, acc: dict[str, int]) -> list[ResponseStreamEvent]:
        if state.deferred_completed is None:
            return []
        return [_maybe_fold_response_completed_usage(state.deferred_completed, acc["output_tokens"])]

    def accumulate_stream_usage(self, acc: dict[str, int], state: _ResponsesStreamState) -> None:
        # All-owned continuation: fold this iteration's output_tokens from the
        # dropped ``response.completed`` event into the running total.
        if state.deferred_completed is not None:
            iter_response = getattr(state.deferred_completed, "response", None)
            iter_usage = getattr(iter_response, "usage", None) if iter_response is not None else None
            if iter_usage is not None:
                acc["output_tokens"] += getattr(iter_usage, "output_tokens", 0) or 0

    async def advance_stream_transcript(
        self,
        transcript: list[Any],
        state: _ResponsesStreamState,
        pool: ToolBackend,
    ) -> None:
        transcript.extend(
            {
                "type": "function_call",
                "call_id": spec["call_id"],
                "name": spec["name"],
                "arguments": spec["arguments"] or "{}",
            }
            for spec in state.owned_specs
        )

        for spec in state.owned_specs:
            try:
                parsed_args = json.loads(spec["arguments"] or "{}")
            except json.JSONDecodeError:
                parsed_args = {}
            try:
                text = await pool.call_tool(spec["name"], parsed_args)
            except Exception as exc:  # noqa: BLE001 — same tool-error-as-message idiom as the non-stream loop
                logger.warning("MCP tool %s execution failed: %s", spec["name"], exc)
                text = f"[tool error] {exc}"
            transcript.append({"type": "function_call_output", "call_id": spec["call_id"], "output": text})


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
