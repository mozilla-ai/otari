"""Shared SSE streaming utilities for gateway routes."""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from contextlib import aclosing, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import anyio
from any_llm.types.completion import CompletionUsage

from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.usage import (
    GatewayUsage,
    cache_read_tokens_of,
    cache_tokens_in_prompt_of,
    cache_write_1h_tokens_of,
    cache_write_tokens_of,
    provider_web_searches_of,
    reasoning_tokens_of,
)
from gateway.log_config import logger
from gateway.model_labeling import relabel_model

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig

C = TypeVar("C")  # Chunk type emitted by the upstream stream
S = TypeVar("S")  # Settlement type, opaque to this module
T = TypeVar("T")


# Most chunks held behind a designated cost carrier, the carrier included. The
# real terminal suffix is short (a usage chunk, or ``message_delta`` plus
# ``message_stop``); anything longer means the carrier was not terminal, and the
# stream resumes rather than waiting on settlement to release the rest.
_MAX_TERMINAL_BUFFER_CHUNKS = 4


class IncompleteStreamError(Exception):
    """The upstream ended a stream without the event that completes it, and raised nothing.

    Carries counts only, never content, so its text is safe to log and to store
    on a usage row.
    """

    def __init__(self, chunks_received: int) -> None:
        self.chunks_received = chunks_received
        super().__init__(f"upstream stream ended without completing ({chunks_received} chunks received)")


@dataclass(frozen=True)
class StreamFormat:
    """SSE formatting configuration for a streaming protocol."""

    done_marker: str
    error_payload: str
    yield_done_on_error: bool
    keepalive: str


_OPENAI_ERROR = json.dumps({"error": {"message": "An error occurred during streaming", "type": "server_error"}})
_ANTHROPIC_ERROR = json.dumps(
    {"type": "error", "error": {"type": "api_error", "message": "An error occurred during streaming"}}
)


def openai_error_event(
    fmt: StreamFormat,
    code: str | None,
    message: str | None = None,
    provider_failure: Mapping[str, object] | None = None,
) -> str:
    """``fmt``'s error event, carrying ``code`` and ``message`` in the OpenAI error object where given.

    ``message`` must already be safe to show the caller; without one the event says only that streaming failed.
    ``provider_failure`` adds the fields a provider's own failure may show (``provider``, ``status``, ``tag``,
    ``retryable``), under ``error.provider_failure``.
    """
    if code is None and message is None and provider_failure is None:
        return fmt.error_payload
    error: dict[str, object] = {"message": message or "An error occurred during streaming", "type": "server_error"}
    if code is not None:
        error["code"] = code
    if provider_failure is not None:
        error["provider_failure"] = dict(provider_failure)
    return fmt.error_payload.replace(_OPENAI_ERROR, json.dumps({"error": error}))


# An SSE comment line: conformant parsers (including the OpenAI SDKs) drop it, so
# it keeps the socket warm without ever surfacing as content.
_SSE_COMMENT_KEEPALIVE = ": keepalive\n\n"

OPENAI_STREAM_FORMAT = StreamFormat(
    done_marker="data: [DONE]\n\n",
    error_payload=f"data: {_OPENAI_ERROR}\n\n",
    yield_done_on_error=True,
    keepalive=_SSE_COMMENT_KEEPALIVE,
)

RESPONSES_STREAM_FORMAT = StreamFormat(
    done_marker="data: [DONE]\n\n",
    error_payload=f"event: error\ndata: {_OPENAI_ERROR}\n\n",
    yield_done_on_error=False,
    keepalive=_SSE_COMMENT_KEEPALIVE,
)

# The events a Responses stream ends a response with.
RESPONSES_TERMINAL_EVENTS = frozenset({"response.completed", "response.incomplete", "response.failed"})

ANTHROPIC_STREAM_FORMAT = StreamFormat(
    done_marker="event: done\ndata: {}\n\n",
    error_payload=f"event: error\ndata: {_ANTHROPIC_ERROR}\n\n",
    yield_done_on_error=False,
    # ``ping`` is part of the Messages stream vocabulary, so existing Anthropic
    # clients already tolerate it; a bare SSE comment is not in that vocabulary.
    keepalive=f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n",
)


def merge_stream_usage(current: CompletionUsage, update: CompletionUsage) -> CompletionUsage:
    """Merge usage data, keeping the last non-zero value for each field."""
    # Provider extras (e.g. Ollama's timing, read by provider_latency_ms_of) ride
    # along; the update's win, as they arrive on the final chunk.
    extras = GatewayUsage.external_extras(current) | GatewayUsage.external_extras(update)
    return GatewayUsage(
        **extras,
        prompt_tokens=update.prompt_tokens or current.prompt_tokens,
        completion_tokens=update.completion_tokens or current.completion_tokens,
        total_tokens=update.total_tokens or current.total_tokens,
        cache_read_tokens=cache_read_tokens_of(update) or cache_read_tokens_of(current),
        cache_write_tokens=cache_write_tokens_of(update) or cache_write_tokens_of(current),
        cache_write_1h_tokens=cache_write_1h_tokens_of(update) or cache_write_1h_tokens_of(current),
        # All chunks in one stream share a provider convention. Keep it separate
        # (Anthropic) if any chunk reported it that way, so the seed's default of
        # "in prompt" never masks the Anthropic shape.
        cache_tokens_in_prompt=cache_tokens_in_prompt_of(update) and cache_tokens_in_prompt_of(current),
        reasoning_tokens=reasoning_tokens_of(update) or reasoning_tokens_of(current),
        provider_web_searches=provider_web_searches_of(update) or provider_web_searches_of(current),
    )


def _describe_settlement_error(exc: Exception) -> str:
    """Name a settlement failure without the row data a database error's message can carry."""
    if isinstance(exc, DATABASE_ERRORS):
        return describe_database_error(exc)
    return type(exc).__name__


_KEEPALIVE_DUE = object()
"""Sentinel yielded by ``_chunks_or_keepalive`` when the upstream went idle."""


async def _chunks_or_keepalive(stream: AsyncIterator[Any], interval_seconds: float) -> AsyncGenerator[Any, None]:
    """Yield upstream chunks, emitting ``_KEEPALIVE_DUE`` once per idle interval.

    The pending ``__anext__`` is held as a task across timeouts rather than being
    awaited under ``wait_for``: cancelling that await would discard a chunk that
    arrives moments after the interval elapses. A non-positive interval disables
    the idle tracking and iterates the upstream directly.
    """
    if interval_seconds <= 0:
        async for chunk in stream:
            yield chunk
        return

    iterator = stream.__aiter__()
    pending: asyncio.Task[Any] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())
            done, _ = await asyncio.wait((pending,), timeout=interval_seconds)
            if not done:
                yield _KEEPALIVE_DUE
                continue
            finished, pending = pending, None
            try:
                chunk = finished.result()
            except StopAsyncIteration:
                return
            yield chunk
    finally:
        if pending is not None:
            pending.cancel()
            # Unshielded, a canceled scope would cancel ``pending`` again on every
            # loop pass and cut short the cleanup its one cancellation starts.
            with anyio.CancelScope(shield=True), suppress(BaseException):
                await pending


async def streaming_generator(
    stream: AsyncIterator[Any],
    format_chunk: Callable[[Any], str],
    extract_usage: Callable[[Any], CompletionUsage | None],
    fmt: StreamFormat,
    on_complete: Callable[[CompletionUsage], Awaitable[S | None]],
    on_error: Callable[[BaseException], Awaitable[None]],
    label: str,
    on_no_usage: Callable[[], Awaitable[None]] | None = None,
    on_incomplete: Callable[[], Awaitable[None]] | None = None,
    display_model: str | None = None,
    keepalive_interval_seconds: float = 0.0,
    is_cost_carrier: Callable[[Any], bool] | None = None,
    attach_settlement: Callable[[Any, S], bool] | None = None,
    on_first_chunk: Callable[[], None] | None = None,
    error_payload: Callable[[BaseException], str] | None = None,
    completes_stream: Callable[[Any], bool] | None = None,
) -> AsyncGenerator[str, None]:
    """Shared SSE streaming generator with usage tracking and error handling.

    Args:
        stream: Async iterator of chunks from the provider
        format_chunk: Formats a chunk into an SSE string
        extract_usage: Extracts usage from a chunk, or returns None if no usage present
        fmt: SSE format configuration (done marker, error payload, etc.)
        display_model: When set (a configured alias was used), each chunk's
            ``model`` field is relabeled to this before formatting, so the
            underlying provider/model never appears on the wire.
        on_complete: Called with aggregated usage after successful streaming that
            included usage data
        on_error: Called with the raised exception on failure. The exception
            itself, not a message, so the caller can classify it (e.g. record the
            upstream HTTP status on the usage log) as well as render it.
        label: Identifier for error log messages (e.g., "openai:gpt-4")
        on_no_usage: Called when streaming completes successfully but the provider
            sent no usage data. Lets the caller bill per ``stream_missing_usage_policy``
            instead of silently skipping the request. When omitted, a no-usage
            stream is not billed (legacy behavior).
        on_incomplete: Called when the stream neither completes nor errors normally
            (e.g. client disconnect mid-stream). Used to release any budget
            reservation so it does not leak.
        keepalive_interval_seconds: Emit ``fmt.keepalive`` whenever the upstream has
            produced nothing for this long, including while terminal settlement is
            pending. 0 disables.
        is_cost_carrier: Identifies a provider object that can carry cost. It is
            held, with what follows it, until usage settles. The last match in
            the stream wins, so a predicate that also matches a non-terminal
            object costs no buffering beyond the chunks between that match and
            the next one.
        attach_settlement: Mutates that carrier with the opaque settlement value.
        on_first_chunk: Called synchronously, at most once, the moment the first
            non-keepalive chunk is about to be formatted and yielded. Lets the
            caller record time-to-first-token without this generator knowing
            anything about how that timing gets used or persisted.
        error_payload: Renders the SSE error event for the exception that ended
            the stream, so a dialect can say what kind of failure it was. When
            omitted, ``fmt.error_payload`` is sent for every failure.
        completes_stream: Says, after each chunk, whether the stream so far is
            complete; its answer after the last chunk decides. When set, an
            upstream that ends incomplete is failed with
            :class:`IncompleteStreamError` through the error path, so the client
            gets an error event rather than a 200 that silently carried nothing.

    """
    usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    has_usage = False
    settled = False
    seen_first_chunk = False
    terminal_buffer: list[Any] = []
    cost_carrier: Any | None = None
    buffering_terminal = False
    overflow_logged = False
    settlement_task: asyncio.Task[S | None] | None = None
    chunks_received = 0
    completed = completes_stream is None

    keepalive_interval = keepalive_interval_seconds if fmt.keepalive else 0.0

    def _format(chunk: Any) -> str:
        if display_model is not None:
            chunk = relabel_model(chunk, display_model)
        return format_chunk(chunk)

    def _format_and_mark_first(chunk: Any) -> str:
        # Format before marking: on_first_chunk must fire only once a chunk
        # has actually been turned into something we are about to yield, not
        # when it was merely pulled off the upstream. This also covers a
        # chunk that reaches the client via the terminal buffer rather than
        # the immediate yield below.
        nonlocal seen_first_chunk
        formatted = _format(chunk)
        if not seen_first_chunk:
            seen_first_chunk = True
            if on_first_chunk is not None:
                on_first_chunk()
        return formatted

    try:
        async with aclosing(_chunks_or_keepalive(stream, keepalive_interval)) as source:
            async for chunk in source:
                if chunk is _KEEPALIVE_DUE:
                    yield fmt.keepalive
                    continue
                chunks_received += 1
                if completes_stream is not None:
                    completed = completes_stream(chunk)
                chunk_usage = extract_usage(chunk)
                if chunk_usage:
                    usage = merge_stream_usage(usage, chunk_usage)
                    has_usage = True

                if is_cost_carrier is not None and is_cost_carrier(chunk):
                    # A later carrier supersedes an earlier one: flushing here keeps
                    # the stream flowing instead of holding it behind a carrier that
                    # is not the last, and leaves cost on the chunk that really ends
                    # the stream.
                    for buffered_chunk in terminal_buffer:
                        yield _format_and_mark_first(buffered_chunk)
                    terminal_buffer.clear()
                    cost_carrier = chunk
                    terminal_buffer.append(chunk)
                    buffering_terminal = True
                    continue

                if buffering_terminal:
                    if len(terminal_buffer) >= _MAX_TERMINAL_BUFFER_CHUNKS:
                        # The designated carrier was not terminal after all. Stop
                        # holding it so the stream keeps flowing, but stay open to
                        # a later carrier rather than giving up on cost outright.
                        if not overflow_logged:
                            logger.debug(
                                "Terminal usage carrier was not terminal for %s; streaming without holding it",
                                # codeql[py/clear-text-logging-sensitive-data]
                                label,
                            )
                            overflow_logged = True
                        for buffered_chunk in terminal_buffer:
                            yield _format_and_mark_first(buffered_chunk)
                        terminal_buffer.clear()
                        cost_carrier = None
                        buffering_terminal = False
                        yield _format_and_mark_first(chunk)
                        continue
                    terminal_buffer.append(chunk)
                    continue

                yield _format_and_mark_first(chunk)

        if not completed:
            raise IncompleteStreamError(chunks_received)

        # Once the upstream is exhausted, usage settles before the buffered terminal
        # suffix and the done marker. A settlement a client disconnect cut off
        # would never run again, as the stream already counts as settled, so it
        # is shielded from the response's canceled scope.
        settled = True
        settlement: S | None = None
        try:
            if has_usage:
                if keepalive_interval > 0:

                    async def _settle() -> S | None:
                        return await on_complete(usage)

                    settlement_task = asyncio.create_task(_settle())
                    while not settlement_task.done():
                        done, _ = await asyncio.wait(
                            (settlement_task,),
                            timeout=keepalive_interval,
                        )
                        if not done:
                            yield fmt.keepalive
                    settlement = settlement_task.result()
                else:
                    settlement = await _shielded(on_complete, usage)
            elif on_no_usage is not None:
                await _shielded(on_no_usage)
        except Exception as log_err:
            # codeql[py/clear-text-logging-sensitive-data]
            logger.error("Failed to log streaming usage for %s: %s", label, _describe_settlement_error(log_err))
        if settlement is not None and attach_settlement is not None:
            try:
                attach_settlement(cost_carrier, settlement)
            except Exception as attach_err:
                # codeql[py/clear-text-logging-sensitive-data]
                logger.error("Failed to attach streaming settlement for %s: %s", label, attach_err)
        for buffered_chunk in terminal_buffer:
            yield _format_and_mark_first(buffered_chunk)
        terminal_buffer.clear()
        yield fmt.done_marker
    except Exception as e:
        # A carrier may have been buffered just before the upstream failed. It
        # remains an ordinary provider event and must precede the safe error.
        for buffered_chunk in terminal_buffer:
            yield _format_and_mark_first(buffered_chunk)
        terminal_buffer.clear()
        yield error_payload(e) if error_payload is not None else fmt.error_payload
        if fmt.yield_done_on_error:
            yield fmt.done_marker
        settled = True
        try:
            await _shielded(on_error, e)
        except Exception as log_err:
            logger.error("Failed to log streaming error usage: %s", _describe_settlement_error(log_err))
        # codeql[py/clear-text-logging-sensitive-data]
        logger.error("Streaming error for %s: %s", label, e)
    finally:
        # A client disconnect cancels the response's anyio scope, which then
        # cancels every await made inside it, these included.
        with anyio.CancelScope(shield=True):
            if settlement_task is not None and not settlement_task.done():
                try:
                    await settlement_task
                except Exception as log_err:
                    # codeql[py/clear-text-logging-sensitive-data]
                    logger.error("Failed to log streaming usage for %s: %s", label, _describe_settlement_error(log_err))
            if not settled and on_incomplete is not None:
                try:
                    await on_incomplete()
                except Exception as log_err:
                    logger.error(
                        "Failed to settle incomplete stream for %s: %s",
                        # codeql[py/clear-text-logging-sensitive-data]
                        label,
                        _describe_settlement_error(log_err),
                    )
            # Iterating the source never closes it, so a stream the client abandoned
            # would hold the provider's connection until garbage collection.
            await close_stream_quietly(stream)


async def hold_for_first_chunk(stream: AsyncIterator[C], config: "GatewayConfig") -> AsyncIterator[C]:
    """``stream``, held for its first chunk as ``streaming_first_chunk_hold_ms`` says.

    Holding lets a provider failure before any output raise before the caller
    sends a status line, so it is answered with the status the same request gets
    unstreamed, and a routing policy can still fall over. A hold of 0 returns
    ``stream`` unread, so the response is committed as soon as it opens.
    """
    hold_ms = config.streaming_first_chunk_hold_ms
    if hold_ms <= 0:
        return stream
    return await await_first_chunk(stream, hold_ms / 1000)


async def await_first_chunk(stream: AsyncIterator[C], window_seconds: float) -> AsyncIterator[C]:
    """Wait up to ``window_seconds`` for ``stream``'s first chunk, so a failure before it raises here.

    A provider stream can fail before producing anything: an SDK that refuses a
    keyword only once the stream is iterated, or an upstream that answers the
    request with an error status. Raised here, before the caller has sent a
    status line, that failure can be answered with its own status instead of a
    ``200`` whose body is one error event.

    Returns an iterator that re-emits the first chunk and then the rest of the
    stream. When the window elapses first, the returned iterator resumes the
    same pending read, so a chunk arriving just after the window is not lost and
    a failure after it reaches the caller of the iterator, as it would have
    without this wait. A non-positive window waits for the first chunk however
    long it takes. Nothing here cuts a slow stream off.
    """
    first = asyncio.ensure_future(stream.__anext__())
    # A read left pending is never awaited if the iterator is dropped unread;
    # retrieve its outcome so a late failure is not reported as unhandled.
    first.add_done_callback(lambda done: done.cancelled() or done.exception())
    try:
        await asyncio.wait((first,), timeout=window_seconds if window_seconds > 0 else None)
    except BaseException:
        first.cancel()
        with suppress(BaseException):
            await first
        await close_stream_quietly(stream)
        raise
    if first.done():
        try:
            first.result()
        except StopAsyncIteration:
            return _empty_async_iter()
        except BaseException:
            await close_stream_quietly(stream)
            raise
    return _Prefetched(first, stream)


class _Prefetched(AsyncIterator[C]):
    """A first read, finished or still pending, followed by the rest of the stream.

    A class rather than a generator so ``aclose`` reaches the source stream and
    any pending read even when iteration never started: closing an unstarted
    generator runs none of its body, which would leave the provider's
    connection open until garbage collection.
    """

    def __init__(self, first: asyncio.Future[C], remaining: AsyncIterator[C]) -> None:
        self._first: asyncio.Future[C] | None = first
        self._remaining = remaining

    def __aiter__(self) -> AsyncIterator[C]:
        return self

    async def __anext__(self) -> C:
        first, self._first = self._first, None
        if first is None:
            return await self._remaining.__anext__()
        try:
            return await first
        finally:
            if not first.done():
                first.cancel()

    async def aclose(self) -> None:
        first, self._first = self._first, None
        if first is not None and not first.done():
            first.cancel()
            with suppress(BaseException):
                await first
        await close_stream_quietly(self._remaining)


async def _empty_async_iter() -> AsyncIterator[C]:
    return
    yield  # unreachable; makes this a generator


async def _shielded(settle: Callable[..., Awaitable[T]], *args: Any) -> T:
    """Run a settlement callback to the end even when the client has disconnected.

    A disconnect cancels the response's task scope, and anyio then cancels every
    await left inside it, so a settlement's database writes would stop with the
    stream and the request would go unbilled. A client that hangs up the
    moment it reads the last event, as Claude Code's ``WebSearch`` does, lands
    there on every request.
    """
    with anyio.CancelScope(shield=True):
        result = await settle(*args)
    return result


async def close_stream_quietly(stream: AsyncIterator[Any]) -> None:
    """Close ``stream`` if it can be closed, even while a client disconnect is canceling the caller."""
    close = getattr(stream, "aclose", None)
    if callable(close):
        with anyio.CancelScope(shield=True), suppress(BaseException):
            await close()
