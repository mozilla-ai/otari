"""Unit tests for the first-chunk timeout policy in ``iterate_streaming_attempts``.

The first-chunk timeout is a failover trigger: it abandons a slow attempt so the
next one in the prioritized fallback plan can be tried. Non-final attempts get the
tight failover budget; the sole/final attempt has nothing to fall over to, so it
gets ``final_attempt_extra_seconds`` of grace on top of the budget. That grace lets
a slow-but-valid first token stream through, while keeping the wait bounded, so a
genuinely hung upstream still times out at ``budget + grace``.
"""

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from gateway.streaming import StreamingAttemptFailure, iterate_streaming_attempts

# Small, well-separated durations keep the tests fast while staying robust on a
# loaded CI box. Failover budget is tight; the terminal grace extends it.
_BUDGET_SECONDS = 0.05
_GRACE_SECONDS = 0.25  # terminal budget = 0.30
_WITHIN_GRACE_DELAY = 0.15  # > budget, < budget + grace
_BEYOND_GRACE_DELAY = 0.45  # > budget + grace


class _RetryableError(Exception):
    """A build-time upstream failure the classifier treats as retryable."""


async def _stream(delay: float, chunks: tuple[str, ...]) -> AsyncIterator[str]:
    """Async stream whose first ``__anext__`` blocks for ``delay`` seconds."""
    import asyncio

    if delay:
        await asyncio.sleep(delay)
    for chunk in chunks:
        yield chunk


def _attempt(
    name: str,
    *,
    delay: float = 0.0,
    chunks: tuple[str, ...] = ("chunk",),
    build_error: Exception | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(name=name, delay=delay, chunks=chunks, build_error=build_error)


def _harness() -> tuple[Any, Any, Any, list[tuple[str, str]]]:
    """Build the ``build_stream`` / ``classify_error`` / ``on_attempt_failed``
    callbacks plus the list that records abandoned attempts."""
    failures: list[tuple[str, str]] = []

    async def build_stream(attempt: SimpleNamespace) -> AsyncIterator[str]:
        if attempt.build_error is not None:
            raise attempt.build_error
        return _stream(attempt.delay, attempt.chunks)

    def classify_error(exc: BaseException) -> tuple[bool, str]:
        return True, type(exc).__name__

    async def on_attempt_failed(attempt: SimpleNamespace, failure: StreamingAttemptFailure) -> None:
        failures.append((attempt.name, failure.error_class))

    return build_stream, classify_error, on_attempt_failed, failures


async def _drain(stream: AsyncIterator[str]) -> list[str]:
    return [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_sole_attempt_slow_first_chunk_within_grace_streams() -> None:
    """A single-attempt request whose first chunk exceeds the failover budget but
    lands within the terminal grace must stream, not 504."""
    build_stream, classify_error, on_attempt_failed, failures = _harness()
    attempts = [_attempt("only", delay=_WITHIN_GRACE_DELAY, chunks=("hi", "there"))]

    chosen, stream = await iterate_streaming_attempts(
        attempts=attempts,
        build_stream=build_stream,
        classify_error=classify_error,
        on_attempt_failed=on_attempt_failed,
        first_chunk_timeout_seconds=_BUDGET_SECONDS,
        final_attempt_extra_seconds=_GRACE_SECONDS,
    )

    assert chosen.name == "only"
    assert await _drain(stream) == ["hi", "there"]
    assert failures == []  # not abandoned: within budget + grace


@pytest.mark.asyncio
async def test_sole_attempt_beyond_grace_still_times_out() -> None:
    """The terminal grace is bounded: a first chunk slower than budget + grace
    still times out, so a genuinely hung upstream cannot hang forever."""
    build_stream, classify_error, on_attempt_failed, failures = _harness()
    attempts = [_attempt("only", delay=_BEYOND_GRACE_DELAY)]

    with pytest.raises(TimeoutError):
        await iterate_streaming_attempts(
            attempts=attempts,
            build_stream=build_stream,
            classify_error=classify_error,
            on_attempt_failed=on_attempt_failed,
            first_chunk_timeout_seconds=_BUDGET_SECONDS,
            final_attempt_extra_seconds=_GRACE_SECONDS,
        )

    assert failures == [("only", "timeout")]


@pytest.mark.asyncio
async def test_non_final_slow_attempt_fails_over_at_the_budget() -> None:
    """A non-final attempt is capped at the failover budget (grace does not apply)
    and fails over even for a delay the terminal grace would have tolerated."""
    build_stream, classify_error, on_attempt_failed, failures = _harness()
    attempts = [
        _attempt("primary", delay=_WITHIN_GRACE_DELAY, chunks=("primary-chunk",)),
        _attempt("fallback", delay=0.0, chunks=("fallback-chunk",)),
    ]

    chosen, stream = await iterate_streaming_attempts(
        attempts=attempts,
        build_stream=build_stream,
        classify_error=classify_error,
        on_attempt_failed=on_attempt_failed,
        first_chunk_timeout_seconds=_BUDGET_SECONDS,
        final_attempt_extra_seconds=_GRACE_SECONDS,
    )

    assert chosen.name == "fallback"
    assert await _drain(stream) == ["fallback-chunk"]
    assert failures == [("primary", "timeout")]


@pytest.mark.asyncio
async def test_final_attempt_of_chain_gets_the_grace() -> None:
    """The last attempt in a chain has no successor, so it gets the terminal grace
    even though an earlier attempt failed over into it."""
    build_stream, classify_error, on_attempt_failed, failures = _harness()
    attempts = [
        _attempt("primary", build_error=_RetryableError("primary down")),
        _attempt("fallback", delay=_WITHIN_GRACE_DELAY, chunks=("fallback-chunk",)),
    ]

    chosen, stream = await iterate_streaming_attempts(
        attempts=attempts,
        build_stream=build_stream,
        classify_error=classify_error,
        on_attempt_failed=on_attempt_failed,
        first_chunk_timeout_seconds=_BUDGET_SECONDS,
        final_attempt_extra_seconds=_GRACE_SECONDS,
    )

    assert chosen.name == "fallback"
    assert await _drain(stream) == ["fallback-chunk"]
    assert failures == [("primary", "_RetryableError")]


async def _raising_stream(exc: Exception) -> AsyncIterator[str]:
    """Async stream that raises on its first ``__anext__`` (upstream error before
    the first chunk), rather than timing out or failing at build time."""
    raise exc
    yield  # unreachable; makes this a generator


@pytest.mark.asyncio
async def test_failure_reason_buckets_by_abandonment_phase() -> None:
    """Each abandonment site tags the failure with its coarse ``reason`` bucket:
    ``build_error`` when opening the stream fails, ``timeout`` when the first-chunk
    wait elapses, and ``upstream_error`` when the stream raises before yielding."""
    reasons: list[str] = []

    def classify_error(exc: BaseException) -> tuple[bool, str]:
        return True, type(exc).__name__

    async def on_attempt_failed(attempt: SimpleNamespace, failure: StreamingAttemptFailure) -> None:
        reasons.append(failure.reason)

    async def build_stream_ok_but_raises(_attempt: SimpleNamespace) -> AsyncIterator[str]:
        return _raising_stream(_RetryableError("upstream blew up before first chunk"))

    async def build_stream_that_fails(_attempt: SimpleNamespace) -> AsyncIterator[str]:
        raise _RetryableError("could not open stream")

    # build_error: build_stream raises before any stream exists.
    with pytest.raises(_RetryableError):
        await iterate_streaming_attempts(
            attempts=[_attempt("only")],
            build_stream=build_stream_that_fails,
            classify_error=classify_error,
            on_attempt_failed=on_attempt_failed,
            first_chunk_timeout_seconds=_BUDGET_SECONDS,
        )

    # timeout: the stream opens but the first chunk never arrives in time.
    build_stream, _classify, _on_failed, _failures = _harness()
    with pytest.raises(TimeoutError):
        await iterate_streaming_attempts(
            attempts=[_attempt("only", delay=_BEYOND_GRACE_DELAY)],
            build_stream=build_stream,
            classify_error=classify_error,
            on_attempt_failed=on_attempt_failed,
            first_chunk_timeout_seconds=_BUDGET_SECONDS,
        )

    # upstream_error: the stream opens but raises before yielding a first chunk.
    with pytest.raises(_RetryableError):
        await iterate_streaming_attempts(
            attempts=[_attempt("only")],
            build_stream=build_stream_ok_but_raises,
            classify_error=classify_error,
            on_attempt_failed=on_attempt_failed,
            first_chunk_timeout_seconds=_BUDGET_SECONDS,
        )

    assert reasons == ["build_error", "timeout", "upstream_error"]


@pytest.mark.asyncio
async def test_default_extra_is_zero_preserves_uniform_cap() -> None:
    """With no grace passed, every attempt (including the sole one) is capped at
    the failover budget, the historical uniform-cap behavior."""
    build_stream, classify_error, on_attempt_failed, failures = _harness()
    attempts = [_attempt("only", delay=_WITHIN_GRACE_DELAY)]

    with pytest.raises(TimeoutError):
        await iterate_streaming_attempts(
            attempts=attempts,
            build_stream=build_stream,
            classify_error=classify_error,
            on_attempt_failed=on_attempt_failed,
            first_chunk_timeout_seconds=_BUDGET_SECONDS,
        )

    assert failures == [("only", "timeout")]
