"""The context window a read adds to each model request's span."""

from gateway.schemas.traces import _span_public
from gateway.services.traces._reading import _context_windows
from gateway.types.traces import SpanRecord


def _llm(span_id: str, model: str | None, provider: str | None = "anthropic") -> SpanRecord:
    return SpanRecord(
        span_id=span_id, kind="llm", origin="gateway", name="chat", outcome="ok", model=model, provider=provider
    )


def test_the_window_is_looked_up_once_per_model_and_is_absent_for_an_unknown_one() -> None:
    lookups: list[tuple[str | None, str]] = []

    def window_of(provider: str | None, model: str) -> int | None:
        lookups.append((provider, model))
        return {"claude-sonnet-4-5": 200_000}.get(model)

    tool = SpanRecord(span_id="t", kind="tool", origin="gateway", name="x", outcome="ok")
    windows = _context_windows(
        [_llm("a", "claude-sonnet-4-5"), _llm("b", "claude-sonnet-4-5"), _llm("c", "mystery"), _llm("d", None), tool],
        window_of,
    )

    assert windows == {"a": 200_000, "b": 200_000}
    assert lookups == [("anthropic", "claude-sonnet-4-5"), ("anthropic", "mystery")]


def test_a_failing_lookup_leaves_the_window_unknown_instead_of_failing_the_read() -> None:
    def window_of(provider: str | None, model: str) -> int | None:
        raise RuntimeError("catalog unavailable")

    assert _context_windows([_llm("a", "claude-sonnet-4-5")], window_of) == {}


def test_the_public_span_names_its_window_or_null() -> None:
    span = _llm("a", "claude-sonnet-4-5")

    assert _span_public(span, frozenset(), {"a": 200_000}).context_window == 200_000
    assert _span_public(span, frozenset(), {}).context_window is None
