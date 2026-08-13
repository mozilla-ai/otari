"""Unit tests for `gateway.services.file_extractors` converter reuse.

The extraction results themselves are markitdown's business; what is tested here
is that the gateway builds exactly one converter. Constructing a ``MarkItDown``
initializes magika, which loads an ONNX model and starts an inference thread
pool that is never released, so one-per-call leaked memory and threads.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from gateway.services import file_extractors


@pytest.fixture(autouse=True)
def _reset_converter() -> Any:
    file_extractors._converter = None
    yield
    file_extractors._converter = None


def test_converter_is_built_once(monkeypatch: pytest.MonkeyPatch) -> None:
    built = 0

    class FakeMarkItDown:
        def __init__(self, **_: Any) -> None:
            nonlocal built
            built += 1

    monkeypatch.setitem(
        __import__("sys").modules,
        "markitdown",
        type("m", (), {"MarkItDown": FakeMarkItDown}),
    )

    first = file_extractors._get_converter()
    second = file_extractors._get_converter()

    assert first is second
    assert built == 1


def test_concurrent_callers_share_one_converter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real leak: unlocked, every racing extraction builds its own."""
    built = 0

    class SlowMarkItDown:
        def __init__(self, **_: Any) -> None:
            nonlocal built
            built += 1
            # Widen the race window an unlocked implementation would lose.
            import time

            time.sleep(0.05)

    monkeypatch.setitem(
        __import__("sys").modules,
        "markitdown",
        type("m", (), {"MarkItDown": SlowMarkItDown}),
    )

    async def main() -> list[Any]:
        results: list[Any] = await asyncio.gather(
            *(asyncio.to_thread(file_extractors._get_converter) for _ in range(8))
        )
        return results

    converters = asyncio.run(main())

    assert built == 1, "a racing caller built a second converter"
    assert all(c is converters[0] for c in converters)


def test_failing_converter_build_degrades_instead_of_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """A magika/ONNX load failure must stay a failed extraction.

    The caller relies on ``ok=False`` to reach its PDF-rasterize / vision
    fallback; an exception escaping here skips the whole request's
    normalization instead of just this one document.
    """

    class Exploding:
        def __init__(self, **_: Any) -> None:
            raise RuntimeError("magika model failed to load")

    monkeypatch.setitem(
        __import__("sys").modules,
        "markitdown",
        type("m", (), {"MarkItDown": Exploding}),
    )

    result = file_extractors._extract_sync(b"data", ".txt")

    assert result.ok is False
    assert "magika model failed to load" in result.detail


def test_missing_markitdown_reports_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def no_markitdown(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "markitdown":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_markitdown)

    assert file_extractors._get_converter() is None
    result = file_extractors._extract_sync(b"data", ".txt")
    assert result.ok is False
    assert "markitdown not installed" in result.detail
