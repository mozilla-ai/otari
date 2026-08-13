"""Unit tests for `gateway.heap` — no database required."""

from __future__ import annotations

import platform

import pytest

from gateway import heap


def test_release_free_heap_never_raises() -> None:
    """Callers treat this as fire-and-forget, so it must swallow everything."""
    heap.release_free_heap()
    heap.release_free_heap()


def test_release_free_heap_swallows_a_failing_libc(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_: int) -> int:
        raise OSError("malloc_trim exploded")

    monkeypatch.setattr(heap, "_MALLOC_TRIM", boom)
    heap.release_free_heap()


def test_release_free_heap_is_a_noop_without_glibc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(heap, "_MALLOC_TRIM", None)
    heap.release_free_heap()
    assert heap.malloc_trim_available() is False


def test_release_free_heap_calls_into_libc(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def record(pad: int) -> int:
        calls.append(pad)
        return 0

    monkeypatch.setattr(heap, "_MALLOC_TRIM", record)

    heap.release_free_heap()

    assert calls == [0], "malloc_trim takes a pad of 0 to release everything it can"


@pytest.mark.skipif(platform.system() != "Linux", reason="malloc_trim is a glibc extension")
def test_malloc_trim_resolves_on_linux() -> None:
    """The deployment target is glibc Linux; a silent no-op there would hide the fix."""
    assert heap.malloc_trim_available() is True
