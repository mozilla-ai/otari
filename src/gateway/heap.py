"""Return freed heap to the kernel after allocation-heavy work.

glibc's allocator keeps freed chunks in its per-thread arenas instead of handing
them back, so a burst of large, short-lived allocations leaves a process at its
high-water mark. The supervised web-extraction worker calls
:func:`release_free_heap` after every parser job so its bounded, long-lived
process does not retain libxml2's freed allocations.

This is a mitigation rather than a repair: it returns already-free memory held
by the allocator.

Availability is resolved once at import. ``malloc_trim`` is a GNU extension, so
this is a no-op on musl and macOS rather than an error. Callers treat releasing
memory as best-effort and must not branch on whether it happened.
"""

from __future__ import annotations

import ctypes
import platform
from typing import Callable

from gateway.log_config import logger


def _resolve_malloc_trim() -> Callable[[int], int] | None:
    """Look up ``malloc_trim`` in the running libc, or return None if absent."""
    if platform.system() != "Linux":
        return None
    try:
        libc = ctypes.CDLL("libc.so.6")
        trim = libc.malloc_trim
    except (OSError, AttributeError):
        # Not glibc (musl names its libc differently and omits the symbol).
        return None
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return trim


_MALLOC_TRIM = _resolve_malloc_trim()


def malloc_trim_available() -> bool:
    """Whether this platform can return free heap to the kernel."""
    return _MALLOC_TRIM is not None


def release_free_heap() -> None:
    """Best-effort release of free heap held by the allocator.

    Safe to call from the event loop: the call walks the arena free lists and
    completes in single-digit milliseconds for the heap sizes the gateway
    reaches. Never raises, so callers can treat it as fire-and-forget.
    """
    if _MALLOC_TRIM is None:
        return
    try:
        _MALLOC_TRIM(0)
    except Exception as exc:  # noqa: BLE001 — releasing memory must never fail a request
        logger.debug("malloc_trim failed: %s", exc)
