"""Killable, resource-bounded extraction for remotely retrieved documents."""

from __future__ import annotations

import asyncio
import multiprocessing
import queue
import sys
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from time import monotonic
from typing import Callable, Literal

WEB_FETCH_EXTRACTION_TIMEOUT_SECONDS = 5.0
WEB_FETCH_MAX_PDF_PAGES = 100
WEB_FETCH_MAX_INTERMEDIATE_TEXT_BYTES = 256 * 1024
WEB_FETCH_EXTRACTION_MEMORY_BYTES = 256 * 1024 * 1024
WEB_FETCH_MAX_PENDING_EXTRACTIONS = 8

ExtractionKind = Literal["html", "pdf"]
WorkerTarget = Callable[[Connection, int], None]


class ExtractionError(RuntimeError):
    """A remote document could not be extracted within the fixed limits."""


class ExtractionQueueFullError(ExtractionError):
    """The bounded extraction worker queue has no remaining capacity."""


@dataclass(frozen=True, slots=True)
class ExtractedText:
    """Bounded text returned by the isolated parser process."""

    text: str
    truncated: bool = False


@dataclass(slots=True)
class _PendingExtraction:
    identifier: int
    kind: ExtractionKind
    payload: str | bytes
    expires_at: float
    future: Future[ExtractedText]


def _truncate_worker_text(value: str) -> ExtractedText:
    encoded = value.encode("utf-8")
    if len(encoded) <= WEB_FETCH_MAX_INTERMEDIATE_TEXT_BYTES:
        return ExtractedText(value, False)
    bounded = encoded[:WEB_FETCH_MAX_INTERMEDIATE_TEXT_BYTES].decode("utf-8", errors="ignore")
    return ExtractedText(bounded, True)


def _apply_worker_limits(memory_bytes: int) -> None:
    """Apply limits before importing native parsers in the child process."""
    if not sys.platform.startswith("linux"):
        return
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _extract_html(value: str) -> ExtractedText:
    import trafilatura

    result = trafilatura.extract(
        value,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )
    text = str(result).strip() if result else ""
    if not text:
        raise ExtractionError("HTML has no extractable text")
    return _truncate_worker_text(text)


def _extract_pdf(value: bytes) -> ExtractedText:
    from gateway.services.file_extractors import extract_bounded_pdf_text_sync

    result = extract_bounded_pdf_text_sync(
        value,
        max_pages=WEB_FETCH_MAX_PDF_PAGES,
        max_text_bytes=WEB_FETCH_MAX_INTERMEDIATE_TEXT_BYTES,
    )
    if not result.ok:
        raise ExtractionError("PDF has no extractable text")
    return ExtractedText(result.text, result.truncated)


def _worker_main(connection: Connection, memory_bytes: int) -> None:
    """Serve extraction jobs until the parent closes the private pipe."""
    try:
        _apply_worker_limits(memory_bytes)
        from gateway.heap import release_free_heap

        while True:
            message = connection.recv()
            if message is None:
                return
            identifier, kind, payload = message
            try:
                if kind == "html" and isinstance(payload, str):
                    result = _extract_html(payload)
                elif kind == "pdf" and isinstance(payload, bytes):
                    result = _extract_pdf(payload)
                else:
                    raise ExtractionError("invalid extraction job")
            except BaseException:
                response = (identifier, False, "", False)
            else:
                response = (identifier, True, result.text, result.truncated)
            finally:
                release_free_heap()
            connection.send(response)
    except (EOFError, BrokenPipeError, OSError):
        return
    finally:
        connection.close()


class ExtractionSupervisor:
    """Serialize parser jobs through one replaceable child process.

    A parent thread owns the child and its pipe. Async callers receive ordinary
    futures, so the singleton is safe to use from multiple application event
    loops without binding its queue or synchronization primitives to one loop.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = WEB_FETCH_EXTRACTION_TIMEOUT_SECONDS,
        max_pending: int = WEB_FETCH_MAX_PENDING_EXTRACTIONS,
        memory_bytes: int = WEB_FETCH_EXTRACTION_MEMORY_BYTES,
        worker_target: WorkerTarget = _worker_main,
    ) -> None:
        if timeout_seconds <= 0 or max_pending <= 0 or memory_bytes <= 0:
            raise ValueError("extraction limits must be positive")
        self._timeout_seconds = timeout_seconds
        self._max_pending = max_pending
        self._memory_bytes = memory_bytes
        self._worker_target = worker_target
        self._context = multiprocessing.get_context("spawn")
        self._queue: queue.Queue[_PendingExtraction | None] = queue.Queue()
        self._state_lock = threading.Lock()
        self._pending_count = 0
        self._next_identifier = 1
        self._closed = False
        self._thread = threading.Thread(target=self._dispatch, name="otari-extract-supervisor", daemon=True)
        self._thread.start()

    async def extract_html(self, value: str) -> ExtractedText:
        return await self._submit("html", value)

    async def extract_pdf(self, value: bytes) -> ExtractedText:
        return await self._submit("pdf", value)

    async def _submit(self, kind: ExtractionKind, payload: str | bytes) -> ExtractedText:
        future: Future[ExtractedText] = Future()
        with self._state_lock:
            if self._closed:
                raise ExtractionError("extraction worker is closed")
            if self._pending_count >= self._max_pending:
                raise ExtractionQueueFullError("extraction queue is full")
            identifier = self._next_identifier
            self._next_identifier += 1
            self._pending_count += 1
        self._queue.put(
            _PendingExtraction(
                identifier=identifier,
                kind=kind,
                payload=payload,
                expires_at=monotonic() + self._timeout_seconds,
                future=future,
            )
        )
        return await asyncio.wrap_future(future)

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=self._timeout_seconds + 1)

    def _finish(
        self,
        pending: _PendingExtraction,
        *,
        result: ExtractedText | None = None,
        error: ExtractionError | None = None,
    ) -> None:
        with self._state_lock:
            self._pending_count -= 1
        if pending.future.cancelled():
            return
        if error is not None:
            pending.future.set_exception(error)
        elif result is not None:
            pending.future.set_result(result)
        else:
            pending.future.set_exception(ExtractionError("extraction worker failed"))

    def _fail_queued(self, error: ExtractionError) -> None:
        while True:
            try:
                pending = self._queue.get_nowait()
            except queue.Empty:
                return
            if pending is None:
                self._queue.put(None)
                return
            self._finish(pending, error=error)

    @staticmethod
    def _stop_worker(process: BaseProcess | None, connection: Connection | None) -> None:
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is None:
            return
        if process.is_alive():
            process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)

    def _start_worker(self) -> tuple[BaseProcess, Connection]:
        parent, child = self._context.Pipe()
        process = self._context.Process(
            target=self._worker_target,
            args=(child, self._memory_bytes),
            name="otari-web-extraction",
            daemon=True,
        )
        process.start()
        child.close()
        return process, parent

    def _dispatch(self) -> None:
        process: BaseProcess | None = None
        connection: Connection | None = None
        try:
            while True:
                pending = self._queue.get()
                if pending is None:
                    return
                if monotonic() >= pending.expires_at:
                    error = ExtractionError("extraction deadline exceeded")
                    self._stop_worker(process, connection)
                    process = None
                    connection = None
                    self._finish(pending, error=error)
                    self._fail_queued(error)
                    continue
                if process is None or not process.is_alive() or connection is None:
                    self._stop_worker(process, connection)
                    try:
                        process, connection = self._start_worker()
                    except BaseException:
                        process = None
                        connection = None
                        error = ExtractionError("extraction worker could not start")
                        self._finish(pending, error=error)
                        self._fail_queued(error)
                        continue
                try:
                    connection.send((pending.identifier, pending.kind, pending.payload))
                    remaining = pending.expires_at - monotonic()
                    if remaining <= 0 or not connection.poll(remaining):
                        raise TimeoutError
                    identifier, ok, text, truncated = connection.recv()
                    if identifier != pending.identifier:
                        raise ExtractionError("extraction worker protocol failed")
                    if not ok:
                        self._finish(pending, error=ExtractionError("content extraction failed"))
                        continue
                except BaseException as exc:
                    error = (
                        ExtractionError("extraction deadline exceeded")
                        if isinstance(exc, TimeoutError)
                        else ExtractionError("content extraction failed")
                    )
                    self._stop_worker(process, connection)
                    process = None
                    connection = None
                    self._finish(pending, error=error)
                    self._fail_queued(error)
                    continue
                self._finish(pending, result=ExtractedText(str(text), bool(truncated)))
        finally:
            self._stop_worker(process, connection)
            self._fail_queued(ExtractionError("extraction worker is closed"))


_shared_supervisor: ExtractionSupervisor | None = None
_shared_supervisor_lock = threading.Lock()


def get_extraction_supervisor() -> ExtractionSupervisor:
    """Return the process-wide extraction supervisor, creating it lazily."""
    global _shared_supervisor
    with _shared_supervisor_lock:
        if _shared_supervisor is None:
            _shared_supervisor = ExtractionSupervisor()
        return _shared_supervisor
