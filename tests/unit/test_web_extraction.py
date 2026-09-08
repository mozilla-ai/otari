"""Tests for the supervised web extraction process."""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import Future
from multiprocessing.connection import Connection
from time import monotonic

import pytest

from gateway.services.web_extraction import (
    WEB_FETCH_EXTRACTION_MEMORY_BYTES,
    WEB_FETCH_EXTRACTION_TIMEOUT_SECONDS,
    WEB_FETCH_MAX_INTERMEDIATE_TEXT_BYTES,
    WEB_FETCH_MAX_PDF_PAGES,
    WEB_FETCH_MAX_PENDING_EXTRACTIONS,
    ExtractedText,
    ExtractionError,
    ExtractionQueueFullError,
    ExtractionSupervisor,
    _PendingExtraction,
    _truncate_worker_text,
    _worker_main,
)
from gateway.services.web_retrieval_network import MAX_DECODED_BODY_BYTES


def _crash_worker(_connection: Connection, _memory_bytes: int) -> None:
    os._exit(17)


def _hang_worker(_connection: Connection, _memory_bytes: int) -> None:
    import time

    time.sleep(10)


class _CancelOnCompletionFuture(Future[ExtractedText]):
    def set_result(self, result: ExtractedText) -> None:
        self.cancel()
        super().set_result(result)

    def set_exception(self, exception: BaseException | None) -> None:
        self.cancel()
        super().set_exception(exception)


def test_worker_output_truncation_preserves_utf8() -> None:
    result = _truncate_worker_text("é" * WEB_FETCH_MAX_INTERMEDIATE_TEXT_BYTES)

    assert result.truncated is True
    assert len(result.text.encode("utf-8")) <= WEB_FETCH_MAX_INTERMEDIATE_TEXT_BYTES
    assert "�" not in result.text


@pytest.mark.asyncio
async def test_html_extraction_runs_in_worker_and_returns_markdown() -> None:
    supervisor = ExtractionSupervisor()
    try:
        result = await supervisor.extract_html("<html><body><h1>Heading</h1><p>Article body.</p></body></html>")
    finally:
        supervisor.close()

    assert "Heading" in result.text
    assert "Article body" in result.text
    assert len(result.text.encode("utf-8")) <= WEB_FETCH_MAX_INTERMEDIATE_TEXT_BYTES


@pytest.mark.asyncio
async def test_failed_extraction_does_not_poison_next_job() -> None:
    supervisor = ExtractionSupervisor()
    try:
        with pytest.raises(ExtractionError):
            await supervisor.extract_html("<html></html>")
        result = await supervisor.extract_html("<html><body><p>next job succeeds</p></body></html>")
    finally:
        supervisor.close()

    assert "next job succeeds" in result.text


@pytest.mark.asyncio
async def test_crashed_worker_is_replaced_before_next_job() -> None:
    supervisor = ExtractionSupervisor(worker_target=_crash_worker)
    try:
        with pytest.raises(ExtractionError):
            await supervisor.extract_html("<html><body>crash</body></html>")
        supervisor._worker_target = _worker_main  # noqa: SLF001
        result = await supervisor.extract_html("<html><body><p>recovered</p></body></html>")
    finally:
        supervisor.close()

    assert "recovered" in result.text


@pytest.mark.asyncio
async def test_submission_deadline_includes_worker_startup_and_recovers() -> None:
    supervisor = ExtractionSupervisor(timeout_seconds=0.000_001)
    try:
        with pytest.raises(ExtractionError, match="deadline"):
            await supervisor.extract_html("<html><body>too late</body></html>")
        supervisor._timeout_seconds = WEB_FETCH_EXTRACTION_TIMEOUT_SECONDS  # noqa: SLF001
        result = await supervisor.extract_html("<html><body><p>recovered</p></body></html>")
    finally:
        supervisor.close()

    assert "recovered" in result.text


@pytest.mark.asyncio
async def test_timeout_fails_current_and_queued_jobs_then_recovers() -> None:
    supervisor = ExtractionSupervisor(timeout_seconds=0.05, worker_target=_hang_worker)
    jobs = [
        asyncio.create_task(supervisor.extract_html(f"<html><body>{index}</body></html>"))
        for index in range(3)
    ]
    try:
        results = await asyncio.gather(*jobs, return_exceptions=True)
        assert all(isinstance(result, ExtractionError) for result in results)
        supervisor._worker_target = _worker_main  # noqa: SLF001
        supervisor._timeout_seconds = WEB_FETCH_EXTRACTION_TIMEOUT_SECONDS  # noqa: SLF001
        recovered = await supervisor.extract_html("<html><body><p>recovered</p></body></html>")
    finally:
        supervisor.close()

    assert "recovered" in recovered.text


@pytest.mark.asyncio
async def test_queue_capacity_counts_running_job() -> None:
    supervisor = ExtractionSupervisor(timeout_seconds=0.05, max_pending=1, worker_target=_hang_worker)
    first = asyncio.create_task(supervisor.extract_html("<html><body>first</body></html>"))
    await asyncio.sleep(0)
    try:
        with pytest.raises(ExtractionQueueFullError):
            await supervisor.extract_html("<html><body>second</body></html>")
        with pytest.raises(ExtractionError):
            await first
    finally:
        supervisor.close()


@pytest.mark.parametrize("error", [None, ExtractionError("failed")])
def test_finishing_a_job_is_atomic_with_cancellation(error: ExtractionError | None) -> None:
    supervisor = ExtractionSupervisor(worker_target=_hang_worker)
    future = _CancelOnCompletionFuture()
    pending = _PendingExtraction(
        identifier=1,
        kind="html",
        payload="payload",
        expires_at=monotonic() + 1,
        future=future,
    )
    supervisor._pending_count = 1  # noqa: SLF001
    try:
        supervisor._finish(  # noqa: SLF001
            pending,
            result=None if error else ExtractedText("done"),
            error=error,
        )
    finally:
        supervisor.close()

    assert future.cancelled() is False
    if error is None:
        assert future.result() == ExtractedText("done")
    else:
        assert future.exception() is error


@pytest.mark.asyncio
async def test_max_sized_payload_cannot_block_past_its_deadline() -> None:
    supervisor = ExtractionSupervisor(timeout_seconds=0.05, worker_target=_hang_worker)
    try:
        with pytest.raises(ExtractionError, match="deadline"):
            await asyncio.wait_for(
                supervisor.extract_pdf(b"x" * MAX_DECODED_BODY_BYTES),
                timeout=2,
            )
        supervisor._worker_target = _worker_main  # noqa: SLF001
        supervisor._timeout_seconds = WEB_FETCH_EXTRACTION_TIMEOUT_SECONDS  # noqa: SLF001
        recovered = await supervisor.extract_html("<html><body><p>recovered</p></body></html>")
    finally:
        supervisor.close()

    assert "recovered" in recovered.text


def test_extraction_limits_are_fixed_when_environment_names_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_WEB_FETCH_EXTRACTION_TIMEOUT_SECONDS", "999")
    monkeypatch.setenv("OTARI_WEB_FETCH_MAX_PDF_PAGES", "999")
    monkeypatch.setenv("OTARI_WEB_FETCH_MAX_INTERMEDIATE_TEXT_BYTES", "999999999")
    monkeypatch.setenv("OTARI_WEB_FETCH_EXTRACTION_MEMORY_BYTES", "999999999")
    monkeypatch.setenv("OTARI_WEB_FETCH_MAX_PENDING_EXTRACTIONS", "999")

    assert WEB_FETCH_EXTRACTION_TIMEOUT_SECONDS == 5.0
    assert WEB_FETCH_MAX_PDF_PAGES == 100
    assert WEB_FETCH_MAX_INTERMEDIATE_TEXT_BYTES == 256 * 1024
    assert WEB_FETCH_EXTRACTION_MEMORY_BYTES == 256 * 1024 * 1024
    assert WEB_FETCH_MAX_PENDING_EXTRACTIONS == 8
