"""Bounded, private forwarding and safe receiver failures."""

import logging
from collections.abc import Iterator

import httpx
import pytest

from gateway.exceptions.feedback_exceptions import FeedbackDeliveryError
from gateway.log_config import logger as gateway_logger
from gateway.schemas.feedback import FeedbackSubmission
from gateway.services.feedback import FeedbackService


@pytest.fixture
def gateway_logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """The gateway logger does not propagate, so caplog sees it only with its handler attached."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger="gateway")
    try:
        yield caplog
    finally:
        gateway_logger.removeHandler(caplog.handler)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("upstream", "expected"), [(204, None), (200, 503), (302, 503), (413, 413), (422, 422), (429, 429), (500, 503)]
)
async def test_receiver_statuses(upstream: int, expected: int | None, gateway_logs: pytest.LogCaptureFixture) -> None:
    calls = []

    def receive(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            upstream,
            text="PRIVATE upstream details",
            headers={"Retry-After": "999999", "Location": "https://elsewhere.test"},
        )

    service = FeedbackService(transport=httpx.MockTransport(receive))
    if expected is None:
        await service.submit(FeedbackSubmission(message="PRIVATE feedback"))
    else:
        with pytest.raises(FeedbackDeliveryError) as error:
            await service.submit(FeedbackSubmission(message="PRIVATE feedback"))
        assert error.value.status_code == expected
        assert error.value.retry_after == (3600 if expected == 429 else None)
        assert f"Feedback receiver answered {upstream}" in gateway_logs.text
    assert len(calls) == 1
    assert "PRIVATE" not in gateway_logs.text


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_after", ["-1", "Wed, 21 Oct 2015 07:28:00 GMT", "1" * 1000, ""])
async def test_untrusted_retry_header(retry_after: str) -> None:
    service = FeedbackService(
        transport=httpx.MockTransport(lambda request: httpx.Response(429, headers={"Retry-After": retry_after}))
    )
    with pytest.raises(FeedbackDeliveryError) as error:
        await service.submit(FeedbackSubmission(message="Idea"))
    assert error.value.retry_after is None


@pytest.mark.asyncio
async def test_timeout_has_no_automatic_retry() -> None:
    calls = 0

    def receive(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("private upstream details", request=request)

    with pytest.raises(FeedbackDeliveryError) as error:
        await FeedbackService(transport=httpx.MockTransport(receive)).submit(FeedbackSubmission(message="Idea"))
    assert error.value.status_code == 503
    assert calls == 1
