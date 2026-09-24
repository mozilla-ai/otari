"""Bounded, private forwarding and safe receiver failures."""

import logging

import httpx
import pytest

from gateway.exceptions.feedback_exceptions import FeedbackDeliveryError
from gateway.schemas.feedback import FeedbackSubmission
from gateway.services.feedback import FeedbackService


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("upstream", "expected"), [(204, None), (200, 503), (302, 503), (413, 413), (422, 422), (429, 429), (500, 503)]
)
async def test_receiver_statuses(upstream: int, expected: int | None, caplog: pytest.LogCaptureFixture) -> None:
    calls = []

    def receive(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            upstream,
            text="PRIVATE upstream details",
            headers={"Retry-After": "999999", "Location": "https://elsewhere.test"},
        )

    service = FeedbackService(transport=httpx.MockTransport(receive))
    with caplog.at_level(logging.DEBUG):
        if expected is None:
            await service.submit(FeedbackSubmission(message="PRIVATE feedback"))
        else:
            with pytest.raises(FeedbackDeliveryError) as error:
                await service.submit(FeedbackSubmission(message="PRIVATE feedback"))
            assert error.value.status_code == expected
            assert error.value.retry_after == (3600 if expected == 429 else None)
    assert len(calls) == 1
    assert "PRIVATE" not in caplog.text


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
