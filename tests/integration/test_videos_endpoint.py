"""The videos endpoint: a job starts under the budget, is read by its owner, and is billed once when it completes."""

import time
from collections.abc import Callable
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest
from any_llm.types.video import VideoError, VideoJob, VideoOutput
from any_llm.utils.exception_handler import convert_exception
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT
from gateway.core.error_codes import ERROR_CODE_HEADER, PRICING_REQUIRED
from gateway.models.usage import UsageLog
from gateway.services.inference import VideoJobService

from .conftest import held_reservations

_MODEL = "openai:sora-2"
_OWNER = "video-owner"
_OTHER = "video-other"


class _ServesVideo:
    SUPPORTS_VIDEO_GENERATION = True


class _NoVideo:
    SUPPORTS_VIDEO_GENERATION = False


def _job(**overrides: Any) -> VideoJob:
    fields: dict[str, Any] = {"id": "video_abc", "model": "sora-2", "status": "queued", "created_at": 1_700_000_000}
    fields.update(overrides)
    return VideoJob(**fields)


def _price(client: TestClient, headers: dict[str, str], rates: dict[str, Any]) -> None:
    response = client.post(f"{API_ROOT}/pricing", json={"model_key": _MODEL, "rates": rates}, headers=headers)
    assert response.status_code == 200, response.text


def _per_second(client: TestClient, headers: dict[str, str], price: float = 0.1) -> None:
    _price(client, headers, {"video": {"output": {"price": price, "unit": "second"}}})


def _user_key(
    client: TestClient, headers: dict[str, str], user_id: str = _OWNER, *, budget: float = 100.0, **key_fields: Any
) -> dict[str, str]:
    budget_row = client.post(f"{API_ROOT}/budgets", json={"max_budget": budget}, headers=headers)
    assert budget_row.status_code == 200, budget_row.text
    user = client.post(
        f"{API_ROOT}/users", json={"user_id": user_id, "budget_id": budget_row.json()["budget_id"]}, headers=headers
    )
    assert user.status_code == 200, user.text
    key = client.post(
        f"{API_ROOT}/keys", json={"key_name": f"{user_id}-key", "user_id": user_id, **key_fields}, headers=headers
    )
    assert key.status_code == 200, key.text
    return {API_KEY_HEADER: f"Bearer {key.json()['key']}"}


def _start(client: TestClient, headers: dict[str, str], job: VideoJob | None = None, **body: Any) -> Any:
    with (
        patch("gateway.api.routes.videos.avideo_generation", new=AsyncMock(return_value=job or _job())),
        patch("gateway.api.routes.videos.AnyLLM.get_provider_class", return_value=_ServesVideo),
    ):
        return client.post(f"{API_ROOT}/videos", json={"model": _MODEL, "prompt": "a cat", **body}, headers=headers)


def _rows(make_session: Callable[[], Session], endpoint: str, *, expected: int, timeout: float = 3.0) -> list[UsageLog]:
    """The usage rows under ``endpoint``, polled because a background log writer commits them."""
    deadline = time.monotonic() + timeout
    while True:
        with make_session() as db:
            rows = list(db.execute(select(UsageLog).where(UsageLog.endpoint == endpoint)).scalars().all())
            if len(rows) >= expected:
                for row in rows:
                    db.expunge(row)
                return rows
        if time.monotonic() > deadline:
            return rows
        time.sleep(0.1)


def _spend(client: TestClient, headers: dict[str, str], user_id: str = _OWNER) -> tuple[float, float]:
    user = client.get(f"{API_ROOT}/users/{user_id}", headers=headers).json()
    return user["spend"], user["reserved"]


# ---------------------------------------------------------------------------
# Starting a job
# ---------------------------------------------------------------------------


def test_starting_a_job_requires_authentication(client: TestClient) -> None:
    response = client.post(f"{API_ROOT}/videos", json={"model": _MODEL, "prompt": "a cat"})

    assert response.status_code == 401


def test_a_job_starts_under_the_estimate_for_its_length_and_settles_at_no_cost(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header)

    with held_reservations() as held:
        response = _start(client, headers, seconds=4)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "id": "video_abc",
        "object": "video",
        "model": "sora-2",
        "provider": "openai",
        "status": "queued",
        "created_at": 1_700_000_000,
        "completed_at": None,
        "progress": None,
        "seconds": None,
        "size": None,
        "error": None,
    }
    assert held == [Decimal("0.4")]
    (row,) = _rows(db_session_factory, "/v1/videos", expected=1)
    assert (row.status, row.cost, row.user_id, row.model, row.provider) == ("success", None, _OWNER, "sora-2", "openai")
    assert _spend(client, master_key_header) == (0.0, 0.0)


def test_a_job_that_completes_inside_the_start_call_is_billed_at_once(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header)
    done = _job(status="completed", seconds=5, outputs=[VideoOutput(url="https://cdn.example/v.mp4")])

    started = _start(client, headers, done, seconds=5)
    read = _read(client, headers, done)

    assert started.status_code == 200, started.text
    assert started.json()["status"] == "completed"
    assert read.status_code == 200, read.text
    (row,) = _rows(db_session_factory, "/v1/videos/completed", expected=1)
    assert row.cost == Decimal("0.5")
    assert _spend(client, master_key_header) == (0.5, 0.0)
    listed = client.get(f"{API_ROOT}/videos", headers=headers).json()["data"]
    assert [(job["id"], job["billed"]) for job in listed] == [("video_abc", True)]


def test_a_job_completed_in_the_start_call_is_read_and_served_without_the_provider(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header)
    done = _job(status="completed", seconds=5, outputs=[VideoOutput(url="https://cdn.example/v.mp4")])
    assert _start(client, headers, done, seconds=5).status_code == 200
    forgotten = AsyncMock(side_effect=RuntimeError("no such job in this process"))

    with patch("gateway.api.routes.videos.aretrieve_video", new=forgotten):
        read = client.get(f"{API_ROOT}/videos/video_abc", headers=headers)
    with (
        patch("gateway.api.routes.videos.aretrieve_video", new=forgotten),
        patch("gateway.api.routes.videos.adownload_video", new=AsyncMock(return_value=b"mp4")) as fetched,
    ):
        content = client.get(f"{API_ROOT}/videos/video_abc/content", headers=headers)

    assert read.status_code == 200, read.text
    assert (read.json()["status"], read.json()["seconds"]) == ("completed", 5)
    assert content.status_code == 200, content.text
    assert fetched.call_args.args[1].url == "https://cdn.example/v.mp4"
    forgotten.assert_not_awaited()


def test_a_job_found_completed_on_a_read_is_remembered_for_the_next_read(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header)
    assert _start(client, headers).status_code == 200
    done = _job(status="completed", seconds=3, outputs=[VideoOutput(url="https://cdn.example/v.mp4")])

    first = _read(client, headers, done)
    with patch("gateway.api.routes.videos.aretrieve_video", new=AsyncMock(side_effect=RuntimeError("gone"))):
        second = client.get(f"{API_ROOT}/videos/video_abc", headers=headers)

    assert first.status_code == 200 and second.status_code == 200, (first.text, second.text)
    assert second.json()["seconds"] == 3


def test_a_job_with_no_length_is_estimated_at_the_longest_default(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header)

    with held_reservations() as held:
        response = _start(client, headers)

    assert response.status_code == 200, response.text
    assert held == [Decimal("0.8")]


def test_the_request_is_forwarded_as_any_llm_names_it(client: TestClient, master_key_header: dict[str, str]) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header)

    with (
        patch("gateway.api.routes.videos.avideo_generation", new=AsyncMock(return_value=_job())) as started,
        patch("gateway.api.routes.videos.AnyLLM.get_provider_class", return_value=_ServesVideo),
    ):
        response = client.post(
            f"{API_ROOT}/videos",
            json={"model": _MODEL, "prompt": "a cat", "seconds": 4, "aspect_ratio": "16:9", "seed": 7},
            headers=headers,
        )

    assert response.status_code == 200, response.text
    started.assert_awaited_once()
    args, kwargs = started.call_args
    assert args == ("sora-2", "a cat")
    assert {key: kwargs[key] for key in ("seconds", "aspect_ratio", "seed")} == {
        "seconds": 4,
        "aspect_ratio": "16:9",
        "seed": 7,
    }
    assert "model" not in kwargs and "user" not in kwargs


def test_a_model_with_no_per_second_video_price_is_refused_before_the_provider_is_called(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _price(client, master_key_header, {"text": {"input": {"price": 3}, "output": {"price": 15}}})
    headers = _user_key(client, master_key_header)

    with (
        patch("gateway.api.routes.videos.avideo_generation", new=AsyncMock(return_value=_job())) as started,
        patch("gateway.api.routes.videos.AnyLLM.get_provider_class", return_value=_ServesVideo),
    ):
        response = client.post(f"{API_ROOT}/videos", json={"model": _MODEL, "prompt": "a cat"}, headers=headers)

    assert response.status_code == 402, response.text
    assert response.headers[ERROR_CODE_HEADER] == PRICING_REQUIRED
    assert "per-second video price" in response.json()["detail"]
    started.assert_not_awaited()
    assert _spend(client, master_key_header) == (0.0, 0.0)


def test_a_provider_that_does_not_serve_video_is_refused_before_it_is_called(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header)

    with (
        patch("gateway.api.routes.videos.avideo_generation", new=AsyncMock(return_value=_job())) as started,
        patch("gateway.api.routes.videos.AnyLLM.get_provider_class", return_value=_NoVideo),
    ):
        response = client.post(f"{API_ROOT}/videos", json={"model": _MODEL, "prompt": "a cat"}, headers=headers)

    assert response.status_code == 422, response.text
    assert "does not support video generation" in response.json()["detail"]
    started.assert_not_awaited()
    assert _spend(client, master_key_header) == (0.0, 0.0)


def test_the_estimate_for_the_length_asked_for_is_what_the_budget_gates(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header, budget=0.5)

    refused = _start(client, headers)
    accepted = _start(client, headers, seconds=4)

    assert refused.status_code == 403, refused.text
    assert "budget" in refused.json()["detail"].lower()
    assert accepted.status_code == 200, accepted.text


def test_a_provider_failure_refunds_the_reservation(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header)

    with (
        patch("gateway.api.routes.videos.avideo_generation", new=AsyncMock(side_effect=RuntimeError("down"))),
        patch("gateway.api.routes.videos.AnyLLM.get_provider_class", return_value=_ServesVideo),
    ):
        response = client.post(f"{API_ROOT}/videos", json={"model": _MODEL, "prompt": "a cat"}, headers=headers)

    assert response.status_code == 502, response.text
    (row,) = _rows(db_session_factory, "/v1/videos", expected=1)
    assert row.status == "error"
    assert _spend(client, master_key_header) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Reading a job
# ---------------------------------------------------------------------------


def _read(client: TestClient, headers: dict[str, str], job: VideoJob, video_id: str = "video_abc") -> Any:
    with patch("gateway.api.routes.videos.aretrieve_video", new=AsyncMock(return_value=job)):
        return client.get(f"{API_ROOT}/videos/{video_id}", headers=headers)


def test_a_job_is_read_from_the_provider_by_its_owner_and_the_master_key(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header)
    assert _start(client, headers).status_code == 200
    running = _job(status="in_progress", progress=40)

    by_owner = _read(client, headers, running)
    by_master = _read(client, master_key_header, running)

    assert by_owner.status_code == 200, by_owner.text
    assert (by_owner.json()["status"], by_owner.json()["progress"]) == ("in_progress", 40)
    assert by_master.status_code == 200, by_master.text


def test_another_users_job_and_an_unknown_id_are_not_found(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _per_second(client, master_key_header)
    owner = _user_key(client, master_key_header)
    other = _user_key(client, master_key_header, _OTHER)
    assert _start(client, owner).status_code == 200

    with patch("gateway.api.routes.videos.aretrieve_video", new=AsyncMock(return_value=_job())) as asked:
        foreign = client.get(f"{API_ROOT}/videos/video_abc", headers=other)
        unknown = client.get(f"{API_ROOT}/videos/video_nope", headers=owner)

    assert (foreign.status_code, unknown.status_code) == (404, 404)
    asked.assert_not_awaited()


def test_a_completed_job_is_billed_once_for_the_seconds_the_provider_reports(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header)
    assert _start(client, headers, seconds=4).status_code == 200
    done = _job(status="completed", seconds=6, outputs=[VideoOutput(url="https://cdn.example/v.mp4")])

    first = _read(client, headers, done)
    second = _read(client, headers, done)

    assert first.status_code == 200 and second.status_code == 200, (first.text, second.text)
    (row,) = _rows(db_session_factory, "/v1/videos/completed", expected=1)
    assert row.cost == Decimal("0.6")
    assert row.billing_meters == {"seconds": 6}
    assert row.pricing_breakdown == [{"meter": "seconds", "units": 6, "unit": "second", "unit_rate": 0.1, "cost": 0.6}]
    assert row.pricing_source == "deployment"
    assert (row.user_id, row.counts_toward_budget) == (_OWNER, True)
    assert _spend(client, master_key_header) == (0.6, 0.0)
    listed = client.get(f"{API_ROOT}/videos", headers=headers).json()["data"]
    assert [(job["id"], job["seconds"], job["billed"]) for job in listed] == [("video_abc", 4, True)]


def test_a_completed_job_the_provider_gives_no_length_for_is_billed_for_the_length_asked(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header)
    assert _start(client, headers, seconds=5).status_code == 200

    response = _read(client, headers, _job(status="completed"))

    assert response.status_code == 200, response.text
    (row,) = _rows(db_session_factory, "/v1/videos/completed", expected=1)
    assert row.cost == Decimal("0.5")
    assert _spend(client, master_key_header) == (0.5, 0.0)


def test_a_completed_job_whose_length_nobody_knows_is_recorded_once_with_no_cost(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header)
    assert _start(client, headers).status_code == 200

    first = _read(client, headers, _job(status="completed"))
    second = _read(client, headers, _job(status="completed"))

    assert first.status_code == 200 and second.status_code == 200
    (row,) = _rows(db_session_factory, "/v1/videos/completed", expected=1)
    assert (row.cost, row.billing_meters) == (None, None)
    assert _spend(client, master_key_header) == (0.0, 0.0)


def test_a_failed_job_is_never_billed(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header)
    assert _start(client, headers, seconds=4).status_code == 200

    response = _read(client, headers, _job(status="failed", error=VideoError(message="moderation", code="blocked")))

    assert response.status_code == 200, response.text
    assert response.json()["error"] == {"message": "moderation", "code": "blocked"}
    assert _rows(db_session_factory, "/v1/videos/completed", expected=1, timeout=0.5) == []
    assert _spend(client, master_key_header) == (0.0, 0.0)


def test_a_job_started_on_a_budget_exempt_key_is_recorded_and_not_charged(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header, exclude_from_budget=True)
    assert _start(client, headers, seconds=4).status_code == 200

    response = _read(client, headers, _job(status="completed", seconds=4))

    assert response.status_code == 200, response.text
    (row,) = _rows(db_session_factory, "/v1/videos/completed", expected=1)
    assert (row.cost, row.counts_toward_budget) == (Decimal("0.4"), False)
    assert _spend(client, master_key_header) == (0.0, 0.0)


def test_a_job_the_gateway_cannot_record_is_a_500(client: TestClient, master_key_header: dict[str, str]) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header)
    down = OperationalError("INSERT INTO video_jobs", {}, Exception("connection reset"))

    with patch.object(VideoJobService, "record", new=AsyncMock(side_effect=down)):
        response = _start(client, headers)

    assert response.status_code == 500, response.text
    assert "connection reset" not in response.text


def test_a_rate_limited_retrieval_is_a_429_and_a_rejected_credential_a_424(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header)
    assert _start(client, headers).status_code == 200
    request = httpx.Request("GET", "https://api.openai.com/v1/videos/video_abc")
    limited = convert_exception(
        openai.RateLimitError("Error code: 429 - slow down", response=httpx.Response(429, request=request), body=None),
        "openai",
    )
    rejected = convert_exception(
        openai.AuthenticationError(
            "Error code: 401 - Incorrect API key provided: sk-proj-AAAA",
            response=httpx.Response(401, request=request),
            body=None,
        ),
        "openai",
    )

    with patch("gateway.api.routes.videos.aretrieve_video", new=AsyncMock(side_effect=limited)):
        too_many = client.get(f"{API_ROOT}/videos/video_abc", headers=headers)
    with patch("gateway.api.routes.videos.aretrieve_video", new=AsyncMock(side_effect=rejected)):
        unusable = client.get(f"{API_ROOT}/videos/video_abc", headers=headers)

    assert too_many.status_code == 429, too_many.text
    assert unusable.status_code == 424, unusable.text
    assert "sk-proj" not in unusable.text


def test_a_failed_jobs_message_is_redacted(client: TestClient, master_key_header: dict[str, str]) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header)
    assert _start(client, headers).status_code == 200
    failed = _job(status="failed", error=VideoError(message="key sk-proj-AAAAAAAAAAAAAAAAAAAA was refused"))

    response = _read(client, headers, failed)

    assert response.status_code == 200, response.text
    assert "sk-proj-AAAA" not in response.text
    assert response.json()["error"]["message"]


def test_a_provider_failure_on_retrieval_is_a_502(client: TestClient, master_key_header: dict[str, str]) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header)
    assert _start(client, headers).status_code == 200

    with patch("gateway.api.routes.videos.aretrieve_video", new=AsyncMock(side_effect=RuntimeError("down"))):
        response = client.get(f"{API_ROOT}/videos/video_abc", headers=headers)

    assert response.status_code == 502, response.text


# ---------------------------------------------------------------------------
# The video itself
# ---------------------------------------------------------------------------


def test_the_content_of_a_completed_job_is_served_in_its_media_type_and_bills_it_once(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header)
    assert _start(client, headers).status_code == 200
    output = VideoOutput(url="https://cdn.example/v.webm", mime_type="video/webm")
    done = _job(status="completed", seconds=3, outputs=[output])

    with (
        patch("gateway.api.routes.videos.aretrieve_video", new=AsyncMock(return_value=done)),
        patch("gateway.api.routes.videos.adownload_video", new=AsyncMock(return_value=b"\x1aE\xdf\xa3")) as fetched,
    ):
        response = client.get(f"{API_ROOT}/videos/video_abc/content", headers=headers)
    again = _read(client, headers, done)

    assert response.status_code == 200, response.text
    assert (response.headers["content-type"], response.content) == ("video/webm", b"\x1aE\xdf\xa3")
    assert fetched.call_args.args[1] == output
    assert again.status_code == 200
    (row,) = _rows(db_session_factory, "/v1/videos/completed", expected=1)
    assert row.cost == Decimal("0.3")
    assert _spend(client, master_key_header) == (0.3, 0.0)


@pytest.mark.parametrize("job_status", ["queued", "in_progress", "failed"])
def test_the_content_of_a_job_that_has_not_completed_is_a_409(
    client: TestClient, master_key_header: dict[str, str], job_status: str
) -> None:
    _per_second(client, master_key_header)
    headers = _user_key(client, master_key_header)
    assert _start(client, headers).status_code == 200

    with (
        patch("gateway.api.routes.videos.aretrieve_video", new=AsyncMock(return_value=_job(status=job_status))),
        patch("gateway.api.routes.videos.adownload_video", new=AsyncMock()) as fetched,
    ):
        response = client.get(f"{API_ROOT}/videos/video_abc/content", headers=headers)

    assert response.status_code == 409, response.text
    assert f"status: {job_status}" in response.json()["detail"]
    fetched.assert_not_awaited()


# ---------------------------------------------------------------------------
# Listing jobs
# ---------------------------------------------------------------------------


def test_a_key_lists_its_own_jobs_and_the_master_key_every_job(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _per_second(client, master_key_header)
    owner = _user_key(client, master_key_header)
    other = _user_key(client, master_key_header, _OTHER)
    assert _start(client, owner, _job(id="video_1")).status_code == 200
    assert _start(client, other, _job(id="video_2")).status_code == 200
    assert _start(client, owner, _job(id="video_3"), seconds=4).status_code == 200

    mine = client.get(f"{API_ROOT}/videos", headers=owner).json()
    everything = client.get(f"{API_ROOT}/videos", headers=master_key_header).json()
    newest = client.get(f"{API_ROOT}/videos?limit=1", headers=master_key_header).json()

    assert mine["object"] == "list"
    assert [job["id"] for job in mine["data"]] == ["video_3", "video_1"]
    assert mine["data"][0] == {
        "id": "video_3",
        "object": "video",
        "model": "sora-2",
        "provider": "openai",
        "created_at": mine["data"][0]["created_at"],
        "seconds": 4,
        "billed": False,
    }
    assert {job["id"] for job in everything["data"]} == {"video_1", "video_2", "video_3"}
    assert [job["id"] for job in newest["data"]] == ["video_3"]
