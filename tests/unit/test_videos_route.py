"""The videos route's pure parts: the request shape and the job's answer."""

import pytest
from any_llm.types.video import VideoError as ProviderVideoError
from any_llm.types.video import VideoGenerationParams, VideoJob

from gateway.api.routes.videos import job_response, no_video_price_detail
from gateway.core.config import API_ROOT
from gateway.schemas.inference import VideoGenerationRequest
from gateway.services.upstream_redaction import MAX_EXPOSED_DETAIL_CHARS


def test_the_request_accepts_every_video_generation_param() -> None:
    """A field any-llm adds to ``VideoGenerationParams`` fails here until the route accepts and forwards it."""
    expected = {"model" if name == "model_id" else name for name in VideoGenerationParams.model_fields}

    assert expected <= set(VideoGenerationRequest.model_fields), sorted(
        expected - set(VideoGenerationRequest.model_fields)
    )


@pytest.mark.parametrize("seconds", [0, 3601])
def test_a_length_outside_one_second_to_an_hour_is_refused(seconds: int) -> None:
    with pytest.raises(ValueError, match="seconds"):
        VideoGenerationRequest(model="gemini:veo", prompt="a cat", seconds=seconds)


def test_the_answer_carries_the_providers_report_under_the_instance_that_runs_it() -> None:
    job = VideoJob(
        id="video_1",
        model="veo-3.0-generate-001",
        status="failed",
        created_at=1_700_000_000,
        seconds=8,
        error=ProviderVideoError(message="content policy", code="blocked"),
    )

    answer = job_response(job, provider="gemini-eu")

    assert answer.model_dump() == {
        "id": "video_1",
        "object": "video",
        "model": "veo-3.0-generate-001",
        "provider": "gemini-eu",
        "status": "failed",
        "created_at": 1_700_000_000,
        "completed_at": None,
        "progress": None,
        "seconds": 8,
        "size": None,
        "error": {"message": "content policy", "code": "blocked"},
    }


def test_a_failed_jobs_message_is_redacted_and_capped() -> None:
    message = "key sk-proj-AAAAAAAAAAAAAAAAAAAA was refused " + "x" * 1000
    job = VideoJob(id="video_1", model="veo", status="failed", error=ProviderVideoError(message=message))

    answer = job_response(job, provider="gemini")

    assert answer.error is not None
    assert "sk-proj-AAAA" not in answer.error.message
    assert len(answer.error.message) <= MAX_EXPOSED_DETAIL_CHARS


def test_the_no_price_detail_says_what_to_set_and_where() -> None:
    detail = no_video_price_detail("gemini:veo")

    assert "per-second video price" in detail
    assert f"POST {API_ROOT}/pricing" in detail
