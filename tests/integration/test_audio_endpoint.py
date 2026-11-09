"""Tests for the POST /api/v1/audio/transcriptions and POST /api/v1/audio/speech endpoints."""

from datetime import UTC, datetime
from typing import Any, Never
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.audio import Transcription
from fastapi import Request
from fastapi.testclient import TestClient
from openai.types.audio.transcription import UsageDuration
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import (
    API_ROOT,
    COST_USD_HEADER,
    PRICING_SOURCE_HEADER,
    USAGE_UNIT_HEADER,
    USAGE_UNITS_HEADER,
)
from gateway.core.database import release_session
from gateway.services.pricing_service import (
    configure_default_pricing,
    default_model_pricing,
)


def _mock_transcription_response() -> Transcription:
    """Build a real Transcription for testing."""
    return Transcription(text="Hello, world!")


FAKE_AUDIO_BYTES = b"fake-audio-content-mp3"


# === Transcription tests ===


def test_transcription_requires_auth_before_reading_the_body(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/v1/audio/transcriptions requires authentication."""

    def unread(self: Request, **_: object) -> Never:
        raise AssertionError("the body was parsed before authentication")

    monkeypatch.setattr(Request, "form", unread)
    resp = client.post(
        f"{API_ROOT}/audio/transcriptions",
        files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
        data={"model": "openai:whisper-1"},
    )
    assert resp.status_code == 401


def test_transcription_body_over_the_route_limit_is_refused_before_parsing(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("gateway.api.routes.audio._MAX_AUDIO_UPLOAD_BYTES", 16)

    def unread(self: Request, **_: object) -> Never:
        raise AssertionError("the oversized body was parsed")

    monkeypatch.setattr(Request, "form", unread)
    response = client.post(
        f"{API_ROOT}/audio/transcriptions",
        headers=api_key_header,
        files={"file": ("large.mp3", b"x" * (16 + 1024 * 1024 + 1), "audio/mpeg")},
        data={"model": "openai:whisper-1"},
    )
    assert response.status_code == 413


def test_transcription_frees_the_auth_connection_before_reading_the_body(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow client must not pin the connection authentication checked out."""
    events: list[str] = []
    release = release_session
    form = Request.form

    async def recording_release(session: AsyncSession) -> bool:
        events.append("release")
        return await release(session)

    def recording_form(self: Request, **kwargs: Any) -> Any:
        events.append("form")
        return form(self, **kwargs)

    monkeypatch.setattr("gateway.api.deps.release_session", recording_release)
    monkeypatch.setattr(Request, "form", recording_form)
    with patch(
        "gateway.api.routes.audio.atranscription", new_callable=AsyncMock, return_value=_mock_transcription_response()
    ):
        resp = client.post(
            f"{API_ROOT}/audio/transcriptions",
            files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
            data={"model": "openai:whisper-1"},
            headers=api_key_header,
        )

    assert resp.status_code == 200
    assert events == ["release", "form"]


def test_transcription_with_api_key(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /api/v1/audio/transcriptions works with API key authentication."""
    mock_resp = _mock_transcription_response()
    with patch("gateway.api.routes.audio.atranscription", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            f"{API_ROOT}/audio/transcriptions",
            files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
            data={"model": "openai:whisper-1"},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "Hello, world!"


def test_transcription_forwards_file_name(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """The provider receives the upload's name and bytes, not anonymous bytes."""
    mock = AsyncMock(return_value=_mock_transcription_response())
    with patch("gateway.api.routes.audio.atranscription", mock):
        resp = client.post(
            f"{API_ROOT}/audio/transcriptions",
            files={"file": ("clip.m4a", b"audio-data", "audio/mp4")},
            data={"model": "openai:whisper-1"},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    assert mock.await_args is not None
    sent = mock.await_args.kwargs["file"]
    assert sent.name == "clip.m4a"
    assert sent.read() == b"audio-data"


def test_transcription_master_key_requires_user(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """POST /api/v1/audio/transcriptions with master key requires 'user' field."""
    mock_resp = _mock_transcription_response()
    with patch("gateway.api.routes.audio.atranscription", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            f"{API_ROOT}/audio/transcriptions",
            files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
            data={"model": "openai:whisper-1"},
            headers=master_key_header,
        )
    assert resp.status_code == 400
    assert "user" in resp.json()["detail"].lower()


def test_transcription_master_key_with_user(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """POST /api/v1/audio/transcriptions with master key + user field succeeds."""
    mock_resp = _mock_transcription_response()
    with patch("gateway.api.routes.audio.atranscription", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            f"{API_ROOT}/audio/transcriptions",
            files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
            data={"model": "openai:whisper-1", "user": test_user["user_id"]},
            headers=master_key_header,
        )
    assert resp.status_code == 200


def test_transcription_provider_error(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /api/v1/audio/transcriptions returns 502 when the provider fails."""
    with patch(
        "gateway.api.routes.audio.atranscription",
        new_callable=AsyncMock,
        side_effect=RuntimeError("provider down"),
    ):
        resp = client.post(
            f"{API_ROOT}/audio/transcriptions",
            files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
            data={"model": "openai:whisper-1"},
            headers=api_key_header,
        )
    assert resp.status_code == 502
    assert "provider" in resp.json()["detail"].lower()


def test_transcription_logs_usage(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /api/v1/audio/transcriptions creates a usage log entry."""
    mock_resp = _mock_transcription_response()
    user_id = api_key_obj["user_id"]

    with patch("gateway.api.routes.audio.atranscription", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            f"{API_ROOT}/audio/transcriptions",
            files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
            data={"model": "openai:whisper-1"},
            headers=api_key_header,
        )
    assert resp.status_code == 200

    usage_resp = client.get(f"{API_ROOT}/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    transcription_logs = [log for log in logs if log["endpoint"] == "/v1/audio/transcriptions"]
    assert len(transcription_logs) >= 1
    assert transcription_logs[0]["status"] == "success"
    assert transcription_logs[0]["prompt_tokens"] == 0


def test_transcription_logs_error_on_failure(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /api/v1/audio/transcriptions logs an error entry when the provider fails."""
    user_id = api_key_obj["user_id"]

    with patch(
        "gateway.api.routes.audio.atranscription",
        new_callable=AsyncMock,
        side_effect=RuntimeError("provider down"),
    ):
        resp = client.post(
            f"{API_ROOT}/audio/transcriptions",
            files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
            data={"model": "openai:whisper-1"},
            headers=api_key_header,
        )
    assert resp.status_code == 502

    usage_resp = client.get(f"{API_ROOT}/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    error_logs = [log for log in logs if log["endpoint"] == "/v1/audio/transcriptions" and log["status"] == "error"]
    assert len(error_logs) >= 1
    assert "provider down" in error_logs[0]["error_message"]


@pytest.mark.parametrize("extra_field", ["language", "prompt", "response_format", "temperature"])
def test_transcription_optional_fields(
    client: TestClient,
    api_key_header: dict[str, str],
    extra_field: str,
) -> None:
    """POST /api/v1/audio/transcriptions forwards optional fields."""
    mock_resp = _mock_transcription_response()
    values = {"language": "en", "prompt": "Previous context", "response_format": "verbose_json", "temperature": "0.2"}
    with patch("gateway.api.routes.audio.atranscription", new_callable=AsyncMock, return_value=mock_resp) as mock:
        resp = client.post(
            f"{API_ROOT}/audio/transcriptions",
            files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
            data={"model": "openai:whisper-1", extra_field: values[extra_field]},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    call_kwargs = mock.call_args.kwargs
    assert extra_field in call_kwargs


@pytest.mark.parametrize("extra_field", ["language", "prompt", "response_format", "temperature"])
def test_transcription_empty_optional_field_uses_default(
    client: TestClient, api_key_header: dict[str, str], extra_field: str
) -> None:
    with patch(
        "gateway.api.routes.audio.atranscription", new_callable=AsyncMock, return_value=_mock_transcription_response()
    ) as mock:
        resp = client.post(
            f"{API_ROOT}/audio/transcriptions",
            files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
            data={"model": "openai:whisper-1", extra_field: ""},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    assert extra_field not in mock.call_args.kwargs


def test_transcription_empty_model_is_missing(client: TestClient, api_key_header: dict[str, str]) -> None:
    resp = client.post(
        f"{API_ROOT}/audio/transcriptions",
        files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
        data={"model": ""},
        headers=api_key_header,
    )
    assert resp.status_code == 422
    assert any(error["type"] == "missing" and error["loc"] == ["body", "model"] for error in resp.json()["detail"])


# === Speech tests ===


def test_speech_requires_auth(client: TestClient) -> None:
    """POST /api/v1/audio/speech requires authentication."""
    resp = client.post(
        f"{API_ROOT}/audio/speech",
        json={"model": "openai:tts-1", "input": "Hello", "voice": "alloy"},
    )
    assert resp.status_code == 401


def test_speech_with_api_key(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /api/v1/audio/speech works with API key authentication."""
    with patch("gateway.api.routes.audio.aspeech", new_callable=AsyncMock, return_value=FAKE_AUDIO_BYTES):
        resp = client.post(
            f"{API_ROOT}/audio/speech",
            json={"model": "openai:tts-1", "input": "Hello", "voice": "alloy"},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content == FAKE_AUDIO_BYTES


def test_speech_content_type_matches_format(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /api/v1/audio/speech returns correct Content-Type for each format."""
    format_types = {
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
    }
    for fmt, expected_type in format_types.items():
        with patch("gateway.api.routes.audio.aspeech", new_callable=AsyncMock, return_value=FAKE_AUDIO_BYTES):
            resp = client.post(
                f"{API_ROOT}/audio/speech",
                json={"model": "openai:tts-1", "input": "Hi", "voice": "alloy", "response_format": fmt},
                headers=api_key_header,
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == expected_type, f"Expected {expected_type} for {fmt}"


def test_speech_master_key_requires_user(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """POST /api/v1/audio/speech with master key requires 'user' field."""
    with patch("gateway.api.routes.audio.aspeech", new_callable=AsyncMock, return_value=FAKE_AUDIO_BYTES):
        resp = client.post(
            f"{API_ROOT}/audio/speech",
            json={"model": "openai:tts-1", "input": "Hello", "voice": "alloy"},
            headers=master_key_header,
        )
    assert resp.status_code == 400
    assert "user" in resp.json()["detail"].lower()


def test_speech_master_key_with_user(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """POST /api/v1/audio/speech with master key + user field succeeds."""
    with patch("gateway.api.routes.audio.aspeech", new_callable=AsyncMock, return_value=FAKE_AUDIO_BYTES):
        resp = client.post(
            f"{API_ROOT}/audio/speech",
            json={
                "model": "openai:tts-1",
                "input": "Hello",
                "voice": "alloy",
                "user": test_user["user_id"],
            },
            headers=master_key_header,
        )
    assert resp.status_code == 200


def test_speech_provider_error(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /api/v1/audio/speech returns 502 when the provider fails."""
    with patch(
        "gateway.api.routes.audio.aspeech",
        new_callable=AsyncMock,
        side_effect=RuntimeError("provider down"),
    ):
        resp = client.post(
            f"{API_ROOT}/audio/speech",
            json={"model": "openai:tts-1", "input": "Hello", "voice": "alloy"},
            headers=api_key_header,
        )
    assert resp.status_code == 502
    assert "provider" in resp.json()["detail"].lower()


def test_speech_logs_usage(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /api/v1/audio/speech creates a usage log entry."""
    user_id = api_key_obj["user_id"]

    with patch("gateway.api.routes.audio.aspeech", new_callable=AsyncMock, return_value=FAKE_AUDIO_BYTES):
        resp = client.post(
            f"{API_ROOT}/audio/speech",
            json={"model": "openai:tts-1", "input": "Hello", "voice": "alloy"},
            headers=api_key_header,
        )
    assert resp.status_code == 200

    usage_resp = client.get(f"{API_ROOT}/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    speech_logs = [log for log in logs if log["endpoint"] == "/v1/audio/speech"]
    assert len(speech_logs) >= 1
    assert speech_logs[0]["status"] == "success"
    assert speech_logs[0]["prompt_tokens"] == 0


def test_speech_logs_error_on_failure(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /api/v1/audio/speech logs an error entry when the provider fails."""
    user_id = api_key_obj["user_id"]

    with patch(
        "gateway.api.routes.audio.aspeech",
        new_callable=AsyncMock,
        side_effect=RuntimeError("provider down"),
    ):
        resp = client.post(
            f"{API_ROOT}/audio/speech",
            json={"model": "openai:tts-1", "input": "Hello", "voice": "alloy"},
            headers=api_key_header,
        )
    assert resp.status_code == 502

    usage_resp = client.get(f"{API_ROOT}/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    error_logs = [log for log in logs if log["endpoint"] == "/v1/audio/speech" and log["status"] == "error"]
    assert len(error_logs) >= 1
    assert "provider down" in error_logs[0]["error_message"]


@pytest.mark.parametrize("extra_field", ["response_format", "speed", "instructions"])
def test_speech_optional_fields(
    client: TestClient,
    api_key_header: dict[str, str],
    extra_field: str,
) -> None:
    """POST /api/v1/audio/speech forwards optional fields."""
    values = {"response_format": "opus", "speed": 1.5, "instructions": "Speak slowly"}
    with patch("gateway.api.routes.audio.aspeech", new_callable=AsyncMock, return_value=FAKE_AUDIO_BYTES) as mock:
        resp = client.post(
            f"{API_ROOT}/audio/speech",
            json={
                "model": "openai:tts-1",
                "input": "Hello",
                "voice": "alloy",
                extra_field: values[extra_field],
            },
            headers=api_key_header,
        )
    assert resp.status_code == 200
    call_kwargs = mock.call_args.kwargs
    assert extra_field in call_kwargs
    assert call_kwargs[extra_field] == values[extra_field]


def test_transcription_billing_meters_tracked_with_pricing(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
) -> None:
    """POST /api/v1/audio/transcriptions records auditable charge lines alongside cost.

    A per-request rate (input_price_per_million holds the per-million-request
    rate) yields one request meter and one charge line.
    """
    client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "openai:whisper-1",
            "input_price_per_million": 2.0,
            "output_price_per_million": 0.0,
            "unit": "request",
        },
        headers=master_key_header,
    )

    mock_resp = _mock_transcription_response()
    with patch("gateway.api.routes.audio.atranscription", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            f"{API_ROOT}/audio/transcriptions",
            files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
            data={"model": "openai:whisper-1"},
            headers=api_key_header,
        )
    assert resp.status_code == 200

    usage_resp = client.get(
        f"{API_ROOT}/usage",
        params={"endpoint": "/v1/audio/transcriptions", "status": "success"},
        headers=master_key_header,
    )
    logs = usage_resp.json()
    assert len(logs) >= 1
    latest = logs[0]
    assert latest["cost"] == pytest.approx(0.000002)
    assert latest["billing_meters"] == {"requests": 1}
    assert latest["pricing_breakdown"] == [
        {"meter": "request", "units": 1, "unit_rate": pytest.approx(0.000002), "cost": pytest.approx(0.000002)}
    ]


def test_transcription_unpriced_records_no_charge_lines(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
) -> None:
    """An unpriced audio model is recorded with no cost: unknown, not free."""
    with patch(
        "gateway.api.routes.audio.atranscription",
        new_callable=AsyncMock,
        return_value=_mock_transcription_response(),
    ):
        resp = client.post(
            f"{API_ROOT}/audio/transcriptions",
            files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
            data={"model": "openai:unpriced-audio-model"},
            headers=api_key_header,
        )
    assert resp.status_code == 200

    usage_resp = client.get(
        f"{API_ROOT}/usage",
        params={"endpoint": "/v1/audio/transcriptions", "status": "success"},
        headers=master_key_header,
    )
    latest = usage_resp.json()[0]
    assert latest["cost"] is None
    assert latest["billing_meters"] is None
    assert latest["pricing_breakdown"] is None


def test_speech_billing_meters_tracked_with_pricing(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
) -> None:
    """POST /api/v1/audio/speech records auditable charge lines alongside cost."""
    client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "openai:tts-1",
            "input_price_per_million": 4.0,
            "output_price_per_million": 0.0,
            "unit": "request",
        },
        headers=master_key_header,
    )

    with patch("gateway.api.routes.audio.aspeech", new_callable=AsyncMock, return_value=FAKE_AUDIO_BYTES):
        resp = client.post(
            f"{API_ROOT}/audio/speech",
            json={"model": "openai:tts-1", "input": "Hello", "voice": "alloy"},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    # The body is audio, so the settled cost rides the headers.
    assert resp.headers[COST_USD_HEADER] == "0.000004"
    assert resp.headers[PRICING_SOURCE_HEADER] == "deployment"
    assert resp.headers[USAGE_UNIT_HEADER] == "requests"
    assert resp.headers[USAGE_UNITS_HEADER] == "1"

    usage_resp = client.get(
        f"{API_ROOT}/usage", params={"endpoint": "/v1/audio/speech", "status": "success"}, headers=master_key_header
    )
    logs = usage_resp.json()
    assert len(logs) >= 1
    latest = logs[0]
    # input_price_per_million = 4.0 is USD per million requests, so one request is 4e-06.
    assert latest["cost"] == pytest.approx(0.000004)
    assert latest["billing_meters"] == {"requests": 1}
    assert latest["pricing_breakdown"] == [
        {"meter": "request", "units": 1, "unit_rate": pytest.approx(0.000004), "cost": pytest.approx(0.000004)}
    ]


def test_speech_at_a_character_rate_names_the_characters_billed_on_the_headers(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
) -> None:
    """The headers say what the usage row says: twelve characters at $15 per million."""
    resp = client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "openai:tts-1",
            "input_price_per_million": 15.0,
            "output_price_per_million": 0.0,
            "unit": "character",
        },
        headers=master_key_header,
    )
    assert resp.status_code == 200, resp.text

    with patch("gateway.api.routes.audio.aspeech", new_callable=AsyncMock, return_value=FAKE_AUDIO_BYTES):
        resp = client.post(
            f"{API_ROOT}/audio/speech",
            json={"model": "openai:tts-1", "input": "Hello, world", "voice": "alloy"},
            headers=api_key_header,
        )
    assert resp.status_code == 200, resp.text
    assert resp.content == FAKE_AUDIO_BYTES
    assert resp.headers[COST_USD_HEADER] == "0.000180"
    assert resp.headers[PRICING_SOURCE_HEADER] == "deployment"
    assert resp.headers[USAGE_UNIT_HEADER] == "characters"
    assert resp.headers[USAGE_UNITS_HEADER] == "12"

    usage_resp = client.get(
        f"{API_ROOT}/usage", params={"endpoint": "/v1/audio/speech", "status": "success"}, headers=master_key_header
    )
    latest = usage_resp.json()[0]
    assert latest["cost"] == pytest.approx(0.00018)
    assert latest["billing_meters"] == {"characters": 12}


def test_speech_unpriced_records_no_charge_lines(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
) -> None:
    """An unpriced speech model is recorded with no cost, unknown rather than free: there is no cost to report."""
    with patch("gateway.api.routes.audio.aspeech", new_callable=AsyncMock, return_value=FAKE_AUDIO_BYTES):
        resp = client.post(
            f"{API_ROOT}/audio/speech",
            json={"model": "openai:unpriced-speech-model", "input": "Hello", "voice": "alloy"},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    assert COST_USD_HEADER not in resp.headers
    assert PRICING_SOURCE_HEADER not in resp.headers
    assert USAGE_UNIT_HEADER not in resp.headers

    usage_resp = client.get(
        f"{API_ROOT}/usage", params={"endpoint": "/v1/audio/speech", "status": "success"}, headers=master_key_header
    )
    latest = usage_resp.json()[0]
    assert latest["cost"] is None
    assert latest["billing_meters"] is None
    assert latest["pricing_breakdown"] is None


def test_a_catalog_token_rate_prices_nothing_a_transcription_without_usage_reports(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
) -> None:
    """A catalog token rate reads as tokens, so a transcription that reports none is unpriced, never one request.

    gpt-4o-transcribe is in the dataset per million tokens.
    """
    configure_default_pricing(True)
    try:
        # Pin the premise: without a dataset entry, the assertions below
        # would pass for the wrong reason and stop guarding anything.
        assert default_model_pricing("openai", "gpt-4o-transcribe", datetime.now(UTC)) is not None
        with patch(
            "gateway.api.routes.audio.atranscription",
            new_callable=AsyncMock,
            return_value=_mock_transcription_response(),
        ):
            resp = client.post(
                f"{API_ROOT}/audio/transcriptions",
                files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
                data={"model": "openai:gpt-4o-transcribe"},
                headers=api_key_header,
            )
        assert resp.status_code == 200
    finally:
        configure_default_pricing(False)

    usage_resp = client.get(
        f"{API_ROOT}/usage",
        params={"endpoint": "/v1/audio/transcriptions", "status": "success"},
        headers=master_key_header,
    )
    latest = usage_resp.json()[0]
    assert latest["model"] == "gpt-4o-transcribe"
    assert latest["cost"] is None
    assert latest["billing_meters"] is None
    assert latest["pricing_breakdown"] is None


def test_a_transcription_is_priced_per_minute_from_the_catalog(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
) -> None:
    """whisper-1 has no text price in the catalog, only audio input per minute, and that prices the length reported."""
    configure_default_pricing(True)
    try:
        pricing = default_model_pricing("openai", "whisper-1", datetime.now(UTC))
        assert pricing is not None and pricing.unit == "minute"
        result = Transcription(text="Hello", usage=UsageDuration(seconds=90, type="duration"))
        with patch("gateway.api.routes.audio.atranscription", new_callable=AsyncMock, return_value=result):
            resp = client.post(
                f"{API_ROOT}/audio/transcriptions",
                files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
                data={"model": "openai:whisper-1"},
                headers=api_key_header,
            )
        assert resp.status_code == 200, resp.text
    finally:
        configure_default_pricing(False)

    usage_resp = client.get(
        f"{API_ROOT}/usage",
        params={"endpoint": "/v1/audio/transcriptions", "status": "success"},
        headers=master_key_header,
    )
    latest = usage_resp.json()[0]
    assert latest["model"] == "whisper-1"
    # A minute and a half at $0.006 a minute.
    assert latest["cost"] == pytest.approx(0.009)
    assert latest["billing_meters"] == {"minutes": 1.5}
    assert latest["pricing_breakdown"] == [
        {"meter": "minutes", "units": 1.5, "unit": "minute", "unit_rate": 0.006, "cost": pytest.approx(0.009)}
    ]


@pytest.mark.parametrize(("form", "expected_status"), [({"language": "en"}, 400), ({}, 502)])
def test_transcription_param_the_provider_sdk_refuses(
    client: TestClient,
    api_key_header: dict[str, str],
    form: dict[str, str],
    expected_status: int,
) -> None:
    """A form field the caller sent and the SDK refuses is named in a 400; the same refusal of a field the
    caller never sent stays the gateway's 502."""
    with patch(
        "gateway.api.routes.audio.atranscription",
        new_callable=AsyncMock,
        side_effect=TypeError("transcribe() got an unexpected keyword argument 'language'"),
    ):
        resp = client.post(
            f"{API_ROOT}/audio/transcriptions",
            files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
            data={"model": "openai:whisper-1", **form},
            headers=api_key_header,
        )
    assert resp.status_code == expected_status
    assert ("'language'" in resp.json()["detail"]) is (expected_status == 400)
