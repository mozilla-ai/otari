"""Tests for the POST /api/v1/audio/transcriptions and POST /api/v1/audio/speech endpoints."""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from any_llm.types.audio import Transcription
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from gateway.core.config import API_ROOT
from gateway.services.pricing_service import (
    configure_default_pricing,
    default_model_pricing,
)


def _mock_transcription_response() -> Transcription:
    """Build a real Transcription for testing."""
    return Transcription(text="Hello, world!")


FAKE_AUDIO_BYTES = b"fake-audio-content-mp3"


# === Transcription tests ===


def test_transcription_requires_auth(client: TestClient) -> None:
    """POST /api/v1/audio/transcriptions requires authentication."""
    resp = client.post(
        f"{API_ROOT}/audio/transcriptions",
        files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
        data={"model": "openai:whisper-1"},
    )
    assert resp.status_code == 401


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

    Audio bills per request like moderations (input_price_per_million holds the
    per-million-request rate), so a priced model yields one request meter and one
    charge line.
    """
    client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "openai:whisper-1",
            "input_price_per_million": 2.0,
            "output_price_per_million": 0.0,
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
    """An unpriced audio model is free by design, so it records no charge line."""
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
    assert latest["cost"] == 0.0
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


def test_speech_unpriced_records_no_charge_lines(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
) -> None:
    """An unpriced speech model is free by design, so it records no charge line."""
    with patch("gateway.api.routes.audio.aspeech", new_callable=AsyncMock, return_value=FAKE_AUDIO_BYTES):
        resp = client.post(
            f"{API_ROOT}/audio/speech",
            json={"model": "openai:unpriced-speech-model", "input": "Hello", "voice": "alloy"},
            headers=api_key_header,
        )
    assert resp.status_code == 200

    usage_resp = client.get(
        f"{API_ROOT}/usage", params={"endpoint": "/v1/audio/speech", "status": "success"}, headers=master_key_header
    )
    latest = usage_resp.json()[0]
    assert latest["cost"] == 0.0
    assert latest["billing_meters"] is None
    assert latest["pricing_breakdown"] is None


def test_transcription_ignores_models_dev_defaults(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
) -> None:
    """Default (models.dev) rates never price audio, even with default_pricing on.

    Those rates are USD per million tokens, but audio bills per request, so
    honoring one would write a charge line at the wrong unit for a rate the
    operator never configured. gpt-4o-transcribe is in the dataset, so it is the
    case that would regress.
    """
    configure_default_pricing(True)
    try:
        # Pin the premise: without a dataset entry to ignore, the assertions below
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
    assert latest["cost"] == 0.0
    assert latest["billing_meters"] is None
    assert latest["pricing_breakdown"] is None


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


# === Transcription upload tests ===

_TRANSCRIPTIONS_URL = f"{API_ROOT}/audio/transcriptions"
_TRANSCRIPTION_PATCH_TARGET = "gateway.api.routes.audio.atranscription"


def _clip(data: bytes = FAKE_AUDIO_BYTES) -> dict[str, Any]:
    return {"file": ("clip.mp3", data, "audio/mpeg")}


def test_transcription_parses_no_upload_without_credentials(client: TestClient) -> None:
    """The auth check answers before the body is read, so an anonymous upload is never stored."""
    with patch("gateway.api.routes.audio.bounded_body") as bounded:
        resp = client.post(_TRANSCRIPTIONS_URL, files=_clip(b"\x00" * 4096), data={"model": "openai:whisper-1"})
    assert resp.status_code == 401
    bounded.assert_not_called()


def test_transcription_refuses_an_upload_over_the_cap(client: TestClient, api_key_header: dict[str, str]) -> None:
    with (
        patch("gateway.api.routes.audio._MAX_AUDIO_BODY_BYTES", 2048),
        patch(_TRANSCRIPTION_PATCH_TARGET, new_callable=AsyncMock) as mock,
    ):
        resp = client.post(
            _TRANSCRIPTIONS_URL, files=_clip(b"\x00" * 4096), data={"model": "openai:whisper-1"}, headers=api_key_header
        )
    assert resp.status_code == 413
    assert "maximum upload size" in resp.json()["detail"]
    mock.assert_not_called()


def test_transcription_refuses_a_file_over_the_cap_inside_the_body_cap(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    """The body cap leaves room for the form framing, so a file just over its own cap is still refused."""
    with (
        patch("gateway.api.routes.audio._MAX_AUDIO_UPLOAD_BYTES", 16),
        patch(_TRANSCRIPTION_PATCH_TARGET, new_callable=AsyncMock) as mock,
    ):
        resp = client.post(
            _TRANSCRIPTIONS_URL, files=_clip(b"\x00" * 64), data={"model": "openai:whisper-1"}, headers=api_key_header
        )
    assert resp.status_code == 413
    mock.assert_not_called()


def _blocked_user(client: TestClient, headers: dict[str, str]) -> str:
    assert client.post(f"{API_ROOT}/users", json={"user_id": "clip-user", "blocked": True}, headers=headers).is_success
    return "clip-user"


def _over_budget_user(client: TestClient, headers: dict[str, str]) -> str:
    budget = client.post(f"{API_ROOT}/budgets", json={"max_budget": 0.0}, headers=headers).json()
    created = client.post(
        f"{API_ROOT}/users", json={"user_id": "clip-user", "budget_id": budget["budget_id"]}, headers=headers
    )
    assert created.is_success
    return "clip-user"


@pytest.mark.parametrize(
    ("make_user", "status_code"),
    [
        pytest.param(lambda _client, _headers: None, 400, id="master-key-without-user"),
        pytest.param(_blocked_user, 403, id="blocked-user"),
        pytest.param(_over_budget_user, 403, id="over-budget-user"),
    ],
)
def test_transcription_reads_no_file_for_a_refused_request(
    client: TestClient, master_key_header: dict[str, str], make_user: Any, status_code: int
) -> None:
    """A request the gateway refuses before the provider call never loads its file."""
    user = make_user(client, master_key_header)
    form = {"model": "openai:whisper-1", **({} if user is None else {"user": user})}
    upload = MagicMock(
        size=len(FAKE_AUDIO_BYTES),
        filename="clip.mp3",
        content_type="audio/mpeg",
        seek=AsyncMock(),
        read=AsyncMock(return_value=FAKE_AUDIO_BYTES),
    )
    with (
        patch("gateway.api.routes.audio.form_file", return_value=upload),
        patch(_TRANSCRIPTION_PATCH_TARGET, new_callable=AsyncMock) as mock,
    ):
        resp = client.post(_TRANSCRIPTIONS_URL, files=_clip(), data=form, headers=master_key_header)
    assert resp.status_code == status_code, resp.text
    upload.read.assert_not_called()
    mock.assert_not_called()


def test_transcription_interrupted_upload(client: TestClient, api_key_header: dict[str, str]) -> None:
    body = MagicMock(form=MagicMock(side_effect=ClientDisconnect()))
    with patch("gateway.api.routes.audio.bounded_body", return_value=body):
        resp = client.post(
            _TRANSCRIPTIONS_URL, files=_clip(), data={"model": "openai:whisper-1"}, headers=api_key_header
        )
    assert resp.status_code == 400
    assert "interrupted" in resp.json()["detail"]


@pytest.mark.parametrize(
    ("data", "files", "field"),
    [
        pytest.param({"model": "openai:whisper-1", "file": "/etc/passwd"}, [], "file", id="file-as-text"),
        pytest.param({"model": "openai:whisper-1", "temperature": "warm"}, [], "temperature", id="bad-temperature"),
        pytest.param({}, [], "model", id="missing-model"),
    ],
)
def test_transcription_refuses_a_field_of_the_wrong_shape(
    client: TestClient, api_key_header: dict[str, str], data: dict[str, str], files: list[Any], field: str
) -> None:
    """The form is read by the route, so a bad field is still the 422 FastAPI gave."""
    sent = [*files, *([("file", ("clip.mp3", FAKE_AUDIO_BYTES, "audio/mpeg"))] if "file" not in data else [])]
    with patch(_TRANSCRIPTION_PATCH_TARGET, new_callable=AsyncMock) as mock:
        resp = client.post(_TRANSCRIPTIONS_URL, data=data, files=sent, headers=api_key_header)
    assert resp.status_code == 422, resp.text
    assert ["body", field] in [error["loc"] for error in resp.json()["detail"]]
    mock.assert_not_called()


def test_transcription_refuses_a_second_file(client: TestClient, api_key_header: dict[str, str]) -> None:
    """One file is read, so a text field sent as a file is one file too many rather than stored."""
    files = [
        ("file", ("clip.mp3", FAKE_AUDIO_BYTES, "audio/mpeg")),
        ("model", ("m.txt", b"openai:whisper-1", "text/plain")),
    ]
    with patch(_TRANSCRIPTION_PATCH_TARGET, new_callable=AsyncMock) as mock:
        resp = client.post(_TRANSCRIPTIONS_URL, files=files, headers=api_key_header)
    assert resp.status_code == 400
    assert "Too many files" in resp.json()["detail"]
    mock.assert_not_called()
