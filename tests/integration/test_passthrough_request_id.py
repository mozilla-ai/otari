"""Each passthrough route sends the ``Otari-Request-ID`` its usage row is recorded under.

Embeddings, decisions and search have tests of their own; these cover the
routes that share the scaffold and had none, speech included, which builds its
own response and so reaches the id through the outcome's headers.
"""

import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT, REQUEST_ID_HEADER

from .test_audio_endpoint import FAKE_AUDIO_BYTES
from .test_images_endpoint import _mock_images_response
from .test_inline_cost import _settlement
from .test_moderations_endpoint import _mock_moderation_response
from .test_rerank_endpoint import RERANK_PAYLOAD, _mock_rerank_response

# (route, patched provider call, request body, provider result)
_ROUTES: list[tuple[str, str, dict[str, Any], Any]] = [
    (
        "images/generations",
        "gateway.api.routes.images.aimage_generation",
        {"model": "openai:dall-e-3", "prompt": "a cute cat"},
        _mock_images_response(),
    ),
    ("rerank", "gateway.api.routes.rerank.arerank", RERANK_PAYLOAD, _mock_rerank_response()),
    (
        "moderations",
        "gateway.api.routes.moderations.amoderation",
        {"model": "openai:omni-moderation-latest", "input": "hello"},
        _mock_moderation_response(),
    ),
    (
        "audio/speech",
        "gateway.api.routes.audio.aspeech",
        {"model": "openai:tts-1", "input": "Hello", "voice": "alloy"},
        FAKE_AUDIO_BYTES,
    ),
]
_IDS = [route for route, *_ in _ROUTES]


@pytest.mark.parametrize(("route", "target", "body", "result"), _ROUTES, ids=_IDS)
def test_a_success_is_answered_and_recorded_under_its_request_id(
    client: TestClient, api_key_header: dict[str, str], route: str, target: str, body: dict[str, Any], result: Any
) -> None:
    with patch(target, new_callable=AsyncMock, return_value=result):
        resp = client.post(f"{API_ROOT}/{route}", json=body, headers=api_key_header)
    assert resp.status_code == 200, resp.text
    request_id = resp.headers[REQUEST_ID_HEADER]
    uuid.UUID(request_id)

    settled = _settlement(client, request_id, api_key_header)
    assert settled.status_code == 200, settled.text
    assert settled.json()["request_id"] == request_id
    assert settled.json()["status"] == "success"
    assert settled.json()["row_count"] == 1


@pytest.mark.parametrize(("route", "target", "body", "result"), _ROUTES, ids=_IDS)
def test_a_provider_failure_is_answered_and_recorded_under_its_request_id(
    client: TestClient, api_key_header: dict[str, str], route: str, target: str, body: dict[str, Any], result: Any
) -> None:
    with patch(target, new_callable=AsyncMock, side_effect=RuntimeError("provider down")):
        resp = client.post(f"{API_ROOT}/{route}", json=body, headers=api_key_header)
    assert resp.status_code == 502, resp.text
    request_id = resp.headers[REQUEST_ID_HEADER]

    settled = _settlement(client, request_id, api_key_header)
    assert settled.status_code == 200, settled.text
    assert settled.json()["status"] == "error"
    assert settled.json()["row_count"] == 1
