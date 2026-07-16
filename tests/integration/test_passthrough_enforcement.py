"""Budget / blocked-user enforcement regression tests for billable routes.

Every billable pass-through route (embeddings, moderations, rerank, images,
audio transcription, audio speech) resolves a billed user and reserves budget
*before* the provider call. These tests lock that in: a blocked user and an
over-budget user must be rejected with 403 on every one of those routes, before
any provider is reached. This is the guard that would have caught the
``/v1/batches`` enforcement bypass (nothing asserted a given route actually runs
the budget gate).

The provider entrypoint for each route is patched to raise if it is ever
called, so no real provider is needed and, more importantly, a route that
skipped enforcement would reach the (raising) provider and fail the 403
assertion instead of silently passing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response

from gateway.core.config import API_KEY_HEADER

# Sentinel raised if a provider entrypoint is invoked. Enforcement rejects the
# request before the provider call, so reaching this is itself the failure.
_PROVIDER_REACHED = AssertionError("provider entrypoint must not be reached when enforcement rejects the request")


def _send_embeddings(client: TestClient, headers: dict[str, str], user_id: str) -> Response:
    return client.post(
        "/v1/embeddings",
        json={"model": "openai:text-embedding-3-small", "input": "hello world", "user": user_id},
        headers=headers,
    )


def _send_moderations(client: TestClient, headers: dict[str, str], user_id: str) -> Response:
    return client.post(
        "/v1/moderations",
        json={"model": "openai:omni-moderation-latest", "input": "hello world", "user": user_id},
        headers=headers,
    )


def _send_rerank(client: TestClient, headers: dict[str, str], user_id: str) -> Response:
    return client.post(
        "/v1/rerank",
        json={
            "model": "cohere:rerank-v3.5",
            "query": "what is the capital of france",
            "documents": ["paris is the capital of france", "berlin is the capital of germany"],
            "user": user_id,
        },
        headers=headers,
    )


def _send_images(client: TestClient, headers: dict[str, str], user_id: str) -> Response:
    return client.post(
        "/v1/images/generations",
        json={"model": "openai:dall-e-3", "prompt": "a red bicycle", "user": user_id},
        headers=headers,
    )


def _send_transcription(client: TestClient, headers: dict[str, str], user_id: str) -> Response:
    return client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.mp3", b"fake-audio-bytes", "audio/mpeg")},
        data={"model": "openai:whisper-1", "user": user_id},
        headers=headers,
    )


def _send_speech(client: TestClient, headers: dict[str, str], user_id: str) -> Response:
    return client.post(
        "/v1/audio/speech",
        json={"model": "openai:tts-1", "input": "hello world", "voice": "alloy", "user": user_id},
        headers=headers,
    )


# (id, provider entrypoint to patch, request sender). The patch target is the
# any-llm symbol as imported into the route module.
_BILLABLE_ROUTES: list[tuple[str, str, Callable[[TestClient, dict[str, str], str], Response]]] = [
    ("embeddings", "gateway.api.routes.embeddings.aembedding", _send_embeddings),
    ("moderations", "gateway.api.routes.moderations.amoderation", _send_moderations),
    ("rerank", "gateway.api.routes.rerank.arerank", _send_rerank),
    ("images", "gateway.api.routes.images.aimage_generation", _send_images),
    ("audio_transcription", "gateway.api.routes.audio.atranscription", _send_transcription),
    ("audio_speech", "gateway.api.routes.audio.aspeech", _send_speech),
]

_ROUTE_PARAMS = [pytest.param(patch_target, send, id=route_id) for route_id, patch_target, send in _BILLABLE_ROUTES]


def _create_blocked_user(client: TestClient, headers: dict[str, str], user_id: str) -> None:
    resp = client.post("/v1/users", json={"user_id": user_id, "blocked": True}, headers=headers)
    assert resp.status_code == 200


def _create_over_budget_user(client: TestClient, headers: dict[str, str], user_id: str) -> None:
    """A user pinned to a zero-dollar budget: any billable request is already at
    the cap, so ``reserve_budget`` rejects with 403 (mirrors the chat-completions
    over-budget test)."""
    budget = client.post("/v1/budgets", json={"max_budget": 0.0}, headers=headers)
    assert budget.status_code == 200
    budget_id = budget.json()["budget_id"]
    created = client.post(
        "/v1/users",
        json={"user_id": user_id, "budget_id": budget_id},
        headers=headers,
    )
    assert created.status_code == 200


@pytest.mark.parametrize(("patch_target", "send"), _ROUTE_PARAMS)
def test_billable_route_rejects_blocked_user(
    client: TestClient,
    master_key_header: dict[str, str],
    patch_target: str,
    send: Callable[[TestClient, dict[str, str], str], Response],
) -> None:
    """Each billable route rejects a blocked user with 403 before the provider."""
    user_id = "enforce-blocked-user"
    _create_blocked_user(client, master_key_header, user_id)

    with patch(patch_target, side_effect=_PROVIDER_REACHED):
        response = send(client, master_key_header, user_id)

    assert response.status_code == 403
    assert "blocked" in response.json()["detail"].lower()


@pytest.mark.parametrize(("patch_target", "send"), _ROUTE_PARAMS)
def test_billable_route_rejects_over_budget_user(
    client: TestClient,
    master_key_header: dict[str, str],
    patch_target: str,
    send: Callable[[TestClient, dict[str, str], str], Response],
) -> None:
    """Each billable route rejects an over-budget user with 403 before the provider."""
    user_id = "enforce-over-budget-user"
    _create_over_budget_user(client, master_key_header, user_id)

    with patch(patch_target, side_effect=_PROVIDER_REACHED):
        response = send(client, master_key_header, user_id)

    assert response.status_code == 403
    assert "budget" in response.json()["detail"].lower()


def test_batches_rejects_blocked_user(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """``/v1/batches`` rejects a blocked user with 403 before the provider.

    Batches bills the API key's own virtual user when no ``user`` body field is
    sent, so we block that virtual user and drive the route with the API key.
    Enforcement landed with issue #258.
    """
    api_key_header = {API_KEY_HEADER: f"Bearer {api_key_obj['key']}"}
    virtual_user_id = f"apikey-{api_key_obj['id']}"

    blocked = client.patch(
        f"/v1/users/{virtual_user_id}",
        json={"blocked": True},
        headers=master_key_header,
    )
    assert blocked.status_code == 200

    with patch("gateway.api.routes.batches.acreate_batch", side_effect=_PROVIDER_REACHED):
        response = client.post(
            "/v1/batches",
            json={
                "model": "openai:gpt-4o-mini",
                "requests": [
                    {"custom_id": "req-1", "body": {"messages": [{"role": "user", "content": "hi"}]}},
                ],
            },
            headers=api_key_header,
        )

    assert response.status_code == 403
    assert "blocked" in response.json()["detail"].lower()
